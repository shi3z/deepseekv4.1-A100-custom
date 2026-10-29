"""Prefill execution policy for DeepSeek-V4.1 multi-turn requests (deterministic heuristic).

Selects the prefill execution path:
  - reuse_gpu_slot: In-VRAM GPU slot continuation (Level 1 cache)
  - reuse_anchor_block256: Restore host RAM snapshot, replay suffix in 256-token blocks
  - reuse_anchor_block512: Restore host RAM snapshot, replay suffix in 512-token blocks
  - full_prefill: Full chunked prefill from start_pos=0
  - safe_recompute: Recompute from scratch when memory is critical, epoch mismatched, or cache invalid

CRITICAL SYSTEM INVARIANT:
  Only GPUs 0, 1, 2, 3 may be monitored or used. GPUs 4, 5, 6, 7 are strictly prohibited.
"""

from __future__ import annotations

import dataclasses
import os
import time
from typing import Any, Dict, Optional, Tuple

# Permitted GPU indices - NEVER reference GPUs 4..7
ALLOWED_GPU_DEVICES = (0, 1, 2, 3)


def get_allowed_gpus_min_free_mem_gb() -> float:
    """Query minimum free VRAM in GiB across GPUs 0, 1, 2, 3 ONLY.

    Strictly guarantees that GPUs 4, 5, 6, 7 are NEVER queried or referenced.
    """
    min_free_bytes = float("inf")
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            for d in ALLOWED_GPU_DEVICES:
                h = pynvml.nvmlDeviceGetHandleByIndex(d)
                info = pynvml.nvmlDeviceGetMemoryInfo(h)
                if info.free < min_free_bytes:
                    min_free_bytes = info.free
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        # Secondary fallback using torch without device count discovery outside 0..3
        try:
            import torch
            if torch.cuda.is_available():
                for d in ALLOWED_GPU_DEVICES:
                    try:
                        free_b, _ = torch.cuda.mem_get_info(d)
                        if free_b < min_free_bytes:
                            min_free_bytes = free_b
                    except Exception:
                        pass
        except Exception:
            pass

    if min_free_bytes == float("inf"):
        return 0.0
    return round(min_free_bytes / (1024 ** 3), 3)


@dataclasses.dataclass
class PrefillObservation:
    """Observation of the prefill situation used to pick the execution path."""
    prompt_tokens: int
    lcp_tokens: int
    suffix_tokens: int
    cache_hit_type: str  # "gpu_slot" | "anchor_snapshot" | "none"
    gpu_slot_id: int  # -1 if none
    gpu_slot_lcp: int  # 0 if none
    anchor_base_tokens: int  # 0 if none
    active_slots: int
    free_slots: int
    gpu_min_free_mem_gb: float
    epoch_matched: bool
    is_multi_turn: bool
    estimated_replay_tokens: int

    def to_state_dict(self) -> Dict[str, Any]:
        """Compact dictionary form (logged with the decision)."""
        return {
            "prompt_tokens": int(self.prompt_tokens),
            "lcp_tokens": int(self.lcp_tokens),
            "suffix_tokens": int(self.suffix_tokens),
            "cache_hit_type": str(self.cache_hit_type),
            "gpu_slot_id": int(self.gpu_slot_id),
            "gpu_slot_lcp": int(self.gpu_slot_lcp),
            "anchor_base_tokens": int(self.anchor_base_tokens),
            "active_slots": int(self.active_slots),
            "free_slots": int(self.free_slots),
            "gpu_min_free_mem_gb": float(self.gpu_min_free_mem_gb),
            "epoch_matched": bool(self.epoch_matched),
            "is_multi_turn": bool(self.is_multi_turn),
            "estimated_replay_tokens": int(self.estimated_replay_tokens),
        }


@dataclasses.dataclass
class PrefillDecision:
    """Prefill execution decision."""
    policy: str  # "reuse_gpu_slot" | "reuse_anchor_block256" | "reuse_anchor_block512" | "full_prefill" | "safe_recompute"
    block_size: int  # 0, 256, 512, etc.
    confidence: float
    reason_code: str
    probabilities: Dict[str, float] = dataclasses.field(default_factory=dict)
    fallback_used: bool = False
    latency_ms: float = 0.0
    model: str = "heuristic-rule"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "block_size": self.block_size,
            "confidence": round(self.confidence, 4),
            "reason_code": self.reason_code,
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "fallback_used": self.fallback_used,
            "latency_ms": round(self.latency_ms, 2),
            "model": self.model,
        }


def decide_policy(obs: PrefillObservation, reason: str = "heuristic") -> PrefillDecision:
    """Deterministic prefill policy: GPU-slot continuation > host snapshot replay > full prefill, with safety rules first."""
    # 1. Critical safety rule: epoch mismatch or severe memory pressure
    if not obs.epoch_matched:
        return PrefillDecision(
            policy="safe_recompute",
            block_size=0,
            confidence=1.0,
            reason_code=f"{reason}:epoch_mismatch",
            fallback_used=True,
            model="heuristic-rule",
        )

    # Low free VRAM used to force a cold recompute (and evict the matching entry). That is backwards: restoring a
    # snapshot and replaying the suffix needs less transient memory than a cold prefill of the whole prompt, and
    # nvml's "free" ignores PyTorch's reusable cached blocks, so during a long prefill it reads ~0 and every
    # concurrent request went cold (2026-10-29: 47K-token cold prefills instead of 2.6K-token replays).
    # DSV41_SAFE_RECOMPUTE_FREE_GB > 0 re-enables the rule with that threshold.
    if obs.gpu_min_free_mem_gb < float(os.environ.get("DSV41_SAFE_RECOMPUTE_FREE_GB", "0")):
        return PrefillDecision(
            policy="safe_recompute",
            block_size=0,
            confidence=1.0,
            reason_code=f"{reason}:low_vram_{obs.gpu_min_free_mem_gb:.2f}gb",
            fallback_used=True,
            model="heuristic-rule",
        )

    # 2. Level 1 GPU Slot Hit
    if obs.cache_hit_type == "gpu_slot" and obs.gpu_slot_id >= 0 and obs.gpu_slot_lcp >= 64:
        # If suffix is not overwhelmingly huge (> 75% of prompt), GPU slot reuse is fastest
        if obs.suffix_tokens <= max(64, int(obs.prompt_tokens * 0.75)):
            return PrefillDecision(
                policy="reuse_gpu_slot",
                block_size=512,  # GPU continuation default chunk size
                confidence=1.0,
                reason_code=f"{reason}:gpu_slot_hit_lcp_{obs.gpu_slot_lcp}",
                fallback_used=True,
                model="heuristic-rule",
            )

    # 3. Level 2 Host Anchor Snapshot Hit
    if obs.cache_hit_type == "anchor_snapshot" and obs.anchor_base_tokens >= 64:
        if obs.suffix_tokens <= max(64, int(obs.prompt_tokens * 0.75)):
            if obs.suffix_tokens <= 1024:
                return PrefillDecision(
                    policy="reuse_anchor_block256",
                    block_size=256,
                    confidence=1.0,
                    reason_code=f"{reason}:anchor_short_suffix_{obs.suffix_tokens}",
                    fallback_used=True,
                    model="heuristic-rule",
                )
            else:
                return PrefillDecision(
                    policy="reuse_anchor_block512",
                    block_size=512,
                    confidence=1.0,
                    reason_code=f"{reason}:anchor_long_suffix_{obs.suffix_tokens}",
                    fallback_used=True,
                    model="heuristic-rule",
                )

    # 4. Default: full prefill from scratch
    return PrefillDecision(
        policy="full_prefill",
        block_size=0,
        confidence=1.0,
        reason_code=f"{reason}:cache_miss_or_large_suffix",
        fallback_used=True,
        model="heuristic-rule",
    )


def verify_and_constrain(decision: PrefillDecision, obs: PrefillObservation) -> PrefillDecision:
    """Hard-rule safety verification over a (possibly forced) decision."""
    policy = decision.policy
    reason = decision.reason_code
    fallback_used = decision.fallback_used
    block_size = decision.block_size

    # Invariant 1: Epoch mismatch must NEVER reuse cache
    if not obs.epoch_matched and policy in ("reuse_gpu_slot", "reuse_anchor_block256", "reuse_anchor_block512"):
        policy = "safe_recompute"
        block_size = 0
        reason = "guardrail:epoch_mismatch"
        fallback_used = True

    # Invariant 2: Cache miss must NEVER attempt cache reuse
    elif obs.cache_hit_type == "none" and policy in ("reuse_gpu_slot", "reuse_anchor_block256", "reuse_anchor_block512"):
        policy = "full_prefill"
        block_size = 0
        reason = "guardrail:cache_miss"
        fallback_used = True

    # Invariant 3: If no GPU slot is available, cannot reuse GPU slot
    elif obs.cache_hit_type != "gpu_slot" and policy == "reuse_gpu_slot":
        if obs.cache_hit_type == "anchor_snapshot":
            policy = "reuse_anchor_block512" if obs.suffix_tokens > 1024 else "reuse_anchor_block256"
            block_size = 512 if policy == "reuse_anchor_block512" else 256
            reason = "guardrail:downgraded_to_anchor"
        else:
            policy = "full_prefill"
            block_size = 0
            reason = "guardrail:downgraded_to_full_prefill"
        fallback_used = True

    # Invariant 4: Low confidence threshold (< 0.80) -> fallback to deterministic heuristic
    elif decision.confidence < 0.80:
        fb = decide_policy(obs, reason=f"guardrail:low_confidence_{decision.confidence:.2f}")
        return fb

    # Assign correct block_size if missing
    if policy == "reuse_anchor_block256":
        block_size = 256
    elif policy == "reuse_anchor_block512":
        block_size = 512
    elif policy == "reuse_gpu_slot":
        block_size = 512

    return PrefillDecision(
        policy=policy,
        block_size=block_size,
        confidence=decision.confidence,
        reason_code=reason,
        probabilities=decision.probabilities,
        fallback_used=fallback_used,
        latency_ms=decision.latency_ms,
        model=decision.model,
    )
