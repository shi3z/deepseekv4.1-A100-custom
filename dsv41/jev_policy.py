"""Jev System 1 Execution Policy Controller for DeepSeek-V4.1 Long Prefill.

Uses TypeSafe AI Jev (System 1 judgment model) as an execution policy controller
to dynamically select the optimal prefill execution path for multi-turn requests:
  - reuse_gpu_slot: In-VRAM GPU slot continuation (Level 1 cache)
  - reuse_anchor_block256: Restore host RAM snapshot, replay suffix in 256-token blocks
  - reuse_anchor_block512: Restore host RAM snapshot, replay suffix in 512-token blocks
  - full_prefill: Full chunked prefill from start_pos=0
  - safe_recompute: Recompute from scratch when memory is critical, epoch mismatched, or cache invalid

CRITICAL SYSTEM INVARIANT:
  Only GPUs 0, 1, 2, 3 may be monitored or used. GPUs 4, 5, 6, 7 are strictly prohibited.
"""

from __future__ import annotations

import concurrent.futures
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
    """Observation state provided to Jev for policy decision."""
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
        """Compact dictionary format suitable for Jev state."""
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
class JevDecision:
    """Structured decision output from Jev / verification controller."""
    policy: str  # "reuse_gpu_slot" | "reuse_anchor_block256" | "reuse_anchor_block512" | "full_prefill" | "safe_recompute"
    block_size: int  # 0, 256, 512, etc.
    confidence: float
    reason_code: str
    probabilities: Dict[str, float] = dataclasses.field(default_factory=dict)
    fallback_used: bool = False
    latency_ms: float = 0.0
    model: str = "jev-latest"

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


# Jev Choice criteria descriptions for structured judgment
JEV_POLICY_CRITERIA = {
    "reuse_gpu_slot": 'Select ONLY when cache_hit_type is "gpu_slot". Continues directly on GPU slot with zero host PCIe transfers.',
    "reuse_anchor_block256": 'Select when cache_hit_type is "anchor_snapshot" and suffix_tokens <= 1024. Restores snapshot and replays in 256-token blocks.',
    "reuse_anchor_block512": 'Select when cache_hit_type is "anchor_snapshot" and suffix_tokens > 1024. Restores snapshot and replays in 512-token blocks.',
    "full_prefill": 'Select when cache_hit_type is "none", or when suffix_tokens exceeds 75% of prompt_tokens.',
    "safe_recompute": 'Select when epoch_matched is False, gpu_min_free_mem_gb < 1.0, or cache corruption is suspected.',
}


def fallback_policy(obs: PrefillObservation, reason: str = "fallback") -> JevDecision:
    """Safe, deterministic baseline policy heuristic."""
    # 1. Critical safety rule: epoch mismatch or severe memory pressure
    if not obs.epoch_matched:
        return JevDecision(
            policy="safe_recompute",
            block_size=0,
            confidence=1.0,
            reason_code=f"{reason}:epoch_mismatch",
            fallback_used=True,
            model="heuristic-rule",
        )

    if obs.gpu_min_free_mem_gb < 0.2:
        return JevDecision(
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
            return JevDecision(
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
                return JevDecision(
                    policy="reuse_anchor_block256",
                    block_size=256,
                    confidence=1.0,
                    reason_code=f"{reason}:anchor_short_suffix_{obs.suffix_tokens}",
                    fallback_used=True,
                    model="heuristic-rule",
                )
            else:
                return JevDecision(
                    policy="reuse_anchor_block512",
                    block_size=512,
                    confidence=1.0,
                    reason_code=f"{reason}:anchor_long_suffix_{obs.suffix_tokens}",
                    fallback_used=True,
                    model="heuristic-rule",
                )

    # 4. Default: full prefill from scratch
    return JevDecision(
        policy="full_prefill",
        block_size=0,
        confidence=1.0,
        reason_code=f"{reason}:cache_miss_or_large_suffix",
        fallback_used=True,
        model="heuristic-rule",
    )


def verify_and_constrain(decision: JevDecision, obs: PrefillObservation) -> JevDecision:
    """Hard-rule safety verification layer over Jev's output."""
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
        fb = fallback_policy(obs, reason=f"guardrail:low_confidence_{decision.confidence:.2f}")
        return fb

    # Assign correct block_size if missing
    if policy == "reuse_anchor_block256":
        block_size = 256
    elif policy == "reuse_anchor_block512":
        block_size = 512
    elif policy == "reuse_gpu_slot":
        block_size = 512

    return JevDecision(
        policy=policy,
        block_size=block_size,
        confidence=decision.confidence,
        reason_code=reason,
        probabilities=decision.probabilities,
        fallback_used=fallback_used,
        latency_ms=decision.latency_ms,
        model=decision.model,
    )


class JevPolicyController:
    """Controller that calls Jev (TypeSafe System 1) and enforces safety constraints."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        if not self.api_key:
            # Check local .env or jev-arcade .env if not found in environment
            for path in (
                os.path.join(os.getcwd(), ".env"),
                "/mnt/ssdraid/git/deepseekv4.1/.env",
                "/home/shi3z/jev-arcade/.env",
            ):
                if os.path.exists(path):
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line.startswith("TYPESAFE_API_KEY="):
                                    self.api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                                    break
                    except Exception:
                        pass
                if self.api_key:
                    break

        self.model_name = model or os.environ.get("TYPESAFE_MODEL", "jev-latest")
        self.client = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="jev_ctl")
        self.enabled = os.environ.get("DSV41_ENABLE_JEV_POLICY", "1") == "1"

        if self.api_key and self.enabled:
            try:
                from typesafe_sdk import TypeSafeClient
                self.client = TypeSafeClient(api_key=self.api_key)
            except Exception as e:
                print(f"[jev-policy] Failed to initialize TypeSafeClient ({e}), will use fallback heuristic", flush=True)

    def _call_jev_raw(self, state: Dict[str, Any]) -> Tuple[str, float, Dict[str, float], str]:
        """Perform raw Jev System 1 call via SDK."""
        if not self.client:
            raise RuntimeError("TypeSafe client not initialized")

        from typesafe_sdk import Choice

        res = self.client.system_one(
            state=state,
            questions={
                "prefill_policy": Choice(
                    instructions="Based on cache_hit_type, suffix_tokens, and memory state, choose the single execution policy that maximizes prefill throughput while guaranteeing safety.",
                    criteria=JEV_POLICY_CRITERIA,
                )
            },
        )
        ans = res.answers["prefill_policy"]
        return ans.choice, float(ans.confidence), ans.probabilities, str(res.model)

    def decide_policy(
        self,
        obs: PrefillObservation,
        timeout_ms: Optional[float] = None,
    ) -> JevDecision:
        """Dynamically decide prefill execution policy with bounded latency and hard safety."""
        if not self.enabled or not self.client:
            return fallback_policy(obs, reason="jev_disabled_or_no_client")

        # Configurable timeout (default 350ms)
        if timeout_ms is None:
            timeout_ms = float(os.environ.get("DSV41_JEV_TIMEOUT_MS", "350.0"))

        timeout_s = max(0.05, timeout_ms / 1000.0)
        t0 = time.perf_counter()

        future = self._executor.submit(self._call_jev_raw, obs.to_state_dict())
        try:
            choice, confidence, probs, model_id = future.result(timeout=timeout_s)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            raw_decision = JevDecision(
                policy=choice,
                block_size=512 if "512" in choice else (256 if "256" in choice else 0),
                confidence=confidence,
                reason_code="jev_decision",
                probabilities=probs,
                fallback_used=False,
                latency_ms=elapsed_ms,
                model=model_id,
            )
            # Run through hard verification rules
            final_decision = verify_and_constrain(raw_decision, obs)
            return final_decision

        except concurrent.futures.TimeoutError:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            fb = fallback_policy(obs, reason=f"timeout_{elapsed_ms:.1f}ms")
            fb.latency_ms = elapsed_ms
            return fb
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            fb = fallback_policy(obs, reason=f"exception_{exc.__class__.__name__}")
            fb.latency_ms = elapsed_ms
            return fb
