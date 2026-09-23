"""Structured JSON Lines Logger for Prefill Execution Policy Decisions.

Records observation state, Jev System 1 decisions, and ground-truth execution
runtimes to build distillation datasets for local policy heuristics.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Optional

from dsv41.jev_policy import JevDecision, PrefillObservation, get_allowed_gpus_min_free_mem_gb


class PolicyLogger:
    """Thread-safe append-only JSONL logger for policy execution data."""

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path or os.environ.get(
            "DSV41_POLICY_LOG_FILE",
            os.path.join(os.getcwd(), "results", "prefix_policy_decisions.jsonl"),
        )
        self._lock = threading.Lock()
        # Ensure target directory exists
        log_dir = os.path.dirname(self.log_path)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
            except Exception:
                pass

    def log_decision_and_outcome(
        self,
        observation: PrefillObservation,
        decision: JevDecision,
        actual_policy: str,
        prefill_time_s: float,
        reused_tokens: int,
        new_tokens: int,
        total_tokens: int,
        req_id: str = "",
        success: bool = True,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record an end-to-end execution sample for evaluation and distillation."""
        effective_tok_s = round(total_tokens / max(prefill_time_s, 1e-6), 2)
        new_tok_s = round(new_tokens / max(prefill_time_s, 1e-6), 2)
        mem_after = get_allowed_gpus_min_free_mem_gb()

        record = {
            "timestamp": time.time(),
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "req_id": req_id,
            "observation": observation.to_state_dict(),
            "jev_decision": decision.to_dict(),
            "actual_policy": actual_policy,
            "outcome": {
                "prefill_time_s": round(prefill_time_s, 4),
                "total_tokens": total_tokens,
                "reused_tokens": reused_tokens,
                "new_tokens": new_tokens,
                "effective_tok_s": effective_tok_s,
                "new_tok_s": new_tok_s,
                "gpu_min_free_mem_gb_after": mem_after,
                "success": success,
                "error": error,
            },
        }

        # Thread-safe append to JSONL
        try:
            line = json.dumps(record, ensure_ascii=False) + "\n"
            with self._lock:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception as exc:
            print(f"[policy-logger] Error writing policy record: {exc}", flush=True)

        return record
