"""Generation engine shared by the REPL and the OpenAI-compatible server: loads the model once, keeps
the CUDA-graph decode runtime, and streams tokens for one request at a time (the runtime is single
sequence; requests are serialized with a lock)."""
from __future__ import annotations
from array import array
import hashlib

import os
import sys
import threading
import time
import queue
import collections
from dataclasses import dataclass, field
from typing import Iterator

import torch

from .decode import DecodeRuntime
from .load import load_model
import traceback

DEFAULT_CKPT = "/mnt/ssd/models/DeepSeek-V4.1-Flash-Abliterated" if os.path.exists("/mnt/ssd/models/DeepSeek-V4.1-Flash-Abliterated") else "/mnt/ssd/models/DeepSeek-V4.1-Flash"
CKPT = os.environ.get("DSV41_CKPT", DEFAULT_CKPT)


@dataclass
class GenParams:
    max_new_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.95
    stop: list[str] = field(default_factory=list)
    seed: int | None = None
    repetition_penalty: float = 1.10
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.10
    penalty_window: int = 256
    progressive_penalty: float = 1.5
    ban_cycles: bool = True
    loop_detect: bool = True
    min_loop_match: int = 48
    min_loop_cycle: int = 1


_WHITESPACE_TOKEN_IDS = {200, 201, 223, 262, 271, 290}


def apply_penalties(
    logits: torch.Tensor,
    tokens: list[int],
    repetition_penalty: float = 1.10,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.10,
    window: int = 256,
    progressive_penalty: float = 1.5,
    ban_cycles: bool = True,
) -> torch.Tensor:
    """Apply cycle suppression, progressive penalty, and repetition/frequency penalties to 1D logits."""
    if not tokens:
        return logits

    penalized = logits.clone()
    n = len(tokens)

    # 1. Consecutive cycle suppression:
    # Detect if the last (L-1) tokens match the previous cycle of length L (L in 3..256).
    # If true, the token tokens[-L] would complete a consecutive repeat of that cycle.
    if ban_cycles and n >= 5:
        max_L = min(256, (n + 1) // 2)
        for L in range(3, max_L + 1):
            if tokens[-2 * L + 1 : -L] == tokens[-L + 1 :]:
                banned_tok = tokens[-L]
                penalized[banned_tok] -= 50.0

    context_tokens = tokens[-window:] if window > 0 and n > window else tokens
    if not context_tokens:
        return penalized

    # 2. Multiplicative repetition penalty
    if repetition_penalty != 1.0 and repetition_penalty > 0:
        unique_toks = list(set(context_tokens))
        tok_idx = torch.tensor(unique_toks, dtype=torch.long, device=logits.device)
        vals = penalized[tok_idx]
        penalized_vals = torch.where(
            vals > 0,
            vals / float(repetition_penalty),
            vals * float(repetition_penalty),
        )
        penalized[tok_idx] = penalized_vals

    # 3. Additive frequency and progressive repetition penalty
    if presence_penalty != 0.0 or frequency_penalty != 0.0 or progressive_penalty > 0.0:
        counts = collections.Counter(context_tokens)
        toks = list(counts.keys())
        cnts = [counts[t] for t in toks]
        tok_idx = torch.tensor(toks, dtype=torch.long, device=logits.device)
        cnt_vals = torch.tensor(cnts, dtype=logits.dtype, device=logits.device)

        penalties = float(presence_penalty) + float(frequency_penalty) * cnt_vals
        if progressive_penalty > 0.0:
            prog_mask = torch.tensor(
                [1.0 if t not in _WHITESPACE_TOKEN_IDS else 0.0 for t in toks],
                dtype=logits.dtype,
                device=logits.device,
            )
            prog = torch.clamp(cnt_vals - 1.0, min=0.0) * float(progressive_penalty) * prog_mask
            penalties += prog
        penalized[tok_idx] -= penalties

    return penalized


def detect_loop(
    tokens: list[int],
    min_match: int = 48,
    min_cycle: int = 1,
) -> tuple[int, int, int] | None:
    """Detect if the trailing `min_match` tokens are an exact repetition of a prior block.

    Returns (cycle_len, trim_count, prev_pos) if a repetition loop is detected, else None.
    - cycle_len: distance between the previous block and the current repeating block
    - trim_count: number of duplicate tokens that should be trimmed from the end
    - prev_pos: index in tokens where the prior matching sequence started
    """
    n = len(tokens)
    if n < min_match + min_cycle:
        return None
    tail = tokens[-min_match:]
    head = tail[0]
    search_end = n - min_match - min_cycle
    for p in range(search_end, -1, -1):
        if tokens[p] == head and tokens[p : p + min_match] == tail:
            cycle_len = (n - min_match) - p
            if cycle_len >= min_match:
                back = 0
                while (
                    (n - min_match - 1 - back >= p + min_match)
                    and (p - 1 - back >= 0)
                    and (tokens[n - min_match - 1 - back] == tokens[p - 1 - back])
                ):
                    back += 1
                total_match = min_match + back
                trim_count = total_match
            else:
                k = 0
                while n - 1 - k - cycle_len >= 0 and tokens[n - 1 - k] == tokens[n - 1 - k - cycle_len]:
                    k += 1
                trim_count = k
            return cycle_len, trim_count, p
    return None


class _BatchRequest:
    def __init__(self, prompt_ids: list[int], params: GenParams, max_new: int, gen: torch.Generator | None, req_id: str = "", images=None, token_types=None):
        self.req_id = req_id or f"req_{id(self):x}"
        self.prompt_ids = prompt_ids
        self.params = params
        self.max_new = max_new
        self.gen = gen
        self.images = images
        self.token_types = token_types
        self.pos = 0
        self.next_token = 0
        self.out_tokens: list[int] = []
        self.done_event = threading.Event()
        self.result_text = ""
        self.result_count = 0
        self.finish_reason = "stop"
        self.error: Exception | None = None
        self.start_time = time.perf_counter()
        self.first_token_time = 0.0
        self.live_text = ""
        self.slot_id = -1
        self.decode_tok_s = 0.0


def sample_token(logits: torch.Tensor, temperature: float, top_p: float, gen: torch.Generator | None) -> int:
    if temperature <= 0:
        try:
            return int(logits.argmax(dim=-1).item())
        except Exception:
            return 0
    try:
        # Sanitize logits against NaN/Inf
        logits_f = torch.nan_to_num(logits.float(), nan=-1e4, posinf=1e4, neginf=-1e4)
        scaled_logits = logits_f / max(float(temperature), 1e-4)
        probs = torch.softmax(scaled_logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        if 0 < top_p < 1:
            sp, si = probs.sort(descending=True)
            keep = (sp.cumsum(-1) - sp) < top_p  # keep tokens until cumulative mass passes top_p
            keep[..., 0] = True  # Always keep at least the top-1 token
            sp = sp * keep
            mass = sp.sum()
            if mass <= 0 or torch.isnan(mass) or torch.isinf(mass):
                return int(si[0].item())
            norm_sp = sp / mass
            if torch.isnan(norm_sp).any() or torch.isinf(norm_sp).any() or (norm_sp < 0).any():
                return int(si[0].item())
            idx = torch.multinomial(norm_sp, 1, generator=gen)
            return int(si.gather(-1, idx).item())

        mass = probs.sum()
        if mass <= 0 or torch.isnan(mass) or torch.isinf(mass):
            return int(logits_f.argmax(dim=-1).item())
        norm_probs = probs / mass
        if torch.isnan(norm_probs).any() or torch.isinf(norm_probs).any() or (norm_probs < 0).any():
            return int(logits_f.argmax(dim=-1).item())
        return int(torch.multinomial(norm_probs, 1, generator=gen).item())
    except Exception as exc:
        print(f"[sample_token] fallback to argmax due to: {exc}", flush=True)
        try:
            return int(logits.argmax(dim=-1).item())
        except Exception:
            return 0


class Engine:
    def __init__(self, ckpt: str = CKPT, devices: list[int] | None = None, max_seq_len: int = 8192,
                 budgets: dict[int, float] | None = None, use_graphs: bool = True, thinking_mode: str = "chat",
                 offload_experts=False, hot_experts: int = 0, route_stats: str = "", ep: bool = False, ep_shards: list[int] | None = None, mtp: int = 0, mtp_device: int | None = None,
                 max_seqs: int = 1):
        from transformers import AutoTokenizer

        sys.path.insert(0, os.path.join(ckpt, "encoding"))
        from encoding import encode_messages, parse_message_from_completion_text  # type: ignore

        self._encode = encode_messages
        self._parse = parse_message_from_completion_text
        self.thinking_mode = thinking_mode
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.max_seq_len = max_seq_len
        self.mtp = int(mtp)
        # Last-request prefix reuse.
        #
        # Only exact append-only prompt growth is reused:
        #
        #   old_prompt == new_prompt[:len(old_prompt)]
        #
        # Anything else falls back to a full prefill.
        self._prefix_prompt_ids: list[int] | None = None
        self._prefix_len: int = 0
        self.mtp_device = mtp_device
        if self.mtp < 0 or self.mtp > 5:
            raise ValueError("--mtp must be 0..5")
        _env_seqs = os.environ.get("DSV41_MAX_SEQS")
        self.max_seqs = int(_env_seqs) if _env_seqs else int(max_seqs)
        if self.max_seqs > 1:
            self.max_decode_slots = self.max_seqs - 1
            max_batch = self.max_decode_slots * (1 + self.mtp)
        else:
            self.max_decode_slots = 1
            max_batch = 1 + self.mtp
        print(f"[engine-init] max_seqs={self.max_seqs} max_decode_slots={self.max_decode_slots} max_batch={max_batch}", flush=True)
        self.model = load_model(ckpt, devices or list(range(torch.cuda.device_count())), max_seq_len=max_seq_len, max_batch=max_batch, max_seqs=self.max_seqs,
                                budgets_gb=budgets, tokenizer=self.tok, offload_experts=offload_experts,
                                hot_experts=hot_experts, route_stats=route_stats, ep=ep, ep_shards=ep_shards)
        if offload_experts:
            from .decode import OffloadDecodeRuntime
            self.rt = OffloadDecodeRuntime(self.model, use_graphs=use_graphs)
            self.rt_b1 = None
        elif ep:
            from .ep import EPRuntime
            self.rt = EPRuntime(self.model, use_graphs=use_graphs)
            if self.max_seqs > 1 and self.rt.B > 1:
                print(f"[engine-init] creating dedicated B=1 decode runtime for single-request speed...", flush=True)
                self.rt_b1 = EPRuntime(self.model, use_graphs=use_graphs, max_batch=1)
            else:
                self.rt_b1 = None
        else:
            self.rt = DecodeRuntime(self.model, use_graphs=use_graphs)
            if self.max_seqs > 1 and self.rt.B > 1:
                self.rt_b1 = DecodeRuntime(self.model, use_graphs=use_graphs, max_batch=1)
            else:
                self.rt_b1 = None
        if use_graphs:
            self.rt.capture()
            if self.rt_b1 is not None:
                print(f"[engine-init] capturing dedicated B=1 CUDA graphs...", flush=True)
                self.rt_b1.capture()
                print(f"[engine-init] dedicated B=1 CUDA graphs captured successfully", flush=True)
        self.ds = None

        if self.mtp:
            if offload_experts:
                raise ValueError(
                    "--mtp is not supported with --offload-experts"
                )

            from .dspark import DSparkRows
            from .load import Checkpoint

            if self.mtp_device is None:
                mtp_dev = self.model.blocks[-1].device
            else:
                mtp_dev = torch.device(
                    f"cuda:{self.mtp_device}"
                )

            free_b, total_b = torch.cuda.mem_get_info(mtp_dev)

            print(
                f"[mtp] loading DSpark on {mtp_dev} "
                f"free={free_b / 2**30:.2f} GiB "
                f"total={total_b / 2**30:.2f} GiB",
                flush=True,
            )

            last = mtp_dev

            self.ds = DSparkRows(
                Checkpoint(ckpt),
                self.model.args,
                last,
                self.model.embed,
                self.model.head,
                self.model.shared,
                len(self.model.blocks),
                self.rt,
            )

            # Tell eager prefill to retain the three target-layer
            # hidden states needed to seed DSpark.
            self.model.collect_main_hidden = self.ds.targets

            # DSpark itself always has a 5-token draft block.
            self.ds.capture(1)

            print(
                f"[mtp] enabled drafts={self.mtp} "
                f"verify_rows={1 + self.mtp}",
                flush=True,
            )
        self.lock = threading.Lock()
        self.eos = self.tok.eos_token_id
        self.ckpt = ckpt
        if "abliterated" in ckpt.lower():
            self.model_name = os.environ.get("DSV41_MODEL_NAME", "deepseek-v4.1-flash-abliterated")
        else:
            self.model_name = os.environ.get("DSV41_MODEL_NAME", "deepseek-v4.1-flash")
        self._jev_engine = None
        self.stats_tracker = None
        self.last_prefill_stats: dict | None = None
        self.current_phase = "idle"
        self.current_context_tokens = 0
        self.prefill_history: collections.deque = collections.deque(maxlen=30)
        self._slot_lock = threading.Lock()
        self.slot_states: dict[int, dict] = {}
        self._slot_tokens: dict[int, list[int]] = {}
        self._slot_tokens_lock = threading.Lock()
        self._last_decode_log_time = 0.0
        self.last_decode_tok_s: float | None = None
        self.last_finish_reason: str = "stop"
        slots_to_track = list(range(1, self.max_seqs)) if self.max_seqs > 1 else [0]
        for s in slots_to_track:
            self.slot_states[s] = {
                "slot_id": s,
                "status": "idle",
                "req_id": "",
                "prompt_tokens": 0,
                "reused_tokens": 0,
                "reused_slot": -1,
                "generated_tokens": 0,
                "max_tokens": 0,
                "tok_s": 0.0,
                "start_time": 0.0,
                "first_tok_time": 0.0,
                "elapsed_s": 0.0,
                "recent_text": "",
                "completed_at": 0.0,
            }
        if self.max_seqs > 1:
            self._init_batch_scheduler()

    def _record_prefill_stats(
        self,
        mode: str,
        total: int,
        reused: int,
        new_tokens: int,
        dt: float,
        prefill_type: str | None = None,
        lcp: int | None = None,
        base_tokens: int | None = None,
        suffix_tokens: int | None = None,
    ):
        if not prefill_type:
            if mode in ("gpu_slot_hit", "GPU_HIT") or "gpu" in str(mode).lower():
                prefill_type = "gpu_hit"
            elif reused > 0 or mode in ("LCP-HIT", "REPLAY"):
                prefill_type = "host_replay"
            else:
                prefill_type = "cold"

        lcp_val = lcp if lcp is not None else reused
        base_val = base_tokens if base_tokens is not None else reused
        suffix_val = suffix_tokens if suffix_tokens is not None else new_tokens

        self.last_prefill_stats = {
            "mode": mode,
            "prefill_type": prefill_type,
            "total_tokens": total,
            "reused_tokens": reused,
            "new_tokens": new_tokens,
            "lcp": lcp_val,
            "base_tokens": base_val,
            "suffix_tokens": suffix_val,
            "hit_rate_pct": round(reused / max(total, 1) * 100, 1),
            "time_s": round(dt, 3),
            "new_tok_s": round(new_tokens / max(dt, 1e-9), 1),
            "effective_tok_s": round(total / max(dt, 1e-9), 1),
            "timestamp": time.time(),
        }
        self.current_context_tokens = total
        self.prefill_history.append(self.last_prefill_stats)
        if self.stats_tracker is not None:
            try:
                self.stats_tracker.record_prefill(self.last_prefill_stats)
            except Exception:
                pass

    @property
    def jev_engine(self):
        if self._jev_engine is None:
            from .jev import JevEngine
            self._jev_engine = JevEngine(self.model, self.tok)
        return self._jev_engine

    def jev_inference(self, prompt: str, schema: dict, max_batch: int = 32) -> tuple[dict, dict]:
        with self.lock:
            self.current_phase = "jev"
            try:
                assembled, metrics = self.jev_engine.process_request(prompt, schema, max_batch=max_batch)
                tot_tokens = metrics.get("prompt_tokens", 0)
                reused = metrics.get("prefix_saved_tokens", 0)
                new_tok = max(0, tot_tokens - reused)
                dt_prefill = metrics.get("prefill_time_s", 0.0)
                self._record_prefill_stats("JEV", tot_tokens, reused, new_tok, dt_prefill)
                if self.stats_tracker is not None:
                    hit_str = "HIT" if metrics.get("cache_hit") else "MISS"
                    self.stats_tracker.record_cache_event(
                        f"Jev structured output: {metrics.get('num_fields', 0)} fields, schema {hit_str}, saved {reused}/{tot_tokens} tokens ({metrics.get('cache_hit_latency_ms', 0):.1f}ms cache hit, total {metrics.get('total_latency_ms', 0):.1f}ms)",
                        event_type="hit" if metrics.get("cache_hit") else "info"
                    )
                return assembled, metrics
            finally:
                self.current_phase = "idle"

    def get_cache_stats(self) -> dict:
        try:
            entries, total_bytes = self._prefix_cache_stats()
        except Exception:
            entries, total_bytes = 0, 0
        jev_schemas = 0
        jev_requests = 0
        jev_stats = {}
        if self._jev_engine is not None and getattr(self._jev_engine, "prefix_tree", None) is not None:
            pt = self._jev_engine.prefix_tree
            jev_schemas = len(pt.schema_nodes)
            jev_requests = len(getattr(pt, "request_nodes", {}))
            jev_stats = getattr(pt, "stats", {})

        # Dynamic KV cache allocation
        shared = getattr(self.model, "shared", None)
        kv_stats = {}
        max_context = getattr(self, "max_seq_len", 1048576)
        current_ctx = getattr(self, "current_context_tokens", 0)
        if shared is not None and hasattr(shared, "compress_kv"):
            for (owner, dev), t in shared.compress_kv.items():
                if dev == self.model.blocks[0].device:
                    ratio = self.model.args.compress_ratios[owner]
                    kv_stats[f"owner_{owner}"] = {
                        "allocated_rows": t.size(1),
                        "max_rows": shared.cache_max_rows.get(owner, 0),
                        "ratio": ratio,
                        "allocated_tokens": t.size(1) * ratio,
                    }

        # Active decode slots with live progress and text preview
        slots_list = []
        active_cnt = 0
        combined_tok_s = 0.0
        now_t = time.perf_counter()
        with getattr(self, "_slot_lock", threading.Lock()):
            for s_id in sorted(getattr(self, "slot_states", {}).keys()):
                st = dict(self.slot_states[s_id])
                # Reset stale completed slots after 60 seconds of inactivity
                if st.get("status") == "completed" and (now_t - st.get("completed_at", 0)) > 60.0:
                    st["status"] = "idle"
                    st["recent_text"] = ""
                    st["generated_tokens"] = 0
                    st["tok_s"] = 0.0
                    self.slot_states[s_id] = st
                if st.get("status") in ("generating", "prefilling"):
                    active_cnt += 1
                    combined_tok_s += st.get("tok_s", 0.0)
                slots_list.append(st)

        return {
            "prefix_entries": entries,
            "prefix_bytes": total_bytes,
            "prefix_gb": round(total_bytes / (1024 ** 3), 2),
            "jev_schemas": jev_schemas,
            "jev_requests": jev_requests,
            "jev_stats": jev_stats,
            "kv_cache": kv_stats,
            "current_context_tokens": current_ctx,
            "max_context_tokens": max_context,
            "context_pct": round(current_ctx / max(max_context, 1) * 100, 2),
            "active_slots": slots_list,
            "slots": slots_list,
            "active_decode_slots": active_cnt,
            "combined_decode_tok_s": round(combined_tok_s, 1),
            "last_prefill": getattr(self, "last_prefill_stats", None),
            "prefill_history": list(getattr(self, "prefill_history", [])),
            "current_phase": "decode" if active_cnt > 0 else getattr(self, "current_phase", "idle"),
        }

    # ---------------------------------------------------------------- prompts
    def chat_prompt(self, messages: list[dict], thinking_mode: str | None = None) -> str:
        return self._encode(messages, thinking_mode=thinking_mode or self.thinking_mode)

    def format_chat(self, messages: list[dict], thinking_mode: str | None = None) -> tuple[list[int], list | None, torch.Tensor | None]:
        """Format chat messages with multimodal image support.
        Returns: (prompt_tokens, images, token_types_tensor)."""
        from .vision import parse_tagged_text, prepare_vl_inputs, VisionConfig

        # Normalize <image>...</image> inside string content
        normalized_messages = []
        for m in messages:
            m_copy = dict(m)
            c = m_copy.get("content")
            if isinstance(c, str) and "<image>" in c and "</image>" in c:
                m_copy["content"] = parse_tagged_text(c)
            normalized_messages.append(m_copy)

        prompt, media_data = self._encode(
            normalized_messages,
            thinking_mode=thinking_mode or self.thinking_mode,
            return_multi_modal_data=True,
        )
        images_raw = media_data.get("images", []) if isinstance(media_data, dict) else []
        if not images_raw:
            return self.tok.encode(prompt), None, None

        v_cfg = VisionConfig.from_cfg(self.model.args.cfg)
        tokens, token_types, image_inputs = prepare_vl_inputs(prompt, images_raw, self.tok, v_cfg)
        dev0 = self.model.blocks[0].device
        token_types_tensor = torch.tensor([token_types], device=dev0, dtype=torch.long)
        return tokens, [image_inputs], token_types_tensor

    @staticmethod
    def _fallback_parse_dsml(text: str) -> list[dict]:
        import re, json
        tool_calls = []
        invokes = re.findall(
            r'<｜DSML｜ invoke name=[\"\'](.*?)[\"\']>(.*?)(?:</｜DSML｜ invoke>|(?=<｜DSML｜ invoke)|(?=</｜DSML｜ calls>)|$)',
            text,
            re.DOTALL,
        )
        for name, body in invokes:
            args = {}
            params = re.findall(
                r'<｜DSML｜ parameter name=[\"\'](.*?)[\"\'](?: string=[\"\'].*?[\"\'])?>(.*?)(?:</｜DSML｜ parameter>|(?=<｜DSML｜ parameter)|(?=</｜DSML｜ invoke)|$)',
                body,
                re.DOTALL,
            )
            for pname, pval in params:
                pval = re.sub(r'</?(?:｜DSML｜|think|analysis).*?>', '', pval).strip()
                try:
                    args[pname] = json.loads(pval)
                except Exception:
                    args[pname] = pval
            tool_calls.append({
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            })
        return tool_calls

    def parse_completion(self, text: str, thinking_mode: str | None = None) -> dict:
        """Structured assistant message (content / reasoning_content / tool_calls). The official parser
        wants the completion to end with the EOS string; we generate without it, so append it."""
        eos = self.tok.eos_token or ""
        reasoning = None
        clean_text = text

        if "</think>" in clean_text:
            parts = clean_text.split("</think>", 1)
            reasoning = parts[0].strip()
            if "<think>" in reasoning:
                reasoning = reasoning.split("<think>", 1)[1].strip()
            clean_text = parts[1]
        elif "<think>" in clean_text:
            reasoning = clean_text.split("<think>", 1)[1].strip()
            clean_text = ""

        if "<｜DSML｜ calls>" in clean_text:
            idx = clean_text.find("<｜DSML｜ calls>")
            prefix = clean_text[:idx].rstrip("\n")
            rest = clean_text[idx:]
            clean_text = prefix + "\n\n" + rest

            if "<｜DSML｜ parameter" in clean_text and "</｜DSML｜ parameter>" not in clean_text:
                clean_text += "</｜DSML｜ parameter>"
            if "<｜DSML｜ invoke" in clean_text and "</｜DSML｜ invoke>" not in clean_text:
                clean_text += "\n</｜DSML｜ invoke>"
            if "</｜DSML｜ calls>" not in clean_text:
                clean_text += "\n</｜DSML｜ calls>"

        try:
            res = self._parse(clean_text + eos, thinking_mode=thinking_mode or self.thinking_mode)
            if reasoning and not res.get("reasoning_content"):
                res["reasoning_content"] = reasoning
            return res
        except Exception:
            pass

        # Fallback DSML parser if official parser failed on malformed/special tokens
        if "<｜DSML｜" in text:
            tool_calls = self._fallback_parse_dsml(text)
            if tool_calls:
                content = clean_text
                if "<｜DSML｜" in content:
                    content = content[:content.find("<｜DSML｜")].strip()
                return {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning,
                    "tool_calls": tool_calls,
                }

        return {"role": "assistant", "content": clean_text.strip(), "reasoning_content": reasoning, "tool_calls": []}

    # ---------------------------------------------------------------- generation
    @torch.inference_mode()
    def _can_reuse_prefix(
        self,
        prompt_ids: list[int],
    ) -> int:
        """Return reusable prefix length, or 0.

        Deliberately conservative: reuse only when the entire previous
        prompt is an exact prefix of the new prompt.
        """
        old = self._prefix_prompt_ids

        if not old:
            return 0

        old_len = len(old)

        if len(prompt_ids) < old_len:
            return 0

        if prompt_ids[:old_len] != old:
            # Diagnostic: how much of the old request actually matches?
            lim = min(len(old), len(prompt_ids))
            lcp = 0

            # Python loop is diagnostic-only and runs once/request.
            while lcp < lim and old[lcp] == prompt_ids[lcp]:
                lcp += 1

            print(
                f"[prefix-cache] NEAR-MISS "
                f"old={len(old):,} "
                f"new={len(prompt_ids):,} "
                f"lcp={lcp:,} "
                f"old_match={100.0*lcp/max(len(old),1):.2f}%",
                flush=True,
            )

            return 0

        return old_len

    def _remember_prefill_prefix(
        self,
        prompt_ids: list[int],
    ):
        # Copy: callers may reuse/mutate their list.
        self._prefix_prompt_ids = list(prompt_ids)
        self._prefix_len = len(prompt_ids)

    def _timed_prefill_forward(
        self,
        token_ids,
        start_pos: int,
        *,
        total_tokens: int,
        reused: int,
    ):
        """Run one prefill forward and report wall-clock throughput.

        model.forward() is effectively synchronous for the current EP
        prefill path because remote expert streams are joined before it
        returns.  Synchronizing the returned tensor's device makes the
        final timing boundary explicit without touching unrelated GPUs.
        """
        n_new = len(token_ids)

        t0 = time.perf_counter()

        logits = self.model.forward(
            torch.tensor(
                [token_ids],
                dtype=torch.long,
            ),
            start_pos,
        )

        # Do not synchronize cuda:5/6/7: they may belong to other jobs.
        # Only ensure the output device has completed its queued work.
        try:
            if torch.is_tensor(logits) and logits.is_cuda:
                torch.cuda.synchronize(logits.device)
            elif isinstance(logits, (tuple, list)):
                for z in logits:
                    if torch.is_tensor(z) and z.is_cuda:
                        torch.cuda.synchronize(z.device)
                        break
        except Exception:
            pass

        dt = time.perf_counter() - t0

        new_rate = n_new / dt if dt > 0 else 0.0
        effective_rate = total_tokens / dt if dt > 0 else 0.0

        print(
            f"[prefill-bench] "
            f"new={n_new:,} "
            f"reused={reused:,} "
            f"total={total_tokens:,} "
            f"start={start_pos:,} "
            f"time={dt:.3f}s "
            f"new_tok_s={new_rate:,.1f} "
            f"effective_tok_s={effective_rate:,.1f}",
            flush=True,
        )

        return logits

    def _prefix_cache_slots(self):
        """Find mutable attention-cache tensors.

        We deliberately snapshot only runtime attention caches, never
        model weights or MoE routing-table caches.

        Currently recognised:
          * *.window_kv_cache
          * *.compress_kv[key]

        Shared cache objects reached by multiple layers are de-duplicated.
        """
        slots = []
        seen_obj = set()
        seen_slot = set()

        def visit(obj, depth=0):
            if obj is None or depth > 12:
                return

            oid = id(obj)
            if oid in seen_obj:
                return
            seen_obj.add(oid)

            if torch.is_tensor(obj):
                return

            if isinstance(
                obj,
                (str, bytes, int, float, bool, type(None)),
            ):
                return

            if isinstance(obj, (list, tuple)):
                for x in obj:
                    visit(x, depth + 1)
                return

            if isinstance(obj, dict):
                for x in obj.values():
                    if not torch.is_tensor(x):
                        visit(x, depth + 1)
                return

            dct = getattr(obj, "__dict__", None)
            if not isinstance(dct, dict):
                return

            for name, val in dct.items():

                # Sliding-window attention cache.
                if (
                    name == "window_kv_cache"
                    and torch.is_tensor(val)
                ):
                    k = ("attr", id(obj), name)

                    if k not in seen_slot:
                        seen_slot.add(k)
                        slots.append(
                            ("attr", obj, name, val)
                        )

                    continue

                # Dynamic compressed cache mirrors.
                if (
                    name in ("compress_kv", "index_k")
                    and isinstance(val, dict)
                ):
                    # Canonical single-copy per owner:
                    # In multi-device setups, compress_kv and index_k are mirrored across all GPUs.
                    # Snapshotting each mirror causes 4x redundant host-RAM bloat.
                    # We pick one canonical mirror per owner to persist, and broadcast upon restore.
                    dedup_mirrors = os.environ.get("DSV41_PREFIX_DEDUP_MIRRORS", "1") == "1"
                    if dedup_mirrors:
                        owner_candidates = {}
                        for key, t in val.items():
                            if not torch.is_tensor(t):
                                continue
                            owner = key[0] if isinstance(key, tuple) else key
                            owner_candidates.setdefault(owner, []).append((key, t))
                        for owner in sorted(owner_candidates.keys(), key=lambda x: str(x)):
                            candidates = owner_candidates[owner]
                            # Deterministic preference: pick primary device if matching layer's device, else sort by str(dev)
                            layer = getattr(self.model, "layers", {}).get(owner, None) if hasattr(getattr(self.model, "layers", None), "__getitem__") else None
                            layer_dev = getattr(layer, "device", None) if layer else None
                            candidates.sort(key=lambda item: (0 if layer_dev is not None and isinstance(item[0], tuple) and len(item[0]) > 1 and str(item[0][1]) == str(layer_dev) else 1, str(item[0])))
                            key, t = candidates[0]
                            k = ("dict", id(val), repr(key))
                            if k not in seen_slot:
                                seen_slot.add(k)
                                slots.append(("dict", val, key, t))
                    else:
                        for key, t in val.items():
                            if not torch.is_tensor(t):
                                continue

                            k = ("dict", id(val), repr(key))

                            if k in seen_slot:
                                continue

                            seen_slot.add(k)

                            slots.append(
                                ("dict", val, key, t)
                            )

                    # Do not recursively walk tensor values.
                    continue

                # Avoid known non-attention runtime tensor caches.
                if name in (
                    "_cache",
                    "_stage",
                    "_ep_prefill_streams",
                ):
                    continue

                if torch.is_tensor(val):
                    continue

                if isinstance(val, (list, tuple, dict)):
                    visit(val, depth + 1)
                    continue

                mod = getattr(
                    getattr(val, "__class__", None),
                    "__module__",
                    "",
                )

                if (
                    mod.startswith("dsv41")
                    or hasattr(val, "__dict__")
                ):
                    visit(val, depth + 1)

        visit(self.model)
        # Draft attention rings must rewind with the main prefix.
        visit(getattr(self, "ds", None))


        # PREFIX-FULLSTATE-EXTRAS
        #
        # Prefix execution state is more than the large KV tables.
        # Compressor partial groups and SharedAttn owner metadata must
        # rewind together with the caches.
        seen = {
            (kind, id(holder), repr(key))
            for kind, holder, key, _ in slots
        }

        def add_tensor_attr(holder, name):
            val = getattr(holder, name, None)
            if not torch.is_tensor(val):
                return
            ident = ("attr", id(holder), repr(name))
            if ident in seen:
                return
            slots.append(("attr", holder, name, val))
            seen.add(ident)

        def add_scalar_attr(holder, name):
            val = getattr(holder, name, None)
            if not isinstance(val, (int, bool)):
                return
            ident = ("scalar_attr", id(holder), repr(name))
            if ident in seen:
                return

            # Fourth tuple member is a tiny synthetic tensor so existing
            # snapshot-signature/tmpfs machinery can describe this slot.
            slots.append(
                (
                    "scalar_attr",
                    holder,
                    name,
                    torch.tensor(
                        int(val),
                        dtype=torch.int64,
                    ),
                )
            )
            seen.add(ident)

        visited_extra = set()

        def scan_extra(obj, depth=0):
            if obj is None or depth > 12:
                return

            oid = id(obj)
            if oid in visited_extra:
                return
            visited_extra.add(oid)

            cls = obj.__class__.__name__

            if cls == "NgramHashState":
                add_tensor_attr(obj, "cache")

            if cls == "Compressor":
                for name in (
                    "kv_state",
                    "score_state",
                    "kv_ring",
                    "score_ring",
                ):
                    add_tensor_attr(obj, name)

            if cls == "SharedAttn":
                # topk_idxs and candidates are NOT persistent prefix
                # state.  They are derived per-forward from the current
                # query and must be recomputed after restore.
                for name in (
                    "kv_owner",
                    "index_owner",
                ):
                    add_scalar_attr(obj, name)

            d = getattr(obj, "__dict__", None)
            if not isinstance(d, dict):
                return

            for val in d.values():
                if torch.is_tensor(val):
                    continue

                if isinstance(val, dict):
                    for x in val.values():
                        if not torch.is_tensor(x):
                            scan_extra(x, depth + 1)

                elif isinstance(val, (list, tuple)):
                    for x in val:
                        if not torch.is_tensor(x):
                            scan_extra(x, depth + 1)

                elif hasattr(val, "__dict__"):
                    scan_extra(val, depth + 1)

        scan_extra(self.model)

        # PREFIX-DROP-EPHEMERAL
        #
        # These tensors are outputs of the current attention/indexer
        # computation, not state required to resume a prefix.
        #
        # Restoring them is both incorrect and expensive.  In
        # particular topk_idxs may be an inference tensor whose shape
        # depends on the old query length.
        _ephemeral_names = {
            "topk_idxs",
            "candidates",
        }

        _before = len(slots)

        slots = [
            slot
            for slot in slots
            if not (
                slot[0] == "attr"
                and slot[2] in _ephemeral_names
            )
        ]

        _dropped = _before - len(slots)

        if _dropped:
            print(
                f"[prefix-snapshot] "
                f"DROP-EPHEMERAL slots={_dropped}",
                flush=True,
            )

        return slots

    @torch.inference_mode()
    def _snapshot_prefix_state(self, used_tokens=None):
        """Copy attention state to host RAM, slicing caches to used rows."""
        slots = self._prefix_cache_slots()

        if not slots:
            raise RuntimeError(
                "prefix snapshot found zero cache tensors"
            )

        snap = []
        total = 0
        allocated_total = 0
        inventory = []

        for slot_no, (kind, holder, key, src) in enumerate(slots):

            # scalar_attr carries a synthetic CPU tensor representing
            # the current integer metadata value.
            if kind == "scalar_attr":
                src = torch.tensor(
                    int(getattr(holder, key)),
                    dtype=torch.int64,
                )

            allocated_bytes = src.numel() * src.element_size()
            allocated_total += allocated_bytes

            # Dynamic compressed/index caches, Engram history, and SWA window
            # may be allocated well beyond the logical prefix. Persist only rows
            # that can be read when restoring this anchor; unused capacity is recreated locally.
            save_src = src
            if used_tokens is not None and src.ndim >= 2:
                if (
                    kind == "dict"
                    and isinstance(key, tuple)
                    and key
                    and hasattr(getattr(self.model, "shared", None), "cache_max_rows")
                    and key[0] in self.model.shared.cache_max_rows
                ):
                    owner = int(key[0])
                    cr = getattr(self.model.args, "compress_ratios", None) if hasattr(self.model, "args") else None
                    if isinstance(cr, dict):
                        ratio = int(cr.get(owner, 1))
                    elif isinstance(cr, (list, tuple)) and 0 <= owner < len(cr):
                        ratio = int(cr[owner])
                    else:
                        ratio = 1
                    rows = max(1, min(src.shape[1], int(used_tokens) // max(1, ratio) + 1))
                    save_src = src[:, :rows].contiguous()
                elif (
                    kind == "attr"
                    and key == "cache"
                    and holder.__class__.__name__ == "NgramHashState"
                ):
                    rows = max(1, min(src.shape[1], int(used_tokens)))
                    save_src = src[:, :rows].contiguous()
                elif (
                    (key == "window_kv_cache" or (isinstance(key, str) and "window_kv" in key))
                    and src.shape[1] > 128
                ):
                    rows = max(1, min(src.shape[1], 128, int(used_tokens)))
                    save_src = src[:, :rows].contiguous()
            elif src.ndim >= 2 and (key == "window_kv_cache" or (isinstance(key, str) and "window_kv" in key)) and src.shape[1] > 128:
                save_src = src[:, :128].contiguous()

            nbytes = save_src.numel() * save_src.element_size()
            total += nbytes
            used_rows = (
                int(save_src.shape[1])
                if save_src.ndim >= 2
                else None
            )
            inventory.append({
                "slot": slot_no, "kind": kind, "key": repr(key),
                "shape": tuple(int(x) for x in src.shape),
                "dtype": str(src.dtype), "allocated_bytes": allocated_bytes,
                "used_rows": used_rows, "used_bytes": nbytes,
            })

            # Pinned memory is preferred for future fast restores.
            # Fall back to normal host memory if the system pin limit
            # refuses a large allocation.
            try:
                host = torch.empty_like(
                    save_src,
                    device="cpu",
                    pin_memory=True,
                )
            except Exception:
                host = torch.empty_like(
                    save_src,
                    device="cpu",
                )

            host.copy_(save_src, non_blocking=False)

            snap.append(
                (kind, holder, key, host)
            )

        saved_pct = 100.0 * (1.0 - total / max(allocated_total, 1))
        print(
            f"[prefix-snapshot] SAVE slots={len(snap)} "
            f"logical={total / 2**20:,.1f}MiB "
            f"allocated={allocated_total / 2**20:,.1f}MiB "
            f"saved={saved_pct:.1f}%",
            flush=True,
        )
        if os.environ.get("DSV41_DEBUG_PREFIX_INVENTORY", "0") == "1":
            for item in inventory:
                print(
                    f"[prefix-inventory] slot={item['slot']} "
                    f"kind={item['kind']} key={item['key']} "
                    f"shape={item['shape']} dtype={item['dtype']} "
                    f"allocated={item['allocated_bytes']} "
                    f"used_rows={item['used_rows']} used={item['used_bytes']}",
                    flush=True,
                )
            print(
                f"[prefix-inventory] TOTAL allocated={allocated_total} "
                f"logical={total} saved={saved_pct:.1f}%", flush=True
            )

        return snap, total

    @torch.inference_mode()
    def _restore_prefix_state(self, snap):
        """Restore a previously captured attention-cache snapshot."""
        touched = set()

        try:
            for kind, holder, key, src in snap:
                if kind == "scalar_attr":
                    setattr(
                        holder,
                        key,
                        int(src.item()),
                    )
                    continue

                if kind == "attr":
                    dst = getattr(holder, key)
                else:
                    dst = holder[key]

                if not torch.is_tensor(dst):
                    raise RuntimeError(
                        f"prefix snapshot target disappeared: "
                        f"{kind}:{key!r}"
                    )

                if dst.dtype != src.dtype:
                    raise RuntimeError(
                        f"prefix snapshot dtype mismatch: "
                        f"{dst.dtype} != {src.dtype}"
                    )

                if dst.numel() < src.numel():
                    raise RuntimeError(
                        f"prefix snapshot target shrank: "
                        f"{dst.numel()} < {src.numel()}"
                    )

                # Dynamic compressed caches can have grown since SAVE.
                # Restore the old allocated prefix; later rows are logically
                # invisible because start_pos determines valid cache length.
                # prefix-restore-check
                if not torch.is_tensor(src) or not torch.is_tensor(dst):
                    raise RuntimeError(
                        f"prefix restore non-tensor slot: "
                        f"kind={kind!r} key={key!r}"
                    )

                if src.dtype != dst.dtype:
                    raise RuntimeError(
                        f"prefix restore dtype mismatch: "
                        f"kind={kind!r} key={key!r} "
                        f"src={src.dtype} dst={dst.dtype}"
                    )

                if src.numel() > dst.numel():
                    raise RuntimeError(
                        f"prefix restore size mismatch: "
                        f"kind={kind!r} key={key!r} "
                        f"src={tuple(src.shape)} "
                        f"dst={tuple(dst.shape)}"
                    )

                try:
                    if (
                        dst.ndim >= 2
                        and src.ndim == dst.ndim
                        and src.shape[0] <= dst.shape[0]
                        and src.shape[1] <= dst.shape[1]
                    ):
                        dst[:src.shape[0], :src.shape[1]].copy_(
                            src,
                            non_blocking=src.is_pinned(),
                        )
                    else:
                        dst.view(-1)[:src.numel()].copy_(
                            src.view(-1),
                            non_blocking=src.is_pinned(),
                        )
                except Exception as exc:
                    print(
                        f"[prefix-snapshot] RESTORE-FAIL "
                        f"kind={kind!r} "
                        f"key={key!r} "
                        f"src_shape={tuple(src.shape)} "
                        f"dst_shape={tuple(dst.shape)} "
                        f"src_dtype={src.dtype} "
                        f"dst_dtype={dst.dtype} "
                        f"device={dst.device} "
                        f"error={exc}",
                        flush=True,
                    )
                    raise

                if dst.is_cuda:
                    touched.add(dst.device)

                # Broadcast canonical mirror to other GPU devices for shared tables
                if (
                    kind == "dict"
                    and isinstance(key, tuple)
                    and len(key) >= 1
                    and isinstance(holder, dict)
                    and hasattr(getattr(self.model, "shared", None), "compress_kv")
                    and (holder is getattr(self.model.shared, "compress_kv", None)
                         or holder is getattr(self.model.shared, "index_k", None))
                ):
                    owner = key[0]
                    for (o, dev), mirror in holder.items():
                        if o == owner and dev != dst.device:
                            if mirror.ndim >= 2 and dst.ndim >= 2:
                                mirror[:src.shape[0], :src.shape[1]].copy_(
                                    dst[:src.shape[0], :src.shape[1]],
                                    non_blocking=True,
                                )
                            else:
                                mirror.view(-1)[:src.numel()].copy_(
                                    dst.view(-1)[:src.numel()],
                                    non_blocking=True,
                                )
                            if mirror.is_cuda:
                                touched.add(mirror.device)
        finally:
            for dev in touched:
                torch.cuda.synchronize(dev)

        print(
            f"[prefix-snapshot] RESTORE "
            f"slots={len(snap)}",
            flush=True,
        )

    @staticmethod
    def _prompt_lcp(a, b):
        if not a or not b:
            return 0

        n = min(len(a), len(b))
        i = 0

        while i < n and a[i] == b[i]:
            i += 1

        return i

    def _capture_main_hidden_row(self, rows):
        """Keep hidden rows produced by token replay for MTP seeding."""
        mh = getattr(self.model, "main_hidden", None)

        if (
            isinstance(mh, list)
            and mh
            and torch.is_tensor(mh[0])
            and mh[0].numel()
        ):
            rows.append(mh[0].detach().clone())

            keep = int(
                os.environ.get(
                    "DSV41_PREFIX_HIDDEN_KEEP",
                    "512",
                )
            )

            if len(rows) > keep:
                del rows[:-keep]

    def _restore_replayed_main_hidden(self, rows):
        """Expose replayed hidden rows to the existing MTP setup."""
        if not rows:
            return

        mh = getattr(self.model, "main_hidden", None)

        if not (
            isinstance(mh, list)
            and mh
        ):
            return

        try:
            cat = torch.cat(rows, dim=0)
            mh[0] = cat
        except Exception as e:
            print(
                f"[prefix-cache] WARNING: "
                f"could not combine main_hidden: {e}",
                flush=True,
            )

    def _set_prefix_replay_mode(self, enabled: bool):
        """Enable safe single-token prefill attention only during
        prefix rollback/replay.

        Normal generation and MTP decode continue to use the optimized
        sparse_attn_decode kernel.
        """
        seen = set()

        def visit(obj, depth=0):
            if obj is None or depth > 12:
                return

            oid = id(obj)
            if oid in seen:
                return
            seen.add(oid)

            if torch.is_tensor(obj):
                return

            if isinstance(
                obj,
                (str, bytes, int, float, bool, type(None)),
            ):
                return

            if hasattr(obj, "window_kv_cache"):
                try:
                    setattr(
                        obj,
                        "_prefix_replay_prefill",
                        bool(enabled),
                    )
                except Exception:
                    pass

            if isinstance(obj, dict):
                for x in obj.values():
                    if not torch.is_tensor(x):
                        visit(x, depth + 1)
                return

            if isinstance(obj, (list, tuple)):
                for x in obj:
                    visit(x, depth + 1)
                return

            d = getattr(obj, "__dict__", None)
            if not isinstance(d, dict):
                return

            for name, x in d.items():
                if name in (
                    "_cache",
                    "_stage",
                    "_ep_prefill_streams",
                ):
                    continue

                if torch.is_tensor(x):
                    continue

                if isinstance(x, (dict, list, tuple)):
                    visit(x, depth + 1)
                elif hasattr(x, "__dict__"):
                    visit(x, depth + 1)

        visit(self.model)

    def _write_prefix_draft_hidden(self, hidden, end_pos):
        """Commit actual hidden history before publishing a prefix anchor."""
        ds = getattr(self, "ds", None)
        if ds is None:
            return
        rows = hidden.reshape(-1, hidden.shape[-1])[-ds.win:]
        positions = torch.arange(end_pos - len(rows), end_pos, device=ds.device)
        ds.write_main_rows(rows, torch.zeros_like(positions), positions)

    @torch.inference_mode()
    def _replay_prefix_tail(
        self,
        prompt_ids,
        start_pos,
        *,
        snapshot_at=None,
    ):
        """Replay cached suffix using production decode runtime.

        For MTP:
          - all but the final prompt token use self.rt.step()
          - final prompt token uses model.forward()
          - collect_main_hidden is explicitly enabled for that final
            forward, exactly as in normal MTP initialization

        This gives DSpark the target-layer hidden state without putting
        the entire replay through slow model.forward().
        """

        total = len(prompt_ids)

        if start_pos >= total:
            return None, None

        logits = None

        # ------------------------------------------------------------
        # snapshot_at may be:
        #
        #   None
        #   int
        #   iterable[int]
        #
        # Preserve the legacy return type for None/int callers.
        # Multi-anchor callers receive:
        #
        #   [(position, (snapshot, bytes)), ...]
        # ------------------------------------------------------------
        _multi_snapshot = isinstance(
            snapshot_at,
            (list, tuple, set),
        )

        if snapshot_at is None:
            _snapshot_positions = set()
        elif _multi_snapshot:
            _snapshot_positions = {
                int(p)
                for p in snapshot_at
                if start_pos <= int(p) <= total
            }
        else:
            _snapshot_positions = {
                int(snapshot_at)
            }

        _captured_positions = set()
        new_snapshot = None
        new_snapshots = []

        def _capture_prefix_anchor(pos):
            nonlocal new_snapshot

            pos = int(pos)

            if (
                pos not in _snapshot_positions
                or pos in _captured_positions
            ):
                return

            try:
                snap = self._snapshot_prefix_state(used_tokens=pos)
            except TypeError as exc:
                # Preserve compatibility with lightweight test doubles and
                # older callers that provide a no-argument snapshot hook.
                if "used_tokens" not in str(exc):
                    raise
                snap = self._snapshot_prefix_state()
            _captured_positions.add(pos)

            if _multi_snapshot:
                new_snapshots.append(
                    (pos, snap)
                )
            else:
                new_snapshot = snap

            if os.environ.get(
                "DSV41_DEBUG_PREFIX_ANCHORS",
                "0",
            ) == "1":
                _snap, _bytes = snap
                print(
                    f"[prefix-anchor] "
                    f"SAVE base={pos:,} "
                    f"size={_bytes/2**20:,.1f}MiB",
                    flush=True,
                )

        use_mtp_tail = bool(
            getattr(self, "mtp", 0)
            and getattr(self, "ds", None) is not None
        )

        replay_total = total - start_pos

        # Reserve final prompt token for a real model.forward() when MTP
        # needs main_hidden.
        decode_end = total - 1 if use_mtp_tail else total

        t0 = time.perf_counter()

        report_every = max(
            128,
            int(
                os.environ.get(
                    "DSV41_PREFIX_REPLAY_REPORT",
                    "512",
                )
            ),
        )

        done = 0

        # ------------------------------------------------------------
        # Prefix replay.
        #
        # Scalar mode:
        #     one rt.step() per token.
        #
        # Block mode:
        #     use the model's causal multi-token continuation path.
        #
        # This path is intentionally limited to prefix replay. Normal
        # autoregressive decode and MTP verification continue to use the
        # production EPRuntime path.
        #
        #   DSV41_PREFIX_BLOCK_REPLAY=1
        #   DSV41_PREFIX_BLOCK_SIZE=128
        #
        # IMPORTANT:
        # The current model.py has dedicated start_pos>0 / seqlen>1
        # handling for the window ring and compressed-cache continuation.
        # ------------------------------------------------------------

        block_replay = (
            os.environ.get(
                "DSV41_PREFIX_BLOCK_REPLAY",
                "1",
            ) == "1"
        )

        block_size = max(
            2,
            int(
                os.environ.get(
                    "DSV41_PREFIX_BLOCK_SIZE",
                    "128",
                )
            ),
        )

        block_min = max(
            2,
            int(
                os.environ.get(
                    "DSV41_PREFIX_BLOCK_MIN",
                    "16",
                )
            ),
        )

        decode_tokens = decode_end - start_pos

        if (
            block_replay
            and decode_tokens >= block_min
        ):
            model = self.model

            set_mode = getattr(
                self,
                "_set_prefix_replay_mode",
                None,
            )

            # main_hidden is only required for the final MTP prompt token.
            # Do not collect 3*dim target-layer activations for every
            # replay chunk.
            old_collect = getattr(
                model,
                "collect_main_hidden",
                None,
            )

            if old_collect is not None:
                model.collect_main_hidden = tuple(self.ds.targets) if use_mtp_tail else ()

            pos = start_pos

            try:
                while pos < decode_end:

                    # Snapshot state BEFORE consuming token at `pos`.
                    # Works for both legacy single-anchor and
                    # multi-anchor snapshot_at.
                    _capture_prefix_anchor(pos)

                    end_pos = min(
                        decode_end,
                        pos + block_size,
                    )

                    # Never jump across a pending snapshot anchor.
                    #
                    # Example:
                    #   pos=65,500, block end=65,756
                    #   anchors=[65,644, 66,668, ...]
                    #
                    # Stop at 65,644.  The next loop iteration calls
                    # _capture_prefix_anchor(65,644) before consuming it.
                    _pending_anchors = [
                        a
                        for a in _snapshot_positions
                        if (
                            a not in _captured_positions
                            and pos < a < end_pos
                        )
                    ]

                    if _pending_anchors:
                        end_pos = min(
                            _pending_anchors
                        )

                    if end_pos <= pos:
                        raise RuntimeError(
                            "prefix block replay made no progress: "
                            f"pos={pos} end={end_pos}"
                        )

                    chunk_ids = prompt_ids[
                        pos:end_pos
                    ]

                    if set_mode is not None:
                        set_mode(True)

                    try:
                        logits = model.forward(
                            torch.tensor(
                                [chunk_ids],
                                dtype=torch.long,
                            ),
                            pos,
                        )
                    finally:
                        if set_mode is not None:
                            set_mode(False)

                    if use_mtp_tail:
                        self._write_prefix_draft_hidden(model.main_hidden, end_pos)
                    n = end_pos - pos
                    done += n
                    pos = end_pos

                    # Synchronize only at reporting boundaries.
                    # Do NOT introduce a per-token synchronization.
                    should_report = (
                        done % report_every < n
                        or pos == decode_end
                    )

                    if should_report:
                        if (
                            torch.is_tensor(logits)
                            and logits.is_cuda
                        ):
                            torch.cuda.synchronize(
                                logits.device
                            )

                        now = time.perf_counter()

                        print(
                            f"[prefix-replay-block] "
                            f"start={start_pos:,} "
                            f"done={done:,}/"
                            f"{replay_total:,} "
                            f"pos={pos:,} "
                            f"block={block_size:,} "
                            f"time={now-t0:.3f}s "
                            f"tok_s="
                            f"{done/max(now-t0,1e-9):,.1f}",
                            flush=True,
                        )

            finally:
                if old_collect is not None:
                    model.collect_main_hidden = (
                        old_collect
                    )

        else:
            # --------------------------------------------------------
            # Existing production scalar replay.
            # --------------------------------------------------------
            for pos in range(
                start_pos,
                decode_end,
            ):

                _capture_prefix_anchor(pos)

                token = int(
                    prompt_ids[pos]
                )

                logits = self.rt.step(
                    token,
                    pos,
                )

                if use_mtp_tail:
                    hidden = torch.cat([
                        self.rt.main_hid[lid][0:1].to(self.ds.device)
                        for lid in self.ds.targets
                    ], dim=-1)
                    self._write_prefix_draft_hidden(hidden, pos + 1)

                # Local CUDA synchronization for prefix-replay debugging.
                # Unlike CUDA_LAUNCH_BLOCKING=1 this does not serialize
                # model loading / graph capture during startup.
                if os.environ.get(
                    "DSV41_DEBUG_PREFIX_SYNC",
                    "0",
                ) == "1":
                    for _d in getattr(
                        self.rt,
                        "devs",
                        [],
                    ):
                        torch.cuda.synchronize(_d)

                    if (
                        pos % 32 == 0
                        or 32600 <= pos <= 32850
                    ):
                        print(
                            f"[prefix-sync] "
                            f"pos={pos:,}=OK",
                            flush=True,
                        )

                done += 1

                if (
                    done % report_every == 0
                    or pos + 1 == decode_end
                ):
                    now = time.perf_counter()

                    print(
                        f"[prefix-replay-decode] "
                        f"start={start_pos:,} "
                        f"done={done:,}/"
                        f"{replay_total:,} "
                        f"pos={pos+1:,} "
                        f"time={now-t0:.3f}s "
                        f"tok_s="
                        f"{done/max(now-t0,1e-9):,.1f}",
                        flush=True,
                    )

        # ------------------------------------------------------------
        # Final MTP prompt token.
        #
        # The critical part:
        #
        #     model.collect_main_hidden = self.ds.targets
        #
        # Model.forward() only creates model.main_hidden when this set
        # contains target block ids.
        # ------------------------------------------------------------
        if use_mtp_tail:
            pos = total - 1
            token = int(prompt_ids[pos])

            _capture_prefix_anchor(pos)

            ds = self.ds
            model = self.model

            targets = tuple(ds.targets)

            if not targets:
                raise RuntimeError(
                    "MTP DSpark has empty targets"
                )

            # Remove any stale capture so success below proves that this
            # exact final-token forward generated main_hidden.
            if hasattr(model, "main_hidden"):
                delattr(model, "main_hidden")

            old_collect = getattr(
                model,
                "collect_main_hidden",
                None,
            )

            model.collect_main_hidden = targets

            print(
                f"[prefix-replay-mtp-tail] "
                f"collect_main_hidden={targets} "
                f"pos={pos:,}",
                flush=True,
            )

            set_mode = getattr(
                self,
                "_set_prefix_replay_mode",
                None,
            )

            if set_mode is not None:
                set_mode(True)

            try:
                logits = model.forward(
                    torch.tensor(
                        [[token]],
                        dtype=torch.long,
                    ),
                    pos,
                )
            finally:
                if set_mode is not None:
                    set_mode(False)

                # Normal Engine initialization normally keeps this set
                # permanently. Preserve whatever policy existed before
                # this replay instead of changing global behavior.
                if old_collect is None:
                    try:
                        delattr(
                            model,
                            "collect_main_hidden",
                        )
                    except AttributeError:
                        pass
                else:
                    model.collect_main_hidden = old_collect

            done += 1

            if not hasattr(model, "main_hidden"):
                raise RuntimeError(
                    "prefix MTP tail forward did not create "
                    "model.main_hidden; "
                    f"targets={targets} pos={pos}"
                )

            mh = model.main_hidden
            self._write_prefix_draft_hidden(mh, total)

            if not torch.is_tensor(mh):
                raise RuntimeError(
                    "prefix MTP tail produced non-tensor "
                    f"main_hidden: {type(mh)}"
                )

            if mh.numel() == 0:
                raise RuntimeError(
                    "prefix MTP tail produced empty main_hidden"
                )

            # Expected model.forward representation is [B,S,3*dim].
            if mh.ndim < 2:
                raise RuntimeError(
                    "prefix MTP tail main_hidden has invalid shape "
                    f"{tuple(mh.shape)}"
                )

            now = time.perf_counter()

            print(
                f"[prefix-replay-mtp-tail] "
                f"main_hidden=OK "
                f"shape={tuple(mh.shape)} "
                f"dtype={mh.dtype} "
                f"device={mh.device} "
                f"done={done:,}/{replay_total:,} "
                f"time={now-t0:.3f}s "
                f"tok_s={done/max(now-t0,1e-9):,.1f}",
                flush=True,
            )

        # Snapshot at total means state after the entire prompt.
        _capture_prefix_anchor(total)

        if _multi_snapshot:
            new_snapshots.sort(
                key=lambda x: x[0]
            )
            return logits, new_snapshots

        return logits, new_snapshot

    def _prefix_tmpfs_dir(self):
        """tmpfs directory used only for process-restart persistence.

        /dev/shm is RAM-backed on Linux, so this does not touch SSD.
        Override with DSV41_PREFIX_CACHE_DIR.
        """
        path = os.environ.get(
            "DSV41_PREFIX_CACHE_DIR",
            "/dev/shm/dsv41-prefix-cache",
        )

        if not path:
            return None

        try:
            os.makedirs(path, exist_ok=True)
        except Exception as exc:
            print(
                f"[prefix-tmpfs] DISABLED "
                f"dir={path!r} error={exc}",
                flush=True,
            )
            return None

        return path

    @staticmethod
    def _prefix_tmpfs_hash(base_ids):
        """Stable filename derived from the cached token prefix."""
        a = array(
            "q",
            (int(x) for x in base_ids),
        )

        return hashlib.sha256(
            a.tobytes()
        ).hexdigest()[:24]

    @staticmethod
    def _prefix_slot_signature(slot):
        """Serializable identity for one mutable cache slot.

        holder object identity is intentionally NOT included because
        Python/model objects are recreated after server restart.
        """
        kind, holder, key, tensor = slot

        shape = tuple(int(x) for x in tensor.shape)
        if (
            (kind == "dict" and isinstance(key, tuple) and key)
            or (kind == "attr" and key == "cache" and holder.__class__.__name__ == "NgramHashState")
            or (kind == "attr" and (key == "window_kv_cache" or "window_kv" in str(key)))
        ):
            try:
                shape = (shape[0], -1, *shape[2:])
            except Exception:
                pass
        return {
            "kind": str(kind),
            "holder_class": (
                holder.__class__.__module__
                + "."
                + holder.__class__.__qualname__
            ),
            "key": repr(key),
            "shape": shape,
            "dtype": str(tensor.dtype),
        }

    def _prefix_current_slot_signatures(self):
        slots = self._prefix_cache_slots()

        return (
            slots,
            [
                self._prefix_slot_signature(x)
                for x in slots
            ],
        )

    def _persist_prefix_cache_entry(
        self,
        ent,
        previous=None,
    ):
        """Persist one host-RAM prefix snapshot into tmpfs.

        torch.save is acceptable here because the target is tmpfs, not
        SSD.  The live restore path still uses the in-process pinned
        tensors and does not read this file unless the server restarts.
        """
        root = self._prefix_tmpfs_dir()

        if root is None:
            return

        snapshot = ent.get("snapshot")

        if not snapshot:
            return

        base_ids = ent["base_ids"]

        h = self._prefix_tmpfs_hash(base_ids)

        final_path = os.path.join(
            root,
            f"prefix-{len(base_ids):09d}-{h}.pt",
        )

        tmp_path = (
            final_path
            + f".tmp-{os.getpid()}-{time.time_ns()}"
        )

        slot_signatures = [
            self._prefix_slot_signature(slot)
            for slot in snapshot
        ]

        # Save only portable data.  Never save holder objects.  Large
        # canonical cache tensors are split into immutable 1024-row blocks;
        # anchors then share blocks by content hash instead of duplicating
        # the entire prefix.
        block_rows = int(os.environ.get("DSV41_PREFIX_BLOCK_ROWS", "1024"))
        block_root = os.path.join(root, "blocks")
        os.makedirs(block_root, exist_ok=True)
        tensors = []
        block_refs = []
        unique_refs = set()
        reused_blocks = 0
        new_blocks = 0
        physical_delta = 0
        for kind, holder, key, host in snapshot:
            h = host.detach().cpu()
            if kind == "dict" and h.ndim >= 2 and h.shape[1] > block_rows:
                refs = []
                for start in range(0, h.shape[1], block_rows):
                    blk = h[:, start:start + block_rows].contiguous()
                    digest = hashlib.sha256(blk.view(torch.uint8).numpy().tobytes()).hexdigest()[:32]
                    path = os.path.join(block_root, digest + ".pt")
                    if not os.path.exists(path):
                        tmp = path + f".tmp-{os.getpid()}-{time.time_ns()}"
                        torch.save(blk, tmp)
                        os.replace(tmp, path)
                        new_blocks += 1
                        try:
                            physical_delta += os.path.getsize(path)
                        except OSError:
                            pass
                    else:
                        reused_blocks += 1
                    unique_refs.add(digest)
                    refs.append(digest)
                block_refs.append({"rows": int(h.shape[1]), "refs": refs})
                tensors.append(torch.empty((0,), dtype=h.dtype))
            else:
                block_refs.append(None)
                tensors.append(h)

        payload = {
            "format": "dsv41-prefix-tmpfs-v2-blocks",
            "block_rows": block_rows,
            "block_refs": block_refs,
            "cache_epoch": os.environ.get(
                "DSV41_PREFIX_CACHE_EPOCH",
                "fullstate-v5-index-engram-dspark",
            ),
            "created_at": time.time(),
            "prompt_ids": list(ent["prompt_ids"]),
            "base_ids": list(base_ids),
            "bytes": int(ent["bytes"]),
            "slot_signatures": slot_signatures,
            "tensors": tensors,
            "max_seq_len": int(self.max_seq_len),
            "model_id": str(
                getattr(
                    self.tok,
                    "name_or_path",
                    "",
                )
            ),
        }

        t0 = time.perf_counter()

        try:
            torch.save(
                payload,
                tmp_path,
            )

            os.replace(
                tmp_path,
                final_path,
            )

        except Exception as exc:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass

            print(
                f"[prefix-tmpfs] SAVE-FAILED "
                f"base={len(base_ids):,} "
                f"error={exc}",
                flush=True,
            )
            return

        dt = time.perf_counter() - t0

        ent["persist_path"] = final_path

        print(
            f"[prefix-tmpfs] SAVE "
            f"base={len(base_ids):,} "
            f"size={ent['bytes']/2**20:,.1f}MiB "
            f"time={dt:.3f}s "
            f"path={final_path}",
            flush=True,
        )

        # Report physical immutable-block accounting separately from the
        # logical snapshot payload.  This is intentionally derived from the
        # files on disk so it remains meaningful after a restart.
        try:
            physical_total = sum(
                os.path.getsize(os.path.join(block_root, name))
                for name in os.listdir(block_root)
                if name.endswith(".pt")
            )
            manifest_bytes = os.path.getsize(final_path)
        except OSError:
            physical_total = 0
            manifest_bytes = 0
        logical_bytes = int(ent.get("bytes", 0))
        dedup_ratio = (
            logical_bytes / physical_total
            if physical_total
            else 0.0
        )
        print(
            f"[prefix-store] anchor={len(base_ids):,} "
            f"logical={logical_bytes/2**20:,.1f}MiB "
            f"unique_blocks={len(unique_refs)} "
            f"reused_blocks={reused_blocks} new_blocks={new_blocks} "
            f"physical_total={physical_total/2**20:,.1f}MiB "
            f"physical_delta={physical_delta/2**20:,.1f}MiB "
            f"manifest_bytes={manifest_bytes} "
            f"dedup_ratio={dedup_ratio:.3f}",
            flush=True,
        )

        # If this entry replaced an older snapshot from the same live
        # cache family, remove the obsolete tmpfs file by default.
        if (
            previous is not None
            and not bool(
                int(
                    os.environ.get(
                        "DSV41_PREFIX_CACHE_KEEP_HISTORY",
                        "0",
                    )
                )
            )
        ):
            old_path = previous.get(
                "persist_path"
            )

            if (
                old_path
                and old_path != final_path
            ):
                try:
                    os.unlink(old_path)

                    print(
                        f"[prefix-tmpfs] DROP-OLD "
                        f"path={old_path}",
                        flush=True,
                    )
                except FileNotFoundError:
                    pass
                except Exception as exc:
                    print(
                        f"[prefix-tmpfs] "
                        f"DROP-OLD-WARN {exc}",
                        flush=True,
                    )

                # Reclaim immutable blocks only after the old manifest has
                # been removed.  Every remaining manifest is scanned and
                # its block references are retained, so shared blocks can
                # never be collected while another anchor uses them.
                if os.environ.get("DSV41_PREFIX_BLOCK_GC", "1") == "1":
                    self._gc_prefix_blocks(root)

        self._prune_prefix_tmpfs()

    @staticmethod
    def _gc_prefix_blocks(root):
        block_root = os.path.join(root, "blocks")
        if not os.path.isdir(block_root):
            return
        refs = set()
        for name in os.listdir(root):
            if not (name.startswith("prefix-") and name.endswith(".pt")):
                continue
            try:
                payload = torch.load(
                    os.path.join(root, name),
                    map_location="cpu",
                    weights_only=False,
                )
                for spec in payload.get("block_refs", []) or []:
                    if spec:
                        refs.update(spec.get("refs", []))
            except Exception:
                # A malformed manifest is left for fsck; never collect
                # blocks based on incomplete metadata.
                return
        removed = 0
        removed_bytes = 0
        for name in os.listdir(block_root):
            if not name.endswith(".pt") or name[:-3] in refs:
                continue
            path = os.path.join(block_root, name)
            try:
                removed_bytes += os.path.getsize(path)
                os.unlink(path)
                removed += 1
            except OSError:
                pass
        if removed:
            print(
                f"[prefix-store-gc] removed={removed} "
                f"bytes={removed_bytes/2**20:.1f}MiB "
                f"referenced={len(refs)}",
                flush=True,
            )

    def _prune_prefix_tmpfs(self):
        """Bound tmpfs usage using the same entry/count style limits."""
        root = self._prefix_tmpfs_dir()

        if root is None:
            return

        try:
            paths = [
                os.path.join(root, name)
                for name in os.listdir(root)
                if (
                    name.startswith("prefix-")
                    and name.endswith(".pt")
                )
            ]

            paths.sort(
                key=lambda x: os.stat(x).st_mtime,
                reverse=True,
            )

        except Exception:
            return

        max_entries = max(
            1,
            int(
                os.environ.get(
                    "DSV41_PREFIX_TMPFS_ENTRIES",
                    os.environ.get(
                        "DSV41_PREFIX_CACHE_ENTRIES",
                        "512",
                    ),
                )
            ),
        )

        max_gb = float(
            os.environ.get(
                "DSV41_PREFIX_TMPFS_GB",
                "512",
            )
        )

        max_bytes = int(
            max_gb * 2**30
        )

        total = 0
        keep = []

        for path in paths:
            try:
                sz = os.path.getsize(path)
            except OSError:
                continue

            if (
                len(keep) < max_entries
                and total + sz <= max_bytes
            ):
                keep.append(path)
                total += sz
                continue

            try:
                os.unlink(path)

                print(
                    f"[prefix-tmpfs] EVICT "
                    f"path={path}",
                    flush=True,
                )
            except Exception:
                pass

    def _load_prefix_cache_from_tmpfs(
        self,
        entries,
    ):
        """Load saved snapshots after a Python process restart."""
        root = self._prefix_tmpfs_dir()

        if root is None:
            return

        try:
            paths = [
                os.path.join(root, name)
                for name in os.listdir(root)
                if (
                    name.startswith("prefix-")
                    and name.endswith(".pt")
                )
            ]

            # newest first
            paths.sort(
                key=lambda x: os.stat(x).st_mtime,
                reverse=True,
            )

        except Exception as exc:
            print(
                f"[prefix-tmpfs] SCAN-FAILED "
                f"error={exc}",
                flush=True,
            )
            return

        if not paths:
            print(
                f"[prefix-tmpfs] LOAD "
                f"entries=0 dir={root}",
                flush=True,
            )
            return

        current_slots, current_sig = (
            self._prefix_current_slot_signatures()
        )

        current_model = str(
            getattr(
                self.tok,
                "name_or_path",
                "",
            )
        )

        max_entries = max(
            1,
            int(
                os.environ.get(
                    "DSV41_PREFIX_CACHE_ENTRIES",
                    "16",
                )
            ),
        )

        max_gb = float(
            os.environ.get(
                "DSV41_PREFIX_CACHE_GB",
                "64",
            )
        )

        max_bytes = int(
            max_gb * 2**30
        )

        total_bytes = 0
        loaded = 0
        skipped = 0

        for path in paths:

            if loaded >= max_entries:
                break

            try:
                payload = torch.load(
                    path,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                # Older torch without weights_only=
                try:
                    payload = torch.load(
                        path,
                        map_location="cpu",
                    )
                except Exception as exc:
                    print(
                        f"[prefix-tmpfs] SKIP "
                        f"path={path} error={exc}",
                        flush=True,
                    )
                    skipped += 1
                    continue
            except Exception as exc:
                print(
                    f"[prefix-tmpfs] SKIP "
                    f"path={path} error={exc}",
                    flush=True,
                )
                skipped += 1
                continue

            if payload.get("format") not in (
                "dsv41-prefix-tmpfs-v1",
                "dsv41-prefix-tmpfs-v2-blocks",
            ):
                skipped += 1
                continue

            _expected_epoch = os.environ.get(
                "DSV41_PREFIX_CACHE_EPOCH",
                "fullstate-v5-index-engram-dspark",
            )

            if (
                payload.get("cache_epoch")
                != _expected_epoch
            ):
                print(
                    f"[prefix-tmpfs] SKIP "
                    f"reason=epoch "
                    f"saved={payload.get('cache_epoch')!r} "
                    f"expected={_expected_epoch!r} "
                    f"path={path}",
                    flush=True,
                )
                skipped += 1
                continue

            if (
                int(
                    payload.get(
                        "max_seq_len",
                        -1,
                    )
                )
                != int(self.max_seq_len)
            ):
                print(
                    f"[prefix-tmpfs] SKIP "
                    f"reason=max_seq_len "
                    f"path={path}",
                    flush=True,
                )
                skipped += 1
                continue

            saved_model = str(
                payload.get(
                    "model_id",
                    "",
                )
            )

            if (
                saved_model
                and current_model
                and saved_model != current_model
            ):
                print(
                    f"[prefix-tmpfs] SKIP "
                    f"reason=model "
                    f"path={path}",
                    flush=True,
                )
                skipped += 1
                continue

            saved_sig = payload.get(
                "slot_signatures",
                [],
            )

            if saved_sig != current_sig:
                print(
                    f"[prefix-tmpfs] SKIP "
                    f"reason=slot-layout "
                    f"path={path}",
                    flush=True,
                )
                skipped += 1
                continue

            tensors = payload.get(
                "tensors",
                [],
            )
            block_refs = payload.get("block_refs") or []
            if block_refs and len(block_refs) == len(tensors):
                block_root = os.path.join(root, "blocks")
                rebuilt = list(tensors)
                try:
                    for i, spec in enumerate(block_refs):
                        if not spec:
                            continue
                        parts = [torch.load(os.path.join(block_root, ref + ".pt"), map_location="cpu", weights_only=False) for ref in spec["refs"]]
                        rebuilt[i] = torch.cat(parts, dim=1)
                    tensors = rebuilt
                except Exception as exc:
                    print(f"[prefix-tmpfs] SKIP reason=missing-block error={exc} path={path}", flush=True)
                    skipped += 1
                    continue

            if (
                len(tensors)
                != len(current_slots)
            ):
                skipped += 1
                continue

            snap = []
            bad = False

            for current_slot, src in zip(
                current_slots,
                tensors,
            ):
                kind, holder, key, dst = (
                    current_slot
                )

                if kind == "scalar_attr":
                    if (
                        not torch.is_tensor(src)
                        or src.numel() != 1
                        or src.dtype != torch.int64
                    ):
                        bad = True
                        break

                    snap.append(
                        (
                            kind,
                            holder,
                            key,
                            src.detach().cpu().clone(),
                        )
                    )
                    continue

                if (
                    torch.is_tensor(src)
                    and torch.is_tensor(dst)
                    and src.dtype == dst.dtype
                    and kind == "dict"
                    and src.ndim == dst.ndim
                    and tuple(src.shape[:1]) == tuple(dst.shape[:1])
                    and tuple(src.shape[2:]) == tuple(dst.shape[2:])
                    and src.shape[1] > dst.shape[1]
                ):
                    try:
                        owner = key[0] if isinstance(key, tuple) else None
                        table_kind = "index_k" if "index_k" in str(key) or "index_k" in str(holder) else "compress_kv"
                        shared = getattr(self.model, "shared", None)
                        if shared and hasattr(shared, "grow_cache_rows") and owner is not None:
                            shared.grow_cache_rows(table_kind, owner, src.shape[1])
                            table = getattr(shared, table_kind, None)
                            if table and key in table:
                                dst = table[key]
                    except Exception as grow_err:
                        pass

                _shape_ok = (
                    torch.is_tensor(src)
                    and torch.is_tensor(dst)
                    and src.dtype == dst.dtype
                    and (
                        tuple(src.shape) == tuple(dst.shape)
                        or (
                            kind in ("dict", "attr")
                            and src.ndim == dst.ndim
                            and tuple(src.shape[:1]) == tuple(dst.shape[:1])
                            and tuple(src.shape[2:]) == tuple(dst.shape[2:])
                            and src.shape[1] <= dst.shape[1]
                        )
                    )
                )
                if not _shape_ok:
                    bad = True
                    break

                # Restore HOT cache tensors as pinned host RAM when
                # possible, preserving the fast H2D restore path.
                try:
                    host = torch.empty_like(
                        src,
                        device="cpu",
                        pin_memory=True,
                    )
                except Exception:
                    host = torch.empty_like(
                        src,
                        device="cpu",
                    )

                host.copy_(src)

                snap.append(
                    (
                        kind,
                        holder,
                        key,
                        host,
                    )
                )

            if bad:
                print(
                    f"[prefix-tmpfs] SKIP "
                    f"reason=tensor-layout "
                    f"path={path}",
                    flush=True,
                )
                skipped += 1
                continue

            nbytes = int(
                payload.get(
                    "bytes",
                    sum(
                        x.numel()
                        * x.element_size()
                        for x in tensors
                    ),
                )
            )

            if (
                total_bytes + nbytes
                > max_bytes
            ):
                continue

            ent = {
                "prompt_ids": list(
                    payload["prompt_ids"]
                ),
                "base_ids": list(
                    payload["base_ids"]
                ),
                "snapshot": snap,
                "bytes": nbytes,
                "last_used": float(
                    payload.get(
                        "created_at",
                        time.time(),
                    )
                ),
                "hits": 0,
                "persist_path": path,
            }

            entries.append(ent)

            loaded += 1
            total_bytes += nbytes

            print(
                f"[prefix-tmpfs] LOADED "
                f"base={len(ent['base_ids']):,} "
                f"size={nbytes/2**20:,.1f}MiB",
                flush=True,
            )

        print(
            f"[prefix-tmpfs] LOAD-DONE "
            f"entries={loaded} "
            f"ram={total_bytes/2**30:,.2f}GiB "
            f"skipped={skipped} "
            f"dir={root}",
            flush=True,
        )

    def _prefix_cache_entries(self):
        """Host-RAM prefix snapshot LRU.

        On the first access after process startup, recover compatible
        snapshots from tmpfs so debugging restarts do not require a
        fresh prefill.
        """
        entries = getattr(
            self,
            "_rolling_prefix_entries",
            None,
        )

        if entries is None:
            entries = []
            self._rolling_prefix_entries = entries

        if not getattr(
            self,
            "_prefix_tmpfs_loaded",
            False,
        ):
            # Mark before loading to avoid accidental recursion.
            self._prefix_tmpfs_loaded = True

            self._load_prefix_cache_from_tmpfs(
                entries
            )

        return entries

    def _find_prefix_cache_entry(
        self,
        prompt_ids,
    ):
        """Find the deepest cached base that is an exact prefix.

        This allows unrelated Claude/LiteLLM request families to coexist
        instead of constantly replacing one global last-request cache.
        """
        entries = self._prefix_cache_entries()

        best = None
        best_base = 0

        for ent in entries:
            base_ids = ent["base_ids"]
            n = len(base_ids)

            if n <= best_base:
                continue

            if len(prompt_ids) < n:
                continue

            if prompt_ids[:n] == base_ids:
                best = ent
                best_base = n

        return best

    def _store_prefix_cache_entry(
        self,
        *,
        prompt_ids,
        base_ids,
        snapshot,
        snapshot_bytes,
        previous=None,
    ):
        # Do not rewrite an identical stable base.
        #
        # A HIT may finish without advancing the snapshot base.  In that
        # case the old snapshot is already the exact state we want.
        # Re-inserting it only burns host copies and ~405 MiB tmpfs I/O.
        if previous is not None:
            _prev_base = previous.get("base_ids")
            if (
                _prev_base is not None
                and list(_prev_base) == list(base_ids)
            ):
                previous["prompt_ids"] = list(prompt_ids)
                previous["last_used"] = time.time()
                previous["hits"] = int(
                    previous.get("hits", 0)
                ) + 1

                print(
                    f"[prefix-cache] SAME-BASE-SKIP "
                    f"base={len(base_ids):,} "
                    f"tmpfs=unchanged",
                    flush=True,
                )

                return previous

        """Insert/update one prefix snapshot and enforce LRU bounds."""
        entries = self._prefix_cache_entries()

        now = time.monotonic()

        ent = {
            "prompt_ids": list(prompt_ids),
            "base_ids": list(base_ids),
            "snapshot": snapshot,
            "bytes": int(snapshot_bytes),
            "last_used": now,
            "hits": (
                int(previous.get("hits", 0))
                if previous is not None
                else 0
            ),
        }

        if previous is not None:
            try:
                entries.remove(previous)
            except ValueError:
                pass

        entries.append(ent)

        max_entries = max(
            1,
            int(
                os.environ.get(
                    "DSV41_PREFIX_CACHE_ENTRIES",
                    "64",
                )
            ),
        )

        max_gb = float(
            os.environ.get(
                "DSV41_PREFIX_CACHE_GB",
                "128",
            )
        )

        max_bytes = int(max_gb * (2**30))

        # Oldest first.
        entries.sort(
            key=lambda x: x["last_used"]
        )

        total_bytes = sum(
            e["bytes"]
            for e in entries
        )

        while (
            len(entries) > max_entries
            or total_bytes > max_bytes
        ):
            victim = entries.pop(0)
            total_bytes -= victim["bytes"]

            print(
                f"[prefix-cache] EVICT "
                f"base={len(victim['base_ids']):,} "
                f"size={victim['bytes']/2**20:,.1f}MiB",
                flush=True,
            )
            if self.stats_tracker is not None:
                try:
                    self.stats_tracker.record_cache_event(
                        f"Prefix cache evict: base={len(victim['base_ids']):,} tokens ({victim['bytes']/2**20:.1f} MiB)",
                        event_type="evict",
                    )
                except Exception:
                    pass

        print(
            f"[prefix-cache] STORE "
            f"base={len(base_ids):,} "
            f"entries={len(entries)} "
            f"ram={total_bytes/2**30:,.2f}GiB",
            flush=True,
        )

        self._persist_prefix_cache_entry(
            ent,
            previous=previous,
        )

        if self.stats_tracker is not None:
            try:
                self.stats_tracker.record_cache_event(
                    f"Prefix cache saved: base={len(base_ids):,} tokens, snapshot={snapshot_bytes / (1024**2):.1f} MiB, RAM total={total_bytes / (1024**3):.2f} GiB",
                    event_type="store",
                )
            except Exception:
                pass

        return ent

    def _prefix_cache_stats(self):
        entries = self._prefix_cache_entries()

        total = sum(
            e["bytes"]
            for e in entries
        )

        return len(entries), total

    def _find_best_gpu_slot(self, prompt_ids: list[int]) -> tuple[int, int]:
        """Find the GPU sequence slot with the longest common prefix against prompt_ids.
        Returns (best_slot, best_lcp_length). If no slot matches, returns (-1, 0)."""
        best_slot = -1
        best_lcp = 0
        with getattr(self, "_slot_tokens_lock", threading.Lock()):
            items = list(self._slot_tokens.items())

        for slot_id, tokens in items:
            if not tokens:
                continue
            lcp = self._prompt_lcp(tokens, prompt_ids)
            # Prefer slot 0 on tie because slot 0 is already the prefill scratchpad
            if lcp > best_lcp or (lcp == best_lcp and slot_id == 0 and best_slot != 0):
                best_lcp = lcp
                best_slot = slot_id
        return best_slot, best_lcp

    @torch.inference_mode()
    def _forward_prefix_continuation(
        self,
        prompt_ids: list[int],
        start_pos: int,
        chunk_size: int = 512,
    ) -> torch.Tensor:
        """Prefill only the tail tokens [start_pos : len(prompt_ids)] using model.forward.
        Slot 0 must already contain the valid KV cache up to start_pos."""
        total = len(prompt_ids)
        if start_pos >= total:
            raise ValueError(f"start_pos {start_pos} >= total {total}")

        chunk_size = max(
            16,
            int(os.environ.get("DSV41_GPU_PREFIX_CHUNK_SIZE", str(chunk_size))),
        )

        use_mtp_tail = bool(
            getattr(self, "mtp", 0)
            and getattr(self, "ds", None) is not None
        )

        model = self.model
        old_collect = getattr(model, "collect_main_hidden", None)
        if old_collect is not None and use_mtp_tail:
            model.collect_main_hidden = tuple(self.ds.targets)

        self._set_prefix_replay_mode(True)
        pos = start_pos
        logits = None
        try:
            while pos < total:
                end_pos = min(total, pos + chunk_size)
                chunk_ids = prompt_ids[pos:end_pos]
                input_tensor = torch.tensor([chunk_ids], dtype=torch.long)
                logits = model.forward(input_tensor, pos)
                if use_mtp_tail and hasattr(model, "main_hidden") and model.main_hidden is not None:
                    self._write_prefix_draft_hidden(model.main_hidden, end_pos)
                pos = end_pos
        finally:
            self._set_prefix_replay_mode(False)

        if logits is None:
            raise RuntimeError(
                f"No logits produced in _forward_prefix_continuation (start_pos={start_pos}, total={total})"
            )
        return logits

    @torch.inference_mode()
    def _prefill_with_gpu_slot_reuse(
        self,
        prompt_ids: list[int],
        best_slot: int,
        best_lcp: int,
        req_id: str = "",
    ) -> tuple[torch.Tensor, int]:
        """Reuse existing GPU KV cache from best_slot, forward only the suffix delta on slot 0."""
        t0 = time.perf_counter()
        total = len(prompt_ids)

        # 1. If best_slot is not slot 0, copy its per-sequence state to slot 0 on GPU
        if best_slot != 0:
            self.rt.copy_seq(best_slot, 0, req_id=req_id)
            devices = getattr(self.rt, "devices", None) or getattr(self, "devices", None)
            if devices:
                for d in devices:
                    if isinstance(d, (int, torch.device)) or (isinstance(d, str) and "cuda" in str(d)):
                        try:
                            torch.cuda.synchronize(d)
                        except Exception:
                            pass

        # 2. Align reuse_pos to compression ratio boundary (multiple of 16).
        # If best_lcp matches the entire prompt, keep at least 1-16 tokens to compute logits.
        if best_lcp >= total:
            reuse_pos = max(0, total - 16)
        else:
            reuse_pos = (best_lcp // 16) * 16

        suffix_len = total - reuse_pos
        print(
            f"[gpu-slot-cache] HIT best_slot={best_slot} "
            f"lcp={best_lcp:,}/{total:,} ({best_lcp/total*100:.1f}%) "
            f"reuse_pos={reuse_pos:,} suffix={suffix_len:,}",
            flush=True,
        )

        # 3. Forward suffix tokens on slot 0
        chunk_size = int(os.environ.get("DSV41_GPU_PREFIX_CHUNK_SIZE", "512"))
        logits = self._forward_prefix_continuation(prompt_ids, reuse_pos, chunk_size=chunk_size)

        if getattr(logits, "is_cuda", False):
            torch.cuda.synchronize(logits.device)

        dt = time.perf_counter() - t0
        print(
            f"[gpu-slot-cache] COMPLETED suffix={suffix_len:,} in {dt:.3f}s "
            f"({suffix_len / max(dt, 1e-4):.1f} tok/s) "
            f"effective_speed={total / max(dt, 1e-4):,.1f} tok/s",
            flush=True,
        )

        self._record_prefill_stats(
            mode="gpu_slot_hit",
            prefill_type="gpu_hit",
            total=total,
            reused=reuse_pos,
            new_tokens=suffix_len,
            lcp=best_lcp,
            base_tokens=reuse_pos,
            suffix_tokens=suffix_len,
            dt=dt,
        )

        return logits, reuse_pos

    @torch.inference_mode()
    def _prefill_with_prefix_reuse(
        self,
        prompt_ids: list[int],
        images=None,
        token_types=None,
    ):
        if images is not None:
            _t0 = time.perf_counter()
            dev0 = self.model.blocks[0].device
            logits = self.model.forward(
                torch.tensor([prompt_ids], dtype=torch.long, device=dev0),
                0,
                images=images,
                token_types=token_types,
            )
            if torch.is_tensor(logits) and logits.is_cuda:
                torch.cuda.synchronize(logits.device)
            _dt = time.perf_counter() - _t0
            _n = len(prompt_ids)
            print(
                f"[vision-prefill] full-prefill tokens={_n:,} "
                f"time={_dt:.3f}s "
                f"tok_s={_n/max(_dt,1e-9):,.1f}",
                flush=True,
            )
            self._remember_prefill_prefix(prompt_ids)
            self._record_prefill_stats(
                total_tokens=_n,
                reused_tokens=0,
                suffix_tokens=_n,
                dt=_dt,
                prefill_type="cold_vision",
            )
            return logits, 0

        # ------------------------------------------------------------
        # Debug correctness switch.
        #
        # Completely bypass prefix snapshots / continuation replay and
        # perform a normal start_pos=0 full prefill.
        #
        #   DSV41_DISABLE_PREFIX_CACHE=1
        #
        # This is intentionally checked before any cache lookup/restore.
        # ------------------------------------------------------------
        _disable_prefix_cache = (
            os.environ.get(
                "DSV41_DISABLE_PREFIX_CACHE",
                "0",
            ) == "1"
            or bool(
                getattr(
                    self,
                    "_disable_prefix_cache_once",
                    False,
                )
            )
        )

        self._disable_prefix_cache_once = False

        if _disable_prefix_cache:
            _t0 = time.perf_counter()

            logits = self.model.forward(
                torch.tensor(
                    [prompt_ids],
                    dtype=torch.long,
                ),
                0,
            )

            # Synchronize the dependency chain before timing.
            if torch.is_tensor(logits) and logits.is_cuda:
                torch.cuda.synchronize(logits.device)

            _dt = time.perf_counter() - _t0
            _n = len(prompt_ids)

            print(
                f"[prefix-cache] DISABLED "
                f"full-prefill tokens={_n:,} "
                f"time={_dt:.3f}s "
                f"tok_s={_n/max(_dt,1e-9):,.1f}",
                flush=True,
            )

            return logits, 0

        """Rolling-prefix cache.

        Keep a host-RAM snapshot GUARD tokens before the end of the
        previous prompt.

        Claude Code commonly rewrites only a small tail of the prompt.
        If the next prompt still shares the snapshotted base, restore
        that cache and replay only the tail using the already-supported
        one-token continuation path.

        This avoids relying on unverified multi-token start_pos>0
        attention semantics.
        """
        guard = max(
            32,
            int(
                os.environ.get(
                    "DSV41_PREFIX_GUARD",
                    "256",
                )
            ),
        )

        total = len(prompt_ids)

        # ------------------------------------------------------------
        # Level 1 Cache: In-GPU Sequence Slot Cache.
        #
        # If any GPU decode slot or prefill slot 0 already retains a
        # prefix of prompt_ids in VRAM, avoid all host RAM restore, PCIe
        # traffic, and slow rollback.
        # ------------------------------------------------------------
        use_gpu_cache = (
            os.environ.get("DSV41_GPU_PREFIX_CACHE", "1") != "0"
        )
        min_gpu_prefix = int(
            os.environ.get("DSV41_GPU_PREFIX_MIN", "64")
        )
        best_gpu_slot, best_gpu_lcp = self._find_best_gpu_slot(prompt_ids)

        if use_gpu_cache and best_gpu_slot >= 0 and best_gpu_lcp >= min_gpu_prefix:
            try:
                logits, reused_pos = self._prefill_with_gpu_slot_reuse(
                    prompt_ids,
                    best_gpu_slot,
                    best_gpu_lcp,
                )
                return logits, reused_pos
            except Exception as exc:
                print(
                    f"[gpu-slot-cache] GPU continuation failed ({exc}), "
                    f"falling back to host snapshots...",
                    flush=True,
                )
                traceback.print_exc()

        # Search ALL retained host-RAM snapshots.
        entry = self._find_prefix_cache_entry(
            prompt_ids
        )

        if entry is not None:
            old_prompt = entry["prompt_ids"]
            old_base_ids = entry["base_ids"]
            old_snapshot = entry["snapshot"]
            old_base = len(old_base_ids)

            entry["last_used"] = time.monotonic()
            entry["hits"] = int(
                entry.get("hits", 0)
            ) + 1
        else:
            old_prompt = None
            old_base_ids = None
            old_snapshot = None
            old_base = 0

        # For diagnostics, also find the best LCP against every cached
        # request, even when none has a reusable base.
        lcp = 0
        lcp_prompt = None

        for candidate in self._prefix_cache_entries():
            cp = candidate["prompt_ids"]

            c = self._prompt_lcp(
                cp,
                prompt_ids,
            )

            if c > lcp:
                lcp = c
                lcp_prompt = cp

        target_base = max(
            0,
            total - guard,
        )

        reusable = (
            entry is not None
            and old_base > 0
        )

        t0 = time.perf_counter()

        # ========================================================
        # HIT: restore stable base and replay only its tail.
        # ========================================================
        if reusable:
            print(
                f"[prefix-cache] LCP-HIT "
                f"old={len(old_prompt):,} "
                f"new={total:,} "
                f"lcp={lcp:,} "
                f"base={old_base:,} "
                f"replay={total-old_base:,}",
                flush=True,
            )

            self._restore_prefix_state(
                old_snapshot
            )

            # Do NOT roll the snapshot forward on every request.
            #
            # Moving the snapshot to total-guard forces replay to split:
            #
            #   old_base -> new_base
            #   SAVE 405MB
            #   new_base -> total
            #
            # For small/medium suffixes this costs far more than simply
            # keeping the existing stable snapshot.
            #
            # Refresh only when replay from the current base becomes
            # sufficiently large.
            refresh_after = max(
                guard,
                int(
                    os.environ.get(
                        "DSV41_PREFIX_REFRESH_TOKENS",
                        "2048",
                    )
                ),
            )

            replay_from_base = total - old_base

            if replay_from_base >= refresh_after:
                next_base = max(
                    old_base,
                    target_base,
                )

                anchor_stride = max(
                    128,
                    int(
                        os.environ.get(
                            "DSV41_PREFIX_ANCHOR_STRIDE",
                            "1024",
                        )
                    ),
                )

                anchor_max = max(
                    1,
                    int(
                        os.environ.get(
                            "DSV41_PREFIX_ANCHOR_MAX",
                            "4",
                        )
                    ),
                )

                # Build anchors backwards from the newest useful base.
                #
                # Example:
                #
                #   old_base=36688
                #   next_base=40553
                #
                # becomes approximately:
                #
                #   37481, 38505, 39529, 40553
                #
                # This strongly favors the likely next Claude-Code
                # continuation while retaining some rewrite tolerance.
                anchors = [
                    next_base
                ]

                p = next_base - anchor_stride

                while (
                    p > old_base
                    and len(anchors) < anchor_max
                ):
                    anchors.append(p)
                    p -= anchor_stride

                anchors = sorted(
                    set(anchors)
                )

                snapshot_at = anchors

                print(
                    f"[prefix-anchor] "
                    f"PLAN old_base={old_base:,} "
                    f"target={next_base:,} "
                    f"stride={anchor_stride:,} "
                    f"anchors="
                    + ",".join(
                        f"{a:,}"
                        for a in anchors
                    ),
                    flush=True,
                )
            else:
                next_base = old_base
                snapshot_at = None

            logits, snap_at = self._replay_prefix_tail(
                prompt_ids,
                old_base,
                snapshot_at=snapshot_at,
            )

            if logits is None:
                # Identical prompt whose snapshot is at its end.
                # Recompute the final token from a fresh restore.
                self._restore_prefix_state(
                    old_snapshot
                )

                redo = max(0, total - 1)

                logits, _ = self._replay_prefix_tail(
                    prompt_ids,
                    redo,
                    snapshot_at=None,
                )

            if isinstance(snap_at, list):
                # Multi-anchor refresh.
                #
                # Remove/replace the old rolling entry only once.  Each
                # additional anchor becomes an independent prefix-cache
                # entry and is then governed by the existing cache LRU /
                # RAM / tmpfs policy.
                _previous = entry

                for (
                    anchor_pos,
                    anchor_snap_pair,
                ) in snap_at:
                    (
                        anchor_snapshot,
                        anchor_bytes,
                    ) = anchor_snap_pair

                    anchor_base_ids = list(
                        prompt_ids[:anchor_pos]
                    )

                    # Give each anchor its natural prefix as its request
                    # identity as well.  This avoids several anchors of
                    # the same full prompt collapsing onto one key in
                    # implementations that de-duplicate prompt_ids.
                    anchor_prompt_ids = list(
                        prompt_ids[:anchor_pos]
                    )

                    self._store_prefix_cache_entry(
                        prompt_ids=anchor_prompt_ids,
                        base_ids=anchor_base_ids,
                        snapshot=anchor_snapshot,
                        snapshot_bytes=anchor_bytes,
                        previous=_previous,
                    )

                    _previous = None

                    print(
                        f"[prefix-anchor] "
                        f"STORE base={anchor_pos:,} "
                        f"size={anchor_bytes/2**20:,.1f}MiB",
                        flush=True,
                    )

                # The newest anchor is the rolling entry represented by
                # this request for accounting below.
                if snap_at:
                    (
                        next_base,
                        (
                            next_snapshot,
                            snap_bytes,
                        ),
                    ) = snap_at[-1]

                    next_base_ids = list(
                        prompt_ids[:next_base]
                    )
                else:
                    next_snapshot = old_snapshot
                    next_base_ids = old_base_ids

            elif snap_at is not None:
                # Legacy single-anchor behavior.
                next_snapshot, snap_bytes = snap_at
                next_base_ids = list(
                    prompt_ids[:next_base]
                )

                next_bytes = sum(
                    h.numel() * h.element_size()
                    for _, _, _, h in next_snapshot
                )

                self._store_prefix_cache_entry(
                    prompt_ids=prompt_ids,
                    base_ids=next_base_ids,
                    snapshot=next_snapshot,
                    snapshot_bytes=next_bytes,
                    previous=entry,
                )

            else:
                # No refresh; keep existing rolling snapshot.
                next_snapshot = old_snapshot
                next_base_ids = old_base_ids

                next_bytes = sum(
                    h.numel() * h.element_size()
                    for _, _, _, h in next_snapshot
                )

                self._store_prefix_cache_entry(
                    prompt_ids=prompt_ids,
                    base_ids=next_base_ids,
                    snapshot=next_snapshot,
                    snapshot_bytes=next_bytes,
                    previous=entry,
                )

            dt = time.perf_counter() - t0

            replay_n = total - old_base

            print(
                f"[prefill-bench] "
                f"mode=LCP-HIT "
                f"new={replay_n:,} "
                f"reused={old_base:,} "
                f"total={total:,} "
                f"time={dt:.3f}s "
                f"new_tok_s="
                f"{replay_n/max(dt,1e-9):,.1f} "
                f"effective_tok_s="
                f"{total/max(dt,1e-9):,.1f}",
                flush=True,
            )

            self._record_prefill_stats(
                "LCP-HIT",
                total,
                old_base,
                replay_n,
                dt,
                prefill_type="host_replay",
                lcp=lcp,
                base_tokens=old_base,
                suffix_tokens=replay_n,
            )
            return logits, old_base

        # ========================================================
        # MISS
        # ========================================================

        entries, cache_bytes = self._prefix_cache_stats()

        print(
            f"[prefix-cache] MISS "
            f"new={total:,} "
            f"best_lcp={lcp:,} "
            f"entries={entries} "
            f"ram={cache_bytes/2**30:,.2f}GiB",
            flush=True,
        )

        # Very small prompts: ordinary full prefill.  There is no
        # meaningful stable base to snapshot yet.
        if target_base <= 0:
            logits = self.model.forward(
                torch.tensor(
                    [prompt_ids],
                    dtype=torch.long,
                ),
                0,
            )

            # Too small to snapshot.  Do not destroy unrelated
            # cached sessions.

            dt = time.perf_counter() - t0

            print(
                f"[prefill-bench] "
                f"mode=FULL "
                f"new={total:,} "
                f"reused=0 "
                f"total={total:,} "
                f"time={dt:.3f}s "
                f"new_tok_s="
                f"{total/max(dt,1e-9):,.1f} "
                f"effective_tok_s="
                f"{total/max(dt,1e-9):,.1f}",
                flush=True,
            )

            self._record_prefill_stats(
                "FULL",
                total,
                0,
                total,
                dt,
                prefill_type="cold",
                lcp=lcp,
                base_tokens=0,
                suffix_tokens=total,
            )
            return logits, 0

        # --------------------------------------------------------
        # Cold construction of rolling snapshot:
        #
        #   [ stable base ............. ][ guard tail ]
        #       full prefill                1-token replay
        #                ^
        #                SAVE
        # --------------------------------------------------------

        base_ids = prompt_ids[:target_base]

        _prefill_t0 = time.perf_counter()

        # Avoid materialising attention activations for a 100K+ prompt in
        # one call. Continuation chunks preserve the persistent caches and
        # keep the peak temporary allocation bounded by HC_PREFILL_CHUNK.
        _force_legacy_chunk = os.environ.get("DSV41_FORCE_LEGACY_BUILD_LOOP", "0") == "1"
        if not _force_legacy_chunk:
            logits = self.model.forward(
                torch.tensor([base_ids], dtype=torch.long),
                0,
            )
        else:
            _chunk = max(1, int(os.environ.get("DSV41_HC_PREFILL_CHUNK", "1024")))
            logits = None
            for _s in range(0, len(base_ids), _chunk):
                _e = min(_s + _chunk, len(base_ids))
                logits = self.model.forward(
                    torch.tensor([base_ids[_s:_e]], dtype=torch.long),
                    _s,
                )

        # CUDA work can still be asynchronous.  Synchronize only the
        # GPUs actually used by this model, not every CUDA device.
        _prefill_devices = set()

        for _obj in getattr(self.model, "layers", []):
            _dev = getattr(_obj, "device", None)
            if isinstance(_dev, torch.device) and _dev.type == "cuda":
                _prefill_devices.add(_dev)

        if not _prefill_devices:
            # Fallback: the returned logits device is sufficient for
            # timing the dependency chain in the common case.
            if torch.is_tensor(logits) and logits.is_cuda:
                _prefill_devices.add(logits.device)

        for _dev in _prefill_devices:
            torch.cuda.synchronize(_dev)

        _prefill_dt = time.perf_counter() - _prefill_t0
        _prefill_n = len(base_ids)

        print(
            f"[prefill-main] "
            f"tokens={_prefill_n:,} "
            f"time={_prefill_dt:.3f}s "
            f"tok_s={_prefill_n/max(_prefill_dt,1e-9):,.1f}",
            flush=True,
        )

        if getattr(self, "ds", None) is not None:
            self._write_prefix_draft_hidden(self.model.main_hidden, target_base)

        snap, snap_bytes = (
            self._snapshot_prefix_state(used_tokens=target_base)
        )

        logits2, _ = self._replay_prefix_tail(
            prompt_ids,
            target_base,
            snapshot_at=None,
        )

        if logits2 is not None:
            logits = logits2

        self._store_prefix_cache_entry(
            prompt_ids=prompt_ids,
            base_ids=base_ids,
            snapshot=snap,
            snapshot_bytes=snap_bytes,
            previous=None,
        )

        dt = time.perf_counter() - t0

        print(
            f"[prefix-cache] BASE "
            f"base={target_base:,} "
            f"guard={total-target_base:,} "
            f"snapshot={snap_bytes/2**20:,.1f}MiB",
            flush=True,
        )

        print(
            f"[prefill-bench] "
            f"mode=BUILD "
            f"new={total:,} "
            f"reused=0 "
            f"total={total:,} "
            f"time={dt:.3f}s "
            f"new_tok_s="
            f"{total/max(dt,1e-9):,.1f} "
            f"effective_tok_s="
            f"{total/max(dt,1e-9):,.1f}",
            flush=True,
        )

        self._record_prefill_stats(
            "BUILD",
            total,
            0,
            total,
            dt,
            prefill_type="cold",
            lcp=lcp,
            base_tokens=0,
            suffix_tokens=total,
        )
        return logits, 0

    @torch.inference_mode()
    def generate(self, prompt_ids: list[int], p: GenParams, images=None, token_types=None) -> Iterator[tuple[int, str]]:
        """Yields (token_id, text_piece) as they are produced. Holds the engine lock for the duration."""
        assert len(prompt_ids) < self.max_seq_len, f"prompt of {len(prompt_ids)} tokens exceeds max_seq_len={self.max_seq_len}"
        # MTP keeps DSpark resident on GPU4. At very long cold-prefill
        # lengths there may not be enough headroom for the indexer's
        # temporary buffers; use the regular decode path for this request
        # instead of returning HTTP 507. Prefix reuse and block replay stay
        # enabled. The threshold is configurable for larger-memory hosts.
        _mtp_long_limit = int(os.environ.get("DSV41_MTP_LONG_PROMPT_LIMIT", "65536"))
        if self.mtp and self.ds is None:
            # A previous long-prompt request may have unloaded DSpark. Do not
            # route a subsequent request into the MTP path without its state.
            print("[mtp] DSpark unavailable; forcing plain decode", flush=True)
            self.mtp = 0
        if self.mtp and len(prompt_ids) > _mtp_long_limit:
            print(f"[mtp] long-prompt fallback prompt={len(prompt_ids):,} limit={_mtp_long_limit:,} mode=plain", flush=True)
            saved_mtp = self.mtp
            self.mtp = 0
            # DSpark itself occupies most of GPU4. Drop its module before
            # the plain prefill; merely skipping speculative decoding does
            # not release those allocations and still yields HTTP 507.
            if self.ds is not None and os.environ.get("DSV41_MTP_UNLOAD_ON_LONG", "1") == "1":
                import gc
                self.ds = None
                self.model.collect_main_hidden = ()
                gc.collect()
                torch.cuda.empty_cache()
                print("[mtp] long-prompt DSpark unloaded", flush=True)
            # Keep plain mode after unloading DSpark. Restoring self.mtp here
            # races with concurrent Claude retries that can enter the MTP
            # path while ds is still absent. A server restart re-enables MTP.
            yield from self.generate(prompt_ids, p, images=images, token_types=token_types)
            return
        max_new = min(p.max_new_tokens, self.max_seq_len - len(prompt_ids) - 1)
        # Claude may request 32K output even for a 100K+ context. Keep
        # long-context requests bounded so the gateway can finish instead
        # of timing out during slow single-token decode. Override per host.
        if len(prompt_ids) > _mtp_long_limit:
            _long_cap = int(os.environ.get("DSV41_LONG_PROMPT_MAX_NEW", os.environ.get("DSV41_INTERACTIVE_MAX_NEW", "65536")))
            max_new = min(max_new, _long_cap)
        gen = None
        if p.seed is not None:
            gen = torch.Generator(device=self.model.blocks[-1].device)
            gen.manual_seed(p.seed)
        with self.lock:
            self.current_phase = "prefill"
            self.current_context_tokens = len(prompt_ids)
            try:
                if self.mtp and images is None:
                    yield from self._generate_mtp_locked(
                        prompt_ids,
                        p,
                        max_new,
                        gen,
                    )
                    return

                logits, _prefix_reused = self._prefill_with_prefix_reuse(
                    prompt_ids,
                    images=images,
                    token_types=token_types,
                )
                with getattr(self, "_slot_tokens_lock", threading.Lock()):
                    self._slot_tokens[0] = list(prompt_ids)
                self.current_phase = "decode"
                pos = len(prompt_ids)
                out: list[int] = []
                decoded_upto = 0
                pending = ""
                t0_single = time.perf_counter()
                last_single_log = t0_single
                with getattr(self, "_slot_lock", threading.Lock()):
                    st = self.slot_states.get(0, {})
                    st["slot_id"] = 0
                    st["status"] = "generating"
                    st["req_id"] = "single"
                    st["prompt_tokens"] = len(prompt_ids)
                    st["generated_tokens"] = 0
                    st["max_tokens"] = max_new
                    st["start_time"] = t0_single
                    st["first_tok_time"] = t0_single
                    st["recent_text"] = ""
                    st["tok_s"] = 0.0
                    self.slot_states[0] = st
                for step in range(max_new):
                    step_logits = logits[0]
                    if out and (
                        p.repetition_penalty != 1.0
                        or p.presence_penalty != 0.0
                        or p.frequency_penalty != 0.0
                        or p.progressive_penalty > 0.0
                        or p.ban_cycles
                    ):
                        step_logits = apply_penalties(
                            step_logits,
                            out,
                            repetition_penalty=p.repetition_penalty,
                            presence_penalty=p.presence_penalty,
                            frequency_penalty=p.frequency_penalty,
                            window=p.penalty_window,
                            progressive_penalty=p.progressive_penalty,
                            ban_cycles=p.ban_cycles,
                        )
                    t = sample_token(step_logits, p.temperature, p.top_p, gen)
                    if t == self.eos:
                        print(
                         f"[generate] STOP=eos step={step} pos={pos} "
                         f"max_new={max_new}",
                         flush=True,
                        )
                        break
                    out.append(t)
                    with getattr(self, "_slot_tokens_lock", threading.Lock()):
                        if 0 in self._slot_tokens:
                            self._slot_tokens[0].append(t)
                    if self.stats_tracker is not None:
                        try:
                            self.stats_tracker.record_decode_tokens(1)
                        except Exception:
                            pass
                    # decode incrementally; hold back a partial multi-byte character
                    text = self.tok.decode(out[decoded_upto:])
                    if "\ufffd" in text:
                        piece = ""
                    else:
                        piece, decoded_upto = text, len(out)
                    now_single = time.perf_counter()
                    dt_single = max(now_single - t0_single, 1e-4)
                    tok_s_single = round(len(out) / dt_single, 1)
                    with getattr(self, "_slot_lock", threading.Lock()):
                        st = self.slot_states.get(0, {})
                        st["generated_tokens"] = len(out)
                        st["tok_s"] = tok_s_single
                        st["elapsed_s"] = round(dt_single, 2)
                        st["recent_text"] = self.tok.decode(out[-100:], errors="replace")
                        self.slot_states[0] = st
                    if now_single - last_single_log >= 1.0:
                        last_single_log = now_single
                        tail = self.tok.decode(out[-30:], errors="replace").replace("\n", " ").replace("\r", "")
                        print(f"[decode] slot 0: {len(out)}/{max_new} tok ({tok_s_single:.1f} t/s) \"{tail}\"", flush=True)
                    if piece:
                        pending += piece
                        if p.stop and any(s in pending for s in p.stop):
                            cut = min(pending.find(s) for s in p.stop if s in pending)
                            if cut > 0:
                                yield t, pending[:cut]
                            return
                        yield t, pending
                        pending = ""
                    if getattr(p, "loop_detect", True):
                        min_match = getattr(p, "min_loop_match", 48)
                        min_cycle = getattr(p, "min_loop_cycle", 1)
                        if len(out) >= min_match + min_cycle:
                            loop_res = detect_loop(out, min_match=min_match, min_cycle=min_cycle)
                            if loop_res is not None:
                                cycle_len, trim_count, prev_pos = loop_res
                                print(
                                    f"[generate] REPETITION LOOP DETECTED on slot 0: "
                                    f"cycle={cycle_len} tokens, trimming {trim_count} duplicate tokens (prev_pos={prev_pos}). Stopping.",
                                    flush=True,
                                )
                                if trim_count > 0:
                                    del out[-trim_count:]
                                    with getattr(self, "_slot_tokens_lock", threading.Lock()):
                                        if 0 in self._slot_tokens:
                                            self._slot_tokens[0] = self._slot_tokens[0][:-trim_count]
                                return
                    logits = self.rt.step(t, pos)
                    pos += 1
                if out[decoded_upto:]:
                    tail = self.tok.decode(out[decoded_upto:])
                    if tail:
                        yield out[-1], tail
            finally:
                self.current_phase = "idle"
                dt_final = max(time.perf_counter() - t0_single, 1e-4) if "t0_single" in locals() else 0.0
                with getattr(self, "_slot_lock", threading.Lock()):
                    if 0 in self.slot_states:
                        st = self.slot_states[0]
                        st["status"] = "completed"
                        st["generated_tokens"] = len(out) if "out" in locals() else 0
                        st["tok_s"] = round(len(out) / dt_final, 1) if ("out" in locals() and dt_final > 0) else 0.0
                        st["elapsed_s"] = round(dt_final, 2)
                        st["recent_text"] = self.tok.decode(out[-1500:], errors="replace") if ("out" in locals() and out) else ""
                        st["completed_at"] = time.perf_counter()
                if "out" in locals() and dt_final > 0:
                    self.last_decode_tok_s = round(len(out) / dt_final, 1)

    def _generate_mtp_locked(
        self,
        prompt_ids: list[int],
        p: GenParams,
        max_new: int,
        gen: torch.Generator | None,
    ) -> Iterator[tuple[int, str]]:
        """Single-sequence DSpark speculative decoding.

        The caller already holds self.lock.
        """

        ds = self.ds
        rt = self.rt
        model = self.model

        assert ds is not None

        def _mtp_sync_probe(label):
            if os.environ.get("DSV41_DEBUG_MTP_SYNC", "0") != "1":
                return
            torch.cuda.synchronize(last if "last" in locals() else ds.device)
            print(
                f"[mtp-sync] {label}=OK",
                flush=True,
            )

        def _mtp_sync_probe(label):
            if os.environ.get("DSV41_DEBUG_MTP_SYNC", "0") != "1":
                return
            torch.cuda.synchronize(last if "last" in locals() else ds.device)
            print(
                f"[mtp-sync] {label}=OK",
                flush=True,
            )

        # ------------------------------------------------------------
        # Prefill
        # ------------------------------------------------------------

        logits, prefix_reused = self._prefill_with_prefix_reuse(
            prompt_ids
        )

        T = len(prompt_ids)
        last = model.blocks[-1].device

        _mtp_sync_probe("after-prefix-prefill")

        _mtp_sync_probe("after-prefix-prefill")

        if not hasattr(model, "main_hidden"):
            raise RuntimeError(
                "MTP prefill did not collect model.main_hidden"
            )

        # DSpark window attention only needs the tail of the main
        # sequence. Avoid feeding tens of thousands of rows through
        # DSpark just to overwrite the same ring.
        mh = model.main_hidden[0]

        win = int(ds.win)

        # mh contains only rows computed by THIS prefill call.
        mh_rows = mh.shape[0]
        mh_start_pos = T - mh_rows

        take = min(win, mh_rows)
        first_local = mh_rows - take
        first_pos = mh_start_pos + first_local

        tail = mh[first_local:]

        ds.write_main_rows(
            tail,
            torch.zeros(
                take,
                dtype=torch.int64,
                device=last,
            ),
            torch.arange(
                first_pos,
                T,
                dtype=torch.int64,
                device=last,
            ),
        )

        _mtp_sync_probe("after-write-main-rows")

        _mtp_sync_probe("after-write-main-rows")

        # Hidden state whose forward produced the first sampled token.
        main_h = mh[-1].clone()

        # Large prefill hidden capture is no longer needed.
        del model.main_hidden
        del mh, tail

        p_last = T - 1
        written_max = p_last

        current = sample_token(
            logits[0],
            p.temperature,
            p.top_p,
            gen,
        )

        if os.environ.get("DSV41_DEBUG_MTP_SYNC", "0") == "1":
            print(
                f"[mtp-sync] sampled-current="
                f"{int(current)} "
                f"logits_vocab={int(logits.shape[-1])}",
                flush=True,
            )
            torch.cuda.synchronize(last)
            print("[mtp-sync] after-sample=OK", flush=True)

        if os.environ.get("DSV41_DEBUG_MTP_SYNC", "0") == "1":
            print(
                f"[mtp-sync] sampled-current="
                f"{int(current)} "
                f"logits_vocab={int(logits.shape[-1])}",
                flush=True,
            )
            torch.cuda.synchronize(last)
            print("[mtp-sync] after-sample=OK", flush=True)

        # ------------------------------------------------------------
        # Streaming decoder state
        # ------------------------------------------------------------

        self.current_phase = "decode"
        out: list[int] = []
        decoded_upto = 0
        pending = ""
        produced = 0

        n_verify_steps = 0
        n_accepted = 0

        def emit(tokens):
            nonlocal decoded_upto
            nonlocal pending
            nonlocal produced

            for t in tokens:
                if produced >= max_new:
                    return True

                if t == self.eos:
                    print(
                        f"[generate] STOP=eos "
                        f"produced={produced} "
                        f"mtp_accept={n_accepted}/{n_verify_steps}",
                        flush=True,
                    )
                    return True

                out.append(int(t))
                produced += 1

                text = self.tok.decode(
                    out[decoded_upto:]
                )

                if "�" in text:
                    piece = ""
                else:
                    piece = text
                    decoded_upto = len(out)

                if piece:
                    pending += piece

                    if p.stop and any(
                        s in pending
                        for s in p.stop
                    ):
                        cut = min(
                            pending.find(s)
                            for s in p.stop
                            if s in pending
                        )

                        if cut > 0:
                            yield int(t), pending[:cut]

                        return True

                    yield int(t), pending
                    pending = ""

            return False

        # First token came directly from the prefill logits.
        stopped = yield from emit([current])

        if stopped:
            return

        # ------------------------------------------------------------
        # Speculative loop
        # ------------------------------------------------------------

        while produced < max_new:

            try:
                draft_all = ds.draft_rows(
                    torch.tensor(
                        [current],
                        dtype=torch.int64,
                        device=last,
                    ),
                    torch.tensor(
                        [p_last],
                        dtype=torch.int64,
                        device=last,
                    ),
                    main_h.view(1, -1),
                    torch.tensor(
                        [written_max],
                        dtype=torch.int64,
                        device=last,
                    ),
                )
            except Exception as e:
                print(
                    "\n[MTP DRAFT ERROR]",
                    repr(e),
                    flush=True,
                )
                traceback.print_exc()
                raise

            if os.environ.get(
                "DSV41_DEBUG_MTP_SYNC",
                "0",
            ) == "1":
                torch.cuda.synchronize(ds.device)
                print(
                    "[mtp-sync] after-draft-rows=OK",
                    flush=True,
                )

            drafts = (
                draft_all[0, :self.mtp]
                .to("cpu")
                .tolist()
            )

            K = 1 + len(drafts)

            row_tokens = [current] + drafts

            positions = list(
                range(
                    p_last + 1,
                    p_last + 1 + K,
                )
            )

            pmax = p_last + K

            # --------------------------------------------------------
            # MTP verifier input guard.
            #
            # Catch invalid token/position indices BEFORE CUDA sees
            # them. A bad embedding index otherwise kills the whole
            # CUDA context with IndexKernel.cu:94.
            # --------------------------------------------------------
            _embed = getattr(model, "embed", None)

            if _embed is None:
                _embed_rows = -1
            elif torch.is_tensor(_embed):
                _embed_rows = int(_embed.shape[0])
            elif hasattr(_embed, "weight"):
                _embed_rows = int(_embed.weight.shape[0])
            else:
                _embed_rows = -1

            _head = getattr(model, "head", None)

            if torch.is_tensor(_head):
                _head_rows = int(_head.shape[0])
            elif hasattr(_head, "weight"):
                _head_rows = int(_head.weight.shape[0])
            else:
                _head_rows = int(logits.shape[-1])

            _bad_tokens = []

            if _embed_rows > 0:
                _bad_tokens = [
                    (i, int(t))
                    for i, t in enumerate(row_tokens)
                    if int(t) < 0 or int(t) >= _embed_rows
                ]

            _max_seq = int(
                getattr(
                    model,
                    "max_seq_len",
                    getattr(self, "max_seq_len", 1048576),
                )
            )

            _bad_pos = [
                (i, int(x))
                for i, x in enumerate(positions)
                if int(x) < 0 or int(x) >= _max_seq
            ]

            if os.environ.get(
                "DSV41_DEBUG_MTP_SYNC",
                "0",
            ) == "1":
                print(
                    f"[mtp-verify-input] "
                    f"tokens={list(map(int, row_tokens))} "
                    f"positions={positions} "
                    f"pmax={pmax} "
                    f"embed_rows={_embed_rows} "
                    f"head_rows={_head_rows} "
                    f"bad_tokens={_bad_tokens} "
                    f"bad_pos={_bad_pos}",
                    flush=True,
                )

            if _bad_tokens:
                raise RuntimeError(
                    "MTP verifier produced token outside main "
                    f"embedding vocabulary: {_bad_tokens}; "
                    f"embed_rows={_embed_rows}, "
                    f"head_rows={_head_rows}"
                )

            if _bad_pos:
                raise RuntimeError(
                    "MTP verifier position out of range: "
                    f"{_bad_pos}; max_seq={_max_seq}"
                )

            # Establish that all CUDA work before the verifier itself
            # completed successfully.
            if os.environ.get(
                "DSV41_DEBUG_MTP_SYNC",
                "0",
            ) == "1":
                for _d in getattr(rt, "devs", [last]):
                    torch.cuda.synchronize(_d)

                print(
                    "[mtp-sync] before-verify-step=OK",
                    flush=True,
                )

            verify_logits = rt.step(
                row_tokens,
                positions,
                seq=[0] * K,
                pmax=[pmax] * K,
            )

            if os.environ.get(
                "DSV41_DEBUG_MTP_SYNC",
                "0",
            ) == "1":
                for _d in getattr(rt, "devs", [last]):
                    torch.cuda.synchronize(_d)

                print(
                    "[mtp-sync] after-verify-step=OK",
                    flush=True,
                )

            # Target-layer hidden states corresponding to every
            # verifier row. Needed for the next DSpark invocation.
            mh_all = torch.cat(
                [
                    rt.main_hid[lid].to(last)
                    for lid in ds.targets
                ],
                dim=-1,
            )

            ds.write_main_rows(
                mh_all,
                torch.zeros(
                    K,
                    dtype=torch.int64,
                    device=last,
                ),
                torch.tensor(
                    positions,
                    dtype=torch.int64,
                    device=last,
                ),
            )

            # Sample verifier outputs exactly as the ordinary engine
            # would sample them.
            verified = [
                sample_token(
                    verify_logits[i],
                    p.temperature,
                    p.top_p,
                    gen,
                )
                for i in range(K)
            ]

            acc = 0

            for i, draft in enumerate(drafts):
                if verified[i] == draft:
                    acc += 1
                else:
                    break

            # Accepted drafts followed by the verifier's first
            # non-draft token (or the bonus token after all accepted).
            new_tokens = drafts[:acc] + [verified[acc]]

            n_verify_steps += 1
            n_accepted += acc

            # Row `acc` is the forward pass which produced the new
            # bonus token.
            main_h = mh_all[acc].clone()

            # mtp_run.py uses this same logical state update.
            old_p = p_last
            p_last = old_p + 1 + acc
            written_max = old_p + K

            current = verified[acc]

            stopped = yield from emit(new_tokens)

            if stopped:
                return

        if out[decoded_upto:]:
            tail = self.tok.decode(
                out[decoded_upto:]
            )

            if tail:
                yield out[-1], tail

        if n_verify_steps:
            print(
                f"[mtp] steps={n_verify_steps} "
                f"accepted={n_accepted} "
                f"accepted/step={n_accepted / n_verify_steps:.2f} "
                f"tokens={produced}",
                flush=True,
            )

    @torch.inference_mode()
    def generate_text(self, prompt_ids: list[int], p: GenParams, images=None, token_types=None) -> tuple[str, int]:
        if self.max_seqs <= 1:
            pieces, n = [], 0
            for _, piece in self.generate(prompt_ids, p, images=images, token_types=token_types):
                pieces.append(piece)
                n += 1
            return "".join(pieces), n

        return self._generate_text_batched(prompt_ids, p, images=images, token_types=token_types)

    def _init_batch_scheduler(self):
        self._batch_queue: queue.Queue[_BatchRequest] = queue.Queue()
        self._batch_stop_event = threading.Event()
        # Decode slots are 1 .. max_seqs - 1 (slot 0 is reserved for prefill scratchpad)
        self._free_decode_slots = list(range(1, self.max_seqs))
        self._active_slots: dict[int, _BatchRequest] = {}
        self._batch_thread = threading.Thread(target=self._batch_worker_loop, daemon=True)
        self._batch_thread.start()
        print(
            f"[batched-engine] initialized with {self.max_decode_slots} concurrent decode slots "
            f"(total max_seqs={self.max_seqs}, max_batch={self.rt.B})",
            flush=True,
        )

    @torch.inference_mode()
    def _batch_worker_loop(self):
        B = self.rt.B
        while not self._batch_stop_event.is_set():
            # 1. Prefill pending requests into available decode slots
            while self._free_decode_slots and not self._batch_queue.empty():
                try:
                    req = self._batch_queue.get_nowait()
                except queue.Empty:
                    break

                # Smart slot assignment: if a GPU slot retains the best prefix and is currently free, reuse it!
                best_gpu_slot, best_gpu_lcp = (0, 0) if req.images is not None else self._find_best_gpu_slot(req.prompt_ids)
                if best_gpu_slot in self._free_decode_slots and best_gpu_slot != 0:
                    self._free_decode_slots.remove(best_gpu_slot)
                    slot_id = best_gpu_slot
                else:
                    slot_id = self._free_decode_slots.pop(0)

                req.slot_id = slot_id
                with self._slot_lock:
                    st = self.slot_states.get(slot_id, {})
                    st["slot_id"] = slot_id
                    st["status"] = "prefilling"
                    st["req_id"] = req.req_id
                    st["prompt_tokens"] = len(req.prompt_ids)
                    st["reused_tokens"] = best_gpu_lcp
                    st["reused_slot"] = best_gpu_slot
                    st["lcp"] = best_gpu_lcp
                    st["prefill_type"] = "gpu_hit" if best_gpu_lcp > 0 else "cold"
                    st["suffix_tokens"] = max(0, len(req.prompt_ids) - best_gpu_lcp)
                    st["generated_tokens"] = 0
                    st["max_tokens"] = req.max_new
                    st["start_time"] = time.perf_counter()
                    st["first_tok_time"] = 0.0
                    st["elapsed_s"] = 0.0
                    st["tok_s"] = 0.0
                    st["recent_text"] = ""
                    self.slot_states[slot_id] = st
                try:
                    with self.lock:
                        # Prefill using slot 0 (fully compatible with prefix-cache / snapshots / GPU cache)
                        logits, _reused = self._prefill_with_prefix_reuse(req.prompt_ids, images=req.images, token_types=req.token_types)
                        first_tok = sample_token(logits[0], req.params.temperature, req.params.top_p, req.gen)
                        # Copy per-sequence cache state from slot 0 to target decode slot
                        self.rt.copy_seq(0, slot_id, req_id=req.req_id)
                        with getattr(self, "_slot_tokens_lock", threading.Lock()):
                            self._slot_tokens[0] = list(req.prompt_ids)
                            self._slot_tokens[slot_id] = list(req.prompt_ids)
                        pf_stats = getattr(self, "last_prefill_stats", None)
                        with self._slot_lock:
                            st = self.slot_states.get(slot_id, {})
                            if pf_stats:
                                st["prefill_type"] = pf_stats.get("prefill_type", st.get("prefill_type", "cold"))
                                st["lcp"] = pf_stats.get("lcp", st.get("lcp", 0))
                                st["reused_tokens"] = pf_stats.get("reused_tokens", st.get("reused_tokens", 0))
                                st["suffix_tokens"] = pf_stats.get("suffix_tokens", max(0, len(req.prompt_ids) - st.get("reused_tokens", 0)))
                                st["prefill_time_s"] = pf_stats.get("time_s", 0.0)
                                st["prefill_eff_s"] = pf_stats.get("effective_tok_s", 0.0)
                            self.slot_states[slot_id] = st
                        req.pos = len(req.prompt_ids)
                        req.next_token = first_tok
                        if first_tok == self.eos:
                            req.result_text = ""
                            req.result_count = 0
                            req.done_event.set()
                            self._free_decode_slots.append(slot_id)
                            with self._slot_lock:
                                st = self.slot_states.get(slot_id, {})
                                st["status"] = "completed"
                                st["generated_tokens"] = 0
                                st["recent_text"] = ""
                                st["completed_at"] = time.perf_counter()
                                self.slot_states[slot_id] = st
                            print(f"[batched-engine] prompt immediately reached EOS for slot={slot_id}", flush=True)
                        else:
                            req.out_tokens.append(first_tok)
                            with getattr(self, "_slot_tokens_lock", threading.Lock()):
                                if slot_id in self._slot_tokens:
                                    self._slot_tokens[slot_id].append(first_tok)
                            req.first_token_time = time.perf_counter()
                            first_piece = self.tok.decode([first_tok], errors="replace")
                            req.live_text = first_piece
                            self._active_slots[slot_id] = req
                            with self._slot_lock:
                                st = self.slot_states.get(slot_id, {})
                                st["status"] = "generating"
                                st["first_tok_time"] = req.first_token_time
                                st["generated_tokens"] = 1
                                st["recent_text"] = first_piece
                                self.slot_states[slot_id] = st
                            print(f"[batched-engine] prefilled slot={slot_id} prompt_tokens={len(req.prompt_ids)}", flush=True)
                except Exception as e:
                    print(f"[batched-engine] prefill error on slot={slot_id}: {e}", flush=True)
                    traceback.print_exc()
                    if "out of memory" in str(e).lower() and torch.cuda.is_available():
                        for d in range(torch.cuda.device_count()):
                            try:
                                with torch.cuda.device(d):
                                    torch.cuda.empty_cache()
                            except Exception:
                                pass
                    req.error = e
                    req.done_event.set()
                    self._free_decode_slots.append(slot_id)
                    with getattr(self, "_slot_tokens_lock", threading.Lock()):
                        if slot_id in self._slot_tokens:
                            del self._slot_tokens[slot_id]
                    with self._slot_lock:
                        st = self.slot_states.get(slot_id, {})
                        st["status"] = "idle"
                        self.slot_states[slot_id] = st

            # 2. Decode active slots in batch
            if not self._active_slots:
                time.sleep(0.001)
                continue

            with self.lock:
                active_ids = list(self._active_slots.keys())
                n_active = len(active_ids)

                # Adaptive decode dispatch:
                # If only 1 request is active and dedicated B=1 runtime is available,
                # execute on B=1 graph to achieve ~60+ tok/s (no MoE sorting/bucketing overhead).
                if n_active == 1 and self.rt_b1 is not None:
                    s_id = active_ids[0]
                    req = self._active_slots[s_id]
                    try:
                        logits = self.rt_b1.step([req.next_token], [req.pos], seq=[s_id], pmax=[req.pos])
                    except Exception as exc:
                        print(f"[batched-engine] single-decode step error: {exc}", flush=True)
                        traceback.print_exc()
                        req.error = exc
                        req.done_event.set()
                        self._free_decode_slots.append(s_id)
                        del self._active_slots[s_id]
                        with getattr(self, "_slot_tokens_lock", threading.Lock()):
                            if s_id in self._slot_tokens:
                                del self._slot_tokens[s_id]
                        with self._slot_lock:
                            st = self.slot_states.get(s_id, {})
                            st["status"] = "idle"
                            self.slot_states[s_id] = st
                        continue
                else:
                    toks, poss, seqs, pmaxs = [], [], [], []
                    for s_id in active_ids:
                        req = self._active_slots[s_id]
                        toks.append(req.next_token)
                        poss.append(req.pos)
                        seqs.append(s_id)
                        pmaxs.append(req.pos)

                    # Padding to fixed batch size B with slot 0 (dummy rows)
                    for _ in range(B - n_active):
                        toks.append(0)
                        poss.append(0)
                        seqs.append(0)
                        pmaxs.append(0)

                    try:
                        logits = self.rt.step(toks, poss, seq=seqs, pmax=pmaxs)
                    except Exception as exc:
                        print(f"[batched-engine] decode step error: {exc}", flush=True)
                        traceback.print_exc()
                        for s_id in active_ids:
                            req = self._active_slots[s_id]
                            req.error = exc
                            req.done_event.set()
                            self._free_decode_slots.append(s_id)
                            with getattr(self, "_slot_tokens_lock", threading.Lock()):
                                if s_id in self._slot_tokens:
                                    del self._slot_tokens[s_id]
                            with self._slot_lock:
                                st = self.slot_states.get(s_id, {})
                                st["status"] = "idle"
                                self.slot_states[s_id] = st
                        self._active_slots.clear()
                        continue

                # Sample tokens per slot with optional repetition/presence/frequency penalties
                sampled_tokens = []
                for idx, s_id in enumerate(active_ids):
                    req = self._active_slots[s_id]
                    slot_logits = logits[idx]
                    if req.out_tokens and (
                        req.params.repetition_penalty != 1.0
                        or req.params.presence_penalty != 0.0
                        or req.params.frequency_penalty != 0.0
                        or req.params.progressive_penalty > 0.0
                        or req.params.ban_cycles
                    ):
                        slot_logits = apply_penalties(
                            slot_logits,
                            req.out_tokens,
                            repetition_penalty=req.params.repetition_penalty,
                            presence_penalty=req.params.presence_penalty,
                            frequency_penalty=req.params.frequency_penalty,
                            window=req.params.penalty_window,
                            progressive_penalty=req.params.progressive_penalty,
                            ban_cycles=req.params.ban_cycles,
                        )
                    try:
                        if req.params.temperature <= 0:
                            sampled_tokens.append(int(slot_logits.argmax(dim=-1).item()))
                        else:
                            sampled_tokens.append(sample_token(slot_logits, req.params.temperature, req.params.top_p, req.gen))
                    except Exception as exc:
                        print(f"[batched-engine] sampling fallback on slot={s_id}: {exc}", flush=True)
                        try:
                            sampled_tokens.append(int(slot_logits.argmax(dim=-1).item()))
                        except Exception:
                            sampled_tokens.append(self.eos)

                finished = []
                now = time.perf_counter()
                if self.stats_tracker is not None and n_active > 0:
                    try:
                        self.stats_tracker.record_decode_tokens(n_active)
                    except Exception:
                        pass
                for idx, s_id in enumerate(active_ids):
                    req = self._active_slots[s_id]
                    t = sampled_tokens[idx]
                    req.pos += 1

                    is_eos = (t == self.eos)
                    if not is_eos:
                        req.out_tokens.append(t)
                        with getattr(self, "_slot_tokens_lock", threading.Lock()):
                            if s_id in self._slot_tokens:
                                self._slot_tokens[s_id].append(t)
                    is_max = (len(req.out_tokens) >= req.max_new)

                    # Fast stop condition check: inspect small trailing window to avoid O(N^2) decodes
                    is_stopped = False
                    if req.params.stop and not is_eos:
                        tail_text = self.tok.decode(req.out_tokens[-32:], errors="replace")
                        for s in req.params.stop:
                            if s in tail_text:
                                full_text = self.tok.decode(req.out_tokens, errors="replace")
                                cut = full_text.find(s)
                                if cut != -1:
                                    req.result_text = full_text[:cut]
                                    req.result_count = len(req.out_tokens)
                                    is_stopped = True
                                    break

                    # Degenerate repetition loop detection & auto-truncation
                    if not is_stopped and not is_eos and getattr(req.params, "loop_detect", True):
                        min_match = getattr(req.params, "min_loop_match", 48)
                        min_cycle = getattr(req.params, "min_loop_cycle", 1)
                        if len(req.out_tokens) >= min_match + min_cycle:
                            loop_res = detect_loop(req.out_tokens, min_match=min_match, min_cycle=min_cycle)
                            if loop_res is not None:
                                cycle_len, trim_count, prev_pos = loop_res
                                print(
                                    f"[batched-engine] REPETITION LOOP DETECTED on slot={s_id}: "
                                    f"cycle={cycle_len} tokens, trimming {trim_count} duplicate tokens (prev_pos={prev_pos}). Stopping.",
                                    flush=True,
                                )
                                if trim_count > 0:
                                    req.out_tokens = req.out_tokens[:-trim_count]
                                    with getattr(self, "_slot_tokens_lock", threading.Lock()):
                                        if s_id in self._slot_tokens:
                                            self._slot_tokens[s_id] = self._slot_tokens[s_id][:-trim_count]
                                req.result_text = self.tok.decode(req.out_tokens, errors="replace")
                                req.result_count = len(req.out_tokens)
                                is_stopped = True

                    if is_eos or is_max or is_stopped:
                        if not is_stopped:
                            req.result_text = self.tok.decode(req.out_tokens, errors="replace")
                            req.result_count = len(req.out_tokens)
                        req.finish_reason = "length" if is_max else "stop"
                        req.done_event.set()
                        finished.append(s_id)
                        dt_gen = max(now - (req.first_token_time or req.start_time), 1e-4)
                        tok_s = round(req.result_count / dt_gen, 1)
                        req.decode_tok_s = tok_s
                        self.last_decode_tok_s = tok_s
                        with self._slot_lock:
                            st = self.slot_states.get(s_id, {})
                            st["status"] = "completed"
                            st["generated_tokens"] = req.result_count
                            st["tok_s"] = tok_s
                            st["elapsed_s"] = round(now - req.start_time, 2)
                            st["recent_text"] = req.result_text[-1500:] if len(req.result_text) > 1500 else req.result_text
                            st["completed_at"] = now
                            self.slot_states[s_id] = st
                        print(
                            f"[batched-engine] finished slot={s_id} tokens={req.result_count} "
                            f"in {dt_gen:.2f}s ({tok_s} tok/s) "
                            f"reason={'eos' if is_eos else ('max' if is_max else 'stop')}",
                            flush=True,
                        )
                    else:
                        req.next_token = t
                        piece = self.tok.decode([t], errors="replace")
                        req.live_text += piece
                        if len(req.live_text) > 2000:
                            req.live_text = req.live_text[-1500:]
                        n_tok = len(req.out_tokens)
                        dt_gen = max(now - (req.first_token_time or req.start_time), 1e-4)
                        tok_s = round(n_tok / dt_gen, 1)
                        with self._slot_lock:
                            st = self.slot_states.get(s_id, {})
                            st["status"] = "generating"
                            st["generated_tokens"] = n_tok
                            st["tok_s"] = tok_s
                            st["elapsed_s"] = round(now - req.start_time, 2)
                            st["recent_text"] = req.live_text
                            self.slot_states[s_id] = st

                # Periodic console decode log across active slots in batch (approx every 1s)
                if now - getattr(self, "_last_decode_log_time", 0.0) >= 1.0:
                    self._last_decode_log_time = now
                    parts = []
                    for sid in active_ids:
                        if sid not in finished:
                            r = self._active_slots[sid]
                            n = len(r.out_tokens)
                            dt = max(now - (r.first_token_time or r.start_time), 1e-4)
                            ts = n / dt
                            tail = (r.live_text[-35:] if len(r.live_text) > 35 else r.live_text).replace("\n", " ").replace("\r", "")
                            parts.append(f"slot {sid}: {n}/{r.max_new} tok ({ts:.1f} t/s) \"{tail}\"")
                    if parts:
                        print(f"[decode] {' | '.join(parts)}", flush=True)

                for s_id in finished:
                    del self._active_slots[s_id]
                    self._free_decode_slots.append(s_id)

    @torch.inference_mode()
    def _generate_text_batched(self, prompt_ids: list[int], p: GenParams, images=None, token_types=None) -> tuple[str, int]:
        _mtp_long_limit = int(os.environ.get("DSV41_MTP_LONG_PROMPT_LIMIT", "65536"))
        max_new = min(p.max_new_tokens, self.max_seq_len - len(prompt_ids) - 1)
        if len(prompt_ids) > _mtp_long_limit:
            _long_cap = int(os.environ.get("DSV41_LONG_PROMPT_MAX_NEW", os.environ.get("DSV41_INTERACTIVE_MAX_NEW", "65536")))
            max_new = min(max_new, _long_cap)
        gen = None
        if p.seed is not None:
            gen = torch.Generator(device=self.model.blocks[-1].device)
            gen.manual_seed(p.seed)

        req = _BatchRequest(prompt_ids, p, max_new, gen, images=images, token_types=token_types)
        self._batch_queue.put(req)
        req.done_event.wait()
        if req.error:
            raise req.error
        self.last_decode_tok_s = getattr(req, "decode_tok_s", None)
        self.last_finish_reason = getattr(req, "finish_reason", "stop")
        return req.result_text, req.result_count


def parse_budgets(spec: str) -> dict[int, float] | None:
    return {int(k): float(v) for k, v in (kv.split(":") for kv in spec.split(",") if kv)} or None
