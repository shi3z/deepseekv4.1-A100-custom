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
from dataclasses import dataclass, field
from typing import Iterator

import torch

from .decode import DecodeRuntime
from .load import load_model
import traceback

CKPT = "/mnt/ssd/models/DeepSeek-V4.1-Flash"


@dataclass
class GenParams:
    max_new_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.95
    stop: list[str] = field(default_factory=list)
    seed: int | None = None


class _BatchRequest:
    def __init__(self, prompt_ids: list[int], params: GenParams, max_new: int, gen: torch.Generator | None):
        self.prompt_ids = prompt_ids
        self.params = params
        self.max_new = max_new
        self.gen = gen
        self.pos = 0
        self.next_token = 0
        self.out_tokens: list[int] = []
        self.done_event = threading.Event()
        self.result_text = ""
        self.result_count = 0
        self.error: Exception | None = None


def sample_token(logits: torch.Tensor, temperature: float, top_p: float, gen: torch.Generator | None) -> int:
    if temperature <= 0:
        return int(logits.argmax(dim=-1).item())
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    if 0 < top_p < 1:
        sp, si = probs.sort(descending=True)
        keep = (sp.cumsum(-1) - sp) < top_p  # keep tokens until cumulative mass passes top_p
        sp = sp * keep
        idx = torch.multinomial(sp / sp.sum(), 1, generator=gen)
        return int(si.gather(-1, idx).item())
    return int(torch.multinomial(probs, 1, generator=gen).item())


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
        self.model_name = "deepseek-v4.1-flash"
        self._jev_engine = None
        if self.max_seqs > 1:
            self._init_batch_scheduler()

    @property
    def jev_engine(self):
        if self._jev_engine is None:
            from .jev import JevEngine
            self._jev_engine = JevEngine(self.model, self.tok)
        return self._jev_engine

    def jev_inference(self, prompt: str, schema: dict, max_batch: int = 32) -> tuple[dict, dict]:
        with self.lock:
            return self.jev_engine.process_request(prompt, schema, max_batch=max_batch)

    def get_cache_stats(self) -> dict:
        try:
            entries, total_bytes = self._prefix_cache_stats()
        except Exception:
            entries, total_bytes = 0, 0
        jev_schemas = 0
        jev_requests = 0
        if self._jev_engine is not None and getattr(self._jev_engine, "prefix_tree", None) is not None:
            pt = self._jev_engine.prefix_tree
            jev_schemas = len(pt.schema_nodes)
            jev_requests = len(getattr(pt, "request_nodes", {}))
        return {
            "prefix_entries": entries,
            "prefix_bytes": total_bytes,
            "jev_schemas": jev_schemas,
            "jev_requests": jev_requests,
        }

    # ---------------------------------------------------------------- prompts
    def chat_prompt(self, messages: list[dict], thinking_mode: str | None = None) -> str:
        return self._encode(messages, thinking_mode=thinking_mode or self.thinking_mode)

    def parse_completion(self, text: str, thinking_mode: str | None = None) -> dict:
        """Structured assistant message (content / reasoning_content / tool_calls). The official parser
        wants the completion to end with the EOS string; we generate without it, so append it."""
        eos = self.tok.eos_token or ""
        try:
            return self._parse(text + eos, thinking_mode=thinking_mode or self.thinking_mode)
        except Exception:
            return {"role": "assistant", "content": text, "reasoning_content": None, "tool_calls": []}

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

            # Dynamic compressed/index caches may be allocated well beyond
            # the logical prefix. Persist only rows that can be read when
            # restoring this anchor; unused capacity is recreated locally.
            save_src = src
            if (
                used_tokens is not None
                and kind == "dict"
                and isinstance(key, tuple)
                and key
                and key[0] in self.model.shared.cache_max_rows
                and src.ndim >= 2
            ):
                owner = int(key[0])
                ratio = int(self.model.args.compress_ratios[owner])
                rows = max(1, min(src.shape[1], int(used_tokens) // max(1, ratio) + 1))
                save_src = src[:, :rows].contiguous()

            nbytes = save_src.numel() * save_src.element_size()
            total += nbytes
            used_rows = (
                int(save_src.shape[1])
                if kind == "dict" and save_src.ndim >= 2
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
                "0",
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
        if kind == "dict" and isinstance(key, tuple) and key:
            try:
                if int(key[0]) >= 0:
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
                    "512",
                )
            ),
        )

        max_gb = float(
            os.environ.get(
                "DSV41_PREFIX_CACHE_GB",
                "256",
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

                _shape_ok = (
                    torch.is_tensor(src)
                    and torch.is_tensor(dst)
                    and src.dtype == dst.dtype
                    and (
                        tuple(src.shape) == tuple(dst.shape)
                        or (
                            kind == "dict"
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

        return ent

    def _prefix_cache_stats(self):
        entries = self._prefix_cache_entries()

        total = sum(
            e["bytes"]
            for e in entries
        )

        return len(entries), total

    @torch.inference_mode()
    def _prefill_with_prefix_reuse(
        self,
        prompt_ids: list[int],
    ):
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

        return logits, 0

    @torch.inference_mode()
    def generate(self, prompt_ids: list[int], p: GenParams) -> Iterator[tuple[int, str]]:
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
            yield from self.generate(prompt_ids, p)
            return
        max_new = min(p.max_new_tokens, self.max_seq_len - len(prompt_ids) - 1)
        # Claude may request 32K output even for a 100K+ context. Keep
        # long-context requests bounded so the gateway can finish instead
        # of timing out during slow single-token decode. Override per host.
        if len(prompt_ids) > _mtp_long_limit:
            max_new = min(max_new, int(os.environ.get("DSV41_LONG_PROMPT_MAX_NEW", "2048")))
        gen = None
        if p.seed is not None:
            gen = torch.Generator(device=self.model.blocks[-1].device)
            gen.manual_seed(p.seed)
        with self.lock:
            if self.mtp:
                yield from self._generate_mtp_locked(
                    prompt_ids,
                    p,
                    max_new,
                    gen,
                )
                return

            logits, _prefix_reused = self._prefill_with_prefix_reuse(
                prompt_ids
            )
            pos = len(prompt_ids)
            out: list[int] = []
            decoded_upto = 0
            pending = ""
            for step in range(max_new):
                t = sample_token(logits[0], p.temperature, p.top_p, gen)
                if t == self.eos:
                    print(
                     f"[generate] STOP=eos step={step} pos={pos} "
                     f"max_new={max_new}",
                     flush=True,
                    )

                    break
                out.append(t)
                # decode incrementally; hold back a partial multi-byte character
                text = self.tok.decode(out[decoded_upto:])
                if "�" in text:
                    piece = ""
                else:
                    piece, decoded_upto = text, len(out)
                if piece:
                    pending += piece
                    if p.stop and any(s in pending for s in p.stop):
                        cut = min(pending.find(s) for s in p.stop if s in pending)
                        if cut > 0:
                            yield t, pending[:cut]
                        return
                    yield t, pending
                    pending = ""
                logits = self.rt.step(t, pos)
                pos += 1
            if out[decoded_upto:]:
                tail = self.tok.decode(out[decoded_upto:])
                if tail:
                    yield out[-1], tail

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
    def generate_text(self, prompt_ids: list[int], p: GenParams) -> tuple[str, int]:
        if self.max_seqs <= 1:
            pieces, n = [], 0
            for _, piece in self.generate(prompt_ids, p):
                pieces.append(piece)
                n += 1
            return "".join(pieces), n

        return self._generate_text_batched(prompt_ids, p)

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

    def _batch_worker_loop(self):
        B = self.rt.B
        while not self._batch_stop_event.is_set():
            # 1. Prefill pending requests into available decode slots
            while self._free_decode_slots and not self._batch_queue.empty():
                try:
                    req = self._batch_queue.get_nowait()
                except queue.Empty:
                    break
                slot_id = self._free_decode_slots.pop(0)
                try:
                    with self.lock:
                        # Prefill using slot 0 (fully compatible with prefix-cache / snapshots)
                        logits, _reused = self._prefill_with_prefix_reuse(req.prompt_ids)
                        first_tok = sample_token(logits[0], req.params.temperature, req.params.top_p, req.gen)
                        # Copy per-sequence cache state from slot 0 to target decode slot
                        self.rt.copy_seq(0, slot_id)
                        req.pos = len(req.prompt_ids)
                        req.next_token = first_tok
                        if first_tok == self.eos:
                            req.result_text = ""
                            req.result_count = 0
                            req.done_event.set()
                            self._free_decode_slots.append(slot_id)
                            print(f"[batched-engine] prompt immediately reached EOS for slot={slot_id}", flush=True)
                        else:
                            req.out_tokens.append(first_tok)
                            self._active_slots[slot_id] = req
                            print(f"[batched-engine] prefilled slot={slot_id} prompt_tokens={len(req.prompt_ids)}", flush=True)
                except Exception as e:
                    print(f"[batched-engine] prefill error on slot={slot_id}: {e}", flush=True)
                    traceback.print_exc()
                    req.error = e
                    req.done_event.set()
                    self._free_decode_slots.append(slot_id)

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
                        self._active_slots.clear()
                        continue

                # Batch sampling optimization:
                # When all slots are greedy (temperature <= 0), execute a single batched argmax
                # and transfer all tokens to CPU in one round trip.
                all_greedy = all(self._active_slots[s].params.temperature <= 0 for s in active_ids)
                if all_greedy:
                    sampled_tokens = logits[:n_active].argmax(dim=-1).tolist()
                else:
                    sampled_tokens = []
                    greedy_cached = None
                    for idx, s_id in enumerate(active_ids):
                        req = self._active_slots[s_id]
                        if req.params.temperature <= 0:
                            if greedy_cached is None:
                                greedy_cached = logits[:n_active].argmax(dim=-1).tolist()
                            sampled_tokens.append(greedy_cached[idx])
                        else:
                            slot_logits = logits[idx]
                            sampled_tokens.append(sample_token(slot_logits, req.params.temperature, req.params.top_p, req.gen))

                finished = []
                for idx, s_id in enumerate(active_ids):
                    req = self._active_slots[s_id]
                    t = sampled_tokens[idx]
                    req.pos += 1

                    is_eos = (t == self.eos)
                    if not is_eos:
                        req.out_tokens.append(t)
                    is_max = (len(req.out_tokens) >= req.max_new)

                    # Fast stop condition check: inspect small trailing window to avoid O(N^2) decodes
                    is_stopped = False
                    if req.params.stop and not is_eos:
                        tail_text = self.tok.decode(req.out_tokens[-32:])
                        for s in req.params.stop:
                            if s in tail_text:
                                full_text = self.tok.decode(req.out_tokens)
                                cut = full_text.find(s)
                                if cut != -1:
                                    req.result_text = full_text[:cut]
                                    req.result_count = len(req.out_tokens)
                                    is_stopped = True
                                    break

                    if is_eos or is_max or is_stopped:
                        if not is_stopped:
                            req.result_text = self.tok.decode(req.out_tokens)
                            req.result_count = len(req.out_tokens)
                        req.done_event.set()
                        finished.append(s_id)
                        print(
                            f"[batched-engine] finished slot={s_id} tokens={req.result_count} "
                            f"reason={'eos' if is_eos else ('max' if is_max else 'stop')}",
                            flush=True,
                        )
                    else:
                        req.next_token = t

                for s_id in finished:
                    del self._active_slots[s_id]
                    self._free_decode_slots.append(s_id)

    @torch.inference_mode()
    def _generate_text_batched(self, prompt_ids: list[int], p: GenParams) -> tuple[str, int]:
        _mtp_long_limit = int(os.environ.get("DSV41_MTP_LONG_PROMPT_LIMIT", "65536"))
        max_new = min(p.max_new_tokens, self.max_seq_len - len(prompt_ids) - 1)
        if len(prompt_ids) > _mtp_long_limit:
            max_new = min(max_new, int(os.environ.get("DSV41_LONG_PROMPT_MAX_NEW", "2048")))
        gen = None
        if p.seed is not None:
            gen = torch.Generator(device=self.model.blocks[-1].device)
            gen.manual_seed(p.seed)

        req = _BatchRequest(prompt_ids, p, max_new, gen)
        self._batch_queue.put(req)
        req.done_event.wait()
        if req.error:
            raise req.error
        return req.result_text, req.result_count


def parse_budgets(spec: str) -> dict[int, float] | None:
    return {int(k): float(v) for k, v in (kv.split(":") for kv in spec.split(",") if kv)} or None
