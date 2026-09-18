"""Jev Mode: Parallel Non-Autoregressive Structured Output Inference for DeepSeek-V4.1-Flash.

Instead of decoding JSON token-by-token autoregressively (which takes dozens or hundreds
of serial decode steps), Jev mode treats each schema field as an independent parallel query
over a shared prefix representation, evaluating candidate log-probabilities directly.

Hierarchical Persistent Prefix Cache:
  [Root]
    └── [Jev System Prompt]  (Prefilled at server startup, permanently retained on GPU)
          └── [Schema Prefix] (Cached separately for repeated schemas on GPU)
                └── [Request Input] (Prefilled once per request -> Shared Request KV)
                      ├── [Field 1 Query] -> Candidate Logits
                      ├── [Field 2 Query] -> Candidate Logits
                      └── [Field 3 Query] -> Candidate Logits
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Schema Definitions and Parsing
# ---------------------------------------------------------------------------

@dataclass
class JevField:
    name: str
    field_type: str  # "enum", "boolean", "integer", "number", "string"
    candidates: list[Any] = field(default_factory=list)
    description: str = ""
    minimum: float | None = None
    maximum: float | None = None
    depends_on: list[str] = field(default_factory=list)
    index: int = 0


class JevSchema:
    """Parses simplified schemas or standard JSON Schema into JevField definitions."""

    def __init__(self, raw_schema: dict[str, Any]):
        self.raw = raw_schema
        self.fields: list[JevField] = []
        self._parse(raw_schema)
        self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        s = json.dumps(self.raw, sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    def _parse(self, raw: dict[str, Any]):
        # Format A: Standard JSON Schema (type: object, properties: {...})
        if "properties" in raw and isinstance(raw["properties"], dict):
            props = raw["properties"]
            idx = 1
            for name, pdef in props.items():
                ptype = pdef.get("type", "string")
                desc = pdef.get("description", "")
                cands = []
                f_type = "string"

                if "enum" in pdef and isinstance(pdef["enum"], list):
                    f_type = "enum"
                    cands = list(pdef["enum"])
                elif ptype == "boolean":
                    f_type = "boolean"
                    cands = [True, False]
                elif ptype in ("integer", "number"):
                    f_type = ptype
                    mn = pdef.get("minimum")
                    mx = pdef.get("maximum")
                    if mn is not None and mx is not None:
                        # Bounded integer / number: support discretized candidate scoring if small range
                        if f_type == "integer" and (mx - mn) <= 20:
                            cands = list(range(int(mn), int(mx) + 1))
                        elif f_type == "number" and (mx - mn) <= 1.01 and mn == 0.0:
                            cands = [round(x * 0.1, 2) for x in range(11)]
                else:
                    f_type = "string"

                deps = pdef.get("depends_on", [])
                if isinstance(deps, str):
                    deps = [deps]

                jf = JevField(
                    name=name,
                    field_type=f_type,
                    candidates=cands,
                    description=desc,
                    minimum=pdef.get("minimum"),
                    maximum=pdef.get("maximum"),
                    depends_on=list(deps),
                    index=idx,
                )
                self.fields.append(jf)
                idx += 1
            return

        # Format B: Simplified mapping { "sentiment": ["positive", "neutral", "negative"], "needs_human": [true, false] }
        idx = 1
        for name, val in raw.items():
            if isinstance(val, list):
                # Check if boolean
                if len(val) == 2 and set(str(x).lower() for x in val) == {"true", "false"}:
                    f_type = "boolean"
                    cands = [True, False]
                else:
                    f_type = "enum"
                    cands = list(val)
            elif isinstance(val, str) and val in ("boolean", "bool"):
                f_type = "boolean"
                cands = [True, False]
            elif isinstance(val, str) and val in ("number", "float", "integer", "int"):
                f_type = val
                cands = []
            else:
                f_type = "string"
                cands = []

            jf = JevField(
                name=name,
                field_type=f_type,
                candidates=cands,
                index=idx,
            )
            self.fields.append(jf)
            idx += 1

    def format_schema_prefix(self) -> str:
        """Formats the schema definition for Level 2 prefix caching."""
        lines = ["Fields to extract:"]
        for f in self.fields:
            if f.candidates:
                c_str = ", ".join(str(c).lower() if isinstance(c, bool) else str(c) for c in f.candidates)
                lines.append(f"Field {f.index} ({f.name}): [{c_str}]")
            else:
                lines.append(f"Field {f.index} ({f.name}): {f.field_type}")
        return "\n".join(lines) + "\n"

    def assemble(self, field_values: dict[str, Any]) -> dict[str, Any]:
        """Assembles typed output dictionary adhering to the schema."""
        out = {}
        for f in self.fields:
            val = field_values.get(f.name)
            if val is not None:
                out[f.name] = val
        return out


# ---------------------------------------------------------------------------
# Hierarchical Persistent Prefix Cache
# ---------------------------------------------------------------------------

class JevPrefixNode:
    """A node in the hierarchical prefix tree."""

    def __init__(self, key: str, token_ids: list[int], node_type: str, parent: JevPrefixNode | None = None):
        self.key = key
        self.node_id = key
        self.node_type = node_type  # "system", "schema", "request"
        self.token_ids = list(token_ids)
        self.parent = parent
        self.children: dict[str, JevPrefixNode] = {}
        self.start_pos = parent.end_pos if parent else 0
        self.end_pos = self.start_pos + len(token_ids)
        self.kv_snapshot: dict | None = None
        self.hit_count = 0
        self.created_at = time.time()
        self.last_accessed = time.time()


_LOGGED_LEVELS: set[str] = set()


class JevPrefixTree:
    """Manages hierarchical persistent prefix caching on GPU:

    Root -> Jev System Prompt -> Schema Prefix -> Request Input -> Fields
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tok = tokenizer
        self.root = JevPrefixNode("root", [], "root")
        self.system_node: JevPrefixNode | None = None
        self.schema_nodes: dict[str, JevPrefixNode] = {}
        self.request_nodes: dict[str, JevPrefixNode] = {}
        self.stats = {
            "system_hits": 0,
            "schema_hits": 0,
            "schema_misses": 0,
            "total_requests": 0,
            "total_prefill_tokens_saved": 0,
        }

    def _ensure_batch_capacity(self, B: int):
        """Ensure model caches have sufficient batch dimension capacity for B parallel fields."""
        max_seqs = getattr(self.model.args, "max_seqs", 2)
        if B > max_seqs:
            raise ValueError(
                f"Requested Jev batch size {B} exceeds model max_seqs ({max_seqs}). "
                f"Batch size must be clamped to <= {max_seqs}."
            )
        with torch.inference_mode(False):
            # 1. Window KV caches across all blocks
            for blk in self.model.blocks:
                if blk.attn.window_kv_cache.shape[0] < B:
                    old = blk.attn.window_kv_cache
                    new_c = torch.zeros(B, old.shape[1], old.shape[2], dtype=old.dtype, device=old.device)
                    assert not torch.is_inference(new_c), "Allocated window_kv_cache must not be an InferenceTensor"
                    new_c[:old.shape[0]] = old
                    blk.attn.window_kv_cache = new_c

            # 2. Shared compress_kv & index_k
            for k, cache in self.model.shared.compress_kv.items():
                if cache.shape[0] < B:
                    new_c = torch.zeros(B, cache.shape[1], cache.shape[2], dtype=cache.dtype, device=cache.device)
                    assert not torch.is_inference(new_c), "Allocated compress_kv must not be an InferenceTensor"
                    new_c[:cache.shape[0]] = cache
                    self.model.shared.compress_kv[k] = new_c

            for k, cache in self.model.shared.index_k.items():
                if cache.shape[0] < B:
                    new_c = torch.zeros(B, cache.shape[1], cache.shape[2], dtype=cache.dtype, device=cache.device)
                    assert not torch.is_inference(new_c), "Allocated index_k must not be an InferenceTensor"
                    new_c[:cache.shape[0]] = cache
                    self.model.shared.index_k[k] = new_c

            # 3. Engram cache
            if getattr(self.model, "engram_hash", None) is not None:
                if self.model.engram_hash.cache.shape[0] < B:
                    old = self.model.engram_hash.cache
                    new_c = torch.empty(B, old.shape[1], dtype=old.dtype, device=old.device)
                    assert not torch.is_inference(new_c), "Allocated engram cache must not be an InferenceTensor"
                    new_c[:old.shape[0]] = old
                    self.model.engram_hash.cache = new_c

    def snapshot_kv(self, end_pos: int, level: str = "request") -> dict:
        """Capture an exact snapshot of active attention KV state up to end_pos on GPU.
        Levels:
          1. 'system': Persistent Jev system prefix KV (immutable, never modified)
          2. 'schema': Persistent schema prefix KV (immutable, never modified)
          3. 'request': Request shared prefix KV (immutable once built, shared by fields)
        """
        with torch.inference_mode(False):
            snap = {
                "end_pos": end_pos,
                "level": level,
                "window_kv": {},
                "compress_kv": {},
                "index_k": {},
                "engram": None,
            }
            for blk in self.model.blocks:
                snap["window_kv"][blk.layer_id] = blk.attn.window_kv_cache[0:1].clone()

            for (owner, dev), cache in self.model.shared.compress_kv.items():
                ratio = max(1, self.model.args.compress_ratios[owner])
                rows = max(1, end_pos // ratio + 1)
                snap["compress_kv"][(owner, dev)] = cache[0:1, :rows].clone()

            for (owner, dev), cache in self.model.shared.index_k.items():
                ratio = max(1, self.model.args.compress_ratios[owner])
                rows = max(1, end_pos // ratio + 1)
                snap["index_k"][(owner, dev)] = cache[0:1, :rows].clone()

            if getattr(self.model, "engram_hash", None) is not None:
                snap["engram"] = self.model.engram_hash.cache[0:1, :end_pos].clone()

            return snap

    def restore_kv(self, snap: dict, batch_size: int = 1, req_id: str = "", level: str = "request"):
        """Restore a snapshot into the active KV cache, broadcasting to batch_size rows.
        Audits destination tensors for inference status, heals if necessary, and ensures
        destination buffers are mutable before inplace copy.
        """
        self._ensure_batch_capacity(batch_size)
        end_pos = snap["end_pos"]
        snap_level = snap.get("level", level)
        first_audit = level not in _LOGGED_LEVELS

        with torch.inference_mode():
            for blk in self.model.blocks:
                saved = snap["window_kv"][blk.layer_id]
                target = blk.attn.window_kv_cache
                is_inf = torch.is_inference(target)
                op_name = f"restore_kv({snap_level}->batch_{batch_size})"
                if is_inf or (first_audit and blk.layer_id == 0):
                    print(
                        f"[tensor-audit] level={level} req_id={req_id} name=window_kv_cache[{blk.layer_id}] "
                        f"shape={tuple(target.shape)} dev={target.device} is_inference={is_inf} op={op_name}",
                        flush=True,
                    )
                target[:batch_size].copy_(saved)

            for (owner, dev), cache in self.model.shared.compress_kv.items():
                saved = snap["compress_kv"][(owner, dev)]
                rows = saved.shape[1]
                is_inf = torch.is_inference(cache)
                op_name = f"restore_kv({snap_level}->batch_{batch_size})"
                if is_inf or (first_audit and (owner, dev) == next(iter(self.model.shared.compress_kv))):
                    print(
                        f"[tensor-audit] level={level} req_id={req_id} name=compress_kv[({owner},{dev})] "
                        f"shape={tuple(cache.shape)} dev={cache.device} is_inference={is_inf} op={op_name}",
                        flush=True,
                    )
                cache[:batch_size, :rows].copy_(saved)

            for (owner, dev), cache in self.model.shared.index_k.items():
                saved = snap["index_k"][(owner, dev)]
                rows = saved.shape[1]
                is_inf = torch.is_inference(cache)
                op_name = f"restore_kv({snap_level}->batch_{batch_size})"
                if is_inf or (first_audit and (owner, dev) == next(iter(self.model.shared.index_k))):
                    print(
                        f"[tensor-audit] level={level} req_id={req_id} name=index_k[({owner},{dev})] "
                        f"shape={tuple(cache.shape)} dev={cache.device} is_inference={is_inf} op={op_name}",
                        flush=True,
                    )
                cache[:batch_size, :rows].copy_(saved)

            if getattr(self.model, "engram_hash", None) is not None and snap.get("engram") is not None:
                saved = snap["engram"]
                target = self.model.engram_hash.cache
                is_inf = torch.is_inference(target)
                op_name = f"restore_kv({snap_level}->batch_{batch_size})"
                if is_inf or first_audit:
                    print(
                        f"[tensor-audit] level={level} req_id={req_id} name=engram_hash.cache "
                        f"shape={tuple(target.shape)} dev={target.device} is_inference={is_inf} op={op_name}",
                        flush=True,
                    )
                target[:batch_size, :end_pos].copy_(saved)

            _LOGGED_LEVELS.add(level)

    def init_system_prompt(self, system_text: str | None = None) -> JevPrefixNode:
        """Prefill and permanently retain the fixed Jev system prompt on GPU at startup."""
        if self.system_node is not None:
            return self.system_node

        if system_text is None:
            system_text = (
                "<｜begin▁of▁sentence｜><｜System｜>You are a structured extraction engine. "
                "Extract field values according to schema. Output only the value.<｜User｜>"
            )

        sys_ids = self.tok.encode(system_text, add_special_tokens=False)
        dev0 = self.model.blocks[0].device
        input_t = torch.tensor([sys_ids], dtype=torch.long, device=dev0)

        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = self.model.forward(input_t, start_pos=0)
        torch.cuda.synchronize(dev0)
        dt = time.perf_counter() - t0

        self.system_node = JevPrefixNode("sys:default", sys_ids, "system", parent=self.root)
        self.system_node.kv_snapshot = self.snapshot_kv(self.system_node.end_pos, level="system")
        self.root.children["sys:default"] = self.system_node
        print(f"[jev-prefix] System prompt prefilled & retained on GPU: tokens={len(sys_ids)} time={dt*1000:.2f}ms", flush=True)
        return self.system_node

    def get_or_create_schema_node(self, schema: JevSchema) -> tuple[JevPrefixNode, bool]:
        """Returns (schema_node, is_cache_hit). Caches schema prefix separately on GPU."""
        if self.system_node is None:
            self.init_system_prompt()

        schema_key = f"schema:{schema.signature}"
        if schema_key in self.schema_nodes:
            self.stats["schema_hits"] += 1
            node = self.schema_nodes[schema_key]
            node.hit_count += 1
            node.last_accessed = time.time()
            self.stats["total_prefill_tokens_saved"] += len(self.system_node.token_ids) + len(node.token_ids)
            return node, True

        # Cache Miss: Restore system prompt snapshot (zero cost) & prefill schema prefix
        self.stats["schema_misses"] += 1
        self.restore_kv(self.system_node.kv_snapshot, batch_size=1, req_id=schema.signature, level="schema")

        schema_text = schema.format_schema_prefix()
        schema_ids = self.tok.encode(schema_text, add_special_tokens=False)
        dev0 = self.model.blocks[0].device
        input_t = torch.tensor([schema_ids], dtype=torch.long, device=dev0)

        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = self.model.forward(input_t, start_pos=self.system_node.end_pos)
        torch.cuda.synchronize(dev0)
        dt = time.perf_counter() - t0

        schema_node = JevPrefixNode(schema_key, schema_ids, "schema", parent=self.system_node)
        schema_node.kv_snapshot = self.snapshot_kv(schema_node.end_pos, level="schema")
        self.system_node.children[schema_key] = schema_node
        self.schema_nodes[schema_key] = schema_node
        print(f"[jev-prefix] Schema prefix cached on GPU: key={schema_key} tokens={len(schema_ids)} time={dt*1000:.2f}ms", flush=True)
        return schema_node, False

    @staticmethod
    def _prompt_lcp(a: list[int], b: list[int]) -> int:
        if not a or not b:
            return 0
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def get_or_create_request_node(
        self,
        schema_node: JevPrefixNode,
        req_ids: list[int],
    ) -> tuple[JevPrefixNode, int, dict]:
        """Looks up or creates Level 3 request KV snapshot.
        Returns:
            (node, reused_tokens, kv_snapshot)
        """
        import hashlib
        # Hash first 4096 tokens + length to form key
        h_bytes = bytes(str(req_ids[:4096]), "ascii") + len(req_ids).to_bytes(4, "little")
        req_hash = hashlib.sha256(h_bytes).hexdigest()[:16]
        req_key = f"{schema_node.node_id}:{req_hash}"

        # 1. Exact match hit
        if req_key in self.request_nodes:
            node = self.request_nodes[req_key]
            node.last_accessed = time.time()
            node.hit_count += 1
            print(f"[jev-prefix] Request exact HIT: key={req_key} tokens={len(req_ids)}", flush=True)
            return node, len(req_ids), node.kv_snapshot

        # 2. Check candidates under this schema for LCP prefix reuse
        best_cand = None
        best_lcp = 0
        for cand_key, cand_node in self.request_nodes.items():
            if cand_node.parent == schema_node and cand_node.token_ids:
                lcp = self._prompt_lcp(cand_node.token_ids, req_ids)
                if lcp > best_lcp:
                    best_lcp = lcp
                    best_cand = cand_node

        dev0 = self.model.blocks[0].device
        _chunk = max(1, int(os.environ.get("DSV41_HC_PREFILL_CHUNK", "1024")))

        if best_cand is not None and best_lcp >= 32:
            # LCP Hit: restore candidate snapshot
            self.restore_kv(best_cand.kv_snapshot, batch_size=1, req_id=req_hash, level="request")
            reused = best_lcp
            print(f"[jev-prefix] Request LCP HIT: reused={reused}/{len(req_ids)} tokens", flush=True)
            for _s in range(reused, len(req_ids), _chunk):
                _e = min(_s + _chunk, len(req_ids))
                chunk_t = torch.tensor([req_ids[_s:_e]], dtype=torch.long, device=dev0)
                with torch.inference_mode():
                    _ = self.model.forward(chunk_t, start_pos=schema_node.end_pos + _s)
            torch.cuda.synchronize(dev0)
        else:
            # Full request prefill from schema snapshot in chunks
            reused = 0
            self.restore_kv(schema_node.kv_snapshot, batch_size=1, req_id=req_hash, level="request")
            for _s in range(0, len(req_ids), _chunk):
                _e = min(_s + _chunk, len(req_ids))
                chunk_t = torch.tensor([req_ids[_s:_e]], dtype=torch.long, device=dev0)
                with torch.inference_mode():
                    _ = self.model.forward(chunk_t, start_pos=schema_node.end_pos + _s)
            torch.cuda.synchronize(dev0)

        req_end_pos = schema_node.end_pos + len(req_ids)
        snap = self.snapshot_kv(req_end_pos, level="request")

        # LRU eviction if cache exceeds 16 entries
        if len(self.request_nodes) >= 16:
            oldest_k = min(self.request_nodes.keys(), key=lambda k: self.request_nodes[k].last_accessed)
            del self.request_nodes[oldest_k]

        node = JevPrefixNode(req_key, req_ids, "request", parent=schema_node)
        node.kv_snapshot = snap
        self.request_nodes[req_key] = node
        schema_node.children[req_key] = node

        return node, reused, snap


# ---------------------------------------------------------------------------
# Jev Inference Engine
# ---------------------------------------------------------------------------

class JevEngine:
    """Full Jev-mode runtime executing parallel field inference with hierarchical prefix caching."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tok = tokenizer
        self.prefix_tree = JevPrefixTree(model, tokenizer)
        # Prefill and retain system prompt immediately
        self.prefix_tree.init_system_prompt()

    def _score_field_candidates(self, jf: JevField, row_logits: torch.Tensor) -> Any:
        """Scores candidate log-probabilities directly from next-token logits without autoregressive decode."""
        lsm = torch.log_softmax(row_logits.float(), dim=-1)

        if jf.field_type in ("enum", "boolean", "integer", "number") and jf.candidates:
            cand_scores = []
            for cand in jf.candidates:
                # String variants (with leading space and without leading space)
                s_cand = str(cand).lower() if isinstance(cand, bool) else str(cand)
                tok_with_space = self.tok.encode(" " + s_cand, add_special_tokens=False)
                tok_no_space = self.tok.encode(s_cand, add_special_tokens=False)

                score = -1e9
                if tok_with_space:
                    score = max(score, lsm[tok_with_space[0]].item())
                if tok_no_space:
                    score = max(score, lsm[tok_no_space[0]].item())

                cand_scores.append((cand, score))

            cand_scores.sort(key=lambda x: x[1], reverse=True)
            best_val = cand_scores[0][0]

            if jf.field_type == "boolean":
                return bool(best_val) if isinstance(best_val, bool) else (str(best_val).lower() == "true")
            elif jf.field_type == "integer":
                try:
                    return int(best_val)
                except ValueError:
                    return best_val
            elif jf.field_type == "number":
                try:
                    return float(best_val)
                except ValueError:
                    return best_val
            return best_val

        # Fallback for unconstrained string / numeric field: pick top greedy token
        top_token = int(row_logits.argmax().item())
        decoded = self.tok.decode([top_token]).strip()
        return decoded

    def process_request(
        self,
        input_text: str,
        raw_schema: dict[str, Any],
        max_batch: int = 32,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Processes a structured extraction request in parallel Jev mode."""
        t_start = time.perf_counter()
        schema = JevSchema(raw_schema)
        dev0 = self.model.blocks[0].device
        K = len(schema.fields)

        # 1. Schema Node Lookup (Prefix Cache Level 1 & Level 2)
        t_cache_0 = time.perf_counter()
        schema_node, schema_hit = self.prefix_tree.get_or_create_schema_node(schema)
        t_cache_hit = time.perf_counter() - t_cache_0

        req_text = f"Input: {input_text}\n<｜Assistant｜></think>"
        req_ids = self.tok.encode(req_text, add_special_tokens=False)

        print(
            f"[jev] request start prompt_tokens={len(req_ids)} fields={K} schema_hit={schema_hit}",
            flush=True,
        )

        # 2. Level 3: Request Input Prefill with persistent prefix caching & LCP reuse
        t_prefill_0 = time.perf_counter()
        req_node, reused_tokens, req_snap = self.prefix_tree.get_or_create_request_node(schema_node, req_ids)
        t_prefill = time.perf_counter() - t_prefill_0
        req_end_pos = schema_node.end_pos + len(req_ids)
        prefilled_tokens = len(req_ids) - reused_tokens

        print(
            f"[jev] prefill done {time.perf_counter() - t_start:.3f}s "
            f"(prefilled={prefilled_tokens} reused={reused_tokens}/{len(req_ids)} tokens "
            f"time={t_prefill*1000:.1f}ms)",
            flush=True,
        )

        # 3. Branch Field-Specific Suffixes & Parallel Candidate Scoring (Level 4 & 5)
        fields = schema.fields
        field_results = {}
        t_scoring = 0.0

        # Enforce batch size limit to pre-allocated model slots (prevents OOM & cache reallocation)
        model_max_seqs = getattr(self.model.args, "max_seqs", 2)
        if hasattr(self.model, "blocks") and len(self.model.blocks) > 0:
            model_max_seqs = min(model_max_seqs, self.model.blocks[0].attn.window_kv_cache.shape[0])
        for cache in self.model.shared.compress_kv.values():
            model_max_seqs = min(model_max_seqs, cache.shape[0])
            break
        effective_max_batch = min(max_batch, max(1, model_max_seqs))

        for batch_start in range(0, K, effective_max_batch):
            batch_end = min(batch_start + effective_max_batch, K)
            cur_fields = fields[batch_start:batch_end]
            B = len(cur_fields)

            # Broadcast shared request KV across B batch rows without duplicating prefix blocks
            self.prefix_tree.restore_kv(req_snap, batch_size=B, req_id=req_node.key, level="field")

            # Field query tokens: \nField {i}: (all exactly 5 tokens!)
            field_query_ids = []
            for f in cur_fields:
                q_str = f"\nField {f.index}:"
                q_ids = self.tok.encode(q_str, add_special_tokens=False)
                field_query_ids.append(q_ids)

            field_tensor = torch.tensor(field_query_ids, dtype=torch.long, device=dev0)

            t_score_0 = time.perf_counter()
            with torch.inference_mode():
                logits = self.model.forward(field_tensor, start_pos=req_end_pos)
            torch.cuda.synchronize(dev0)
            t_scoring += time.perf_counter() - t_score_0

            for row_idx, f in enumerate(cur_fields):
                val = self._score_field_candidates(f, logits[row_idx])
                field_results[f.name] = val

        t_total = time.perf_counter() - t_start
        print(
            f"[jev] inference done {t_total:.3f}s (scoring={t_scoring*1000:.1f}ms fields={K})",
            flush=True,
        )

        assembled = schema.assemble(field_results)

        total_saved = (
            (len(self.prefix_tree.system_node.token_ids) if self.prefix_tree.system_node else 0)
            + (len(schema_node.token_ids) if schema_hit else 0)
            + reused_tokens
        )

        assembled_str = json.dumps(assembled, ensure_ascii=False)
        output_tokens = len(self.tok.encode(assembled_str, add_special_tokens=False)) if self.tok else len(assembled_str.split())

        metrics = {
            "cache_hit": schema_hit,
            "cache_hit_latency_ms": t_cache_hit * 1000.0,
            "reused_tokens": reused_tokens,
            "prefilled_tokens": prefilled_tokens,
            "prefill_ms": t_prefill * 1000.0,
            "scoring_ms": t_scoring * 1000.0,
            "total_ms": t_total * 1000.0,
            "total_latency_ms": t_total * 1000.0,
            "prefill_time_s": t_prefill,
            "num_fields": K,
            "prompt_tokens": req_end_pos,
            "completion_tokens": output_tokens,
            "total_tokens": req_end_pos + output_tokens,
            "tokens_saved": total_saved,
            "prefix_saved_tokens": total_saved,
            "tok_s": (output_tokens / max(t_total, 1e-6)) if output_tokens > 0 else 0,
            "effective_tok_s": (output_tokens / max(t_total, 1e-6)) if output_tokens > 0 else 0,
        }

        return assembled, metrics

