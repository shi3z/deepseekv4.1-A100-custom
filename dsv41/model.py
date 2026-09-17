"""DeepSeek-V4.1-Flash on Ampere: our own single-process, multi-GPU (layer pipeline) runtime.

Numerics follow the released architecture definition, but every Hopper/Blackwell-only piece is
replaced: FP8 dense weights are dequantized to bf16 at load (cuBLAS), FP4 experts stay packed in VRAM
and go through our Triton grouped GEMM, the sparse attention / Sinkhorn run in torch, and the two
Engram hash tables live in host RAM (see engram.py). Activations are fake-quantized to FP8/FP4 at the
same points the reference quantizes them, so the bf16 GEMMs see the same rounded inputs."""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

import torch
import torch.nn.functional as F

from .w8 import W8, linear_w, oproj_a
from .fused import fake_quant_fp4, fake_quant_fp8, hc_post as _fused_hc_post, hc_pre as _fused_hc_pre, hc_split_sinkhorn, rmsnorm, rope_, sparse_attn_decode, swiglu_quant
from .kernels import sparse_attn as sparse_attn_prefill
from .moe_kernels import GroupedPairs, grouped_fp4_gemm, grouped_fp4_gemm_chunked
from . import cukern
from .quant import maybe_compile
import os


@dataclass
class Args:
    cfg: dict
    max_batch_size: int = 1  # rows decoded together
    max_seqs: int = 0  # sequence slots (caches); 0 = max_batch_size
    max_seq_len: int = 16384

    def __getattr__(self, k):
        try:
            return self.cfg[k]
        except KeyError as e:
            raise AttributeError(k) from e


PROF: dict[str, float] | None = None  # set to a dict to collect per-component seconds (with syncs)
ROUTE_STATS: dict | None = None  # layer -> expert hit counts


def _tick(key: str, t0: float) -> float:
    if PROF is not None:
        torch.cuda.synchronize()
        t = time.perf_counter()
        PROF[key] += t - t0
        return t
    return t0


# --------------------------------------------------------------------------- small pieces
def linear_fp8(x: torch.Tensor, w) -> torch.Tensor:
    """A Linear whose checkpoint weight is FP8: the reference quantizes the activation to FP8 (per-32,
    power-of-two scale) before the GEMM; we do the same rounding, then the GEMM (w: bf16 tensor or W8)."""
    return linear_w(fake_quant_fp8(x, 32), w)


@lru_cache(8)
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow, device) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs).to(device)



@lru_cache(32)
def get_rope_tables(
    dim,
    seqlen,
    original_seq_len,
    base,
    factor,
    beta_fast,
    beta_slow,
    device,
):
    """Shared RoPE tables.

    The old Attention.__init__ reused freqs_cis only opportunistically,
    then materialized fresh contiguous cos/sin arrays for every layer.

    At large max_seq_len that wastes substantial VRAM.  Layers with the
    same RoPE configuration on the same GPU can safely share these
    immutable tables.
    """
    freqs_cis = precompute_freqs_cis(
        dim,
        seqlen,
        original_seq_len,
        base,
        factor,
        beta_fast,
        beta_slow,
        device,
    )

    # view_as_real itself is a view.  Do not materialize an unnecessary
    # full [seq, dim/2, 2] temporary.
    cs = torch.view_as_real(freqs_cis)

    cos = cs[..., 0].contiguous()
    sin = cs[..., 1].contiguous()

    return freqs_cis, cos, sin

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """In-place rotation of the last dim (pairs as complex). x: [b, s, d] or [b, s, h, d]."""
    y = x
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if xc.ndim == 3:
        freqs_cis = freqs_cis.view(1, xc.size(1), xc.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, xc.size(1), 1, xc.size(-1))
    y.copy_(torch.view_as_real(xc * freqs_cis).flatten(-2))
    return y


@lru_cache(4)
def get_window_topk_idxs(
    window_size: int,
    bsz: int,
    seqlen: int,
    start_pos: int,
    device,
):
    """Return physical ring-buffer KV indices for every query token.

    Shape:
        [bsz, seqlen, width]

    For continuation prefill (start_pos > 0, seqlen > 1), each query row
    gets its own causal sliding window.  This is important because the
    whole continuation chunk is already written into the ring buffer
    before attention runs; later tokens in the same chunk must therefore
    be excluded from earlier query rows.

    The returned values are physical slots in window_kv_cache, not
    absolute token positions.
    """

    if seqlen <= 0:
        return torch.empty(
            (bsz, 0, 0),
            dtype=torch.int32,
            device=device,
        )

    # Keep the old compact representation for an initial prompt.  It
    # avoids materialising a full window width when the prompt itself is
    # shorter than the window.
    if start_pos == 0:
        qpos = torch.arange(
            seqlen,
            dtype=torch.long,
        ).unsqueeze(1)

        width = min(
            seqlen,
            window_size,
        )

        # Absolute positions represented by each row.
        abs_idx = (
            (qpos - window_size + 1).clamp_min(0)
            + torch.arange(
                width,
                dtype=torch.long,
            ).unsqueeze(0)
        )

        valid = abs_idx <= qpos

        # Initial prefill uses non-wrapped KV order.
        idxs = torch.where(
            valid,
            abs_idx,
            -1,
        )

    else:
        # --------------------------------------------------------
        # Continuation prefill / decode.
        #
        # Query i corresponds to absolute position:
        #
        #     q = start_pos + i
        #
        # Its causal window consists of:
        #
        #     q-window+1 ... q
        #
        # and each absolute position maps to the physical ring slot:
        #
        #     abs_pos % window_size
        # --------------------------------------------------------

        qpos = (
            start_pos
            + torch.arange(
                seqlen,
                dtype=torch.long,
            )
        ).unsqueeze(1)

        rel = torch.arange(
            window_size,
            dtype=torch.long,
        ).unsqueeze(0)

        abs_idx = (
            qpos
            - window_size
            + 1
            + rel
        )

        valid = (
            (abs_idx >= 0)
            & (abs_idx <= qpos)
        )

        physical = torch.remainder(
            abs_idx,
            window_size,
        )

        idxs = torch.where(
            valid,
            physical,
            -1,
        )

    return (
        idxs
        .to(torch.int32)
        .unsqueeze(0)
        .expand(bsz, -1, -1)
        .contiguous()
        .to(device)
    )

def select_candidate_blocks(logits, compress_lens, topk_blocks: int, block_size: int) -> torch.Tensor:
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


# --------------------------------------------------------------------------- shared attention state
class SharedAttn:
    """Shared compressed attention caches.

    Cache topology remains FULL MIRROR: every owner has a cache on every
    participating device.

    The difference is capacity management.  Instead of allocating the
    complete max_seq_len cache at startup, mirrors start small and grow
    geometrically when more compressed rows are actually needed.
    """

    def __init__(self):
        self.compress_kv: dict[tuple[int, torch.device], torch.Tensor] = {}
        self.index_k: dict[tuple[int, torch.device], torch.Tensor] = {}

        # Hard maximum number of compressed rows for each owner.
        # Filled by load_model().
        self.cache_max_rows: dict[int, int] = {}

        self.kv_owner: int = -1
        self.index_owner: int = -1
        self._topk_idxs_map: dict[int, torch.Tensor] = {}
        self._candidates_map: dict[int, torch.Tensor] = {}
        self._current_chunk_idx: int = 0

    @property
    def topk_idxs(self) -> torch.Tensor | None:
        return self._topk_idxs_map.get(getattr(self, "_current_chunk_idx", 0))

    @topk_idxs.setter
    def topk_idxs(self, val: torch.Tensor | None):
        chunk_idx = getattr(self, "_current_chunk_idx", 0)
        if val is None:
            self._topk_idxs_map.pop(chunk_idx, None)
        else:
            self._topk_idxs_map[chunk_idx] = val

    @property
    def candidates(self) -> torch.Tensor | None:
        return self._candidates_map.get(getattr(self, "_current_chunk_idx", 0))

    @candidates.setter
    def candidates(self, val: torch.Tensor | None):
        chunk_idx = getattr(self, "_current_chunk_idx", 0)
        if val is None:
            self._candidates_map.pop(chunk_idx, None)
        else:
            self._candidates_map[chunk_idx] = val

    def cleanup_chunk(self, chunk_idx: int):
        self._topk_idxs_map.pop(chunk_idx, None)
        self._candidates_map.pop(chunk_idx, None)

    @torch.inference_mode()
    def _ensure_capacity(
        self,
        table: dict,
        owner: int,
        need_rows: int,
        kind: str,
    ):
        """Grow every mirror of one owner's cache when necessary."""

        keys = [
            key
            for key in table.keys()
            if key[0] == owner
        ]

        if not keys:
            raise KeyError(
                f"{kind}: no cache mirrors exist for owner {owner}"
            )

        max_rows = self.cache_max_rows.get(owner)

        if max_rows is not None and need_rows > max_rows:
            raise RuntimeError(
                f"{kind}: owner {owner} needs {need_rows:,} rows, "
                f"but configured maximum is {max_rows:,}"
            )

        current = table[keys[0]].size(1)

        if need_rows <= current:
            return

        # Geometric growth.  This makes reallocations rare:
        #
        # 32K logical context
        #   -> 64K
        #   -> 128K
        #   -> 256K
        #   -> 512K
        #   -> 1M
        #
        # Rows themselves are compressed according to the owner's ratio.
        # Long cold-prefill can run close to the VRAM limit. In exact mode
        # avoid doubling a 64K allocation to 128K when only a few more
        # rows are needed; the temporary attention buffers need that headroom.
        if os.environ.get("DSV41_EXACT_CACHE_GROW", "0") == "1":
            target = need_rows
        else:
            target = max(need_rows, max(current * 2, 1))

        if max_rows is not None:
            target = min(target, max_rows)

        print(
            f"[cache-grow] {kind} owner={owner} "
            f"rows={current:,}->{target:,} "
            f"need={need_rows:,} mirrors={len(keys)}",
            flush=True,
        )

        # Grow mirrors one at a time so we do not hold all old+new
        # mirrors simultaneously.
        for key in keys:
            old = table[key]

            if old.size(1) >= need_rows:
                continue

            new = torch.empty(
                old.size(0),
                target,
                old.size(2),
                dtype=old.dtype,
                device=old.device,
            )

            # Only existing rows need preserving.  Future rows need not
            # be initialized because callers never read beyond the
            # current logical compressed length.
            new[:, :old.size(1)].copy_(old)

            table[key] = new

            del old

    def write_compress_kv(
        self,
        owner: int,
        bsz: int,
        pos: int,
        rows: torch.Tensor,
    ):
        self.kv_owner = owner

        end = pos + rows.size(1)

        self._ensure_capacity(
            self.compress_kv,
            owner,
            end,
            "compress_kv",
        )

        for (o, dev), cache in self.compress_kv.items():
            if o == owner:
                if dev == rows.device:
                    cache[:bsz, pos:end].copy_(rows)
                else:
                    with torch.cuda.device(dev):
                        target_row = rows.to(dev, non_blocking=True)
                        cache[:bsz, pos:end].copy_(
                            target_row,
                            non_blocking=True,
                        )
                        target_row.record_stream(
                            torch.cuda.current_stream(dev)
                        )

    def write_index_k(
        self,
        owner: int,
        bsz: int,
        pos: int,
        rows: torch.Tensor,
    ):
        self.index_owner = owner

        end = pos + rows.size(1)

        self._ensure_capacity(
            self.index_k,
            owner,
            end,
            "index_k",
        )

        for (o, dev), cache in self.index_k.items():
            if o == owner:
                if dev == rows.device:
                    cache[:bsz, pos:end].copy_(rows)
                else:
                    with torch.cuda.device(dev):
                        target_row = rows.to(dev, non_blocking=True)
                        cache[:bsz, pos:end].copy_(
                            target_row,
                            non_blocking=True,
                        )
                        target_row.record_stream(
                            torch.cuda.current_stream(dev)
                        )

# --------------------------------------------------------------------------- attention
class Compressor:
    def __init__(self, args: Args, layer_id: int, w: dict, device):
        self.ratio = args.compress_ratios[layer_id]
        self.norm_w = w["compressor.norm.weight"]
        self.eps = args.norm_eps
        self.wkv = w["compressor.wkv.weight"].float() if self.ratio > 1 else w["compressor.wkv.weight"]
        if self.ratio > 1:
            self.wgate = w["compressor.wgate.weight"].float()
            shape = (args.max_seqs or args.max_batch_size, self.ratio, args.head_dim)
            self.kv_state = torch.zeros(shape, dtype=torch.float32, device=device)
            self.score_state = torch.full(shape, -torch.inf, dtype=torch.float32, device=device)
            # ring of the raw kv / score of the last RING positions per sequence (slot = position % RING): the static
            # per-row path pools a pair from it, so several positions of one sequence can be in flight at once
            self.RING = 8
            self.kv_ring = torch.zeros(args.max_seqs or args.max_batch_size, self.RING, args.head_dim, dtype=torch.float32, device=device)
            self.score_ring = torch.full((args.max_seqs or args.max_batch_size, self.RING, args.head_dim), -torch.inf, dtype=torch.float32, device=device)

    def __call__(self, x: torch.Tensor, start_pos: int) -> torch.Tensor | None:
        bsz, seqlen, _ = x.size()
        ratio, dtype = self.ratio, x.dtype
        if ratio == 1:
            return rmsnorm(F.linear(x, self.wkv), self.norm_w, self.eps)
        xf = x.float()
        kv, score = F.linear(xf, self.wkv), F.linear(xf, self.wgate)
        # DecodeRuntime.compressor2 updates the ring, not the legacy partial
        # group buffers. Rebuild a continuation's partial group BEFORE this
        # chunk overwrites its ring slots (large chunks wrap the entire ring).
        if start_pos > 0:
            partial = start_pos % ratio
            if partial:
                previous = torch.arange(start_pos - partial, start_pos, device=x.device) % self.RING
                self.kv_state[:bsz, :partial].copy_(self.kv_ring[:bsz, previous])
                self.score_state[:bsz, :partial].copy_(self.score_ring[:bsz, previous])
        n_ring = min(self.RING, seqlen)  # keep the last positions in the ring
        for j in range(seqlen - n_ring, seqlen):
            self.kv_ring[:bsz, (start_pos + j) % self.RING] = kv[:, j]
            self.score_ring[:bsz, (start_pos + j) % self.RING] = score[:, j]
        if start_pos == 0:
            should = seqlen >= ratio
            rem = seqlen % ratio
            cut = seqlen - rem
            if rem:
                kv, self.kv_state[:bsz, :rem] = kv.split([cut, rem], dim=1)
                score, self.score_state[:bsz, :rem] = score.split([cut, rem], dim=1)
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            # Pool complete groups together. A Python loop per compression
            # group launched thousands of tiny CUDA operations per block.
            # Only the first partial group needs the previous ring state.
            pooled = []
            j = 0
            partial = start_pos % ratio
            if partial:
                take = min(ratio - partial, seqlen)
                self.kv_state[:bsz, partial:partial + take] = kv[:, :take]
                self.score_state[:bsz, partial:partial + take] = score[:, :take]
                j = take
                if partial + take == ratio:
                    pooled.append((self.kv_state[:bsz] *
                                   self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True))

            end = j + ((seqlen - j) // ratio) * ratio
            if end > j:
                groups = kv[:, j:end].unflatten(1, (-1, ratio))
                gates = score[:, j:end].unflatten(1, (-1, ratio))
                pooled.append((groups * gates.softmax(dim=2)).sum(dim=2))
                # Preserve even unused state slots for snapshot compatibility.
                self.kv_state[:bsz] = kv[:, end - ratio:end]
                self.score_state[:bsz] = score[:, end - ratio:end]
            if end < seqlen:
                tail = seqlen - end
                self.kv_state[:bsz, :tail] = kv[:, end:]
                self.score_state[:bsz, :tail] = score[:, end:]
            if not pooled:
                return None
            kv = torch.cat(pooled, dim=1) if len(pooled) > 1 else pooled[0]

        return rmsnorm(
            kv.to(dtype),
            self.norm_w,
            self.eps,
        )


class Indexer:
    def __init__(self, args: Args, layer_id: int, w: dict, device, shared: SharedAttn):
        self.shared = shared
        self.device = device
        self.layer_id = layer_id
        self.owns_k = layer_id in args.kv_source_layers
        self.index_owner = max([o for o in args.kv_source_layers if o <= layer_id], default=-1)
        self.ratio = args.compress_ratios[layer_id]
        self.is_candidate_source = layer_id == args.candidate_source_layer
        self.uses_candidates = 0 <= args.candidate_source_layer < layer_id
        self.candidate_topk_blocks = args.candidate_topk_blocks
        self.candidate_block_size = args.candidate_block_size
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.topk = args.index_topk
        self.softmax_scale = self.head_dim**-0.5
        self.wq_b = w["indexer.wq_b.weight"]  # bf16 (from fp8)
        self.weights_proj = w["indexer.weights_proj.weight"]
        self.freqs_cis = None
        if self.owns_k:
            self.wk = w["indexer.wk.weight"]
            self.k_norm_w = w["indexer.k_norm.weight"]
            self.eps = args.norm_eps

    def __call__(self, x, qr, latent, start_pos: int, offset: int):
        bsz, seqlen, _ = x.size()
        ratio, rd, end_pos = self.ratio, self.rope_head_dim, start_pos + seqlen

        if self.owns_k:
            # consumers up to the next owner read this layer's keys
            self.shared.index_owner = self.layer_id

        if self.owns_k and latent is not None:
            k = rmsnorm(F.linear(latent, self.wk), self.k_norm_w, self.eps)
            if start_pos == 0:
                rope_(
                    k,
                    rd,
                    self.cos,
                    self.sin,
                    0,
                    pos_stride=ratio,
                )
            else:
                # Multi-token continuation can emit more than one
                # compressed key.  The first emitted key belongs to
                # the compression group containing start_pos; later
                # keys are `ratio` tokens apart.
                first_group_start = (
                    start_pos
                    - (start_pos % ratio)
                )

                rope_(
                    k,
                    rd,
                    self.cos,
                    self.sin,
                    first_group_start,
                    pos_stride=ratio,
                )

            k = fake_quant_fp4(k, 32)
            self.shared.write_index_k(
                self.layer_id,
                bsz,
                start_pos // ratio,
                k,
            )

        q = linear_fp8(qr, self.wq_b).unflatten(
            -1, (self.n_heads, self.head_dim)
        )
        rope_(q, rd, self.cos, self.sin, start_pos)
        q = fake_quant_fp4(q, 32)

        index_k = self.shared.index_k[
            (self.index_owner, self.device)
        ][:bsz, : end_pos // ratio]

        weights = F.linear(x, self.weights_proj) * (
            self.softmax_scale * self.n_heads**-0.5
        )

        # ===============================================================
        # CHUNKED PREFILL
        #
        # Original:
        #   [B,S,H,T] fp32
        #
        # 5767 x 32 x 5767 x 4 bytes ~= 3.96 GiB.
        #
        # Chunk only the query dimension. This is mathematically the same
        # operation, but peak scratch becomes roughly:
        #
        #   [B,128,H,T]
        #
        # Decode keeps the original path.
        # ===============================================================
        # Continuation prefill has the same score scratch as cold prefill.
        # Bound it for every multi-token call, including prefix replay.
        query_chunk = max(1, int(os.environ.get("DSV41_INDEX_QUERY_CHUNK", "64")))
        if seqlen > 1:
            QUERY_CHUNK = query_chunk

            key_len = index_k.size(1)
            key_pos = torch.arange(
                key_len,
                device=x.device,
            )

            # One conversion per layer instead of per query chunk.
            index_k_f = index_k.float()
            kT = index_k_f.transpose(1, 2)

            idx_chunks = []
            candidate_chunks = [] if self.is_candidate_source else None

            topk = min(self.topk, end_pos // ratio)

            for q0 in range(0, seqlen, QUERY_CHUNK):
                q1 = min(q0 + QUERY_CHUNK, seqlen)

                qc = q[:, q0:q1].float()
                wc = weights[:, q0:q1].float()

                # Memory-efficient head accumulation: [B, C, T] instead of [B, C, H, T]
                # Completely eliminates multi-GiB scratch tensors for long contexts (1M tokens).
                score = torch.zeros(
                    bsz,
                    q1 - q0,
                    key_len,
                    dtype=torch.float32,
                    device=x.device,
                )
                for h_idx in range(self.n_heads):
                    sh = torch.bmm(qc[:, :, h_idx], kT)
                    sh.relu_()
                    sh.mul_(wc[:, :, h_idx].unsqueeze(-1))
                    score.add_(sh)

                # Preserve original causal/compression visibility.
                compress_lens = (
                    (
                        start_pos
                        + torch.arange(
                            q0,
                            q1,
                            device=x.device,
                        )
                        + 1
                    )
                    // ratio
                ).unsqueeze(-1)

                score.masked_fill_(
                    key_pos >= compress_lens,
                    -torch.inf,
                )

                if self.is_candidate_source:
                    cand = select_candidate_blocks(
                        score,
                        compress_lens,
                        self.candidate_topk_blocks,
                        self.candidate_block_size,
                    )
                    candidate_chunks.append(cand)

                elif self.uses_candidates:
                    cand = self.shared.candidates[:, q0:q1]

                    if cand.device != score.device:
                        cand = cand.to(
                            score.device,
                            non_blocking=True,
                        )

                    score.masked_fill_(
                        ~cand,
                        -torch.inf,
                    )

                idx = (
                    score
                    .topk(
                        topk,
                        dim=-1,
                        sorted=False,
                    )
                    .indices
                    .sort(dim=-1)
                    .values
                )

                idx_chunks.append(
                    torch.where(
                        idx < compress_lens,
                        idx + offset,
                        -1,
                    ).int()
                )

                del qc, wc, score

            if self.is_candidate_source:
                self.shared.candidates = torch.cat(
                    candidate_chunks,
                    dim=1,
                )

            return torch.cat(idx_chunks, dim=1)

        # ===============================================================
        # Original path for decode / short prefill
        # ===============================================================

        index_score = torch.einsum(
            "bshd,btd->bsht",
            q.float(),
            index_k.float(),
        )

        # Original was:
        #
        # (index_score.relu_() *
        #  weights.float().unsqueeze(-1)).sum(dim=2)
        #
        # mul_ avoids another huge allocation.
        index_score.relu_()
        index_score.mul_(
            weights.float().unsqueeze(-1)
        )
        index_score = index_score.sum(dim=2)

        if start_pos == 0:
            compress_lens = (
                torch.arange(
                    1,
                    seqlen + 1,
                    device=x.device,
                ) // ratio
            ).unsqueeze(-1)

            index_score.masked_fill_(
                torch.arange(
                    index_k.size(1),
                    device=x.device,
                ) >= compress_lens,
                -torch.inf,
            )
        else:
            # ----------------------------------------------------
            # Causal continuation visibility.
            #
            # Old code used:
            #
            #     compress_lens = end_pos // ratio
            #
            # for the whole continuation chunk.  That exposed keys
            # produced near the END of the chunk to queries near its
            # BEGINNING.
            #
            # Query i has absolute position:
            #
            #     start_pos + i
            #
            # and may see only compression groups fully completed by
            # that position:
            #
            #     (start_pos + i + 1) // ratio
            #
            # Shape [S,1] broadcasts against [B,S,T].
            # ----------------------------------------------------
            compress_lens = (
                (
                    start_pos
                    + torch.arange(
                        seqlen,
                        device=x.device,
                    )
                    + 1
                )
                // ratio
            ).unsqueeze(-1)

            key_pos = torch.arange(
                index_k.size(1),
                device=x.device,
            )

            index_score.masked_fill_(
                key_pos >= compress_lens,
                -torch.inf,
            )

        if self.is_candidate_source:
            self.shared.candidates = select_candidate_blocks(
                index_score,
                compress_lens,
                self.candidate_topk_blocks,
                self.candidate_block_size,
            )

        elif self.uses_candidates:
            index_score = index_score.masked_fill(
                ~self.shared.candidates.to(index_score.device),
                -torch.inf,
            )

        topk = min(self.topk, end_pos // ratio)

        idxs = (
            index_score
            .topk(
                topk,
                dim=-1,
                sorted=False,
            )
            .indices
            .sort(dim=-1)
            .values
        )

        return torch.where(
            idxs < compress_lens,
            idxs + offset,
            -1,
        ).int()

class Attention:
    def __init__(self, args: Args, layer_id: int, w: dict, device, shared: SharedAttn):
        self.args = args
        self.layer_id = layer_id
        self.device = device
        self.shared = shared
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.n_groups = args.o_groups
        self.o_lora_rank = args.o_lora_rank
        self.window = args.window_size
        self.ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps
        self.softmax_scale = self.head_dim**-0.5
        self.attn_sink = w["attn.attn_sink"].float()
        self.wq_a, self.q_norm_w, self.wq_b = w["attn.wq_a.weight"], w["attn.q_norm.weight"], w["attn.wq_b.weight"]
        self.wkv, self.kv_norm_w = w["attn.wkv.weight"], w["attn.kv_norm.weight"]
        self.wo_a = w["attn.wo_a.weight"]  # block-diagonal over groups: rows g*rank..(g+1)*rank see only group g (see w8.oproj_a)
        self.wo_b = w["attn.wo_b.weight"]
        self.is_kv_source = layer_id in args.kv_source_layers
        self.is_index_source = layer_id in args.index_source_layers
        self.kv_owner = max([o for o in args.kv_source_layers if o <= layer_id], default=-1)
        self.compressor = Compressor(args, layer_id, {k[5:]: v for k, v in w.items() if k.startswith("attn.compressor")}, device) if self.is_kv_source else None
        self.indexer = Indexer(args, layer_id, {k[5:]: v for k, v in w.items() if k.startswith("attn.indexer")}, device, shared) if self.is_index_source else None
        self.window_kv_cache = torch.zeros(args.max_seqs or args.max_batch_size, self.window, self.head_dim, dtype=torch.bfloat16, device=device)
        if self.ratio:
            original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
        else:
            original_seq_len, rope_theta = 0, args.rope_theta
        self.freqs_cis, self.cos, self.sin = get_rope_tables(
            self.rope_head_dim,
            args.max_seq_len,
            original_seq_len,
            rope_theta,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
            device,
        )
        if self.indexer is not None:
            self.indexer.freqs_cis = self.freqs_cis
            self.indexer.cos, self.indexer.sin = self.cos, self.sin

    def _window_kv(self, x, freqs_cis, start_pos):
        """Build causal sliding-window KV.

        Important continuation invariant:

        For start_pos > 0 and seqlen > 1 we MUST NOT use the mutable
        ring buffer itself as the attention KV after writing the whole
        chunk.

        Doing so lets earlier queries in the chunk observe ring slots
        already overwritten by later (future) tokens.

        Instead:

          1. copy the required OLD history out of the ring
          2. concatenate old history + current chunk into temporary KV
          3. build per-query causal indices into that temporary KV
          4. commit the current chunk into the persistent ring

        The temporary tensor remains independent after the ring commit,
        so attention sees the correct historical contents.
        """
        bsz, seqlen, _ = x.size()
        win = self.window

        kv = rmsnorm(
            linear_fp8(x, self.wkv),
            self.kv_norm_w,
            self.eps,
        )

        rope_(
            kv,
            self.rope_head_dim,
            self.cos,
            self.sin,
            start_pos,
        )

        kv = fake_quant_fp8(kv, 32)

        # ----------------------------------------------------
        # Normal initial prefill.
        # ----------------------------------------------------
        if start_pos == 0:
            if seqlen <= win:
                self.window_kv_cache[
                    :bsz,
                    :seqlen,
                ] = kv
            else:
                cut = seqlen % win

                self.window_kv_cache[
                    :bsz,
                    cut:win,
                ], self.window_kv_cache[
                    :bsz,
                    :cut,
                ] = kv[:, -win:].split(
                    [win - cut, cut],
                    dim=1,
                )

            window_kv = kv

            topk_idxs = get_window_topk_idxs(
                win,
                bsz,
                seqlen,
                start_pos,
                x.device,
            )

            return window_kv, topk_idxs

        # ----------------------------------------------------
        # Fast scalar decode.
        # Keep the original behaviour.
        # ----------------------------------------------------
        if seqlen == 1:
            slot = start_pos % win

            self.window_kv_cache[
                :bsz,
                slot,
            ] = kv.squeeze(1)

            window_kv = self.window_kv_cache[:bsz]

            topk_idxs = get_window_topk_idxs(
                win,
                bsz,
                1,
                start_pos,
                x.device,
            )

            return window_kv, topk_idxs

        # ----------------------------------------------------
        # Causal multi-token continuation.
        # ----------------------------------------------------

        # A query at start_pos needs at most win-1 OLD tokens plus
        # itself.  Capture them BEFORE modifying the ring.
        hist_len = min(
            int(start_pos),
            int(win - 1),
        )

        if hist_len:
            hist_abs = torch.arange(
                start_pos - hist_len,
                start_pos,
                device=x.device,
                dtype=torch.long,
            )

            hist_slots = torch.remainder(
                hist_abs,
                win,
            )

            # Advanced indexing creates a separate temporary tensor.
            history = self.window_kv_cache[
                :bsz,
                hist_slots,
            ].clone()

            window_kv = torch.cat(
                [history, kv],
                dim=1,
            )
        else:
            window_kv = kv

        # temp layout:
        #
        #   [ old-history ][ current chunk ]
        #     hist_len       seqlen
        #
        # Query i is temporary position hist_len+i and may attend to
        # at most the preceding win-1 positions plus itself.
        temp_len = hist_len + seqlen
        width = min(
            win,
            temp_len,
        )

        q = (
            hist_len
            + torch.arange(
                seqlen,
                device=x.device,
                dtype=torch.long,
            )
        ).unsqueeze(1)

        rel = torch.arange(
            width,
            device=x.device,
            dtype=torch.long,
        ).unsqueeze(0)

        first = (
            q - win + 1
        ).clamp_min(0)

        idxs = first + rel

        valid = idxs <= q

        idxs = torch.where(
            valid,
            idxs,
            -1,
        )

        topk_idxs = (
            idxs
            .to(torch.int32)
            .unsqueeze(0)
            .expand(bsz, -1, -1)
            .contiguous()
        )

        # Now that the historical values used by attention have been
        # copied, commit the whole new chunk to the persistent ring.
        #
        # If seqlen >= win only the final win tokens can survive.
        if seqlen >= win:
            tail = kv[:, -win:]

            end_pos = start_pos + seqlen
            first_abs = end_pos - win

            slots = torch.remainder(
                torch.arange(
                    first_abs,
                    end_pos,
                    device=x.device,
                    dtype=torch.long,
                ),
                win,
            )

            self.window_kv_cache[
                :bsz,
                slots,
            ] = tail

        else:
            slots = torch.remainder(
                start_pos
                + torch.arange(
                    seqlen,
                    device=x.device,
                    dtype=torch.long,
                ),
                win,
            )

            self.window_kv_cache[
                :bsz,
                slots,
            ] = kv

        return window_kv, topk_idxs

    def _compress_kv(self, x, qr, start_pos, offset):
        bsz, seqlen, _ = x.size()
        ratio = self.ratio
        compress_len = (start_pos + seqlen) // ratio
        latent = self.compressor(x, start_pos) if self.is_kv_source else None
        if self.is_kv_source:
            self.shared.kv_owner = self.layer_id
        # indexer needs the pre-RoPE latent, so it runs before the cache write
        if self.is_index_source:
            if compress_len == 0:
                idxs = torch.empty(bsz, seqlen, 0, dtype=torch.int32, device=x.device)
            else:
                idxs = self.indexer(x, qr, latent, start_pos, offset)
            self.shared.topk_idxs = idxs
        else:
            idxs = self.shared.topk_idxs.to(x.device, non_blocking=True)
        if latent is not None:
            latent = latent.contiguous()
            if start_pos == 0:
                rope_(
                    latent,
                    self.rope_head_dim,
                    self.cos,
                    self.sin,
                    0,
                    pos_stride=ratio,
                )
            else:
                # The first compressed row belongs to the compression
                # group containing start_pos.  Additional rows are
                # spaced by `ratio` absolute token positions.
                first_group_start = (
                    start_pos
                    - (start_pos % ratio)
                )

                rope_(
                    latent,
                    self.rope_head_dim,
                    self.cos,
                    self.sin,
                    first_group_start,
                    pos_stride=ratio,
                )
            latent = fake_quant_fp4(latent, 16, scale_e4m3=True)
            self.shared.write_compress_kv(self.layer_id, bsz, start_pos // ratio, latent)
        return self.shared.compress_kv[(self.kv_owner, self.device)][:bsz, :compress_len], idxs

    def __call__(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        rd = self.rope_head_dim
        qr = rmsnorm(linear_fp8(x, self.wq_a), self.q_norm_w, self.eps)
        q = linear_fp8(qr, self.wq_b).unflatten(-1, (self.n_heads, self.head_dim))
        rope_(q, rd, self.cos, self.sin, start_pos)
        kv, topk_idxs = self._window_kv(x, freqs_cis, start_pos)

        if (
            os.environ.get(
                "DSV41_DEBUG_COLD_PREFILL_STAGE",
                "0",
            ) == "1"
            and start_pos == 0
        ):
            torch.cuda.synchronize(self.device)
            print(
                f"[cold-prefill-stage] "
                f"layer={self.layer_id} "
                f"AFTER-WINDOW "
                f"seqlen={seqlen} "
                f"kv={tuple(kv.shape)} "
                f"topk={tuple(topk_idxs.shape)}",
                flush=True,
            )


        if (
            os.environ.get("DSV41_DEBUG_ATTN_STAGE", "0") == "1"
            and 32760 <= start_pos <= 32780
        ):
            torch.cuda.synchronize(self.device)
            print(
                f"[attn-stage] layer={self.layer_id} "
                f"pos={start_pos} AFTER-WINDOW "
                f"kv={tuple(kv.shape)} "
                f"topk={tuple(topk_idxs.shape)}",
                flush=True,
            )

        if self.ratio:
            if (
                os.environ.get("DSV41_DEBUG_ATTN_STAGE", "0") == "1"
                and 32760 <= start_pos <= 32780
            ):
                print(
                    f"[attn-stage] layer={self.layer_id} "
                    f"pos={start_pos} BEFORE-COMPRESS "
                    f"ratio={self.ratio} "
                    f"offset={kv.size(1)}",
                    flush=True,
                )

            ckv, cidx = self._compress_kv(
                x,
                qr,
                start_pos,
                kv.size(1),
            )

            if (
                os.environ.get(
                    "DSV41_DEBUG_COLD_PREFILL_STAGE",
                    "0",
                ) == "1"
                and start_pos == 0
            ):
                torch.cuda.synchronize(self.device)
                print(
                    f"[cold-prefill-stage] "
                    f"layer={self.layer_id} "
                    f"AFTER-COMPRESS "
                    f"ckv={tuple(ckv.shape)} "
                    f"cidx={tuple(cidx.shape)}",
                    flush=True,
                )


            if (
                os.environ.get("DSV41_DEBUG_ATTN_STAGE", "0") == "1"
                and 32760 <= start_pos <= 32780
            ):
                torch.cuda.synchronize(self.device)
                print(
                    f"[attn-stage] layer={self.layer_id} "
                    f"pos={start_pos} AFTER-COMPRESS "
                    f"ckv={tuple(ckv.shape)} "
                    f"cidx={tuple(cidx.shape)}",
                    flush=True,
                )

            kv = torch.cat([kv, ckv], dim=1)
            topk_idxs = torch.cat([topk_idxs, cidx], dim=-1)

        # ------------------------------------------------------------
        # Prefix replay boundary guard.
        #
        # Catch a malformed sparse-attention index before it reaches
        # torch advanced indexing / gather and poisons the CUDA context.
        # Enabled only for explicit debugging.
        # ------------------------------------------------------------
        if (
            os.environ.get(
                "DSV41_DEBUG_ATTN_BOUNDARY",
                "0",
            ) == "1"
            and 32760 <= start_pos <= 32780
        ):
            # Make sure a failure reported below really belongs to this
            # attention invocation rather than an earlier async kernel.
            torch.cuda.synchronize(self.device)

            valid_idx = topk_idxs[topk_idxs >= 0]

            if valid_idx.numel():
                idx_min = int(valid_idx.min().item())
                idx_max = int(valid_idx.max().item())
            else:
                idx_min = -1
                idx_max = -1

            kv_rows = int(kv.size(1))

            ckv_rows = (
                int(ckv.size(1))
                if self.ratio
                else 0
            )

            cidx_valid = (
                cidx[cidx >= 0]
                if self.ratio
                else None
            )

            if (
                cidx_valid is not None
                and cidx_valid.numel()
            ):
                cidx_min = int(cidx_valid.min().item())
                cidx_max = int(cidx_valid.max().item())
            else:
                cidx_min = -1
                cidx_max = -1

            print(
                f"[attn-boundary] "
                f"layer={self.layer_id} "
                f"pos={start_pos} "
                f"ratio={self.ratio} "
                f"window={self.window} "
                f"kv_rows={kv_rows} "
                f"ckv_rows={ckv_rows} "
                f"idx=[{idx_min},{idx_max}] "
                f"cidx=[{cidx_min},{cidx_max}] "
                f"topk_shape={tuple(topk_idxs.shape)}",
                flush=True,
            )

            if idx_max >= kv_rows:
                raise RuntimeError(
                    "ATTN OOB BEFORE CUDA: "
                    f"layer={self.layer_id} "
                    f"pos={start_pos} "
                    f"ratio={self.ratio} "
                    f"idx_max={idx_max} "
                    f"kv_rows={kv_rows} "
                    f"cidx_max={cidx_max} "
                    f"ckv_rows={ckv_rows}"
                )

        # Prefix-cache replay deliberately uses the prefill kernel even
        # for a single token.  The optimized decode Triton kernel assumes
        # the normal decode layout and currently fails to compile for the
        # rollback/replay path at arbitrary start_pos.
        force_prefill = bool(
            getattr(self, "_prefix_replay_prefill", False)
        )

        if seqlen == 1 and not force_prefill:
            o = sparse_attn_decode(
                q,
                kv,
                self.attn_sink,
                topk_idxs,
                self.softmax_scale,
            )
        else:
            o = sparse_attn_prefill(
                q,
                kv,
                self.attn_sink,
                topk_idxs,
                self.softmax_scale,
            )

            if (
                os.environ.get(
                    "DSV41_DEBUG_COLD_PREFILL_STAGE",
                    "0",
                ) == "1"
                and start_pos == 0
            ):
                torch.cuda.synchronize(self.device)
                print(
                    f"[cold-prefill-stage] "
                    f"layer={self.layer_id} "
                    f"AFTER-ATTN "
                    f"kv_rows={kv.size(1)} "
                    f"topk={tuple(topk_idxs.shape)}",
                    flush=True,
                )

        rope_(o, rd, self.cos, self.sin, start_pos, inverse=True)
        o = o.view(bsz, seqlen, self.n_groups, -1)
        o = oproj_a(o, self.wo_a, self.n_groups, self.o_lora_rank)
        return linear_fp8(o, self.wo_b)


# --------------------------------------------------------------------------- MoE
class MoE:
    def __init__(self, args: Args, layer_id: int, w: dict, device):
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_experts = args.n_routed_experts
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.gate_temp = args.cfg.get("gate_temp", 1.0)
        self.norm_topk_prob = args.cfg.get("norm_topk_prob", True)
        self.route_scale = args.route_scale
        self.swiglu_limit = args.swiglu_limit
        self.gate_w = w["ffn.gate.weight"].float()
        self.gate_bias = w["ffn.gate.bias"].float()
        self.w13 = w["experts.w13"]  # uint8 [E, 2*inter, dim/2]  (w1 rows then w3 rows)
        self.s13 = w["experts.s13"]  # uint8 [E, 2*inter, dim/32]
        self.w2 = w["experts.w2"]  # uint8 [E, dim, inter/2]
        self.s2 = w["experts.s2"]  # uint8 [E, dim, inter/32]
        self.inter = self.w2.shape[2] * 2
        self.offload = bool(w.get("experts.offload", False))  # experts in host RAM (streamed to the GPU, or computed on the CPU)
        self.ep = w.get("experts.ep")  # expert-parallel shards: [{device, start, n, w13, s13, w2, s2}] (see dsv41/ep.py)
        self.host = w.get("experts.host")  # HostExperts: compute the selected experts on the CPU
        self.hot = w.get("experts.hot")  # GPU-resident subset (slot map) computed on the GPU, overlapping the CPU
        if self.host is not None:
            self.x_host = torch.empty(self.dim, dtype=torch.bfloat16, pin_memory=True)
            self.y_host = torch.empty(self.dim, dtype=torch.float32, pin_memory=True)
            self.y_host.zero_()
        # shared expert: one GEMM for gate and up (rows [w1; w3])
        w1, w3 = w["ffn.shared_experts.w1.weight"], w["ffn.shared_experts.w3.weight"]
        self.sh_w13 = W8.cat([w1, w3]) if isinstance(w1, W8) else torch.cat([w1, w3], dim=0).contiguous()
        self.sh_w2 = w["ffn.shared_experts.w2.weight"]
        self.device = device
        self._cache: dict = {}

    def gate(self, x: torch.Tensor):
        scores = F.linear(x.float(), self.gate_w) / self.gate_temp
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        indices = (scores + self.gate_bias).topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights /= weights.sum(dim=-1, keepdim=True) + 1e-20
        weights *= self.route_scale
        return weights, indices

    def swiglu(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return _swiglu(gate, up, self.swiglu_limit)

    def shared_expert(self, x: torch.Tensor) -> torch.Tensor:
        gu = linear_fp8(x, self.sh_w13).float()
        h = swiglu_quant(gu, None, self.inter, self.swiglu_limit)  # bf16, already FP8-rounded for w2
        return linear_w(h, self.sh_w2).float()

    def _forward_ep(self, xq, eid, tok, weights, n_tok, n_pairs):
        """Parallel low-memory EP prefill with routed-xq compaction.

        Remote shards normally receive only the token rows that actually
        route to experts resident on that shard.

        Set:
            DSV41_EP_COMPACT_XQ=0
        to restore full-xq peer copies for an A/B benchmark.

        The pair output is reduced directly into [n_tok, dim], so the
        old [n_pairs, dim] FP32 owner buffer remains eliminated.
        """
        owner = xq.device

        compact_xq = os.environ.get(
            "DSV41_EP_COMPACT_XQ",
            "1",
        ) != "0"

        y = torch.zeros(
            n_tok,
            self.dim,
            dtype=torch.float32,
            device=owner,
        )

        wflat = weights.flatten().float()

        if not hasattr(self, "_ep_prefill_streams"):
            self._ep_prefill_streams = {}

        jobs = []

        remote_full_rows = 0
        remote_sent_rows = 0

        # --------------------------------------------------------------
        # Route preparation on layer owner.
        #
        # tok is monotonically nondecreasing because it originates from
        # repeat_interleave(topk).  After sel preserves pair order,
        # unique_consecutive is sufficient and avoids a sorting unique().
        # --------------------------------------------------------------
        for sh in self.ep:
            sel = (
                (eid >= sh["start"])
                & (eid < sh["start"] + sh["n"])
            ).nonzero().flatten()

            m = int(sel.numel())
            if m == 0:
                continue

            d = sh["device"]

            stream = self._ep_prefill_streams.get(d)
            if stream is None:
                stream = torch.cuda.Stream(device=d)
                self._ep_prefill_streams[d] = stream

            pair_tok_owner = tok[sel].to(torch.int64)

            job = {
                "sh": sh,
                "sel": sel,
                "m": m,
                "device": d,
                "stream": stream,
                "pair_tok_owner": pair_tok_owner,
                "le_owner": (
                    eid[sel] - sh["start"]
                ).to(torch.int32),
                "wt_owner": wflat[sel].contiguous(),
            }

            if d != owner:
                remote_full_rows += n_tok

                if compact_xq:
                    uniq, inv = torch.unique_consecutive(
                        pair_tok_owner,
                        return_inverse=True,
                    )

                    job["uniq_tok_owner"] = uniq
                    job["tk_compute_owner"] = inv.to(torch.int32)

                    remote_sent_rows += int(uniq.numel())
                else:
                    job["tk_compute_owner"] = pair_tok_owner.to(
                        torch.int32
                    )
                    remote_sent_rows += n_tok
            else:
                # Local owner shard needs no P2P transfer and therefore
                # uses original xq row indices directly.
                job["tk_compute_owner"] = pair_tok_owner.to(
                    torch.int32
                )

            jobs.append(job)

        # Finish route tensors before remote devices consume them.
        ev_route = torch.cuda.Event()
        ev_route.record(torch.cuda.current_stream(owner))

        # --------------------------------------------------------------
        # Launch expert work.
        #
        # For compact remote jobs, create only routed xq rows on owner,
        # copy them to the destination, wait only for that copy, release
        # the temporary source, then launch expert compute.
        #
        # This is deliberately conservative about source lifetime.  GPU N
        # can compute while the next shard's compact input is prepared.
        # --------------------------------------------------------------
        for job in jobs:
            sh = job["sh"]
            d = job["device"]
            m = job["m"]
            stream = job["stream"]

            stream.wait_event(ev_route)

            if d == owner:
                xs = xq

            elif compact_xq:
                # Gather only rows needed by this expert shard.
                xs_src = xq.index_select(
                    0,
                    job["uniq_tok_owner"],
                )

                ev_gather = torch.cuda.Event()
                ev_gather.record(torch.cuda.current_stream(owner))
                stream.wait_event(ev_gather)

                with torch.cuda.device(d), torch.cuda.stream(stream):
                    xs = xs_src.to(
                        d,
                        non_blocking=True,
                    )
                xs_src.record_stream(stream)
                del xs_src

            else:
                with torch.cuda.device(d), torch.cuda.stream(stream):
                    xs = xq.to(
                        d,
                        non_blocking=True,
                    )

            with torch.cuda.device(d), torch.cuda.stream(stream):
                le = job["le_owner"].to(
                    d,
                    dtype=torch.int32,
                    non_blocking=True,
                )

                tk_compute = job["tk_compute_owner"].to(
                    d,
                    dtype=torch.int32,
                    non_blocking=True,
                )

                wt = job["wt_owner"].to(
                    d,
                    non_blocking=True,
                ).contiguous()

                local_rows = torch.arange(
                    m,
                    dtype=torch.int32,
                    device=d,
                )

                ones = torch.ones(
                    m,
                    device=d,
                )

                p1 = GroupedPairs(
                    le,
                    tk_compute,
                    local_rows,
                    ones,
                    64,
                )

                gu = grouped_fp4_gemm(
                    xs,
                    sh["w13"],
                    sh["s13"],
                    p1,
                    m,
                )

                hq = swiglu_quant(
                    gu,
                    wt,
                    self.inter,
                    self.swiglu_limit,
                )

                p2 = GroupedPairs(
                    le,
                    local_rows,
                    local_rows,
                    ones,
                    64,
                )

                ys = grouped_fp4_gemm(
                    hq,
                    sh["w2"],
                    sh["s2"],
                    p2,
                    m,
                )

                job["ys_remote"] = ys
                job["xs_remote"] = xs

                del gu
                del hq

        # --------------------------------------------------------------
        # Low-memory gather: one shard at a time.
        # --------------------------------------------------------------
        for job in jobs:
            stream = job["stream"]
            ev_done = torch.cuda.Event()
            ev_done.record(stream)
            torch.cuda.current_stream(owner).wait_event(ev_done)

            ys = job["ys_remote"]

            if ys.device == owner:
                ys_owner = ys
            else:
                ys_owner = ys.to(owner, non_blocking=True)
                ys.record_stream(torch.cuda.current_stream(owner))

            y.index_add_(
                0,
                job["pair_tok_owner"],
                ys_owner,
            )

            del ys_owner
            del job["ys_remote"]
            del job["xs_remote"]

        # One representative routing/transfer statistic is enough.
        # Avoid log spam during 1-token prefix replay.
        # Print EP transfer statistics only for real multi-token prefill.
        if (
            self.layer_id == 0
            and remote_full_rows
            and n_tok >= 64
        ):
            ratio = remote_sent_rows / remote_full_rows
            print(
                f"[ep-xq] layer=0 "
                f"compact={int(compact_xq)} "
                f"remote_rows={remote_sent_rows:,}/"
                f"{remote_full_rows:,} "
                f"ratio={ratio:.3f} "
                f"saved={(1.0-ratio)*100:.1f}%",
                flush=True,
            )

        return y

    def _pair_tables(self, n_tok: int, device):
        """Constant index tensors for a dispatch of n_tok tokens (cached: no per-step allocations)."""
        key = (n_tok, device)
        t = self._cache.get(key)
        if t is None:
            n_pairs = n_tok * self.topk
            tok = torch.arange(n_tok, device=device, dtype=torch.int32).repeat_interleave(self.topk)
            pair_rows = torch.arange(n_pairs, device=device, dtype=torch.int32)
            ones = torch.ones(n_pairs, device=device)
            t = self._cache[key] = (tok, pair_rows, ones)
        return t

    # ---- expert offload: host -> GPU staging (shared per device)
    _stage: dict = {}

    def _staging(self, n: int, kind: str):
        """GPU buffers for `n` experts (and pinned host buffers for gathered decode experts)."""
        key = (self.device, kind, n)
        b = MoE._stage.get(key)
        if b is None:
            E, n13, kh = self.w13.shape
            dim, ih = self.w2.shape[1], self.w2.shape[2]
            g = lambda *shape: torch.empty(*shape, dtype=torch.uint8, device=self.device)
            b = {"w13": g(n, n13, kh), "s13": g(n, n13, kh * 2 // 32), "w2": g(n, dim, ih), "s2": g(n, dim, ih * 2 // 32)}
            MoE._stage[key] = b
        return b

    def _decode_offload(self, xq, indices, weights, n_tok, n_pairs, tok, pair_rows, ones):
        """Gather the selected experts from host RAM into GPU staging buffers, then the normal GEMV."""
        b = self._staging(n_pairs, "decode")
        eids = indices.flatten().tolist()  # one sync per layer
        # DMA each selected expert straight from the pinned host tensors (no CPU-side gather)
        for i, e in enumerate(eids):
            for gk, src in (("w13", self.w13), ("s13", self.s13), ("w2", self.w2), ("s2", self.s2)):
                b[gk][i].copy_(src[e], non_blocking=True)
        local = torch.arange(n_pairs, device=xq.device, dtype=torch.int32)  # staged experts are in pair order
        gu = cukern.fp4_gemv_pairs(xq, b["w13"][:n_pairs], b["s13"][:n_pairs], tok.to(torch.int32), local, ones, n_pairs)
        hq = swiglu_quant(gu, weights.flatten().float().contiguous(), self.inter, self.swiglu_limit)
        y2 = cukern.fp4_gemv_pairs(hq, b["w2"][:n_pairs], b["s2"][:n_pairs], pair_rows.to(torch.int32), local, ones, n_pairs)
        return y2.view(n_tok, self.topk, self.dim).sum(dim=1)

    def split_hot(self, ids: list[int], wts: list[float]):
        """(gpu slot ids [6], gpu weights [6], cold ids, cold weights): cold experts map to the zero dummy slot."""
        if not self.hot:
            return None, None, ids, wts
        slot, dummy = self.hot["slot"], self.hot["dummy"]
        gs, gw, cid, cw = [], [], [], []
        for e, w in zip(ids, wts):
            k = slot.get(e)
            if k is None:
                gs.append(dummy); gw.append(0.0); cid.append(e); cw.append(w)
            else:
                gs.append(k); gw.append(w)
        return gs, gw, cid, cw

    def hot_experts_gpu(self, xq, slot_ids, slot_w):
        """GPU GEMV over the hot slots (static shapes; cold pairs hit the dummy slot with weight 0)."""
        n = slot_ids.numel()
        hot = self.hot
        local = torch.arange(n, device=xq.device, dtype=torch.int32)
        tok0 = torch.zeros(n, device=xq.device, dtype=torch.int32)
        ones = torch.ones(n, device=xq.device)
        gu = cukern.fp4_gemv_pairs(xq.view(1, -1), hot["w13"], hot["s13"], tok0, slot_ids, ones, n)
        hq = swiglu_quant(gu, slot_w, self.inter, self.swiglu_limit)
        y2 = cukern.fp4_gemv_pairs(hq, hot["w2"], hot["s2"], local, slot_ids, ones, n)
        return y2.sum(dim=0, keepdim=True)

    def _decode_cpu(self, xq, indices, weights):
        """Selected experts on the CPU (cold) and on the GPU (hot cache); only x/y cross PCIe."""
        self.x_host.copy_(xq.view(-1))  # sync D2H
        ids = indices.flatten().tolist()
        wts = weights.flatten().tolist()
        gs, gw, cid, cw = self.split_hot(ids, wts)
        y_gpu = None
        if gs is not None:
            y_gpu = self.hot_experts_gpu(xq, torch.tensor(gs, device=xq.device, dtype=torch.int32), torch.tensor(gw, device=xq.device))
        if cid:
            y = self.host.forward(self.x_host, cid, cw, float(self.swiglu_limit))
            self.y_host.copy_(y)
            y_c = self.y_host.to(xq.device, non_blocking=True).view(1, self.dim)
        else:
            y_c = torch.zeros(1, self.dim, device=xq.device)
        return y_c if y_gpu is None else y_c + y_gpu

    def _prefill_offload(self, xq, indices, weights, n_tok, n_pairs, tok, pair_rows, ones, chunk: int = 64):
        """Stream all experts through the GPU in chunks; each pair is computed in the chunk of its expert."""
        E = self.w13.shape[0]
        b = self._staging(chunk, "prefill")
        eid = indices.flatten()
        gu = torch.zeros(n_pairs, 2 * self.inter, device=xq.device, dtype=torch.float32)
        for c0 in range(0, E, chunk):
            n = min(chunk, E - c0)
            for gk, src in (("w13", self.w13), ("s13", self.s13)):
                b[gk][:n].copy_(src[c0 : c0 + n], non_blocking=True)
            sel = ((eid >= c0) & (eid < c0 + n)).nonzero().flatten()
            if sel.numel() == 0:
                continue
            p1 = GroupedPairs((eid[sel] - c0).to(torch.int32), tok[sel], pair_rows[sel], ones[sel], 64)
            grouped_fp4_gemm(xq, b["w13"][:n], b["s13"][:n], p1, n_pairs, out=gu)
        hq = swiglu_quant(gu, weights.flatten().float().contiguous(), self.inter, self.swiglu_limit)
        y = torch.zeros(n_tok, self.dim, device=xq.device, dtype=torch.float32)
        for c0 in range(0, E, chunk):
            n = min(chunk, E - c0)
            for gk, src in (("w2", self.w2), ("s2", self.s2)):
                b[gk][:n].copy_(src[c0 : c0 + n], non_blocking=True)
            sel = ((eid >= c0) & (eid < c0 + n)).nonzero().flatten()
            if sel.numel() == 0:
                continue
            p2 = GroupedPairs((eid[sel] - c0).to(torch.int32), pair_rows[sel], tok[sel], ones[sel], 64)
            grouped_fp4_gemm(hq, b["w2"][:n], b["s2"][:n], p2, n_tok, out=y)
        return y

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.reshape(-1, self.dim)
        n_tok = x.size(0)
        weights, indices = self.gate(x)
        if ROUTE_STATS is not None:  # expert usage histogram per layer (for the hot-expert cache design)
            ROUTE_STATS.setdefault(self.layer_id, torch.zeros(self.n_experts, dtype=torch.int64)).add_(
                torch.bincount(indices.flatten().cpu(), minlength=self.n_experts))
        # routed experts: pairs (token, expert)
        n_pairs = n_tok * self.topk
        tok, pair_rows, ones = self._pair_tables(n_tok, x.device)
        xq = fake_quant_fp8(x, 32)
        eid = indices.flatten().to(torch.int32)
        if self.ep:
            y = self._forward_ep(xq, eid, tok, weights, n_tok, n_pairs)
        elif self.offload:
            if n_tok == 1 and self.host is not None:
                y = self._decode_cpu(xq, indices, weights)
            elif n_tok <= 16:
                y = self._decode_offload(xq, indices, weights, n_tok, n_pairs, tok, pair_rows, ones)
            else:
                y = self._prefill_offload(xq, indices, weights, n_tok, n_pairs, tok, pair_rows, ones)
        elif n_tok <= 16:
            # decode: bandwidth-bound CUDA GEMV, one output row per (token, expert) pair, no atomics
            gu = cukern.fp4_gemv_pairs(xq, self.w13, self.s13, tok.to(torch.int32), eid, ones, n_pairs)
            hq = swiglu_quant(gu, weights.flatten().float().contiguous(), self.inter, self.swiglu_limit)
            y2 = cukern.fp4_gemv_pairs(hq, self.w2, self.s2, pair_rows.to(torch.int32), eid, ones, n_pairs)
            y = y2.view(n_tok, self.topk, self.dim).sum(dim=1)
        else:
            block_m = 64

            # ---------------------------------------------------------------
            # Memory-bounded MoE prefill
            #
            # Do the complete expert path per token chunk:
            #
            #   GEMM1 -> SwiGLU -> GEMM2 -> expert reduction
            #
            # and write only the reduced [tokens, dim] result into y.
            #
            # This deliberately avoids grouped_fp4_gemm_chunked(), whose
            # final torch.cat() rebuilt the full multi-GiB pair tensor.
            # ---------------------------------------------------------------
            moe_chunk = int(
                os.environ.get("DSV41_MOE_PREFILL_CHUNK", "4096")
            )
            moe_chunk = max(1, moe_chunk)

            if n_tok <= moe_chunk:
                p1 = GroupedPairs(
                    eid,
                    tok,
                    pair_rows,
                    ones,
                    block_m,
                    n_experts=self.n_routed_experts if hasattr(self, "n_routed_experts") else 384,
                )

                gu = grouped_fp4_gemm(
                    xq,
                    self.w13,
                    self.s13,
                    p1,
                    n_pairs,
                )

                hq = swiglu_quant(
                    gu,
                    weights.flatten().float().contiguous(),
                    self.inter,
                    self.swiglu_limit,
                )

                p2 = p1.make_p2(
                    pair_rows,
                    pair_rows,
                    ones,
                )

                y = grouped_fp4_gemm(
                    hq,
                    self.w2,
                    self.s2,
                    p2,
                    n_pairs,
                ).view(
                    n_tok,
                    self.topk,
                    self.dim,
                ).sum(dim=1)

            else:
                if os.environ.get("DSV41_DEBUG", "0") == "1":
                    print(
                        f"[moe-prefill] layer={self.layer_id} "
                        f"tokens={n_tok:,} chunk={moe_chunk:,}",
                        flush=True,
                    )

                # Only the final reduced result persists across chunks.
                # This is far smaller than [n_pairs, 2*inter].
                y = torch.empty(
                    (n_tok, self.dim),
                    dtype=torch.float32,
                    device=x.device,
                )

                for t0 in range(0, n_tok, moe_chunk):
                    t1 = min(t0 + moe_chunk, n_tok)
                    nc = t1 - t0
                    npairs = nc * self.topk

                    # Slice token-local inputs.
                    xc = xq[t0:t1]
                    wc = weights[t0:t1]
                    ic = indices[t0:t1]

                    # Build pair tables for THIS chunk only.
                    ctok, cpair_rows, cones = self._pair_tables(
                        nc,
                        x.device,
                    )

                    ceid = ic.flatten().to(
                        device=x.device,
                        dtype=torch.int32,
                        non_blocking=True,
                    ).contiguous()

                    ctok = ctok.to(
                        device=x.device,
                        dtype=torch.int32,
                        non_blocking=True,
                    ).contiguous()

                    cpair_rows = cpair_rows.to(
                        device=x.device,
                        dtype=torch.int32,
                        non_blocking=True,
                    ).contiguous()

                    cones = cones.to(
                        device=x.device,
                        dtype=torch.float32,
                        non_blocking=True,
                    ).contiguous()

                    p1 = GroupedPairs(
                        ceid,
                        ctok,
                        cpair_rows,
                        cones,
                        block_m,
                        n_experts=self.n_routed_experts if hasattr(self, "n_routed_experts") else 384,
                    )

                    # Temporary size now depends on nc, not total n_tok.
                    gu = grouped_fp4_gemm(
                        xc,
                        self.w13,
                        self.s13,
                        p1,
                        npairs,
                    )

                    hq = swiglu_quant(
                        gu,
                        wc.flatten().float().contiguous(),
                        self.inter,
                        self.swiglu_limit,
                    )

                    p2 = p1.make_p2(
                        cpair_rows,
                        cpair_rows,
                        cones,
                    )

                    yc = grouped_fp4_gemm(
                        hq,
                        self.w2,
                        self.s2,
                        p2,
                        npairs,
                    ).view(
                        nc,
                        self.topk,
                        self.dim,
                    ).sum(dim=1)

                    y[t0:t1].copy_(yc)

                    # Release pair-sized intermediates immediately.
                    del gu
                    del hq
                    del yc
                    del p1
                    del p2
                    del ceid
                    del ctok
                    del cpair_rows
                    del cones

        y += self.shared_expert(x)
        return y.to(x.dtype).view(shape)


def _hc_mix_proj(x, fn, eps):
    """
    HC mix projection.

    Long prefill used to materialize the complete flattened activation
    in fp32:

        [B, S, HC*D] fp32

    which becomes several GiB for 30K+ token prompts.

    The operation is token-local, so chunking the sequence dimension is
    mathematically identical while greatly reducing peak VRAM.
    """
    chunk = int(
        os.environ.get(
            "DSV41_HC_PREFILL_CHUNK",
            "2048",
        )
    )

    # Decode / short prompt: preserve the original fast path.
    if (
        chunk <= 0
        or x.ndim < 3
        or x.shape[1] <= chunk
    ):
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(
            xf.square().mean(-1, keepdim=True) + eps
        )
        return F.linear(xf, fn) * rsqrt

    B = x.shape[0]
    S = x.shape[1]
    out_features = fn.shape[0]

    # F.linear(float32, ...) produces float32 here.
    # Allocate only the comparatively small final projection once.
    y = torch.empty(
        (B, S, out_features),
        dtype=torch.float32,
        device=x.device,
    )

    if os.environ.get(
        "DSV41_DEBUG_HC_PREFILL",
        "0",
    ) == "1":
        print(
            f"[hc-prefill] "
            f"tokens={S:,} "
            f"chunk={chunk:,} "
            f"in_flat={x.flatten(2).shape[-1]:,} "
            f"out={out_features:,} "
            f"device={x.device}",
            flush=True,
        )

    for s0 in range(0, S, chunk):
        s1 = min(
            s0 + chunk,
            S,
        )

        # Only this slice is promoted to fp32.
        xf = x[:, s0:s1].flatten(2).float()

        rsqrt = torch.rsqrt(
            xf.square().mean(
                -1,
                keepdim=True,
            )
            + eps
        )

        yc = F.linear(
            xf,
            fn,
        )

        yc.mul_(rsqrt)

        y[:, s0:s1].copy_(yc)

        # Drop the large fp32 temporaries immediately.
        del xf
        del rsqrt
        del yc

    return y


def _hc_pre(x, pre_mix):
    return _fused_hc_pre(x, pre_mix)


def _hc_post(x, residual, post, comb):
    return _fused_hc_post(x, residual, post, comb)


def _swiglu(gate, up, limit: float):
    gate, up = gate.float(), up.float()
    if limit > 0:
        up = torch.clamp(up, min=-limit, max=limit)
        gate = torch.clamp(gate, max=limit)
    return F.silu(gate) * up


# --------------------------------------------------------------------------- block
class Block:
    def __init__(self, args: Args, layer_id: int, w: dict, device, shared: SharedAttn):
        self.layer_id = layer_id
        self.device = device
        self.eps = args.norm_eps
        self.hc = args.hc_mult
        self.sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        self.attn = Attention(args, layer_id, w, device, shared)
        self.ffn = MoE(args, layer_id, w, device)
        self.attn_norm_w = w["attn_norm.weight"]
        self.ffn_norm_w = w["ffn_norm.weight"]
        self.hc_attn = (w["hc_attn_fn"].float(), w["hc_attn_scale"].float(), w["hc_attn_base"].float())
        self.hc_ffn = (w["hc_ffn_fn"].float(), w["hc_ffn_scale"].float(), w["hc_ffn_base"].float())
        self.engram = None

    def hc_mixes(self, x, fn, scale, base):
        mixes = _hc_mix_proj(x, fn, self.eps)
        return hc_split_sinkhorn(mixes, scale, base, self.hc, self.sinkhorn_iters, self.hc_eps)

    def hc_pre(self, x, pre_mix):
        return _hc_pre(x, pre_mix)

    def hc_post(self, x, residual, post, comb):
        return _hc_post(x, residual, post, comb)

    def __call__(self, x: torch.Tensor, start_pos: int, pre_mix: torch.Tensor):
        t = time.perf_counter()
        residual = x
        attn_pre, attn_post, attn_comb = self.hc_mixes(x, *self.hc_attn)
        x = self.hc_pre(x, pre_mix)
        x = rmsnorm(x, self.attn_norm_w, self.eps)
        t = _tick("hc+norm", t)
        x = self.attn(x, start_pos)
        t = _tick("attention", t)
        x = self.hc_post(x, residual, attn_post, attn_comb)
        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, *self.hc_ffn)
        x = self.hc_pre(x, attn_pre)
        x = rmsnorm(x, self.ffn_norm_w, self.eps)
        t = _tick("hc+norm", t)
        x = self.ffn(x)
        t = _tick("moe", t)
        x = self.hc_post(x, residual, ffn_post, ffn_comb)
        _tick("hc+norm", t)
        return x, ffn_pre


# --------------------------------------------------------------------------- transformer
class Transformer:
    def __init__(self, args: Args):
        self.args = args
        self.blocks: list[Block] = []
        self.shared = SharedAttn()
        self.embed = None  # bf16 [vocab, dim] on device of layer 0
        self.head = None  # bf16 [vocab, dim] on device of last layer
        self.norm_w = None
        self.engram_hash = None
        self.hc = args.hc_mult

    def reset_cache(self):
        """Zero all KV and index caches to avoid cross-request contamination."""
        for blk in self.blocks:
            if hasattr(blk.attn, "window_kv_cache") and blk.attn.window_kv_cache is not None:
                blk.attn.window_kv_cache.zero_()
            c = blk.attn.compressor
            if c is not None and getattr(c, "ratio", 1) > 1:
                if hasattr(c, "kv_state"):
                    c.kv_state.zero_()
                if hasattr(c, "score_state"):
                    c.score_state.fill_(-torch.inf)
                if hasattr(c, "kv_ring"):
                    c.kv_ring.zero_()
                if hasattr(c, "score_ring"):
                    c.score_ring.fill_(-torch.inf)
        if hasattr(self, "shared"):
            for cache in self.shared.compress_kv.values():
                cache.zero_()
            for cache in self.shared.index_k.values():
                cache.zero_()
            self.shared._topk_idxs_map.clear()
            self.shared._candidates_map.clear()
            self.shared._current_chunk_idx = 0

    def _get_stages(self) -> list[dict]:
        """Group contiguous blocks on the same device into pipeline stages."""
        if hasattr(self, "_stages_cache") and self._stages_cache is not None:
            return self._stages_cache
        stages = []
        if not self.blocks:
            return stages
        current_dev = self.blocks[0].device
        current_blocks = []
        for blk in self.blocks:
            if blk.device == current_dev:
                current_blocks.append(blk)
            else:
                stages.append({
                    "device": current_dev,
                    "blocks": current_blocks,
                })
                current_dev = blk.device
                current_blocks = [blk]
        if current_blocks:
            stages.append({
                "device": current_dev,
                "blocks": current_blocks,
            })
        self._stages_cache = stages
        return stages

    @torch.inference_mode()
    def _forward_sequential(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """input_ids [b, s] (long) -> logits for the last position [b, vocab] (fp32)."""
        dev0 = self.blocks[0].device
        input_ids = input_ids.to(dev0)
        hashes = self.engram_hash(input_ids, start_pos) if self.engram_hash is not None else None
        h = F.embedding(input_ids, self.embed)
        h = h.unsqueeze(2).repeat(1, 1, self.hc, 1)
        pre_mix = h.new_zeros(h.size(0), h.size(1), self.hc, dtype=torch.float32)
        pre_mix[:, :, 0] = 1.0
        targets = set(getattr(self, "collect_main_hidden", ()))
        main_hiddens = []
        for blk in self.blocks:
            t = time.perf_counter()
            if h.device != blk.device:
                h = h.to(blk.device, non_blocking=True)
                pre_mix = pre_mix.to(blk.device, non_blocking=True)
            t = _tick("transfer", t)
            if blk.engram is not None:
                h = blk.engram(h, hashes[:, :, blk.engram.layer_hash_index, :])
                _tick("engram", t)
            if blk.layer_id in targets:  # DSpark reads the attention input of its target layers
                main_hiddens.append(h.mean(dim=2))
            h, pre_mix = blk(h, start_pos, pre_mix)
        if main_hiddens:
            # MTP/DSpark only consumes the recent main attention window.
            # Keeping every target-layer hidden row for a 100K+ prompt
            # creates several GiB of avoidable GPU state and can surface
            # as an asynchronous illegal-memory error on the next CUDA op.
            keep = int(self.args.cfg.get("window_size", 0))
            if keep > 0 and input_ids.shape[1] > keep:
                main_hiddens = [m[:, -keep:] for m in main_hiddens]
            self.main_hidden = torch.cat([m.to(main_hiddens[-1].device) for m in main_hiddens], dim=-1)  # [b, s, 3*dim]
        h = self.blocks[-1].hc_pre(h, pre_mix)[:, -1]
        h = rmsnorm(h, self.norm_w, self.args.norm_eps)
        return F.linear(h, self.head).float()

    @torch.inference_mode()
    def forward_pipelined(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        chunk_size: int = 2048,
        progress_interval: int = 0,
    ) -> torch.Tensor:
        """Pipelined prefill across multi-GPU stages with chunked prompt tokens.

        Divides prompt into M chunks and executes a pipelined FIFO schedule across K stages.
        Each stage runs on its dedicated CUDA stream with CUDA event synchronization.
        """
        stages = self._get_stages()
        K = len(stages)
        dev0 = stages[0]["device"]
        last_dev = stages[-1]["device"]
        input_ids = input_ids.to(dev0)
        B, S = input_ids.shape

        if not hasattr(self, "_stage_streams") or len(self._stage_streams) != K:
            self._stage_streams = [torch.cuda.Stream(device=s["device"]) for s in stages]

        chunk_size = max(128, int(chunk_size))
        M = (S + chunk_size - 1) // chunk_size

        # Pre-ensure capacity for all KV and index owners upfront so no reallocations happen mid-pipeline
        for owner in self.shared.cache_max_rows:
            ratio = int(self.args.compress_ratios[owner])
            needed = (start_pos + S + ratio - 1) // ratio
            self.shared._ensure_capacity(self.shared.compress_kv, owner, needed, "compress_kv")
            self.shared._ensure_capacity(self.shared.index_k, owner, needed, "index_k")

        hashes = self.engram_hash(input_ids, start_pos) if self.engram_hash is not None else None

        targets = set(getattr(self, "collect_main_hidden", ()))
        chunk_main_hiddens: dict[int, list[torch.Tensor]] = {lid: [] for lid in targets}

        chunk_activations: list[list[tuple[torch.Tensor, torch.Tensor] | None]] = [
            [None for _ in range(M)] for _ in range(K)
        ]
        ev_stage_done: list[list[torch.cuda.Event]] = [
            [torch.cuda.Event() for _ in range(M)] for _ in range(K)
        ]

        T = M + K - 1
        final_logits = None

        ev_start = torch.cuda.Event()
        ev_start.record(torch.cuda.current_stream(dev0))
        for st in self._stage_streams:
            st.wait_event(ev_start)

        t_pipe_start = time.perf_counter()
        last_progress_time = t_pipe_start
        last_progress_tokens = 0

        for t in range(T):
            for k in range(K - 1, -1, -1):
                m = t - k
                if not (0 <= m < M):
                    continue

                c0 = m * chunk_size
                c1 = min(c0 + chunk_size, S)
                p_m = start_pos + c0
                stage = stages[k]
                dev_k = stage["device"]
                stream_k = self._stage_streams[k]
                stage_blocks = stage["blocks"]

                with torch.cuda.device(dev_k), torch.cuda.stream(stream_k):
                    if k == 0:
                        chunk_ids = input_ids[:, c0:c1].to(dev_k, non_blocking=True)
                        h = F.embedding(chunk_ids, self.embed)
                        h = h.unsqueeze(2).repeat(1, 1, self.hc, 1)
                        pre_mix = h.new_zeros(h.size(0), h.size(1), self.hc, dtype=torch.float32)
                        pre_mix[:, :, 0] = 1.0
                    else:
                        stream_k.wait_event(ev_stage_done[k - 1][m])
                        h_prev, pre_mix_prev = chunk_activations[k - 1][m]
                        h = h_prev.to(dev_k, non_blocking=True)
                        pre_mix = pre_mix_prev.to(dev_k, non_blocking=True)
                        h_prev.record_stream(stream_k)
                        pre_mix_prev.record_stream(stream_k)
                        chunk_activations[k - 1][m] = None

                    self.shared._current_chunk_idx = m

                    chunk_hashes = hashes[:, c0:c1] if hashes is not None else None

                    for blk in stage_blocks:
                        if blk.engram is not None and chunk_hashes is not None:
                            blk_hash = chunk_hashes[:, :, blk.engram.layer_hash_index, :]
                            if blk_hash.device != dev_k:
                                blk_hash = blk_hash.to(dev_k, non_blocking=True)
                            h = blk.engram(h, blk_hash)
                        if blk.layer_id in targets:
                            chunk_main_hiddens[blk.layer_id].append(h.mean(dim=2))
                        h, pre_mix = blk(h, p_m, pre_mix)

                    if k < K - 1:
                        chunk_activations[k][m] = (h, pre_mix)
                    else:
                        if m == M - 1:
                            h_last = self.blocks[-1].hc_pre(h, pre_mix)[:, -1]
                            h_last = rmsnorm(h_last, self.norm_w, self.args.norm_eps)
                            final_logits = F.linear(h_last, self.head).float()
                        self.shared.cleanup_chunk(m)

                    ev_stage_done[k][m].record(stream_k)

                    # Periodic progress logging for long contexts
                    if progress_interval > 0 and k == K - 1 and ((m + 1) % progress_interval == 0 or m == M - 1):
                        ev_stage_done[k][m].synchronize()
                        now = time.perf_counter()
                        cur_tokens = c1
                        dt_total = now - t_pipe_start
                        dt_chunk = now - last_progress_time
                        tok_chunk = cur_tokens - last_progress_tokens
                        inst_spd = tok_chunk / max(dt_chunk, 1e-6)
                        avg_spd = cur_tokens / max(dt_total, 1e-6)
                        last_progress_time = now
                        last_progress_tokens = cur_tokens
                        mem_info = " / ".join(
                            f"{(torch.cuda.mem_get_info(s['device'])[1] - torch.cuda.mem_get_info(s['device'])[0])/2**30:.1f}"
                            for s in stages
                        )
                        print(
                            f"[1M-prefill] chunk {m + 1:3d}/{M} ({cur_tokens:8,d}/{S:,} tok, {cur_tokens/S*100:4.1f}%) | "
                            f"time: {dt_total:6.1f}s | "
                            f"speed: {inst_spd:5.1f} tok/s (avg: {avg_spd:5.1f} tok/s) | "
                            f"VRAM used: [{mem_info}] GiB",
                            flush=True,
                        )

        torch.cuda.current_stream(last_dev).wait_stream(self._stage_streams[-1])
        self.shared._current_chunk_idx = 0

        if targets:
            keep = int(self.args.cfg.get("window_size", 0))
            layer_hiddens = []
            for lid in sorted(targets):
                full_layer_h = torch.cat(chunk_main_hiddens[lid], dim=1)
                if keep > 0 and full_layer_h.shape[1] > keep:
                    full_layer_h = full_layer_h[:, -keep:]
                layer_hiddens.append(full_layer_h.to(last_dev, non_blocking=True))
            self.main_hidden = torch.cat(layer_hiddens, dim=-1)

        return final_logits

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """input_ids [b, s] (long) -> logits for the last position [b, vocab] (fp32)."""
        stages = self._get_stages()
        seqlen = input_ids.shape[1]
        chunk_size = int(os.environ.get("DSV41_PIPELINE_CHUNK_SIZE", "2048"))
        use_pipeline = (
            os.environ.get("DSV41_PIPELINE", "1") != "0"
            and len(stages) > 1
            and seqlen >= chunk_size * 2
        )
        if use_pipeline:
            return self.forward_pipelined(input_ids, start_pos=start_pos, chunk_size=chunk_size)
        return self._forward_sequential(input_ids, start_pos=start_pos)
