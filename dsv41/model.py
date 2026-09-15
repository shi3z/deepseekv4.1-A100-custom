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
from .moe_kernels import GroupedPairs, grouped_fp4_gemm
from . import cukern
from .quant import maybe_compile


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
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int, device):
    if start_pos == 0:
        end = torch.arange(seqlen).unsqueeze(1)
        idxs = (end - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        idxs = torch.where(idxs > end, -1, idxs)
    else:
        oldest = start_pos % window_size + 1
        idxs = torch.cat([torch.arange(oldest, window_size), torch.arange(oldest)])
        idxs = torch.where(idxs > start_pos, -1, idxs)
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous().to(device)


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
    """Caches produced by source layers and read by later layers, possibly on other GPUs. Each owner's
    cache is mirrored on every device; the owner pushes only the new rows to all mirrors. Consumers
    read the cache of the most recent owner before them (set by that owner when it runs)."""

    def __init__(self):
        self.compress_kv: dict[tuple[int, torch.device], torch.Tensor] = {}
        self.index_k: dict[tuple[int, torch.device], torch.Tensor] = {}
        self.kv_owner: int = -1
        self.index_owner: int = -1
        self.topk_idxs: torch.Tensor | None = None
        self.candidates: torch.Tensor | None = None

    def write_compress_kv(self, owner: int, bsz: int, pos: int, rows: torch.Tensor):
        self.kv_owner = owner
        for (o, dev), cache in self.compress_kv.items():
            if o == owner:
                cache[:bsz, pos : pos + rows.size(1)] = rows.to(dev, non_blocking=True)

    def write_index_k(self, owner: int, bsz: int, pos: int, rows: torch.Tensor):
        self.index_owner = owner
        for (o, dev), cache in self.index_k.items():
            if o == owner:
                cache[:bsz, pos : pos + rows.size(1)] = rows.to(dev, non_blocking=True)


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
            should = (start_pos + 1) % ratio == 0
            slot = start_pos % ratio
            self.kv_state[:bsz, slot] = kv.squeeze(1)
            self.score_state[:bsz, slot] = score.squeeze(1)
            if should:
                kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should:
            return None
        return rmsnorm(kv.to(dtype), self.norm_w, self.eps)


class Indexer:
    def __init__(self, args: Args, layer_id: int, w: dict, device, shared: SharedAttn):
        self.shared = shared
        self.device = device
        self.layer_id = layer_id
        self.owns_k = layer_id in args.kv_source_layers
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
                rope_(k, rd, self.cos, self.sin, 0, pos_stride=ratio)
            else:
                rope_(k, rd, self.cos, self.sin, start_pos + 1 - ratio)

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
            (self.shared.index_owner, self.device)
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
        if start_pos == 0 and seqlen > 128:
            QUERY_CHUNK = 128

            key_len = index_k.size(1)
            key_pos = torch.arange(
                key_len,
                device=x.device,
            )

            # One conversion per layer instead of per query chunk.
            index_k_f = index_k.float()

            idx_chunks = []
            candidate_chunks = [] if self.is_candidate_source else None

            topk = min(self.topk, end_pos // ratio)

            for q0 in range(0, seqlen, QUERY_CHUNK):
                q1 = min(q0 + QUERY_CHUNK, seqlen)

                qc = q[:, q0:q1].float()
                wc = weights[:, q0:q1].float()

                # [B,C,H,D] x [B,T,D] => [B,C,H,T]
                score = torch.einsum(
                    "bshd,btd->bsht",
                    qc,
                    index_k_f,
                )

                # Avoid another full [B,C,H,T] temporary.
                score.relu_()
                score.mul_(wc.unsqueeze(-1))

                # Reduce heads immediately.
                # [B,C,H,T] -> [B,C,T]
                score = score.sum(dim=2)

                # Preserve original causal/compression visibility.
                compress_lens = (
                    torch.arange(
                        q0 + 1,
                        q1 + 1,
                        device=x.device,
                    ) // ratio
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
                    seqlen // ratio,
                    device=x.device,
                ) >= compress_lens,
                -torch.inf,
            )
        else:
            compress_lens = end_pos // ratio

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
        bsz, seqlen, _ = x.size()
        win = self.window
        kv = rmsnorm(linear_fp8(x, self.wkv), self.kv_norm_w, self.eps)
        rope_(kv, self.rope_head_dim, self.cos, self.sin, start_pos)
        kv = fake_quant_fp8(kv, 32)
        if start_pos == 0:
            if seqlen <= win:
                self.window_kv_cache[:bsz, :seqlen] = kv
            else:
                cut = seqlen % win
                self.window_kv_cache[:bsz, cut:win], self.window_kv_cache[:bsz, :cut] = kv[:, -win:].split([win - cut, cut], dim=1)
            window_kv = kv
        else:
            self.window_kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            window_kv = self.window_kv_cache[:bsz]
        return window_kv, get_window_topk_idxs(win, bsz, seqlen, start_pos, x.device)

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
                rope_(latent, self.rope_head_dim, self.cos, self.sin, 0, pos_stride=ratio)
            else:
                rope_(latent, self.rope_head_dim, self.cos, self.sin, start_pos + 1 - ratio)
            latent = fake_quant_fp4(latent, 16, scale_e4m3=True)
            self.shared.write_compress_kv(self.layer_id, bsz, start_pos // ratio, latent)
        return self.shared.compress_kv[(self.shared.kv_owner, self.device)][:bsz, :compress_len], idxs

    def __call__(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        rd = self.rope_head_dim
        qr = rmsnorm(linear_fp8(x, self.wq_a), self.q_norm_w, self.eps)
        q = linear_fp8(qr, self.wq_b).unflatten(-1, (self.n_heads, self.head_dim))
        rope_(q, rd, self.cos, self.sin, start_pos)
        kv, topk_idxs = self._window_kv(x, freqs_cis, start_pos)
        if self.ratio:
            ckv, cidx = self._compress_kv(x, qr, start_pos, kv.size(1))
            kv = torch.cat([kv, ckv], dim=1)
            topk_idxs = torch.cat([topk_idxs, cidx], dim=-1)
        if seqlen == 1:
            o = sparse_attn_decode(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            o = sparse_attn_prefill(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
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
        """Eager / prefill path with sharded experts: each shard computes its pairs on its own device, the
        per-token sums come back to the owner device (used for prefill; decode uses dsv41/ep.py)."""
        # one output row per (token, expert) pair, filled by the shard that holds the expert, summed in a fixed
        # order per token: deterministic (no atomics across a token's experts)
        yp = torch.zeros(n_pairs, self.dim, device=xq.device, dtype=torch.float32)
        wflat = weights.flatten().float()
        for sh in self.ep:
            sel = ((eid >= sh["start"]) & (eid < sh["start"] + sh["n"])).nonzero().flatten()
            if sel.numel() == 0:
                continue
            d = sh["device"]
            xs = xq.to(d)
            le = (eid[sel] - sh["start"]).to(d)
            tk = tok[sel].to(d)
            wt = wflat[sel].to(d)
            m = sel.numel()
            local_rows = torch.arange(m, device=d, dtype=torch.int32)
            ones = torch.ones(m, device=d)
            block_m = 64
            p1 = GroupedPairs(le, tk, local_rows, ones, block_m)
            gu = grouped_fp4_gemm(xs, sh["w13"], sh["s13"], p1, m)
            hq = swiglu_quant(gu, wt.contiguous(), self.inter, self.swiglu_limit)
            p2 = GroupedPairs(le, local_rows, local_rows, ones, block_m)
            ys = grouped_fp4_gemm(hq, sh["w2"], sh["s2"], p2, m)
            yp[sel] = ys.to(xq.device)
        return yp.view(n_tok, self.topk, self.dim).sum(dim=1)

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
            p1 = GroupedPairs(eid, tok, pair_rows, ones, block_m)
            gu = grouped_fp4_gemm(xq, self.w13, self.s13, p1, n_pairs)  # fp32 [pairs, 2*inter]
            hq = swiglu_quant(gu, weights.flatten().float().contiguous(), self.inter, self.swiglu_limit)
            # one output row per pair (no atomics across a token's experts), summed in a fixed order: deterministic prefill
            p2 = GroupedPairs(eid, pair_rows, pair_rows, ones, block_m)
            y = grouped_fp4_gemm(hq, self.w2, self.s2, p2, n_pairs).view(n_tok, self.topk, self.dim).sum(dim=1)
        y += self.shared_expert(x)
        return y.to(x.dtype).view(shape)


def _hc_mix_proj(x, fn, eps):
    xf = x.flatten(2).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return F.linear(xf, fn) * rsqrt


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

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
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
            self.main_hidden = torch.cat([m.to(main_hiddens[-1].device) for m in main_hiddens], dim=-1)  # [b, s, 3*dim]
        h = self.blocks[-1].hc_pre(h, pre_mix)[:, -1]
        h = rmsnorm(h, self.norm_w, self.args.norm_eps)
        return F.linear(h, self.head).float()
