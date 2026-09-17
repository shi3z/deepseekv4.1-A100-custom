"""Static-shape decode path (one token per step) with per-GPU CUDA graphs.

Everything that depends on the position is expressed with device tensors and fixed-size buffers:
the window ring index pattern, the compressor's parity, the compressed-KV / index-key cache writes
(a dummy row absorbs the "no new group yet" case), the indexer's top-k over the whole cache with
future positions masked. Host work per step is reduced to: the Engram table gather (CPU), a few tiny
H2D/D2D copies between GPU segments, and one graph launch per GPU."""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F

from . import cukern
from .cukern import fp4_gemm_tc, fp4_gemv_pairs as cukern_fp4
from .fused import fake_quant_fp4, fake_quant_fp8, rmsnorm, rope_dev_, sparse_attn_decode_split as sparse_attn_decode2, swiglu_quant
from .fused2 import gate_topk, hc_mix, hc_post2_, hc_pre_norm_quant, hc_pre_norm_quant2, hc_sinkhorn, kv_write, norm_quant, sattn2
from .model import Attention, Block, Transformer, _hc_post, _hc_pre, linear_fp8, select_candidate_blocks
from .w8 import linear_w, oproj_a

FUSED2 = os.environ.get("DSV41_FUSED2", "1") == "1"
_CPU_DEBUG = os.environ.get("DSV41_CPU_DEBUG") is not None
FP4_TC = os.environ.get("DSV41_FP4_TC", "1") == "1"  # expert GEMM on tensor cores (cuda/fp4_tc.cu) instead of the GEMV
HC_FORK = os.environ.get("DSV41_HC_FORK", "1") == "1"  # hyper-connection mixes + sinkhorn on a side stream (a parallel graph branch)


class DecodeRuntime:
    def __init__(self, model: Transformer, use_graphs: bool = True, max_batch: int | None = None):
        self.m = model
        args = model.args
        self.cfg = args.cfg
        self.win = self.cfg["window_size"]
        self.rd = self.cfg["rope_head_dim"]
        self.topk = self.cfg["index_topk"]
        self.use_graphs = use_graphs
        # contiguous device segments in layer order
        self.segments: list[tuple[torch.device, list[Block]]] = []
        for blk in model.blocks:
            if self.segments and self.segments[-1][0] == blk.device:
                self.segments[-1][1].append(blk)
            else:
                self.segments.append((blk.device, [blk]))
        self.devices = [d for d, _ in self.segments]
        hc, dim = self.cfg["hc_mult"], self.cfg["dim"]
        B = self.B = int(max_batch) if max_batch is not None else args.max_batch_size  # rows decoded together; row r is a token at position pos[r] of sequence seq[r]
        self.pos = {d: torch.zeros(B, dtype=torch.int64, device=d) for d in self.devices}
        S = self.S = args.max_seqs or B  # sequence slots (the caches are sized by it)
        self.seq = {d: torch.arange(B, dtype=torch.int64, device=d) % S for d in self.devices}
        self.pmax = {d: torch.zeros(B, dtype=torch.int64, device=d) for d in self.devices}  # newest position written to the row's sequence this step
        self.h_in = {d: torch.zeros(B, 1, hc, dim, dtype=torch.bfloat16, device=d) for d in self.devices}
        self.h_out = {d: torch.zeros(B, 1, hc, dim, dtype=torch.bfloat16, device=d) for d in self.devices}
        self.pre_in = {d: torch.zeros(B, 1, hc, dtype=torch.float32, device=d) for d in self.devices}
        self.pre_out = {d: torch.zeros(B, 1, hc, dtype=torch.float32, device=d) for d in self.devices}
        self.topk_buf = {d: torch.full((B, 1, self.topk), -1, dtype=torch.int32, device=d) for d in self.devices}
        n_cand = args.max_seq_len + 1
        self.cand_buf = {d: torch.zeros(B, 1, n_cand, dtype=torch.bool, device=d) for d in self.devices}
        self.tok = torch.zeros(B, 1, dtype=torch.int64, device=self.devices[0])
        self.logits = torch.zeros(B, self.cfg["vocab_size"], dtype=torch.float32, device=self.devices[-1])
        self.eng_in: dict[int, torch.Tensor] = {}
        for blk in model.blocks:
            if blk.engram is not None:
                cols = blk.engram.wkv.shape[1]
                self.eng_in[blk.layer_id] = torch.zeros(B, 1, cols, dtype=torch.bfloat16, device=blk.device)
        # owners: the row written this step (value + index), to propagate to mirrors on other devices
        self.kv_row: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.ik_row: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for blk in model.blocks:
            if blk.attn.is_kv_source:
                d = blk.device
                self.kv_row[blk.layer_id] = (torch.zeros(B, 1, self.cfg["head_dim"], dtype=torch.bfloat16, device=d), torch.zeros(B, dtype=torch.int64, device=d))
                self.ik_row[blk.layer_id] = (torch.zeros(B, 1, self.cfg["index_head_dim"], dtype=torch.bfloat16, device=d), torch.zeros(B, dtype=torch.int64, device=d))
        self.arange_win = {d: torch.arange(self.win, device=d) for d in self.devices}
        # DSpark: the attention inputs of the target layers (mean over the hc copies), captured inside the graphs
        self.target_layers = list(self.cfg.get("dspark_target_layer_ids", []))
        # routing telemetry (DSV41_ROUTE_LOG=1): the gate result of every layer in persistent buffers
        self.route_log = os.environ.get("DSV41_ROUTE_LOG") == "1"
        topk_e = self.cfg["n_activated_experts"]
        self.route_eid = {b.layer_id: torch.zeros(B, topk_e, dtype=torch.int32, device=b.device) for b in model.blocks} if self.route_log else {}
        self.route_wt = {b.layer_id: torch.zeros(B, topk_e, dtype=torch.float32, device=b.device) for b in model.blocks} if self.route_log else {}
        self.main_hid = {lid: torch.zeros(B, dim, dtype=torch.bfloat16, device=model.blocks[lid].device) for lid in self.target_layers if lid < len(model.blocks)}
        self.graphs: dict[torch.device, torch.cuda.CUDAGraph] = {}
        self.kv_owner = -1
        self.index_owner = -1

    # ------------------------------------------------------------------ attention (static)
    def attention2(self, A: Attention, x: torch.Tensor, xq: torch.Tensor, d: torch.device) -> torch.Tensor:
        """x: rmsnorm output bf16 [B, dim] (unquantized, for the compressor/indexer); xq: its fp8 fake-quantized copy."""
        B = x.shape[0]
        pos, seq, pmax = self.pos[d], self.seq[d], self.pmax[d]
        rd, eps = self.rd, A.eps
        qr = norm_quant(linear_w(xq, A.wq_a), A.q_norm_w, eps)  # q_norm output, already fp8-rounded for wq_b / indexer
        q = linear_w(qr, A.wq_b).view(B, 1, A.n_heads, A.head_dim)
        rope_dev_(q, rd, A.cos, A.sin, pos)
        kv_write(linear_w(xq, A.wkv), A.kv_norm_w, A.cos, A.sin, pos, A.window_kv_cache, rd, eps, seq)
        if A.ratio:
            ratio = A.ratio
            compress_len = torch.div(pos + 1, ratio, rounding_mode="floor")  # [B]
            latent = None
            x3 = x.view(B, 1, -1)
            if A.is_kv_source:
                self.kv_owner = A.layer_id
                latent, should = self.compressor2(A, x3, pos, seq)
                cache = self.m.shared.compress_kv[(A.layer_id, d)]
                row = torch.where(should, compress_len - 1, torch.full_like(compress_len, cache.shape[1] - 1))  # [B]
            if A.is_index_source:
                idxs = self.indexer(A, x3, qr.view(B, 1, -1), latent, pos, compress_len, d, row if A.is_kv_source else None)
                self.topk_buf[d].copy_(idxs)
            else:
                idxs = self.topk_buf[d]
            if latent is not None:
                latent = latent.contiguous()
                rope_dev_(latent, rd, A.cos, A.sin, pos, add=1 - ratio)
                latent = fake_quant_fp4(latent, 16, scale_e4m3=True)
                cache[seq, row] = latent[:, 0]
                val, idx = self.kv_row[A.layer_id]
                val.copy_(latent)
                idx.copy_(row)
            ckv = self.m.shared.compress_kv[(self.kv_owner, d)]
            o = sattn2(q, A.window_kv_cache, ckv, idxs, pos, A.attn_sink, A.cos, A.sin, rd, A.softmax_scale, seq, pmax)
        else:
            o = sattn2(q, A.window_kv_cache, None, None, pos, A.attn_sink, A.cos, A.sin, rd, A.softmax_scale, seq, pmax)
        o = oproj_a(o.view(B, 1, A.n_groups, -1), A.wo_a, A.n_groups, A.o_lora_rank)
        return linear_fp8(o, A.wo_b)

    def attention(self, A: Attention, x: torch.Tensor, d: torch.device) -> torch.Tensor:
        pos = self.pos[d]
        rd, eps = self.rd, A.eps
        qr = rmsnorm(linear_fp8(x, A.wq_a), A.q_norm_w, eps)
        q = linear_fp8(qr, A.wq_b).unflatten(-1, (A.n_heads, A.head_dim))
        rope_dev_(q, rd, A.cos, A.sin, pos)
        # sliding window: write this token's KV into the ring, build the ring index pattern
        kv = rmsnorm(linear_fp8(x, A.wkv), A.kv_norm_w, eps)
        rope_dev_(kv, rd, A.cos, A.sin, pos)
        kv = fake_quant_fp8(kv, 32)
        slot = torch.remainder(pos, self.win)
        A.window_kv_cache.index_copy_(1, slot.view(1), kv)
        widx = torch.remainder(self.arange_win[d] + slot + 1, self.win)
        widx = torch.where(widx > pos, -1, widx).to(torch.int32).view(1, 1, self.win)
        if A.ratio:
            ratio = A.ratio
            compress_len = torch.div(pos + 1, ratio, rounding_mode="floor")
            latent = None
            if A.is_kv_source:
                self.kv_owner = A.layer_id
                latent, should = self.compressor(A, x, pos)
                cache = self.m.shared.compress_kv[(A.layer_id, d)]
                row = torch.where(should, compress_len - 1, torch.full_like(compress_len, cache.shape[1] - 1))
            if A.is_index_source:
                idxs = self.indexer(A, x, qr, latent, pos, compress_len, d, row if A.is_kv_source else None)
                self.topk_buf[d].copy_(idxs)
            else:
                idxs = self.topk_buf[d]
            if latent is not None:
                latent = latent.contiguous()
                rope_dev_(latent, rd, A.cos, A.sin, pos, add=1 - ratio)
                latent = fake_quant_fp4(latent, 16, scale_e4m3=True)
                cache.index_copy_(1, row, latent)
                val, idx = self.kv_row[A.layer_id]
                val.copy_(latent)
                idx.copy_(row)
            ckv = self.m.shared.compress_kv[(self.kv_owner, d)]
            o = sparse_attn_decode2(q, A.window_kv_cache, ckv, A.attn_sink, torch.cat([widx, idxs], dim=-1), A.softmax_scale)
        else:
            o = sparse_attn_decode2(q, A.window_kv_cache, None, A.attn_sink, widx, A.softmax_scale)
        rope_dev_(o, rd, A.cos, A.sin, pos, inverse=True)
        o = oproj_a(o.view(1, 1, A.n_groups, -1), A.wo_a, A.n_groups, A.o_lora_rank)
        return linear_fp8(o, A.wo_b)

    def compressor(self, A: Attention, x: torch.Tensor, pos: torch.Tensor):
        C = A.compressor
        if C.ratio == 1:
            latent = rmsnorm(F.linear(x, C.wkv), C.norm_w, C.eps)
            return latent, torch.ones((), dtype=torch.bool, device=x.device)
        xf = x.float()
        kv, score = F.linear(xf, C.wkv), F.linear(xf, C.wgate)
        slot = torch.remainder(pos, C.ratio).view(1)
        C.kv_state.index_copy_(1, slot, kv)
        C.score_state.index_copy_(1, slot, score)
        pooled = (C.kv_state * C.score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
        should = torch.remainder(pos + 1, C.ratio) == 0
        return rmsnorm(pooled.to(x.dtype), C.norm_w, C.eps), should

    def compressor2(self, A: Attention, x: torch.Tensor, pos: torch.Tensor, seq: torch.Tensor):
        """Per-row compressor: x [B, 1, dim] at positions pos of sequences seq. The raw kv / gate score of every position
        go to the sequence's ring (slot = pos % RING); a row whose position closes a group pools the group's members
        from the ring (ratio 2: this position and the previous one). Returns (latent [B, 1, D], should [B] bool)."""
        C = A.compressor
        B = x.shape[0]
        if C.ratio == 1:
            latent = rmsnorm(F.linear(x, C.wkv), C.norm_w, C.eps)
            return latent, torch.ones(B, dtype=torch.bool, device=x.device)
        assert C.ratio == 2
        xf = x.float()
        kv, score = F.linear(xf, C.wkv)[:, 0], F.linear(xf, C.wgate)[:, 0]
        R = C.RING
        if os.environ.get("DSV41_DBG_RING"):
            print("[ring]", A.layer_id, C.kv_ring.shape, C.kv_ring.device, "seq", seq.tolist(), seq.device, "pos", pos.tolist(), "kv", kv.shape, kv.device, flush=True)
        C.kv_ring[seq, pos % R] = kv
        C.score_ring[seq, pos % R] = score
        prev = (pos - 1) % R
        kv2 = torch.stack([C.kv_ring[seq, prev], kv], dim=1)  # [B, 2, D]
        sc2 = torch.stack([C.score_ring[seq, prev], score], dim=1)
        pooled = (kv2 * sc2.softmax(dim=1)).sum(dim=1, keepdim=True)
        should = torch.remainder(pos + 1, C.ratio) == 0
        return rmsnorm(pooled.to(x.dtype), C.norm_w, C.eps), should

    def indexer(self, A: Attention, x, qr, latent, pos, compress_len, d, row):
        I = A.indexer
        ratio, rd = I.ratio, self.rd
        if I.owns_k:
            self.index_owner = A.layer_id
            k = rmsnorm(F.linear(latent, I.wk), I.k_norm_w, I.eps).contiguous()
            rope_dev_(k, rd, I.cos, I.sin, pos, add=1 - ratio)
            k = fake_quant_fp4(k, 32)
            cache = self.m.shared.index_k[(A.layer_id, d)]
            if FUSED2:
                cache[self.seq[d], row] = k[:, 0]
            else:
                cache.index_copy_(1, row.view(1), k)
            val, idx = self.ik_row[A.layer_id]
            val.copy_(k)
            idx.copy_(row.view(-1))
        B = qr.shape[0]
        q = linear_fp8(qr, I.wq_b).unflatten(-1, (I.n_heads, I.head_dim))
        rope_dev_(q, rd, I.cos, I.sin, pos)
        q = fake_quant_fp4(q, 32)
        index_k = self.m.shared.index_k[(self.index_owner, d)]  # [S, max_c + 1, 128] (last row = dummy)
        if FUSED2:
            index_k = index_k.index_select(0, self.seq[d])  # the row's sequence
        weights = F.linear(x, I.weights_proj) * (I.softmax_scale * I.n_heads**-0.5)
        score = torch.einsum("bshd,btd->bsht", q.float(), index_k.float())
        score = (score.relu_() * weights.float().unsqueeze(-1)).sum(dim=2)  # [B, 1, max_c + 1]
        n_pos = score.shape[-1]
        cl = compress_len.view(-1, 1, 1) if compress_len.dim() else compress_len
        score = score.masked_fill(torch.arange(n_pos, device=d) >= cl, -torch.inf)
        if I.is_candidate_source:
            cand = select_candidate_blocks(score, cl, I.candidate_topk_blocks, I.candidate_block_size)
            self.cand_buf[d][..., :n_pos].copy_(cand)
        elif I.uses_candidates:
            score = score.masked_fill(~self.cand_buf[d][..., :n_pos], -torch.inf)
        idxs = score.topk(self.topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        if FUSED2:
            return torch.where(idxs < cl, idxs, -1).to(torch.int32)
        return torch.where(idxs < compress_len, idxs + self.win, -1).to(torch.int32)

    # ------------------------------------------------------------------ block / segment
    def _hc_sub(self, blk: Block, h: torch.Tensor, params, pre_in, norm_w, want_f32=False, out=None, want_perm=False):
        """Hyper-connection split + hc_pre + rmsnorm + fp8 quant for one sub-block.
        Returns ((pre, post, comb), side, x, xq, xf[, xqp]): with HC_FORK the split (hc_mix + sinkhorn) runs on a side
        stream (a parallel graph branch; it is only needed at hc_post) and `side` must be joined by _hc_join first.
        want_perm: also the 8-k permuted copy of xq for the tensor-core expert GEMM."""
        fn, scale, base = params
        if not HC_FORK and h.shape[0] == 1:  # the single fused kernel is one row only
            mixes = hc_mix(h, fn, blk.eps)
            r = hc_pre_norm_quant(h, pre_in, mixes, scale, base, norm_w, blk.eps, blk.hc_eps, blk.sinkhorn_iters, want_f32=want_f32, out=out, want_perm=want_perm)
            return (r[0], r[1], r[2]), None, *r[3:]
        split, side = self._hc_split(blk, h, params)
        r = hc_pre_norm_quant2(h, pre_in, norm_w, blk.eps, want_f32=want_f32, out=out, want_perm=want_perm)
        return split, side, *r

    def _hc_split(self, blk: Block, h: torch.Tensor, params):
        fn, scale, base = params
        d = blk.device
        main = torch.cuda.current_stream(d)
        side = self._side_stream(d)
        side.wait_stream(main)
        with torch.cuda.stream(side):
            mixes = hc_mix(h, fn, blk.eps)
            out = hc_sinkhorn(mixes, scale, base, blk.hc, blk.sinkhorn_iters, blk.hc_eps)
        return out, side

    def _side_stream(self, d):
        ss = getattr(self, "_side", None)
        if ss is None:
            ss = self._side = {}
        if d not in ss:
            ss[d] = torch.cuda.Stream(d)
        return ss[d]

    @staticmethod
    def _hc_join(side, d):
        if side is not None:
            torch.cuda.current_stream(d).wait_stream(side)

    def block2(self, blk: Block, h: torch.Tensor, pre_mix: torch.Tensor):
        """One layer on the residual buffer h [B, 1, hc, dim] (updated in place); pre_mix: [B, hc] fp32."""
        d = blk.device
        (pre_n, post, comb), side, x, xq, _ = self._hc_sub(blk, h, blk.hc_attn, pre_mix, blk.attn_norm_w)
        a = self.attention2(blk.attn, x, xq, d)
        self._hc_join(side, d)
        hc_post2_(a.view(1, -1), h, post, comb)
        (pre_out, post, comb), side, x, xq, xf, xqp = self._hc_sub(blk, h, blk.hc_ffn, pre_n, blk.ffn_norm_w, want_f32=True, want_perm=True)
        y2, ys = self.moe2(blk.ffn, xq, xf, xqp)
        self._hc_join(side, d)
        hc_post2_(None, h, post, comb, y2=y2, ys=ys)
        return h, pre_out

    def _tc_tables(self, n: int, d, topk: int = 0):
        """Constant group tables for n (token, expert) pairs, one group per pair: grp_start = 0..n, the token of pair p
        is p // topk (0 when topk == 0), and the w2 input row of pair p is p."""
        key = (n, d, topk)
        t = self._tc_cache.get(key) if hasattr(self, "_tc_cache") else None
        if t is None:
            if not hasattr(self, "_tc_cache"):
                self._tc_cache = {}
            tok = torch.arange(n, device=d, dtype=torch.int32) // topk if topk else torch.zeros(n, device=d, dtype=torch.int32)
            t = self._tc_cache[key] = (torch.arange(n + 1, device=d, dtype=torch.int32), tok, torch.arange(n, device=d, dtype=torch.int32))
        return t

    def experts_tc(self, xqp, hq_permute, w13, s13, w2, s2, eid, wt, inter, limit, n, d, topk: int = 0, shard=(0, 1 << 30)):
        """n (token, expert, weight) pairs on the tensor-core FP4 GEMM: fp32 [n, dim] in pair order (token-major).
        xqp: 8-k permuted x [B, K]; eid / wt: [B, topk] (pair p = token p // topk). With more than one token the
        pairs are bucketed by expert on the device (sorted; one weight read per distinct expert, up to B tokens per
        group); shard = (first expert, count) restricts the work to this GPU's experts (rows of the others are zero)."""
        B = n // topk if topk else 1
        if B <= 1:
            starts, tok, rows = self._tc_tables(n, d, topk)
            gu = fp4_gemm_tc(xqp, w13, s13, eid.reshape(-1), starts, tok, n, 1, shard_start=shard[0], shard_n=shard[1], zero_out=False)
            hqp = swiglu_quant(gu, wt.reshape(-1), inter, limit, permute=True)
            return fp4_gemm_tc(hqp, w2, s2, eid.reshape(-1), starts, rows, n, 1, shard_start=shard[0], shard_n=shard[1], zero_out=True)
        # ---- bucket the n pairs by expert (all static shapes; padded groups have expert -1 and are skipped)
        e = eid.reshape(-1).to(torch.int64)
        se, order = torch.sort(e, stable=True)
        tok_sorted = (order // topk).to(torch.int32)
        wt_sorted = wt.reshape(-1)[order].contiguous()
        ar = torch.arange(n, device=d)
        is_new = torch.ones(n, dtype=torch.bool, device=d)
        is_new[1:] = se[1:] != se[:-1]
        gmax = cukern.FP4_W_MAX if cukern.FP4_W_LAYOUT else 16  # tokens per group the kernels handle
        if B > gmax:  # split longer runs of the same expert
            gid0 = torch.cumsum(is_new.to(torch.int64), 0) - 1
            first = torch.full((n,), n, dtype=torch.int64, device=d).scatter_reduce_(0, gid0, torch.where(is_new, ar, torch.full_like(ar, n)), "amin")
            is_new = is_new | (((ar - first[gid0]) % gmax) == 0)
        gid = torch.cumsum(is_new.to(torch.int64), 0) - 1
        grp_expert = torch.full((n,), -1, dtype=torch.int32, device=d).scatter_(0, gid, se.to(torch.int32))
        grp_start = torch.full((n + 1,), n, dtype=torch.int32, device=d)
        grp_start.scatter_reduce_(0, gid, torch.where(is_new, ar, torch.full_like(ar, n)).to(torch.int32), "amin")
        gu = fp4_gemm_tc(xqp, w13, s13, grp_expert, grp_start, tok_sorted, n, min(B, gmax), shard_start=shard[0], shard_n=shard[1], zero_out=False)
        hqp = swiglu_quant(gu, wt_sorted, inter, limit, permute=True)
        rows = ar.to(torch.int32)
        y2s = fp4_gemm_tc(hqp, w2, s2, grp_expert, grp_start, rows, n, min(B, gmax), shard_start=shard[0], shard_n=shard[1], zero_out=True)
        inv = torch.empty_like(order).scatter_(0, order, ar)
        return y2s[inv]

    def moe2(self, moe, xq: torch.Tensor, xf: torch.Tensor, xqp=None):
        """Routed experts (fp32 [topk, dim], one row per selected expert, routing weight applied) and the shared expert (bf16 [1, dim])."""
        d = xq.device
        scores = F.linear(xf, moe.gate_w)
        eid, wt = gate_topk(scores, moe.gate_bias, moe.gate_temp, moe.topk, moe.route_scale, moe.score_func, moe.norm_topk_prob and moe.topk > 1,
                            eid=self.route_eid.get(moe.layer_id), wt=self.route_wt.get(moe.layer_id))
        if FP4_TC and xqp is not None:
            B = xq.shape[0]
            y2 = self.experts_tc(xqp, True, moe.w13, moe.s13, moe.w2, moe.s2, eid, wt, moe.inter, moe.swiglu_limit, B * moe.topk, d, topk=moe.topk)
            y2 = y2.view(B, moe.topk, -1)
        else:
            tok, pair_rows, ones = moe._pair_tables(1, d)
            gu = cukern_fp4(xq, moe.w13, moe.s13, tok, eid, ones, moe.topk)
            hq = swiglu_quant(gu, wt, moe.inter, moe.swiglu_limit)
            y2 = cukern_fp4(hq, moe.w2, moe.s2, pair_rows, eid, ones, moe.topk)
        gu_s = linear_w(xq, moe.sh_w13)
        hs = swiglu_quant(gu_s, None, moe.inter, moe.swiglu_limit)
        ys = linear_w(hs, moe.sh_w2)
        return y2, ys

    def route_snapshot(self):
        """(eid [layers, topk] int64, wt [layers, topk] fp32) of the last step (route logging on)."""
        L = sorted(self.route_eid)
        eid = torch.stack([self.route_eid[l].cpu() for l in L]).long()  # [layers, B, topk]
        wt = torch.stack([self.route_wt[l].cpu() for l in L])
        if self.B == 1:
            return eid[:, 0], wt[:, 0]
        return eid, wt

    def block(self, blk: Block, x: torch.Tensor, pre_mix: torch.Tensor):
        residual = x
        attn_pre, attn_post, attn_comb = blk.hc_mixes(x, *blk.hc_attn)
        x = _hc_pre(x, pre_mix)
        x = rmsnorm(x, blk.attn_norm_w, blk.eps)
        x = self.attention(blk.attn, x, blk.device)
        x = _hc_post(x, residual, attn_post, attn_comb)
        residual = x
        ffn_pre, ffn_post, ffn_comb = blk.hc_mixes(x, *blk.hc_ffn)
        x = _hc_pre(x, attn_pre)
        x = rmsnorm(x, blk.ffn_norm_w, blk.eps)
        x = blk.ffn(x)
        x = _hc_post(x, residual, ffn_post, ffn_comb)
        return x, ffn_pre

    def run_segment(self, si: int):
        d, blocks = self.segments[si]
        with torch.cuda.device(d):
            if si == 0:
                h = F.embedding(self.tok, self.m.embed).unsqueeze(2).repeat(1, 1, self.cfg["hc_mult"], 1)
                pre = torch.zeros(self.B, 1, self.cfg["hc_mult"], dtype=torch.float32, device=d)
                pre[:, :, 0] = 1.0
            else:
                h, pre = self.h_in[d], self.pre_in[d]
            if FUSED2:
                h = h.contiguous()
                pre = pre.reshape(self.B, -1)
            for blk in blocks:
                if blk.engram is not None:
                    h = blk.engram.apply(h, self.eng_in[blk.layer_id])
                    if FUSED2:
                        h = h.contiguous()
                if blk.layer_id in self.main_hid:
                    self.main_hid[blk.layer_id].copy_(h.mean(2).view(self.B, -1))
                h, pre = (self.block2 if FUSED2 else self.block)(blk, h, pre)
            if FUSED2:
                pre = pre.view(self.B, 1, -1)
            if si == len(self.segments) - 1:
                hh = _hc_pre(h, pre)[:, -1]
                hh = rmsnorm(hh, self.m.norm_w, self.cfg["norm_eps"])
                self.logits.copy_(F.linear(hh, self.m.head).float())
            else:
                self.h_out[d].copy_(h)
                self.pre_out[d].copy_(pre)

    def _propagate(self, si: int):
        """After a segment: push cache rows written by its owners and the index/candidate buffers to
        the devices of later segments (plain D2D copies, outside the graphs)."""
        d, blocks = self.segments[si]
        later = self.devices[si + 1 :]
        if not later:
            return
        for blk in blocks:
            lid = blk.layer_id
            if lid in self.kv_row:
                val, idx = self.kv_row[lid]
                for dd in later:
                    self.m.shared.compress_kv[(lid, dd)][self.seq[dd], idx.to(dd, non_blocking=True)] = val[:, 0].to(dd, non_blocking=True)
                val, idx = self.ik_row[lid]
                for dd in later:
                    self.m.shared.index_k[(lid, dd)][self.seq[dd], idx.to(dd, non_blocking=True)] = val[:, 0].to(dd, non_blocking=True)
        nd = later[0]
        self.topk_buf[nd].copy_(self.topk_buf[d], non_blocking=True)
        self.cand_buf[nd].copy_(self.cand_buf[d], non_blocking=True)
        self.h_in[nd].copy_(self.h_out[d], non_blocking=True)
        self.pre_in[nd].copy_(self.pre_out[d], non_blocking=True)

    def capture(self):
        for si, (d, _) in enumerate(self.segments):
            with torch.cuda.device(d):
                s = torch.cuda.Stream(d)
                s.wait_stream(torch.cuda.current_stream(d))
                with torch.cuda.stream(s):
                    for _ in range(2):  # warm up (Triton compiles, allocator)
                        self.run_segment(si)
                torch.cuda.current_stream(d).wait_stream(s)
                torch.cuda.synchronize(d)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s):
                    self.run_segment(si)
                self.graphs[d] = g
        torch.cuda.synchronize()
        self._graph_cache_signature = self._cache_signature()

    def copy_seq(self, src: int, dst: int):
        """Copy every per-sequence state (window rings, compressed caches, index keys, compressor rings, Engram history)
        from sequence slot src to slot dst (used to prefill sequences one at a time into slot 0)."""
        for blk in self.m.blocks:
            A = blk.attn
            A.window_kv_cache[dst].copy_(A.window_kv_cache[src])
            C = A.compressor
            if C is not None and C.ratio > 1:
                for t in (C.kv_ring, C.score_ring, C.kv_state, C.score_state):
                    t[dst].copy_(t[src])
        for cache in list(self.m.shared.compress_kv.values()) + list(self.m.shared.index_k.values()):
            cache[dst].copy_(cache[src])
        if self.m.engram_hash is not None:
            self.m.engram_hash.cache[dst].copy_(self.m.engram_hash.cache[src])

    def _cache_signature(self):
        return tuple((name, key, value.data_ptr(), tuple(value.shape))
                     for name in ("compress_kv", "index_k")
                     for key, value in getattr(self.m.shared, name).items())

    def _prepare_decode_cache(self, end_pos):
        """Grow before GPU dispatch; tensor .item() is illegal during capture.

        Keep a dummy row beyond the visible keys. Prefill can also replace
        cache tensors, so validate captured pointers even without growth here.
        """
        shared = self.m.shared
        pending = []
        for name in ("compress_kv", "index_k"):
            table = getattr(shared, name)
            for owner in {key[0] for key in table}:
                ratio = self.m.blocks[owner].attn.ratio
                needed = end_pos // ratio + 1
                if any(v.size(1) < needed for (o, _), v in table.items() if o == owner):
                    pending.append((table, owner, needed, name))
        if pending:
            for device in self.devices:
                torch.cuda.synchronize(device)
            for table, owner, needed, name in pending:
                shared._ensure_capacity(table, owner, needed, name)
        if self.graphs and self._cache_signature() != getattr(self, "_graph_cache_signature", None):
            # Replaying a graph after reallocation reads stale pointers.
            # Recapture performs warm-up writes into model state; use eager
            # execution until an explicit safe recapture instead.
            for device in self.devices:
                torch.cuda.synchronize(device)
            self.graphs.clear()
            print("[decode-cache] cache storage changed; using eager decode", flush=True)

    def set_rows(self, token, pos, seq=None, pmax=None):
        """Fill the row tables: token(s), position(s), sequence id per row (default 0..B-1) and the newest position
        written to each row's sequence this step (default: the row's own position). Ints are broadcast."""
        B = self.B
        as_rows = lambda v, dt: torch.full((B,), int(v), dtype=dt) if isinstance(v, int) else torch.as_tensor(v, dtype=dt).view(-1)
        tok = as_rows(token, torch.int64)
        self.tok.copy_(tok.view(-1, 1))
        p = as_rows(pos, torch.int64)
        # Cache sizing and pointer validation require host synchronization;
        # CUDA graph capture must remain entirely device side.
        if not torch.cuda.is_current_stream_capturing():
            self._prepare_decode_cache(int(p.max().item()) + 1)
        sq = as_rows(seq, torch.int64) if seq is not None else torch.arange(B, dtype=torch.int64) % self.S
        pm = as_rows(pmax, torch.int64) if pmax is not None else p
        for d in self.devices:
            self.pos[d].copy_(p)
            self.seq[d].copy_(sq)
            self.pmax[d].copy_(pm)
        return tok, p, sq

    def _engram_rows(self, tok, p, sq):
        if self.m.engram_hash is None:
            return
        dev0 = self.devices[0]
        hashes = self.m.engram_hash.rows(tok.to(dev0), sq.to(dev0), p.to(dev0))
        for blk in self.m.blocks:
            if blk.engram is not None:
                emb = blk.engram.table.lookup(hashes[:, :, blk.engram.layer_hash_index, :], blk.device).flatten(-2)
                self.eng_in[blk.layer_id].copy_(emb)

    @torch.inference_mode()
    def step(self, token, pos, seq=None, pmax=None) -> torch.Tensor:
        """One decode step for B rows: token(s) at position(s) `pos` of sequence(s) `seq` (see set_rows);
        returns logits [B, vocab] on the last device."""
        tok, p, sq = self.set_rows(token, pos, seq, pmax)
        # the owner bookkeeping mirrors the eager path: sources set themselves as they run (in layer order)
        self.kv_owner = -1
        self.index_owner = -1
        self._engram_rows(tok, p, sq)
        for si, (d, _) in enumerate(self.segments):
            if self.use_graphs and d in self.graphs:
                self.graphs[d].replay()
            else:
                self.run_segment(si)
            self._propagate(si)
        return self.logits


def _gpu_numa_node(d: torch.device) -> int | None:
    try:
        pr = torch.cuda.get_device_properties(d)
        bus = f"{pr.pci_domain_id:04x}:{pr.pci_bus_id:02x}:{pr.pci_device_id:02x}.0"
        return int(open(f"/sys/bus/pci/devices/{bus}/numa_node").read())
    except Exception:
        return None


class OffloadDecodeRuntime(DecodeRuntime):
    """Single-GPU expert-offload decode: each layer is a few CUDA graphs with the host-side expert
    routing (gate result -> CPU experts / hot-slot ids) in between. The dense half of the layer uses
    the fused decode kernels on static buffers (h_buf is the residual stream, updated in place)."""

    def __init__(self, model: Transformer, use_graphs: bool = True):
        super().__init__(model, use_graphs=False)
        assert len(self.devices) == 1, "offload mode runs all layers on one GPU"
        self.d = self.devices[0]
        self.use_graphs = use_graphs
        hc, dim = self.cfg["hc_mult"], self.cfg["dim"]
        d = self.d
        self.h_buf = torch.zeros(1, 1, hc, dim, dtype=torch.bfloat16, device=d)
        self.pre_buf = torch.zeros(hc, dtype=torch.float32, device=d)
        self.pre_mid = torch.zeros(hc, dtype=torch.float32, device=d)
        self.xq_buf = torch.zeros(1, dim, dtype=torch.bfloat16, device=d)
        self.xqp_buf = torch.zeros(1, dim, dtype=torch.bfloat16, device=d)  # 8-k permuted copy (tensor-core expert GEMM)
        self.ffn_pre = torch.zeros(hc, dtype=torch.float32, device=d)
        self.ffn_post = torch.zeros(hc, dtype=torch.float32, device=d)
        self.ffn_comb = torch.zeros(hc, hc, dtype=torch.float32, device=d)
        topk = self.cfg["n_activated_experts"]
        self.gate_w = torch.zeros(topk, dtype=torch.float32, device=d)
        self.gate_idx = torch.zeros(topk, dtype=torch.int32, device=d)
        self.g1: dict[int, torch.cuda.CUDAGraph] = {}
        self.g2: dict[int, torch.cuda.CUDAGraph] = {}
        self.g3: dict[int, torch.cuda.CUDAGraph] = {}
        self.cpu_experts = model.blocks[0].ffn.host is not None
        self.prof = None  # set to a defaultdict(float) to collect per-stage seconds
        self.y_pair = torch.zeros(2, dim, dtype=torch.float32, device=d)  # row 0: CPU experts, row 1: shared (+ hot) experts
        self.y_gpu = self.y_pair[0:1]
        self.y_shared = self.y_pair[1:2]
        self.ys_zero = torch.zeros(1, dim, dtype=torch.bfloat16, device=d)
        self.slot_ids = torch.zeros(topk, dtype=torch.int32, device=d)
        self.slot_w = torch.zeros(topk, dtype=torch.float32, device=d)
        self.n_cold = 0
        # pinned host mirrors: the gate result and x go host-side inside graph 1 (D2H memcpy nodes), the hot-slot
        # table and the CPU expert output go back inside graphs 2 / 3, so the host syncs once per layer
        self.x_host = torch.zeros(dim, dtype=torch.bfloat16, pin_memory=True)
        self.ids_host = torch.zeros(topk, dtype=torch.int32, pin_memory=True)
        self.wts_host = torch.zeros(topk, dtype=torch.float32, pin_memory=True)
        self.slot_ids_host = torch.zeros(topk, dtype=torch.int32, pin_memory=True)
        self.slot_w_host = torch.zeros(topk, dtype=torch.float32, pin_memory=True)
        self.cold_ids_host = torch.zeros(topk, dtype=torch.int32, pin_memory=True)
        self.cold_w_host = torch.zeros(topk, dtype=torch.float32, pin_memory=True)
        self.y_host = torch.zeros(dim, dtype=torch.float32, pin_memory=True)
        # numpy views of the pinned routing tables: filling them is one slice assignment instead of per-element torch ops
        self.slot_ids_np, self.slot_w_np = self.slot_ids_host.numpy(), self.slot_w_host.numpy()
        self.cold_ids_np, self.cold_w_np = self.cold_ids_host.numpy(), self.cold_w_host.numpy()
        self.limit = float(model.blocks[0].ffn.swiglu_limit)
        self.call_times: list[float] = []
        self.capturing = False
        self.hotcache = None
        if self.cpu_experts:
            from . import cpumoe
            cpumoe.lib()
            cpumoe.pin_main_thread(_gpu_numa_node(d))
            if any(blk.ffn.hot for blk in model.blocks) and os.environ.get("DSV41_ADAPTIVE_HOT", "1") == "1":
                from .hotcache import HotCache
                self.hotcache = HotCache(model.blocks, d, swaps_per_token=int(os.environ.get("DSV41_HOT_SWAPS", "2")),
                                         margin=float(os.environ.get("DSV41_HOT_MARGIN", "2.0")), cpus=cpumoe.RESERVED_CPUS or None)

    # ---- the halves of a layer, on static buffers
    def part1(self, blk: Block):
        h = self.h_buf
        if blk.engram is not None:
            h.copy_(blk.engram.apply(h, self.eng_in[blk.layer_id]))
        d = blk.device
        (pre_n, post, comb), side, x, xq, _ = self._hc_sub(blk, h, blk.hc_attn, self.pre_buf, blk.attn_norm_w, out={"pre": self.pre_mid})
        a = self.attention2(blk.attn, x, xq, d)
        self._hc_join(side, d)
        hc_post2_(a.view(1, -1), h, post, comb)
        if HC_FORK:
            self.pre_mid.copy_(pre_n)
        (pre_f, post_f, comb_f), side, _, _, xf, _ = self._hc_sub(blk, h, blk.hc_ffn, self.pre_mid, blk.ffn_norm_w, want_f32=True, want_perm=True,
                                                                out={"pre": self.ffn_pre, "post": self.ffn_post, "comb": self.ffn_comb, "yq": self.xq_buf, "yqp": self.xqp_buf})
        self._hc_join(side, d)
        if HC_FORK:
            self.ffn_pre.copy_(pre_f)
            self.ffn_post.copy_(post_f)
            self.ffn_comb.copy_(comb_f)
        moe = blk.ffn
        scores = F.linear(xf, moe.gate_w)
        gate_topk(scores, moe.gate_bias, moe.gate_temp, moe.topk, moe.route_scale, moe.score_func, moe.norm_topk_prob and moe.topk > 1,
                  eid=self.gate_idx, wt=self.gate_w)
        if self.cpu_experts:  # D2H for the host-side routing (captured as memcpy nodes)
            self.x_host.copy_(self.xq_buf.view(-1), non_blocking=True)
            self.ids_host.copy_(self.gate_idx, non_blocking=True)
            self.wts_host.copy_(self.gate_w, non_blocking=True)

    def _shared(self, moe):
        gu_s = linear_w(self.xq_buf, moe.sh_w13)
        hs = swiglu_quant(gu_s, None, moe.inter, moe.swiglu_limit)
        return linear_w(hs, moe.sh_w2)

    def part2_cpu_shared(self, blk: Block):
        """GPU work that overlaps the CPU expert computation: the shared expert and the hot experts."""
        moe = blk.ffn
        y = self._shared(moe).float()
        if moe.hot:
            self.slot_ids.copy_(self.slot_ids_host, non_blocking=True)  # H2D memcpy nodes at the head of graph 2
            self.slot_w.copy_(self.slot_w_host, non_blocking=True)
            if FP4_TC:
                hot = moe.hot
                y2 = self.experts_tc(self.xqp_buf, True, hot["w13"], hot["s13"], hot["w2"], hot["s2"], self.slot_ids, self.slot_w, moe.inter, moe.swiglu_limit, moe.topk, self.d)
                y = y + y2.sum(dim=0, keepdim=True)
            else:
                y = y + moe.hot_experts_gpu(self.xq_buf, self.slot_ids, self.slot_w)
        self.y_shared.copy_(y)

    def part2_cpu_post(self, blk: Block):
        self.y_gpu.view(-1).copy_(self.y_host, non_blocking=True)  # H2D memcpy node at the head of graph 3
        hc_post2_(None, self.h_buf, self.ffn_post, self.ffn_comb, y2=self.y_pair, ys=self.ys_zero)
        self.pre_buf.copy_(self.ffn_pre)

    def _route(self, blk: Block):
        """After graph 1: wait for its D2H copies (the one host sync per layer), split the experts into hot
        (GPU slots) and cold (CPU), fill the pinned slot / cold tables. Returns the number of cold experts."""
        moe = blk.ffn
        torch.cuda.current_stream(self.d).synchronize()
        ids = self.ids_host.tolist()
        wts = self.wts_host.tolist()
        if self.hotcache is not None and not self.capturing:
            self.hotcache.update(blk.layer_id, ids)
        gs, gw, cid, cw = moe.split_hot(ids, wts)
        if gs is not None:
            self.slot_ids_np[:] = gs
            self.slot_w_np[:] = gw
        n = len(cid)
        if n:
            self.cold_ids_np[:n] = cid
            self.cold_w_np[:n] = cw
        return n

    def _cpu_experts(self, blk: Block, n_cold: int):
        moe = blk.ffn
        t0 = time.perf_counter() if self.prof is not None else 0.0
        if n_cold:
            moe.host.forward_ids(self.x_host.data_ptr(), self.cold_ids_host.data_ptr(), self.cold_w_host.data_ptr(), n_cold, self.y_host.data_ptr(), self.limit)
        else:
            self.y_host.zero_()
        if self.prof is not None:
            dt = time.perf_counter() - t0
            self.prof["  (of which cpumoe_forward call)"] += dt
            self.call_times.append(dt)
            if _CPU_DEBUG:
                print(f"[pycall] E={n_cold} {dt * 1e6:.0f}us", file=sys.stderr, flush=True)

    def part2(self, blk: Block):
        moe = blk.ffn
        n_pairs = moe.topk
        tok, pair_rows, ones = moe._pair_tables(1, self.d)
        b = moe._staging(n_pairs, "decode")
        local = torch.arange(n_pairs, device=self.d, dtype=torch.int32)
        if FP4_TC:
            y2 = self.experts_tc(self.xqp_buf, True, b["w13"], b["s13"], b["w2"], b["s2"], local, self.gate_w, moe.inter, moe.swiglu_limit, n_pairs, self.d)
        else:
            gu = cukern_fp4(self.xq_buf, b["w13"], b["s13"], tok, local, ones, n_pairs)
            hq = swiglu_quant(gu, self.gate_w, moe.inter, moe.swiglu_limit)
            y2 = cukern_fp4(hq, b["w2"], b["s2"], pair_rows, local, ones, n_pairs)
        hc_post2_(None, self.h_buf, self.ffn_post, self.ffn_comb, y2=y2, ys=self._shared(moe))
        self.pre_buf.copy_(self.ffn_pre)

    def _gather(self, moe):
        b = moe._staging(moe.topk, "decode")
        for i, e in enumerate(self.gate_idx.tolist()):  # the one host sync per layer
            for gk, src in (("w13", moe.w13), ("s13", moe.s13), ("w2", moe.w2), ("s2", moe.s2)):
                b[gk][i].copy_(src[e], non_blocking=True)

    def capture(self):
        if not self.use_graphs:
            return
        self.capturing = True
        with torch.cuda.device(self.d):
            s = torch.cuda.Stream(self.d)
            s.wait_stream(torch.cuda.current_stream(self.d))
            for blk in self.m.blocks:
                with torch.cuda.stream(s):
                    for _ in range(2):
                        self.part1(blk)
                        if self.cpu_experts:
                            n_cold = self._route(blk)
                            self.part2_cpu_shared(blk)
                            self._cpu_experts(blk, n_cold)
                            self.part2_cpu_post(blk)
                        else:
                            self._gather(blk.ffn)
                            self.part2(blk)
                torch.cuda.synchronize(self.d)
                g1 = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g1, stream=s, capture_error_mode="thread_local"):
                    self.part1(blk)
                if self.cpu_experts:
                    g2 = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g2, stream=s, capture_error_mode="thread_local"):
                        self.part2_cpu_shared(blk)
                    g3 = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g3, stream=s, capture_error_mode="thread_local"):
                        self.part2_cpu_post(blk)
                    self.g3[blk.layer_id] = g3
                else:
                    g2 = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g2, stream=s, capture_error_mode="thread_local"):
                        self.part2(blk)
                self.g1[blk.layer_id], self.g2[blk.layer_id] = g1, g2
            torch.cuda.current_stream(self.d).wait_stream(s)
            torch.cuda.synchronize(self.d)
        self.capturing = False

    @torch.inference_mode()
    def step(self, token: int, pos: int) -> torch.Tensor:
        d = self.d
        self.tok.fill_(token)
        self.pos[d].fill_(pos)
        self.kv_owner = -1
        self.index_owner = -1
        if self.m.engram_hash is not None:
            hashes = self.m.engram_hash(self.tok, pos)
            for blk in self.m.blocks:
                if blk.engram is not None:
                    emb = blk.engram.table.lookup(hashes[:, :, blk.engram.layer_hash_index, :], d).flatten(-2)
                    self.eng_in[blk.layer_id].copy_(emb)
        self.h_buf.copy_(F.embedding(self.tok, self.m.embed).unsqueeze(2).repeat(1, 1, self.cfg["hc_mult"], 1))
        self.pre_buf.zero_()
        self.pre_buf[0] = 1.0
        if self.hotcache is not None:
            self.hotcache.new_token()
        prof = self.prof
        for blk in self.m.blocks:
            lid = blk.layer_id
            t0 = time.perf_counter() if prof is not None else 0.0
            if self.use_graphs:
                self.g1[lid].replay()
            else:
                self.part1(blk)
            if self.cpu_experts:
                n_cold = self._route(blk)  # syncs on part1
                self.n_cold += n_cold
                if prof is not None:
                    t1 = time.perf_counter(); prof["gpu dense+attn (sync)"] += t1 - t0; t0 = t1
                if self.use_graphs:
                    self.g2[lid].replay()  # shared + hot experts on the GPU, overlapping the CPU cold experts
                else:
                    self.part2_cpu_shared(blk)
                self._cpu_experts(blk, n_cold)
                if prof is not None:
                    t1 = time.perf_counter(); prof["cpu cold experts"] += t1 - t0; t0 = t1
                if self.use_graphs:
                    self.g3[lid].replay()
                else:
                    self.part2_cpu_post(blk)
                if prof is not None:
                    torch.cuda.synchronize(d); prof["gpu post"] += time.perf_counter() - t0
                continue
            self._gather(blk.ffn)
            if self.use_graphs:
                self.g2[lid].replay()
            else:
                self.part2(blk)
        hh = _hc_pre(self.h_buf, self.pre_buf.view(1, 1, -1))[:, -1]
        hh = rmsnorm(hh, self.m.norm_w, self.cfg["norm_eps"])
        self.logits.copy_(F.linear(hh, self.m.head).float())
        return self.logits
