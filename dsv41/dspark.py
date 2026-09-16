"""DSpark (DeepSeek-V4.1's multi-token prediction head): 3 extra blocks that draft `block_size` (5) tokens
in one parallel pass from the last sampled token and the attention inputs of the target layers 37-39.

Reference semantics (inference/model.py, DSparkBlock / forward_spec):
  main_x  = main_norm(main_proj(cat(mean_hc(h_37), mean_hc(h_38), mean_hc(h_39))))          [1, dim]
  the draft blocks keep a sliding-window cache of kv(main_x) for the main positions;
  x       = embed([t, noise, noise, noise, noise]) (t = the token just sampled) at positions pos+1 .. pos+5
  each draft block: window attention of the 5 drafts over the main window + the 5 drafts themselves
  (non-causal), then a 128-expert top-3 MoE, all with hyper-connections like the main blocks;
  head:  logits of the 5 positions + a first-order Markov bias (prev token -> vocab) applied left to right,
  greedy pick, and a confidence score per draft position.
This module is the eager version (measurement / correctness); the static-graph version follows."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .fused import fake_quant_fp8, rmsnorm, rope_
from .load import _dense, load_layer
from .model import Block, _hc_post, _hc_pre, linear_fp8
from .w8 import linear_w, oproj_a


class DSpark:
    def __init__(self, ckpt, args, device, embed: torch.Tensor, head: torch.Tensor, shared, n_layers: int):
        cfg = args.cfg
        self.device = device
        self.dim = cfg["dim"]
        self.hc = cfg["hc_mult"]
        self.block = cfg["dspark_block_size"]
        self.noise = cfg["dspark_noise_token_id"]
        self.targets = list(cfg["dspark_target_layer_ids"])
        # DSpark runs entirely on its own device.  Both embedding and
        # LM head must live there; leaving head on the main model's last
        # GPU causes cuda:N / cuda:M mismatch during draft generation.
        self.embed = embed.to(device)
        self.head = head.to(device)
        self.eps = cfg["norm_eps"]
        self.blocks: list[Block] = []
        self.ws = []
        for i in range(cfg["n_mtp_layers"]):
            lid = cfg["n_layers"] + i  # the DSpark blocks' own ids (window-only attention), independent of a truncated model
            w = load_layer(ckpt, lid, device, prefix=f"mtp.{i}.")
            blk = Block(args, lid, w, device, shared)
            blk.ffn.topk = cfg["dspark_n_activated_experts"]
            blk.ffn.n_experts = cfg["dspark_n_routed_experts"]
            self.blocks.append(blk)
            self.ws.append(w)
        w0, wl = self.ws[0], self.ws[-1]
        self.main_proj = w0["main_proj.weight"]
        self.main_norm_w = w0["main_norm.weight"]
        self.norm_w = wl["norm.weight"]
        self.markov_embed = wl["markov_head.embed.weight"]  # bf16 [vocab, 256]
        self.markov_head = wl["markov_head.head.weight"]  # bf16 [vocab, 256]
        self.conf_w = wl["confidence_head.proj.weight"].float()  # [1, dim + 256]
        self.rd = cfg["rope_head_dim"]
        self.win = cfg["window_size"]

    # ---- main-side state: kv of the projected target hidden for every main position
    @torch.no_grad()
    def write_main(self, main_hidden: torch.Tensor, start_pos: int):
        """main_hidden: bf16 [T, 3*dim] for main positions start_pos .. start_pos+T-1."""
        T = main_hidden.shape[0]
        main_x = rmsnorm(linear_fp8(main_hidden.to(self.device), self.main_proj), self.main_norm_w, self.eps).view(1, T, self.dim)
        for blk in self.blocks:
            A = blk.attn
            kv = rmsnorm(linear_fp8(main_x, A.wkv), A.kv_norm_w, A.eps).contiguous()
            rope_(kv, self.rd, A.cos, A.sin, start_pos)
            kv = fake_quant_fp8(kv, 32)
            for j in range(T):
                A.window_kv_cache[0, (start_pos + j) % self.win] = kv[0, j]

    def _attention(self, A, x: torch.Tensor, pos: int) -> torch.Tensor:
        """x: [1, B, dim] draft inputs at positions pos+1 .. pos+B; attends to the main window (<= pos) and all drafts."""
        B = x.shape[1]
        qr = rmsnorm(linear_fp8(x, A.wq_a), A.q_norm_w, A.eps)
        q = linear_fp8(qr, A.wq_b).view(1, B, A.n_heads, A.head_dim).contiguous()
        rope_(q, self.rd, A.cos, A.sin, pos + 1)
        kv = rmsnorm(linear_fp8(x, A.wkv), A.kv_norm_w, A.eps).contiguous()
        rope_(kv, self.rd, A.cos, A.sin, pos + 1)
        kv = fake_quant_fp8(kv, 32)
        win = A.window_kv_cache[0]  # [win, head_dim]
        K = torch.cat([win, kv[0]], dim=0).float()  # [win + B, d]
        valid = torch.ones(self.win + B, dtype=torch.bool, device=x.device)
        if pos + 1 < self.win:
            valid[pos + 1 : self.win] = False
        s = torch.einsum("bhd,td->bht", q[0].float(), K) * A.softmax_scale  # [B, H, T]
        s = s.masked_fill(~valid, float("-inf"))
        m = s.amax(dim=-1, keepdim=True)
        p = torch.exp(s - m)
        l = p.sum(dim=-1, keepdim=True) + torch.exp(A.attn_sink.view(1, -1, 1) - m)
        o = (torch.einsum("bht,td->bhd", p, K) / l).to(torch.bfloat16).view(1, B, A.n_heads, A.head_dim).contiguous()
        rope_(o, self.rd, A.cos, A.sin, pos + 1, inverse=True)
        o = oproj_a(o.view(1, B, A.n_groups, -1), A.wo_a, A.n_groups, A.o_lora_rank)
        return linear_fp8(o, A.wo_b)

    @torch.no_grad()
    def draft(self, token: int, pos: int):
        """Drafts for positions pos+2 .. pos+1+block given t_{pos+1} = token and the main state up to pos.
        Returns (ids [block], confidence [block] fp32)."""
        B = self.block
        ids = torch.tensor([token] + [self.noise] * (B - 1), device=self.device)
        h = F.embedding(ids, self.embed).view(1, B, 1, self.dim).repeat(1, 1, self.hc, 1)
        pre = h.new_zeros(1, B, self.hc, dtype=torch.float32)
        pre[:, :, 0] = 1.0
        for blk in self.blocks:
            residual = h
            attn_pre, attn_post, attn_comb = blk.hc_mixes(h, *blk.hc_attn)
            x = rmsnorm(_hc_pre(h, pre), blk.attn_norm_w, blk.eps)
            a = self._attention(blk.attn, x, pos)
            h = _hc_post(a, residual, attn_post, attn_comb)
            residual = h
            ffn_pre, ffn_post, ffn_comb = blk.hc_mixes(h, *blk.hc_ffn)
            x = rmsnorm(_hc_pre(h, attn_pre), blk.ffn_norm_w, blk.eps)
            y = blk.ffn(x)
            h = _hc_post(y, residual, ffn_post, ffn_comb)
            pre = ffn_pre
        x = _hc_pre(h, pre)[0]  # [B, dim]
        logits = F.linear(rmsnorm(x, self.norm_w, self.eps), self.head).float()  # [B, vocab]
        out = [token]
        embeds = []
        for i in range(B):
            e = self.markov_embed[out[i]]
            embeds.append(e)
            logits[i] += F.linear(e.float(), self.markov_head.float())
            out.append(int(logits[i].argmax()))
        conf = F.linear(torch.cat([x.float(), torch.stack(embeds).float()], dim=-1), self.conf_w).view(-1)
        return out[1:], conf


# ---------------------------------------------------------------------------- batched (static-shape) draft
from .fused import fake_quant_fp8 as _fq8, rope_dev_
from .fused2 import gate_topk, hc_mix, hc_post2_, hc_pre_norm_quant2, hc_sinkhorn, kv_write, norm_quant, sattn2


class DSparkRows(DSpark):
    """Same computation as DSpark.draft, for S sequences at once on the runtime's fused kernels (rows = S x block).
    The draft blocks' window rings hold kv(main_x) of the main positions (write_main_rows, per row: sequence + position)."""

    def __init__(self, ckpt, args, device, embed, head, shared, n_layers, rt):
        super().__init__(ckpt, args, device, embed, head, shared, n_layers)
        self.rt = rt
        self.S = args.max_seqs or args.max_batch_size  # sequence slots (the caches are sized by it)
        for blk in self.blocks:
            blk.attn.draft_kv = torch.zeros(self.S, self.block, args.head_dim, dtype=torch.bfloat16, device=device)

    @torch.no_grad()
    def write_main_rows(self, main_hidden: torch.Tensor, seq: torch.Tensor, pos: torch.Tensor):
        """main_hidden [R, 3*dim]; seq / pos int64 [R].

        DSpark may live on a different GPU from the main model, so do not
        assume caller-created indexing tensors are already on self.device.
        Triton kv_write requires POS and SEQ to be CUDA pointers on the
        same device as the MTP attention tensors.
        """
        dev = self.device

        main_hidden = main_hidden.to(
            device=dev,
            non_blocking=True,
        )

        seq = seq.to(
            device=dev,
            dtype=torch.int64,
            non_blocking=True,
        ).contiguous()

        pos = pos.to(
            device=dev,
            dtype=torch.int64,
            non_blocking=True,
        ).contiguous()

        main_x = norm_quant(
            linear_fp8(main_hidden, self.main_proj),
            self.main_norm_w,
            self.eps,
        )

        for blk in self.blocks:
            A = blk.attn
            from .w8 import linear_w
            kv_write(linear_w(main_x, A.wkv), A.kv_norm_w, A.cos, A.sin, pos, A.window_kv_cache, self.rd, A.eps, seq)

    def _attention_rows(self, A, x, xq, seq_row, pos_row, pmax_row, plim_row, idx5):
        """x, xq: [R, dim] (R = S * block draft rows); attends to the sequence's main ring (positions <= plim) and all
        block drafts of the sequence (non-causal)."""
        from .w8 import linear_w, oproj_a
        R = x.shape[0]
        qr = norm_quant(linear_w(xq, A.wq_a), A.q_norm_w, A.eps)
        q = linear_w(qr, A.wq_b).view(R, 1, A.n_heads, A.head_dim)
        rope_dev_(q, self.rd, A.cos, A.sin, pos_row)
        kv = rmsnorm(linear_w(xq, A.wkv), A.kv_norm_w, A.eps).view(R, 1, -1).contiguous()
        rope_dev_(kv, self.rd, A.cos, A.sin, pos_row)
        kv = _fq8(kv, 32)
        A.draft_kv[seq_row, self.slot_row] = kv[:, 0]
        o = sattn2(q, A.window_kv_cache, A.draft_kv, idx5, pos_row, A.attn_sink, A.cos, A.sin, self.rd, A.softmax_scale, seq_row, pmax_row, plim=plim_row)
        o = oproj_a(o.view(R, 1, A.n_groups, -1), A.wo_a, A.n_groups, A.o_lora_rank)
        return linear_fp8(o, A.wo_b).view(R, -1)

    def capture(self, S: int):
        """Static buffers + one CUDA graph for draft_rows with S sequences (the draft is ~40 small launches per block)."""
        dev = self.device
        self.g_in = {"tokens": torch.zeros(S, dtype=torch.int64, device=dev), "pos": torch.zeros(S, dtype=torch.int64, device=dev),
                     "mh": torch.zeros(S, 3 * self.dim, dtype=torch.bfloat16, device=dev), "wmax": torch.zeros(S, dtype=torch.int64, device=dev)}
        self.g_out = torch.zeros(S, self.block, dtype=torch.int64, device=dev)
        st = torch.cuda.Stream(dev)
        with torch.cuda.device(dev), torch.cuda.stream(st):
            for _ in range(2):
                self.g_out.copy_(self._draft_rows(self.g_in["tokens"], self.g_in["pos"], self.g_in["mh"], self.g_in["wmax"]))
            torch.cuda.synchronize(dev)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=st, capture_error_mode="thread_local"):
                self.g_out.copy_(self._draft_rows(self.g_in["tokens"], self.g_in["pos"], self.g_in["mh"], self.g_in["wmax"]))
        torch.cuda.synchronize(dev)
        self.graph = g

    @torch.no_grad()
    def draft_rows(self, tokens, pos_last, main_hidden, written_max):
        if getattr(self, "graph", None) is not None and tokens.shape[0] == self.g_out.shape[0]:
            self.g_in["tokens"].copy_(tokens)
            self.g_in["pos"].copy_(pos_last)
            self.g_in["mh"].copy_(main_hidden)
            self.g_in["wmax"].copy_(written_max)
            self.graph.replay()
            return self.g_out
        return self._draft_rows(tokens, pos_last, main_hidden, written_max)

    @torch.no_grad()
    def _draft_rows(self, tokens: torch.Tensor, pos_last: torch.Tensor, main_hidden: torch.Tensor, written_max: torch.Tensor):
        """tokens [S] (the token just accepted, at position pos_last+1... i.e. t_{p+1} with p = pos_last), pos_last [S]
        (position p whose forward produced the token; the main rings hold positions <= written_max [S], only <= p are visible),
        main_hidden [S, 3*dim] (target-layer inputs at position p). Returns drafts int64 [S, block] on the device."""
        S, B = tokens.shape[0], self.block
        dev = self.device
        R = S * B
        rt = self.rt
        j = torch.arange(B, device=dev)
        seq_row = torch.arange(S, device=dev).repeat_interleave(B)
        self.slot_row = j.repeat(S)
        pos_row = (pos_last.repeat_interleave(B) + 1 + self.slot_row)
        pmax_row = written_max.repeat_interleave(B)
        plim_row = pos_last.repeat_interleave(B)
        idx5 = j.to(torch.int32).view(1, 1, B).expand(R, 1, B).contiguous()
        ids = torch.full((S, B), self.noise, dtype=torch.int64, device=dev)
        ids[:, 0] = tokens
        h = F.embedding(ids.view(-1), self.embed).view(R, 1, 1, self.dim).repeat(1, 1, self.hc, 1).contiguous()
        pre = torch.zeros(R, self.hc, device=dev)
        pre[:, 0] = 1.0
        for blk in self.blocks:
            fn, scale, base = blk.hc_attn
            pre_n, post, comb = hc_sinkhorn(hc_mix(h, fn, blk.eps), scale, base, blk.hc, blk.sinkhorn_iters, blk.hc_eps)
            x, xq, _ = hc_pre_norm_quant2(h, pre, blk.attn_norm_w, blk.eps)
            a = self._attention_rows(blk.attn, x, xq, seq_row, pos_row, pmax_row, plim_row, idx5)
            hc_post2_(a, h, post, comb)
            fn, scale, base = blk.hc_ffn
            pre_f, post, comb = hc_sinkhorn(hc_mix(h, fn, blk.eps), scale, base, blk.hc, blk.sinkhorn_iters, blk.hc_eps)
            x, xq, xf, xqp = hc_pre_norm_quant2(h, pre_n, blk.ffn_norm_w, blk.eps, want_f32=True, want_perm=True)
            moe = blk.ffn
            scores = F.linear(xf, moe.gate_w)
            eid, wt = gate_topk(scores, moe.gate_bias, moe.gate_temp, moe.topk, moe.route_scale, moe.score_func, moe.norm_topk_prob and moe.topk > 1)
            y2 = rt.experts_tc(xqp, True, moe.w13, moe.s13, moe.w2, moe.s2, eid, wt, moe.inter, moe.swiglu_limit, R * moe.topk, dev, topk=moe.topk)
            from .w8 import linear_w
            gu_s = linear_w(xq, moe.sh_w13)
            from .fused import swiglu_quant
            ys = linear_w(swiglu_quant(gu_s, None, moe.inter, moe.swiglu_limit), moe.sh_w2)
            hc_post2_(None, h, post, comb, y2=y2.view(R, moe.topk, -1), ys=ys)
            pre = pre_f
        x, _, _ = hc_pre_norm_quant2(h, pre, self.norm_w, self.eps)  # normed [R, dim] (the quantized copy is unused)
        logits = F.linear(x, self.head).float().view(S, B, -1)
        prev = tokens
        out = []

        markov_n = int(self.markov_embed.shape[0])

        for i in range(B):
            # CUDA-graph-safe bounds protection.
            #
            # IMPORTANT:
            # Do not use .item(), .tolist(), CPU copies, printing, or
            # data-dependent Python branching here. _draft_rows() is
            # executed while DSpark.capture() is recording a CUDA graph.
            #
            # Invalid token ids are redirected to a safe embedding row,
            # then their Markov residual is masked to zero.
            valid_prev = (
                (prev >= 0)
                & (prev < markov_n)
            )

            safe_prev = prev.clamp(
                min=0,
                max=markov_n - 1,
            )

            e = self.markov_embed[safe_prev]

            markov_delta = F.linear(
                e.float(),
                self.markov_head.float(),
            )

            markov_delta = markov_delta * (
                valid_prev
                .to(markov_delta.dtype)
                .unsqueeze(-1)
            )

            logits[:, i] += markov_delta

            prev = logits[:, i].argmax(-1)
            out.append(prev)
        return torch.stack(out, dim=1)
