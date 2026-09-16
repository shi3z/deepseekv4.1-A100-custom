"""Expert-parallel decode: one CUDA graph per GPU for the whole token, no host round trips.

Dense layers are pipelined over the GPUs in order (layer L's attention, norms, gate and shared expert
run on its owner), every layer's routed experts are sharded over all GPUs. Per layer the owner pushes
the quantized activation and the routing (6 expert ids + weights) into every peer's inbox with P2P
stores and raises a flag; each GPU computes the selected experts it holds (a masked grouped GEMM on
the FP4 tensor-core kernel), sums them into a partial and pushes it into the owner's inbox, raising
another flag; the owner waits for the partials, adds the shared expert and continues. When the owner
changes (pipeline hop) the residual stream and the attention bookkeeping travel the same way.

All synchronisation is device-side (flag kernels spinning on a per-token sequence number), so the
40 layers of a token are 4 graph launches and one host sync for the logits. Requires P2P access
between all the GPUs (same PCIe root / socket is fine: ~10 us per message)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

import os

from .cukern import fp4_gemm_tc, memcpy_async, p2p_copy, p2p_copy_row, p2p_multicast, p2p_seq_bump, p2p_signal, p2p_stamp, p2p_sum_rows, p2p_wait

# messages by copy engine (cuMemcpyAsync) instead of kernel P2P stores: the engines win for large messages
# (B=16: 164 KB routing packets, multicast 553 -> 76 us) while kernel stores have the lower latency for small
# ones (B=1: 41 vs ~10 us). "auto" switches at DSV41_EP_DMA_BYTES.
EP_DMA_MODE = os.environ.get("DSV41_EP_DMA", "auto")
EP_DMA_BYTES = int(os.environ.get("DSV41_EP_DMA_BYTES", "32768"))

EP_TRACE = os.environ.get("DSV41_EP_TRACE", "0") == "1"  # per-layer device timestamps (owner: 6 points, peer: 2)
# NVLink-pair relay (4 GPUs = two NV12 pairs, cross-pair links are PCIe at ~21 GB/s): the route packet goes to the
# owner's partner over NVLink and to ONE GPU of the other pair over PCIe, which forwards it to its partner over
# NVLink; the partials come back the same way in reverse (the far pair's leaf adds into its partner, one PCIe
# transfer instead of two). Partials travel as bf16 (the fp32 sum over the shard's experts rounded once).
EP_RELAY = os.environ.get("DSV41_EP_RELAY", "1") == "1"
EP_BF16_PART = os.environ.get("DSV41_EP_BF16_PART", "1") == "1"
from .decode import DecodeRuntime, HC_FORK
from .fused import rmsnorm, swiglu_quant
from .fused2 import gate_topk, hc_post2_
from .model import Block, Transformer, _hc_pre


class EPRuntime(DecodeRuntime):
    def __init__(self, model: Transformer, use_graphs: bool = True):
        super().__init__(model, use_graphs=False)
        self.use_graphs = use_graphs
        self.devs = [torch.device(f"cuda:{d}") for d in dict.fromkeys(b.device.index for b in model.blocks)]
        # every device in the pipeline order; expert shards live on all of them
        shards = model.blocks[0].ffn.ep
        assert shards is not None, "load the model with ep=True"
        self.shard = {sh["device"]: (sh["start"], sh["n"]) for sh in shards}
        self.devs = [sh["device"] for sh in shards]  # shard order == device order
        self.nd = len(self.devs)
        self.idx = {d: i for i, d in enumerate(self.devs)}
        nl = len(model.blocks)
        hc, dim, topk = self.cfg["hc_mult"], self.cfg["dim"], self.cfg["n_activated_experts"]
        self.topk_e = topk
        B = self.B
        # peer access (torch enables it lazily on a copy)
        for a in self.devs:
            for b in self.devs:
                if a != b:
                    torch.zeros(1, device=a).to(b)
        # static buffers
        self.seqno = {d: torch.ones(1, dtype=torch.int32, device=d) for d in self.devs}  # per-token flag sequence number (not the row sequence ids)
        self.hop_h = {d: torch.zeros(B, 1, hc, dim, dtype=torch.bfloat16, device=d) for d in self.devs}
        self.hop_pre = {d: torch.zeros(B, hc, dtype=torch.float32, device=d) for d in self.devs}
        self.pre_identity = {d: torch.tensor([[1.0] + [0.0] * (hc - 1)] * B, dtype=torch.float32, device=d) for d in self.devs}
        # routing message: [xqp bf16 [B, dim] | eid int32 [B, topk] | wt fp32 [B, topk]] in one buffer (owner's outbox / peers' inbox)
        nx, ne = B * dim * 2, B * topk * 4
        msg_bytes = -(-(nx + 2 * ne) // 16) * 16
        self.outbox = {d: torch.zeros(msg_bytes, dtype=torch.uint8, device=d) for d in self.devs}
        self.inbox = {d: torch.zeros(msg_bytes, dtype=torch.uint8, device=d) for d in self.devs}
        def views(buf):
            return (buf[:nx].view(torch.bfloat16).view(B, dim), buf[nx : nx + ne].view(torch.int32).view(B, topk), buf[nx + ne : nx + 2 * ne].view(torch.float32).view(B, topk))
        self.xqp_out, self.eid8, self.wt8 = {}, {}, {}
        self.inbox_x, self.inbox_eid, self.inbox_wt = {}, {}, {}
        for d in self.devs:
            self.xqp_out[d], self.eid8[d], self.wt8[d] = views(self.outbox[d])
            self.inbox_x[d], self.inbox_eid[d], self.inbox_wt[d] = views(self.inbox[d])
        self.dma_route = EP_DMA_MODE == "1" or (EP_DMA_MODE == "auto" and msg_bytes > EP_DMA_BYTES)
        self.dma_part = EP_DMA_MODE == "1" or (EP_DMA_MODE == "auto" and B * dim * 4 > EP_DMA_BYTES)
        self.bf16_part = EP_BF16_PART and self.dma_part
        pdt = torch.bfloat16 if self.bf16_part else torch.float32
        self.part_in = {d: torch.zeros(self.nd, B, dim, dtype=pdt, device=d) for d in self.devs}  # [sender, b, dim]
        self.part_out = {d: torch.zeros(B, dim, dtype=torch.float32, device=d) for d in self.devs}  # a peer's summed partial before the DMA
        self.part_out16 = {d: torch.zeros(B, dim, dtype=pdt, device=d) for d in self.devs} if self.bf16_part else self.part_out
        # NVLink pairs (d, d ^ 1) and the relay roles: for owner o, relay(o) is the lower GPU of the other pair
        self.partner = {d: next((p for p in self.devs if p.index == (d.index ^ 1)), None) for d in self.devs}
        self.relay = EP_RELAY and self.dma_route and self.dma_part and self.nd == 4 and all(v is not None for v in self.partner.values())
        self.relay_of = {}
        if self.relay:
            for o in self.devs:
                far = [d for d in self.devs if d != o and d != self.partner[o]]
                self.relay_of[o] = min(far, key=lambda dd: dd.index)
        self.part_relay = {d: torch.zeros(B, dim, dtype=pdt, device=d) for d in self.devs}  # the leaf partner's partial
        self.flag_relay = {d: torch.zeros(nl, dtype=torch.int32, device=d) for d in self.devs}
        self.flag_route = {d: torch.zeros(nl, dtype=torch.int32, device=d) for d in self.devs}
        self.flag_part = {d: torch.zeros(nl, self.nd, dtype=torch.int32, device=d) for d in self.devs}
        self.flag_hop = {d: torch.zeros(nl + 1, dtype=torch.int32, device=d) for d in self.devs}
        # pointer tables for the signal kernels (addresses of peers' flags)
        self.sig_route = {}
        self.sig_part = {}
        self.sig_hop = {}
        self.sig_fwd = {}    # (L): on relay(o), the leaf's route flag
        self.sig_relay = {}  # (L): on the leaf, relay(o)'s flag_relay
        for L, blk in enumerate(model.blocks):
            o = blk.device
            peers = [d for d in self.devs if d != o]
            if self.relay:
                a = self.relay_of[o]
                b = self.partner[a]
                direct = [self.partner[o], a]
                self.sig_route[L] = torch.tensor([self.flag_route[p][L].data_ptr() for p in direct], dtype=torch.int64, device=o)
                self.sig_fwd[L] = torch.tensor([self.flag_route[b][L].data_ptr()], dtype=torch.int64, device=a)
                self.sig_relay[L] = torch.tensor([self.flag_relay[a][L].data_ptr()], dtype=torch.int64, device=b)
                for p in peers:  # the relay raises the owner's slots of both far GPUs; the leaf signals nobody but the relay
                    slots = [self.idx[p]] + ([self.idx[b]] if p == a else [])
                    self.sig_part[(L, p)] = torch.tensor([self.flag_part[o][L, i].data_ptr() for i in slots], dtype=torch.int64, device=p)
            else:
                self.sig_route[L] = torch.tensor([self.flag_route[p][L].data_ptr() for p in peers], dtype=torch.int64, device=o)
                for p in peers:
                    self.sig_part[(L, p)] = torch.tensor([self.flag_part[o][L, self.idx[p]].data_ptr()], dtype=torch.int64, device=p)
            self.sig_inbox = getattr(self, "sig_inbox", {})
            if o not in self.sig_inbox:
                self.sig_inbox[o] = torch.tensor([self.inbox[p].data_ptr() for p in peers], dtype=torch.int64, device=o)
            if L + 1 < nl and model.blocks[L + 1].device != o:
                nxt = model.blocks[L + 1].device
                self.sig_hop[L] = torch.tensor([self.flag_hop[nxt][L + 1].data_ptr()], dtype=torch.int64, device=o)
        self._ptrs = {}
        for L, blk in enumerate(model.blocks):
            o = blk.device
            self._ptrs[("own", L, o)] = torch.tensor([self.flag_part[o][L, self.idx[o]].data_ptr()], dtype=torch.int64, device=o)
        # the candidate buffers travel between devices with 16-byte copies: pad them
        # Candidate masks are only consumed by static-shape decode graphs.
        # Do not make every graph buffer 1M tokens merely because the
        # logical KV limit is 1M; that exhausts graph-capture workspace.
        # Long prefill builds candidates dynamically in the model path.
        _cand_tokens = int(os.environ.get("DSV41_EP_CAND_TOKENS", "131072"))
        _cand_tokens = min(max(_cand_tokens, 1024), model.args.max_seq_len)
        n_cand = -((_cand_tokens + 1) // -16) * 16
        self.cand_buf = {d: torch.zeros(B, 1, n_cand, dtype=torch.bool, device=d) for d in self.devs}
        for d in self.devs:
            self._tc_tables(B * topk, d, topk)
        self.mcast_counter = {d: torch.zeros(1, dtype=torch.int32, device=d) for d in self.devs}
        self.graphs = {}
        self.logits = torch.zeros(B, self.cfg["vocab_size"], dtype=torch.float32, device=self.devs[-1])
        self.first_layer = {d: min(L for L, b in enumerate(model.blocks) if b.device == d) for d in self.devs}
        self.last_layer = {d: max(L for L, b in enumerate(model.blocks) if b.device == d) for d in self.devs}
        self.dry = False  # dry pass: no device-side waits / signals (only to compile and load every kernel)
        # every device works on its own non-blocking stream: a cross-device memcpy on a legacy default stream
        # would synchronise with the peer's default stream and deadlock against the flag waits
        self.streams = {d: torch.cuda.Stream(d) for d in self.devs}
        self.trace = {d: torch.zeros(nl, 16, dtype=torch.int64, device=d) for d in self.devs} if EP_TRACE else None

        # ------------------------------------------------------------
        # Exact preallocation BEFORE CUDA graph capture.
        #
        # Do NOT use SharedAttn._ensure_capacity() here: it grows
        # geometrically (typically 2x), which wastes several GiB across
        # five mirrors and can make cold prefill OOM.
        #
        # We allocate exactly enough compressed rows for a configurable
        # logical-token horizon.  Because this happens before CUDA graph
        # capture, the resulting pointers remain stable during decode.
        # ------------------------------------------------------------
        _prealloc_tokens = int(
            os.environ.get(
                "DSV41_EP_PREALLOC_TOKENS",
                "40000",
            )
        )

        def _exact_grow(table, owner, target, kind):
            keys = [
                key
                for key in list(table.keys())
                if key[0] == owner
            ]

            if not keys:
                return

            max_rows = self.m.shared.cache_max_rows.get(owner)

            if max_rows is not None:
                target = min(
                    int(target),
                    int(max_rows),
                )

            current = int(table[keys[0]].size(1))

            if target <= current:
                print(
                    f"[ep-prealloc-exact] "
                    f"{kind} owner={owner} "
                    f"rows={current:,} already>=target={target:,}",
                    flush=True,
                )
                return

            print(
                f"[ep-prealloc-exact] "
                f"{kind} owner={owner} "
                f"rows={current:,}->{target:,} "
                f"mirrors={len(keys)}",
                flush=True,
            )

            # Grow mirrors one at a time to limit peak allocation.
            for key in keys:
                old = table[key]
                old_rows = int(old.size(1))

                if old_rows >= target:
                    continue

                new_cache = torch.empty(
                    old.size(0),
                    target,
                    old.size(2),
                    dtype=old.dtype,
                    device=old.device,
                )

                new_cache[:, :old_rows].copy_(old)

                table[key] = new_cache
                del old

        if _prealloc_tokens > 0:
            owners = sorted(
                set(
                    owner
                    for owner, _dev
                    in (
                        list(self.m.shared.compress_kv.keys())
                        + list(self.m.shared.index_k.keys())
                    )
                )
            )

            for owner in owners:
                A = self.m.blocks[owner].attn

                ratio = max(
                    int(A.ratio),
                    1,
                )

                # ceil(logical_tokens / compression_ratio)
                target = (
                    _prealloc_tokens
                    + ratio
                    - 1
                ) // ratio

                if any(
                    o == owner
                    for o, _dev
                    in self.m.shared.compress_kv.keys()
                ):
                    _exact_grow(
                        self.m.shared.compress_kv,
                        owner,
                        target,
                        "compress_kv",
                    )

                if any(
                    o == owner
                    for o, _dev
                    in self.m.shared.index_k.keys()
                ):
                    _exact_grow(
                        self.m.shared.index_k,
                        owner,
                        target,
                        "index_k",
                    )

                print(
                    f"[ep-prealloc] "
                    f"owner={owner} "
                    f"ratio={ratio} "
                    f"tokens={_prealloc_tokens:,} "
                    f"rows={target:,}",
                    flush=True,
                )


    def _stamp(self, d, L, k):
        if self.trace is not None:
            p2p_stamp(self.trace[d][L, k], d)

    def _wait(self, flags, d):
        if not self.dry:
            p2p_wait(flags, self.seqno[d], d)

    def _signal(self, ptrs, d):
        if not self.dry:
            p2p_signal(ptrs, self.seqno[d], d)

    # ------------------------------------------------------------------ pieces
    def _experts_shard(self, d, xqp, eid, wt, moe):
        """This GPU's experts for the B * topk (token, expert) pairs: fp32 [B * topk, dim] (rows of other shards zero)."""
        start, n = self.shard[d]
        sh = [s for s in moe.ep if s["device"] == d][0]
        return self.experts_tc(xqp, True, sh["w13"], sh["s13"], sh["w2"], sh["s2"], eid, wt, moe.inter, moe.swiglu_limit,
                               self.B * self.topk_e, d, topk=self.topk_e, shard=(start, n))

    def _push_cache_rows(self, blk: Block, d):
        """After an owner ran a KV / index source layer: mirror the written rows to the later devices."""
        lid = blk.layer_id
        if lid not in self.kv_row:
            return
        later = [dd for dd in self.devs if self.idx[dd] > self.idx[d]]
        val, idx = self.kv_row[lid]
        for dd in later:
            p2p_copy_row(self.m.shared.compress_kv[(lid, dd)], idx, val, d, self.seq[d])
        val, idx = self.ik_row[lid]
        for dd in later:
            p2p_copy_row(self.m.shared.index_k[(lid, dd)], idx, val, d, self.seq[d])

    def _owner_layer(self, blk: Block, d, h, pre):
        L = blk.layer_id
        moe = blk.ffn
        self._stamp(d, L, 0)
        if blk.engram is not None:
            h.copy_(blk.engram.apply(h, self.eng_in[L]))
        if L in self.main_hid:  # DSpark reads the attention inputs of its target layers
            self.main_hid[L].copy_(h.mean(2).view(self.B, -1))
        (pre_n, post, comb), side, x, xq, _ = self._hc_sub(blk, h, blk.hc_attn, pre, blk.attn_norm_w)
        a = self.attention2(blk.attn, x, xq, d)
        self._push_cache_rows(blk, d)
        self._hc_join(side, d)
        hc_post2_(a.view(1, -1), h, post, comb)
        self._stamp(d, L, 1)
        (pre_out, post, comb), side, x, xq, xf, xqp = self._hc_sub(blk, h, blk.hc_ffn, pre_n, blk.ffn_norm_w, want_f32=True, want_perm=True,
                                                                 out={"yqp": self.xqp_out[d]})
        self._stamp(d, L, 8)
        scores = F.linear(xf, moe.gate_w)
        eid, wt = self.eid8[d], self.wt8[d]
        gate_topk(scores, moe.gate_bias, moe.gate_temp, moe.topk, moe.route_scale, moe.score_func, moe.norm_topk_prob and moe.topk > 1, eid=eid, wt=wt)
        if self.route_log:  # telemetry: keep this layer's routing (DSV41_ROUTE_LOG=1)
            self.route_eid[L].copy_(eid)
            self.route_wt[L].copy_(wt)
        self._stamp(d, L, 9)
        # routing + activation to every peer, then the flags
        if not self.dry:
            if self.dma_route:
                targets = [self.partner[d], self.relay_of[d]] if self.relay else [p for p in self.devs if p != d]
                for p in targets:
                    memcpy_async(self.inbox[p], self.outbox[d], d)
                p2p_signal(self.sig_route[L], self.seqno[d], d)
            else:
                p2p_multicast(self.sig_inbox[d], self.outbox[d], self.sig_route[L], self.seqno[d], d, counter=self.mcast_counter[d])
        self._stamp(d, L, 2)
        xqp = self.xqp_out[d]
        # own shard and, on a second stream, the shared expert, while the peers work
        main = torch.cuda.current_stream(d)
        side2 = self._side_stream2(d)
        side2.wait_stream(main)
        with torch.cuda.stream(side2):
            ys = self._shared_expert(moe, xq)
        y_loc = self._experts_shard(d, xqp, eid, wt, moe)
        if self.bf16_part:
            p2p_sum_rows(self.part_out[d], y_loc, d, groups=self.B)
            self.part_in[d][self.idx[d]].copy_(self.part_out[d])
        else:
            p2p_sum_rows(self.part_in[d][self.idx[d]], y_loc, d, groups=self.B)
        main.wait_stream(side2)
        self._stamp(d, L, 3)
        # wait for the peers' partials (own slot is raised by a local signal so the whole row can be waited on)
        self._signal(self._own_part_ptr(L, d), d)
        self._wait(self.flag_part[d][L], d)
        self._stamp(d, L, 4)
        self._hc_join(side, d)
        hc_post2_(None, h, post, comb, y2=self.part_in[d], ys=ys, y2_sum_first=True)
        self._stamp(d, L, 5)
        return pre_out

    def _own_part_ptr(self, L, d):
        return self._ptrs[("own", L, d)]

    def _side_stream2(self, d):
        ss = getattr(self, "_side2", None)
        if ss is None:
            ss = self._side2 = {}
        if d not in ss:
            ss[d] = torch.cuda.Stream(d)
        return ss[d]

    def _shared_expert(self, moe, xq):
        from .w8 import linear_w
        gu_s = linear_w(xq, moe.sh_w13)
        hs = swiglu_quant(gu_s, None, moe.inter, moe.swiglu_limit)
        return linear_w(hs, moe.sh_w2)

    def _peer_layer(self, blk: Block, d):
        L = blk.layer_id
        o = blk.device
        role = None
        if self.relay:
            a = self.relay_of[o]
            role = "relay" if d == a else "leaf" if d == self.partner[a] else "partner"
        self._wait(self.flag_route[d][L : L + 1], d)
        self._stamp(d, L, 6)
        if role == "relay" and not self.dry:  # forward the route packet to the leaf over NVLink
            memcpy_async(self.inbox[self.partner[d]], self.inbox[d], d)
            p2p_signal(self.sig_fwd[L], self.seqno[d], d)
        y = self._experts_shard(d, self.inbox_x[d], self.inbox_eid[d], self.inbox_wt[d], blk.ffn)
        if self.dma_part:
            p2p_sum_rows(self.part_out[d], y, d, groups=self.B)
            if role == "leaf":  # into the relay partner, which adds it to its own partial
                if self.bf16_part:
                    self.part_out16[d].copy_(self.part_out[d])
                memcpy_async(self.part_relay[a], self.part_out16[d], d)
                self._signal(self.sig_relay[L], d)
                self._stamp(d, L, 7)
                return
            if role == "relay":
                self._wait(self.flag_relay[d][L : L + 1], d)
                self.part_out[d].add_(self.part_relay[d])
            if self.bf16_part:
                self.part_out16[d].copy_(self.part_out[d])
            memcpy_async(self.part_in[o][self.idx[d]], self.part_out16[d], d)
        else:
            p2p_sum_rows(self.part_in[o][self.idx[d]], y, d, groups=self.B)  # straight into the owner's inbox row
        self._signal(self.sig_part[(L, d)], d)
        self._stamp(d, L, 7)

    def token_begin(self, d):
        hc = self.cfg["hc_mult"]
        if d == self.devs[0]:
            with torch.cuda.device(d):
                self.hop_h[d].copy_(F.embedding(self.tok, self.m.embed).unsqueeze(2).repeat(1, 1, hc, 1))
                self.hop_pre[d].copy_(self.pre_identity[d])
        self._pre = {dd: self.hop_pre[dd] for dd in self.devs}
        self.kv_owner = -1
        self.index_owner = -1

    def layer_section(self, L, d):
        """Layer L's share of the token on device d (owner or peer section)."""
        blocks = self.m.blocks
        blk = blocks[L]
        h = self.hop_h[d]
        with torch.cuda.device(d):
            if blk.device == d:
                if L == self.first_layer[d] and self.idx[d] > 0:
                    self._wait(self.flag_hop[d][L : L + 1], d)
                pre = self._owner_layer(blk, d, h, self._pre[d])
                self._pre[d] = pre
                if L == self.last_layer[d]:
                    if self.idx[d] + 1 < self.nd:
                        nxt = blocks[L + 1].device
                        p2p_copy(self.hop_h[nxt], h, d)
                        p2p_copy(self.hop_pre[nxt], pre, d)
                        p2p_copy(self.topk_buf[nxt], self.topk_buf[d], d)
                        p2p_copy(self.cand_buf[nxt], self.cand_buf[d], d)
                        self._signal(self.sig_hop[L], d)
                    else:
                        hh = _hc_pre(h, pre.view(self.B, 1, -1))[:, -1]
                        hh = rmsnorm(hh, self.m.norm_w, self.cfg["norm_eps"])
                        self.logits.copy_(F.linear(hh, self.m.head).float())
            else:
                self._peer_layer(blk, d)
        # bookkeeping of the source layers (attention2 on later owners reads these; same on every device)
        if blk.attn.is_kv_source:
            self.kv_owner = L
        if blk.attn.indexer is not None and blk.attn.indexer.owns_k:
            self.index_owner = L

    def token_end(self, d):
        with torch.cuda.device(d):
            p2p_seq_bump(self.seqno[d], d)

    def token_graph(self, d):
        """The whole token on device d (owner and peer sections in layer order)."""
        self.token_begin(d)
        for L in range(len(self.m.blocks)):
            self.layer_section(L, d)
        self.token_end(d)

    # ------------------------------------------------------------------ driver
    def capture(self):
        # 1) dry pass without waits: compiles / loads every kernel on every device (a module load while the device
        #    spins in a wait deadlocks the host), 2) two real passes with the devices interleaved per layer,
        # 3) one graph per device
        # Graph decode uses bounded static candidate storage. Keep the
        # model's logical context limit unchanged, but clamp only the
        # capture-time shape metadata so a 1M-context model does not make
        # temporary graph workspaces explode.
        _logical_max_seq = self.m.args.max_seq_len
        _graph_limit = min(_logical_max_seq, int(os.environ.get("DSV41_EP_GRAPH_TOKENS", "131072")))
        self.graph_token_limit = _graph_limit
        self.m.args.max_seq_len = _graph_limit

        # Capture against compact, dedicated RoPE storage. Capturing pointers
        # into the full 256K-1M tables triggers an illegal access in the
        # Triton decode RoPE kernel on sm80 even though position zero is used.
        # Keep these clones alive because the graph owns their pointers.
        self._graph_rope_tables = []
        _rope_restore = []
        for blk in self.m.blocks:
            attn = blk.attn
            old_cos, old_sin = attn.cos, attn.sin
            graph_cos = old_cos[:_graph_limit].clone()
            graph_sin = old_sin[:_graph_limit].clone()
            _rope_restore.append((attn, old_cos, old_sin))
            attn.cos, attn.sin = graph_cos, graph_sin
            if attn.indexer is not None:
                attn.indexer.cos, attn.indexer.sin = graph_cos, graph_sin
            self._graph_rope_tables.append((graph_cos, graph_sin))
        self.dry = True
        for d in self.devs:
            with torch.cuda.stream(self.streams[d]):
                self.token_graph(d)
            torch.cuda.synchronize(d)
        self.dry = False
        self.m.args.max_seq_len = _logical_max_seq
        for d in self.devs:
            self.seqno[d].fill_(1)
            self.flag_route[d].zero_()
            self.flag_part[d].zero_()
            self.flag_hop[d].zero_()
            self.flag_relay[d].zero_()
        for _ in range(2):
            self._eager_token()
            for d in self.devs:
                torch.cuda.synchronize(d)
        if not self.use_graphs:
            return
        for d in self.devs:
            with torch.cuda.device(d):
                s = self.streams[d]
                torch.cuda.synchronize(d)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s, capture_error_mode="thread_local"):
                    self.token_graph(d)
                self.graphs[d] = g
        for d in self.devs:
            torch.cuda.synchronize(d)
        self._graph_cache_signature = self._cache_signature()
        for attn, old_cos, old_sin in _rope_restore:
            attn.cos, attn.sin = old_cos, old_sin
            if attn.indexer is not None:
                attn.indexer.cos, attn.indexer.sin = old_cos, old_sin

    def _eager_token(self):
        """One token without graphs, the devices interleaved per layer on their own streams."""
        for d in self.devs:
            with torch.cuda.stream(self.streams[d]):
                self.token_begin(d)
        for L in range(len(self.m.blocks)):
            for d in self.devs:
                with torch.cuda.stream(self.streams[d]):
                    self.layer_section(L, d)
        for d in self.devs:
            with torch.cuda.stream(self.streams[d]):
                self.token_end(d)

    @torch.inference_mode()
    def step(self, token, pos, seq=None, pmax=None) -> torch.Tensor:
        tok, p, sq = self.set_rows(token, pos, seq, pmax)
        self._engram_rows(tok, p, sq)

        # Debug dynamic-cache growth vs captured CUDA Graph pointers.
        # Switch to eager execution only around/after the 32K boundary.
        _debug_eager = False

        if torch.is_tensor(p):
            _decode_pos = int(p.max().item())
        elif isinstance(p, (list, tuple)):
            _decode_pos = max(int(v) for v in p)
        else:
            _decode_pos = int(p)
        if _decode_pos >= getattr(self, "graph_token_limit", self.m.args.max_seq_len):
            _debug_eager = True

        if os.environ.get(
            "DSV41_DEBUG_EP_EAGER_BOUNDARY",
            "0",
        ) == "1":
            if torch.is_tensor(p):
                _debug_pos = int(p.flatten()[0].item())
            elif isinstance(p, (list, tuple)):
                _debug_pos = int(p[0])
            else:
                _debug_pos = int(p)

            _debug_eager = _debug_pos >= 32760

            if _debug_eager and (
                _debug_pos <= 32780
                or _debug_pos % 256 == 0
            ):
                print(
                    f"[ep-boundary] pos={_debug_pos} "
                    f"mode=EAGER",
                    flush=True,
                )

        if (
            self.use_graphs
            and self.graphs
            and not _debug_eager
        ):
            for d in self.devs:
                self.graphs[d].replay()
        else:
            self._eager_token()

        torch.cuda.synchronize(self.devs[-1])
        return self.logits


def trace_report(rt: "EPRuntime") -> str:
    """Average per-layer timeline (us) from the DSV41_EP_TRACE stamps of the last token."""
    if rt.trace is None:
        return "no trace"
    tr = {d: rt.trace[d].cpu() for d in rt.devs}
    rows = []
    agg = {}
    for L, blk in enumerate(rt.m.blocks):
        o = blk.device
        t = tr[o][L]
        peers = [d for d in rt.devs if d != o]
        pc = [tr[d][L, 7].item() - tr[d][L, 6].item() for d in peers]  # peer compute + push (its own clock)
        parts = {"attn+hc": t[1] - t[0], "  hc_sub2": t[8] - t[1], "  gate+topk": t[9] - t[8], "  multicast": t[2] - t[9],
                 "own experts || shared": t[3] - t[2], "wait partials": t[4] - t[3],
                 "hc_post": t[5] - t[4], "layer": t[5] - t[0], "peer compute+push (max)": max(pc), "peer compute+push (mean)": sum(pc) / len(pc)}
        for k, v in parts.items():
            agg.setdefault(k, []).append(float(v))
    return "\n".join(f"  {k:28s} {sum(v) / len(v) / 1000:7.1f} us" for k, v in agg.items())
