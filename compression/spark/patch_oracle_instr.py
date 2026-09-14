"""Instrument the oracle experiment: separate speculative I/O from demand I/O, and count both.

Everything here is off unless `model.oracle` is set, and the counters are plain integers, so the
shipped path is unchanged. What it adds:

  * speculative loads go to their own thread pool, so a layer's OWN misses are never queued behind
    a prefetch for a later layer (demand I/O outranks speculative I/O);
  * resolve records, per call, whether it was speculative and how many experts it hit or loaded,
    which is what separates "depth helped" from "depth read more bytes";
  * every (layer, expert) actually read is remembered, so a record read twice -- prefetched,
    evicted, read again -- shows up as duplicate rather than hiding inside the total.
"""
import os

p = os.path.expanduser("~/dsv41-spark/work/engine/experts.py")
s = open(p).read()

a = """        self.pool = ThreadPoolExecutor(io_threads, thread_name_prefix="expert-io")"""
b = """        self.pool = ThreadPoolExecutor(io_threads, thread_name_prefix="expert-io")
        # a100-vq: speculative (oracle / predictor) loads never share the demand queue
        self.spec_pool = ThreadPoolExecutor(max(2, io_threads // 2), thread_name_prefix="expert-spec")
        self.io = {"demand_hits": 0, "demand_loads": 0, "spec_hits": 0, "spec_loads": 0,
                   "dup_loads": 0, "seen": set()}"""
assert a in s, "pool line not found"
s = s.replace(a, b, 1)

a = "    def resolve(self, layer: int, experts: torch.Tensor, prefill: bool, split: bool = False) -> torch.Tensor:"
b = "    def resolve(self, layer: int, experts: torch.Tensor, prefill: bool, split: bool = False,\n                spec: bool = False) -> torch.Tensor:"
assert a in s, "resolve signature not found"
s = s.replace(a, b, 1)

a = """            t0 = time.perf_counter()
            # everything queued so far -- i.e. through the previous layer. The slots below were"""
b = """            io = self.io
            io["spec_hits" if spec else "demand_hits"] += len(slot_of) - len(to_load)
            io["spec_loads" if spec else "demand_loads"] += len(to_load)
            for k, _s in to_load:
                if k in io["seen"]:
                    io["dup_loads"] += 1
                io["seen"].add(k)
            t0 = time.perf_counter()
            # everything queued so far -- i.e. through the previous layer. The slots below were"""
assert a in s, "split branch not found"
s = s.replace(a, b, 1)

a = "            futs = [self.pool.submit(self._load_into_slot, *ks) for ks in to_load]"
b = "            pool = self.spec_pool if spec else self.pool\n            futs = [pool.submit(self._load_into_slot, *ks) for ks in to_load]"
assert a in s, "submit not found"
s = s.replace(a, b, 1)
open(p, "w").write(s)
print("patched engine/experts.py: spec pool + io counters")

p = os.path.expanduser("~/dsv41-spark/work/engine/model.py")
s = open(p).read()
a = "                        r = self.store.resolve(nxt, self.oracle[nxt], True, split=True)"
b = "                        r = self.store.resolve(nxt, self.oracle[nxt], True, split=True, spec=True)"
assert a in s, "oracle resolve not found"
s = s.replace(a, b, 1)

a = """            freqs = self.freqs_c if w.ratio else self.freqs_w
            h, pre_mix = self.block(h, pre_mix, w, L, S, sh, self.c.win[L], freqs, prefill, self.store,
                                    self.store.arena, a.n_routed_experts)"""
b = """            freqs = self.freqs_c if w.ratio else self.freqs_w
            _tl = time.perf_counter()
            _tm = self.stats.get("moe_s", 0.0)
            h, pre_mix = self.block(h, pre_mix, w, L, S, sh, self.c.win[L], freqs, prefill, self.store,
                                    self.store.arena, a.n_routed_experts)
            if self.oracle is not None or LAYER_TIMING:
                d = self.stats.setdefault("layer_ms", {})
                tot = (time.perf_counter() - _tl) * 1e3
                moe = (self.stats.get("moe_s", 0.0) - _tm) * 1e3
                e = d.setdefault(L, [0.0, 0.0, 0])
                e[0] += tot - moe      # attention + engram + dense
                e[1] += moe            # routing, waiting for experts, and the MoE itself
                e[2] += 1"""
assert a in s, "layer loop not found"
s = s.replace(a, b, 1)
a = "ORACLE_DEPTH = int(os.environ.get(\"DSV41_ORACLE_DEPTH\", \"1\"))"
b = a + "\nLAYER_TIMING = os.environ.get(\"DSV41_LAYER_TIMING\") == \"1\""
assert a in s
s = s.replace(a, b, 1)
open(p, "w").write(s)
print("patched engine/model.py: per-layer non-MoE / MoE timing")
