"""Oracle prefetch: what would a perfect future-routing predictor be worth?

Resident-first took a 512-token prefill chunk from 21.5 s to 20.0 s and stopped there, with the
disk idle 22 % of the time. The remaining idle is structural: layer L+1's router reads layer L's
output, so while attention, engram and the dense parts of layer L+1 run -- 5.9 s of the 7.1 s of
GPU work -- nothing knows which experts to fetch yet.

Before building a predictor it is worth knowing the ceiling. Prefill is deterministic here (two
runs agree bit for bit), so a first pass can record the routing and a second can be handed it as an
oracle. That is an upper bound no predictor can beat, and it costs no model work to measure.

    model.oracle = {L: indices}   # from a recorded pass
    for L in layers:
        prefetch(oracle[L + 1])   # <- the thing a predictor would have to guess
        attention / engram / dense / moe for L
"""
import os
import shutil

p = os.path.expanduser("~/dsv41-spark/work/engine/model.py")
bak = p + ".pre-oracle.bak"
if not os.path.exists(bak):
    shutil.copy2(p, bak)
s = open(bak).read()

a = """    def _tap(self, name, L, t):"""
b = """    #: {layer: top-k expert ids} from a recorded pass; set to prefetch one layer ahead
    oracle = None
    _pf = None

    def _tap(self, name, L, t):"""
assert a in s, "tap not found"
s = s.replace(a, b, 1)

a = """            freqs = self.freqs_c if w.ratio else self.freqs_w
            h, pre_mix = self.block(h, pre_mix, w, L, S, sh, self.c.win[L], freqs, prefill, self.store,
                                    self.store.arena, a.n_routed_experts)
            self._tap("h", L, h); self._tap("pre_mix", L, pre_mix)"""
b = """            if self.oracle is not None and prefill and (L + 1) in self.oracle:
                # start the next layer's experts now, while this layer's attention and dense parts
                # run. A predictor would have to produce these ids; the oracle just knows them.
                r = self.store.resolve(L + 1, self.oracle[L + 1], True, split=True)
                self._pf = (L + 1, r[2]) if isinstance(r, tuple) else None
            freqs = self.freqs_c if w.ratio else self.freqs_w
            h, pre_mix = self.block(h, pre_mix, w, L, S, sh, self.c.win[L], freqs, prefill, self.store,
                                    self.store.arena, a.n_routed_experts)
            self._tap("h", L, h); self._tap("pre_mix", L, pre_mix)"""
assert a in s, "layer loop not found"
s = s.replace(a, b, 1)

a = """        res = store.resolve(L, indices, prefill, split=SPLIT_MOE and prefill)"""
b = """        if self._pf is not None and self._pf[0] == L:
            self._pf[1]()          # the prefetch for this layer: let it land before we route
            self._pf = None
        res = store.resolve(L, indices, prefill, split=SPLIT_MOE and prefill)"""
assert a in s, "moe resolve call not found"
s = s.replace(a, b, 1)
open(p, "w").write(s)
print("patched engine/model.py: model.oracle prefetches one layer ahead")
