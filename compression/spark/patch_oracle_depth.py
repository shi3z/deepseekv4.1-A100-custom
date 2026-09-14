"""How far ahead does the prefetch have to see?

One layer of lookahead took a 512-token prefill chunk from 20.1 s to 16.6 s and the disk from
3.51 to 4.55 GB/s -- still short of the 5.6 GB/s it sustains when kept busy, so the queue is
running dry between layers. DSV41_ORACLE_DEPTH says how many layers ahead to start; that is the
number a real predictor would have to reach.
"""
import os

p = os.path.expanduser("~/dsv41-spark/work/engine/model.py")
s = open(p).read()
a = """            if self.oracle is not None and prefill and (L + 1) in self.oracle:
                # start the next layer's experts now, while this layer's attention and dense parts
                # run. A predictor would have to produce these ids; the oracle just knows them.
                r = self.store.resolve(L + 1, self.oracle[L + 1], True, split=True)
                self._pf = (L + 1, r[2]) if isinstance(r, tuple) else None"""
b = """            if self.oracle is not None and prefill:
                # start the coming layers' experts now, while this layer's attention and dense
                # parts run. A predictor would have to produce these ids; the oracle knows them.
                for d in range(1, ORACLE_DEPTH + 1):
                    nxt = L + d
                    if nxt in self.oracle and nxt not in self._pf_done:
                        r = self.store.resolve(nxt, self.oracle[nxt], True, split=True)
                        self._pf_done[nxt] = r[2] if isinstance(r, tuple) else None"""
assert a in s, "oracle prefetch block not found"
s = s.replace(a, b, 1)

a = """    oracle = None
    _pf = None"""
b = """    oracle = None
    _pf_done = {}"""
assert a in s
s = s.replace(a, b, 1)

a = """        if self._pf is not None and self._pf[0] == L:
            self._pf[1]()          # the prefetch for this layer: let it land before we route
            self._pf = None"""
b = """        w8 = self._pf_done.pop(L, None)
        if w8 is not None:
            w8()                   # the prefetch for this layer: let it land before we route"""
assert a in s
s = s.replace(a, b, 1)

a = "SPLIT_MOE = os.environ.get(\"DSV41_SPLIT_MOE\") == \"1\""
b = (a + "\n#: layers of lookahead for the oracle prefetch experiment (a100-vq/oracle_probe.py)\n"
     "ORACLE_DEPTH = int(os.environ.get(\"DSV41_ORACLE_DEPTH\", \"1\"))")
assert a in s
s = s.replace(a, b, 1)
open(p, "w").write(s)
print("patched engine/model.py: DSV41_ORACLE_DEPTH layers of lookahead")
