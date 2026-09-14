"""Classify the residual demand misses at depth=1.

At depth=1 the oracle prefetches layer L's experts during layer L-1 and the wait for them runs
before layer L routes, so in principle every expert should be resident when the MoE starts.
Measured, 95.8 % are. This records, for every expert the prefetch touched, when it was submitted,
when its read started and finished, when it was installed, when it was evicted, and when it was
first needed -- which is enough to say why the other 4.2 % were not there:

  A never submitted        (layer 0 has no previous layer; anything outside the oracle)
  B submitted, read not finished
  C read finished, not installed
  D installed, then evicted by the transient ring before it was needed
  E delayed behind a duplicate / competing load
  F anything else

Opt-in: DSV41_MISS_TRACE=1. The dict is only touched when it is on.
"""
import os

p = os.path.expanduser("~/dsv41-spark/work/engine/experts.py")
s = open(p).read()

a = """        self.io = {"demand_hits": 0, "demand_loads": 0, "spec_hits": 0, "spec_loads": 0,
                   "dup_loads": 0, "seen": set()}"""
b = """        self.io = {"demand_hits": 0, "demand_loads": 0, "spec_hits": 0, "spec_loads": 0,
                   "dup_loads": 0, "seen": set()}
        # a100-vq: why a demand miss was a miss (DSV41_MISS_TRACE=1); see patch_miss_trace.py
        self.trace = {} if os.environ.get("DSV41_MISS_TRACE") == "1" else None
        self.miss_why = {}"""
assert a in s, "io counters not found"
s = s.replace(a, b, 1)

# ---- submit timestamps, and the classification of a demand miss
a = """            io = self.io
            io["spec_hits" if spec else "demand_hits"] += len(slot_of) - len(to_load)
            io["spec_loads" if spec else "demand_loads"] += len(to_load)"""
b = """            io = self.io
            io["spec_hits" if spec else "demand_hits"] += len(slot_of) - len(to_load)
            io["spec_loads" if spec else "demand_loads"] += len(to_load)
            if self.trace is not None:
                now = time.perf_counter()
                if spec:
                    for k, _s2 in to_load:
                        self.trace.setdefault(k, {})["submit"] = now
                else:
                    for k, _s2 in to_load:
                        r = self.trace.get(k)
                        if r is None or "submit" not in r:
                            why = "A_never_submitted"
                        elif "io_done" not in r:
                            why = "B_read_unfinished"
                        elif "install" not in r:
                            why = "C_install_pending"
                        elif r.get("evict", 0.0) > r.get("install", 0.0):
                            why = "D_evicted"
                        elif r.get("dup"):
                            why = "E_duplicate_delay"
                        else:
                            why = "F_other"
                        self.miss_why[why] = self.miss_why.get(why, 0) + 1
                        r = self.trace.setdefault(k, {})
                        r["reloaded"] = r.get("reloaded", 0) + 1"""
assert a in s, "io counter block not found"
s = s.replace(a, b, 1)

# ---- eviction timestamps
a = """        old = self.slot_key.pop(slot, None)
        if old is not None:
            self.transient_map.pop(old, None)"""
b = """        old = self.slot_key.pop(slot, None)
        if old is not None:
            self.transient_map.pop(old, None)
            if self.trace is not None and old in self.trace:
                self.trace[old]["evict"] = time.perf_counter()"""
assert a in s, "transient evict not found"
s = s.replace(a, b, 1)

# ---- io / install timestamps
a = """    def _load_into_slot(self, key: tuple, slot: int, prefix: str | None = None):"""
b = """    def _load_into_slot(self, key: tuple, slot: int, prefix: str | None = None):
        if self.trace is not None:
            r = self.trace.setdefault(key, {})
            r["io_start"] = time.perf_counter()
            try:
                return self._load_into_slot_inner(key, slot, prefix)
            finally:
                r["io_done"] = r["install"] = time.perf_counter()
        return self._load_into_slot_inner(key, slot, prefix)

    def _load_into_slot_inner(self, key: tuple, slot: int, prefix: str | None = None):"""
assert a in s, "_load_into_slot not found"
s = s.replace(a, b, 1)
open(p, "w").write(s)
print("patched engine/experts.py: miss trace")
