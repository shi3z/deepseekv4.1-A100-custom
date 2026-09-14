"""Clear the gate event once the layer's loads have landed.

`GATE` is the event a split pass records so the H2D copies can overlap the resident MoE instead of
queueing behind it. It was left set after prefill, and the decode step is captured into a CUDA
graph: waiting on an event recorded outside the capture is illegal, so every decode request came
back with `operation failed due to a previous error during capture`. The event is only meaningful
for the loads of the call that recorded it, so drop it when they are done.
"""
import os

p = os.path.expanduser("~/dsv41-spark/work/engine/experts.py")
s = open(p).read()
a = """            def _wait():
                for f in futs:
                    f.result()
                self.stats["load_s"] += time.perf_counter() - t0"""
b = """            def _wait():
                for f in futs:
                    f.result()
                _cs.GATE = None      # only valid for these loads; decode is graph-captured
                self.stats["load_s"] += time.perf_counter() - t0"""
assert a in s, "_wait not found"
open(p, "w").write(s.replace(a, b, 1))
print("patched engine/experts.py: the gate event is cleared after the loads")
