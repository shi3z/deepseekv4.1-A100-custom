"""Resident-first is a prefill-only path: the decode step is captured into a CUDA graph.

The engine graph-captures the decode step (that is what `build_routing_small`'s static shapes are
for). Splitting the MoE call needs a host-side wait on the loader threads and a CUDA event
recorded mid-step, neither of which is legal during capture -- every decode request came back with
`operation failed due to a previous error during capture`. Prefill is not captured, which is also
where the win is: 21.5 s -> 20.0 s on a 512-token chunk.
"""
import os

p = os.path.expanduser("~/dsv41-spark/work/engine/model.py")
s = open(p).read()
a = "        res = store.resolve(L, indices, prefill, split=SPLIT_MOE)"
b = "        res = store.resolve(L, indices, prefill, split=SPLIT_MOE and prefill)"
assert a in s, "split call site not found"
open(p, "w").write(s.replace(a, b, 1))
print("patched engine/model.py: split only during prefill")
