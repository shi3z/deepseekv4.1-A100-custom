"""Round the split path exactly where the unsplit one does, so the two are bit-identical.

`moe_fn` returns bf16 and the caller casts it back to fp32, so the shipped path rounds the routed
sum once. A split call would round two partial sums instead, which measured 0.66 % on the logit sum
after forty layers -- not wrong, but it makes "this changes nothing" untestable. With the sum kept
in fp32 inside the kernels (cb3_moe SPLIT) and one bf16 round here, the two paths agree bit for bit.
"""
import os
import shutil

p = os.path.expanduser("~/dsv41-spark/work/engine/model.py")
bak = p + ".pre-split.bak"
s = open(p).read()
a = """            routed = self.moe_fn(y, slots, weights, arena, a.swiglu_limit).float()
            wait_for_misses()
            routed = routed + self.moe_fn(y, pend, weights, arena, a.swiglu_limit).float()"""
b = """            routed = self.moe_fn(y, slots, weights, arena, a.swiglu_limit).float()
            wait_for_misses()
            routed = routed + self.moe_fn(y, pend, weights, arena, a.swiglu_limit).float()
            # the unsplit path rounds the routed sum once, on the way out of moe_fn
            routed = routed.to(torch.bfloat16).float()"""
assert a in s, "split branch not found"
open(p, "w").write(s.replace(a, b, 1))
print("patched engine/model.py: one bf16 round, where the unsplit path has it")
