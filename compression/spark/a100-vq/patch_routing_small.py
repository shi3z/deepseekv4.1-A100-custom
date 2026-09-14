"""Let build_routing_small survive a masked pass.

It sorts the (token, k) pairs by slot and writes each run into its own BM-wide block, on the
assumption that no slot owns more than BM pairs -- true at decode size. A masked pass breaks it:
every pair whose expert is still loading carries slot -1, so they all sort into ONE run, and the
scatter walks past that block into the next real slot's rows. The kernels already skip a block
whose slot is -1, so the fix is not to write those pairs at all; with no -1 present the scatter is
element for element what it was.
"""
import os
import shutil

p = os.path.expanduser("~/dsv41-spark/work/tools/fp4_moe.py")
bak = p + ".pre-split.bak"
if not os.path.exists(bak):
    shutil.copy2(p, bak)
s = open(bak).read()

a = """    block_pair = torch.full((P * BM,), -1, dtype=torch.int32, device=flat.device)
    block_pair[blk * BM + rank] = order.to(torch.int32)
    block_slot = torch.full((P,), -1, dtype=torch.int32, device=flat.device)
    block_slot[blk] = ss"""
b = """    block_pair = torch.full((P * BM,), -1, dtype=torch.int32, device=flat.device)
    # A masked pass (a100-vq/patch_engine_split.py runs the resident experts while the misses are
    # still loading) marks a pair's slot -1, and those all sort into one run, which is the one case
    # that breaks "no slot owns more than BM pairs" and scatters past the block. The kernels skip a
    # -1 block anyway, so leave those pairs out; with no -1 present this is the original scatter.
    keep = ss >= 0
    block_pair[(blk * BM + rank)[keep]] = order.to(torch.int32)[keep]
    block_slot = torch.full((P,), -1, dtype=torch.int32, device=flat.device)
    block_slot[blk[keep]] = ss[keep]"""
assert a in s, "build_routing_small scatter not found"
open(p, "w").write(s.replace(a, b, 1))
print("patched tools/fp4_moe.py: build_routing_small skips masked pairs")
