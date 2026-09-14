"""Make the masked-pair fix capture-safe.

The first version dropped the -1 pairs with boolean indexing, `x[keep]`, whose output size depends
on the data -- so it needs a host sync and the decode step, which the engine captures into a CUDA
graph, died with `operation not permitted when stream is capturing`. Same effect with static
shapes: send every masked pair to one spare row past the end of the table, which no block reads.

`block_slot` needs nothing: the -1 pairs sort into block 0, and writing -1 there is exactly the
"skip this block" marker the kernels already honour.
"""
import os

p = os.path.expanduser("~/dsv41-spark/work/tools/fp4_moe.py")
s = open(p).read()
a = """    block_pair = torch.full((P * BM,), -1, dtype=torch.int32, device=flat.device)
    # A masked pass (a100-vq/patch_engine_split.py runs the resident experts while the misses are
    # still loading) marks a pair's slot -1, and those all sort into one run, which is the one case
    # that breaks "no slot owns more than BM pairs" and scatters past the block. The kernels skip a
    # -1 block anyway, so leave those pairs out; with no -1 present this is the original scatter.
    keep = ss >= 0
    block_pair[(blk * BM + rank)[keep]] = order.to(torch.int32)[keep]
    block_slot = torch.full((P,), -1, dtype=torch.int32, device=flat.device)
    block_slot[blk[keep]] = ss[keep]"""
b = """    # One spare row past the end: a masked pass (a100-vq/patch_engine_split.py runs the resident
    # experts while the misses are still loading) marks a pair's slot -1, and those all sort into a
    # single run, the one case that breaks "no slot owns more than BM pairs" and would scatter into
    # the next real slot's rows. Sending them to the spare row keeps every shape static, which the
    # graph-captured decode step requires; with no -1 present this is the original scatter.
    block_pair = torch.full((P * BM + 1,), -1, dtype=torch.int32, device=flat.device)
    valid = ss >= 0
    dump = torch.full_like(blk, P * BM)
    block_pair[torch.where(valid, blk * BM + rank, dump)] = torch.where(
        valid, order.to(torch.int32), torch.full_like(order, -1, dtype=torch.int32))
    block_slot = torch.full((P,), -1, dtype=torch.int32, device=flat.device)
    block_slot[blk] = ss"""
assert a in s, "the first fix is not in place"
open(p, "w").write(s.replace(a, b, 1))
print("patched tools/fp4_moe.py: static-shape masked scatter")
