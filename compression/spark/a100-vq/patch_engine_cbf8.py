"""Add EXPERT_FORMAT=cbf8 to the engine, on top of the vq12/vq6 patches. Keeps a .pre-cbf8.bak."""
import os, shutil
p = os.path.expanduser("~/dsv41-spark/work/engine/v41_engine.py")
bak = p + ".pre-cbf8.bak"
if not os.path.exists(bak):
    shutil.copy2(p, bak)
s = open(bak).read()

anchor = '        if self.expert_format == "vq6":'
ins = '''        if self.expert_format == "cbf8":
            # a100-vq/cbf8: CB3's record byte for byte, its eight `cb` bytes read as int8 level
            # codes instead of E2M1 codes. Two prmt fetch an arbitrary fp16 level's low and high
            # byte and two more interleave them, so the codebook stops being the E2M1 grid without
            # the decode ever touching memory -- and `cvt.rn.f16x2.e2m1x2` is no longer needed.
            import sys as _sys
            _sys.path.insert(0, os.path.expanduser("~/dsv41-spark/a100-vq"))
            import cbf8_moe as F8M
            from cbf8 import CBF8
            cb3_cls = F8M.CBF8Arena
            self._cbf8 = CBF8(device)
            cb3_moe_fn = F8M.moe_forward_cbf8
            self.kernel = "triton-cbf8"
            log("using the Triton CBF8 (eight free fp16 levels, register-only) MoE kernel "
                "for the routed experts")
''' + anchor
assert anchor in s, "the vq6 branch is missing: run patch_engine_vq6.py first"
s = s.replace(anchor, ins, 1)

a2 = '''            if self.expert_format == "vq6":
                return a.attach(self._vq6)'''
ins2 = '''            if self.expert_format == "cbf8":
                return a.attach(self._cbf8)
            if self.expert_format == "vq6":
                return a.attach(self._vq6)'''
assert a2 in s
s = s.replace(a2, ins2, 1)
s = s.replace('assert self.expert_format in ("fp4", "cb3", "vq12", "vq6", "tiered"), self.expert_format',
              'assert self.expert_format in ("fp4", "cb3", "vq12", "vq6", "cbf8", "tiered"), self.expert_format', 1)
s = s.replace('choices=["fp4", "cb3", "vq12", "vq6", "tiered"]',
              'choices=["fp4", "cb3", "vq12", "vq6", "cbf8", "tiered"]', 1)
open(p, "w").write(s)
print("engine patched for EXPERT_FORMAT=cbf8")
