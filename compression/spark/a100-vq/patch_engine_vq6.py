"""Add EXPERT_FORMAT=vq6 to the engine, on top of the vq12 patch. Idempotent; keeps a .pre-vq6.bak."""
import os, shutil
p = os.path.expanduser("~/dsv41-spark/work/engine/v41_engine.py")
bak = p + ".pre-vq6.bak"
if not os.path.exists(bak):
    shutil.copy2(p, bak)
s = open(bak).read()

anchor = '        if self.expert_format == "vq12":'
ins = '''        if self.expert_format == "vq6":
            # a100-vq/vq6: dim-2 vector quantisation, 64 entries, the same 3 bit/weight and the
            # same slot geometry. The codebook is small enough that a warp's gather stays inside a
            # sector, which is what VQ12's 4096-entry table costs on this GPU.
            import sys as _sys
            _sys.path.insert(0, os.path.expanduser("~/dsv41-spark/a100-vq"))
            import vq6_moe as V6M
            from vq6 import VQ6
            cb3_cls = V6M.VQ6Arena
            self._vq6 = VQ6(os.environ.get("DSV41_VQ6_CODEBOOK",
                                           os.path.expanduser("~/dsv41-spark/a100-vq/vq2_3.npz")),
                            device)
            cb3_moe_fn = V6M.moe_forward_vq6
            self.kernel = "triton-vq6"
            log("using the Triton VQ6 (dim-2 vector codebook) MoE kernel for the routed experts")
''' + anchor
assert anchor in s, "the vq12 branch is missing: run patch_engine_vq12.py first"
s = s.replace(anchor, ins, 1)

a2 = '''            if self.expert_format == "vq12":
                return a.attach(self._vq12)'''
ins2 = '''            if self.expert_format == "vq6":
                return a.attach(self._vq6)
            if self.expert_format == "vq12":
                return a.attach(self._vq12)'''
assert a2 in s, "make_expert_arena's vq12 line not found"
s = s.replace(a2, ins2, 1)
s = s.replace('assert self.expert_format in ("fp4", "cb3", "vq12", "tiered"), self.expert_format',
              'assert self.expert_format in ("fp4", "cb3", "vq12", "vq6", "tiered"), self.expert_format', 1)
s = s.replace('choices=["fp4", "cb3", "vq12", "tiered"]', 'choices=["fp4", "cb3", "vq12", "vq6", "tiered"]', 1)
open(p, "w").write(s)
print("engine patched for EXPERT_FORMAT=vq6")
