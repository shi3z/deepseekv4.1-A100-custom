"""Add EXPERT_FORMAT=vq12 to the engine. Idempotent; keeps a .pre-vq12.bak."""
import os, shutil
p = os.path.expanduser("~/dsv41-spark/work/engine/v41_engine.py")
bak = p + ".pre-vq12.bak"
if not os.path.exists(bak):
    shutil.copy2(p, bak)
s = open(bak).read()

a1 = '        if self.expert_format == "cb3":\n            import cb3_moe as C3\n            from engine.codebook_sim import CodebookSim'
ins1 = '''        if self.expert_format == "vq12":
            # a100-vq/vq12: CB3's slot geometry and rate, a dim-4 vector codebook instead of a
            # per-row scalar one. Same bytes, half the quantisation damage (+4.23 % wikitext PPL
            # against the FP4 checkpoint where CB3 costs +8.54 %).
            import sys as _sys
            _sys.path.insert(0, os.path.expanduser("~/dsv41-spark/a100-vq"))
            import vq12_moe as VQM
            from vq12 import VQ12
            cb3_cls = VQM.VQ12Arena
            self._vq12 = VQ12(os.environ.get("DSV41_VQ12_CODEBOOK",
                                             os.path.expanduser("~/dsv41-spark/a100-vq/vq_3.0.npz")),
                              device)
            cb3_moe_fn = VQM.moe_forward_vq
            self.kernel = "triton-vq12"
            log("using the Triton VQ12 (dim-4 vector codebook) MoE kernel for the routed experts")
''' + a1
assert a1 in s, "cb3 branch not found"
s = s.replace(a1, ins1, 1)

a2 = '''            if cb3_cls is None:
                return fp4_arena_cls(n_slots, device)
            a = cb3_cls(n_slots, device)
            a.sim = self._cb3_sim
            return a'''
ins2 = '''            if cb3_cls is None:
                return fp4_arena_cls(n_slots, device)
            a = cb3_cls(n_slots, device)
            if self.expert_format == "vq12":
                return a.attach(self._vq12)
            a.sim = self._cb3_sim
            return a'''
assert a2 in s, "make_expert_arena not found"
s = s.replace(a2, ins2, 1)
s = s.replace('assert self.expert_format in ("fp4", "cb3", "tiered"), self.expert_format',
              'assert self.expert_format in ("fp4", "cb3", "vq12", "tiered"), self.expert_format', 1)
s = s.replace('choices=["fp4", "cb3", "tiered"]', 'choices=["fp4", "cb3", "vq12", "tiered"]', 1)
open(p, "w").write(s)
print("engine patched for EXPERT_FORMAT=vq12")
