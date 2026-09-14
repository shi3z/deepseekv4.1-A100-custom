"""Resident-first MoE: stop waiting for every expert before starting the layer.

Measured on one Spark, a 512-token prefill chunk with the full CB3 store and a 94 GB arena:

    wall 21.2 s      GPU busy 7.1 s      NVMe 70.7 GB at 3.34 GB/s
    the disk is idle 30 % of the time, and reaches 4.78 GB/s while it is busy
    io_threads 12 -> 24 -> 48 changes nothing (21.2 / 21.2 / 21.1 s): the queue is not the limit

Per layer that is [load 122 experts, 0.37 s] then [compute, 0.18 s], strictly serialised, because
`ExpertStore.resolve` ends in `list(self.pool.map(...))` -- every miss has to land before the MoE
starts. Two thirds of the layer's experts are already resident, so two thirds of the compute can
run while the rest arrive, and the shared expert (dense bf16, touches no expert store at all) can
run even earlier.

Opt-in: DSV41_SPLIT_MOE=1. Without it `resolve` keeps its old signature and the engine behaves
exactly as before.
"""
import os
import shutil

W = os.path.expanduser("~/dsv41-spark/work")


def patch(path, edits, tag):
    bak = path + f".pre-{tag}.bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    s = open(bak).read()
    for a, b in edits:
        assert a in s, f"{os.path.basename(path)}: anchor not found:\n{a[:120]}"
        s = s.replace(a, b, 1)
    open(path, "w").write(s)
    print(f"patched {os.path.relpath(path, W)}")


# ---------------------------------------------------------------- the parts buffer must start at 0
# A masked pass writes only the blocks whose slot is >= 0, so the rows it skips have to be zero
# rather than whatever was in the allocation, or the final sum over k picks up garbage.
patch(f"{W}/tools/cb3_moe.py", [(
    "UNPACK_BATCH = int(os.environ.get(\"DSV41_CB3_UNPACK_BATCH\", 32))",
    "UNPACK_BATCH = int(os.environ.get(\"DSV41_CB3_UNPACK_BATCH\", 32))\n"
    "# with DSV41_SPLIT_MOE the routed experts are computed in two masked passes, so the rows a\n"
    "# pass skips must be zero rather than uninitialised (see a100-vq/patch_engine_split.py)\n"
    "SPLIT = os.environ.get(\"DSV41_SPLIT_MOE\") == \"1\"\n"
    "_parts = (lambda *a, **k: torch.zeros(*a, **k)) if SPLIT else (lambda *a, **k: torch.empty(*a, **k))"
)] + [(
    "    parts = torch.empty((P, DIM), dtype=torch.float32, device=dev)",
    "    parts = _parts((P, DIM), dtype=torch.float32, device=dev)"
)] * 4, "split")

patch(f"{W}/engine/experts.py", [(
    """        if to_load:
            t0 = time.perf_counter()
            list(self.pool.map(lambda ks: self._load_into_slot(*ks), to_load))
            self.stats["load_s"] += time.perf_counter() - t0
        self.stats["resolve_s"] += time.perf_counter() - t_res
        return slots""",
    """        if to_load and split:
            # a100-vq: hand the residents back now and the misses as a second mask, so the caller
            # can run the MoE on what the arena already holds while NVMe fills the rest.
            pend = np.full(self.n_experts, -1, dtype=np.int32)
            for (_lyr, e), s in to_load:
                pend[e] = s
                lut[e] = -1
            res_slots = torch.from_numpy(lut[ex.astype(np.intp)]).to(experts.device)
            pend_slots = torch.from_numpy(pend[ex.astype(np.intp)]).to(experts.device)
            t0 = time.perf_counter()
            futs = [self.pool.submit(self._load_into_slot, *ks) for ks in to_load]

            def _wait():
                for f in futs:
                    f.result()
                self.stats["load_s"] += time.perf_counter() - t0

            self.stats["resolve_s"] += time.perf_counter() - t_res
            return res_slots, pend_slots, _wait
        if to_load:
            t0 = time.perf_counter()
            list(self.pool.map(lambda ks: self._load_into_slot(*ks), to_load))
            self.stats["load_s"] += time.perf_counter() - t0
        self.stats["resolve_s"] += time.perf_counter() - t_res
        return slots"""
), (
    "        slots = torch.from_numpy(lut[ex.astype(np.intp)]).to(experts.device)\n"
    "        self.stats[\"host_set_s\"]",
    "        if False:\n            pass\n"
    "        slots = torch.from_numpy(lut[ex.astype(np.intp)]).to(experts.device)\n"
    "        self.stats[\"host_set_s\"]"
)], "split")

# `resolve`'s signature, wherever it is declared
p = f"{W}/engine/experts.py"
s = open(p).read()
import re
m = re.search(r"def resolve\(self, ([^)]*)\)", s)
assert m, "resolve() signature not found"
if "split" not in m.group(1):
    s = s.replace(m.group(0), f"def resolve(self, {m.group(1)}, split: bool = False)", 1)
    open(p, "w").write(s)
    print("resolve() takes split=")

patch(f"{W}/engine/model.py", [(
    """        slots = store.resolve(L, indices, prefill)
        routed = self.moe_fn(y, slots, weights, arena, a.swiglu_limit).float()
        shared = R.expert_ffn(y, w.sh_w1, w.sh_w2, w.sh_w3, a.swiglu_limit).float()""",
    """        res = store.resolve(L, indices, prefill, split=SPLIT_MOE)
        if isinstance(res, tuple):
            slots, pend, wait_for_misses = res
            # the shared expert is dense bf16 and needs nothing from the expert store, and the
            # resident routed experts are already in the arena: both can run while the misses are
            # still coming off NVMe.
            shared = R.expert_ffn(y, w.sh_w1, w.sh_w2, w.sh_w3, a.swiglu_limit).float()
            routed = self.moe_fn(y, slots, weights, arena, a.swiglu_limit).float()
            wait_for_misses()
            routed = routed + self.moe_fn(y, pend, weights, arena, a.swiglu_limit).float()
        else:
            slots = res
            routed = self.moe_fn(y, slots, weights, arena, a.swiglu_limit).float()
            shared = R.expert_ffn(y, w.sh_w1, w.sh_w2, w.sh_w3, a.swiglu_limit).float()"""
), (
    "MAX_CHUNK = int(os.environ.get(\"DSV41_PREFILL_CHUNK\", 2048))",
    "MAX_CHUNK = int(os.environ.get(\"DSV41_PREFILL_CHUNK\", 2048))\n"
    "# a100-vq/patch_engine_split.py: run the residents (and the shared expert) while the misses load\n"
    "SPLIT_MOE = os.environ.get(\"DSV41_SPLIT_MOE\") == \"1\""
)], "split")
print("done -- enable with DSV41_SPLIT_MOE=1")
