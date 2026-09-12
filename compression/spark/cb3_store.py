"""Read pre-packed CB3 expert slots straight off NVMe, so a miss costs neither the FP4->CB3 pack
nor the wider FP4 read.

Measured on gx10-b872 (unpruned stream, ARENA_GB=94, DSV41_BLOCK=1):

    FP4 arena            108.7 ms/tok   NVMe 0.233 GB/tok   hit 0.9585    9.16 tok/s
    CB3 arena, pack on   148.5 ms/tok   NVMe 0.069 GB/tok   hit 0.9872    6.71 tok/s
      every miss                        (the CB3 arena holds 31 % more experts, so the traffic
                                         collapses -- and the pack, 20.8 ms/expert, more than
                                         eats it: ~3.7 misses/token is ~76 ms of fill)

This module removes the fill: the record written by a100-vq/pack_store.py is exactly the 12 slot
tensors of cb3_moe.CB3ArenaV2 concatenated, so a miss is one O_DIRECT pread of 14,454,784 B (a
multiple of 4096, so no alignment slack) into a pinned buffer plus 12 H2D slice copies.

Opt-in: the engine only attaches a store when DSV41_CB3_STORE names one, so nothing changes for a
run that does not ask for it.
"""

from __future__ import annotations

import json
import os
import threading

import torch

ALIGN = 4096


class CB3Store:
    def __init__(self, path: str, device, punched_path: str = ""):
        meta = json.load(open(path + ".json"))
        # every format that shares the CB3 slot geometry reads the same way; which codebook
        # decodes it is the engine's EXPERT_FORMAT, not the store's business
        assert meta["format"] in ("cb3_v2", "vq12_in_cb3_slots", "vq12_from_fp4"), meta["format"]
        self.format = meta["format"]
        # set by maybe_open when the run's EXPERT_FORMAT does not match this file's: the record is
        # read as it is and rewritten in the slot (see a100-vq/vq12_fallback.py)
        self.convert = None
        self.stride = int(meta["stride"])
        assert self.stride % ALIGN == 0, f"record stride {self.stride} is not {ALIGN}-aligned"
        self.offsets = {k: (int(o), int(n), tuple(s)) for k, (o, n, s) in meta["offsets"].items()}
        self.records = {tuple(int(x) for x in k.split(",")): int(v) for k, v in meta["records"].items()}
        self.fd = os.open(path + ".bin", os.O_RDONLY | os.O_DIRECT)
        self.device = device
        self._tls = threading.local()
        self.n_hits = 0
        # experts whose FP4 bytes were freed by a100-vq/punch_fp4.py: reading them would return
        # zeros, so a miss on one that is NOT in the store has to fail loudly instead
        self.punched = set()
        if punched_path and os.path.exists(punched_path):
            self.punched = {tuple(x) for x in json.load(open(punched_path))["punched"]}

    def guard(self, key) -> None:
        k = (int(key[0]), int(key[1]))
        if k in self.punched:
            raise RuntimeError(
                f"expert {k} is not in the CB3 store but its FP4 bytes were punched out of the "
                f"checkpoint; the store and models/fp4_punched.json disagree. Re-pack it with "
                f"a100-vq/pack_store.py or restore the shard.")

    def has(self, key) -> bool:
        return (int(key[0]), int(key[1])) in self.records

    def _buf(self):
        b = getattr(self._tls, "buf", None)
        if b is None:
            # pinned and 4096-aligned: O_DIRECT refuses anything else
            raw = torch.empty(self.stride + ALIGN, dtype=torch.uint8, pin_memory=True)
            off = (-raw.data_ptr()) % ALIGN
            b = self._tls.buf = raw[off:off + self.stride]
            self._tls.raw = raw
            self._tls.mv = b.numpy().data
        return b, self._tls.mv

    def load(self, arena, slot: int, key, stream) -> int:
        buf, mv = self._buf()
        rec = self.records[(int(key[0]), int(key[1]))]
        got = os.preadv(self.fd, [mv], rec * self.stride)
        assert got == self.stride, (got, self.stride)
        compute = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            stream.wait_stream(compute)
            for name, (o, n, shape) in self.offsets.items():
                getattr(arena, name)[slot].view(-1).copy_(buf[o:o + n], non_blocking=True)
            if self.convert is not None:
                self.convert(arena, slot)
        stream.synchronize()
        self.n_hits += 1
        return slot


def maybe_open(device):
    """CB3Store (or MultiStore) named by DSV41_CB3_STORE (colon separated), or None."""
    p = os.environ.get("DSV41_CB3_STORE", "")
    if not p:
        return None
    p = os.path.expanduser(p)
    stores = [x for x in p.split(":") if os.path.exists(x + ".bin") and os.path.exists(x + ".json")]
    if not stores:
        return None
    punched = os.environ.get("DSV41_FP4_PUNCHED", os.path.join(os.path.dirname(stores[0]),
                                                               "fp4_punched.json"))
    st = (CB3Store(stores[0], device, punched) if len(stores) == 1
          else MultiStore([CB3Store(s, device) for s in stores], punched))
    # a VQ12 run reading a CB3 record: re-encode it in the slot rather than raise. The A100-packed
    # VQ12 store does not cover every expert -- the disk does not hold two full stores -- and the
    # FP4 bytes behind the uncovered ones are punched.
    if os.environ.get("EXPERT_FORMAT", "") == "vq12":
        import vq12_fallback
        conv = vq12_fallback.Converter(device)
        for f in getattr(st, "stores", [st]):
            if f.format == "cb3_v2":
                f.convert = conv
        st.converter = conv
        n_cb3 = sum(1 for f in st.records.values() if f.format == "cb3_v2") \
            if isinstance(st, MultiStore) else (len(st.records) if st.format == "cb3_v2" else 0)
        print(f"[vq12] {len(st.records) - n_cb3} experts come from a VQ12 store, {n_cb3} would be "
              f"re-encoded from CB3 on the miss path", flush=True)
    return st


class MultiStore:
    """Several record files behind one interface, so a store can be extended without rewriting it."""

    def __init__(self, stores, punched_path: str = ""):
        self.stores = stores
        self.records = {}
        for st in stores:
            for k in st.records:
                # a later file overrides an earlier one, so a store can be extended without being
                # rewritten -- except that a record already in the run's own format wins over one
                # that would have to be converted on the miss path
                cur = self.records.get(k)
                if cur is None or cur.format == st.format or st.format != "cb3_v2":
                    self.records[k] = st
        self.punched = set()
        if punched_path and os.path.exists(punched_path):
            self.punched = {tuple(x) for x in json.load(open(punched_path))["punched"]}

    def has(self, key) -> bool:
        return (int(key[0]), int(key[1])) in self.records

    def guard(self, key) -> None:
        CB3Store.guard(self, key)

    def load(self, arena, slot: int, key, stream) -> int:
        return self.records[(int(key[0]), int(key[1]))].load(arena, slot, key, stream)
