"""Free the FP4 bytes of routed experts that already live in the CB3 store, without breaking the
checkpoint.

The shards cannot simply be deleted: of the 510.3 GB, the routed experts are 288.8 GB but the other
214.3 GB -- 203.1 GB of it the Engram tables, plus the dense/attention/head weights and the 384
DSpark draft experts -- sits in the same files and is read every step. `fallocate --punch-hole`
frees the expert blocks while leaving every other tensor at its original offset, so nothing that
reads the checkpoint has to change.

Three rules keep this from touching anything it should not:

  * only tensors named `layers.<n>.ffn.experts.<e>.*` are considered -- never `mtp.*` (the DSpark
    draft experts stay FP4) and never anything else;
  * only experts whose (layer, expert) is a record in the store, so the bytes are recoverable;
  * the punched range is aligned INWARD to 4096, because safetensors packs tensors 8-byte aligned
    and a partial block at either edge can hold a neighbour's bytes. At most 8 KB per tensor is
    left behind; over the whole model that is under 1 GB.

Writes a sidecar listing what was punched, which the engine patch uses to fail loudly rather than
silently feed zeros if a punched expert is ever missing from the store.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import struct
import time

ALIGN = 4096
FALLOC_FL_KEEP_SIZE = 0x01
FALLOC_FL_PUNCH_HOLE = 0x02

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.fallocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64, ctypes.c_int64]


def punch(fd: int, off: int, length: int):
    if length <= 0:
        return 0
    if _libc.fallocate(fd, FALLOC_FL_KEEP_SIZE | FALLOC_FL_PUNCH_HOLE, off, length) != 0:
        e = ctypes.get_errno()
        raise OSError(e, os.strerror(e), f"fallocate(punch, {off}, {length})")
    return length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash"))
    ap.add_argument("--store", default=os.path.expanduser("~/dsv41-spark/models/cb3_store"),
                    help="colon separated list of stores whose experts may be punched")
    ap.add_argument("--out", default=os.path.expanduser("~/dsv41-spark/models/fp4_punched.json"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    have = {}
    for s in a.store.split(":"):
        meta = json.load(open(s + ".json"))
        for k in meta["records"]:
            have[tuple(int(x) for x in k.split(","))] = s
    print(f"{len(have)} experts are in the store(s) and may be punched")

    already = set()
    if os.path.exists(a.out):
        already = {tuple(x) for x in json.load(open(a.out))["punched"]}
        print(f"{len(already)} already punched in a previous run")

    idx = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
    by_file: dict[str, list[str]] = {}
    for n, f in idx.items():
        if n.startswith("layers.") and ".ffn.experts." in n:
            by_file.setdefault(f, []).append(n)

    freed = kept = 0
    punched: set = set(already)
    t0 = time.time()
    for fi, (f, names) in enumerate(sorted(by_file.items())):
        p = os.path.join(a.model, f)
        with open(p, "rb") as fh:
            hn = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(hn))
        base = 8 + hn
        todo = []
        for n in names:
            parts = n.split(".")
            key = (int(parts[1]), int(parts[4]))
            if key not in have or key in already:
                kept += hdr[n]["data_offsets"][1] - hdr[n]["data_offsets"][0]
                continue
            s, e = hdr[n]["data_offsets"]
            lo = -(-(base + s) // ALIGN) * ALIGN      # round the start UP
            hi = ((base + e) // ALIGN) * ALIGN        # round the end DOWN
            todo.append((lo, hi - lo, key))
        if not todo:
            continue
        if a.dry_run:
            freed += sum(t[1] for t in todo)
            punched.update(t[2] for t in todo)
            continue
        fd = os.open(p, os.O_RDWR)
        try:
            for lo, ln, key in todo:
                freed += punch(fd, lo, ln)
                punched.add(key)
        finally:
            os.close(fd)
        if (fi + 1) % 6 == 0:
            print(f"  {fi+1}/{len(by_file)} shards, {freed/1e9:.1f} GB freed, {time.time()-t0:.0f}s",
                  flush=True)
    print(f"{'would free' if a.dry_run else 'freed'} {freed/1e9:.1f} GB; "
          f"{kept/1e9:.1f} GB of expert bytes left in place (not in the store)")
    if not a.dry_run:
        json.dump({"punched": sorted(list(punched)), "model": a.model, "store": a.store},
                  open(a.out, "w"))
        print(f"wrote {a.out} ({len(punched)} experts)")


if __name__ == "__main__":
    main()
