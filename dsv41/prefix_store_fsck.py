"""Read-only integrity check for immutable prefix-cache blocks."""
import argparse
import hashlib
import os
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="/dev/shm/dsv41-prefix-cache")
    args = ap.parse_args()
    root = args.root
    block_root = os.path.join(root, "blocks")
    referenced = set()
    manifests = missing = corrupt = 0
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        if not (name.startswith("prefix-") and name.endswith(".pt")):
            continue
        manifests += 1
        path = os.path.join(root, name)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"manifest={name} ERROR={exc}")
            corrupt += 1
            continue
        for spec in payload.get("block_refs", []) or []:
            if not spec:
                continue
            for digest in spec.get("refs", []):
                referenced.add(digest)
                block = os.path.join(block_root, digest + ".pt")
                if not os.path.exists(block):
                    print(f"missing manifest={name} block={digest}")
                    missing += 1
                    continue
                try:
                    tensor = torch.load(block, map_location="cpu", weights_only=False)
                    actual = hashlib.sha256(
                        tensor.contiguous().view(torch.uint8).numpy().tobytes()
                    ).hexdigest()[:32]
                    if actual != digest:
                        print(f"corrupt manifest={name} block={digest} actual={actual}")
                        corrupt += 1
                except Exception as exc:
                    print(f"corrupt block={digest} ERROR={exc}")
                    corrupt += 1
    blocks = {
        n[:-3] for n in os.listdir(block_root)
        if n.endswith(".pt")
    } if os.path.isdir(block_root) else set()
    orphan = sorted(blocks - referenced)
    total_bytes = sum(
        os.path.getsize(os.path.join(block_root, n + ".pt"))
        for n in blocks
    ) if os.path.isdir(block_root) else 0
    print(
        f"[prefix-fsck] manifests={manifests} blocks={len(blocks)} "
        f"referenced={len(referenced)} orphan={len(orphan)} "
        f"missing={missing} corrupt={corrupt} "
        f"physical={total_bytes/2**20:.1f}MiB"
    )
    return 1 if missing or corrupt else 0


if __name__ == "__main__":
    sys.exit(main())
