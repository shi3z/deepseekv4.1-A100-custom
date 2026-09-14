"""Compare each dense-FP4 config's fixed-prefill logits against the FP8 baseline.

Same 256 tokens, same teacher-forced positions, so position i sees the same context in every
run -- which is exactly what the generated-token comparison could not promise.
"""
from __future__ import annotations

import os
import sys

import torch

base = torch.load(os.path.expanduser("~/pre_base.pt")).float()
print(f"{'config':10s} {'top1 agree':>10s} {'top5 overlap':>13s} {'cosine':>9s} "
      f"{'rel L2':>8s} {'max|d|':>8s} {'KL(b||c)':>9s}")
b1 = base.argmax(-1)
b5 = base.topk(5, -1).indices
bp = torch.log_softmax(base, -1)
for tag in sys.argv[1:]:
    c = torch.load(os.path.expanduser(f"~/pre_{tag}.pt")).float()
    c1 = c.argmax(-1)
    c5 = c.topk(5, -1).indices
    ov = (b5.unsqueeze(-1) == c5.unsqueeze(-2)).any(-1).float().mean()
    cos = torch.nn.functional.cosine_similarity(base.flatten(0, -2), c.flatten(0, -2), -1).mean()
    rel = (c - base).norm() / base.norm()
    kl = torch.nn.functional.kl_div(torch.log_softmax(c, -1), bp,
                                    log_target=True, reduction="batchmean")
    print(f"{tag:10s} {(b1 == c1).float().mean():10.4f} {ov:13.4f} {cos:9.5f} "
          f"{rel:8.4f} {(c - base).abs().max():8.4f} {kl:9.5f}")
