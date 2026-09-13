"""Turn a plain text file into the JSONL `V41Engine.teacher_forced` scores, in fixed token chunks.

The A100 measures wikitext/code perplexity with ppl.py; the Spark cannot run that harness (its FP4
expert bytes are punched) and the A100 cannot run CBF8 (its expert GEMM decodes E2M1 in registers
and has no per-row codebook). So the two formats are compared on the Spark, on the same text, and
the result is carried onto the A100's scale through CB3, which both boxes can measure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="category=path pairs")
    ap.add_argument("--len", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=16, help="chunks per category")
    ap.add_argument("--model", default=os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash"))
    ap.add_argument("--out", default=os.path.expanduser("~/dsv41-spark/work/corpus/eval_ppl.jsonl"))
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    n = 0
    with open(a.out, "w") as f:
        for spec in a.src:
            cat, path = spec.split("=", 1)
            ids = tok.encode(open(path, encoding="utf-8").read(), add_special_tokens=False)
            for i in range(a.chunks):
                s = i * a.len
                if s + a.len > len(ids):
                    break
                f.write(json.dumps({"id": f"{cat}-{i}", "category": cat, "source": path,
                                    "text": tok.decode(ids[s:s + a.len])}) + "\n")
                n += 1
    print(f"{n} sequences of {a.len} tokens -> {a.out}")


if __name__ == "__main__":
    main()
