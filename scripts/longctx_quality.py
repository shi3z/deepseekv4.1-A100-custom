"""Long-context retrieval quality of the dsv41 server, over HTTP, at the lengths it actually serves.

The README's long-context evidence is one 499K needle checked by hand.  This measures it systematically, so that a
change which trades exactness for speed -- `DSV41_CED` (layers 21-39 over the last 128 tokens only), the selection
budget `index_topk`, a different `DSV41_SLOT_CAPS`, or raising max_seq_len to 1M -- can be judged instead of
assumed.

Design: ONE long context per length, carrying one needle per depth, prefilled once; then one question per needle,
each a separate request that shares the whole context as its prefix.  The first request pays the cold prefill (39
minutes at 499K), the rest hit the prefix cache (419K restored in 12.8 s, measured 2026-11-04), so a sweep over
depths costs one prefill instead of one per depth.  That is also what a real session looks like: a long document
and then many turns.

  PYTHONPATH=. python scripts/longctx_quality.py --lengths 8k,32k --depths 0.1,0.5,0.9 --label ced0
  PYTHONPATH=. python scripts/longctx_quality.py --lengths 128k,256k --filler ja_corpus.txt --out results/longctx_ced0.json

Each row records whether the needle came back, the usage the server reports (prompt tokens and how many of them came
from a cache), the time to first token and the decode rate, so a quality regression and a speed change are visible
in the same table.
"""
import argparse, json, os, random, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CKPT = os.environ.get("DSV41_CKPT", "/mnt/raid6/models/DeepSeek-V4.1-Flash-1017-abliterated")

# A needle has to be unique, unguessable from the filler, and checkable by exact string match.  Each depth gets its
# own record id and value so one question cannot be answered from another needle.
NEEDLE = "【記録 {tag}】第{idx}四半期の{item}の在庫数は {value} 個であった。"
ITEMS = ["特殊ボルト", "絶縁シート", "冷却ファン", "光学レンズ", "基板コネクタ", "防振ゴム", "耐熱テープ", "軸受リング"]
QUESTION = ("上の文章の中に「記録 {tag}」という記録があります。その記録に書かれている在庫数を、"
            "数字だけで答えてください。説明は不要です。")


def tokens(text, tok):
    return len(tok(text, add_special_tokens=False)["input_ids"])


def build_filler(tok, target_tokens, path=None):
    """Filler text of about target_tokens tokens.  A real file if one is given (prose is the honest case), else a
    synthetic report whose sentences vary, so the model cannot shortcut by compressing a repeated block."""
    if path:
        base = open(path, encoding="utf-8").read()
        if not base.strip():
            raise SystemExit(f"{path} is empty")
        out = base
        while tokens(out, tok) < target_tokens:
            out += "\n\n" + base
        ids = tok(out, add_special_tokens=False)["input_ids"][:target_tokens]
        return tok.decode(ids)
    rng = random.Random(0)
    lines, n = [], 0
    depts = ["第一製造部", "品質保証部", "調達部", "技術開発部", "物流管理部", "設備保全部"]
    acts = ["定例点検を実施した", "工程の見直しを行った", "部材の受け入れ検査を完了した",
            "設備の稼働記録を更新した", "作業手順書を改訂した", "испытание"[:0] + "試作品の評価を開始した"]
    while n < target_tokens:
        line = (f"{rng.randint(2020, 2026)}年{rng.randint(1, 12)}月{rng.randint(1, 28)}日、"
                f"{rng.choice(depts)}は{rng.choice(acts)}。担当者は{rng.randint(1, 40)}名、"
                f"所要時間は{rng.randint(1, 48)}時間、記録番号は R-{rng.randint(10000, 99999)} である。")
        lines.append(line)
        n += tokens(line, tok) + 1
        if len(lines) % 400 == 0:
            lines.append(f"--- 第{len(lines)//400}節 ---")
    ids = tok("\n".join(lines), add_special_tokens=False)["input_ids"][:target_tokens]
    return tok.decode(ids)


def build_context(tok, length_tokens, depths, seed=0):
    """A context of about length_tokens tokens with one needle per depth.  Returns (text, needles)."""
    rng = random.Random(seed)
    needles = []
    for i, d in enumerate(depths):
        tag = f"{chr(ord('A') + i)}{rng.randint(100, 999)}"
        needles.append({"depth": d, "tag": tag, "value": f"{rng.randint(1000, 99999):,}",
                        "item": ITEMS[i % len(ITEMS)], "idx": rng.randint(1, 4)})
    budget = length_tokens - sum(tokens(NEEDLE.format(**n), tok) + 4 for n in needles) - 64
    filler = build_filler(tok, max(256, budget), FILLER_PATH)
    ids = tok(filler, add_special_tokens=False)["input_ids"]
    pieces, last = [], 0
    for n in sorted(needles, key=lambda x: x["depth"]):
        cut = min(len(ids), max(0, int(len(ids) * n["depth"])))
        pieces.append(tok.decode(ids[last:cut]))
        pieces.append("\n\n" + NEEDLE.format(**n) + "\n\n")
        last = cut
    pieces.append(tok.decode(ids[last:]))
    return "".join(pieces), needles


def ask(base_url, model, prompt, max_new, timeout_s):
    """One non-streaming request; returns (text, stats).  Non-stream so the server's usage and prefill_policy come
    back in the body."""
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_new, "temperature": 0}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    if "error" in d:
        return None, {"error": str(d["error"])[:300], "seconds": round(dt, 1)}
    msg = d["choices"][0]["message"]
    u = d.get("usage") or {}
    pt = u.get("prompt_tokens") or 0
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", u.get("prompt_cache_hit_tokens", 0))
    return (msg.get("content") or ""), {
        "seconds": round(dt, 1), "prompt_tokens": pt, "completion_tokens": u.get("completion_tokens"),
        "cached_tokens": cached, "cache_hit_pct": round(100 * cached / max(1, pt), 1),
        "policy": (d.get("prefill_policy") or {}).get("policy"),
        "finish": d["choices"][0].get("finish_reason")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--lengths", default="8k,32k", help="comma separated, e.g. 32k,128k,256k,499k")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--filler", default=None, help="text file to fill the context with; omitted = synthetic report")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--timeout", type=int, default=5400, help="per request; a cold 499K prefill takes ~2,400 s")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    global FILLER_PATH
    FILLER_PATH = a.filler

    def n_tokens(x):
        x = x.strip().lower()
        return int(float(x[:-1]) * (1024 if x.endswith("k") else 1024 * 1024)) if x[-1] in "km" else int(x)

    lengths = [n_tokens(x) for x in a.lengths.split(",") if x.strip()]
    depths = [float(x) for x in a.depths.split(",") if x.strip()]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    res = {"label": a.label, "base_url": a.base_url, "lengths": lengths, "depths": depths,
           "filler": a.filler or "synthetic", "rows": []}
    for L in lengths:
        ctx, needles = build_context(tok, L, depths)
        real = tokens(ctx, tok)
        print(f"\n=== context {L} tokens requested, {real:,} built, {len(needles)} needles", flush=True)
        for n in needles:
            prompt = ctx + "\n\n" + QUESTION.format(tag=n["tag"])
            text, st = ask(a.base_url, a.model, prompt, a.max_new, a.timeout)
            hit = text is not None and n["value"].replace(",", "") in (text or "").replace(",", "").replace(" ", "")
            row = {"length": L, "built_tokens": real, "depth": n["depth"], "tag": n["tag"],
                   "want": n["value"], "hit": bool(hit), "answer": (text or "")[:120], **st}
            res["rows"].append(row)
            print(f"  depth {n['depth']:.2f} tag {n['tag']}: hit={hit} want={n['value']} "
                  f"got={(text or '').strip()[:40]!r} {json.dumps({k: st[k] for k in ('seconds', 'prompt_tokens', 'cache_hit_pct', 'policy') if k in st})}",
                  flush=True)
        got = [r for r in res["rows"] if r["length"] == L]
        print(f"  == {L}: {sum(r['hit'] for r in got)}/{len(got)} retrieved", flush=True)
    total = res["rows"]
    print(f"\n=== {a.label}: {sum(r['hit'] for r in total)}/{len(total)} retrieved overall")
    for L in lengths:
        g = [r for r in total if r["length"] == L]
        print(f"  {L:>8}: {sum(r['hit'] for r in g)}/{len(g)}  "
              + "  ".join(f"d{r['depth']:.1f}={'o' if r['hit'] else 'x'}" for r in g))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
        print("wrote", a.out)


FILLER_PATH = None

if __name__ == "__main__":
    main()
