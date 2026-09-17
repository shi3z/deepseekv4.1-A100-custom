"""Exercise long cold prefill and append-only reuse through a running local API.

python -m dsv41.test_long_session --tokens 117896 --out results/long-session.json
The synthetic conversation is a capacity/transport check, not a Claude task.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path


def main():
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url', default='http://127.0.0.1:8000')
    ap.add_argument('--ckpt', default='/mnt/ssd/models/DeepSeek-V4.1-Flash')
    ap.add_argument('--tokens', type=int, default=117896)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    unit = tok.encode('Reference note: Water freezes at zero degrees Celsius.\n', add_special_tokens=False)
    ids = (unit * (args.tokens // len(unit) + 1))[:args.tokens]
    prompt = tok.decode(ids) + '\nSummarize the reference note in one short sentence.\n'
    records = []
    # /completions exposes the raw token count and permits exact append-only
    # prompts, independently of any external gateway's chat template.
    for suffix in ('', '\nReply once more using fewer words.\n'):
        body = dict(model='deepseek-v4.1-flash', prompt=prompt + suffix,
                    max_tokens=16, temperature=0, stream=False)
        request = urllib.request.Request(args.url + '/v1/completions',
                                         json.dumps(body).encode(),
                                         {'Content-Type': 'application/json'})
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=3600) as response:
            result = json.load(response)
        assert result.get('choices'), result
        assert result['usage']['completion_tokens'] > 0, result
        records.append(dict(seconds=time.perf_counter()-start, response=result))
        Path(args.out).write_text(json.dumps(records, indent=2, ensure_ascii=False))
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
