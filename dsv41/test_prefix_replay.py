"""Full-model scalar/block replay audit. Run exclusively with serving stopped.

python -m dsv41.test_prefix_replay --out results/prefix-audit-20260916/ab.jsonl
Only GPUs 2,0,1,3,4 are allowed. Outputs exact differences, not an assertion
that different floating point kernels must produce bit-identical results.
"""
import argparse
import json
import time
from pathlib import Path
import torch
from .engine import Engine


def inventory(engine):
    """Named mutable state; omit weights and communication pointer tables."""
    out = {}
    m, rt = engine.model, engine.rt
    for i, b in enumerate(m.blocks):
        out[f'model.blocks.{i}.window_kv_cache'] = ('persistent', b.attn.window_kv_cache)
        c = getattr(b.attn, 'compressor', None)
        for k in ('kv_ring', 'score_ring', 'kv_state', 'score_state'):
            if c is not None and hasattr(c, k):
                out[f'model.blocks.{i}.compressor.{k}'] = ('derived-legacy' if k.endswith('_state') else 'persistent', getattr(c, k))
    for k in ('compress_kv', 'index_k'):
        for key, v in getattr(m.shared, k).items():
            out[f'shared.{k}.{key}'] = ('persistent', v)
    for k in ('kv_owner', 'index_owner', 'topk_idxs', 'candidates'):
        out[f'shared.{k}'] = ('ephemeral', getattr(m.shared, k))
    if m.engram_hash is not None:
        out['engram.cache'] = ('persistent', m.engram_hash.cache)
    out['model.main_hidden'] = ('output', getattr(m, 'main_hidden', None))
    # Runtime scratch, coordination state and outputs are classified explicitly.
    for k in ('pos', 'seq', 'pmax', 'tok', 'eng_in', 'main_hid', 'cand_buf', 'topk_buf',
              'kv_row', 'ik_row', 'kv_owner', 'index_owner', 'seqno', 'flag_route',
              'flag_part', 'flag_hop', 'mcast_counter', 'logits'):
        def add(name, v):
            if isinstance(v, dict):
                for key, val in v.items(): add(f'{name}.{key}', val)
            elif isinstance(v, (tuple, list)):
                for j, val in enumerate(v): add(f'{name}.{j}', val)
            else: out[name] = ('output' if k in ('main_hid', 'logits') else 'ephemeral', v)
        add(f'rt.{k}', getattr(rt, k, None))
    if engine.ds is not None:
        for i, b in enumerate(engine.ds.blocks):
            out[f'dspark.{i}.window_kv_cache'] = ('persistent', b.attn.window_kv_cache)
    return out


def freeze(inv):
    return {k: (kind, v.detach().cpu().clone() if torch.is_tensor(v) else v)
            for k, (kind, v) in inv.items()}


def difference(name, kind, a, b):
    row = dict(name=name, kind=kind)
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        row['equal'] = a is None and b is None or type(a) is type(b) and a == b
        return row
    row.update(shape=list(a.shape), dtype=str(a.dtype), other_shape=list(b.shape))
    if a.shape != b.shape:
        row['equal'] = False
        return row
    same = (a == b) | (torch.isnan(a) & torch.isnan(b)) if a.is_floating_point() else a == b
    bad = ~same
    row['mismatch_count'] = int(bad.sum())
    row['equal'] = not row['mismatch_count']
    row['first_mismatch_index'] = bad.nonzero()[0].tolist() if bad.any() else None
    delta = torch.where(same, 0, (a.double() - b.double()).abs())
    row['max_abs_diff'] = float(delta.max()) if delta.numel() else 0
    return row


@torch.inference_mode()
def run(a):
    e = Engine(devices=[2,0,1,3,4], ep=True, ep_shards=[77,77,77,77,76],
               max_seq_len=131072, mtp=a.mtp, mtp_device=4)
    ids = e.tok.encode(('A conversation about astronomy, music and careful reasoning. ' * 400))
    ids = (ids * 20)[:a.base + 3 + 512]
    e.model.collect_main_hidden = tuple(e.rt.target_layers)
    e.model.forward(torch.tensor([ids[:a.base]]), 0)
    # Seed from scalar execution too: exercises switching with a partial group.
    for p in range(a.base, a.base + 3): e.rt.step(ids[p], p)
    base = a.base + 3
    snap, _ = e._snapshot_prefix_state()
    with open(a.out, 'w') as f:
        def emit(row):
            f.write(json.dumps(row, allow_nan=True) + '\n'); f.flush()
        for n in (1,2,4,16,128,512):
            states, logits = [], []
            for mode in ('scalar', 'block'):
                e._restore_prefix_state(snap)
                t = time.perf_counter()
                if mode == 'scalar':
                    for p in range(base, base+n): out = e.rt.step(ids[p], p)
                    mh = torch.cat([e.rt.main_hid[l][0:1].to(e.ds.device if e.ds else e.model.blocks[-1].device)
                                    for l in e.rt.target_layers], -1).unsqueeze(1)
                    e.model.main_hidden = mh
                else:
                    e._set_prefix_replay_mode(True)
                    try: out = e.model.forward(torch.tensor([ids[base:base+n]]), base)
                    finally: e._set_prefix_replay_mode(False)
                for d in e.rt.devs: torch.cuda.synchronize(d)
                emit(dict(event='timing', mode=mode, n=n, seconds=time.perf_counter()-t))
                logits.append(out[0].float().cpu().clone())
                states.append(freeze(inventory(e)))
            emit(dict(n=n, **difference('final_logits', 'output', logits[0], logits[1])))
            print('AB', n, 'logit max', (logits[0]-logits[1]).abs().max().item(),
                  'argmax', [v.argmax().item() for v in logits], flush=True)
            for name, (kind, v) in states[0].items():
                emit(dict(n=n, **difference(name, kind, v, states[1][name][1])))
    print('AUDIT DONE', a.out, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--base', type=int, default=1024)
    ap.add_argument('--mtp', type=int, choices=[0,5], default=0)
    run(ap.parse_args())
