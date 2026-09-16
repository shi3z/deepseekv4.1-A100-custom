"""CPU regressions for snapshot coverage and scalar multi-anchor capture."""
import unittest
from types import SimpleNamespace
import torch
from .engine import Engine

class SharedAttn:
    def __init__(self):
        self.compress_kv = {(2, 'cpu'): torch.ones(1, 4, 2)}
        self.index_k = {(2, 'cpu'): torch.ones(1, 4, 2) * 2}
        self.kv_owner = 2
        self.index_owner = 2
        self.topk_idxs = torch.ones(1, 4)
        self.candidates = torch.ones(1, 4)

class NgramHashState:
    def __init__(self): self.cache = torch.arange(8).reshape(1, 8)

class PrefixStateTests(unittest.TestCase):
    def test_snapshot_covers_cross_request_history_and_draft(self):
        e = Engine.__new__(Engine)
        e.model = SimpleNamespace(shared=SharedAttn(), engram_hash=NgramHashState())
        ring = torch.ones(1, 8, 2) * 3
        e.ds = SimpleNamespace(blocks=[SimpleNamespace(window_kv_cache=ring)])
        snap, _ = e._snapshot_prefix_state()
        expected = [x.clone() for _, _, _, x in snap]
        for kind, holder, key, v in e._prefix_cache_slots():
            if kind == 'scalar_attr': setattr(holder, key, 99)
            else: v.zero_()
        e._restore_prefix_state(snap)
        current = e._prefix_cache_slots()
        self.assertEqual(len(current), 6)
        for (_, _, _, actual), want in zip(current, expected):
            self.assertTrue(torch.equal(actual, want))
        self.assertTrue(torch.equal(e.model.shared.index_k[(2, 'cpu')], torch.ones(1,4,2)*2))
        self.assertTrue(torch.equal(e.model.engram_hash.cache, torch.arange(8).reshape(1,8)))
        self.assertTrue(torch.equal(ring, torch.ones(1,8,2)*3))

    def test_scalar_captures_each_anchor(self):
        import os
        from unittest.mock import patch
        e = Engine.__new__(Engine)
        e.mtp = 0; e.ds = None
        e.rt = SimpleNamespace(step=lambda token, pos: torch.tensor([pos]))
        count = []
        e._snapshot_prefix_state = lambda: (count.append(1) or [], 0)
        with patch.dict(os.environ, {'DSV41_PREFIX_BLOCK_REPLAY': '0'}):
            _, snaps = e._replay_prefix_tail(list(range(8)), 2, snapshot_at=[3,5,8])
        self.assertEqual([p for p, _ in snaps], [3,5,8])
    def test_restore_into_inference_tensor(self):
        e = Engine.__new__(Engine)
        e.model = SimpleNamespace(shared=SharedAttn(), engram_hash=NgramHashState())
        snap, _ = e._snapshot_prefix_state()
        with torch.inference_mode():
            e.model.shared.compress_kv[(2, 'cpu')] = torch.empty(1, 8, 2)
        # Should succeed without RuntimeError: Inplace update to inference tensor outside InferenceMode
        e._restore_prefix_state(snap)
        self.assertTrue(torch.equal(e.model.shared.compress_kv[(2, 'cpu')][:, :4], torch.ones(1, 4, 2)))

if __name__ == '__main__': unittest.main()
