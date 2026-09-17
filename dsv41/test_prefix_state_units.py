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
        self.cache_max_rows = {2: 128}

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

    def test_multi_mirror_dedup_and_broadcast(self):
        e = Engine.__new__(Engine)
        shared = SharedAttn()
        shared.compress_kv[(2, 'dev2')] = torch.ones(1, 4, 2)
        shared.index_k[(2, 'dev2')] = torch.ones(1, 4, 2) * 2
        e.model = SimpleNamespace(shared=shared, engram_hash=NgramHashState(), args=SimpleNamespace(compress_ratios={2: 1}))

        slots = e._prefix_cache_slots()
        ckv_slots = [s for s in slots if s[0] == 'dict' and s[1] is shared.compress_kv]
        idx_slots = [s for s in slots if s[0] == 'dict' and s[1] is shared.index_k]
        self.assertEqual(len(ckv_slots), 1)
        self.assertEqual(len(idx_slots), 1)

        snap, _ = e._snapshot_prefix_state()
        shared.compress_kv[(2, 'cpu')].zero_()
        shared.compress_kv[(2, 'dev2')].zero_()
        shared.index_k[(2, 'cpu')].zero_()
        shared.index_k[(2, 'dev2')].zero_()

        e._restore_prefix_state(snap)
        self.assertTrue(torch.equal(shared.compress_kv[(2, 'cpu')], torch.ones(1, 4, 2)))
        self.assertTrue(torch.equal(shared.compress_kv[(2, 'dev2')], torch.ones(1, 4, 2)))
        self.assertTrue(torch.equal(shared.index_k[(2, 'cpu')], torch.ones(1, 4, 2) * 2))
        self.assertTrue(torch.equal(shared.index_k[(2, 'dev2')], torch.ones(1, 4, 2) * 2))

    def test_ngram_hash_and_swa_slicing(self):
        e = Engine.__new__(Engine)
        shared = SharedAttn()
        engram = NgramHashState()
        engram.cache = torch.arange(100).reshape(1, 100)
        e.model = SimpleNamespace(shared=shared, engram_hash=engram, args=SimpleNamespace(compress_ratios={2: 1}))
        ring = torch.ones(1, 200, 4)
        e.ds = SimpleNamespace(blocks=[SimpleNamespace(window_kv_cache=ring)])

        snap, total = e._snapshot_prefix_state(used_tokens=20)
        snap_map = {repr(key): src for kind, holder, key, src in snap}
        self.assertEqual(snap_map["'cache'"].shape, (1, 20))
        self.assertEqual(snap_map["'window_kv_cache'"].shape, (1, 20, 4))

    def test_find_best_gpu_slot(self):
        e = Engine.__new__(Engine)
        e._slot_tokens = {}
        # Empty slots
        slot, lcp = e._find_best_gpu_slot([1, 2, 3])
        self.assertEqual(slot, -1)
        self.assertEqual(lcp, 0)

        # Slot 1 has partial match, Slot 2 has longer match
        e._slot_tokens = {
            1: [10, 20, 30, 40],
            2: [10, 20, 30, 40, 50, 60],
            3: [99, 99],
        }
        slot, lcp = e._find_best_gpu_slot([10, 20, 30, 40, 50, 60, 70, 80])
        self.assertEqual(slot, 2)
        self.assertEqual(lcp, 6)

        # Tie: prefer slot 0
        e._slot_tokens = {
            0: [1, 2, 3, 4],
            1: [1, 2, 3, 4],
        }
        slot, lcp = e._find_best_gpu_slot([1, 2, 3, 4, 5])
        self.assertEqual(slot, 0)
        self.assertEqual(lcp, 4)

    def test_gpu_slot_prefill_reuse(self):
        e = Engine.__new__(Engine)
        copied = []
        forwarded = []
        e.rt = SimpleNamespace(copy_seq=lambda src, dst, req_id="": copied.append((src, dst)))
        e._forward_prefix_continuation = lambda prompt_ids, start_pos, chunk_size=512: (
            forwarded.append((prompt_ids, start_pos)) or torch.tensor([[1.0, 2.0]])
        )
        e.last_prefill_stats = None
        e.current_context_tokens = 0
        import collections
        e.prefill_history = collections.deque(maxlen=10)
        e.stats_tracker = None

        prompt = list(range(100)) # 100 tokens
        logits, reused = e._prefill_with_gpu_slot_reuse(prompt, best_slot=2, best_lcp=85)
        # best_lcp 85 -> aligned to 16: 80
        self.assertEqual(reused, 80)
        self.assertEqual(copied, [(2, 0)])
        self.assertEqual(forwarded, [(prompt, 80)])
        self.assertEqual(e.last_prefill_stats["mode"], "gpu_slot_hit")
        self.assertEqual(e.last_prefill_stats["reused_tokens"], 80)
        self.assertEqual(e.last_prefill_stats["new_tokens"], 20)

        # Exact match (best_lcp == 100) -> reuse_pos max(0, 100 - 16) = 84
        copied.clear()
        forwarded.clear()
        logits, reused = e._prefill_with_gpu_slot_reuse(prompt, best_slot=0, best_lcp=100)
        self.assertEqual(reused, 84)
        self.assertEqual(copied, []) # best_slot is 0, no copy needed!
        self.assertEqual(forwarded, [(prompt, 84)])

    def test_copy_seq_with_inference_tensors(self):
        from .decode import DecodeRuntime
        rt = DecodeRuntime.__new__(DecodeRuntime)
        with torch.inference_mode():
            wkv = torch.ones(5, 8, 4)
            ckv = torch.ones(5, 16, 4)
            idx_k = torch.ones(5, 16, 2)
            c_ring = torch.ones(5, 8, 4)
            s_ring = torch.ones(5, 8, 4)
            c_state = torch.ones(5, 8, 4)
            s_state = torch.ones(5, 8, 4)
            eng_cache = torch.ones(5, 32, dtype=torch.int64)

        compressor = SimpleNamespace(
            ratio=2,
            kv_ring=c_ring,
            score_ring=s_ring,
            kv_state=c_state,
            score_state=s_state,
        )
        attn = SimpleNamespace(layer_id=0, window_kv_cache=wkv, compressor=compressor)
        block = SimpleNamespace(layer_id=0, attn=attn)
        shared = SimpleNamespace(
            compress_kv={(0, 'cpu'): ckv},
            index_k={(0, 'cpu'): idx_k},
        )
        engram_hash = SimpleNamespace(cache=eng_cache)
        rt.m = SimpleNamespace(blocks=[block], shared=shared, engram_hash=engram_hash)

        # Set slot 0 with distinct values
        with torch.inference_mode():
            wkv[0].fill_(42.0)
            ckv[0].fill_(43.0)
            idx_k[0].fill_(44.0)
            c_ring[0].fill_(45.0)
            eng_cache[0].fill_(46)

        # Ensure calling copy_seq OUTSIDE inference mode works seamlessly without RuntimeError
        self.assertFalse(torch.is_inference_mode_enabled())
        rt.copy_seq(0, 2)

        self.assertTrue(torch.equal(attn.window_kv_cache[2], attn.window_kv_cache[0]))
        self.assertTrue(torch.equal(shared.compress_kv[(0, 'cpu')][2], shared.compress_kv[(0, 'cpu')][0]))
        self.assertTrue(torch.equal(shared.index_k[(0, 'cpu')][2], shared.index_k[(0, 'cpu')][0]))
        self.assertTrue(torch.equal(compressor.kv_ring[2], compressor.kv_ring[0]))
        self.assertTrue(torch.equal(engram_hash.cache[2], engram_hash.cache[0]))

if __name__ == '__main__': unittest.main()
