"""Regression checks for continuation compression and bounded indexer scratch."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from . import model


class LongPrefillTests(unittest.TestCase):
    def test_compressor_block_matches_scalar_and_state(self):
        torch.manual_seed(42)
        for ratio in (2, 4, 8):
            args = model.Args(dict(compress_ratios=[ratio], head_dim=16, norm_eps=1e-6))
            weights = {'compressor.norm.weight': torch.ones(16),
                       'compressor.wkv.weight': torch.randn(16, 16),
                       'compressor.wgate.weight': torch.randn(16, 16)}
            for start in range(1, ratio + 1):
                for count in (1, 3, 16, 129):
                    block = model.Compressor(args, 0, weights, 'cpu')
                    scalar = model.Compressor(args, 0, weights, 'cpu')
                    x = torch.randn(1, start + count, 16)
                    def norm(v, w, eps):
                        return v * torch.rsqrt(v.square().mean(-1, keepdim=True) + eps) * w
                    with patch.object(model, 'rmsnorm', norm):
                        block(x[:, :start], 0)
                        scalar(x[:, :start], 0)
                        actual = block(x[:, start:], start)
                        pieces = [scalar(x[:, i:i+1], i) for i in range(start, start + count)]
                    pieces = [v for v in pieces if v is not None]
                    if pieces:
                        torch.testing.assert_close(actual, torch.cat(pieces, 1), atol=2e-5, rtol=2e-5)
                    else:
                        self.assertIsNone(actual)
                    for name in ('kv_state', 'score_state', 'kv_ring', 'score_ring'):
                        torch.testing.assert_close(getattr(block, name), getattr(scalar, name), atol=2e-5, rtol=2e-5)

    def test_decode_cache_growth_and_graph_invalidation(self):
        from .decode import DecodeRuntime
        rt = DecodeRuntime.__new__(DecodeRuntime)
        shared = model.SharedAttn()
        shared.cache_max_rows = {0: 100}
        shared.compress_kv = {(0, 'cpu'): torch.ones(1, 3, 4)}
        shared.index_k = {(0, 'cpu'): torch.ones(1, 3, 2)}
        rt.m = SimpleNamespace(shared=shared, blocks=[SimpleNamespace(attn=SimpleNamespace(ratio=2))])
        rt.devices = []
        rt.graphs = {'cpu': object()}
        rt._graph_cache_signature = rt._cache_signature()
        rt._prepare_decode_cache(4)
        self.assertTrue(rt.graphs)
        with patch.dict(os.environ, {'DSV41_EXACT_CACHE_GROW': '1'}):
            rt._prepare_decode_cache(6)
        self.assertFalse(rt.graphs)
        for table in (shared.compress_kv, shared.index_k):
            self.assertEqual(table[(0, 'cpu')].shape[1], 4)
            self.assertTrue(table[(0, 'cpu')][:, :3].eq(1).all())
        # Prefill can replace storage without needing growth at decode time.
        rt.graphs = {'cpu': object()}
        rt._graph_cache_signature = rt._cache_signature()
        shared.index_k[(0, 'cpu')] = shared.index_k[(0, 'cpu')].clone()
        rt._prepare_decode_cache(6)
        self.assertFalse(rt.graphs)

    def test_indexer_chunk_matches_full_with_candidates(self):
        torch.manual_seed(17)
        for start in (0, 3, 1025):
            for mode in ('plain', 'source', 'consumer'):
                n, heads, dim, ratio = 137, 2, 8, 2
                keys = (start+n)//ratio
                obj = model.Indexer.__new__(model.Indexer)
                obj.ratio, obj.rope_head_dim = ratio, 0
                obj.owns_k = False
                obj.device = 'cpu'
                obj.n_heads, obj.head_dim = heads, dim
                obj.wq_b = torch.randn(heads*dim, dim)
                obj.weights_proj = torch.randn(heads, dim)
                obj.softmax_scale = dim**-0.5
                obj.cos = obj.sin = None
                obj.topk = 16
                obj.is_candidate_source = mode == 'source'
                obj.uses_candidates = mode == 'consumer'
                obj.candidate_topk_blocks = 4
                obj.candidate_block_size = 8
                obj.shared = SimpleNamespace(index_owner=0, index_k={(0, 'cpu'): torch.randn(1, keys, dim)},
                                             candidates=torch.rand(1, n, keys) > .3)
                x, qr = torch.randn(1, n, dim), torch.randn(1, n, dim)
                with patch.object(model, 'linear_fp8', F.linear), patch.object(model, 'rope_', lambda *a, **k: None), patch.object(model, 'fake_quant_fp4', lambda x, *a: x):
                    with patch.dict(os.environ, {'DSV41_INDEX_QUERY_CHUNK': '10000'}):
                        expected = obj(x, qr, None, start, 128)
                        candidates = obj.shared.candidates.clone()
                    with patch.dict(os.environ, {'DSV41_INDEX_QUERY_CHUNK': '32'}):
                        actual = obj(x, qr, None, start, 128)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(obj.shared.candidates, candidates, rtol=0, atol=0)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
