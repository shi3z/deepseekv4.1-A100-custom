"""Engram (conditional n-gram memory) with the hash tables resident in host RAM.

Hash ids depend only on token ids, so for every forward we compute the ids on the GPU, gather the
FP8 rows + E8M0 scales on the CPU (189 GiB of tables never touch VRAM), ship ~12 KB per token to
the GPU and dequantize there. The hashing itself follows the checkpoint's definition (compressed
vocab, per-layer multipliers, XOR-rolling n-gram hash into prime-sized buckets)."""
import os
import numpy as np
import torch
from sympy import isprime

from .quant import e8m0_to_float


def find_next_prime(start: int, seen: set[int]) -> int:
    c = start + 1
    while not isprime(c) or c in seen:
        c += 1
    return c


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Token id -> compressed id, where tokens that normalize alike (case, accents, whitespace) collapse."""
    from tokenizers import Regex, normalizers

    sentinel = ""
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " "),
    ])
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for tid in range(len(tokenizer)):
        text = backend.decode([tid], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(tid)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[tid] = new_id
    return lookup, len(key_to_new)


def compute_hash_multipliers(layer_ids, max_ngram_size: int, vocab_size: int) -> torch.Tensor:
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // vocab_size) // 2)
    rows = []
    for lid in layer_ids:
        g = np.random.default_rng(10007 * lid)
        v = g.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.tensor(v * 2 + 1))
    return torch.stack(rows)


class EngramLayout:
    def __init__(self, cfg: dict):
        self.layer_ids = tuple(cfg["engram_layer_ids"])
        self.max_ngram_size = cfg["engram_max_ngram_size"]
        self.n_heads = cfg["engram_n_heads"]
        self.head_dim = cfg["engram_head_dim"]
        self.num_embeddings = tuple(cfg["engram_num_embeddings"])
        primes, seen = [], set()
        for _ in self.layer_ids:
            per_ngram = []
            for _ in range(self.max_ngram_size - 1):
                sizes, cur = [], cfg["engram_vocab_size"] - 1
                for _ in range(self.n_heads):
                    cur = find_next_prime(cur, seen)
                    seen.add(cur)
                    sizes.append(cur)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        self.primes = tuple(primes)
        self.n_hash_cols = (self.max_ngram_size - 1) * self.n_heads


class NgramHashState:
    """Maps positions to the hash ids of the n-grams ending there; keeps the compressed-id history
    across prefill/decode. Runs on `device`."""

    DEAD = -1

    def __init__(self, cfg: dict, layout: EngramLayout, tokenizer, max_batch: int, max_seq: int, device):
        self.layout = layout
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        assert vocab_size == cfg["engram_compressed_vocab_size"], (vocab_size, cfg["engram_compressed_vocab_size"])
        self.pad_id = token_map[cfg["engram_pad_id"]]
        flat = [[p for per_ngram in layer for p in per_ngram] for layer in layout.primes]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        self.primes = torch.tensor(layout.primes, device=device)  # [n_layers, ngram-1, heads]
        self.offsets = torch.tensor(np.array(offsets), device=device)  # [n_layers, cols]
        self.multipliers = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, vocab_size).to(device)
        self.token_map = torch.tensor(token_map, device=device)
        self.cache = torch.empty(max_batch, max_seq, dtype=torch.int64, device=device)

    @torch.inference_mode()
    def rows(self, input_ids: torch.Tensor, seq: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Per-row decode: input_ids [B] at positions pos [B] of sequences seq [B] -> hash ids [B, 1, n_layers, n_hash_cols]."""
        B = input_ids.shape[0]
        compressed = self.token_map[input_ids]
        self.cache[seq, pos] = compressed
        hist = self.cache[seq]  # [B, max_seq]
        positions = pos.view(B, 1)
        tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(self.layout.max_ngram_size):
            source = hist.gather(1, (positions - shift).clamp_min(0))
            blocked = blocked | (positions < shift) | (source == self.DEAD)
            tokens.append(torch.where(blocked, self.pad_id, source))
        tokens = torch.stack(tokens, dim=-1)  # [B, 1, max_ngram]
        return self._hash(tokens)

    def __call__(self, input_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        """input_ids [B, L] -> hash ids [B, L, n_engram_layers, n_hash_cols] (int64, on device)."""
        batch, seqlen = input_ids.shape
        compressed = self.token_map[input_ids]
        self.cache[:batch, start_pos : start_pos + seqlen] = compressed
        positions = torch.arange(start_pos, start_pos + seqlen, device=input_ids.device).expand(batch, seqlen)
        tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(self.layout.max_ngram_size):
            source = self.cache[:batch].gather(1, (positions - shift).clamp_min(0))
            blocked = blocked | (positions < shift) | (source == self.DEAD)
            tokens.append(torch.where(blocked, self.pad_id, source))
        tokens = torch.stack(tokens, dim=-1)  # [B, L, max_ngram]
        return self._hash(tokens)

    def _hash(self, tokens: torch.Tensor) -> torch.Tensor:
        products = tokens.unsqueeze(2) * self.multipliers  # [B, L, n_layers, max_ngram]
        rolling, hashes = products[..., 0], []
        for i in range(1, self.layout.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets


class HostEngramTable:
    """One layer's hash table (FP8 rows + E8M0 scales) in host memory, gathered by index."""

    def __init__(self, weight_u8: torch.Tensor, scale_u8: torch.Tensor, block: int = 32):
        assert weight_u8.dtype == torch.uint8 and scale_u8.dtype == torch.uint8
        self.weight = weight_u8  # [rows, 256] bytes of float8_e4m3fn
        self.scale = scale_u8  # [rows, 8]
        self.block = block

    def lookup(self, ids: torch.Tensor, device) -> torch.Tensor:
        """ids: int64 [...]; returns dequantized bf16 [..., 256] on `device`."""
        flat = ids.reshape(-1).cpu()
        rows = torch.index_select(self.weight, 0, flat).pin_memory().to(device, non_blocking=True)
        scales = torch.index_select(self.scale, 0, flat).pin_memory().to(device, non_blocking=True)
        vals = rows.view(torch.float8_e4m3fn).float().unflatten(-1, (-1, self.block)) * e8m0_to_float(scales).unsqueeze(-1)
        return vals.flatten(-2).to(torch.bfloat16).view(*ids.shape, -1)


class Engram(torch.nn.Module):
    """Gated write of the n-gram lookup into the hc_mult residual copies (reference semantics)."""

    def __init__(self, dim: int, hc_mult: int, layout: EngramLayout, table: HostEngramTable,
                 wkv: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor, eps: float):
        super().__init__()
        self.dim, self.hc_mult, self.eps = dim, hc_mult, eps
        self.table = table
        self.wkv = wkv  # bf16 [dim*(hc+1), n_hash_cols*head_dim] (dequantized from fp8)
        self.qk_weight = (q_weight.float() * k_weight.float())  # [hc, dim]
        self.clamp_value = 1e-6

    def forward(self, x: torch.Tensor, hash_ids: torch.Tensor) -> torch.Tensor:
        """x: [B, L, hc, dim] bf16; hash_ids: [B, L, n_hash_cols].

        Long prefill is processed in sequence chunks so apply() never
        materializes x.float() for the entire prompt at once.

        Engram application is token-local once hash_ids are known, so
        chunking along L preserves the computation.
        """
        B, L = x.shape[:2]

        chunk = int(
            os.environ.get("DSV41_ENGRAM_PREFILL_CHUNK", "256")
        )
        chunk = max(1, chunk)

        if L <= chunk:
            emb = self.table.lookup(
                hash_ids,
                x.device,
            ).flatten(-2)

            return self.apply(x, emb)

        if os.environ.get("DSV41_DEBUG", "0") == "1":
            print(
                f"[engram-prefill] tokens={L:,} chunk={chunk:,} "
                f"device={x.device}",
                flush=True,
            )

        # Modify x chunk-by-chunk. The caller immediately replaces h with
        # this return value, so keeping a second full [B,L,hc,dim] tensor
        # would only waste VRAM.
        for s0 in range(0, L, chunk):
            s1 = min(s0 + chunk, L)

            xs = x[:, s0:s1]

            emb = self.table.lookup(
                hash_ids[:, s0:s1],
                x.device,
            ).flatten(-2)

            ys = self.apply(xs, emb)

            xs.copy_(ys)

            del emb
            del ys

        return x

    def apply(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """The GPU half: emb [B, L, cols*head_dim] bf16 (already gathered + dequantized)."""
        from .w8 import linear_w
        kv = linear_w(emb, self.wkv)
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        h = x.float()
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (h * self.qk_weight * key).sum(-1) * rstd * self.dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)
