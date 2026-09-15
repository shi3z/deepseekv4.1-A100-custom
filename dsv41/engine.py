"""Generation engine shared by the REPL and the OpenAI-compatible server: loads the model once, keeps
the CUDA-graph decode runtime, and streams tokens for one request at a time (the runtime is single
sequence; requests are serialized with a lock)."""
from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator

import torch

from .decode import DecodeRuntime
from .load import load_model

CKPT = "/mnt/ssd/models/DeepSeek-V4.1-Flash"


@dataclass
class GenParams:
    max_new_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.95
    stop: list[str] = field(default_factory=list)
    seed: int | None = None


def sample_token(logits: torch.Tensor, temperature: float, top_p: float, gen: torch.Generator | None) -> int:
    if temperature <= 0:
        return int(logits.argmax(dim=-1).item())
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    if 0 < top_p < 1:
        sp, si = probs.sort(descending=True)
        keep = (sp.cumsum(-1) - sp) < top_p  # keep tokens until cumulative mass passes top_p
        sp = sp * keep
        idx = torch.multinomial(sp / sp.sum(), 1, generator=gen)
        return int(si.gather(-1, idx).item())
    return int(torch.multinomial(probs, 1, generator=gen).item())


class Engine:
    def __init__(self, ckpt: str = CKPT, devices: list[int] | None = None, max_seq_len: int = 8192,
                 budgets: dict[int, float] | None = None, use_graphs: bool = True, thinking_mode: str = "chat",
                 offload_experts=False, hot_experts: int = 0, route_stats: str = "", ep: bool = False, ep_shards: list[int] | None = None):
        from transformers import AutoTokenizer

        sys.path.insert(0, os.path.join(ckpt, "encoding"))
        from encoding import encode_messages, parse_message_from_completion_text  # type: ignore

        self._encode = encode_messages
        self._parse = parse_message_from_completion_text
        self.thinking_mode = thinking_mode
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.max_seq_len = max_seq_len
        self.model = load_model(ckpt, devices or list(range(torch.cuda.device_count())), max_seq_len=max_seq_len,
                                budgets_gb=budgets, tokenizer=self.tok, offload_experts=offload_experts,
                                hot_experts=hot_experts, route_stats=route_stats, ep=ep, ep_shards=ep_shards)
        if offload_experts:
            from .decode import OffloadDecodeRuntime
            self.rt = OffloadDecodeRuntime(self.model, use_graphs=use_graphs)
        elif ep:
            from .ep import EPRuntime
            self.rt = EPRuntime(self.model, use_graphs=use_graphs)
        else:
            self.rt = DecodeRuntime(self.model, use_graphs=use_graphs)
        if use_graphs:
            self.rt.capture()
        self.lock = threading.Lock()
        self.eos = self.tok.eos_token_id
        self.model_name = "deepseek-v4.1-flash"

    # ---------------------------------------------------------------- prompts
    def chat_prompt(self, messages: list[dict], thinking_mode: str | None = None) -> str:
        return self._encode(messages, thinking_mode=thinking_mode or self.thinking_mode)

    def parse_completion(self, text: str, thinking_mode: str | None = None) -> dict:
        """Structured assistant message (content / reasoning_content / tool_calls). The official parser
        wants the completion to end with the EOS string; we generate without it, so append it."""
        eos = self.tok.eos_token or ""
        try:
            return self._parse(text + eos, thinking_mode=thinking_mode or self.thinking_mode)
        except Exception:
            return {"role": "assistant", "content": text, "reasoning_content": None, "tool_calls": []}

    # ---------------------------------------------------------------- generation
    @torch.inference_mode()
    def generate(self, prompt_ids: list[int], p: GenParams) -> Iterator[tuple[int, str]]:
        """Yields (token_id, text_piece) as they are produced. Holds the engine lock for the duration."""
        assert len(prompt_ids) < self.max_seq_len, f"prompt of {len(prompt_ids)} tokens exceeds max_seq_len={self.max_seq_len}"
        max_new = min(p.max_new_tokens, self.max_seq_len - len(prompt_ids) - 1)
        gen = None
        if p.seed is not None:
            gen = torch.Generator(device=self.model.blocks[-1].device)
            gen.manual_seed(p.seed)
        with self.lock:
            logits = self.model.forward(torch.tensor([prompt_ids], dtype=torch.long), 0)
            pos = len(prompt_ids)
            out: list[int] = []
            decoded_upto = 0
            pending = ""
            for step in range(max_new):
                t = sample_token(logits[0], p.temperature, p.top_p, gen)
                if t == self.eos:
                    print(
                     f"[generate] STOP=eos step={step} pos={pos} "
                     f"max_new={max_new}",
                     flush=True,
                    )

                    break
                out.append(t)
                # decode incrementally; hold back a partial multi-byte character
                text = self.tok.decode(out[decoded_upto:])
                if "�" in text:
                    piece = ""
                else:
                    piece, decoded_upto = text, len(out)
                if piece:
                    pending += piece
                    if p.stop and any(s in pending for s in p.stop):
                        cut = min(pending.find(s) for s in p.stop if s in pending)
                        if cut > 0:
                            yield t, pending[:cut]
                        return
                    yield t, pending
                    pending = ""
                logits = self.rt.step(t, pos)
                pos += 1
            if out[decoded_upto:]:
                tail = self.tok.decode(out[decoded_upto:])
                if tail:
                    yield out[-1], tail

    def generate_text(self, prompt_ids: list[int], p: GenParams) -> tuple[str, int]:
        pieces, n = [], 0
        for _, piece in self.generate(prompt_ids, p):
            pieces.append(piece)
            n += 1
        return "".join(pieces), n


def parse_budgets(spec: str) -> dict[int, float] | None:
    return {int(k): float(v) for k, v in (kv.split(":") for kv in spec.split(",") if kv)} or None
