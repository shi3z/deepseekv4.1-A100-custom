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
import traceback

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
                 offload_experts=False, hot_experts: int = 0, route_stats: str = "", ep: bool = False, ep_shards: list[int] | None = None, mtp: int = 0, mtp_device: int | None = None):
        from transformers import AutoTokenizer

        sys.path.insert(0, os.path.join(ckpt, "encoding"))
        from encoding import encode_messages, parse_message_from_completion_text  # type: ignore

        self._encode = encode_messages
        self._parse = parse_message_from_completion_text
        self.thinking_mode = thinking_mode
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.max_seq_len = max_seq_len
        self.mtp = int(mtp)
        self.mtp_device = mtp_device
        if self.mtp < 0 or self.mtp > 5:
            raise ValueError("--mtp must be 0..5")
        self.model = load_model(ckpt, devices or list(range(torch.cuda.device_count())), max_seq_len=max_seq_len, max_batch=1 + self.mtp, max_seqs=1,
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
        self.ds = None

        if self.mtp:
            if offload_experts:
                raise ValueError(
                    "--mtp is not supported with --offload-experts"
                )

            from .dspark import DSparkRows
            from .load import Checkpoint

            if self.mtp_device is None:
                mtp_dev = self.model.blocks[-1].device
            else:
                mtp_dev = torch.device(
                    f"cuda:{self.mtp_device}"
                )

            free_b, total_b = torch.cuda.mem_get_info(mtp_dev)

            print(
                f"[mtp] loading DSpark on {mtp_dev} "
                f"free={free_b / 2**30:.2f} GiB "
                f"total={total_b / 2**30:.2f} GiB",
                flush=True,
            )

            last = mtp_dev

            self.ds = DSparkRows(
                Checkpoint(ckpt),
                self.model.args,
                last,
                self.model.embed,
                self.model.head,
                self.model.shared,
                len(self.model.blocks),
                self.rt,
            )

            # Tell eager prefill to retain the three target-layer
            # hidden states needed to seed DSpark.
            self.model.collect_main_hidden = self.ds.targets

            # DSpark itself always has a 5-token draft block.
            self.ds.capture(1)

            print(
                f"[mtp] enabled drafts={self.mtp} "
                f"verify_rows={1 + self.mtp}",
                flush=True,
            )
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
            if self.mtp:
                yield from self._generate_mtp_locked(
                    prompt_ids,
                    p,
                    max_new,
                    gen,
                )
                return

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

    def _generate_mtp_locked(
        self,
        prompt_ids: list[int],
        p: GenParams,
        max_new: int,
        gen: torch.Generator | None,
    ) -> Iterator[tuple[int, str]]:
        """Single-sequence DSpark speculative decoding.

        The caller already holds self.lock.
        """

        ds = self.ds
        rt = self.rt
        model = self.model

        assert ds is not None

        # ------------------------------------------------------------
        # Prefill
        # ------------------------------------------------------------

        logits = model.forward(
            torch.tensor(
                [prompt_ids],
                dtype=torch.long,
            ),
            0,
        )

        T = len(prompt_ids)
        last = model.blocks[-1].device

        if not hasattr(model, "main_hidden"):
            raise RuntimeError(
                "MTP prefill did not collect model.main_hidden"
            )

        # DSpark window attention only needs the tail of the main
        # sequence. Avoid feeding tens of thousands of rows through
        # DSpark just to overwrite the same ring.
        mh = model.main_hidden[0]

        win = int(ds.win)
        first = max(0, T - win)

        tail = mh[first:T]

        ds.write_main_rows(
            tail,
            torch.zeros(
                T - first,
                dtype=torch.int64,
                device=last,
            ),
            torch.arange(
                first,
                T,
                dtype=torch.int64,
                device=last,
            ),
        )

        # Hidden state whose forward produced the first sampled token.
        main_h = mh[-1].clone()

        # Large prefill hidden capture is no longer needed.
        del model.main_hidden
        del mh, tail

        p_last = T - 1
        written_max = p_last

        current = sample_token(
            logits[0],
            p.temperature,
            p.top_p,
            gen,
        )

        # ------------------------------------------------------------
        # Streaming decoder state
        # ------------------------------------------------------------

        out: list[int] = []
        decoded_upto = 0
        pending = ""
        produced = 0

        n_verify_steps = 0
        n_accepted = 0

        def emit(tokens):
            nonlocal decoded_upto
            nonlocal pending
            nonlocal produced

            for t in tokens:
                if produced >= max_new:
                    return True

                if t == self.eos:
                    print(
                        f"[generate] STOP=eos "
                        f"produced={produced} "
                        f"mtp_accept={n_accepted}/{n_verify_steps}",
                        flush=True,
                    )
                    return True

                out.append(int(t))
                produced += 1

                text = self.tok.decode(
                    out[decoded_upto:]
                )

                if "�" in text:
                    piece = ""
                else:
                    piece = text
                    decoded_upto = len(out)

                if piece:
                    pending += piece

                    if p.stop and any(
                        s in pending
                        for s in p.stop
                    ):
                        cut = min(
                            pending.find(s)
                            for s in p.stop
                            if s in pending
                        )

                        if cut > 0:
                            yield int(t), pending[:cut]

                        return True

                    yield int(t), pending
                    pending = ""

            return False

        # First token came directly from the prefill logits.
        stopped = yield from emit([current])

        if stopped:
            return

        # ------------------------------------------------------------
        # Speculative loop
        # ------------------------------------------------------------

        while produced < max_new:

            try:
                draft_all = ds.draft_rows(
                    torch.tensor(
                        [current],
                        dtype=torch.int64,
                        device=last,
                    ),
                    torch.tensor(
                        [p_last],
                        dtype=torch.int64,
                        device=last,
                    ),
                    main_h.view(1, -1),
                    torch.tensor(
                        [written_max],
                        dtype=torch.int64,
                        device=last,
                    ),
                )
            except Exception as e:
                print(
                    "\n[MTP DRAFT ERROR]",
                    repr(e),
                    flush=True,
                )
                traceback.print_exc()
                raise

            drafts = (
                draft_all[0, :self.mtp]
                .to("cpu")
                .tolist()
            )

            K = 1 + len(drafts)

            row_tokens = [current] + drafts

            positions = list(
                range(
                    p_last + 1,
                    p_last + 1 + K,
                )
            )

            pmax = p_last + K

            verify_logits = rt.step(
                row_tokens,
                positions,
                seq=[0] * K,
                pmax=[pmax] * K,
            )

            # Target-layer hidden states corresponding to every
            # verifier row. Needed for the next DSpark invocation.
            mh_all = torch.cat(
                [
                    rt.main_hid[lid].to(last)
                    for lid in ds.targets
                ],
                dim=-1,
            )

            ds.write_main_rows(
                mh_all,
                torch.zeros(
                    K,
                    dtype=torch.int64,
                    device=last,
                ),
                torch.tensor(
                    positions,
                    dtype=torch.int64,
                    device=last,
                ),
            )

            # Sample verifier outputs exactly as the ordinary engine
            # would sample them.
            verified = [
                sample_token(
                    verify_logits[i],
                    p.temperature,
                    p.top_p,
                    gen,
                )
                for i in range(K)
            ]

            acc = 0

            for i, draft in enumerate(drafts):
                if verified[i] == draft:
                    acc += 1
                else:
                    break

            # Accepted drafts followed by the verifier's first
            # non-draft token (or the bonus token after all accepted).
            new_tokens = drafts[:acc] + [verified[acc]]

            n_verify_steps += 1
            n_accepted += acc

            # Row `acc` is the forward pass which produced the new
            # bonus token.
            main_h = mh_all[acc].clone()

            # mtp_run.py uses this same logical state update.
            old_p = p_last
            p_last = old_p + 1 + acc
            written_max = old_p + K

            current = verified[acc]

            stopped = yield from emit(new_tokens)

            if stopped:
                return

        if out[decoded_upto:]:
            tail = self.tok.decode(
                out[decoded_upto:]
            )

            if tail:
                yield out[-1], tail

        if n_verify_steps:
            print(
                f"[mtp] steps={n_verify_steps} "
                f"accepted={n_accepted} "
                f"accepted/step={n_accepted / n_verify_steps:.2f} "
                f"tokens={produced}",
                flush=True,
            )

    def generate_text(self, prompt_ids: list[int], p: GenParams) -> tuple[str, int]:
        pieces, n = [], 0
        for _, piece in self.generate(prompt_ids, p):
            pieces.append(piece)
            n += 1
        return "".join(pieces), n


def parse_budgets(spec: str) -> dict[int, float] | None:
    return {int(k): float(v) for k, v in (kv.split(":") for kv in spec.split(",") if kv)} or None
