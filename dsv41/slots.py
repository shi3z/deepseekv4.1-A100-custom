"""Per-slot cache capacities (DSV41_SLOT_CAPS), e.g. "512k,128k,128k,128k,512k".

Every sequence slot gets its own context capacity instead of max_seq_len for all of them: slot 0 (the prefill
scratchpad, which must hold the largest request) and the decode slots 1.. in order.  The compressed-KV / index-K
caches become one flat buffer per (owner, device) in which slot s occupies k_s consecutive "units" of
unit_rows(ratio) rows (unit = the smallest capacity).  The kernels and the Python code keep addressing the cache
as [row, position]: the tensor they see is an as_strided view [n_units, max_rows, dim] with row stride = one
unit, so logical row r_s = (first unit of slot s) and position p land at flat row r_s * U + p, i.e. inside the
slot's own units as long as p < k_s * U.  Rows r_s + 1 .. r_s + k_s - 1 of the view alias the slot's later
units and are never used as sequence ids; the view has rows 0 .. r_last (n_units), and the buffer holds
r_last + max(k) units, which is exactly sum(caps) when the largest decode slot is listed last.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _tokens(x: str) -> int:
    x = x.strip().lower()
    if x.endswith("k"):
        return int(float(x[:-1]) * 1024)
    if x.endswith("m"):
        return int(float(x[:-1]) * 1024 * 1024)
    return int(x)


@dataclass(frozen=True)
class SlotLayout:
    caps: tuple[int, ...]   # tokens per slot; index 0 = prefill scratchpad
    unit: int               # smallest capacity (tokens); every capacity is a multiple of it
    k: tuple[int, ...]      # units per slot
    rows: tuple[int, ...]   # logical cache row (first unit) of each slot
    n_units: int            # logical rows of every per-sequence tensor = last slot's row + 1

    @property
    def decode_rows(self) -> list[int]:
        return list(self.rows[1:])

    @property
    def cap_by_row(self) -> dict[int, int]:
        return {r: c for r, c in zip(self.rows, self.caps)}

    def unit_rows(self, ratio: int) -> int:
        """Rows of one unit for an owner with compression `ratio` (k * unit_rows >= cap // ratio + 1 for every slot)."""
        return self.unit // max(1, int(ratio)) + 1

    def cap_rows(self, ratio: int) -> dict[int, int]:
        u = self.unit_rows(ratio)
        return {r: kk * u for r, kk in zip(self.rows, self.k)}

    def max_rows(self, ratio: int) -> int:
        return max(self.k) * self.unit_rows(ratio)

    @property
    def flat_units(self) -> int:
        """Units the flat buffer needs so that the [n_units, max_rows] view stays inside it: the last row plus the
        largest capacity.  Equals the sum of all capacities when the largest decode slot is listed last (no waste)."""
        return self.rows[-1] + max(self.k)

    def flat_rows(self, ratio: int) -> int:
        return self.flat_units * self.unit_rows(ratio)

    def describe(self) -> str:
        return ", ".join(f"slot{i}(row {r}): {c:,} tok" for i, (r, c) in enumerate(zip(self.rows, self.caps)))


def parse_slot_caps(spec: str, max_seq_len: int) -> SlotLayout:
    caps = [_tokens(x) for x in spec.split(",") if x.strip()]
    if len(caps) < 2:
        raise ValueError(f"DSV41_SLOT_CAPS needs the scratchpad slot and at least one decode slot: {spec!r}")
    unit = min(caps)
    if unit <= 0 or any(c % unit for c in caps):
        raise ValueError(f"DSV41_SLOT_CAPS: every capacity must be a multiple of the smallest one ({unit}): {caps}")
    if caps[0] < max(caps):
        raise ValueError(f"DSV41_SLOT_CAPS: slot 0 (prefill scratchpad) must be the largest capacity: {caps}")
    if max(caps) > max_seq_len:
        raise ValueError(f"DSV41_SLOT_CAPS: capacity {max(caps)} exceeds max_seq_len {max_seq_len}")
    k = [c // unit for c in caps]
    rows, acc = [], 0
    for kk in k:
        rows.append(acc)
        acc += kk
    return SlotLayout(tuple(caps), unit, tuple(k), tuple(rows), rows[-1] + 1)


def slot_layout_from_env(max_seq_len: int) -> SlotLayout | None:
    spec = os.environ.get("DSV41_SLOT_CAPS", "").strip()
    return parse_slot_caps(spec, max_seq_len) if spec else None
