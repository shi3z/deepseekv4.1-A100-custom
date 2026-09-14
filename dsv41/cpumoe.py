"""ctypes wrapper for the CPU MoE library (dsv41/cpu/moe_cpu.cpp)."""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_lib = None


def lib():
    global _lib
    if _lib is None:
        src = os.path.join(HERE, "cpu", "moe_cpu.cpp")
        tag = hashlib.sha1(open(src, "rb").read()).hexdigest()[:12]
        so = os.path.join(HERE, "cpu", f".moe_cpu.{tag}.so")
        if not os.path.exists(so):
            # Default icelake-server: AVX-512 + VNNI + BF16 without requiring AMX (Sapphire Rapids).
            # Override with DSV41_CPU_MARCH=sapphirerapids|cooperlake|native|...
            march = os.environ.get("DSV41_CPU_MARCH", "icelake-server")
            subprocess.run(["g++", "-O3", f"-march={march}", "-fopenmp", "-shared", "-fPIC", "-o", so, src], check=True)
        _lib = ctypes.CDLL(so)
        _lib.cpumoe_alloc.restype = ctypes.c_void_p
        _lib.cpumoe_alloc.argtypes = [ctypes.c_size_t]
        _lib.cpumoe_bw_test.restype = ctypes.c_double
        _lib.cpumoe_bw_test.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int]
        _lib.cpumoe_load_rows.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        _lib.cpumoe_load_expert.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] + [ctypes.c_void_p] * 6 + [ctypes.c_int, ctypes.c_int]
        _lib.cpumoe_local_fraction.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        _lib.cpumoe_local_fraction.restype = ctypes.c_double
        _lib.cpumoe_forward.restype = ctypes.c_int
        _lib.cpumoe_forward_ids.restype = ctypes.c_int
        _lib.cpumoe_forward_ids.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_size_t] * 4 + [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p]
        _lib.cpumoe_set_int8(int(os.environ.get("DSV41_CPU_INT8", "2")))
        _lib.cpumoe_set_chunks(int(os.environ.get("DSV41_CH1", "96")), int(os.environ.get("DSV41_CH3", "64")))
        cpus = os.environ.get("DSV41_CPU_LIST") or default_cpu_list()  # e.g. "0-11,24-35,12-23,36-47" (node 0 first, then node 1)
        if cpus:
            lst = []
            for part in cpus.split(","):
                a, _, b = part.partition("-")
                lst.extend(range(int(a), int(b or a) + 1))
            arr = (ctypes.c_int * len(lst))(*lst)
            got = _lib.cpumoe_init_list(arr, len(lst), int(os.environ.get("DSV41_CORES_PER_NODE", str(len(lst) // 2))))
            assert got == len(lst), f"OpenMP gave {got} threads, wanted {len(lst)} (set OMP_THREAD_LIMIT / check cpu affinity)"
        else:
            threads = int(os.environ.get("DSV41_CPU_THREADS", "24"))
            cores_per_node = int(os.environ.get("DSV41_CORES_PER_NODE", "12"))
            _lib.cpumoe_init(threads, cores_per_node)
    return _lib


RESERVED_CPUS: list[int] = []  # hardware threads left free for the main thread / CUDA driver (see default_cpu_list)


def _expand(cpulist: str) -> list[int]:
    out = []
    for part in cpulist.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def _siblings(cpu: int) -> list[int]:
    try:
        return _expand(open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read().strip())
    except OSError:
        return [cpu]


def default_cpu_list(reserve_cores: int | None = None) -> str:
    """Worker cpus grouped by NUMA node (node 0 first) from sysfs, leaving `reserve_cores` whole physical cores
    (all their hardware threads) per node (default DSV41_RESERVE_CORES=1) for the Python main thread, the CUDA
    driver and the hot-cache copy thread: with active OpenMP waiting the workers spin between layers and would
    otherwise starve those threads, and a busy hyperthread sibling slows the worker sharing its core."""
    import glob
    nodes = sorted(glob.glob("/sys/devices/system/node/node[0-9]*"))
    per_node = []
    for n in nodes:
        try:
            per_node.append(_expand(open(os.path.join(n, "cpulist")).read().strip()))
        except OSError:
            return ""
    if not per_node:
        return ""
    r = int(os.environ.get("DSV41_RESERVE_CORES", "1")) if reserve_cores is None else reserve_cores
    workers, reserved = [], []
    m = None
    for c in per_node:
        cores = []  # distinct physical cores of this node, in cpu order
        seen = set()
        for x in c:
            if x in seen:
                continue
            sib = [y for y in _siblings(x) if y in c]
            seen.update(sib)
            cores.append(sib)
        keep = cores[: len(cores) - r] if r < len(cores) else cores
        drop = cores[len(cores) - r:] if r < len(cores) else []
        w = sorted(x for core in keep for x in core)
        workers.append(w)
        reserved.extend(x for core in drop for x in core)
        m = len(w) if m is None else min(m, len(w))
    RESERVED_CPUS[:] = reserved
    os.environ.setdefault("DSV41_CORES_PER_NODE", str(m))
    return ",".join(",".join(str(x) for x in w[:m]) for w in workers)


def pin_main_thread(node: int | None = None):
    """Pin the calling (main) thread to the reserved cpus (of `node`, if given and available)."""
    cpus = RESERVED_CPUS
    if node is not None:
        try:
            nc = set(_expand(open(f"/sys/devices/system/node/node{node}/cpulist").read().strip()))
            cpus = [c for c in RESERVED_CPUS if c in nc] or RESERVED_CPUS
        except OSError:
            pass
    if cpus:
        os.sched_setaffinity(0, set(cpus))


class HostExperts:
    """One layer's experts in NUMA-split host memory: w13 [E, N13, K/2], s13 [E, N13, K/32], w2 [E, dim, inter/2], s2."""
    _warned = False

    def __init__(self, E: int, inter: int, dim: int):
        self.E, self.inter, self.dim = E, inter, dim
        self.K = dim
        L = lib()
        self.n13 = 2 * inter
        self.rb13, self.rs13 = dim // 2, dim // 32
        self.rb2, self.rs2 = inter // 2, inter // 32
        self.bytes13 = self.n13 * self.rb13
        self.bytes_s13 = self.n13 * self.rs13
        self.bytes2 = dim * self.rb2
        self.bytes_s2 = dim * self.rs2
        self.w13 = L.cpumoe_alloc(E * self.bytes13)
        self.s13 = L.cpumoe_alloc(E * self.bytes_s13)
        self.w2 = L.cpumoe_alloc(E * self.bytes2)
        self.s2 = L.cpumoe_alloc(E * self.bytes_s2)
        assert self.w13 and self.s13 and self.w2 and self.s2, "host allocation failed"

        self.gu = torch.empty(16 * self.n13, dtype=torch.float32)
        self.h = torch.empty(16 * inter, dtype=torch.bfloat16)
        self.out = torch.empty(dim, dtype=torch.float32)
        self._lib = L
        self.gu_ptr, self.h_ptr = self.gu.data_ptr(), self.h.data_ptr()

    def load_expert(self, e: int, w1: torch.Tensor, w3: torch.Tensor, s1: torch.Tensor, s3: torch.Tensor, w2: torch.Tensor, s2: torch.Tensor):
        """CPU tensors (contiguous uint8/int8) straight from the checkpoint mmap."""
        L = lib()
        srcs = [t.contiguous().view(torch.uint8) for t in (w1, w3, s1, s3, w2, s2)]
        for t, n in zip(srcs, (self.inter * self.rb13, self.inter * self.rb13, self.inter * self.rs13, self.inter * self.rs13, self.dim * self.rb2, self.dim * self.rs2)):
            assert t.numel() == n, (t.shape, n)
        L.cpumoe_load_expert(ctypes.c_void_p(self.w13), ctypes.c_void_p(self.s13), ctypes.c_void_p(self.w2), ctypes.c_void_p(self.s2), e,
                             *[ctypes.c_void_p(t.data_ptr()) for t in srcs], self.inter, self.dim)

    def local_fraction(self) -> float:
        """Fraction of (sampled) pages that live on the NUMA node whose threads read them (1.0 = perfect)."""
        L = lib()
        fr = [L.cpumoe_local_fraction(ctypes.c_void_p(self.w13), self.E, self.n13, self.rb13), L.cpumoe_local_fraction(ctypes.c_void_p(self.w2), self.E, self.dim, self.rb2)]
        return sum(fr) / len(fr)

    def views(self):
        """torch uint8 views [E, N, ...] over the host buffers (no copy)."""
        def v(ptr, n, shape):
            return torch.frombuffer((ctypes.c_uint8 * n).from_address(ptr), dtype=torch.uint8).view(*shape)
        E = self.E
        return (v(self.w13, E * self.bytes13, (E, self.n13, self.rb13)), v(self.s13, E * self.bytes_s13, (E, self.n13, self.rs13)),
                v(self.w2, E * self.bytes2, (E, self.dim, self.rb2)), v(self.s2, E * self.bytes_s2, (E, self.dim, self.rs2)))

    def load_layer(self, w1: list, w3: list, s1: list, s3: list, w2: list, s2: list):
        """All experts of the layer at once (lists of E contiguous CPU tensors, e.g. mmap views of the checkpoint)."""
        L = lib()
        E = self.E
        assert len(w1) == E
        P = ctypes.c_void_p * E
        ptrs = [P(*[t.contiguous().data_ptr() for t in lst]) for lst in (w1, w3, s1, s3, w2, s2)]
        self._keep = (w1, w3, s1, s3, w2, s2)  # keep the source tensors alive during the copy
        L.cpumoe_load_layer(ctypes.c_void_p(self.w13), ctypes.c_void_p(self.s13), ctypes.c_void_p(self.w2), ctypes.c_void_p(self.s2),
                            *ptrs, E, self.inter, self.dim)
        self._keep = None

    def forward_ids(self, x_ptr: int, ids_ptr: int, wts_ptr: int, n: int, out_ptr: int, limit: float) -> None:
        """Lean entry for the decode loop: raw pointers to a bf16 [K] activation, int32 [n] expert ids, fp32 [n] weights
        and the fp32 [dim] output (all host memory, e.g. pinned tensors). No allocations, one ctypes call."""
        if n == 0:
            return
        r = self._lib.cpumoe_forward_ids(self.w13, self.s13, self.w2, self.s2, self.bytes13, self.bytes_s13, self.bytes2, self.bytes_s2,
                                         ids_ptr, n, x_ptr, wts_ptr, out_ptr, self.K, self.inter, self.dim, limit, self.gu_ptr, self.h_ptr)
        assert r == 0

    def forward(self, x_bf16: torch.Tensor, expert_ids: list[int], weights: list[float], limit: float) -> torch.Tensor:
        """x_bf16: CPU bf16 [K]; returns fp32 [dim] = sum_e w_e * expert_e(x)."""
        L = lib()
        E = len(expert_ids)
        P = ctypes.c_void_p * E
        p13 = P(*[self.w13 + e * self.bytes13 for e in expert_ids])
        ps13 = P(*[self.s13 + e * self.bytes_s13 for e in expert_ids])
        p2 = P(*[self.w2 + e * self.bytes2 for e in expert_ids])
        ps2 = P(*[self.s2 + e * self.bytes_s2 for e in expert_ids])
        wts = (ctypes.c_float * E)(*weights)
        x = x_bf16.contiguous()
        assert x.dtype == torch.bfloat16 and x.numel() == self.K
        r = L.cpumoe_forward(p13, ps13, p2, ps2, E, ctypes.c_void_p(x.data_ptr()), wts, ctypes.c_void_p(self.out.data_ptr()),
                             self.K, self.inter, self.dim, ctypes.c_float(limit),
                             ctypes.c_void_p(self.gu.data_ptr()), ctypes.c_void_p(self.h.data_ptr()))
        assert r == 0
        return self.out


def bw_test(gib: float = 8.0) -> dict:
    """Read bandwidth in GB/s: both nodes together, and each node alone (threads pinned, memory first-touched locally)."""
    L = lib()
    n = int(gib * 2**30)
    p = L.cpumoe_alloc(n)
    src = torch.ones(n, dtype=torch.uint8)
    L.cpumoe_load_rows(ctypes.c_void_p(p), ctypes.c_void_p(src.data_ptr()), 2, n // 2)  # first touch: half per node
    return {"both": L.cpumoe_bw_test(ctypes.c_void_p(p), n, 5, -1), "node0": L.cpumoe_bw_test(ctypes.c_void_p(p), n, 5, 0), "node1": L.cpumoe_bw_test(ctypes.c_void_p(p), n, 5, 1)}
