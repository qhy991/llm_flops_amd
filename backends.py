"""gfx942 kernel backend helpers for GLM-5 micro-benchmarks."""
from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch

from common import (
    BLOCK_SIZE_GEMM,
    BLOCK_SIZE_KV,
    D_QK,
    HIDDEN_DIM,
    MOE_INTERMEDIATE_SIZE,
    NUM_EXPERTS_PER_TOK,
    NUM_HEADS,
    QK_NOPE_HEAD_DIM,
    V_HEAD_DIM,
    cuda_graph_bench,
    prepare_sglang_fp8_weight,
)

AITER_GEMM_FN: Optional[Callable] = None
TRITON_GEMM_FN: Optional[Callable] = None
BMM_FP8_FN: Optional[Callable] = None


def init_backends() -> None:
    global AITER_GEMM_FN, TRITON_GEMM_FN, BMM_FP8_FN

    try:
        from sglang.srt.layers.quantization.fp8_utils import aiter_w8a8_block_fp8_linear
        AITER_GEMM_FN = aiter_w8a8_block_fp8_linear
    except ImportError:
        pass

    try:
        from sglang.srt.layers.quantization.fp8_utils import triton_w8a8_block_fp8_linear
        TRITON_GEMM_FN = triton_w8a8_block_fp8_linear
    except ImportError:
        pass

    try:
        import sgl_kernel
        if hasattr(sgl_kernel, "bmm_fp8"):
            BMM_FP8_FN = sgl_kernel.bmm_fp8
    except ImportError:
        pass

    if torch.version.hip:
        BMM_FP8_FN = None


def pick_fp8_gemm() -> Tuple[Callable, str]:
    if AITER_GEMM_FN is not None:
        return AITER_GEMM_FN, "aiter"
    if TRITON_GEMM_FN is not None:
        return TRITON_GEMM_FN, "triton"
    raise RuntimeError("No FP8 GEMM backend (set SGLANG_PATH and SGLANG_USE_AITER=1)")


def gemm_backend_name() -> str:
    return pick_fp8_gemm()[1]


def bmm_backend_name() -> str:
    return "sgl_kernel.bmm_fp8" if BMM_FP8_FN else "torch.bmm"


def _cast_bmm_fp8(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    fp8_dtype = torch.float8_e4m3fnuz if torch.version.hip else torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max
    abs_max = tensor.abs().amax()
    scale = (abs_max / fp8_max).clamp(min=1e-12).float()
    return (tensor / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype), scale.unsqueeze(0)


def bench_fp8_gemm(m: int, k: int, n: int, device: torch.device) -> float:
    gemm_fn, _ = pick_fp8_gemm()
    x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
    _, w_fp8, w_scale = prepare_sglang_fp8_weight(n, k, device=device)
    return cuda_graph_bench(
        lambda: gemm_fn(x, w_fp8, BLOCK_SIZE_GEMM, w_scale, input_scale=None, bias=None)
    )


def bench_bf16_gemm(m: int, k: int, n: int, device: torch.device) -> float:
    x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
    w = torch.randn(n, k, dtype=torch.bfloat16, device=device)
    out = torch.empty(m, n, dtype=torch.bfloat16, device=device)
    return cuda_graph_bench(lambda: torch.mm(x, w.t(), out=out))


def bench_bmm(batch: int, m: int, k: int, n: int, device: torch.device) -> float:
    if BMM_FP8_FN is None:
        a = torch.randn(batch, m, k, dtype=torch.bfloat16, device=device)
        b = torch.randn(batch, k, n, dtype=torch.bfloat16, device=device)
        return cuda_graph_bench(lambda: torch.bmm(a, b))

    a_bf16 = torch.randn(batch, m, k, dtype=torch.bfloat16, device=device)
    b_bf16 = torch.randn(batch, k, n, dtype=torch.bfloat16, device=device)
    a_fp8, a_scale = _cast_bmm_fp8(a_bf16.reshape(-1, k))
    b_fp8, b_scale = _cast_bmm_fp8(b_bf16.reshape(-1, n))
    a_fp8 = a_fp8.view(batch, m, k)
    b_fp8 = b_fp8.view(batch, k, n)
    return cuda_graph_bench(
        lambda: BMM_FP8_FN(a_fp8, b_fp8, a_scale, b_scale, torch.bfloat16)
    )


def bench_mla_decode_pytorch(m: int, s: int, device: torch.device) -> float:
    head_dim = D_QK
    v_head_dim = V_HEAD_DIM
    num_pages = max(1, (s + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV)
    kv_cache = torch.randn(num_pages * BLOCK_SIZE_KV, head_dim, device=device, dtype=torch.bfloat16)
    kv_indices = torch.arange(min(s, kv_cache.shape[0]), device=device, dtype=torch.int32)
    q = torch.randn(m, NUM_HEADS, head_dim, device=device, dtype=torch.bfloat16)
    o = torch.empty(m, NUM_HEADS, v_head_dim, device=device, dtype=torch.bfloat16)
    kv_flat = kv_cache.view(-1, head_dim)

    if m == 1:
        kv_seq = kv_flat[kv_indices.long()]

        def run_bs1():
            scores = torch.matmul(q[0], kv_seq.T)
            attn = torch.softmax(scores, dim=-1)
            o[0] = torch.matmul(attn, kv_seq[:, :v_head_dim])

        return cuda_graph_bench(run_bs1)

    def run_multi():
        for i in range(m):
            kv_seq = kv_flat[kv_indices.long()]
            scores = torch.matmul(q[i], kv_seq.T)
            attn = torch.softmax(scores, dim=-1)
            o[i] = torch.matmul(attn, kv_seq[:, :v_head_dim])

    return cuda_graph_bench(run_multi)


def bench_mla_prefill_pytorch(m: int, device: torch.device) -> float:
    """Causal prefill attention (PyTorch reference, graph-captured)."""
    scale = 1.0 / math.sqrt(D_QK)
    q = torch.randn(1, NUM_HEADS, m, D_QK, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, NUM_HEADS, m, D_QK, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, NUM_HEADS, m, V_HEAD_DIM, device=device, dtype=torch.bfloat16)
    mask = torch.triu(torch.ones(m, m, device=device, dtype=torch.bool), diagonal=1)

    def run():
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores.masked_fill_(mask.view(1, 1, m, m), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        torch.matmul(attn, v)

    return cuda_graph_bench(run)


def bench_moe_dense_triple(m_tokens: int, device: torch.device) -> Tuple[float, float, float]:
    return (
        bench_fp8_gemm(m_tokens, HIDDEN_DIM, MOE_INTERMEDIATE_SIZE, device),
        bench_fp8_gemm(m_tokens, HIDDEN_DIM, MOE_INTERMEDIATE_SIZE, device),
        bench_fp8_gemm(m_tokens, MOE_INTERMEDIATE_SIZE, HIDDEN_DIM, device),
    )


def bench_moe_grouped_aiter(
    m_per_expert: list,
    k: int,
    n: int,
    device: torch.device,
) -> Tuple[float, int]:
    """Grouped MoE GEMM via per-expert AITER FP8 slices (ROCm analogue to DeepGEMM grouped)."""
    gemm_fn, _ = pick_fp8_gemm()
    alignment = 128
    aligned_m = [(((m + alignment - 1) // alignment) * alignment) if m > 0 else 0 for m in m_per_expert]
    total_m = sum(aligned_m)

    xs = []
    ws = []
    for m in aligned_m:
        if m <= 0:
            xs.append(None)
            ws.append(None)
            continue
        x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        _, w_fp8, w_scale = prepare_sglang_fp8_weight(n, k, device=device)
        xs.append(x)
        ws.append((w_fp8, w_scale))

    def run():
        for x, w in zip(xs, ws):
            if x is None:
                continue
            w_fp8, w_scale = w
            gemm_fn(x, w_fp8, BLOCK_SIZE_GEMM, w_scale, input_scale=None, bias=None)

    avg_ms = cuda_graph_bench(run)
    return avg_ms, total_m
