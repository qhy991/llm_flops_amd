"""
Shared utilities for GLM-5 operator benchmarks on AMD MI300X (gfx942).

Adapted from https://github.com/lixiuhong/llm_flops for ROCm / AITER / SGLang.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Bootstrap: add AITER + SGLang to sys.path
# ---------------------------------------------------------------------------

def bootstrap(
    aiter_path: Optional[str] = None,
    sglang_path: Optional[str] = None,
) -> None:
    os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx942")
    os.environ.setdefault("SGLANG_USE_AITER", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    aiter_path = aiter_path or os.environ.get("AITER_PATH", "/root/repos/aiter")
    sglang_path = sglang_path or os.environ.get("SGLANG_PATH", "/root/repos/sglang/python")
    for p in (aiter_path, sglang_path):
        if p and p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# GLM-5 model constants (aligned with llm_flops)
# ---------------------------------------------------------------------------

HIDDEN_DIM = 6144
Q_LORA_RANK = 2048
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 192
V_HEAD_DIM = 256
NUM_HEADS = 64
INDEX_N_HEADS = 32
INDEX_HEAD_DIM = 128
MOE_INTERMEDIATE_SIZE = 2048
NUM_EXPERTS_PER_TOK = 8
N_EXPERT_LOCAL = 8  # single-GPU micro-bench

QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
FUSED_QKV_A_OUT = Q_LORA_RANK + KV_LORA_RANK + QK_ROPE_HEAD_DIM
D_QK = KV_LORA_RANK + QK_ROPE_HEAD_DIM
BLOCK_SIZE_KV = 64
BLOCK_SIZE_GEMM = [128, 128]

NUM_WARMUP_DEFAULT = 5
NUM_RUNS_DEFAULT = 20

PREFILL_M_LIST_DEFAULT = [1024, 2048, 4096]
DECODE_M_LIST_DEFAULT = [1, 4, 8, 16, 32]
KV_S_LIST_DEFAULT = [2048, 8192]


def cdiv(a: int, b: int) -> int:
    return -(a // -b)


def cuda_graph_bench(
    run_fn: Callable[[], None],
    num_warmup: int = NUM_WARMUP_DEFAULT,
    num_runs: int = NUM_RUNS_DEFAULT,
) -> float:
    """Warmup, CUDA graph capture + replay, return avg ms per iteration."""
    torch.cuda.synchronize()
    for _ in range(num_warmup):
        run_fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(num_runs):
            run_fn()
    torch.cuda.synchronize()

    for _ in range(num_warmup):
        graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()

    avg_ms = start.elapsed_time(end) / num_runs
    del graph
    return avg_ms


def prepare_sglang_fp8_weight(
    N: int,
    K: int,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SGLang-style FP8 block-scaled weight [N, K]."""
    if device is None:
        device = torch.device("cuda")

    fp8_dtype = torch.float8_e4m3fnuz if torch.version.hip else torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max
    block_n, block_k = BLOCK_SIZE_GEMM
    n_blocks = cdiv(N, block_n)
    k_blocks = cdiv(K, block_k)

    w_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16)
    n_pad = n_blocks * block_n
    k_pad = k_blocks * block_k
    w_padded = torch.zeros(n_pad, k_pad, device=device, dtype=torch.float32)
    w_padded[:N, :K] = w_bf16.float()

    reshaped = w_padded.reshape(n_blocks, block_n, k_blocks, block_k)
    abs_max = reshaped.abs().amax(dim=(1, 3))
    weight_scale = (abs_max / fp8_max).clamp(min=1e-12).float()

    scaled = reshaped / weight_scale[:, None, :, None]
    w_fp8 = (
        scaled.reshape(n_pad, k_pad)[:N, :K]
        .clamp(-fp8_max, fp8_max)
        .to(fp8_dtype)
    )
    return w_bf16, w_fp8, weight_scale
