#!/usr/bin/env python3
"""
GLM-5 sparse MLA prefill benchmark — AMD gfx942.

Upstream: https://github.com/lixiuhong/llm_flops dsa_flashmla.py
Backend: PyTorch sparse gather attention (FlashMLA sparse_fwd not on ROCm).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)

from bench_utils import write_results
from common import (
    D_QK,
    KV_LORA_RANK,
    NUM_HEADS,
    NUM_RUNS_DEFAULT,
    NUM_WARMUP_DEFAULT,
    cuda_graph_bench,
    bootstrap,
)
import common as common_mod

H_Q = NUM_HEADS
H_KV = 1
D_V = KV_LORA_RANK
TOPK = 2048
TOTAL_LEN_DEFAULT = 65536
HIT_RATES_DEFAULT = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
HIT_RATES_QUICK = [0, 50, 90]
TOTAL_LEN_QUICK = 8192
SM_SCALE = D_QK ** -0.5


def make_tensors(s_q: int, s_kv: int, topk: int, device) -> tuple:
    import torch

    q = torch.randn(s_q, H_Q, D_QK, dtype=torch.bfloat16, device=device)
    kv = torch.randn(s_kv, H_KV, D_QK, dtype=torch.bfloat16, device=device)
    topk_actual = min(topk, s_kv)
    indices = torch.stack(
        [torch.randperm(s_kv, device=device)[:topk_actual] for _ in range(s_q * H_KV)]
    ).view(s_q, H_KV, topk_actual).to(torch.int64)
    return q, kv, indices, topk_actual


def bench_sparse_mla_pytorch(s_q: int, s_kv: int, topk: int, device) -> tuple:
    import torch

    q, kv, indices, topk_actual = make_tensors(s_q, s_kv, topk, device)
    kv_flat = kv[:, 0, :]
    idx = indices[:, 0, :].long()

    def run():
        gathered = kv_flat[idx]
        scores = torch.einsum("qhd,qkd->qhk", q, gathered) * SM_SCALE
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("qhk,qkd->qhd", attn, gathered[..., :D_V])
        return out

    avg_ms = cuda_graph_bench(run)

    flops = 2.0 * H_Q * s_q * topk_actual * (D_QK + D_V)
    kv_tokens = min(topk_actual * s_q, s_kv)
    mem_bytes = 2 * (H_Q * s_q * D_QK + kv_tokens * D_QK + H_Q * s_q * D_V)
    tflops = flops / (avg_ms * 1e-3) / 1e12
    tbps = mem_bytes / (avg_ms * 1e-3) / 1e12
    fpb = flops / mem_bytes if mem_bytes > 0 else 0.0
    del q, kv, indices
    return avg_ms, tflops, tbps, fpb


def main() -> None:
    bootstrap()

    parser = argparse.ArgumentParser(description="GLM-5 sparse MLA prefill (gfx942)")
    parser.add_argument("--total-len", type=int, default=TOTAL_LEN_DEFAULT)
    parser.add_argument("--hit-rates", type=int, nargs="+", default=HIT_RATES_DEFAULT)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--warmup", type=int, default=NUM_WARMUP_DEFAULT)
    parser.add_argument("--runs", type=int, default=NUM_RUNS_DEFAULT)
    parser.add_argument("--output-dir", type=str, default=os.path.join(_REPO, "results"))
    args = parser.parse_args()
    if args.quick:
        args.total_len = TOTAL_LEN_QUICK
        args.hit_rates = HIT_RATES_QUICK

    common_mod.NUM_WARMUP_DEFAULT = args.warmup
    common_mod.NUM_RUNS_DEFAULT = args.runs

    import torch

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print("=" * 100)
    print("GLM-5 Sparse MLA Prefill | AMD gfx942 | llm_flops_amd")
    print(f"  backend=pytorch_sparse_gather  total_len={args.total_len}  topk={TOPK}")
    print("=" * 100)
    print(f"{'hit%':>6} {'s_q':>8} {'s_kv':>8} {'avg_ms':>10} {'TFlops':>10} {'TB/s':>10}")
    print("-" * 60)

    all_results = []
    for hit_rate in args.hit_rates:
        s_q = int(args.total_len * (1 - hit_rate / 100))
        s_kv = args.total_len
        if s_q == 0:
            print(f"{hit_rate:>5d}%  (skip: s_q=0)")
            continue
        torch.cuda.empty_cache()
        try:
            avg_ms, tflops, tbps, fpb = bench_sparse_mla_pytorch(s_q, s_kv, TOPK, device)
            print(f"{hit_rate:>5d}% {s_q:>8d} {s_kv:>8d} {avg_ms:>10.3f} {tflops:>10.1f} {tbps:>10.3f}")
            all_results.append({
                "benchmark": "dsa_flashmla",
                "hit_rate": hit_rate,
                "s_q": s_q,
                "s_kv": s_kv,
                "topk": TOPK,
                "h_q": H_Q,
                "d_qk": D_QK,
                "d_v": D_V,
                "avg_ms": avg_ms,
                "tflops": tflops,
                "tbps": tbps,
                "fpb": fpb,
                "status": "OK",
            })
        except Exception as e:
            print(f"{hit_rate:>5d}%   FAILED: {e}")
            all_results.append({
                "benchmark": "dsa_flashmla",
                "hit_rate": hit_rate,
                "s_q": s_q,
                "s_kv": s_kv,
                "topk": TOPK,
                "avg_ms": 0.0,
                "status": "FAIL",
                "error": str(e)[:200],
            })
        time.sleep(0.1)

    write_results(all_results, args.output_dir, "glm5_sparse_prefill_amd")


if __name__ == "__main__":
    main()
