#!/usr/bin/env python3
"""
GLM-5 MoE grouped GEMM benchmark — AMD gfx942.

Upstream: https://github.com/lixiuhong/llm_flops moe_deepgemm.py
Backend: per-expert AITER FP8 GEMM slices (ROCm analogue to DeepGEMM grouped).
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)

from bench_utils import write_results
from common import (
    HIDDEN_DIM,
    MOE_INTERMEDIATE_SIZE,
    N_EXPERT_LOCAL,
    NUM_RUNS_DEFAULT,
    NUM_WARMUP_DEFAULT,
    bootstrap,
)
import common as common_mod
from backends import bench_moe_grouped_aiter, gemm_backend_name, init_backends

TOTAL_TOKENS_DEFAULT = 16 * N_EXPERT_LOCAL
NUM_DISTRIBUTIONS_DEFAULT = 5


def generate_random_m_per_expert(total_tokens: int, n_expert: int) -> list:
    counts = [0] * n_expert
    for _ in range(total_tokens):
        counts[random.randint(0, n_expert - 1)] += 1
    return counts


def format_distribution(m_per_expert: list) -> str:
    avg = statistics.mean(m_per_expert)
    std = statistics.stdev(m_per_expert) if len(m_per_expert) > 1 else 0.0
    sorted_m = sorted(m_per_expert, reverse=True)
    nonzero = sum(1 for m in m_per_expert if m > 0)
    return (
        f"min={min(m_per_expert)}, max={max(m_per_expert)}, avg={avg:.0f}, std={std:.0f}, "
        f"nonzero={nonzero}/{len(m_per_expert)}, top5={sorted_m[:5]}"
    )


def main() -> None:
    bootstrap()
    init_backends()

    parser = argparse.ArgumentParser(description="GLM-5 MoE grouped GEMM (gfx942)")
    parser.add_argument("--total-tokens", type=int, default=TOTAL_TOKENS_DEFAULT)
    parser.add_argument("--num-distributions", type=int, default=NUM_DISTRIBUTIONS_DEFAULT)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--warmup", type=int, default=NUM_WARMUP_DEFAULT)
    parser.add_argument("--runs", type=int, default=NUM_RUNS_DEFAULT)
    parser.add_argument("--output-dir", type=str, default=os.path.join(_REPO, "results"))
    args = parser.parse_args()
    if args.quick:
        args.total_tokens = 8 * N_EXPERT_LOCAL
        args.num_distributions = 2

    common_mod.NUM_WARMUP_DEFAULT = args.warmup
    common_mod.NUM_RUNS_DEFAULT = args.runs

    import torch

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    proj_configs = [
        ("gate_proj", HIDDEN_DIM, MOE_INTERMEDIATE_SIZE),
        ("up_proj", HIDDEN_DIM, MOE_INTERMEDIATE_SIZE),
        ("down_proj", MOE_INTERMEDIATE_SIZE, HIDDEN_DIM),
    ]
    distributions = [
        generate_random_m_per_expert(args.total_tokens, N_EXPERT_LOCAL)
        for _ in range(args.num_distributions)
    ]

    print("=" * 110)
    print("GLM-5 MoE Grouped GEMM | AMD gfx942 | llm_flops_amd")
    print(f"  GEMM={gemm_backend_name()}  experts={N_EXPERT_LOCAL}  total_tokens={args.total_tokens}")
    print(f"  distributions={args.num_distributions}")
    print("=" * 110)

    all_results = []
    for proj_name, k, n in proj_configs:
        print(f"\n=== {proj_name}: K={k}, N={n} ===")
        for dist_idx, m_per_expert in enumerate(distributions):
            print(f"\n  [dist {dist_idx}] {format_distribution(m_per_expert)}")
            try:
                avg_ms, total_m = bench_moe_grouped_aiter(m_per_expert, k, n, device)
                flops = 2.0 * total_m * n * k
                mem = total_m * k + N_EXPERT_LOCAL * n * k + total_m * n * 2
                tflops = flops / (avg_ms * 1e-3) / 1e12
                tbps = mem / (avg_ms * 1e-3) / 1e12
                fpb = flops / mem if mem > 0 else 0.0
                print(
                    f"  -> total_m={total_m}, avg={avg_ms:.3f} ms, "
                    f"{tflops:.1f} TFlops, {tbps:.3f} TB/s"
                )
                all_results.append({
                    "benchmark": "moe_deepgemm",
                    "proj": proj_name,
                    "dist_idx": dist_idx,
                    "total_m": total_m,
                    "K": k,
                    "N": n,
                    "m_min": min(m_per_expert),
                    "m_max": max(m_per_expert),
                    "m_avg": sum(m_per_expert) // N_EXPERT_LOCAL,
                    "avg_ms": avg_ms,
                    "tflops": tflops,
                    "tbps": tbps,
                    "fpb": fpb,
                    "m_per_expert": m_per_expert,
                    "status": "OK",
                })
            except Exception as e:
                print(f"  -> FAILED: {e}")
                all_results.append({
                    "benchmark": "moe_deepgemm",
                    "proj": proj_name,
                    "dist_idx": dist_idx,
                    "total_m": 0,
                    "K": k,
                    "N": n,
                    "avg_ms": 0.0,
                    "status": "FAIL",
                    "error": str(e)[:200],
                    "m_per_expert": m_per_expert,
                })
            time.sleep(0.1)

    write_results(all_results, args.output_dir, "glm5_moe_deepgemm_amd")


if __name__ == "__main__":
    main()
