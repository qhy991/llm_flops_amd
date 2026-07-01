#!/usr/bin/env python3
"""
GLM-5 DSA Indexer GEMM benchmark — AMD gfx942.

Upstream: https://github.com/lixiuhong/llm_flops dsa_indexer.py
Backend: AITER FP8 GEMM (replaces cuBLAS _scaled_mm on CUDA).
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)

from bench_utils import compute_gemm_metrics, write_results
from common import (
    HIDDEN_DIM,
    INDEX_HEAD_DIM,
    INDEX_N_HEADS,
    NUM_RUNS_DEFAULT,
    NUM_WARMUP_DEFAULT,
    Q_LORA_RANK,
    bootstrap,
)
import common as common_mod
from backends import bench_bf16_gemm, bench_fp8_gemm, gemm_backend_name, init_backends

M_LIST_DEFAULT = [16, 256, 512, 1024]
S_LIST_DEFAULT = [65536, 65536 * 2, 65536 * 4]
M_LIST_QUICK = [16, 256]
S_LIST_QUICK = [2048, 8192]
VRAM_LIMIT_BYTES = 60 * (1024 ** 3)

GEMM_SPECS = [
    ("index_k_proj", lambda m, s: s, HIDDEN_DIM, lambda m, s: INDEX_HEAD_DIM, "fp8"),
    ("index_q_upproj", lambda m, s: m, Q_LORA_RANK, lambda m, s: INDEX_N_HEADS * INDEX_HEAD_DIM, "fp8"),
    ("index_weights_proj", lambda m, s: m, HIDDEN_DIM, lambda m, s: INDEX_N_HEADS, "bf16"),
    ("index_score", lambda m, s: INDEX_N_HEADS * m, INDEX_HEAD_DIM, lambda m, s: s, "fp8"),
]


def main() -> None:
    bootstrap()
    init_backends()

    parser = argparse.ArgumentParser(description="GLM-5 DSA indexer GEMM (gfx942)")
    parser.add_argument("--m-list", type=int, nargs="+", default=M_LIST_DEFAULT)
    parser.add_argument("--s-list", type=int, nargs="+", default=S_LIST_DEFAULT)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--warmup", type=int, default=NUM_WARMUP_DEFAULT)
    parser.add_argument("--runs", type=int, default=NUM_RUNS_DEFAULT)
    parser.add_argument("--output-dir", type=str, default=os.path.join(_REPO, "results"))
    args = parser.parse_args()
    if args.quick:
        args.m_list = M_LIST_QUICK
        args.s_list = S_LIST_QUICK

    common_mod.NUM_WARMUP_DEFAULT = args.warmup
    common_mod.NUM_RUNS_DEFAULT = args.runs

    import torch

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print("=" * 120)
    print("GLM-5 DSA Indexer GEMM | AMD gfx942 | llm_flops_amd")
    print(f"  GEMM={gemm_backend_name()}")
    print(f"  M={args.m_list}  S={args.s_list}")
    print("=" * 120)

    all_results = []
    for m in args.m_list:
        for s in args.s_list:
            print(f"\n--- M={m}, S={s} ---")
            for spec_name, m_fn, k, n_fn, dtype in GEMM_SPECS:
                gemm_m = m_fn(m, s)
                gemm_n = n_fn(m, s)
                n_heads_run = 1
                note = ""
                if spec_name == "index_score":
                    out_bytes = gemm_m * gemm_n * 2
                    if out_bytes > VRAM_LIMIT_BYTES:
                        gemm_m = m
                        n_heads_run = INDEX_N_HEADS
                        note = f" (per-head x{INDEX_N_HEADS})"

                shape = f"[{m_fn(m, s)},{k}]×[{k},{n_fn(m, s)}]"
                try:
                    if dtype == "bf16":
                        avg_one = bench_bf16_gemm(gemm_m, k, gemm_n, device)
                    else:
                        avg_one = bench_fp8_gemm(gemm_m, k, gemm_n, device)

                    if n_heads_run > 1:
                        avg_ms = avg_one * n_heads_run
                        full_m = INDEX_N_HEADS * m
                        metrics = compute_gemm_metrics(full_m, k, gemm_n, avg_ms)
                    else:
                        avg_ms = avg_one
                        metrics = compute_gemm_metrics(gemm_m, k, gemm_n, avg_ms)

                    print(
                        f"  {spec_name + note:<24} {avg_ms:>10.4f} ms  "
                        f"{metrics['tflops']:>8.1f} TFlops"
                    )
                    all_results.append({
                        "benchmark": "dsa_indexer",
                        "name": spec_name,
                        "M": m,
                        "S": s,
                        "gemm_M": m_fn(m, s),
                        "K": k,
                        "gemm_N": n_fn(m, s),
                        "shape": shape,
                        "avg_ms": avg_ms,
                        "status": "OK",
                        "note": note.strip(),
                        **metrics,
                    })
                except Exception as e:
                    print(f"  {spec_name:<24}   FAILED: {e}")
                    all_results.append({
                        "benchmark": "dsa_indexer",
                        "name": spec_name,
                        "M": m,
                        "S": s,
                        "gemm_M": m_fn(m, s),
                        "K": k,
                        "gemm_N": n_fn(m, s),
                        "shape": shape,
                        "avg_ms": 0.0,
                        "status": "FAIL",
                        "error": str(e)[:200],
                        "tflops": 0.0,
                        "tbps": 0.0,
                        "fpb": 0.0,
                    })

    write_results(all_results, args.output_dir, "glm5_dsa_indexer_amd")


if __name__ == "__main__":
    main()
