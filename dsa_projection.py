#!/usr/bin/env python3
"""
GLM-5 Attention GEMM/BMM benchmark — AMD gfx942.

Upstream: https://github.com/lixiuhong/llm_flops dsa_projection.py
Backend: AITER FP8 GEMM + torch.bmm (HIP has no bmm_fp8).
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
    KV_LORA_RANK,
    NUM_HEADS,
    NUM_RUNS_DEFAULT,
    NUM_WARMUP_DEFAULT,
    QK_HEAD_DIM,
    QK_NOPE_HEAD_DIM,
    QK_ROPE_HEAD_DIM,
    Q_LORA_RANK,
    V_HEAD_DIM,
    bootstrap,
)
import common as common_mod
from backends import bench_bmm, bench_fp8_gemm, bmm_backend_name, gemm_backend_name, init_backends

M_LIST_DEFAULT = [1024, 4096, 16384, 65536]
M_LIST_QUICK = [1024, 4096]

ATTN_SPECS = [
    ("q_a_proj", "gemm", lambda m: (m, HIDDEN_DIM, Q_LORA_RANK)),
    ("q_b_proj", "gemm", lambda m: (m, Q_LORA_RANK, NUM_HEADS * QK_HEAD_DIM)),
    ("absorbed_W_UK", "bmm", lambda m: (NUM_HEADS, m, QK_NOPE_HEAD_DIM, KV_LORA_RANK)),
    ("kv_a_proj", "gemm", lambda m: (m, HIDDEN_DIM, KV_LORA_RANK + QK_ROPE_HEAD_DIM)),
    ("absorbed_W_UV", "bmm", lambda m: (NUM_HEADS, m, KV_LORA_RANK, V_HEAD_DIM)),
    ("o_proj", "gemm", lambda m: (m, NUM_HEADS * V_HEAD_DIM, HIDDEN_DIM)),
]


def main() -> None:
    bootstrap()
    init_backends()

    parser = argparse.ArgumentParser(description="GLM-5 attention GEMM/BMM (gfx942)")
    parser.add_argument("--m-list", type=int, nargs="+", default=M_LIST_DEFAULT)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--warmup", type=int, default=NUM_WARMUP_DEFAULT)
    parser.add_argument("--runs", type=int, default=NUM_RUNS_DEFAULT)
    parser.add_argument("--output-dir", type=str, default=os.path.join(_REPO, "results"))
    args = parser.parse_args()
    if args.quick:
        args.m_list = M_LIST_QUICK

    common_mod.NUM_WARMUP_DEFAULT = args.warmup
    common_mod.NUM_RUNS_DEFAULT = args.runs

    import torch

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print("=" * 120)
    print("GLM-5 Attention GEMM/BMM | AMD gfx942 | llm_flops_amd")
    print(f"  GEMM={gemm_backend_name()}  BMM={bmm_backend_name()}")
    print(f"  M={args.m_list}  warmup={args.warmup} runs={args.runs}")
    print("=" * 120)

    all_results = []
    for m in args.m_list:
        print(f"\n--- M={m} ---")
        for name, spec_type, params_fn in ATTN_SPECS:
            try:
                if spec_type == "gemm":
                    gM, k, n = params_fn(m)
                    avg_ms = bench_fp8_gemm(gM, k, n, device)
                    shape = f"[{gM},{k}]×[{k},{n}]"
                    metrics = compute_gemm_metrics(gM, k, n, avg_ms)
                    type_label = "GEMM"
                else:
                    batch, gM, k, n = params_fn(m)
                    avg_ms = bench_bmm(batch, gM, k, n, device)
                    shape = f"batch={batch}, [{gM},{k}]×[{k},{n}]"
                    flops = 2.0 * batch * gM * n * k
                    mem = batch * (gM * k + n * k + gM * n * 2)
                    ms = max(avg_ms, 1e-9)
                    metrics = {
                        "tflops": flops / (ms * 1e-3) / 1e12,
                        "tbps": mem / (ms * 1e-3) / 1e12,
                        "fpb": flops / mem if mem > 0 else 0.0,
                    }
                    type_label = "BMM"

                print(
                    f"  {name:<20} {avg_ms:>10.4f} ms  "
                    f"{metrics['tflops']:>8.1f} TFlops  {metrics['tbps']:>8.3f} TB/s"
                )
                all_results.append({
                    "benchmark": "dsa_projection",
                    "name": name,
                    "type": type_label,
                    "M": m,
                    "shape": shape,
                    "avg_ms": avg_ms,
                    "status": "OK",
                    **metrics,
                })
            except Exception as e:
                print(f"  {name:<20}   FAILED: {e}")
                all_results.append({
                    "benchmark": "dsa_projection",
                    "name": name,
                    "type": spec_type.upper(),
                    "M": m,
                    "shape": "",
                    "avg_ms": 0.0,
                    "status": "FAIL",
                    "error": str(e)[:200],
                    "tflops": 0.0,
                    "tbps": 0.0,
                    "fpb": 0.0,
                })

    write_results(all_results, args.output_dir, "glm5_attention_gemm_amd")


if __name__ == "__main__":
    main()
