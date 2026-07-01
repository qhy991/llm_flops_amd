#!/usr/bin/env python3
"""
GLM-5 DECODE phase unified operator benchmark — AMD gfx942 / MI300X.

Upstream reference: https://github.com/lixiuhong/llm_flops bench_glm5_decode.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from typing import Callable, List, Tuple

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)

from common import (
    DECODE_M_LIST_DEFAULT,
    FUSED_QKV_A_OUT,
    HIDDEN_DIM,
    INDEX_HEAD_DIM,
    INDEX_N_HEADS,
    KV_LORA_RANK,
    KV_S_LIST_DEFAULT,
    NUM_HEADS,
    NUM_RUNS_DEFAULT,
    NUM_WARMUP_DEFAULT,
    QK_HEAD_DIM,
    QK_NOPE_HEAD_DIM,
    Q_LORA_RANK,
    V_HEAD_DIM,
    bootstrap,
)
import common as common_mod
from backends import (
    bench_bf16_gemm,
    bench_bmm,
    bench_fp8_gemm,
    bench_mla_decode_pytorch,
    bench_moe_dense_triple,
    bmm_backend_name,
    gemm_backend_name,
    init_backends,
)

MOE_TOP_K = 8


def get_operators(m: int, s: int) -> List[Tuple[str, str, Callable, str, str]]:
    gemm = gemm_backend_name()
    bmm = bmm_backend_name()
    tokens = m * MOE_TOP_K
    ops: List[Tuple[str, str, Callable, str, str]] = [
        ("fused_qkv_a_proj", "Attention", lambda d: bench_fp8_gemm(m, HIDDEN_DIM, FUSED_QKV_A_OUT, d),
         f"[{m},{HIDDEN_DIM}]×[{FUSED_QKV_A_OUT}]", gemm),
        ("q_b_proj", "Attention", lambda d: bench_fp8_gemm(m, Q_LORA_RANK, NUM_HEADS * QK_HEAD_DIM, d),
         f"[{m},{Q_LORA_RANK}]×[{NUM_HEADS * QK_HEAD_DIM}]", gemm),
        ("absorbed_W_UK", "Attention", lambda d: bench_bmm(NUM_HEADS, m, QK_NOPE_HEAD_DIM, KV_LORA_RANK, d),
         f"bmm [{NUM_HEADS},{m},{QK_NOPE_HEAD_DIM}]×[{KV_LORA_RANK}]", bmm),
        ("absorbed_W_UV", "Attention", lambda d: bench_bmm(NUM_HEADS, m, KV_LORA_RANK, V_HEAD_DIM, d),
         f"bmm [{NUM_HEADS},{m},{KV_LORA_RANK}]×[{V_HEAD_DIM}]", bmm),
        ("o_proj", "Attention", lambda d: bench_fp8_gemm(m, NUM_HEADS * V_HEAD_DIM, HIDDEN_DIM, d),
         f"[{m},{NUM_HEADS * V_HEAD_DIM}]×[{HIDDEN_DIM}]", gemm),
        ("mla_decode_attn", "MLA", lambda d, _m=m, _s=s: bench_mla_decode_pytorch(_m, _s, d),
         f"pytorch_mla batch={m} s_kv={s}", "pytorch_mla_graph_safe"),
        ("index_k_proj", "DSA Indexer", lambda d: bench_fp8_gemm(m, HIDDEN_DIM, INDEX_HEAD_DIM, d),
         f"[{m},{HIDDEN_DIM}]×[{INDEX_HEAD_DIM}]", gemm),
        ("index_q_upproj", "DSA Indexer", lambda d: bench_fp8_gemm(m, Q_LORA_RANK, INDEX_N_HEADS * INDEX_HEAD_DIM, d),
         f"[{m},{Q_LORA_RANK}]×[{INDEX_N_HEADS * INDEX_HEAD_DIM}]", gemm),
        ("index_weights_proj", "DSA Indexer", lambda d: bench_bf16_gemm(m, HIDDEN_DIM, INDEX_N_HEADS, d),
         f"[{m},{HIDDEN_DIM}]×[{INDEX_N_HEADS}]", "torch.mm"),
        ("moe_gate_proj", "MoE", lambda d, _t=tokens: bench_fp8_gemm(_t, HIDDEN_DIM, 2048, d),
         f"[{tokens},{HIDDEN_DIM}]×[2048]", gemm),
        ("moe_up_proj", "MoE", lambda d, _t=tokens: bench_fp8_gemm(_t, HIDDEN_DIM, 2048, d),
         f"[{tokens},{HIDDEN_DIM}]×[2048]", gemm),
        ("moe_down_proj", "MoE", lambda d, _t=tokens: bench_fp8_gemm(_t, 2048, HIDDEN_DIM, d),
         f"[{tokens},2048]×[{HIDDEN_DIM}]", gemm),
    ]
    return ops


def main() -> None:
    bootstrap()
    init_backends()

    parser = argparse.ArgumentParser(description="GLM-5 decode benchmark (gfx942)")
    parser.add_argument("--m-list", type=int, nargs="+", default=DECODE_M_LIST_DEFAULT)
    parser.add_argument("--s-list", type=int, nargs="+", default=KV_S_LIST_DEFAULT)
    parser.add_argument("--warmup", type=int, default=NUM_WARMUP_DEFAULT)
    parser.add_argument("--runs", type=int, default=NUM_RUNS_DEFAULT)
    parser.add_argument("--output-dir", type=str, default=os.path.join(_REPO, "results"))
    args = parser.parse_args()

    common_mod.NUM_WARMUP_DEFAULT = args.warmup
    common_mod.NUM_RUNS_DEFAULT = args.runs

    import torch
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print("=" * 120)
    print("GLM-5 DECODE Benchmark | AMD gfx942 | glm5-flops-amd")
    print(f"  GEMM={gemm_backend_name()}  BMM={bmm_backend_name()}  MLA=pytorch  index_score=SKIP")
    print(f"  M={args.m_list}  S={args.s_list}  warmup={args.warmup} runs={args.runs}")
    print("=" * 120)

    all_results = []
    ts = datetime.now().isoformat()

    for m in args.m_list:
        for s in args.s_list:
            print(f"\n--- batch={m}, S={s} ---")
            for name, cat, fn, shape, backend in get_operators(m, s):
                try:
                    ms = fn(device)
                    print(f"  {name:<22} {ms:>10.4f} ms  [{backend}]")
                    all_results.append({
                        "phase": "decode", "timestamp": ts, "name": name, "category": cat,
                        "backend": backend, "M": m, "S": s, "shape": shape,
                        "avg_ms": ms, "status": "OK",
                    })
                except Exception as e:
                    print(f"  {name:<22}   FAILED: {e}")
                    all_results.append({
                        "phase": "decode", "timestamp": ts, "name": name, "category": cat,
                        "backend": backend, "M": m, "S": s, "shape": shape,
                        "avg_ms": 0.0, "status": "FAIL", "error": str(e)[:200],
                    })
                time.sleep(0.03)

    _write_results(all_results, args.output_dir, "glm5_decode_amd")
    _print_summaries(all_results, args.m_list, args.s_list)


def _print_summaries(results, m_list, s_list) -> None:
    for m in m_list:
        for s in s_list:
            subset = [r for r in results if r["M"] == m and r["S"] == s and r["status"] == "OK"]
            subset.sort(key=lambda r: r["avg_ms"], reverse=True)
            total = sum(r["avg_ms"] for r in subset)
            print(f"\nSummary batch={m} S={s} layer-sum={total:.3f} ms")
            for i, r in enumerate(subset[:5], 1):
                pct = 100 * r["avg_ms"] / total if total else 0
                print(f"  {i}. {r['name']:<22} {r['avg_ms']:.4f} ms ({pct:.1f}%)")


def _write_results(results, output_dir: str, prefix: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"{prefix}_{stamp}.csv")
    json_path = os.path.join(output_dir, f"{prefix}_{stamp}.json")
    fields = ["phase", "timestamp", "name", "category", "backend", "M", "S", "shape", "avg_ms", "status", "error"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved:\n  {csv_path}\n  {json_path}")


if __name__ == "__main__":
    main()
