#!/usr/bin/env python3
"""Run all GLM-5 AMD micro-benchmarks."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.environ.get("VENV_PYTHON", sys.executable)

BENCHMARKS = {
    "decode": ("bench_glm5_decode.py", "GLM-5 decode unified"),
    "prefill": ("bench_glm5_prefill.py", "GLM-5 prefill unified"),
    "projection": ("dsa_projection.py", "Attention GEMM/BMM (dsa_projection)"),
    "indexer": ("dsa_indexer.py", "DSA Indexer GEMM (dsa_indexer)"),
    "moe": ("moe_deepgemm.py", "MoE grouped GEMM (moe_deepgemm)"),
    "flashmla": ("dsa_flashmla.py", "Sparse MLA prefill (dsa_flashmla)"),
}

DEEPEP_CMD = [
    "torchrun", "--nproc_per_node=4",
    os.path.join(ROOT, "bench_glm5_deepep.py"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=list(BENCHMARKS.keys()) + ["deepep", "specialized", "all"],
        default="all",
    )
    parser.add_argument("--quick", action="store_true", help="Smaller sweeps for smoke test")
    parser.add_argument("--output-dir", type=str, default=os.path.join(ROOT, "results"))
    args, extra = parser.parse_known_args()

    quick_args = ["--quick"] if args.quick else []
    out_args = ["--output-dir", args.output_dir]

    if args.suite == "specialized":
        suites = ["projection", "indexer", "moe", "flashmla"]
    elif args.suite == "all":
        suites = list(BENCHMARKS.keys()) + ["deepep"]
    elif args.suite == "deepep":
        suites = ["deepep"]
    else:
        suites = [args.suite]

    rc = 0
    for name in suites:
        if name == "deepep":
            cmd = DEEPEP_CMD + quick_args + out_args + ["--scenario", "balanced"] + extra
            print(f"\n{'=' * 70}\n  MoE EP comm (torch.distributed)\n  {' '.join(cmd)}\n{'=' * 70}\n")
            r = subprocess.run(cmd, cwd=ROOT)
            rc = rc or r.returncode
            continue

        script, desc = BENCHMARKS[name]
        path = os.path.join(ROOT, script)
        cmd = [PYTHON, path] + quick_args + out_args + extra
        print(f"\n{'=' * 70}\n  {desc}\n  {' '.join(cmd)}\n{'=' * 70}\n")
        r = subprocess.run(cmd, cwd=ROOT)
        rc = rc or r.returncode

    sys.exit(rc)


if __name__ == "__main__":
    main()
