#!/usr/bin/env python3
"""Run all GLM-5 AMD micro-benchmarks (decode + prefill)."""
from __future__ import annotations

import argparse
import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.environ.get("VENV_PYTHON", sys.executable)

BENCHMARKS = {
    "decode": ("bench_glm5_decode.py", "GLM-5 decode unified (llm_flops style)"),
    "prefill": ("bench_glm5_prefill.py", "GLM-5 prefill unified (llm_flops style)"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=list(BENCHMARKS.keys()) + ["all"], default="all")
    parser.add_argument("--quick", action="store_true",
                        help="Smaller M/S lists for smoke test")
    args, extra = parser.parse_known_args()

    quick_args = []
    if args.quick:
        quick_args = [
            "--m-list", "1", "1024",
            "--s-list", "2048",
            "--warmup", "2",
            "--runs", "5",
        ]

    suites = list(BENCHMARKS.keys()) if args.suite == "all" else [args.suite]
    rc = 0
    for name in suites:
        script, desc = BENCHMARKS[name]
        path = os.path.join(ROOT, script)
        cmd = [PYTHON, path] + quick_args + extra
        print(f"\n{'=' * 70}\n  {desc}\n  {' '.join(cmd)}\n{'=' * 70}\n")
        r = subprocess.run(cmd, cwd=ROOT)
        rc = rc or r.returncode

    sys.exit(rc)


if __name__ == "__main__":
    main()
