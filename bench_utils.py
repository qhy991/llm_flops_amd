"""Shared helpers for specialized llm_flops_amd benchmarks."""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


def gemm_flops(m: int, k: int, n: int) -> float:
    return 2.0 * m * n * k


def gemm_mem_bytes(m: int, k: int, n: int) -> float:
    return m * k + n * k + m * n * 2


def compute_gemm_metrics(m: int, k: int, n: int, avg_ms: float) -> Dict[str, float]:
    flops = gemm_flops(m, k, n)
    mem = gemm_mem_bytes(m, k, n)
    ms = max(avg_ms, 1e-9)
    return {
        "tflops": flops / (ms * 1e-3) / 1e12,
        "tbps": mem / (ms * 1e-3) / 1e12,
        "fpb": flops / mem if mem > 0 else 0.0,
    }


def write_results(
    results: List[Dict[str, Any]],
    output_dir: str,
    prefix: str,
    fields: Optional[List[str]] = None,
) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if fields is None:
        keys: List[str] = []
        for row in results:
            for k in row:
                if k not in keys:
                    keys.append(k)
        fields = keys

    csv_path = os.path.join(output_dir, f"{prefix}_{stamp}.csv")
    json_path = os.path.join(output_dir, f"{prefix}_{stamp}.json")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved:\n  {csv_path}\n  {json_path}")
    return csv_path, json_path
