#!/usr/bin/env python3
"""Compare benchmark JSON against a frozen baseline (acceptance regression check)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_rows(path: Path) -> List[dict]:
    with path.open() as f:
        return json.load(f)


def row_key(r: dict) -> Tuple:
    bench = r.get("benchmark", r.get("phase", ""))
    if bench == "dsa_projection" or r.get("phase") in ("decode", "prefill"):
        if "phase" in r:
            return (r["phase"], r["name"], r["M"], r.get("S", 0))
        return (bench, r["name"], r["M"], 0)
    if bench == "dsa_indexer":
        return (bench, r["name"], r["M"], r["S"])
    if bench == "moe_deepgemm":
        return (bench, r["proj"], r["dist_idx"], r.get("K", 0), r.get("N", 0))
    if bench == "dsa_flashmla":
        return (bench, r["hit_rate"], r.get("s_q", 0), r.get("s_kv", 0))
    if bench == "deepep":
        return (bench, r["scenario"], r.get("rank", 0), r.get("num_tokens", 0))
    return (bench, r.get("name", ""), r.get("M", 0), r.get("S", 0))


def index_rows(rows: List[dict]) -> Dict[Tuple, dict]:
    out = {}
    for r in rows:
        if r.get("status") not in (None, "OK"):
            if r.get("status") == "FAIL" or r.get("avg_ms", 1) == 0 and r.get("error"):
                continue
        if r.get("status") == "FAIL":
            continue
        if "avg_ms" in r and r["avg_ms"] <= 0 and r.get("error"):
            continue
        out[row_key(r)] = r
    return out


def compare(baseline: Dict[Tuple, dict], current: Dict[Tuple, dict], max_regression: float) -> int:
    keys = sorted(set(baseline) | set(current))
    improved = regressed = missing = new = 0

    print(f"{'key':<50} {'base_ms':>10} {'curr_ms':>10} {'chg':>8} {'status':>10}")
    print("-" * 95)

    for key in keys:
        b = baseline.get(key)
        c = current.get(key)
        key_str = str(key)[:50]
        if b and not c:
            missing += 1
            print(f"{key_str:<50} {'—':>10} {'—':>10} {'—':>8} {'MISSING':>10}")
            continue
        if c and not b:
            new += 1
            cms = c.get("avg_ms", c.get("total_ms", 0))
            print(f"{key_str:<50} {'—':>10} {cms:>10.4f} {'—':>8} {'NEW':>10}")
            continue

        bms = b.get("avg_ms", b.get("total_ms", 0))
        cms = c.get("avg_ms", c.get("total_ms", 0))
        if bms <= 0:
            chg = 0.0
        else:
            chg = (bms - cms) / bms * 100
        if chg >= 1.0:
            status = "FASTER"
            improved += 1
        elif chg <= -max_regression:
            status = "REGRESS"
            regressed += 1
        else:
            status = "OK"
        print(f"{key_str:<50} {bms:>10.4f} {cms:>10.4f} {chg:>+7.1f}% {status:>10}")

    print("-" * 95)
    print(
        f"Summary: improved={improved} regressed={regressed} "
        f"within_tol={len(keys)-improved-regressed-missing-new} missing={missing} new={new}"
    )
    return 1 if regressed else 0


def load_baseline_dir(base_path: Path) -> List[dict]:
    rows: List[dict] = []
    if base_path.is_dir():
        for name in sorted(base_path.iterdir()):
            if name.suffix == ".json" and name.name != "manifest.json":
                rows.extend(load_rows(name))
        return rows
    return load_rows(base_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare benchmark JSON to baseline")
    parser.add_argument(
        "--baseline",
        default="results/baseline/v2026.07.01",
        help="Baseline JSON file or directory",
    )
    parser.add_argument("--current", required=True, help="Current run JSON or directory")
    parser.add_argument("--max-regression", type=float, default=5.0)
    args = parser.parse_args()

    base_rows = load_baseline_dir(Path(args.baseline))
    curr_path = Path(args.current)
    curr_rows = load_baseline_dir(curr_path) if curr_path.is_dir() else load_rows(curr_path)

    rc = compare(index_rows(base_rows), index_rows(curr_rows), args.max_regression)
    sys.exit(rc)


if __name__ == "__main__":
    main()
