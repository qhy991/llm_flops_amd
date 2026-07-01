#!/usr/bin/env python3
"""Compare benchmark JSON against a frozen baseline (acceptance regression check)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def load_rows(path: Path) -> List[dict]:
    with path.open() as f:
        return json.load(f)


def index_rows(rows: List[dict]) -> Dict[Tuple, dict]:
    out = {}
    for r in rows:
        if r.get("status") != "OK":
            continue
        key = (r.get("phase", ""), r["name"], r["M"], r["S"])
        out[key] = r
    return out


def compare(baseline: Dict[Tuple, dict], current: Dict[Tuple, dict], max_regression: float) -> int:
    keys = sorted(set(baseline) | set(current))
    improved = regressed = missing = new = 0

    print(f"{'phase':<8} {'name':<22} {'M':>4} {'S':>6} {'base_ms':>10} {'curr_ms':>10} {'chg':>8} {'status':>10}")
    print("-" * 90)

    for key in keys:
        phase, name, m, s = key
        b = baseline.get(key)
        c = current.get(key)
        if b and not c:
            missing += 1
            print(f"{phase:<8} {name:<22} {m:>4} {s:>6} {'—':>10} {'—':>10} {'—':>8} {'MISSING':>10}")
            continue
        if c and not b:
            new += 1
            print(f"{phase:<8} {name:<22} {m:>4} {s:>6} {'—':>10} {c['avg_ms']:>10.4f} {'—':>8} {'NEW':>10}")
            continue

        bms = b["avg_ms"]
        cms = c["avg_ms"]
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
        print(
            f"{phase:<8} {name:<22} {m:>4} {s:>6} {bms:>10.4f} {cms:>10.4f} {chg:>+7.1f}% {status:>10}"
        )

    print("-" * 90)
    print(
        f"Summary: improved={improved} regressed={regressed} "
        f"within_tol={len(keys)-improved-regressed-missing-new} missing={missing} new={new}"
    )
    return 1 if regressed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare benchmark JSON to baseline")
    parser.add_argument(
        "--baseline",
        default="results/baseline/v2026.07.01/decode.json",
        help="Baseline JSON (or directory with decode.json + prefill.json)",
    )
    parser.add_argument("--current", required=True, help="Current run JSON")
    parser.add_argument(
        "--max-regression",
        type=float,
        default=5.0,
        help="Max allowed slowdown %% per op before FAIL (default 5%%)",
    )
    args = parser.parse_args()

    base_path = Path(args.baseline)
    curr_path = Path(args.current)

    base_rows: List[dict] = []
    if base_path.is_dir():
        for name in ("decode.json", "prefill.json"):
            p = base_path / name
            if p.exists():
                base_rows.extend(load_rows(p))
    else:
        base_rows = load_rows(base_path)
        parent = base_path.parent / "prefill.json"
        if parent.exists() and base_path.name == "decode.json":
            base_rows.extend(load_rows(parent))

    curr_rows = load_rows(curr_path)
    if curr_path.parent.joinpath("prefill.json").exists() and "prefill" not in curr_path.name:
        pass
    # If current is only decode, user passes one file; if dir, merge
    if curr_path.is_dir():
        curr_rows = []
        for name in ("decode.json", "prefill.json"):
            p = curr_path / name
            if p.exists():
                curr_rows.extend(load_rows(p))

    rc = compare(index_rows(base_rows), index_rows(curr_rows), args.max_regression)
    sys.exit(rc)


if __name__ == "__main__":
    main()
