#!/usr/bin/env python3
"""
GLM-5 MoE EP communication benchmark — AMD gfx942.

Upstream: https://github.com/lixiuhong/llm_flops bench_glm5_deepep.py
Backend: torch.distributed all_to_all (DeepEP is CUDA-only; not available on ROCm).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)

from common import HIDDEN_DIM, bootstrap

NUM_EXPERTS = 256
NUM_TOPK = 8
M_PER_GPU_LIST_DEFAULT = [512, 1024, 2048, 4096, 8192]
M_PER_GPU_LIST_QUICK = [512, 2048]
NUM_WARMUP = 10
NUM_RUNS = 30

SCENARIOS = {
    "balanced": 0.0,
    "mild": 1.0,
    "medium": 2.0,
    "heavy": 3.0,
}


def init_dist(local_rank: int) -> int:
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return dist.get_rank()


def generate_routing(num_tokens: int, num_experts: int, num_topk: int, skew_std: float, device):
    if skew_std <= 0:
        scores = torch.ones((num_tokens, num_experts), dtype=torch.float32, device=device)
        scores += torch.randn_like(scores) * 0.01
    else:
        popularity = torch.randn(num_experts, device=device) * skew_std
        popularity = popularity.exp()
        popularity = popularity / popularity.sum() * num_experts
        scores = popularity.unsqueeze(0).expand(num_tokens, -1).clone()
        scores += torch.randn((num_tokens, num_experts), device=device) * 0.5
        scores = scores.abs() + 1e-6
    topk_idx = torch.topk(scores, num_topk, dim=-1, sorted=False).indices
    return topk_idx, scores


def bench_fn(fn, num_warmups=NUM_WARMUP, num_tests=NUM_RUNS):
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    for _ in range(num_warmups):
        fn()
    cache.zero_()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for i in range(num_tests):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()
    times = np.array([s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)])[1:]
    return float(np.average(times)), float(np.min(times)), float(np.max(times))


def run_ep_bench(rank, world_size, num_tokens, hidden, topk_idx, scenario_name):
    device = torch.device("cuda", rank)
    experts_per_rank = NUM_EXPERTS // world_size
    recv_tokens = num_tokens * NUM_TOPK // world_size
    recv_tokens = max(recv_tokens, 1)

    send_tensors = [
        torch.randn(recv_tokens, hidden, dtype=torch.bfloat16, device=device)
        for _ in range(world_size)
    ]
    recv_tensors = [
        torch.empty(recv_tokens, hidden, dtype=torch.bfloat16, device=device)
        for _ in range(world_size)
    ]

    def layout_fn():
        owners = topk_idx // experts_per_rank
        _ = owners.unique(sorted=True)

    t_layout, _, _ = bench_fn(layout_fn)
    t_dispatch, _, _ = bench_fn(lambda: dist.all_to_all(recv_tensors, send_tensors))
    t_combine, _, _ = bench_fn(lambda: dist.all_to_all(send_tensors, recv_tensors))

    return {
        "benchmark": "deepep",
        "backend": "torch.distributed all_to_all",
        "scenario": scenario_name,
        "rank": rank,
        "world_size": world_size,
        "num_tokens": num_tokens,
        "recv_tokens": recv_tokens,
        "layout_ms": t_layout * 1000,
        "dispatch_ms": t_dispatch * 1000,
        "combine_ms": t_combine * 1000,
        "total_ms": (t_layout + t_dispatch + t_combine) * 1000,
        "status": "OK",
    }


def worker(local_rank: int, args):
    bootstrap()
    rank = init_dist(local_rank)
    world_size = dist.get_world_size()
    torch.manual_seed(rank)
    device = torch.device("cuda", local_rank)

    scenarios = SCENARIOS if args.scenario == "all" else {args.scenario: SCENARIOS[args.scenario]}
    m_list = M_PER_GPU_LIST_QUICK if args.quick else M_PER_GPU_LIST_DEFAULT

    if rank == 0:
        print("=" * 90)
        print("GLM-5 MoE EP Communication | AMD gfx942 | torch.distributed fallback")
        print(f"  world_size={world_size}  scenario={args.scenario}")
        print("=" * 90)

    all_results = []
    for scenario_name, skew in scenarios.items():
        for m_per_gpu in m_list:
            topk_idx, _ = generate_routing(m_per_gpu, NUM_EXPERTS, NUM_TOPK, skew, device)
            try:
                result = run_ep_bench(rank, world_size, m_per_gpu, args.hidden, topk_idx, scenario_name)
                if rank == 0:
                    print(
                        f"  {scenario_name} M={m_per_gpu}: total={result['total_ms']:.3f} ms "
                        f"(layout={result['layout_ms']:.3f}, dispatch={result['dispatch_ms']:.3f}, "
                        f"combine={result['combine_ms']:.3f})"
                    )
                all_results.append(result)
            except Exception as e:
                if rank == 0:
                    print(f"  {scenario_name} M={m_per_gpu}: FAILED {e}")
                all_results.append({
                    "benchmark": "deepep",
                    "scenario": scenario_name,
                    "rank": rank,
                    "num_tokens": m_per_gpu,
                    "status": "FAIL",
                    "error": str(e)[:200],
                })

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(args.output_dir, f"glm5_deepep_amd_{stamp}.json")
        csv_path = os.path.join(args.output_dir, f"glm5_deepep_amd_{stamp}.csv")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        fields = [
            "scenario", "rank", "world_size", "num_tokens", "recv_tokens",
            "layout_ms", "dispatch_ms", "combine_ms", "total_ms", "status",
        ]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_results)
        print(f"\nSaved:\n  {json_path}\n  {csv_path}")

    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="GLM-5 MoE EP comm (gfx942, torch.distributed)")
    parser.add_argument("--hidden", type=int, default=HIDDEN_DIM)
    parser.add_argument("--scenario", default="balanced", choices=list(SCENARIOS.keys()) + ["all"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", type=str, default=os.path.join(_REPO, "results"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("DeepEP unavailable on ROCm. Launch with:")
        print(
            "  torchrun --nproc_per_node=4 bench_glm5_deepep.py "
            f"--scenario {args.scenario} --output-dir {args.output_dir}"
        )
        return

    if "LOCAL_RANK" not in os.environ:
        print("ERROR: run via torchrun, e.g.:")
        print("  torchrun --nproc_per_node=4 bench_glm5_deepep.py --scenario balanced")
        sys.exit(1)

    worker(int(os.environ["LOCAL_RANK"]), args)


if __name__ == "__main__":
    main()
