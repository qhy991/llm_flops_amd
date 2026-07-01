# Baseline v2026.07.01 — GLM-5 FP8 on MI300X (gfx942)

> 录制时间：2026-07-01  
> 用途：算子级 micro-benchmark 验收基线（L1）。优化后跑同样扫参，用 `compare_results.py` 对比。

## 环境快照

| 项 | 值 |
|----|-----|
| GPU | AMD MI300X × 4（单卡 bench） |
| Arch | gfx942 |
| ROCm | 7.0 |
| PyTorch | 2.10.0+rocm7.0 |
| aiter | `6d0304e8c` |
| sglang | `85fd900` |
| llm_flops_amd | `74caad2` |

## 扫参配置

| 阶段 | M | S | warmup | runs |
|------|---|---|--------|------|
| decode | 1, 4, 8, 16, 32 | 2048, 8192 | 5 | 20 |
| prefill | 1024, 2048, 4096 | 2048, 8192 | 5 | 20 |

**统计**：decode 120/120 OK，prefill 72/72 OK；`index_score` 未测。

## Decode — layer-sum (ms)

| M | S=2048 | S=8192 | Top-1 瓶颈 | Top-1 % |
|---|--------|--------|------------|---------|
| 1 | 0.199 | 0.194 | o_proj | ~22% |
| 4 | 0.362 | 0.329 | mla_decode_attn | 48–53% |
| 8 | 0.561 | 0.486 | mla_decode_attn | 63–68% |
| 16 | 0.937 | 0.808 | mla_decode_attn | 76–79% |
| 32 | 1.757 | 1.496 | mla_decode_attn | 82–85% |

### Decode BS=1, S=2048 — Top-5

| 算子 | avg_ms | 占比 |
|------|--------|------|
| o_proj | 0.0436 | 21.9% |
| mla_decode_attn | 0.0347 | 17.4% |
| fused_qkv_a_proj | 0.0203 | 10.2% |
| moe_up_proj | 0.0187 | 9.4% |
| moe_gate_proj | 0.0186 | 9.3% |

### Decode BS=32, S=2048 — Top-5

| 算子 | avg_ms | 占比 |
|------|--------|------|
| mla_decode_attn | 1.4873 | 84.7% |
| moe_up_proj | 0.0551 | 3.1% |
| moe_gate_proj | 0.0547 | 3.1% |
| o_proj | 0.0492 | 2.8% |
| moe_down_proj | 0.0250 | 1.4% |

## Prefill — layer-sum (ms)

| M | S=2048 | S=8192 | Top-1 瓶颈 | Top-1 % |
|---|--------|--------|------------|---------|
| 1024 | 2.400 | 2.537 | mla_prefill_attn | ~28% |
| 2048 | 5.119 | 5.163 | mla_prefill_attn | ~42% |
| 4096 | 13.461 | 13.754 | mla_prefill_attn | ~59% |

### Prefill M=4096, S=2048 — Top-5

| 算子 | avg_ms | 占比 |
|------|--------|------|
| mla_prefill_attn | 7.9993 | 59.4% |
| moe_gate_proj | 1.1394 | 8.5% |
| moe_up_proj | 1.1361 | 8.4% |
| moe_down_proj | 1.0808 | 8.0% |
| o_proj | 0.9451 | 7.0% |

## 验收用法

```bash
# 重跑全量
python run_all.py

# 与基线对比（默认容忍单算子慢 5%）
python compare_results.py \
  --baseline results/baseline/v2026.07.01 \
  --current results/glm5_decode_amd_YYYYMMDD_HHMMSS.json

# 更严格：单算子慢 2% 即 FAIL
python compare_results.py --max-regression 2.0 --baseline ... --current ...
```

完整原始数据见 `v2026.07.01/decode.json`、`prefill.json` 及对应 CSV。
