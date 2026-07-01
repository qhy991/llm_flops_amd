# llm_flops_amd

GLM-5 算子性能测试工具集 — **AMD MI300X (gfx942 / ROCm)** 版。

仓库：[qhy991/llm_flops_amd](https://github.com/qhy991/llm_flops_amd)  
上游参考：[lixiuhong/llm_flops](https://github.com/lixiuhong/llm_flops)（NVIDIA CUDA + DeepGEMM）

## 与 llm_flops 的对应关系

| llm_flops (CUDA) | llm_flops_amd (gfx942) | 状态 |
|------------------|-------------------------|------|
| `bench_glm5_decode.py` | `bench_glm5_decode.py` | ✅ |
| `bench_glm5_prefill.py` | `bench_glm5_prefill.py` | ✅ |
| `bench_glm5_deepep.py` | `bench_glm5_deepep.py` | ✅ torch.distributed |
| `dsa_flashmla.py` | `dsa_flashmla.py` | ✅ PyTorch sparse |
| `dsa_indexer.py` | `dsa_indexer.py` | ✅ 含 index_score |
| `dsa_projection.py` | `dsa_projection.py` | ✅ |
| `moe_deepgemm.py` | `moe_deepgemm.py` | ✅ AITER grouped |

### gfx942 后端映射

| 算子 | NVIDIA (llm_flops) | AMD (本仓库) |
|------|-------------------|--------------|
| FP8 GEMM | DeepGEMM | SGLang `aiter_w8a8_block_fp8_linear` |
| Absorb BMM | `sgl_kernel.bmm_fp8` | `torch.bmm`（HIP 无 bmm_fp8） |
| MLA decode | `flash_mla_with_kvcache` | PyTorch graph-safe gather |
| MLA prefill | FlashMLA paged | PyTorch causal attention |
| MoE | `fp8_m_grouped_gemm_nt_masked` | 3 路 dense FP8 GEMM |
| index_score | `fp8_paged_mqa_logits` | **暂未实现** |

> **注意**：这是**单卡、单层** micro-benchmark，不含 TP/权重加载/整模型 E2E。layer-sum 不可直接与 `bench_one_batch` 的 ms/tok 对比。

## 依赖

- **硬件**：AMD MI300X（gfx942）或兼容 ROCm GPU
- **Python**：ROCm PyTorch（如 `rocm-torch` venv）
- **仓库**（需单独 clone）：
  - [aiter](https://github.com/ROCm/aiter)
  - [sglang](https://github.com/sgl-project/sglang)

## 快速开始

```bash
# 1. 配置环境
cp setup_env.sh.example setup_env.sh
# 编辑 AITER_PATH / SGLANG_PATH / source 你的 venv
source setup_env.sh

# 2. 冒烟测试
python run_all.py --quick

# 3. 全量 decode 扫表（默认 M=1,4,8,16,32  S=2048,8192）
python bench_glm5_decode.py

# 3. 全量 prefill 扫表（默认 M=1024,2048,4096  S=2048,8192）
python bench_glm5_prefill.py

# 4. 专项算子 benchmark（对应上游 llm_flops 脚本 3-7）
python dsa_projection.py          # Attention GEMM/BMM 大 M 扫参
python dsa_indexer.py             # DSA Indexer 含 index_score
python moe_deepgemm.py            # MoE grouped GEMM + 随机分布
python dsa_flashmla.py            # Sparse MLA vs KV 命中率

# 5. MoE EP 通信（需多卡 torchrun；DeepEP 不可用，用 all_to_all 替代）
torchrun --nproc_per_node=4 bench_glm5_deepep.py --scenario balanced

# 6. 一次跑全部
python run_all.py
python run_all.py --suite specialized   # 仅专项 4 项
python run_all.py --suite deepep        # 仅 EP 通信
```

### 环境变量

| 变量 | 说明 | 默认 |
|------|------|------|
| `AITER_PATH` | aiter 源码根目录 | `/root/repos/aiter` |
| `SGLANG_PATH` | sglang `python/` 目录 | `/root/repos/sglang/python` |
| `SGLANG_USE_AITER` | 启用 AITER FP8 | `1` |
| `PYTORCH_ROCM_ARCH` | GPU arch | `gfx942` |
| `VENV_PYTHON` | Python 解释器 | 当前 `python` |

## 输出

结果写入 `results/`：

- `glm5_decode_amd_YYYYMMDD_HHMMSS.csv` / `.json`
- `glm5_prefill_amd_YYYYMMDD_HHMMSS.csv` / `.json`

终端会打印按 `avg_ms` 降序的 **瓶颈 Top-5** 与 **layer-sum**。

## 验收基线 (Acceptance Baseline)

| 版本 | 内容 | 路径 |
|------|------|------|
| **v2026.07.01** | decode + prefill unified layer-sum | `results/baseline/v2026.07.01/` |
| **v2026.07.02** | 专项 5 脚本（projection/indexer/moe/flashmla/deepep） | `results/baseline/v2026.07.02/` |

### L1 对比

```bash
# unified decode/prefill
python compare_results.py \
  --baseline results/baseline/v2026.07.01 \
  --current results/glm5_decode_amd_YYYYMMDD.json

# 专项算子（目录内所有 json）
python compare_results.py \
  --baseline results/baseline/v2026.07.02 \
  --current results/glm5_dsa_indexer_amd_YYYYMMDD.json
```

- 人类可读摘要：[`results/baseline/BASELINE.md`](results/baseline/BASELINE.md)
- v2026.07.02 统计：**projection 24/24 · indexer 48/48 · moe 15/15 · flashmla 10/10 · deepep 5/5** 全部 OK

### v2026.07.01 要点（unified）

- **Decode BS=1**：layer-sum ≈ **0.20 ms**；`o_proj` + `mla_decode_attn` 居前
- **Decode BS≥4**：`mla_decode_attn` 占 50%→85%
- **Prefill M=4096**：layer-sum ≈ **13.5 ms**；`mla_prefill_attn` ~59%

## 推送到 GitHub

```bash
cd llm_flops_amd
git remote add origin https://github.com/qhy991/llm_flops_amd.git
git push -u origin main
```

## License

与上游 llm_flops 相同用途的 benchmark 代码；依赖 aiter/sglang 遵循各自许可证。
