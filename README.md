# llm_flops_amd

GLM-5 算子性能测试工具集 — **AMD MI300X (gfx942 / ROCm)** 版。

仓库：[qhy991/llm_flops_amd](https://github.com/qhy991/llm_flops_amd)  
上游参考：[lixiuhong/llm_flops](https://github.com/lixiuhong/llm_flops)（NVIDIA CUDA + DeepGEMM）

## 与 llm_flops 的对应关系

| llm_flops (CUDA) | llm_flops_amd (gfx942) | 状态 |
|------------------|-------------------------|------|
| `bench_glm5_decode.py` | `bench_glm5_decode.py` | ✅ |
| `bench_glm5_prefill.py` | `bench_glm5_prefill.py` | ✅ |
| `dsa_projection.py` | 合并在 unified bench 的 Attention GEMM | 部分 |
| `dsa_indexer.py` | 合并在 unified bench 的 Indexer 段 | 部分 |
| `dsa_flashmla.py` | PyTorch MLA fallback | 近似 |
| `moe_deepgemm.py` | 3× dense AITER GEMM | 近似 |
| `bench_glm5_deepep.py` | 未包含（多卡 EP） | ❌ |

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

# 4. 全量 prefill 扫表（默认 M=1024,2048,4096  S=2048,8192）
python bench_glm5_prefill.py

# 5. 一次跑 decode + prefill
python run_all.py
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

## 已知 AMD 实测结论（参考）

**Decode BS=1, S=2048**（layer-sum ≈ 0.20 ms）：

1. `o_proj` ~0.044 ms
2. `mla_decode_attn` ~0.035 ms
3. GEMM/MoE ~0.02 ms 量级

**Decode BS≥4**：`mla_decode_attn`（PyTorch fallback）迅速成为瓶颈（50%→85%）。

## 推送到 GitHub

```bash
cd /root/glm5-flops-amd
git init
git add .
git commit -m "Initial AMD gfx942 GLM-5 operator benchmarks"
git remote add origin https://github.com/<user>/glm5-flops-amd.git
git push -u origin main
```

## License

与上游 llm_flops 相同用途的 benchmark 代码；依赖 aiter/sglang 遵循各自许可证。
