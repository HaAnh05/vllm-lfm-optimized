# LLM Inference Serving Optimization on Resource-Constrained Hardware

This repository presents an empirical optimization study for serving the **LiquidAI LFM2.5-1.2B-Instruct** architecture under a multi-turn, Poisson-distributed workload benchmark.

---

## 🎯 1. System Architecture & Constraints

- **Model:** `LiquidAI/LFM2.5-1.2B-Instruct` (Gated Delta Networks / State-Space Model family)
- **Target Hardware:** 1× NVIDIA H200 MIG instance (18 GB VRAM, 3 CPU Cores, 8 GB RAM)
- **Runtime:** vLLM `v0.22.1` containerized serving layer
- **Workload Profile:** 70 Poisson-distributed multi-turn conversations (330 evaluated requests, 90 warmup requests; prompt length $\approx$ 4,000 tokens, output length $\le$ 200 tokens)
- **Evaluation Metric:** Effective Request Score (ERS), defined as:

$$S_{\text{request}} = 0.5 \cdot s_{\text{ttft}} + 0.5 \cdot s_{\text{tpot}}$$

---

## 🔬 2. Methodology & Key Findings

### 2.1. Memory-Bandwidth Bottleneck Analysis
Standard FP16/BF16 model weight loading ($\approx 2.4\text{ GB}$) incurs a physical memory bandwidth floor, constraining Time-Per-Output-Token (TPOT) to $6\text{ ms}$. 

### 2.2. FP8 Quantization Paradigm Shift
Applying **Online FP8 Weight Quantization** (`--quantization=fp8`) reduces memory bandwidth demand by $50\%$ ($\approx 1.2\text{ GB}$), leveraging native Hopper Tensor Cores to reduce TPOT from $6\text{ ms}$ to $4\text{ ms}$ ($\Delta\text{TPOT} = -33.3\%$), elevating overall ERS from $48.83$ to $59.62$.

### 2.3. Architectural Constraints of GDN
- **Speculative Decoding Incompatibility:** Speculative methods (`ngram_gpu`) cause state corruption during rejection rollbacks in State-Space / GDN architectures, failing context accuracy gates ($0\%$ retention).
- **CPU Core Starvation Mitigation:** With only 3 available CPU cores, logging disables (`--no-enable-log-requests`) and thread contention caps (`TOKENIZERS_PARALLELISM=false`) are strictly required.

---

## 📊 3. Empirical Experimental Trajectory

| Iteration | ERS Score | TTFT p50 (ms) | TTFT p95 (ms) | TPOT (ms) | Failed | Core Intervention & Technical Rationale |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **V1** | **48.42** | 91 | 141 | 6 | 8 | Baseline evaluation (BF16 weights, FP8 KV cache, `--max-model-len=8192`). |
| **V3** | **40.48** | — | — | 5 | — | Removed FP8 KV cache; caused severe VRAM thrashing. |
| **V4** | **37.95** | 55 | 98 | 6 | 91 | Excessive prefill batching (`16384` tokens) induced OOM cascades. |
| **V5** | **40.83** | 50 | 88 | 6 | 76 | Integrated `cutedsl` backend; optimized TTFT but suffered OOMs without FP8 cache. |
| **V6** | **48.83** | 50 | 93 | 6 | 8 | Restored FP8 KV cache + `cutedsl` + Level 3 optimization. Memory-bound TPOT floor at 6 ms. |
| **V7** | **46.68** | 66 | 97 | 6 | 7 | Disabled chunked prefill; regressed TTFT without improving TPOT. |
| **V8–V9** | **0.00** | — | — | — | — | Speculative decoding (`ngram_gpu`) triggered hidden state corruption on GDN rollbacks. |
| **V10** | **48.59** | 52 | 89 | 6 | 8 | Reverted speculative decoding; evaluated `machete` backend and throughput scheduling. |
| **V11** | **59.62** | **49** | **81** | **4** | **7** | **Major Breakthrough:** Deployed custom container with online FP8 weight quantization. |
| **V12** | **60.40** | 50 | **73** | **4** | **4** | Set `VLLM_TORCH_COMPILE_LEVEL=3`, `--gpu-memory-utilization=0.95`, `--max-num-batched-tokens=4096`. Reduced p95 latency to 73ms and failures to 4. |
| **V13 (Ultimate Combined)** | *Ready* | — | — | — | — | Combined FP8 Quantization + `--gpu-memory-utilization=0.96` + `--max-num-seqs=75` + Micro Speculation (`ngram_gpu`, `--spec-tokens=2`) + Level 3 optimization + CPU Thread bounds (`=3`). |

---

## 🛠️ 4. Reproduction & Deployment Guide

### 4.1. CI/CD Image Build Pipeline
Images are built via GitHub Actions using the `.github/workflows/build-docker.yml` workflow:

```bash
git add .
git commit -m "Deploy V13 optimization profile"
git push origin main
```

### 4.2. Production Orchestration

```yaml
services:
  model:
    image: haanh05/vllm-lfm-optimized:v3
    entrypoint: ["python3", "/app/entrypoint.py"]
    command:
      - --model=/model
      - --served-model-name=LFM2.5-1.2B-Instruct
      - --host=0.0.0.0
      - --port=8000
      - --max-model-len=32768
      - --max-num-seqs=80
      - --swap-space=1
      - --gpu-memory-utilization=0.95
      - --tensor-parallel-size=1
      - --enable-prefix-caching
      - --kv-cache-dtype=fp8
      - --quantization=fp8
      - --enable-chunked-prefill
      - --max-num-batched-tokens=4096
      - --gdn-prefill-backend=cutedsl
      - --optimization-level=3
      - --no-enable-log-requests
      - --disable-log-stats
    ports:
      - "8000:8000"
    shm_size: "2g"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```
