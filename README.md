# Optimized vLLM Serving for LiquidAI LFM2.5-1.2B-Instruct

Dự án tối ưu hóa hiệu năng phục vụ mô hình ngôn ngữ lớn (**LLM Inference Serving**) trên hạ tầng GPU NVIDIA H200 (MiG 18GB VRAM, 3 CPU Cores, 8GB RAM) thuộc khuôn khổ cuộc thi tối ưu LLM Serving.

---

## 🚀 1. Hướng dẫn Setup & Chạy (Quickstart)

### 1.1. Yêu cầu Hệ thống
- Hệ điều hành: Ubuntu 24.04 LTS (hoặc môi trường Docker trên Linux/Windows)
- Phần cứng: NVIDIA H200 (MiG 18GB VRAM)
- Driver & CUDA: NVIDIA Driver 590.x / CUDA 13.x
- Công cụ: Docker & Docker Compose, Git

### 1.2. Quy trình Build & Deploy qua CI/CD (GitHub Actions)

Dự án được cấu hình tự động build image trên GitHub Actions và đẩy lên Docker Hub để tránh tốn tài nguyên máy local.

1. **Clone Repository:**
   ```bash
   git clone https://github.com/HaAnh05/vllm-lfm-optimized.git
   cd vllm-lfm-optimized
   ```

2. **Cấu hình Docker Hub Secrets trên GitHub:**
   Vào `Settings -> Secrets and variables -> Actions` trên GitHub Repo và thêm 2 Secrets:
   - `DOCKERHUB_USERNAME`: `haanh05`
   - `DOCKERHUB_TOKEN`: Personal Access Token tạo từ Docker Hub.

3. **Kích hoạt Build Image:**
   Mỗi khi `git push` lên branch `main`, GitHub Actions sẽ tự động build image và push lên Docker Hub tại:
   `haanh05/vllm-lfm-optimized:v2`

4. **Chạy Server bằng Docker Compose:**
   ```bash
   docker compose up -d
   ```

---

## ⚙️ 2. Mô tả Cấu hình & Tối ưu Hiện tại (Version V12)

### 2.1. File `docker-compose.yml`

```yaml
services:
  model:
    image: haanh05/vllm-lfm-optimized:v2
    entrypoint:
      - python3
      - /app/entrypoint.py
    command:
      - --model=/model
      - --served-model-name=LFM2.5-1.2B-Instruct
      - --host=0.0.0.0
      - --port=8000
      - --max-model-len=8192
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

### 2.2. Các Kỹ thuật Tối ưu Cốt lõi
- **FP8 Online Weight Quantization (`--quantization=fp8`):** Nén trọng số mô hình từ BF16 (2.4GB) xuống FP8 (1.2GB), giảm 50% băng thông bộ nhớ và tận dụng lõi Tensor Core FP8 của H200, kéo TPOT từ 6ms xuống 4ms (và đang tối ưu xuống 3ms).
- **FP8 KV Cache (`--kv-cache-dtype=fp8`):** Giảm dung lượng lưu trữ KV cache, giúp phục vụ được nhiều request cùng lúc mà không bị rớt do cạn VRAM.
- **CuteDSL Prefill Backend (`--gdn-prefill-backend=cutedsl`):** Backend được tối ưu riêng cho kiến trúc Gated Delta Networks (GDN), giúp đẩy TTFT xuống sát mốc 49ms.
- **Custom Entrypoint Script (`entrypoint.py`):** Thiết lập `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` chặn fragmentation VRAM, tắt tokenizer parallelism tránh nghẽn 3 CPU cores, và đẩy `VLLM_TORCH_COMPILE_LEVEL=3` để biên dịch tối đa.
- **Cân bằng Prefill & Decode (`--max-num-batched-tokens=4096`):** Giới hạn kích thước prefill chunk để nhường tài nguyên GPU ưu tiên cho luồng decode sinh token nhanh hơn.

---

## 📜 3. Lịch sử Các Phiên bản Tối ưu (Iteration History)

| Phiên bản | Điểm ERS | TTFT p50 | TTFT p95 | TPOT (tbt) | Failed | Thay đổi chính & Phân tích nguyên nhân |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **V1** | **48.42** | 91ms | 141ms | 6ms | 8 | **Baseline:** Cấu hình chuẩn từ BTC với FP8 KV Cache, chunked prefill, `--max-model-len=8192`. |
| **V3** | **40.48** | — | — | 5ms | — | **Thử nghiệm sai lầm:** Bỏ `fp8` KV cache vì nhầm mô hình GDN không phải Transformer. VRAM bị tràn làm rớt điểm nghiêm trọng. |
| **V4** | **37.95** | 55ms | 98ms | 6ms | 91 | Ép `--max-num-batched-tokens=16384` quá đà gây ra Out-Of-Memory (OOM) làm 91 request bị đánh 0 điểm. |
| **V5** | **40.83** | 50ms | 88ms | 6ms | 76 | Thử nghiệm backend `cutedsl` giúp TTFT giảm kỷ lục (50ms), nhưng vẫn bị OOM 76 requests do chưa nạp lại FP8 cache. |
| **V6** | **48.83** | 50ms | 93ms | 6ms | 8 | Khôi phục FP8 KV cache + `cutedsl` + `--optimization-level=3`. Đạt kỷ lục cũ nhưng TPOT vẫn kẹt cứng ở 6ms. |
| **V7** | **46.68** | 66ms | 97ms | 6ms | 7 | Tắt Chunked Prefill với hy vọng bật CUDA Graphs để hạ TPOT, nhưng TPOT không đổi và TTFT bị chậm đi. |
| **V8 / V9** | **0.00** | — | — | — | — | **Thử nghiệm Speculative Decoding (`ngram_gpu`):** Bị dính lỗi Accuracy Gate `long-context probe failed (0%) — truncation` do GDN không hỗ trợ rollback state khi đoán sai token. |
| **V10** | **48.59** | 52ms | 89ms | 6ms | 8 | Bỏ Speculative Decoding, thử backend `machete` và `throughput` mode. Độ ổn định cao nhưng TPOT vẫn chạm sàn 6ms của BF16. |
| **V11 (Docker Image FP8)** | **59.62** | **49ms** | **81ms** | **4ms** | **7** | **BƯỚC TIẾN ĐỘT PHÁ:** Tự build Custom Docker Image với `--quantization=fp8` online weight quantization. Nén weights giúp giảm 50% băng thông bộ nhớ, **bẻ gãy sàn 6ms TPOT xuống 4ms**, điểm tăng vọt +10.8. |
| **V12 (Hiện tại)** | *Đang chấm* | — | — | — | — | Cập nhật `VLLM_TORCH_COMPILE_LEVEL=3`, nâng `--gpu-memory-utilization=0.95` (xóa 7 lỗi failed), và giảm `--max-num-batched-tokens=4096` để đẩy TPOT xuống mốc ~3ms. |

---

## 📈 4. Bài học Kinh nghiệm (Key Takeaways)

1. **Hạn chế phần cứng:** Mô hình bị giới hạn bởi băng thông bộ nhớ (Memory Bandwidth Bound). Đọc 2.4GB trọng số BF16 mất tối thiểu 6ms/token trên H200 MIG.
2. **Giải pháp Quantization:** Muốn hạ TPOT < 6ms, bắt buộc phải dùng **Weight Quantization (FP8/INT4)** để nén dung lượng trọng số cần đọc mỗi nhịp.
3. **Đặc thù GDN (State-Space Model):** GDN không tương thích với Speculative Decoding (`ngram_gpu`) do không rollback được hidden state.
4. **Quản lý CPU overhead:** Hệ thống chỉ có 3 CPU Cores nên việc tắt log (`--no-enable-log-requests`, `--disable-log-stats`) và khóa multi-thread tokenizer là cực kỳ quan trọng để tránh nghẽn CPU.
