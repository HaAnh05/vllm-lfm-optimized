# Đề bài: Tối ưu LLM Inference Serving trên Hạ tầng GPU

## 1. Tổng quan & Hạ tầng hệ thống

Cuộc thi mô phỏng trực tiếp bài toán thực tế của các đội ngũ hạ tầng AI doanh nghiệp: làm thế nào để tối ưu hóa phục vụ mô hình ngôn ngữ lớn (LLM serving) đạt hiệu năng cao nhất (thông lượng cao, độ trễ thấp) trên tài nguyên phần cứng hữu hạn nhưng vẫn đảm bảo độ chính xác của câu trả lời.

### 1.1. Mô hình chỉ định
- Mô hình: LiquidAI/LFM2.5-1.2B-Instruct
- Trọng số tải từ: Hugging Face
- Link tải trọng số: https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct

### 1.2. Hạ tầng đánh giá (môi trường chuẩn hóa)
Hệ thống tự động cấp phát và chạy benchmark trực tiếp vào endpoint của thí sinh trên:
- Phần cứng: 1 instance MiG NVIDIA H200 (18GB VRAM, 3 Core CPU, 8GB RAM)
- Hệ điều hành: Ubuntu 24.04 LTS
- Driver: NVIDIA driver 590.x (hỗ trợ CUDA 13.x)

---

## 2. Nhiệm vụ & Đặc điểm Workload Trace

Thí sinh cần triển khai và tối ưu một LLM inference server để xử lý một tập workload trace multi-turn mô phỏng traffic production thực tế.

### 2.1. Quy mô tải
- Gồm 70 hội thoại đến theo phân phối Poisson
- Tổng cộng 330 request được tính điểm
- Trước đó có 15 hội thoại primer (khởi động) để làm nóng hệ thống và không tính điểm

### 2.2. Giới hạn độ dài
- Context input tối đa khoảng 4k token (khoảng 12k ký tự)
- Output tối đa 200 token

### 2.3. Tính chất Deterministic
- Phân phối Poisson được “đóng băng” thành timestamp cố định trong file trace
- Mọi đội thi đều chạy trên cùng một timeline, đảm bảo tính nhân quả
- Lượt kế tiếp của hội thoại multi-turn chỉ được gửi sau khi lượt trước hoàn tất kèm khoảng “think” mô phỏng người dùng

### 2.4. Bản trace công khai vs bản chấm
- Bản trace công khai đã lược bỏ nội dung prompt, chỉ còn thời điểm đến và số token in/out để provision tải
- Prompt thật chỉ được gửi tới endpoint tại thời điểm hệ thống của BTC chấm điểm để chống học tủ (pre-bake)

---

## 3. Cấu trúc tính điểm tổng hợp

Điểm số cuối cùng của đội thi được chốt sau khi kết hợp hiệu năng phục vụ (vòng online) và hình phạt sụt giảm chất lượng (hậu kiểm sau vòng online):

$$\text{Score} = 100 \times \text{ERS} \times f(\Delta)$$

### 3.1. Điểm hiệu năng phục vụ: ERS (Effective Request Score)

Chấm tự động trong vòng online dựa trên hai chỉ số chính:
- TTFT (Time-To-First-Token)
- TPOT (Time-Per-Output-Token)

ERS là trung bình cộng điểm của tất cả $N$ requests ($N = 330$):

$$\text{ERS} = \frac{1}{N} \sum_{i=1}^{N} S_{\text{request},i} \quad \in [0,1]$$

Điểm của từng request được tính như sau:
- Nếu lỗi, timeout hoặc trả về 0 token: $S_{\text{request}} = 0$
- Nếu xử lý thành công: $S_{\text{request}} = w \cdot s_{\text{ttft}} + (1 - w) \cdot s_{\text{tpot}}$

Trong đó:

$$s_{\text{ttft}} = \left[ \text{clamp}\left( \frac{C_{\text{ttft}} - \text{TTFT}}{C_{\text{ttft}} - F_{\text{ttft}}}, 0, 1 \right) \right]^{\gamma}$$

$$s_{\text{tpot}} = \left[ \text{clamp}\left( \frac{C_{\text{tpot}} - \text{TPOT}_{\text{mean}}}{C_{\text{tpot}} - F_{\text{tpot}}}, 0, 1 \right) \right]^{\gamma}$$

#### Tham số cấu hình hệ thống
- $F_{\text{ttft}}$: Floor của TTFT, đạt mức này hoặc thấp hơn nhận điểm tối đa = 10 ms
- $C_{\text{ttft}}$: Ceiling của TTFT, chạm mức này hoặc cao hơn bị 0 điểm = 400 ms
- $F_{\text{tpot}}$: Floor của TPOT = 1 ms
- $C_{\text{tpot}}$: Ceiling của TPOT = 10 ms
- $\gamma$: Hệ số lũy thừa quy định độ dốc hàm phạt = 2
- $w$: Trọng số ưu tiên của TTFT = 0.5

### 3.2. Cổng chất lượng: Accuracy Gate (hậu kiểm sau vòng online)

Vòng online không chấm accuracy trên từng lượt nộp. Khi vòng online kết thúc, mỗi đội chọn thủ công tối đa 5 bài submissions tốt nhất (giữ nguyên Docker image/digest đã nộp). BTC sẽ hậu kiểm tính hợp lệ và chạy chấm điểm GPQA Diamond full qua lm-evaluation-harness (filter strict-match).

Hàm tính độ suy giảm độ chính xác so với baseline mẫu BF16 (mặc định là 0.4):

$$\Delta = \text{Accuracy}_{\text{baseline}} - \text{Accuracy}_{\text{submission}}$$

Dựa trên $\Delta$, hệ thống áp dụng hàm phạt $f(\Delta)$ theo dạng piecewise linear (đầu ra thuộc $[0,1]$):

$$f(\Delta) = \begin{cases}
1.0 & \text{nếu } \Delta \le 0.1 \\
1.0 - \frac{\Delta - 0.10}{0.06} & \text{nếu } 0.1 < \Delta < 0.16 \\
0.0 & \text{nếu } \Delta \ge 0.16
\end{cases}$$

Điểm chính thức của đội là điểm số tốt nhất trong các bài nộp còn hợp lệ sau hậu kiểm.

---

## 4. Không gian & Phương pháp tối ưu được phép

Thí sinh chỉ được phép sử dụng serving framework vLLM cho bài thi này. Các hướng tiếp cận được khuyến khích bao gồm:

- Quantization: các kỹ thuật online quantization (cân nhắc tác động đến Accuracy Gate)
- KV Cache & Memory: Paged Attention, KV cache quantization (FP8, INT8), prefix caching, semantic caching, offloading xuống CPU/NVMe
- Serving & Scheduling: continuous/dynamic batching, speculative decoding, disaggregated prefill/decode serving, memory-aware scheduling
- System & Runtime: custom CUDA/Triton kernels, tích hợp fused attention kernels (FlashAttention, FlashInfer), tối ưu hóa memory layout và CUDA Graphs

---

## 5. Quy trình nộp bài

### 5.1. Phát triển và đóng gói
- Phát triển mã nguồn tối ưu và đóng gói thành một Docker image công khai trên Docker Hub

### 5.2. Nộp bài
- Gửi file cấu hình docker-compose.yml lên portal của BTC
- Khai báo chính xác đường dẫn image và lệnh thực thi

### 5.3. Đánh giá tự động
- Hệ thống tự động pull image, dựng container trên MiG H200, kiểm tra healthcheck và chạy benchmark tính điểm ERS cập nhật lên leaderboard online

### 5.4. Mẫu docker-compose.yml

```yaml
services:
  model:
    image: vllm/vllm-openai:v0.22.1
    entrypoint:
      - python3
      - -m
      - vllm.entrypoints.openai.api_server
    command:
      - --model=/model
      - --served-model-name=LFM2.5-1.2B-Instruct
      - --host=0.0.0.0
      - --port=8000
      - --max-model-len=32768
      - --gpu-memory-utilization=0.95
      - --tensor-parallel-size=1
      - --enable-prefix-caching
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

---

## 6. Quy định chống gian lận & Phân định thứ hạng

### 6.1. Các hành vi nghiêm cấm (Anti-Cheating)

Giải pháp phải hướng tới việc phục vụ trung thực, sẵn sàng ứng dụng cho production thực tế. Mọi thủ thuật đánh lừa hệ thống đo lường đều bị hủy (void) kết quả:

- Pre-bake / hardcode: tính toán hoặc lưu sẵn đáp án thay vì suy luận thực tế tại thời điểm gọi endpoint
- Dual-path: rẽ nhánh hành vi xử lý khác nhau giữa lúc hệ thống đo độ trễ (vòng online) và lúc kiểm tra chất lượng (GPQA)
- Gaming metrics: đệm ký tự rỗng, cắt ngắn chuỗi sinh trái phép để né cổng hậu kiểm
- Can thiệp hạ tầng: thực hiện cuộc gọi mạng ra ngoài, sửa đổi tokenizer hoặc weights gốc của mô hình, làm bẩn tài nguyên
- Bất trung thực quy trình: thay đổi hoặc tráo đổi Docker image sau khi đã khóa cổng nộp bài

### 6.2. Tiêu chí phân định vùng sát điểm (Tie-break)

Đối với các đội có điểm số bám sát nhau trong vùng nhiễu đo lường (biên độ sai số nhỏ hơn hoặc bằng 1–2 điểm), thứ hạng được phân định tuần tự theo:

1. Mức độ suy giảm độ chính xác $\Delta$ thấp hơn
2. Chỉ số độ trễ p95 TTFT thấp hơn
3. Tốc độ sinh văn bản (generation speed) cao hơn
4. Thời điểm nộp bài hợp lệ sớm hơn

### 6.3. Quy trình khiếu nại & Re-grade

BTC có quyền chạy độc lập nhiều lần trên đúng bản Docker image đã chốt để lấy điểm trung vị nhằm đảm bảo tính công bằng (đặc biệt là nhóm tranh chấp giải thưởng).

BTC sẽ gửi email thông báo kết quả dự kiến trước khi chốt bảng xếp hạng. Mọi khiếu nại phải được gửi về BTC trong thời hạn tối đa 24 giờ kể từ thời điểm nhận thông báo hoặc công bố kết quả.

---

## 7. Thông tin bổ sung

- Mô hình chỉ định: LiquidAI/LFM2.5-1.2B-Instruct
- Link tải weights: https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct
- Link Docker image baseline từ BTC: https://hub.docker.com/layers/vllm/vllm-openai/v0.22.1/images/sha256-55c9bcee9fc66644b139fddae8a7a03e4c0c8a25ab5c64b0ce614554a8abf5d5
