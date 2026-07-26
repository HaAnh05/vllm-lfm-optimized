#!/usr/bin/env python3
"""Custom entrypoint for optimized vLLM serving on H200 MIG (3 CPU cores, 8GB RAM)."""
import os
import sys

# === Performance Environment Variables ===

# Restrict CPU threads to match the 3 physical CPU cores (prevents context switching & thread thrashing)
os.environ["OMP_NUM_THREADS"] = "3"
os.environ["MKL_NUM_THREADS"] = "3"
os.environ["OPENBLAS_NUM_THREADS"] = "3"

# PyTorch memory allocator: expandable segments reduces VRAM fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Reduce logging CPU overhead on 3 CPU cores
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

# Disable tokenizer parallelism to prevent thread contention on 3 CPU cores
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Aggressive PyTorch compilation for kernel fusion
os.environ["VLLM_TORCH_COMPILE_LEVEL"] = "3"

# Launch vLLM server with all arguments passed through
cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"] + sys.argv[1:]
os.execvp(sys.executable, cmd)