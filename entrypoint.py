#!/usr/bin/env python3
"""Custom entrypoint combining V13-V15 optimizations for H200 MIG (3 CPU cores, 8GB RAM)."""
import os
import sys

# === Performance & Thread Bounds (3 CPU Cores) ===
os.environ["OMP_NUM_THREADS"] = "3"
os.environ["MKL_NUM_THREADS"] = "3"
os.environ["OPENBLAS_NUM_THREADS"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === Memory & Compilation Optimizations ===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
os.environ["VLLM_TORCH_COMPILE_LEVEL"] = "3"

# Launch vLLM server with passed arguments
cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"] + sys.argv[1:]
os.execvp(sys.executable, cmd)