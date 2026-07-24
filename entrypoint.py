#!/usr/bin/env python3
"""Custom entrypoint for optimized vLLM serving on H200 MIG (3 CPU cores)."""
import os
import sys

# === Performance Environment Variables ===

# PyTorch memory allocator: expandable segments giảm fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Giảm log overhead (mỗi log line tốn CPU cycles trên 3 cores)
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

# Disable tokenizer parallelism (tránh thread contention trên 3 CPU cores)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Force aggressive torch.compile optimization
os.environ["VLLM_TORCH_COMPILE_LEVEL"] = "3"

# Launch vLLM server with all arguments passed through
cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"] + sys.argv[1:]
os.execvp(sys.executable, cmd)