FROM vllm/vllm-openai:v0.22.1

# Copy custom entrypoint
COPY entrypoint.py /app/entrypoint.py
RUN chmod +x /app/entrypoint.py