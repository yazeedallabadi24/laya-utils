FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set the working directory
WORKDIR /app

# Install dependencies using the CPU-only PyTorch index
RUN uv pip install --system --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "laya[onnx]" onnx

# Pre-cache config and tokenizer files (~3.5MB) so the container can run offline and instantly
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('convaiinnovations/laya', allow_patterns=['rl_agent_config.json', 'tokenizer/*', 'encoder/*'])"

# Copy scripts and pre-built INT8 model if available in context
COPY main.py export_onnx.py laya.int8.onn* ./

# If pre-built model is not in context (e.g. freshly cloned repo), generate and quantize it
RUN if [ ! -f laya.int8.onnx ]; then python export_onnx.py --quantize && rm -f laya.onnx; fi

# Entrypoint forwards any arguments to main.py
ENTRYPOINT ["python", "main.py"]
CMD []

