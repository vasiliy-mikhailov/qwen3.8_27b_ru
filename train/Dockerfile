# RTX 5090 is sm_120, which needs a CUDA 12.8 build of torch; the cu128 wheels
# are the first that ship sm_120 kernels.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv git && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
      torch && \
    pip install --no-cache-dir \
      transformers accelerate peft bitsandbytes datasets safetensors sentencepiece
WORKDIR /work
