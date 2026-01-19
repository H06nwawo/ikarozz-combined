FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.11 python3-pip git wget supervisor && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    vllm diffusers transformers accelerate \
    torch torchvision safetensors pillow flask

WORKDIR /app
COPY flux_server.py /app/
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 8000 8001

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
