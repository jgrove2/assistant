FROM python:3.11-slim AS base

ARG TARGETARCH

RUN apt-get update && apt-get install -y \
    wget \
    curl \
    libgomp1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN if [ "$TARGETARCH" = "arm64" ]; then \
        PIPER_ARCH="aarch64"; \
    else \
        PIPER_ARCH="x86_64"; \
    fi && \
    wget -q "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_${PIPER_ARCH}.tar.gz" -O piper.tar.gz && \
    tar -xzf piper.tar.gz && \
    rm piper.tar.gz

RUN wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx" -O /app/voice.onnx && \
    wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" -O /app/voice.onnx.json

RUN pip install --no-cache-dir flask

COPY server.py .

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

CMD ["python", "server.py"]
