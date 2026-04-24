FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    alsa-utils \
    libsndfile1 \
    gcc \
    g++ \
    make \
    python3-dev \
    libasound2-plugins \
    pulseaudio-utils \
    gosu \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libv4l-dev \
    && rm -rf /var/lib/apt/lists/*

RUN echo 'pcm.!default { type pulse }\nctl.!default { type pulse }' > /etc/asound.conf

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from openwakeword.utils import download_models; download_models()"

COPY src/ ./src/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN useradd -m -u 1000 appuser && \
    groupadd -f -g 44 video && \
    usermod -aG video appuser

ENTRYPOINT ["/entrypoint.sh"]
