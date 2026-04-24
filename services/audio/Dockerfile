FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    gcc \
    g++ \
    make \
    python3-dev \
    libasound2-plugins \
    pulseaudio-utils \
    && rm -rf /var/lib/apt/lists/*

RUN echo 'pcm.!default { type pulse }\nctl.!default { type pulse }' > /etc/asound.conf

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from openwakeword.utils import download_models; download_models()"

COPY . .

RUN useradd -m -u 1000 appuser && \
    groupadd -f -g 44 video && \
    usermod -aG video appuser
USER appuser

CMD ["python", "server.py"]
