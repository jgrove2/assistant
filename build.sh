#!/usr/bin/env bash
set -euo pipefail

BUILDER_NAME="multiarch-builder"

if ! docker buildx inspect "$BUILDER_NAME" &>/dev/null; then
    docker buildx create --name "$BUILDER_NAME" --use
else
    docker buildx use "$BUILDER_NAME"
fi

docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --tag bot:latest \
    --file Dockerfile \
    .

docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --tag piper:latest \
    --file piper/Dockerfile \
    ./piper
