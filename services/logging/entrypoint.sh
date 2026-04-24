#!/bin/sh
DOCKER_GID=$(stat -c '%g' /var/run/docker.sock 2>/dev/null || echo "")
if [ -n "$DOCKER_GID" ] && [ "$DOCKER_GID" != "0" ]; then
    groupadd -f -g "$DOCKER_GID" docker-host
    usermod -aG docker-host appuser
fi
mkdir -p /app/data
exec gosu appuser python server.py
