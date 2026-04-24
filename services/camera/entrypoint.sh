#!/bin/sh
mkdir -p /app/faces/images
chown -R appuser:appuser /app/faces
exec gosu appuser python server.py
