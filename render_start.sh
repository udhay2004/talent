#!/bin/bash
apt-get install -y ffmpeg 2>/dev/null || true
cd /opt/render/project/src
exec gunicorn app:app --workers=1 --timeout=300 --bind=0.0.0.0:$PORT
