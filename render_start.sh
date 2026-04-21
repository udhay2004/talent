#!/bin/bash
apt-get install -y ffmpeg
gunicorn app:app --workers=1 --timeout=300 --bind=0.0.0.0:$PORT
