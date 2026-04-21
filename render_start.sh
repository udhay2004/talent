#!/bin/bash
export PYTHONPATH=/opt/render/project/src
exec gunicorn app:app --workers=1 --timeout=300 --bind=0.0.0.0:$PORT
