#!/bin/bash
# Simple script to run the restaurant demand forecasting API

cd "$(dirname "$0")"
PYTHONPATH=. python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
