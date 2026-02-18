#!/usr/bin/env python3
"""Simple script to run the restaurant demand forecasting API."""
import subprocess
import sys
import os
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).parent
    os.chdir(project_root)
    cmd = [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
    subprocess.run(cmd)
