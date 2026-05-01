#!/usr/bin/env sh
set -eu

python_bin="${PYTHON:-python3}"
PYTHONPATH=src "$python_bin" scripts/verify_phase4.py
