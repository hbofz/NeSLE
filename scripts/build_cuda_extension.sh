#!/usr/bin/env sh
# Kept for callers that still invoke the shell entry point.
exec "${PYTHON:-python3}" "$(dirname "$0")/build_cuda_extension.py"
