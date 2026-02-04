#!/bin/bash
set -xeuo pipefail

# E2E test: GenerationRunner smoke test
# Requires: 1 GPU (NVIDIA A100/H100 or similar)
#
# Runs GenerationRunner with a small model (Qwen2.5-0.5B) and
# GSM8K subset (5 samples), verifies output files exist.

cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

PYTHONUNBUFFERED=1 python3 \
    treetune_tests/treetune_verl/generation/e2e_generation_entrypoint.py

echo "E2E generation test passed!"
