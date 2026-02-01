#!/bin/bash
set -xeuo pipefail

# E2E smoke test: GRPO on GSM8K via Task system
# Requires: 1 GPU

cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

PYTHONUNBUFFERED=1 python3 treetune_tests/tasks/e2e_entrypoint.py

echo "E2E test passed!"
