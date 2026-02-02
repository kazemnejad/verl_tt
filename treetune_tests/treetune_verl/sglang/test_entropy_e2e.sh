#!/bin/bash
set -xeuo pipefail

# E2E test: entropy extraction through full verl GRPO pipeline
# Requires: 1 GPU (NVIDIA A100 or similar)
#
# Verifies that per-token entropy flows from sglang sampler through
# the entropy patches, agent loops, and worker post-processing into
# DataProto.batch["rollout_entropy"].

cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

PYTHONUNBUFFERED=1 python3 \
    treetune_tests/treetune_verl/sglang/e2e_entropy_entrypoint.py

echo "E2E entropy test passed!"
