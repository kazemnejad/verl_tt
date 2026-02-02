# Copyright 2025 Individual Contributor: Amirhossein Kazemnejad
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""E2E test: verify rollout_entropy appears in DataProto during GRPO training.

Runs a single GRPO training step with our entropy-aware pipeline
(VerifyingEntropyManager -> EntropySGLangReplica -> VerifyingEntropyWorker)
and asserts that ``rollout_entropy`` is present in the DataProto batch.

The config points ``agent_loop_manager_class`` at
``VerifyingEntropyManager``, which uses ``VerifyingEntropyWorker`` — a
subclass that writes a JSON signal file when rollout_entropy is found.

Requires: 1 GPU (NVIDIA A100 or similar).
"""

import json
import os
import tempfile

import hydra

# Signal file path — set BEFORE run_with_tasks so Ray propagates it to workers
_SIGNAL_FILE = tempfile.mktemp(prefix="entropy_e2e_", suffix=".json")
os.environ["TREETUNE_E2E_ENTROPY_SIGNAL"] = _SIGNAL_FILE


@hydra.main(
    config_path="e2e_entropy_config",
    config_name="grpo_gsm8k_entropy",
    version_base=None,
)
def main(config):
    from treetune_verl.tasks import run_with_tasks

    run_with_tasks(config)

    # Check signal file written by VerifyingEntropyWorker._postprocess
    signal_path = os.environ.get("TREETUNE_E2E_ENTROPY_SIGNAL", _SIGNAL_FILE)
    if not os.path.exists(signal_path):
        raise AssertionError(
            "rollout_entropy signal file was never written! "
            "VerifyingEntropyWorker._postprocess did not produce rollout_entropy. "
            f"Expected signal at: {signal_path}"
        )

    with open(signal_path) as f:
        signal = json.load(f)

    os.unlink(signal_path)

    assert signal["verified"], "Signal file exists but verified=False"
    assert signal["shape"] == signal["expected_shape"], (
        f"Shape mismatch: {signal['shape']} != {signal['expected_shape']}"
    )
    assert signal["all_non_negative"], f"Entropy values contain significant negatives: min={signal['min_value']}"
    assert signal["num_positive"] > 0, "No positive entropy values found"

    print(
        f"[E2E] E2E entropy test PASSED: "
        f"shape={signal['shape']}, "
        f"min={signal['min_value']:.6f}, "
        f"mean_positive={signal['mean_positive']:.4f}, "
        f"num_positive={signal['num_positive']}"
    )


if __name__ == "__main__":
    main()
