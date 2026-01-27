# Scripts Index

Quick reference for all utility scripts in this directory. The index might not be up to date. If you don't find what you need here, check the scripts directly.

## Model Conversion & Checkpointing

- **converter_hf_to_mcore.py** - Converts HuggingFace model checkpoints to Megatron-Core format. Supports single-rank or distributed conversion (via torchrun) for large models like DeepseekV3, Qwen2/3-MoE, and VL models. Handles pipeline/expert parallelism.

- **legacy_model_merger.py** - Merges FSDP or Megatron distributed checkpoints back into HuggingFace format. Supports both `merge` (save to disk) and `test` (validate against reference) operations. Can optionally upload merged model to HuggingFace Hub.

- **megatron_merge_lora.py** - Merges LoRA adapter weights into the base Megatron model. Uses Ray workers to load model with adapter and save merged weights. Requires the same config as training plus `adapter_path`.

- **init_random_model.py** - Creates a smaller model with random weights for debugging. Takes an existing HF model and a custom config JSON to override dimensions (num_layers, hidden_size, etc). Useful for fast iteration during development.

## Environment & Diagnostics

- **diagnose.py** - Comprehensive system diagnostic script. Checks Python/pip versions, verl installation, OS/hardware info, GPU memory, CUDA versions, network connectivity, and environment variables. Run with `--help` to see all options.

- **install_vllm_sglang_mcore.sh** - One-shot installation script for inference frameworks (vLLM, SGLang) and Megatron dependencies. Installs flash-attention, TransformerEngine, and patches opencv. Control what to install via `USE_MEGATRON` and `USE_SGLANG` env vars.

## Configuration & Documentation

- **generate_trainer_config.sh** - Auto-generates flattened reference YAML configs from the Hydra trainer configs. Runs `print_cfg.py` internally and validates that generated configs are up-to-date via git diff. Used in CI to catch config drift.

- **print_cfg.py** - Simple Hydra config printer. Loads a trainer config and prints it to stdout. Used by `generate_trainer_config.sh` to produce reference YAML files.

- **docs-list.py** - Lists all markdown files under `docs/` and extracts front-matter metadata (summary, read_when hints). Enforces documentation hygiene by flagging files missing required fields.

## Visualization

- **rollout_viewer.py** - TUI (terminal UI) application for browsing rollout JSONL logs. Navigate samples/steps with keyboard shortcuts, search content, filter fields, jump by request_id, and sort by score. Requires `textual==0.52.1`.

## VeOmni (MoE Utilities)

- **veomni/moe_merge.py** - Merges individual MoE expert weights into stacked tensors for efficient loading. Converts `experts.{j}.gate_proj.weight` format to `experts.gate_proj` (stacked). Supports Qwen3-MoE and DeepSeek formats.

- **veomni/moe_split.py** - Reverse of `moe_merge.py`. Splits stacked MoE expert tensors back to individual expert weights. Use this to restore HF-compatible format after training with merged weights.
