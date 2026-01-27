---
summary: How to run verl based code—entrypoints (main_ppo.py, etc.), CLI patterns, local testing tips.
read_when:
  - Running experiments
  - Debugging runs
  - Understanding CLI overrides
  - Setting up local testing
---

# Running verl-based code

RL experiments typically run long and need multiple GPUs/nodes. However, for testing and debugging, you need to run verl code locally on your machine.
This doc explains how verl code runs—entrypoints, config system, and practical examples.
Note: This doc is only for end-to-end running of verl-based experiments and runs. For the unit tests, see `agent-docs/testing-guide.md`.

## The Entrypoint

verl uses Python module execution (`python3 -m`) with Hydra configuration. The offical main entrypoints live in `verl/trainer/main_*.py`:

| Entrypoint | Purpose |
|------------|---------|
| `main_ppo.py` | **Primary**: RL training (PPO, GRPO, DAPO, etc.) |
| `main_eval.py` | Offline evaluation using reward model + ground truth |
| `main_generation.py` | Batch generation from prompts (FSDP-based) |
| `main_generation_server.py` | Generation via OpenAI-compatible server (vLLM/sglang) |
| `fsdp_sft_trainer.py` | Supervised fine-tuning (SFT) |

### How `main_ppo.py` Works

The entrypoint follows this flow:

```
main() [Hydra decorated]
    └── run_ppo(config)
            ├── ray.init()  # Initialize Ray cluster
            └── TaskRunner.remote().run(config)  # Spawn remote task
                    ├── Load tokenizer/processor
                    ├── Create datasets + reward functions
                    ├── Configuring and initializing RayPPOTrainer
                    └── trainer.fit()  # Start training loop in the same process
```

Key design patterns:

1. **Hydra Config**: `@hydra.main(config_path="config", config_name="ppo_trainer")` loads YAML configs
2. **Ray Remote Task**: Training runs as a Ray remote class (`TaskRunner`) to avoid head node scheduling

### TaskRunner Class Breifly Explained

The `TaskRunner` is a Ray remote actor that orchestrates training:

```python
@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        # 1. Configure the trainer
        # 2. Load tokenizer, datasets
        tokenizer = hf_tokenizer(local_path)
        train_dataset = create_rl_dataset(...)
        
        # 3. Initialize trainer
        trainer = RayPPOTrainer(config=config, ...)
        trainer.init_workers()
        trainer.fit()
```

## Running

### General Structure

Run entrypoints as Python modules with Hydra CLI overrides:

```bash
python -m verl.trainer.<entrypoint> \
    <config.path>=<value> \
    <config.nested.path>=<value> \
    ...
```

Note that <config.path>=<value> and <config.nested.path>=<value> are optional as the entrypoint usually sets the default paths and config.

### CLI Override Pattern

Hydra allows overriding any config field from command line using dot notation:

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \              # Change algorithm
    data.train_files=$HOME/data/train.parquet \ # Set data path
    data.train_batch_size=1024 \                # Batch size
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.rollout.name=sglang \     # Rollout backend
    actor_rollout_ref.rollout.n=5 \             # Samples per prompt
    trainer.n_gpus_per_node=4 \                 # set based on the number of GPUs you have (use value from `nvidia-smi` in ur script)
    trainer.nnodes=1 \                          # Number of nodes
    trainer.logger='["console","wandb"]'        # Note: lists need quotes
```

Use `+config.path=value` to add new fields not in the base config.

Useful environment variables for debugging:
- `VERL_LOGGING_LEVEL`: Set the logging level for verl.

For the complete config reference (all fields: data, actor_rollout_ref, critic, reward_model, algorithm, trainer), see `docs/examples/config.rst`.

### Where to Find More Examples

| Location | Description |
|----------|-------------|
| `docs/examples/gsm8k_example.rst` | Complete GSM8K walkthrough (data prep → SFT → PPO) |
| `docs/examples/config.rst` | Detailed config field explanations |
| `docs/start/quickstart.rst` | Minimal quickstart guide |
| `examples/**.sh` | Production-ready shell scripts |
| `recipes/**.sh` | Research recipes with custom trainers |

## See Also

- `verl-framework-guide.md` - Architecture overview
- `ppo-trainer-lifecycle.md` - Training loop details  
- `config-system.md` - Deep dive into Hydra configs
- `testing-guide.md` - Testing practices
