"""Generation entrypoint with Hydra configuration."""

import os
import socket
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from treetune_verl.generation.runner import GenerationRunner
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils.device import auto_set_device


@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    auto_set_device(config)
    run_generation(config)


def run_generation(config) -> None:
    """Initialize Ray, load data, and run generation.

    Data loading follows the same pattern as main_ppo.py:
    tokenizer/processor/dataset created here, passed to runner.
    """
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_kwargs", {}).get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"Ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    print(f"GenerationRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
    pprint(OmegaConf.to_container(config, resolve=True))
    # NOTE: Don't call OmegaConf.resolve(config) globally — rollout defaults
    # contain oc.select interpolations that fail in-place resolution.
    # OmegaConf resolves lazily on attribute access.

    # Task system: resolve train_tasks → config.data.train_files (if configured)
    from treetune_verl.tasks import resolve_tasks_into_config

    resolve_tasks_into_config(config)

    # Data loading (same pattern as main_ppo.py)
    from verl.trainer.main_ppo import create_rl_dataset
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.utils.fs import copy_to_local

    local_path = copy_to_local(
        config.actor_rollout_ref.model.path,
        use_shm=config.actor_rollout_ref.model.get("use_shm", False),
    )
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

    dataset = create_rl_dataset(
        config.data.train_files,
        config.data,
        tokenizer,
        processor,
        max_samples=config.data.get("train_max_samples", -1),
    )

    runner = GenerationRunner(config, dataset, collate_fn)
    runner.run()

    timeline_json_file = config.get("ray_kwargs", {}).get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


if __name__ == "__main__":
    main()
