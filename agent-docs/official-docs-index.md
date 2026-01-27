# verl Official Documentation Index

> Last updated: 2026-01-26.
> Comprehensive navigation guide for verl's 82 documentation files across 14 sections.
> Note: This index might not always be up to date. If you don't find what you need here, check `docs/` directly.
> Note: The official docs themselves might be outdated too. Always use the actual source code as the ground truth.


---

## Root Programming Guides

### HybridFlow Programming Guide
**Path**: `docs/hybrid_flow.rst`

Introduces verl's core HybridFlow architecture for representing RL training as a two-level dataflow: control flow (algorithm logic) and computation flow (neural network training). Covers the design philosophy separating single-process controllers from multi-process computation. **When to read**: Understanding verl's architectural foundation; designing new RL algorithms; learning how controller-worker communication works.

### The Design of `verl.single_controller`
**Path**: `docs/single_controller.rst`

Deep dive into the single_controller module for verl contributors—explains WorkerGroup, ResourcePool, and ClassWithArgs abstractions. Walks through the `generate_sequences` example showing how methods are registered and invoked across distributed workers. **When to read**: Extending verl internals; understanding Ray-based distributed coordination; implementing new Worker classes.

---

## Quickstart

### Installation
**Path**: `docs/start/install.rst`

Complete installation guide covering Python/CUDA requirements, backend choices (FSDP/Megatron for training; vLLM/SGLang/TGI for inference), Docker images, and pip installation. **When to read**: First-time setup; switching between inference backends; troubleshooting environment issues.

### Quickstart: PPO Training on GSM8K
**Path**: `docs/start/quickstart.rst`

Step-by-step tutorial training a Qwen2.5-0.5B model on GSM8K math problems using PPO with rule-based rewards. Includes dataset prep, model download, and training script with all key parameters. **When to read**: Getting your first verl training job running; understanding basic PPO configuration.

### Multinode Training
**Path**: `docs/start/multinode.rst`

Guides for scaling to multiple nodes: manual Ray cluster setup, job submission via `ray job submit`, and automated cloud deployment via SkyPilot. **When to read**: Scaling beyond single-node; deploying on clusters; using SkyPilot on GCP/AWS/Kubernetes.

### Ray Debug Tutorial
**Path**: `docs/start/ray_debug_tutorial.rst`

Tutorial for debugging distributed Ray jobs including breakpoints, accessing logs, and troubleshooting common issues. **When to read**: Debugging distributed training; understanding Ray worker coordination.

### More Resources
**Path**: `docs/start/more_resources.rst`

Collection of additional learning materials, community links, and external resources for verl users. **When to read**: Finding examples, tutorials, and community discussions.

### Agentic RL
**Path**: `docs/start/agentic_rl.rst`

Introduction to agentic reinforcement learning with tool-calling capabilities during rollout. **When to read**: Building agents that interact with environments; multi-turn tool use training.

---

## Data Preparation

### Prepare Data
**Path**: `docs/preparation/prepare_data.rst`

Data preprocessing guide for RL training: required parquet format, field specifications (prompts, responses, rewards), and data loading utilities. **When to read**: Preparing custom datasets; understanding verl's data requirements.

### Reward Function
**Path**: `docs/preparation/reward_function.rst`

How to implement custom reward functions combining rule-based and model-based rewards. Covers RewardManager interface and integration with training loop. **When to read**: Implementing custom rewards; combining multiple reward signals.

---

## Configurations

### Configuration Guide
**Path**: `docs/examples/config.rst`

Complete reference for verl's YAML configuration system covering all parameters: data, actor, critic, rollout, trainer settings. **When to read**: Tuning hyperparameters; understanding configuration hierarchy; debugging config issues.

---

## PPO Example

### PPO Code Architecture
**Path**: `docs/examples/ppo_code_architecture.rst`

Detailed walkthrough of verl's PPO implementation: entry points, trainer structure, worker roles, and data flow between components. **When to read**: Understanding PPO implementation details; modifying the training loop.

### GSM8K Example
**Path**: `docs/examples/gsm8k_example.rst`

Extended GSM8K tutorial with SFT pre-training, checkpoint management, and evaluation. More comprehensive than quickstart. **When to read**: Complete end-to-end example; learning SFT-to-RL pipeline.

### Multi-Modal Example
**Path**: `docs/examples/multi_modal_example.rst`

Training vision-language models (VLMs) with verl, handling multi-modal inputs during rollout and training. **When to read**: Training VLMs; working with image/video inputs in RL.

### SkyPilot Examples
**Path**: `docs/examples/skypilot_examples.rst`

Ready-to-use SkyPilot configurations for cloud deployment: PPO, GRPO, and multi-turn training setups. **When to read**: Deploying on clouds; auto-scaling training jobs.

---

## Algorithms

### PPO
**Path**: `docs/algo/ppo.md`

Proximal Policy Optimization implementation details: clipped objective, GAE computation, and verl-specific optimizations. **When to read**: Understanding PPO internals; tuning PPO hyperparameters.

### GRPO
**Path**: `docs/algo/grpo.md`

Group Relative Policy Optimization—generates N responses per prompt, uses relative ranking for advantage. Simpler than PPO (no critic). **When to read**: Using GRPO as PPO alternative; understanding group-based RL.

### CollabLLM
**Path**: `docs/algo/collabllm.md`

Collaborative LLM training with multiple agent interaction patterns. **When to read**: Multi-agent RL scenarios.

### DAPO
**Path**: `docs/algo/dapo.md`

Decoupled Alignment via PPO—separates alignment objectives for better training stability. **When to read**: Alignment-focused training; DAPO recipe configuration.

### SPIN
**Path**: `docs/algo/spin.md`

Self-Play fINe-tuning—uses self-play dynamics for iterative improvement. **When to read**: Self-play RL approaches.

### SPPO
**Path**: `docs/algo/sppo.md`

Self-Play Preference Optimization—combines preference learning with self-play. **When to read**: Preference-based self-play training.

### Entropy Bonus
**Path**: `docs/algo/entropy.md`

Configuring entropy bonus for exploration: coefficients, scheduling, and interaction with other losses. **When to read**: Tuning exploration; preventing policy collapse.

### OPO
**Path**: `docs/algo/opo.md`

Online Preference Optimization algorithm details. **When to read**: Online preference learning approaches.

### Baseline Methods
**Path**: `docs/algo/baseline.md`

Reference implementations for baseline RL methods used in comparisons. **When to read**: Benchmarking; understanding baseline configurations.

### GPG
**Path**: `docs/algo/gpg.md`

Generalized Policy Gradient algorithm implementation. **When to read**: Alternative policy gradient methods.

### Rollout Correction
**Path**: `docs/algo/rollout_corr.md`

Techniques for correcting distribution mismatch between rollout and training (TIS—Token Importance Sampling). **When to read**: Handling FP8/async mismatch; improving training stability.

### Rollout Correction Math
**Path**: `docs/algo/rollout_corr_math.md`

Mathematical foundations for rollout correction—derivations and theoretical justification. **When to read**: Understanding TIS theory; research on correction methods.

### OTB (Off-The-Batch)
**Path**: `docs/algo/otb.md`

Off-the-batch training techniques for sample efficiency. **When to read**: Improving sample utilization.

---

## PPO Trainer and Workers

### Ray Trainer
**Path**: `docs/workers/ray_trainer.rst`

RayPPOTrainer architecture: resource pools, worker groups, training loop orchestration. **When to read**: Understanding trainer coordination; extending trainer logic.

### FSDP Workers
**Path**: `docs/workers/fsdp_workers.rst`

FSDP backend workers: ActorRolloutRefWorker, CriticWorker implementation and configuration. Recommended for prototyping. **When to read**: Using FSDP backend; debugging worker issues; day-0 model support.

### Megatron Workers
**Path**: `docs/workers/megatron_workers.rst`

Megatron-LM backend for high scalability with 3D parallelism (DP+TP+PP+EP+CP). **When to read**: Scaling to very large models; production deployment.

### SGLang Worker
**Path**: `docs/workers/sglang_worker.rst`

SGLang rollout backend configuration: server mode, async generation, advanced features. **When to read**: Using SGLang for inference; multi-turn rollout.

### TensorRT-LLM Worker
**Path**: `docs/workers/trtllm_worker.rst`

TensorRT-LLM rollout backend for optimized NVIDIA inference. **When to read**: Maximum inference throughput; TensorRT deployment.

### Model Engine
**Path**: `docs/workers/model_engine.rst`

Abstract Model Engine interface: `initialize`, `forward_backward_batch`, `optimizer_step`. Backend-agnostic training API. **When to read**: Understanding backend abstraction; adding new training backends.

---

## Performance Tuning Guide

### DeepSeek Performance Tips
**Path**: `docs/perf/dpsk.md`

Performance optimizations specific to DeepSeek models—MoE handling, memory management. **When to read**: Training DeepSeek models; MoE optimization.

### Best Practices
**Path**: `docs/perf/best_practices.rst`

General performance best practices: batch sizing, memory optimization, GPU utilization tips. **When to read**: Optimizing training throughput; reducing memory usage.

### Performance Tuning
**Path**: `docs/perf/perf_tuning.rst`

Detailed tuning guide: gradient checkpointing, offloading, micro-batching strategies. **When to read**: Systematic performance optimization; debugging slow training.

### vLLM 0.8+ Upgrade Guide
**Path**: `docs/README_vllm0.8.md`

Migration guide for vLLM 0.8+: enabling CUDA graph, V1 engine, dependency updates. **When to read**: Upgrading vLLM version; fixing vLLM compatibility issues.

### Hardware Resource Guide
**Path**: `docs/perf/device_tuning.rst`

Reference tables for GPU requirements by model size (0.5B to 70B+), tested configurations. **When to read**: Planning hardware allocation; finding working configs for specific models.

### verl Profiler System
**Path**: `docs/perf/verl_profiler_system.md`

Built-in profiler architecture: global config, role-level profiling, adding new tools. **When to read**: Profiling training runs; adding custom profiling.

### Nsight Profiling
**Path**: `docs/perf/nsight_profiling.md`

NVIDIA Nsight Systems integration: configuration, discrete mode, finding output files. **When to read**: Deep GPU profiling; CUDA kernel analysis.

---

## Adding New Models

### FSDP Extension
**Path**: `docs/advance/fsdp_extension.rst`

Adding model support to FSDP backend: dtensor_weight_loader implementation for memory-efficient weight sync. Lists supported models (Llama, Qwen, Gemma, etc.). **When to read**: Adding new HuggingFace models; implementing weight loaders.

### Megatron Extension
**Path**: `docs/advance/megatron_extension.rst`

Adding models to Megatron backend: GPTModel, TransformerLayerSpec, model initialization. **When to read**: Megatron model integration; custom architectures.

---

## Advanced Features

### Checkpoint Management
**Path**: `docs/advance/checkpoint.rst`

Checkpoint saving/loading: directory structure for FSDP and Megatron, HuggingFace conversion tool (`verl.model_merger`). **When to read**: Fault tolerance setup; checkpoint conversion; resuming training.

### RoPE Scaling Override
**Path**: `docs/advance/rope.rst`

Enabling RoPE scaling for extended context (e.g., YaRN) via `override_config`. **When to read**: Training with longer contexts than default.

### Attention Implementation Override
**Path**: `docs/advance/attention_implementation.rst`

Switching attention: `flash_attention_2`, `eager`, `sdpa`. Useful for debugging and compatibility. **When to read**: Debugging attention errors; testing different implementations.

### LoRA Support
**Path**: `docs/advance/ppo_lora.rst`

Parameter-efficient RL training with LoRA: FSDP and Megatron configurations, 70B+ training on limited hardware. **When to read**: Training large models efficiently; LoRA adapter management.

### Multi-turn Rollout
**Path**: `docs/sglang_multiturn/multiturn.rst`

Configuring multi-turn conversations: tool definitions, interaction configs, multi-modal tool responses. **When to read**: Building conversational agents; tool-calling during rollout.

### Interaction System
**Path**: `docs/sglang_multiturn/interaction_system.rst`

BaseInteraction interface for multi-turn RL: instance management, async architecture, reward integration. **When to read**: Implementing custom interaction agents; curriculum learning.

### Placement Configuration
**Path**: `docs/advance/placement.rst`

Ray resource pool and worker placement tutorial: GPU sharing, resource isolation. **When to read**: Understanding Ray worker coordination; custom resource allocation.

### DPO Extension
**Path**: `docs/advance/dpo_extension.rst`

Extending verl for DPO (Direct Preference Optimization): step-by-step implementation guide. **When to read**: Implementing new RL algorithms; understanding verl extension patterns.

### Sandbox Fusion Example
**Path**: `docs/examples/sandbox_fusion_example.rst`

Training with remote code execution via Sandbox Fusion: setup, API configuration, Prime reward manager. **When to read**: Code verification rewards; sandboxed execution.

### Rollout Trace
**Path**: `docs/advance/rollout_trace.rst`

Tracing agentic rollouts with wandb Weave or MLflow: configuration, limiting trace volume, viewing trajectories. **When to read**: Debugging multi-turn rollouts; observability.

### Rollout Skip
**Path**: `docs/advance/rollout_skip.rst`

Caching rollout sequences to accelerate repeated experiments. **When to read**: Speeding up iterative experiments; debugging with cached data.

### One-Step Off-Policy Async
**Path**: `docs/advance/one_step_off.md`

Async trainer overlapping generation and training with one-step staleness. 23-40% speedup. **When to read**: Improving GPU utilization; async training setup.

### Agent Loop
**Path**: `docs/advance/agent_loop.rst`

AgentLoopBase interface for custom multi-turn rollout: pluggable loop, request-level load balancing. **When to read**: Custom agentic workflows; tool integration.

### Reward Loop
**Path**: `docs/advance/reward_loop.rst`

Async reward computation: hybrid rewards (rule + DisRM + GenRM), flexible configuration. **When to read**: Complex reward setups; generative reward models.

### Fully Async Policy Trainer
**Path**: `docs/advance/fully_async.md`

Complete async training with resource isolation: TransferQueue integration, 2.35-2.67x speedup. **When to read**: Maximum throughput; production async training.

### TransferQueue
**Path**: `docs/data/transfer_queue.md`

Asynchronous streaming data management for async/streaming training pipelines. **When to read**: Async architecture internals; custom data flow.

### Grafana/Prometheus Monitoring
**Path**: `docs/advance/grafana_prometheus.md`

Rollout monitoring with Prometheus metrics and Grafana dashboards: setup, auto-configuration. **When to read**: Production observability; identifying long-tail issues.

### FP8 Rollout
**Path**: `docs/advance/fp8.md`

FP8 quantized inference: blockwise quantization, TIS integration, 12-18% rollout speedup. **When to read**: Memory optimization; faster inference.

### Async On-Policy Distillation
**Path**: `docs/advance/async-on-policy-distill.md`

Knowledge distillation with async scheduling: teacher top-k distributions, one/two-step off-policy. **When to read**: Distilling from larger models; async distillation.

### MTP Guide
**Path**: `docs/advance/mtp.md`

Multi-Token Prediction for speculative decoding in RL: training and inference configuration. **When to read**: MTP-enabled models (mimo, Qwen-next, DeepSeek); speculative rollout.

---

## Hardware Support

### AMD ROCm Dockerfile
**Path**: `docs/amd_tutorial/amd_build_dockerfile_page.rst`

Building verl Docker image for AMD MI300 GPUs with ROCm platform. **When to read**: AMD GPU setup; ROCm environment.

### AMD Performance Tuning
**Path**: `docs/amd_tutorial/amd_vllm_page.rst`

vLLM sleep mode and CUDA graph workarounds for AMD GPUs. **When to read**: AMD-specific optimizations; troubleshooting ROCm issues.

### Ascend Quickstart
**Path**: `docs/ascend_tutorial/ascend_quick_start.rst`

Huawei Ascend NPU setup: CANN/torch_npu installation, supported hardware (Atlas 200T/900/800T). **When to read**: Ascend NPU deployment; Chinese hardware support.

### Ascend Consistency
**Path**: `docs/ascend_tutorial/ascend_consistency.rst`

Aligning verl and vLLM inference results on Ascend: environment variables, deterministic settings. **When to read**: Ascend reproducibility; debugging inference mismatch.

### Ascend Profiling (Chinese)
**Path**: `docs/ascend_tutorial/ascend_profiling_zh.rst`

NPU profiling guide in Chinese: level0/1/2 collection, FSDP/MindSpeed backends. **When to read**: Ascend performance analysis (Chinese readers).

### Ascend Profiling (English)
**Path**: `docs/ascend_tutorial/ascend_profiling_en.rst`

NPU profiling guide in English: same content as Chinese version. **When to read**: Ascend performance analysis (English readers).

### Ascend Dockerfile Guide
**Path**: `docs/ascend_tutorial/dockerfile_build_guidance.rst`

Building Ascend Docker images: component versions, public image locations. **When to read**: Ascend Docker setup; image selection.

### Ascend SGLang Quickstart
**Path**: `docs/ascend_tutorial/ascend_sglang_quick_start.rst`

SGLang backend on Ascend NPUs: installation, torch_memory_saver setup. **When to read**: SGLang on Ascend; NPU inference.

---

## API References

### Data Interface
**Path**: `docs/api/data.rst`

DataProto class documentation: TensorDict batch, meta_info, core APIs (to, select, union, concat). **When to read**: Working with verl's data structures; understanding data flow.

### Single Controller Interface
**Path**: `docs/api/single_controller.rst`

API docs for Worker, WorkerGroup, ResourcePool, RayWorkerGroup, create_colocated_worker_cls. **When to read**: Implementing custom workers; understanding Ray abstractions.

### Trainer Interface
**Path**: `docs/api/trainer.rst`

RayPPOTrainer API: init_workers, fit; tokenizer utilities; core algo functions (compute_policy_loss, kl_penalty). **When to read**: Customizing trainer; understanding loss computation.

### Utilities
**Path**: `docs/api/utils.rst`

Utility function docs: FSDP utils, checkpoint management, dataset utilities, tracking, metrics. **When to read**: Using helper functions; understanding utility implementations.

---

## Blog

### v0.7 Release
**Path**: `docs/blog/v0.7.md`

verl 0.7 release notes: Hybrid-Controller architecture overview, verl-core/verl-trainer layers, Model Engine backends. **When to read**: Understanding verl evolution; high-level architecture.

---

## FAQ

### Frequently Asked Questions
**Path**: `docs/faq/faq.rst`

Common issues and solutions: Ray debugging, Slurm clusters, tensordict errors, installation problems. **When to read**: Troubleshooting; common pitfalls.

---

## Development Notes

### Sandbox Fusion Tool Integration
**Path**: `docs/sglang_multiturn/sandbox_fusion.rst`

Internal design doc for Sandbox Fusion integration: tool schema, rate limiting, async conventions. **When to read**: Contributing to tool integration; understanding implementation decisions.
