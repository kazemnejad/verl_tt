# PPO Trainer - Design & Lifecycle

## Overview

`RayPPOTrainer` is verl's default trainer for on-policy RL (PPO, GRPO, DAPO, etc.). It's a single-threaded orchestrator implementing the single-controller pattern, coordinating multiple distributed worker groups (SPMD blocks). Located at `verl/trainer/ppo/ray_trainer.py`.

Note that in this trainer, all worker groups are colocated (e.g. Actor, Rollout, Reference) and share the same physical resources (i.e. GPUs).

## Lifecycle

1. Entry script `verl/trainer/main_ppo.py` connects to (or initializes) a Ray cluster and schedules a task.
2. The task creates a `RayPPOTrainer` instance with configuration, `ResourcePoolManager`, and components (tokenizer, processor, role_worker_mapping, reward_fn, etc.).
3. `ResourcePoolManager` maps roles to resource pools (default: single global pool shared by all roles).
4. The trainer fused worker groups that belongs to the same resource pool, initializes/spawns worker groups and allocates resources across the cluster.
5. Once workers are ready, it loads any checkpoint and optionally runs validation.
6. The training loop begins: load batches, dispatch to worker groups, perform RL steps.


### Training Loop Lifecycle (`fit` method)

The `fit()` method (lines 1349-1741) is the main training loop:

1. **Load Checkpoint** → 2. **Initial Validation** → 3. **Epoch Loop**
   - **Batch Loop** (for each batch):
     1. Generation
     2. Reward computation
     3. Log probs computation
     4. Values computation
     5. Advantage computation
     6. Update (actor/critic)
   - Validation, Checkpoint, Logging (at intervals)

### Rough Loop Outline

```python
def fit(self):
    # 1. Load checkpoint if resuming
    current_step = self._load_checkpoint()

    # 2. Initial validation (optional)
    if self.config.trainer.val_before_train:
        self._validate()

    # 3. Epoch loop
    for epoch in range(current_epoch, total_epochs):

        # 4. Batch loop
        for batch_dict in self.train_dataloader:
            # Create DataProto from batch
            batch = DataProto.from_single_dict(batch_dict)

            # 5. SEQUENCE GENERATION
            gen_batch = self._prepare_generation_batch(batch)
            gen_output = self.async_rollout_manager.generate_sequences(gen_batch)
            batch = batch.union(gen_output)

            # 6. REWARD COMPUTATION
            if self.use_rm:
                rm_scores = self.rm_wg.compute_rm_score(batch)
                batch.update(rm_scores=rm_scores)
            rewards = self.reward_fn(batch)
            batch.update(rewards=rewards)

            # 7. LOG PROBABILITY COMPUTATION
            old_log_probs = self._compute_old_log_prob(batch)
            batch.update(old_log_probs=old_log_probs)

            # 8. REFERENCE LOG PROBS (if KL penalty)
            if self.use_reference_policy:
                ref_log_probs = self._compute_ref_log_prob(batch)
                batch.update(ref_log_prob=ref_log_probs)

            # 9. VALUE COMPUTATION (if using critic)
            if self.use_critic:
                values = self._compute_values(batch)
                batch.update(values=values)

            # 10. ADVANTAGE COMPUTATION (on driver process)
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
            )

            # Apply KL penalty to rewards if enabled
            if kl_ctrl_in_reward:
                batch = apply_kl_penalty(batch, kl_coef)

            # 11. CRITIC UPDATE
            if self.use_critic:
                critic_metrics = self._update_critic(batch)

            # 12. ACTOR UPDATE (after critic warmup)
            if global_step >= critic_warmup_steps:
                actor_metrics = self._update_actor(batch)

            # 13. VALIDATION (at intervals)
            if global_step % test_freq == 0:
                val_metrics = self._validate()

            # 14. CHECKPOINT (at intervals)
            if global_step % save_freq == 0:
                self._save_checkpoint()

            # 15. LOGGING
            self._log_metrics(metrics)
```


### Data Flow Through Training Step

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Single Training Step Data Flow                     │
└─────────────────────────────────────────────────────────────────────────┘

1. DATA LOADING
   batch_dict from train_dataloader
   → DataProto.from_single_dict(batch_dict)
   → Repeat n times for n_rollout samples
   Fields: [prompt, data_source, reward_model, extra_info]

                           ↓

2. SEQUENCE GENERATION (async or sync)
   Input: prompts, metadata
   Output: responses, rollout_log_probs
   Workers: async_rollout_manager.generate_sequences()
   Fields: [+ responses, input_ids, attention_mask, position_ids, rollout_log_probs]

                           ↓

3. REWARD COMPUTATION
   └─ Reward Model Scores (if enabled)
      rm_wg.compute_rm_score(batch)
   └─ Token-Level Rewards (rule-based or function)
      reward_fn(batch)
   Fields: [+ rm_scores, rewards]

                           ↓

4. POLICY EVALUATION (compute log probs at current state)
   _compute_old_log_prob(batch)
   → actor_rollout_wg.compute_log_prob()
   Returns: old_log_probs, entropys
   Fields: [+ old_log_probs, entropys]

                           ↓

5. REFERENCE POLICY EVALUATION (if KL penalty needed)
   _compute_ref_log_prob(batch)
   → ref_policy_wg.compute_ref_log_prob() or actor_rollout_wg
   Returns: ref_log_prob
   Fields: [+ ref_log_prob]

                           ↓

6. VALUE ESTIMATION (if critic enabled)
   _compute_values(batch)
   → critic_wg.compute_values()
   Returns: values (V(s))
   Fields: [+ values]

                           ↓

7. ADVANTAGE COMPUTATION (on driver process - CPU)
   compute_advantage(batch)
   ├─ GAE: uses values and rewards
   ├─ GRPO: groups rewards by uid
   └─ Other: custom estimators
   Returns: advantages, returns
   Optionally: applies KL penalty to rewards
   Fields: [+ advantages, returns]

                           ↓

8. CRITIC UPDATE
   _update_critic(batch)
   → critic_wg.update_critic()
   Minimizes: MSE(values - returns)

                           ↓

9. ACTOR UPDATE (after critic warmup)
   _update_actor(batch)
   → actor_rollout_wg.update_actor()
   Maximizes: PPO loss with KL constraint
   Mini-batch training with multiple epochs

                           ↓

10. METRICS & LOGGING
    Compute timing/throughput/variance metrics
    Log to wandb/tensorboard/swanlab
```

## Advantage Estimators

Defined in `verl/trainer/ppo/core_algos.py`:

```python
class AdvantageEstimator(str, Enum):
    GAE = "gae"                    # Generalized Advantage Estimation
    GAE_NO_NORM = "gae_no_norm"    # GAE without normalization
    GRPO = "grpo"                  # Group Relative Policy Optimization
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    RLOO = "rloo"                  # REINFORCE Leave-One-Out
    REMAX = "remax"                # Reward Maximization
    # ... custom estimators via @register_adv_est

@register_adv_est("gae")
def compute_gae_advantage(batch: DataProto, gamma: float, lam: float) -> DataProto:
    """
    GAE: A_t = sum_{l=0}^{inf} (γλ)^l δ_{t+l}
    where δ_t = r_t + γV(s_{t+1}) - V(s_t)
    """
    values = batch.batch["values"]
    rewards = batch.batch["rewards"]

    # Compute TD errors
    deltas = rewards + gamma * values[:, 1:] - values[:, :-1]

    # Compute GAE
    advantages = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        gae = deltas[t] + gamma * lam * gae
        advantages[t] = gae

    # Compute returns
    returns = advantages + values[:, :-1]

    batch.update(advantages=advantages, returns=returns)
    return batch
```

## Checkpointing

```python
def _save_checkpoint(self, step: int):
    """Save model, optimizer, and training state"""
    # Save actor
    self.actor_rollout_wg.save_checkpoint(
        path=f"{checkpoint_dir}/actor_step_{step}",
        ...
    )

    # Save critic (if exists)
    if self.use_critic:
        self.critic_wg.save_checkpoint(
            path=f"{checkpoint_dir}/critic_step_{step}",
            ...
        )

    # Save dataloader state
    self._save_dataloader_state(step)

    # Manage retention (keep only last N checkpoints)
    self._cleanup_old_checkpoints()

def _load_checkpoint(self) -> int:
    """Load from latest checkpoint if resuming"""
    if not self.config.trainer.resume:
        return 0

    latest_checkpoint = self._find_latest_checkpoint()
    if latest_checkpoint is None:
        return 0

    # Load actor
    self.actor_rollout_wg.load_checkpoint(latest_checkpoint)

    # Load critic
    if self.use_critic:
        self.critic_wg.load_checkpoint(latest_checkpoint)

    # Load dataloader state
    step = self._load_dataloader_state(latest_checkpoint)

    return step
```

## Validation

```python
def _validate(self) -> dict:
    """Run validation on validation dataset"""
    all_metrics = []

    for batch_dict in self.val_dataloader:
        batch = DataProto.from_single_dict(batch_dict)

        # Generate sequences
        gen_output = self.actor_rollout_wg.generate_sequences(batch)
        batch = batch.union(gen_output)

        # Compute validation rewards
        rewards = self.val_reward_fn(batch)

        # Collect metrics
        all_metrics.append({
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            ...
        })

    # Aggregate across batches
    val_metrics = aggregate_metrics(all_metrics)
    return val_metrics
```

## Configuration Structure

Refer to `verl/trainer/config/ppo_trainer.yaml` for the configuration structure.

## Extending the Trainer

### Custom Algorithm

```python
class MyCustomTrainer(RayPPOTrainer):
    def fit(self):
        # Override the entire training loop
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                batch = DataProto.from_single_dict(batch_dict)

                # Your custom training logic
                ...

                # Can still use parent's helper methods
                gen_output = self.actor_rollout_wg.generate_sequences(batch)
                ...
```

### Custom Advantage Estimator

```python
from verl.trainer.ppo.core_algos import register_adv_est

@register_adv_est("my_advantage")
def compute_my_advantage(batch: DataProto, **kwargs) -> DataProto:
    """Custom advantage computation"""
    rewards = batch.batch["rewards"]

    # Your advantage logic
    advantages = ...
    returns = ...

    batch.update(advantages=advantages, returns=returns)
    return batch
```

### Custom Policy Loss

```python
from verl.trainer.ppo.core_algos import register_policy_loss

@register_policy_loss("my_loss")
def compute_my_loss(old_log_prob, log_prob, advantages, response_mask, config, **kwargs):
    """Custom policy loss"""
    ratio = torch.exp(log_prob - old_log_prob)

    # Your loss computation
    loss = ...

    metrics = {"my_metric": loss.item()}
    return loss, metrics
```

## Performance Metrics

The trainer computes and logs metrics, including but not limited to:
- Timing: `timing/{generation,reward,update_actor,update_critic}`
- Throughput: `perf/tokens_per_second`, `perf/mfu/actor`
- Training: `train/{policy_loss,value_loss,entropy,kl_divergence}`
- Data: `data/{reward_mean,reward_std,advantage_mean}`

## Key Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| `ray_trainer.py` | 1,741 | Main RayPPOTrainer class |
| `main_ppo.py` | 449 | Entry point with Hydra |
| `core_algos.py` | ~1,900 | Advantage, policy loss, KL penalty |
| `reward.py` | ~300 | Reward computation |
| `metric_utils.py` | ~200 | Metrics aggregation |
| `utils.py` | ~100 | Role enum, helpers |
