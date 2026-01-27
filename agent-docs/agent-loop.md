# Agent Loop - Multi-Turn & Tool Use

## Overview

The agent loop enables multi-turn conversations with tool use for agentic RL. State machine-based approach managing LLM↔tools↔environment interactions.

**Key Capabilities:**
- Multi-turn conversation with turn tracking
- Tool calling with parallel execution
- Environment interaction (user simulation, task environments)
- Async execution with cancellation/resume support
- Multi-modal support (images, videos)

**Why Server-Based Async Rollout?**

Agents interact with environments through tool calls. To avoid GPU idling while waiting for tool responses, asyncio-based coroutines execute rollout requests asynchronously. Server/client separation enables:
1. Load balancing across GPUs, reducing long-tail request impact
2. Isolation of agent-specific features (tracing) from inference engine

---

## System Architecture

```
┌────────────┐
│ PPOTrainer │
└─────┬──────┘
      │ generate_sequences
      ▼
┌─────────────────┐       _initialize_llm_servers      ┌─────────────────────────────────────┐
│  AgentLoop      │ ──────────────────────────────────►│  AsyncServer Pool (SGLang/vLLM)     │
│  Manager        │                                    │  ┌─────────┐ ┌─────────┐ ┌─────────┐│
└─────┬───────────┘                                    │  │ Server0 │ │ Server1 │ │ Server2 ││
      │                                                │  └─────────┘ └─────────┘ └─────────┘│
      │ spawns                                         └──────────────────▲──────────────────┘
      ▼                                                                   │
┌─────────────────────────────────────────────────────────────────────────┼───────────────────┐
│  AgentLoopWorkers                                                       │                   │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐       │                   │
│  │ AgentLoopWorker             │  │ AgentLoopWorker             │       │                   │
│  │ ┌───────┬───────┬─────────┐ │  │ ┌───────┬───────┬─────────┐ │       │                   │
│  │ │prompt0│prompt1│...promptN│ │  │ │prompt0│prompt1│...promptN│ │       │                   │
│  │ └───┬───┴───────┴─────────┘ │  │ └───┬───┴───────┴─────────┘ │       │                   │
│  │     │                       │  │     │                       │       │                   │
│  │     ▼                       │  │     ▼                       │       │                   │
│  │ ┌─────────┐                 │  │ ┌─────────┐                 │       │                   │
│  │ │AgentLoop│◄──┐ (multi-turn)│  │ │AgentLoop│◄──┐ (multi-turn)│       │                   │
│  │ └────┬────┘───┘             │  │ └────┬────┘───┘             │       │                   │
│  │      │                      │  │      │                      │       │                   │
│  │      ▼                      │  │      ▼                      │       │                   │
│  │ ┌──────────────────┐        │  │ ┌──────────────────┐        │       │                   │
│  │ │AsyncLLMServer    │        │  │ │AsyncLLMServer    │        │       │                   │
│  │ │Manager           │────────┼──┼─│Manager           │────────┼───────┘                   │
│  │ └──────────────────┘        │  │ └──────────────────┘        │  generate(token_ids)      │
│  └─────────────────────────────┘  └─────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

**Flow:**
1. **PPOTrainer** calls `generate_sequences` on AgentLoopManager
2. **AgentLoopManager** initializes AsyncServer pool (SGLang/vLLM) and spawns AgentLoopWorkers
3. Each **AgentLoopWorker** handles batch of prompts, runs AgentLoop per prompt
4. **AgentLoop** executes multi-turn state machine (self-loop for turns)
5. **AsyncLLMServerManager** load-balances `generate(token_ids)` calls to server pool

---

## State Machine

```
                    ┌─────────────────────┐
                    │    AgentData        │
                    │  (State Container)  │
                    └──────────┬──────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       State Machine                                   │
│                                                                       │
│  PENDING ───► GENERATING ───► PROCESSING_TOOLS ───┐                  │
│                    │                    │          │                  │
│                    │         ┌──────────┘          │                  │
│                    ▼         ▼                     │                  │
│              INTERACTING ◄───┘                     │                  │
│                    │                               │                  │
│                    └───────► TERMINATED ◄──────────┘                  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Agent States

| State | Description |
|-------|-------------|
| `PENDING` | Initialize prompt, apply chat template |
| `GENERATING` | LLM generation, extract tool calls |
| `PROCESSING_TOOLS` | Execute tool calls (parallel) |
| `INTERACTING` | Get environment/user feedback |
| `TERMINATED` | End of sequence |

**Transitions:**
- `GENERATING` → `PROCESSING_TOOLS`: if tool calls detected
- `GENERATING` → `INTERACTING`: if interaction present and no tools
- `GENERATING` → `TERMINATED`: if done (max turns or stop token)
- `PROCESSING_TOOLS` → `GENERATING`: continue after tools
- `INTERACTING` → `GENERATING` or `TERMINATED`: based on environment response

---

## Core Components

### AgentData (State Container)

Holds all state for a single trajectory:
- **Conversation**: `messages`, `image_data`, `video_data`
- **Tokens**: `prompt_ids`, `response_ids`, `response_mask`, `response_logprobs`
- **Counters**: `user_turns`, `assistant_turns`
- **Rewards**: `turn_scores`, `tool_rewards`
- **Environment**: `interaction`, `tools`, `tool_calls`

### Response Mask Semantics

```
Tokens:  [prompt] [LLM] [tool] [LLM] [env] [LLM]
Mask:    [  0   ] [ 1 ] [ 0  ] [ 1 ] [ 0 ] [ 1 ]

0 = external source (prompt, tool, environment)
1 = LLM-generated (training signal)
```

This enables computing loss only on LLM tokens and proper credit assignment.

---

## Tool System

### BaseTool Interface

| Method | Purpose |
|--------|---------|
| `create(instance_id)` | Initialize tool instance per trajectory |
| `execute(instance_id, parameters)` | Execute tool call → `(response, reward, extra)` |
| `calc_reward(instance_id)` | Final reward calculation |
| `release(instance_id)` | Cleanup resources |

### ToolResponse

Contains: `text`, `image` (optional), `video` (optional)

### Tool Parsing

Extracts tool calls from LLM output. Formats:
- **Hermes**: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
- **GPT-OSS**: Special tokens `<function>...</function>`

---

## Interaction System

### BaseInteraction Interface

| Method | Purpose |
|--------|---------|
| `start_interaction(instance_id)` | Initialize session → initial prompt |
| `generate_response(instance_id, messages)` | Environment response → `(terminate, content, score, extra)` |
| `calculate_score(instance_id)` | Final score |
| `finalize_interaction(instance_id)` | Cleanup |

---

## Async Execution

### AgentLoopWorker

Manages async execution of multiple agent loops in parallel via `asyncio.gather()`.

### AsyncLLMServerManager

- Load balances across multiple LLM servers
- Sticky routing for prefix caching (same `request_id` → same server)
- Least-requests load balancing

**Why Token-Based API (not Chat Completion)?**

Token↔text conversion can be irreversible (e.g., `<think>` token differs when converted from text vs generated by LLM). Training requires exact LLM-generated tokens for accurate advantage computation. Token-based API lets client maintain relationship between tool-generated text and LLM tokens.

### AsyncServer (Inference Engine Adaptation)

| Engine | Implementation |
|--------|----------------|
| **SGLang** | Uses `async_generate` on first GPU of each TP group; AsyncServer calls via Ray actor |
| **vLLM** | Uses `generate` interface; communicates with TP group via ZMQ; direct call in AsyncServer |

### Cancellation & Resume (Fully Async)

`AsyncPartialToolAgentLoop` supports:
- Cancellation via shared `asyncio.Event`
- Resume from cancelled state (preserves `AgentData` in output)
- Partial rollout for interrupted sessions

---

## Configuration

### Required Config Options

To enable agent loop:
- `data.return_raw_chat=True`
- `actor_rollout_ref.rollout.mode=async`

### Dataset Requirements

For tool agent loop, add `agent_name` field to dataset. During rollout, selects `tool_agent_loop` or `single_turn_agent` (default) based on this field.

### Multi-Turn Config

| Setting | Description |
|---------|-------------|
| `max_user_turns` | Max environment/user turns |
| `max_assistant_turns` | Max LLM generation turns |
| `max_parallel_calls` | Parallel tool execution limit |
| `max_tool_response_length` | Max tokens per tool response |
| `format` | Tool call format (`hermes`, `gpt-oss`) |

---

## Output Structure

`AgentLoopOutput` contains:
- `prompt_ids`, `response_ids`, `response_mask`, `response_logprobs`
- `reward_score`, `num_turns`
- `metrics`: `generate_sequences` time, `tool_calls` time, `num_preempted`
- `extra_fields`: `turn_scores`, `tool_rewards`, `is_cancel`, `state`

---

## Where to Look

| Topic | Location |
|-------|----------|
| Main AgentLoopWorker | `verl/experimental/agent_loop/agent_loop.py` |
| ToolAgentLoop state machine | `verl/experimental/agent_loop/tool_agent_loop.py` |
| Tool parser | `verl/experimental/agent_loop/tool_parser.py` |
| Fully async worker | `verl/experimental/fully_async_policy/agent_loop/agent_loop.py` |
| Partial/cancellable loop | `verl/experimental/fully_async_policy/agent_loop/partial_tool_agent_loop.py` |
| BaseTool interface | `verl/tools/base_tool.py` |
| Tool schemas | `verl/tools/schemas.py` |
| Tool registry | `verl/tools/utils/tool_registry.py` |
| BaseInteraction interface | `verl/interactions/base.py` |
| Interaction registry | `verl/interactions/utils/interaction_registry.py` |
| Example tools | `verl/tools/gsm8k_tool.py`, `verl/tools/search_tool.py` |
| Example interactions | `verl/interactions/gsm8k_interaction.py` |
| Multi-turn examples | `examples/sglang_multiturn/` |
| GSM8K tool agent preprocess | `examples/data_preprocess/gsm8k_tool_agent_loop.py` |
| Agentic RL docs | `docs/start/agentic_rl.rst` |
| Rollout trace docs | `docs/advance/rollout_trace` |

---

## Debugging & Tracing

MLflow-based tracing for debugging rollout details:
```bash
pip install mlflow
# Enable in training script, then:
mlflow ui -h 0.0.0.0 -p 5000 --backend-store-uri sqlite:////tmp/mlruns.db
```

Note: "Failed to decode tool call" console errors during training are normal - model sometimes generates malformed tool tags.

See: `docs/advance/rollout_trace`

---

## Nuances & Gotchas

1. **Response mask is critical** for RL - only LLM tokens (mask=1) get gradients
2. **Tool execution is parallel** up to `max_parallel_calls`
3. **Sticky routing** keeps same request on same server for prefix cache hits
4. **Cancellation preserves state** - can resume from `extra_fields` in output
5. **Turn limits** are separate for user and assistant turns
6. **Tool rewards vs turn scores** - tools give immediate rewards, interactions give per-turn scores
7. **Token-based API** - don't use text-based chat completion; token fidelity matters for training
8. **agent_name field** - required in dataset for tool agent loop selection
