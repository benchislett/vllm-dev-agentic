# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Machine

This is a workstation machine equipped with a single NVIDIA RTX PRO 6000 Blackwell GPU which has 96GB of VRAM.

## Project Overview

vLLM is a high-throughput, memory-efficient inference and serving engine for large language models (LLMs). It provides an OpenAI-compatible API server, offline batch inference via the `LLM` class, and supports a wide range of models from HuggingFace.

This repository is a fork of vLLM with additional documentation for Claude Code and some miscellaneous tooling. The main repository (and `gh` default for issues/pulls) is on GitHub at `vllm-project/vllm`. Instructions for claude live on the "main-dev" branch. The branch "main" tracks vLLM upstream main and should always be untouched by commits from this repo. To update main-dev, first fetch and merge upstream/main into main, then rebase main-dev atop it. main-dev will only contain additions and no deletions, with the exception of README.md which is rewritten and should always override any updates to the upstream README.

## Build & Development Commands

Always operate inside of a virtualenv. You may assume that you are already operating with `.venv/` sourced when `claude` is launched, but it helps to add `source .venv/bin/activate` to bash scripts for sanity.

You may assume that `tokens.sh` is sourced when `claude` is launched, but it helps to add `source tokens.sh` to bash scripts for sanity. This provides `HF_TOKEN` for huggingface model access, and may override `HF_HUB` or `HF_HUB_CACHE` to local model cache directories when available. Otherwise, they will go to the default `~/.cache/huggingface/hub/`.

### Installation

vLLM should already be installed in the provided `.venv`, and you can assume it will be present. If you find it is not or need to reinstall:

Prefer precompiled builds whenever possible. When provided, `VLLM_USE_PRECOMPILED` will fetch a compatible wheel and will use the compiled source code (kernels etc.) instead of building the C++/CUDA code from scratch.
When running on newer/older commits, you may run into errors about mismatching arguments or missing kernels, typically with prefixes `torch._C` or `vllm._C`. This indicates the source component is out-of-date, and either needs to be re-installed (preferably still using precompiled wheels) or must be fully rebuilt from scratch. To build from scratch, omit `VLLM_USE_PRECOMPILED`; note you should also add `MAX_JOBS=X` when building this way so that the machine's entire CPU is not eaten up, with X=6 for single-gpu machines and X=32 for multi-gpu servers. When bisecting to previous commits, it may be helpful to create multiple different virtual environments for each of the commits in the bisect line. This is permissible, but should be avoided unless indicated by failures such as the one previously described. If it doesn't crash with an explicit error about missing kernels, it should be fine and multiple virtual envs are not needed.

```bash
VLLM_USE_PRECOMPILED=1 pip install -v -e .
pip install -r requirements/dev.txt # Dev dependencies (lint + test)
```

### Linting & Formatting

Running linting manually is not needed, as `pre-commit install` should be already enabled by the developer, so the hooks will run on `git commit`. If asked to run the linting, use one of these options:

```bash
pre-commit run --all-files              # Run all linters. Use this by default, since all linters are required for pre-commit anyways.
pre-commit run ruff-check --all-files   # Ruff lint only
pre-commit run ruff-format --all-files  # Ruff format only
pre-commit run mypy-local --all-files   # mypy type checking
pre-commit run clang-format --all-files # C++/CUDA formatting
```

The project uses **ruff** for Python linting/formatting and **clang-format** for C++/CUDA. Ruff rules: E, F, UP, B, ISC, SIM, I (isort), G.

### Testing

Tests take a while to run, so it is not recommended to run the full test suite locally. Instead, when working on a feature or bugfix, identify and/or create a few targeted test cases for your problem and run those specifically using `pytest -k`.

Always use `-s` with pytest so that the vLLM server logs are shown (useful for debugging and insights).

Not all tests run in per-PR CI. The full CI pipeline is described in `.buildkite/test-pipeline.yaml`.

### File Header Requirement
All new Python files must include SPDX headers:
```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```

### Commits
Commits must be signed with DCO sign-off. Use `git commit -s` to ensure proper sign-off.

Commit messages should have a short and brief header under 80 characters, all lowercase.
Then, if Claude is writing the commit, two line breaks and a short explanatory paragraph about the changes. 

When pushing in force mode, always prefer `--force-with-lease`.

## Architecture

Most of the time, `vllm/v1/worker/gpu_model_runner.py` will be your starting point when debugging. Breakpoints and log statements are very useful in `execute_model()` before and after the model forward pass, to inspect the state etc.

An upcoming refactor of the model runner called ModelRunnerV2 is located in `vllm/v1/worker/gpu`. Ignore it unless specifically asked.

### V1 Engine (current default)

The codebase has moved to the "V1" architecture under `vllm/v1/`. The legacy engine in `vllm/engine/` (LLMEngine, AsyncLLMEngine) still exists but is essentially unused.

**Request flow through V1:**
1. **Entrypoints** (`vllm/entrypoints/`): API servers (OpenAI-compatible at `openai/api_server.py`), CLI (`cli/`), and offline `LLM` class (`llm.py`). The CLI entry point is `vllm.entrypoints.cli.main:main`.
2. **AsyncLLM** (`vllm/v1/engine/async_llm.py`): The main async engine frontend. Processes inputs, manages request lifecycle, and communicates with EngineCore.
3. **InputProcessor** (`vllm/v1/engine/input_processor.py`): Tokenizes/preprocesses requests before sending to EngineCore.
4. **EngineCoreClient** (`vllm/v1/engine/core_client.py`): Communicates with EngineCore over ZMQ sockets. EngineCore runs in a separate process.
5. **EngineCore** (`vllm/v1/engine/core.py`): The core engine loop. Runs the scheduler, dispatches work to the executor, and collects outputs.
6. **Scheduler** (`vllm/v1/core/sched/scheduler.py`): Decides which requests to process in each iteration, manages KV cache allocation via `KVCacheManager` (`vllm/v1/core/kv_cache_manager.py`).
7. **Executor** (`vllm/v1/executor/`): Manages worker processes. Options: `uniproc_executor.py` (single GPU), `multiproc_executor.py` (multi-GPU), `ray_executor.py` (Ray distributed option for multi-GPU, opt-in).
8. **Worker/ModelRunner** (`vllm/v1/worker/`): `gpu_worker.py` manages a single GPU. `gpu_model_runner.py` handles the actual model forward pass, KV cache, CUDA graphs, and sampling.
9. **OutputProcessor** (`vllm/v1/engine/output_processor.py`): Converts engine outputs back to user-facing `RequestOutput` objects with detokenization.

### Model System

- **Model registry** (`vllm/model_executor/models/registry.py`): Maps HuggingFace architecture names to vLLM model implementations. When adding a new model, also update `tests/models/registry.py` with example models.
- **Model implementations** (`vllm/model_executor/models/`): One file per model family (e.g., `llama.py`, `qwen2.py`). Models use custom layers from `vllm/model_executor/layers/`.
- **Custom layers** (`vllm/model_executor/layers/`): Shared building blocks — `linear.py` (parallelized linears), `vocab_parallel_embedding.py`, `rotary_embedding/`, `layernorm.py`, `fused_moe/` (Mixture of Experts), `attention/` (attention backends), `quantization/` (quantization methods like GPTQ, AWQ, FP8).
- **Model loading** (`vllm/model_executor/model_loader/`): Handles weight loading from various formats (HuggingFace, GGUF, sharded, tensorizer).

### Key Subsystems

- **Attention backends** (`vllm/v1/attention/backends/`): FlashAttention, FlashInfer, etc.
- **Sampling** (`vllm/v1/sample/`): Token sampling with logits processing.
- **Speculative decoding** (`vllm/v1/spec_decode/`): Draft model, EAGLE, Medusa, n-gram proposers. Primary backend with most support is EAGLE, in `vllm/v1/spec_decode/eagle.py`
- **Structured output** (`vllm/v1/structured_output/`): Guided generation backends (xgrammar, outlines, guidance, lm-format-enforcer).
- **Distributed** (`vllm/distributed/`): Tensor/pipeline/expert parallelism, KV cache transfer for disaggregated prefill.
- **Platforms** (`vllm/platforms/`): Hardware abstraction (CUDA, ROCm, CPU, TPU, XPU). Assume CUDA GPU is used by default.
- **Config** (`vllm/config/`): All configuration dataclasses. `VllmConfig` is the top-level config aggregating all sub-configs, in `vllm/config/vllm.py`
- **Multimodal** (`vllm/multimodal/`): Processing for images, audio, video inputs.
- **C++/CUDA kernels** (`csrc/`): Custom CUDA kernels for attention, quantization, cache management, MoE, etc.

### Communication Pattern

The V1 engine uses a **multi-process architecture** with ZMQ for IPC:
- The API server (frontend) runs in the main process
- `EngineCore` runs in a separate process, communicating via ZMQ sockets
- Workers run in further sub-processes (one per GPU)
- Data is serialized with `msgspec` for performance

## Serving Arguments

vLLM has a lot of serving arguments. Many are useful for debugging, benchmarking, profiling, and development.

### Debugging Tips

- `--trust-remote-code`: sometimes needs to be set for some models. For any standard model (Meta, DeepSeek, NVIDIA, OpenAI, Qwen, etc.) this is safe and can be added without concern. Otherwise, omit by default and prompt the user if the vLLM server requests that it be provided.
- `--load-format dummy`: set this to skip loading the model weights, leaving them as uninitialized tensors. Useful when debugging failures that do not depend on model outputs (fatal crashes, missing kernels, startup failures etc.) especially when using large models since startup can take several minutes.
- `--enforce-eager` this mode disables CUDA graphs and torch.compile optimizations/fusions. In addition to greatly speeding up start time, this can make runs a bit more deterministic and exclude/identify a class of errors that occur due to setup mismatch between recording and replay of CUDA graphs. Not suitable for benchmarking or profiling since it will perform much worse.
- `--attention-backend` can be set to `flash_attn` or `triton_attn` when debugging issues stemming from an attention backend, to identify if the problem is misconfiguration in the model runner or a problem with a specific attention backend. On Hopper (H100/H200) GPUs, FlashAttention will be the default most of the time. On Blackwell, `flashinfer` (specifically with the TRTLLM kernels) is the default and most performant backend.
- `--no-enable-prefix-caching` and `--no-enable-chunked-prefill` can be used to further isolate issues that indicate a problem with these features.

### Benchmarking Tips

- `--max-num-batched-tokens`: a bound on the number of tokens in a batch. Must be at least as large as `max_num_seqs`. When this is smaller than the context length, prefills will automatically be "chunked". This can be useful for interactivity ("goodput" etc.) but is often detrimental for performance. Set to at least 8192 when benchmarking, or 32768 when doing high-concurrency or longer-context benchmarks. For extremely long context, you can set this to the next power of two larger than your max ISL. Avoid going beyond 128K as some kernels may not support such large widths, although it should work in theory.
- `--no-enable-prefix-caching`: when benchmarking performance, prefix cache hits often introduce variance within and between runs. Disable this feature whenever you benchmark or profile a scenario that does not explicitly try to measure prefix caching performance.
- `--gpu-memory-utilization`: defaults to 0.9, meaning the server will (try) use 90% of the available VRAM, allocating KV cache accordingly. This is usually appropriate, but when trying to squeeze an extra few % of throughput for an important use-case you can try bumping this to 0.95 or somewhere in between. It can be lowered from the default if the system just barely runs out of memory. If there are still persistent OOM issues, you should check if another process/vllm server is running on the maching.

#### CUDA Graphs:

vLLM records and replays CUDA graphs for performance, which are enabled by default. There are two kinds of graph: PIECEWISE graphs, where the attention is omitted and we record a graph for the rest of the layer and replay it K times for each forward pass, and FULL graphs, where we record the entire forward pass.
If supported by the attention backend, FULL_AND_PIECEWISE is the default mode, meaning we record FULL graphs for DECODE where each request has the same number of query tokens, and PIECEWISE graphs for PREFILL where we have one (or more) requests with a variable number of tokens. If not supported by attention, one set of PIECEWISE graphs will be recorded and used for both prefill and decode. Mixed batches of prefill + decode will run in PIECEWISE mode, with the total token count being the number of prefill tokens + number of decode query tokens.

By default, vLLM will record up to a maximum batch size of 512, meaning 512 decode query tokens or 512 tokens in the prefill. This threshold can be increased be setting `--max-cudagraph-capture-size N`. Recording more graphs consumes a nontrivial amount of VRAM, ~a few GB. Generally this value can be increased to 2048 without concern if the GPU has more than 32GB of VRAM, and can go even higher (but will have diminishing returns of course). If affordable, it should be set to >= the max ISL (up to 4096, don't go higher than this). Note that when using speculative decoding, there are `(1 + num_speculative_tokens)` query tokens for each request, so this value should be set much higher: for example if max_concurrency == 128 and num_speculative_tokens == 7, then we need to set it to at least (128 * (1 + 7)) == `1024`.

#### Parallelism:

Tensor Parallelism (TP) and Expert Parallelism (EP) are the most commonly used methods for using multiple GPUs. Prefer TP in almost all cases, except for higher-concurrency runs of large MoE models such as DeepSeek models.

To use TP for multi-gpu (TP1 is default) set `--tensor-parallel-size N`. To instead use EP, also set `--enable-expert-parallel`. For DeepSeek MLA models, we can use data-parallel attention instead of TP, with `--data-parallel-size` instead of TP size.

#### Speculative Decoding:

When using speculative decoding, the prompt contents matter and benchmarking with random data is not sufficient for most cases. If you are only required to analyze the engine iteration time or collect a profile, then the contents don't typically matter since the batch sizes are consistent no matter how many tokens are accepted or rejected. But for TPS and TPOT calculations, real data must be used (such as MTBench, HumanEval or SPEED-Bench).

## Coding Style

Use minimal comments, especially when the functionality is already clear from variable, function, and parameter names.
Follow existing style and naming conventions as much as possible. Keep style consistent with the local file being modified.

Your primary goal is to accomplish the feature in as few lines of code as possible. Do not 'golf', but be minimal.
Fewer lines of code means less to review and maintain, and a much smoother process overall. Avoid overly verbose comments, big refactors, and modifying unrelated components.
