# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mini-SGLang is a lightweight (~5,000 lines) high-performance LLM inference framework. It's a compact implementation of SGLang providing an OpenAI-compatible API server with advanced optimizations (Radix Cache, Chunked Prefill, Overlap Scheduling, Tensor Parallelism).

**Platform**: Linux only (x86_64/aarch64). Windows/macOS require WSL2 or Docker.

## Development Commands

```bash
# Setup
uv venv --python=3.12
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run server
python -m minisgl --model "Qwen/Qwen3-0.6B"
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell  # Interactive mode
python -m minisgl --model "meta-llama/Llama-3.1-70B-Instruct" --tp 4 --port 30000

# Code quality
black python/                    # Format code
ruff check --fix python/         # Lint with auto-fix
mypy python/minisgl/             # Type checking (strict mode enforced)
pre-commit run --all-files       # Run all pre-commit hooks

# Testing
pytest tests/                    # Run all tests with coverage
pytest tests/core/               # Scheduler tests
pytest tests/kernel/             # Kernel unit tests
pytest tests/quantization/       # AWQ quantization tests
pytest tests/path/test_file.py::test_name  # Run single test
```

## Architecture

Mini-SGLang uses a **multi-process architecture** with ZeroMQ for control messages and NCCL for GPU tensor communication:

```
User → API Server → Tokenizer → Scheduler (Rank 0) → Broadcast to TP Ranks → Engines → Detokenizer → User
```

**Key Processes:**
- **API Server** (`minisgl.server.api_server`): FastAPI server, OpenAI-compatible endpoints
- **Scheduler** (`minisgl.scheduler`): One per GPU, manages computation and batching
- **Engine** (`minisgl.engine`): Per-GPU inference executor, manages model/KV cache/CUDA graphs
- **Tokenizer/Detokenizer** (`minisgl.tokenizer`): Text ↔ token conversion workers

## Code Organization (`python/minisgl/`)

| Module | Purpose |
|--------|---------|
| `core.py` | Core dataclasses: `Req`, `Batch`, `Context`, `SamplingParams` |
| `models/` | Model implementations (Llama, Qwen3) with HF weight loading |
| `layers/` | TP-aware building blocks (linear, embedding, RoPE, etc.) |
| `attention/` | Attention backends (FlashAttention, FlashInfer) |
| `kvcache/` | KV cache management (Radix and Naive strategies) |
| `scheduler/` | Core scheduling logic, request batching |
| `engine/` | Per-GPU inference with CUDA graph support |
| `distributed/` | Tensor parallelism, NCCL communication |
| `kernel/` | Custom CUDA kernels via TVM-FFI |
| `server/` | CLI args, process launching, FastAPI server |
| `message/` | ZMQ message definitions with auto-serialization |

## Code Style

- **Line length**: 100 characters
- **Python version**: 3.10+ (3.12 recommended)
- **Type annotations**: Required (mypy strict mode)
- **Formatters**: black, ruff, clang-format (for CUDA)

## Quantization Support

| Method | GPU Requirement | Description |
|--------|-----------------|-------------|
| `awq` | SM75+ (Turing) | AWQ using Triton kernels |
| `awq_marlin` | SM80+ (Ampere) | AWQ using Marlin CUDA kernels (faster) |

AWQ models are auto-upgraded to AWQ Marlin on compatible GPUs. Quantization config is loaded from `quant_config.json` or `quantize_config.json`.

**Key files:**
- `layers/quantization/awq.py`: AWQ Triton implementation
- `layers/quantization/awq_marlin.py`: AWQ Marlin implementation
- `layers/quantization/marlin_utils.py`: Marlin kernel utilities

## Key Dependencies

- `transformers>=4.56.0,<=4.57.3` (strict version range)
- `flashinfer-python>=0.5.3` (high-performance attention)
- `sgl_kernel>=0.3.17.post1` (custom CUDA kernels including Marlin)
- `apache-tvm-ffi>=0.1.4` (kernel JIT compilation)

## Environment Variables

- `MINISGL_DISABLE_OVERLAP_SCHEDULING=1`: Disable CPU-GPU overlap (for ablation studies)
- `MINISGL_SHELL_TEMPERATURE`, `MINISGL_SHELL_TOP_K`, `MINISGL_SHELL_TOP_P`: Shell sampling params
