# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

vllm-ascend is a hardware plugin enabling [vLLM](https://github.com/vllm-project/vllm) to run on Huawei Ascend NPUs. It implements vLLM's pluggable hardware interface, registering via `entry_points` in `setup.py`. The main package is `vllm_ascend/`.

## Common Commands

```bash
# Install in dev mode
pip install -e .[dev]

# Run unit tests
pytest -sv tests/ut/

# Run a specific test
pytest -sv tests/ut/ops/test_prepare_finalize.py::test_prepare_inputs

# Run E2E tests (requires NPU hardware)
pytest -sv tests/e2e/singlecard/

# Lint and format
bash format.sh          # local pre-commit hooks
bash format.sh ci       # CI mode (includes markdownlint)
ruff check vllm_ascend/
ruff format vllm_ascend/

# Type checking
mypy vllm_ascend/

# Build (requires SOC_VERSION env var or NPU hardware)
pip install -e .
```

## Architecture

### Plugin Registration

The plugin registers itself via `setup.py` entry points:
- `vllm.platform_plugins` -> `NPUPlatform` (in `vllm_ascend/platform.py`)
- `vllm.general_plugins` -> connectors, model loaders, profiling, models

### Key Directories

- `vllm_ascend/patch/` — Monkey-patches for upstream vLLM (platform-level and worker-level)
- `vllm_ascend/worker/` — NPU model runners (v1, v2, 310P variants)
- `vllm_ascend/attention/` — NPU attention implementations
- `vllm_ascend/ops/` — Custom NPU operators
- `vllm_ascend/envs.py` — Centralized environment variable definitions
- `vllm_ascend/ascend_config.py` — Ascend-specific configuration
- `csrc/` — C/C++ custom kernel sources (built via CMake)
- `tests/ut/` — Unit tests (mirrors source structure)
- `tests/e2e/` — End-to-end tests (singlecard/, multicard/, nightly/, weekly/)

### Extension Pattern

This project never adds model files directly. All model-specific functionality uses:

1. **Patching** (`vllm_ascend/patch/`): Monkey-patch upstream vLLM classes
2. **Inheritance**: `NPUModelRunner(GPUModelRunner)`, `AscendSampler`, etc.
3. **Custom operators**: NPU-specific ops in `vllm_ascend/ops/`

## Critical Conventions

### Environment Variables

All env vars must be defined in `vllm_ascend/envs.py` using the `env_variables` dict. Use `VLLM_ASCEND_*` naming. Reference via `from vllm_ascend import envs`. Never hardcode env var names elsewhere.

### NPU Performance Rules

- **Never** use `tensor.item()` in hot paths — causes CPU-NPU sync that blocks `AsyncScheduler`
- Avoid CPU-NPU memory transfers in hot paths
- Prefer in-place operations (`x.add_()`, `x.mul_()`) where safe
- Keep values on device; use `torch.argmax`, `torch.sum` instead of pulling to CPU

### Code Style

- Line length: 120 (configured in `pyproject.toml` for ruff)
- Imports at top of file (exceptions: circular imports, lazy loading, `TYPE_CHECKING`)
- No magic numbers — use named constants
- Classes: `PascalCase`, functions: `snake_case`, constants: `ALL_UPPER_CASE`

### Commits and PRs

- Conventional Commits format with mandatory sign-off: `git commit -s`
- PR title format: `[Type][Module] Description` (e.g., `[BugFix][Worker] Fix padding`)
- Valid types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `chore`
- PRs must come from fork repositories

## Detailed Guidelines

See [AGENTS.md](AGENTS.md) for comprehensive development guidelines including review checklists, patch patterns, and contributor workflow.
