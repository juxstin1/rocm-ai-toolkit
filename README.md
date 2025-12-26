# rocm-ai-toolkit

Command-line AI utilities for ROCm environments.

This repo provides user-facing CLI tools built around the `aikit` package.
Verification is delegated to rocm-truth; this repo does not claim proof of GPU
execution.

Verification repo:

- https://github.com/juxstin1/rocm-truth

## Quick start

```powershell
pip install -e .
ai help
```

## Docs

- `docs/SETUP_ROCM.md`

## Overrides

- `AIKIT_PYTHON` to point at a specific Python executable
- `AIKIT_GPU_INDEX` to set `HIP_VISIBLE_DEVICES` when using the .bat launchers
