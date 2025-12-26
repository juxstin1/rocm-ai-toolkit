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

## Diagnostics

The `diagnostics/` folder includes a HIP clock probe that compiles a small
kernel and measures GPU cycles over wall time.

```powershell
cd diagnostics
build.bat
python clock_probe.py
```

Set `HIPCC` to the full path of `hipcc` if it is not on PATH.

## Overrides

- `AIKIT_PYTHON` to point at a specific Python executable
- `AIKIT_GPU_INDEX` to set `HIP_VISIBLE_DEVICES` when using the .bat launchers

## Contributing

Keep scope user-facing and avoid proof claims. Verification should be delegated
to rocm-truth. Issues and small PRs are welcome.
