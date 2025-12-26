# ROCm Setup Guide for Windows (RX 9070 XT Verified)

This guide documents the exact configuration used to run `aikit` and PyTorch on
an AMD Radeon RX 9070 XT under Windows 11.

Official documentation can be fragmented; this specific combination of versions
has been verified to work for Stable Diffusion, Semantic Search, and
Segmentation.

## The Verified Stack

| Component | Version | Notes |
| --- | --- | --- |
| GPU | Radeon RX 9070 XT | RDNA 4 Architecture |
| OS | Windows 10/11 | Verified on Windows 11 |
| Python | 3.12 | Required for these nightly wheels |
| HIP SDK | 6.4 | Must match or exceed PyTorch ROCm version |
| PyTorch | 2.6.0 (Nightly) | Built with ROCm 6.4.4 support |

## Step-by-Step Installation

### 1. System prerequisites

Before installing Python libraries, you must have the low-level drivers
installed.

1. Visual Studio 2022 Community: install the "Desktop development with C++"
   workload. The HIP SDK requires the MSVC compiler chain.
2. AMD Drivers: ensure you are on the latest Adrenalin drivers for the 9070 XT.
3. AMD HIP SDK 6.4: download and install from https://rocm.docs.amd.com/.
   - Verification: open PowerShell and run `hipconfig --version`. It should
     return `6.4.xxxxx`.

### 2. Python environment

Do not use the system Python. Create a dedicated virtual environment to avoid
DLL conflicts.

```powershell
# Install Python 3.12 if not present
python --version  # Must say 3.12.x

# Create the environment
python -m venv venv

# Activate it
.\venv\Scripts\activate
```

### 3. Installing PyTorch (the tricky part)

Standard `pip install torch` will install the CUDA or CPU version. You must
force the ROCm nightly build.

```powershell
# Uninstall any existing torch versions to be safe
pip uninstall torch torchvision torchaudio

# Install PyTorch Nightly with ROCm 6.2/6.4 compatibility
# Note: The index URL often references the major version (e.g., rocm6.2) even
# for 6.4 builds.
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.2
```

Note: As of verification (Dec 2025), the nightly build
`torch-2.6.0.dev...` provides the ROCm 6.4.4 runtime support required for RDNA
4 cards.

### 4. Installing rocm-ai-toolkit

Once the foundation is laid, install the toolkit in editable mode.

```powershell
cd path\to\rocm-ai-toolkit
pip install -e .
```

## Verification

Do not assume it works just because it installed. Windows can silently fall
back to CPU.

### 1. Verify execution with rocm-truth

Use rocm-truth to prove execution. This repo does not claim proof.

```powershell
git clone https://github.com/juxstin1/rocm-truth
cd rocm-truth
python rocm-truth.py
```

Look for `execution_verified: true` and `status: PASS`.

Watch out for `rocm_stack_verified: false` (this is normal on Windows, since
OS-level verification tools are missing).

### 2. Run a real workload

Use the `aikit` CLI to force a heavy GPU load:

```powershell
# This uses the sentence-transformers library on the GPU
ai find "test verification"
```

## Common pitfalls

- `hipErrorNoBinaryForGpu`: usually means your PyTorch version was built for an
  older ROCm version that does not support the 9070 XT's architecture. Ensure
  you are on the nightly build.
- Silent CPU fallback: if rocm-truth reports FAIL or `execution_verified: false`,
  verify your environment variables. Ensure `HIP_PATH` is set to
  `C:\Program Files\AMD\ROCm\6.4\`.
