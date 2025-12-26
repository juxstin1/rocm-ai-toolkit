# ROCm Diagnostics

Advanced diagnostic tools for validating AMD GPU behavior at the
hardware/kernel level.

## How it works (3 parts)

The spy (C++ kernel):
`clock_probe.cpp` runs on the GPU and spins in a tight loop, counting hardware
cycles with `__clock64()`.

The stopwatch (Python):
`clock_probe.py` launches the kernel, measures wall time on the CPU, and reads
back the cycle count.

The math:
cycles / seconds = MHz. This is derived from the GPU's own counter rather than
driver-reported clocks, which makes it useful for spotting throttle/idle
behavior while workloads run.

## 1. Clock Probe (`clock_probe.py`)

Directly measures the true GPU shader clock frequency during execution.
Unlike `rocm-smi` or Task Manager which report requested or target clocks, this
tool runs a HIP kernel that counts actual hardware cycles (`__clock64()`) over
a measured time window.

Use this to detect:

* Thermal throttling (real clock < target clock).
* Power limit throttling.
* Phantom execution (driver reports success, but clocks never boost).

### Prerequisites

* AMD HIP SDK installed (e.g., ROCm 5.7+ or 6.x).
* `hipcc` available (part of the SDK).

### Usage (Windows)

**Option A: One-Click (Recommended)**
Double-click `run_diagnostics.bat`. This will attempt to auto-locate your ROCm
install, build the kernel, and run the test.

**Option B: Manual**
1. Set your HIPCC path: `set HIPCC="C:\Program Files\AMD\ROCm\6.4\bin\hipcc.exe"`
2. Build: `build.bat`
3. Run: `python clock_probe.py`

### Interpretation

* PASS: "Measured Clk" aligns with your GPU's expected game/boost clock
  (e.g., ~2300-2800 MHz for RDNA3).
* FAIL: 0 MHz or extreme outliers indicate the kernel is not executing correctly
  on the hardware.

### Should you tune the loops?

Leave the default loop count high (e.g., 500,000,000).

For a diagnostic tool like this, duration is accuracy.

Short runs (<100ms): GPU boost latency dominates. The card might stay in idle
state (low voltage/clock) for half the test, skewing the average MHz down.

Long runs (>500ms): this forces the power management firmware to wake up, apply
full voltage, and hit the sustained boost clock. This is what you actually want
to measure.
