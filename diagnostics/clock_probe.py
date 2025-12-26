import ctypes
import time
from pathlib import Path


def load_probe():
    dll_path = Path(__file__).with_name("clock_probe.dll")
    if not dll_path.exists():
        raise FileNotFoundError("clock_probe.dll not found. Run build.bat first.")

    lib = ctypes.CDLL(str(dll_path.resolve()))
    lib.measure_gpu_cycles.restype = ctypes.c_longlong
    lib.measure_gpu_cycles.argtypes = [ctypes.c_int]
    return lib


def measure_true_clock(loops: int) -> float:
    lib = load_probe()
    print(f"Running cycle probe ({loops} iterations)...")

    start_time = time.perf_counter()
    total_cycles = lib.measure_gpu_cycles(loops)
    end_time = time.perf_counter()

    elapsed_sec = end_time - start_time
    real_hz = total_cycles / max(elapsed_sec, 1e-9)
    real_mhz = real_hz / 1_000_000

    print("--- RESULTS ---")
    print(f"Wall Time:    {elapsed_sec:.4f} s")
    print(f"GPU Cycles:   {total_cycles:,}")
    print(f"Measured Clk: {real_mhz:.2f} MHz")

    return real_mhz


if __name__ == "__main__":
    measure_true_clock(1_000_000)
    print("\n--- Warmup Complete ---\n")
    measure_true_clock(500_000_000)
