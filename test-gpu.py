"""
Quick GPU test for RX 9070 XT with PyTorch ROCm 6.4.4
Run: python test-gpu.py
"""
import torch
import time

print("=" * 50)
print("PyTorch ROCm GPU Test")
print("=" * 50)

# Basic info
print(f"\nPyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Count: {torch.cuda.device_count()}")

if not torch.cuda.is_available():
    print("\n[ERROR] No GPU detected!")
    exit(1)

# Show device info
device_name = torch.cuda.get_device_name(0)
print(f"\nActive GPU: {device_name}")

if "9070" not in device_name:
    print("[WARNING] Not using RX 9070 XT! Check HIP_VISIBLE_DEVICES")

# Quick compute test
print("\n--- Running Matrix Multiply Test ---")
device = torch.device("cuda")

# Warmup
x = torch.randn(2000, 2000, device=device)
y = torch.randn(2000, 2000, device=device)
torch.matmul(x, y)
torch.cuda.synchronize()

# Benchmark
print("Computing 8000x8000 matrix multiply...")
start = time.time()
x = torch.randn(8000, 8000, device=device)
y = torch.randn(8000, 8000, device=device)
z = torch.matmul(x, y)
torch.cuda.synchronize()
elapsed = time.time() - start

print(f"Result: {elapsed:.3f} seconds")
print(f"TFLOPS: {(2 * 8000**3) / elapsed / 1e12:.2f}")

# Memory info
mem_allocated = torch.cuda.memory_allocated() / 1024**3
mem_reserved = torch.cuda.memory_reserved() / 1024**3
print(f"\nGPU Memory: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")

print("\n[SUCCESS] GPU is working correctly!")
