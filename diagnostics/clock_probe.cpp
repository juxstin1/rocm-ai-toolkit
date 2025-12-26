#include <hip/hip_runtime.h>
#include <iostream>

#define HIP_CHECK(command)                                    \
  do {                                                        \
    hipError_t status = command;                              \
    if (status != hipSuccess) {                               \
      std::cerr << "Error: " << hipGetErrorString(status)     \
                << std::endl;                                 \
      return -1;                                              \
    }                                                         \
  } while (0)

__global__ void clock_measure_kernel(long long* output, int duration_loops) {
  int tid = threadIdx.x;

  long long start = __clock64();

  // Busy work to keep clocks high (unrollable dependency).
  volatile long long dummy = 0;
  for (int i = 0; i < duration_loops; i++) {
    dummy += i * i;
  }

  // Prevent optimization.
  if (dummy == 12345) {
    output[1] = dummy;
  }

  long long end = __clock64();

  if (tid == 0) {
    output[0] = end - start;
  }
}

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

extern "C" EXPORT long long measure_gpu_cycles(int loops) {
  long long* d_out = nullptr;
  long long h_out[2] = {0, 0};

  HIP_CHECK(hipMalloc(&d_out, 2 * sizeof(long long)));

  clock_measure_kernel<<<1, 256>>>(d_out, loops);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(h_out, d_out, 2 * sizeof(long long),
                      hipMemcpyDeviceToHost));

  hipFree(d_out);
  return h_out[0];
}
