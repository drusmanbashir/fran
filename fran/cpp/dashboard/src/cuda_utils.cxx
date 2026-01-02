#include <iostream>
#include "cuda_utils.h"

int getGPUCount () {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    return 1;
  }
  return device_count;
}   


