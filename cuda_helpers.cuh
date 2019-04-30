#ifndef CUDA_HELPERS_HPP
#define CUDA_HELPERS_HPP

#include <stdint.h>
#include <math_constants.h>
#include <memory>
#include <cuda_runtime_api.h>
#include <cuda.h>

#define CUERR {                                                            \
    cudaError_t err;                                                       \
    if ((err = cudaGetLastError()) != cudaSuccess) {                       \
        std::cout << "CUDA error: " << cudaGetErrorString(err) << " : "    \
                    << __FILE__ << ", line " << __LINE__ << std::endl;       \
        exit(1);                                                           \
    }                                                                      \
}

// transfer constants
#define H2D (cudaMemcpyHostToDevice)
#define D2H (cudaMemcpyDeviceToHost)
#define H2H (cudaMemcpyHostToHost)
#define D2D (cudaMemcpyDeviceToDevice)

#endif