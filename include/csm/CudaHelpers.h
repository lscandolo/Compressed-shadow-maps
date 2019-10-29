#pragma once

#include <cstdio>

// Small fix for ill defined cuda header
#if defined(DEBUG) 
#    define  DEBUG 1
#    define _DEBUG 1
#endif

#if defined(_DEBUG) 
#    define _DEBUG 1
#    define  DEBUG 1
#endif

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#define CudaVerify(e) MxVerify((e) == cudaSuccess)
// #define CudaVerifyErr() MxVerify((cudaGetLastError()) == cudaSuccess)

//#define CudaVerifyErr() checkError(__LINE__)
#define CudaVerifyErr() 

static inline void checkError(const unsigned int line)
{
	cudaDeviceSynchronize();
    cudaError_t errcode = cudaGetLastError();
	if (errcode != cudaSuccess)
	{
		std::printf("CUDA error %d (line %d): %s \n", errcode, line, cudaGetErrorString(errcode));
		//MxVerify((cudaGetLastError()) == cudaSuccess);//!!
	}
}

struct cuda_kernel_launch_size_t
{
	dim3 block;
	dim3 thread;
};

#include <algorithm>

cuda_kernel_launch_size_t
computeLaunchSize1D(dim3 size, dim3 block = dim3(0, 0, 0));

cuda_kernel_launch_size_t
computeLaunchSize2D(dim3 size, dim3 block = dim3(0, 0, 0));

cuda_kernel_launch_size_t
computeLaunchSize3D(dim3 size, dim3 block = dim3(0, 0, 0));

