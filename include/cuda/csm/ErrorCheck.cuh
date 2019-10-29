#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// "Stolen" from http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api

#ifdef _DEBUG
    #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
	#define gpuVerify() { gpuCheckLastError(__FILE__, __LINE__); } 
#else
	#define gpuErrchk(ans) {}
	#define gpuVerify() {} 
#endif

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = false)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

inline void gpuCheckLastError(const char *file, int line)
{
	cudaStreamSynchronize(0);
	gpuAssert(cudaPeekAtLastError(), file, line);
}

