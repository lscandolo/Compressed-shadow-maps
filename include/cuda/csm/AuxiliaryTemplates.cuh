#pragma once

#include <cuda_runtime.h>
#include "../CudaHelpers.h"

#include <thrust/device_vector.h>

template <typename  T>
__global__
void copyTex2DToPtr(cudaTextureObject_t tex, T* ptr, int pixels, dim3 texDim)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= pixels)
		return;

	int y = i / texDim.y;
	int x = i - y * texDim.y;

	ptr[i] = tex2D<float>(tex, x, y);
}

template <typename  T>
void
copyTexture2DToPtr(cudaTextureObject_t tex, T* ptr, dim3 texDim)
{
	size_t pixels = texDim.x * texDim.y;
	dim3 thread(min((size_t)1024, pixels));
	int extra = pixels % thread.x ? 1 : 0;
	dim3 block(pixels / thread.x + extra);
	copyTex2DToPtr<T> << < block, thread >> > (tex, ptr, pixels, texDim);
}

template <typename  T>
__global__
void copyTex3DToPtr(cudaTextureObject_t tex, T* ptr, int pixels, dim3 texDim)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= pixels)
		return;

	int sliceSize = (texDim.x * texDim.y);
	int z = i / sliceSize;
	int rem = i - z * sliceSize;
	int y = rem / texDim.y;
	int x = rem - y * texDim.y;

	ptr[i] = tex3D<float>(tex, x, y, z);
}

template <typename  T>
void
copyTexture3DToPtr(cudaTextureObject_t tex, T* ptr, dim3 texDim)
{
	size_t pixels = texDim.x * texDim.y * texDim.z;
	cuda_kernel_launch_size_t launchSize = computeLaunchSize1D(pixels);
	copyTex3DToPtr<T> <<< launchSize.block, launchSize.thread >>> (tex, ptr, pixels, texDim);
}


//template <typename T>
//void
//copyDeviceVectorToDeviceVector(thrust::device_vector<typename T>::iterator srcBegin,
//                               thrust::device_vector<typename T>::iterator srcEnd,
//							   thrust::device_vector<typename T>::iterator dstBegin)
//{
//		thrust::copy(srcBegin, srcEnd, dstBegin);
//}