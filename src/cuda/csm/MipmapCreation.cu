#ifndef __CUDACC__
	#define __CUDACC__
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <surface_functions.h>

__global__
void propagateMin(cudaSurfaceObject_t in, cudaSurfaceObject_t out, dim3 texDim)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < texDim.x && y < texDim.y)
	{
		float2 tl = make_float2(2*x  , 2*y);
		float2 tr = make_float2(2*x+1, 2*y);
		float2 bl = make_float2(2*x  , 2*y+1);
		float2 br = make_float2(2*x+1, 2*y+1);

		float val =    surf2Dread<float>(in, tl.x * sizeof(float), tl.y);
		val = min(val, surf2Dread<float>(in, tr.x * sizeof(float), tr.y));
		val = min(val, surf2Dread<float>(in, bl.x * sizeof(float), bl.y));
		val = min(val, surf2Dread<float>(in, br.x * sizeof(float), br.y));

		surf2Dwrite(val, out, x * sizeof(float), y);
	}

}

__global__
void propagateMax(cudaSurfaceObject_t in, cudaSurfaceObject_t out, dim3 texDim)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < texDim.x && y < texDim.y)
	{
		float2 tl = make_float2(2 * x, 2 * y);
		float2 tr = make_float2(2 * x + 1, 2 * y);
		float2 bl = make_float2(2 * x, 2 * y + 1);
		float2 br = make_float2(2 * x + 1, 2 * y + 1);

		float val = surf2Dread<float>(in, tl.x * sizeof(float), tl.y);
		val = max(val, surf2Dread<float>(in, tr.x * sizeof(float), tr.y));
		val = max(val, surf2Dread<float>(in, bl.x * sizeof(float), bl.y));
		val = max(val, surf2Dread<float>(in, br.x * sizeof(float), br.y));

		surf2Dwrite(val, out, x * sizeof(float), y);
	}

}

void
createMinMipmapLevel(cudaSurfaceObject_t prevLevel, cudaSurfaceObject_t nextLevel, dim3 texDim)
{
	dim3 thread(32, 32);
	int extraX = texDim.x % thread.x ? 1 : 0;
	int extraY = texDim.y % thread.y ? 1 : 0;
	dim3 block(texDim.x / thread.x + extraX, texDim.y / thread.y + extraY);
	propagateMin <<< block, thread >>> (prevLevel, nextLevel, texDim);
}

void
createMaxMipmapLevel(cudaSurfaceObject_t prevLevel, cudaSurfaceObject_t nextLevel, dim3 texDim)
{
	dim3 thread(32, 32);
	int extraX = texDim.x % thread.x ? 1 : 0;
	int extraY = texDim.y % thread.y ? 1 : 0;
	dim3 block(texDim.x / thread.x + extraX, texDim.y / thread.y + extraY);
	propagateMax <<< block, thread >>> (prevLevel, nextLevel, texDim);
}

