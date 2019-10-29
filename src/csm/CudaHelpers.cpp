#pragma once

#include "csm/CudaHelpers.h"
#include <iostream>

cuda_kernel_launch_size_t
computeLaunchSize1D(dim3 size, dim3 block)
{
	if (block.x == 0)
	{
		block = dim3(1024);
	}

	cuda_kernel_launch_size_t return_info;
	return_info.thread = dim3(std::min(block.x, size.x));
	int extraX = size.x % return_info.thread.x ? 1 : 0;
	return_info.block = dim3(size.x / return_info.thread.x + extraX);

	return return_info;
}

cuda_kernel_launch_size_t
computeLaunchSize2D(dim3 size, dim3 block)
{
	cuda_kernel_launch_size_t return_info;

	if (block.x == 0 || block.y == 0)
	{
		block = dim3(32, 32);
	}

    return_info.thread = dim3(std::min(block.x, size.x), std::min(block.y, size.y));
	int extraX = size.x % return_info.thread.x ? 1 : 0;
	int extraY = size.y % return_info.thread.y ? 1 : 0;
	return_info.block = dim3(size.x / return_info.thread.x + extraX, 
		                     size.y / return_info.thread.y + extraY);

	return return_info;
}

cuda_kernel_launch_size_t
computeLaunchSize3D(dim3 size, dim3 block)
{
	cuda_kernel_launch_size_t return_info;

	if (block.x == 0 || block.y == 0 || block.z == 0)
	{
		block = dim3(16, 8, 8);
	}

    return_info.thread = dim3(std::min(block.x, size.x), std::min(block.y, size.y), std::min(block.z, size.z));
	int extraX = size.x % return_info.thread.x ? 1 : 0;
	int extraY = size.y % return_info.thread.y ? 1 : 0;
	int extraZ = size.z % return_info.thread.z ? 1 : 0;
	return_info.block = dim3(size.x / return_info.thread.x + extraX, 
		                     size.y / return_info.thread.y + extraY, 
							 size.z / return_info.thread.z + extraZ);

	return return_info;
}


	
