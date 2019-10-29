#ifndef __CUDACC__
	#define __CUDACC__
#endif

#include "cuda/csm/QuadTreeCreationHelpers.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <surface_functions.h>
#include <texture_fetch_functions.h>

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/scatter.h>
#include <thrust/scan.h>

#include "csm/CudaHelpers.h"
#include "helpers/ScopeTimer.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace Quadtree {

	void
	createThrustTemporaryResources(size_t size, 
	                               std::vector<thrust::device_vector<int32_t>>& tempGidx,
								   thrust::device_vector<uint32_t>& tempLevels,
								   thrust::device_vector<uint32_t>& tempLidx,
   								   size_t initialTempLevelsSize
								   )
	{
		const uint32_t mipmapLevels = log2(size) + 1;
		tempGidx.resize(mipmapLevels);
		for (size_t i = 0; i < mipmapLevels; ++i)
		{
			size_t current_size = size >> i;
			tempGidx[i].resize(current_size * current_size);
		}

		tempLevels.resize(initialTempLevelsSize);
		tempLidx.resize(initialTempLevelsSize);
	}

	void
	freeThrustTemporaryResources(std::vector<thrust::device_vector<int32_t>>& tempGidx,
	                             thrust::device_vector<uint32_t>& tempLevels,
								 thrust::device_vector<uint32_t>& tempLidx
								 )
	{
		tempGidx.clear();
		tempLevels.clear();
		tempLidx.clear();
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	__global__
	void
	addCount(uint32_t* offset,
	         uint32_t* new_count,
	         uint32_t* last_exists)
	{
			*offset += *new_count + *last_exists;
	}

	void
	createNodeIndices(const size_t size, 
	                  const bool preserveBottomLevel,
	                  const size_t numexists, 
					  const uint8_t lowestlevel,
					  const uint8_t textureLevels,
					  const std::vector<uint32_t*>& existsVector, 
					  thrust::device_vector<uint32_t>& levels, 
					  thrust::device_vector<uint32_t>& lidx, 
					  std::vector<thrust::device_vector<int32_t>>& gidx)
	{
		using namespace thrust::placeholders;

		const uint32_t mipmapLevels = log2(size) + 1;
		const uint32_t topLevel = textureLevels - 1;
		const uint32_t bottomLevel = (topLevel + 1) - mipmapLevels;

		static uint32_t* level_total = 0;
		static uint32_t* offset = 0;

		
		uint32_t* outputLevel       = thrust::raw_pointer_cast(levels.data());
		uint32_t* outputLocalIndex  = thrust::raw_pointer_cast(lidx.data());

		if (!level_total)
		{
			cudaMalloc(&offset, sizeof(uint32_t));
			cudaMalloc(&level_total, sizeof(uint32_t));
		}

		// Set offset to 0
		cudaMemset(offset, 0, sizeof(uint32_t));

		uint8_t levelSkip = preserveBottomLevel ? 1 : 0;


		// Compute indices
		uint32_t ofs = 0;
		for (int32_t i = topLevel; i >= lowestlevel + levelSkip; --i)
		{
			const size_t csize = size >> (i - bottomLevel);
			const size_t csize_sqr = csize * csize;
			uint32_t numonlevel = 0;
			// Copy all the local indices of the nodes on the current level (stream compaction of existing nodes)

			int32_t*  outputGlobalIndex = thrust::raw_pointer_cast(gidx[i].data());
			thrust::device_ptr<int32_t> exists_ptr = thrust::device_pointer_cast((int32_t*)existsVector[i]);
			thrust::device_ptr<int32_t> count_ptr =  thrust::device_pointer_cast(gidx[i].data());
			{//PROFILE_SCOPE("Node Creation-Scan");
				
				thrust::exclusive_scan(exists_ptr,                // Input start
					                   exists_ptr + csize_sqr,    // Input end
									   count_ptr,                    // Output start
									   0,                            // initial value
									   thrust::plus<uint32_t>());    // op
									   
			}

			cudaMemcpy(level_total, outputGlobalIndex + csize_sqr - 1, sizeof(uint32_t), cudaMemcpyDeviceToDevice);

			{//PROFILE_SCOPE("Node Creation-fillNodeIndices");
			fillNodeIndices(i,
				            csize,
							existsVector[i],
							//level_count,
							offset,
							outputLevel,
							outputLocalIndex,
							outputGlobalIndex
							);
			}
			

			// Add the resulting count (in level_count + csize_sqr) to offset
			{
				addCount <<< dim3(1), dim3(1) >>> (offset, level_total, existsVector[i] + csize_sqr - 1);
			}

		}
		CudaVerifyErr();
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	void
	createNodeArrays(const size_t size, 
	                 const bool preserveBottomLevel,
		             const size_t numexists, 
					 const uint8_t lowestlevel,
					 const uint8_t textureLevels,
					 const thrust::device_vector<uint32_t>& levels,
					 const thrust::device_vector<uint32_t>& lidx,
					 const std::vector<thrust::device_vector<int32_t>>& gidx,
					 const std::vector<uint32_t*> existsVector,
					 const std::vector<cudaTextureObject_t>& value_textures,
					 const std::vector<cudaTextureObject_t>& min_textures,
					 const std::vector<cudaTextureObject_t>& max_textures,
					 gpu_node_t*& d_nodes,
					 const uint32_t root_idx,
 					 const uint32_t* bottomLevelPtrs)

	{
		// Assume d_nodes is big enough to fit all numexists nodes
		const uint32_t mipmapLevels = log2(size) + 1;
		const uint32_t topLevel = textureLevels - 1;
		const uint32_t bottomLevel = (topLevel + 1) - mipmapLevels;

		uint8_t levelSkip = preserveBottomLevel ? 1 : 0;

		// Run the kernel which creates the nodes on the GPU
		for (uint32_t i = lowestlevel + levelSkip; i <= topLevel; ++i)
		{
			size_t current_size = size >> (i - bottomLevel);
			const int32_t* gidx_ptr = (i > lowestlevel) ? thrust::raw_pointer_cast(&gidx[i - 1][0]) : NULL;
			const uint32_t* exists_ptr = (i > lowestlevel) ? existsVector[i-1] : NULL;
			createQuadtreeNodes(i,
				                preserveBottomLevel,
				                lowestlevel,
				                current_size,
								thrust::raw_pointer_cast(&levels[0]),
								thrust::raw_pointer_cast(&lidx[0]),
								gidx_ptr, 
								exists_ptr,
								value_textures[i], 
								min_textures[i], 
								max_textures[i],
								numexists, 
								root_idx,
								bottomLevelPtrs,
								d_nodes);
		}
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	void
	resizeTemporaryStorage(const size_t newsize,
	                       size_t&      currentsize,
						   gpu_node_t*& d_nodes,
						   thrust::device_vector<uint32_t>& levels,
						   thrust::device_vector<uint32_t>& lidx)
	{
			if (currentsize < newsize) {
				currentsize = newsize * 1.2;
				levels.resize(currentsize);
				lidx.resize(currentsize );
				cudaFree(d_nodes);
				cudaMalloc(&d_nodes, sizeof(gpu_node_t)* currentsize);
			}
	}

};