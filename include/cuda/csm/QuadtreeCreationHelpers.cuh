#pragma once

#include <cstdint>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "QuadTreeCreation.cuh"

namespace Quadtree {

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	void
	createThrustTemporaryResources(size_t size, 
	                               std::vector<thrust::device_vector<int32_t>>& tempGidx,
								   thrust::device_vector<uint32_t>& tempLevels,
								   thrust::device_vector<uint32_t>& tempLidx,
								   size_t initialTempLevelsSize
								   );

	void
	freeThrustTemporaryResources(std::vector<thrust::device_vector<int32_t>>& tempGidx,
	                             thrust::device_vector<uint32_t>& tempLevels,
								 thrust::device_vector<uint32_t>& tempLidx
								 );


	void
	createNodeIndices(const size_t size, 
	                  const bool preserveBottomLevel,
	                  const size_t numexists,
					  const uint8_t lowestlevel,
					  const uint8_t textureLevels,
					  const std::vector<uint32_t*>& exists, 
					  thrust::device_vector<uint32_t>& levels,
					  thrust::device_vector<uint32_t>& lidx,
					  std::vector<thrust::device_vector<int32_t>>& gidx);

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
					 const uint32_t* bottomLevelPtrs);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	void
	resizeTemporaryStorage(const size_t newsize,
	                       size_t&      currentsize,
						   gpu_node_t*& d_nodes,
						   thrust::device_vector<uint32_t>& levels,
						   thrust::device_vector<uint32_t>& lidx);


};