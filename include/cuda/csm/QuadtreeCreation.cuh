#pragma once

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include "csm/QuadtreeTypes.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//typedef Quadtree::children_t gpu_children_t;


namespace Quadtree{

	typedef node_t gpu_node_t;

	//struct gpu_children_t
	//{
	//	int32_t tl;
	//	int32_t tr;
	//	int32_t bl;
	//	int32_t br;
	//};
	//
	//struct gpu_node_t
	//{
	//	union {
	//		int32_t childrenList[4];
	//		gpu_children_t children;
	//	};
	//
	//	float value;
	//	uint8_t flags;
	//};

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	void
	createQuadtreeLevel(const unsigned int level,
	                    const bool preserveBottomLevel,
	                    cudaTextureObject_t prevMin, 
						cudaTextureObject_t prevMax,
						cudaSurfaceObject_t prevValue,
						uint32_t* prevExists,
						cudaSurfaceObject_t nextMin,
						cudaSurfaceObject_t nextMax,
						cudaSurfaceObject_t nextValue,
						uint32_t* nextExists,
						uint32_t* totalCount,
						uint8_t*  levelExists,
						dim3 texDim);


	void
	createQuadtreeNodes(const uint32_t level, 
	                    const bool preserveBottomLevel,
						const uint32_t lowestlevel,
	                    const size_t levelsize, 
						const uint32_t* d_levels, 
						const uint32_t* d_lidx, 
						const int32_t* d_gidx, 
						const uint32_t* d_exists,
						cudaTextureObject_t values, 
						cudaTextureObject_t mins, 
						cudaTextureObject_t maxs, 
						const size_t numnodes, 
						const uint32_t root_idx, 
						const uint32_t* bottomLevelPtrs,
						gpu_node_t* d_nodes);

	void 
	fillNodeIndices(const uint32_t level,
	                const uint32_t levelsize,
					const uint32_t* exists,
					//const uint32_t* count,
					const uint32_t* offset,
					uint32_t*  outputLevel,
					uint32_t*  outputLocalIndexVector,
					int32_t*   outputGlobalIndexVector);
	
};
