#ifndef __CUDACC__
	#define __CUDACC__
#endif

#include "cuda/csm/QuadTreeCreation.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <surface_functions.h>
#include <texture_fetch_functions.h>
#include <stdint.h>

#include "csm/CudaHelpers.h"

namespace Quadtree {

typedef float flag_t;
#define EMPTY_FLAG	0.0f
#define LEAF_FLAG	1.0f
#define INNER_FLAG	2.0f

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__
void sortValues(float* vals1, bool* isMin1, float* vals2, bool* isMin2)
{

#define IS_LOWER(V,F,i,j) (V[(i)] < V[(j)] || (V[(i)] == V[(j)] &&  F[(i)] && !F[(j)]))
#define COPY(Vsrc, Fsrc, Vdst, Fdst, src, dst) Vdst[(dst)] = Vsrc[(src)]; Fdst[(dst)] = Fsrc[(src)];

			//////////////// Sort pairs of values
			for (int i = 0; i < 4; ++i)
			{
				if (IS_LOWER(vals1, isMin1, i * 2, i * 2 + 1))
				{
					COPY(vals1, isMin1, vals2, isMin2, i * 2, i * 2);
					COPY(vals1, isMin1, vals2, isMin2, i * 2 + 1, i * 2 + 1);
				}
				else
				{
					COPY(vals1, isMin1, vals2, isMin2, i * 2, i * 2 + 1);
					COPY(vals1, isMin1, vals2, isMin2, i * 2 + 1, i * 2);
				}
			}

			//////////////// Sort sets of 4 values
			for (int i = 0; i < 2; ++i)
			{
				int idx1 = i * 4;
				int idx2 = i * 4 + 2;
				int idx1Limit = idx1 + 2;
				int idx2Limit = idx2 + 2;
				int dstIndex = i * 4;

				for (int j = 0; j < 4; ++j)
				{
					if (idx1 >= idx1Limit) {
						COPY(vals2, isMin2, vals1, isMin1, idx2, dstIndex);
						idx2++;
					}
					else if (idx2 >= idx2Limit || IS_LOWER(vals2, isMin2, idx1, idx2)) {
						COPY(vals2, isMin2, vals1, isMin1, idx1, dstIndex);
						idx1++;
					}
					else {
						COPY(vals2, isMin2, vals1, isMin1, idx2, dstIndex);
						idx2++;
					}

					dstIndex++;
				}
			}

			//////////////// Sort set of 8 values
			{
				int idx1 = 0;
				int idx2 = 4;
				int idx1Limit = 4;
				int idx2Limit = 8;
				int dstIndex = 0;

				for (int j = 0; j < 8; ++j)
				{
					if (idx1 >= idx1Limit) {
						COPY(vals1, isMin1, vals2, isMin2, idx2, dstIndex);
						idx2++;
					}
					else if (idx2 >= idx2Limit || IS_LOWER(vals1, isMin1, idx1, idx2)) {
						COPY(vals1, isMin1, vals2, isMin2, idx1, dstIndex);
						idx1++;
					}
					else {
						COPY(vals1, isMin1, vals2, isMin2, idx2, dstIndex);
						idx2++;
					}

					dstIndex++;
				}
			}

#undef IS_LOWER
#undef COPY

		}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	inline __device__ void count_ivals(const float2& ival, const float2& a, const float2& b, const float2& c, unsigned int& count)
	{
		count = 1;
		if (a.x <= ival.x && a.y >= ival.x)
			count++;
		if (b.x <= ival.x && b.y >= ival.x)
			count++;
		if (c.x <= ival.x && c.y >= ival.x)
			count++;

	}

	__global__
	void createLevel(const unsigned int level,
	                 const bool preserveBottomLevel,
	                 cudaTextureObject_t prevMin,
					 cudaTextureObject_t prevMax,
					 cudaSurfaceObject_t prevValue,
					 uint32_t*           prevExists,
					 cudaSurfaceObject_t nextMin,
					 cudaSurfaceObject_t nextMax,
					 cudaSurfaceObject_t nextValue,
					 uint32_t*           nextExists,
					 uint32_t*           totalCount,
					 uint8_t*  levelExists,
					 dim3 texDim)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x >= texDim.x || y >= texDim.y)
			return;

		// Get the bounds of the four children of the current node
		const int2 tl = make_int2(2 * x    , 2 * y);
		const int2 tr = make_int2(2 * x + 1, 2 * y);
		const int2 bl = make_int2(2 * x    , 2 * y + 1);
		const int2 br = make_int2(2 * x + 1, 2 * y + 1);

		const float2 tlBounds = make_float2(tex2D<float>(prevMin, tl.x, tl.y), tex2D<float>(prevMax, tl.x, tl.y)); // 2 x READ
		const float2 trBounds = make_float2(tex2D<float>(prevMin, tr.x, tr.y), tex2D<float>(prevMax, tr.x, tr.y)); // 2 x READ
		const float2 blBounds = make_float2(tex2D<float>(prevMin, bl.x, bl.y), tex2D<float>(prevMax, bl.x, bl.y)); // 2 x READ
		const float2 brBounds = make_float2(tex2D<float>(prevMin, br.x, br.y), tex2D<float>(prevMax, br.x, br.y)); // 2 x READ

		// // Add a minimum interval depth
		//float bias = 0.00001;
		//tlBounds.y = max(tlBounds.x + bias, tlBounds.y);
		//trBounds.y = max(trBounds.x + bias, trBounds.y);
		//blBounds.y = max(blBounds.x + bias, blBounds.y);
		//brBounds.y = max(brBounds.x + bias, brBounds.y);

		// Compute minimum and maximum of the 4 intervals
		const float gmin = max(max(tlBounds.x, trBounds.x), max(blBounds.x, brBounds.x));
		const float gmax = min(min(tlBounds.y, trBounds.y), min(blBounds.y, brBounds.y));

		// Find a new best value for the 4 intervals, also find the min/max values of the intervals covered by the best value
		float newValue = 0.0f;
		float minValue = 0.0;
		float maxValue = 1.0;

		if (gmin <= gmax)
		{
			newValue = (gmin + gmax) * 0.5f;
			minValue = gmin;
			maxValue = gmax;
		}
		else
		{
			unsigned int bestcounter = 0;
			unsigned int counter = 0;

			count_ivals(tlBounds, trBounds, blBounds, brBounds, counter);
			if (counter > bestcounter)
			{
				bestcounter = counter;
				newValue = tlBounds.x;
			}
			count_ivals(trBounds, tlBounds, blBounds, brBounds, counter);
			if (counter > bestcounter)
			{
				bestcounter = counter;
				newValue = trBounds.x;
			}
			count_ivals(blBounds, tlBounds, trBounds, brBounds, counter);
			if (counter > bestcounter)
			{
				bestcounter = counter;
				newValue = blBounds.x;
			}
			count_ivals(brBounds, tlBounds, trBounds, blBounds, counter);
			if (counter > bestcounter)
			{
				bestcounter = counter;
				newValue = brBounds.x;
			}

			if (newValue >= tlBounds.x && newValue <= tlBounds.y)
			{
				minValue = max(minValue, tlBounds.x);
				maxValue = min(maxValue, tlBounds.y);
			}
			if (newValue >= trBounds.x && newValue <= trBounds.y)
			{
				minValue = max(minValue, trBounds.x);
				maxValue = min(maxValue, trBounds.y);
			}
			if (newValue >= blBounds.x && newValue <= blBounds.y)
			{
				minValue = max(minValue, blBounds.x);
				maxValue = min(maxValue, blBounds.y);
			}
			if (newValue >= brBounds.x && newValue <= brBounds.y)
			{
				minValue = max(minValue, brBounds.x);
				maxValue = min(maxValue, brBounds.y);
			}
		}

		// Compute whether the child exists (parent doesn't cover it)
		bool tlExists = tlBounds.x > newValue || tlBounds.y < newValue;
		bool trExists = trBounds.x > newValue || trBounds.y < newValue;
		bool blExists = blBounds.x > newValue || blBounds.y < newValue;
		bool brExists = brBounds.x > newValue || brBounds.y < newValue;

		////////////////////////// Flags update
		// Update the lowest leaf level below this node
		size_t index = y * texDim.x + x;
		size_t tlIndex = tl.y * texDim.x * 2 + tl.x;
		size_t trIndex = tr.y * texDim.x * 2 + tr.x;
		size_t blIndex = bl.y * texDim.x * 2 + bl.x;
		size_t brIndex = br.y * texDim.x * 2 + br.x;

		uint32_t tlCount, trCount, blCount, brCount;
		if (level == 1 && !preserveBottomLevel) {
			tlCount = 1;
			trCount = 1;
			blCount = 1;
			brCount = 1;
		} else {
			tlCount = prevExists[tlIndex]; // 1 x READ
			trCount = prevExists[trIndex]; // 1 x READ
			blCount = prevExists[blIndex]; // 1 x READ
			brCount = prevExists[brIndex]; // 1 x READ
		}

		tlCount -= (tlExists || tlCount > 1) ? 0 : 1;
		trCount -= (trExists || trCount > 1) ? 0 : 1;
		blCount -= (blExists || blCount > 1) ? 0 : 1;
		brCount -= (brExists || brCount > 1) ? 0 : 1;

		prevExists[tlIndex] = tlCount ? 1 : 0; // 1 x WRITE
		prevExists[trIndex] = trCount ? 1 : 0; // 1 x WRITE
		prevExists[blIndex] = blCount ? 1 : 0; // 1 x WRITE
		prevExists[brIndex] = brCount ? 1 : 0; // 1 x WRITE

		if (texDim.x > 1) {
			nextExists[index] = tlCount + trCount + blCount + brCount + 1;  // 1 x WRITE
		} else {
			nextExists[index] = 1;  // 1 x WRITE
			totalCount[0] = tlCount + trCount + blCount + brCount + 1;
		}

		// Update values of the 4 children (set child to empty(=-1) if parent covers it)
		float tlPrev = tlExists ? (tlBounds.x + tlBounds.y) * 0.5f : -1.f;
		float trPrev = trExists ? (trBounds.x + trBounds.y) * 0.5f : -1.f;
		float blPrev = blExists ? (blBounds.x + blBounds.y) * 0.5f : -1.f;
		float brPrev = brExists ? (brBounds.x + brBounds.y) * 0.5f : -1.f;

		if (tlCount)
			surf2Dwrite(tlPrev, prevValue, tl.x * sizeof(float), tl.y);
		if (trCount)
			surf2Dwrite(trPrev, prevValue, tr.x * sizeof(float), tr.y);
		if (blCount)
			surf2Dwrite(blPrev, prevValue, bl.x * sizeof(float), bl.y);
		if (brCount)
			surf2Dwrite(brPrev, prevValue, br.x * sizeof(float), br.y);

		if (tlCount || trCount || blCount || brCount)
			levelExists[level - 1] = 1;

		// If this is the highest level, set the value to the average of the current min/max
		if (texDim.x == 1)
			surf2Dwrite(0.5f * (minValue + maxValue), nextValue, 0, 0);

		// Store the min/value for the node on the next level
		surf2Dwrite(minValue, nextMin, x * sizeof(float), y);
		surf2Dwrite(maxValue, nextMax, x * sizeof(float), y);
	}

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
						dim3 texDim)
	{
			dim3 thread(min(32, texDim.x), min(32, texDim.y));
			const int extraX = texDim.x % thread.x ? 1 : 0;
			const int extraY = texDim.y % thread.y ? 1 : 0;
			dim3 block(texDim.x / thread.x + extraX, texDim.y / thread.y + extraY);
			createLevel <<< block, thread >>> (level,
				                                 preserveBottomLevel,
				                                 prevMin, 
												 prevMax,
												 prevValue, 
												 prevExists,
												 nextMin, 
												 nextMax,
												 nextValue, 
												 nextExists, 
												 totalCount,
												 levelExists,
												 texDim);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	__global__
	void
	fillIndices(const uint32_t level,
	            const uint32_t threadNum,
				const uint32_t* exists,
				const uint32_t* offset,
				uint32_t*  outputLevel,
				uint32_t*  outputLocalIndexVector,
				int32_t*   outputGlobalIndexVector)
	{
			const int i = blockIdx.x * blockDim.x + threadIdx.x;

			if (i >= threadNum)
				return;

			if (!exists[i]) {
				outputGlobalIndexVector[i] = -1;
				return;
			}

			int outputIndex = (*offset) + outputGlobalIndexVector[i];
			
			outputLevel[outputIndex] = level;
			outputLocalIndexVector[outputIndex] = i;

			outputGlobalIndexVector[i] = outputIndex;
	}






	void 
	fillNodeIndices(const uint32_t level,
	                const uint32_t levelsize,
					const uint32_t* existsVector,
					const uint32_t* offset,
					uint32_t*  outputLevel,
					uint32_t*  outputLocalIndexVector,
					int32_t*   outputGlobalIndexVector)
	{
		uint32_t threadNum = levelsize*levelsize;
		cuda_kernel_launch_size_t kernel = computeLaunchSize1D(threadNum);
		fillIndices <<< kernel.block, kernel.thread >>> (level,
			                                               threadNum,
														   existsVector,
														   //countVector,
														   offset,
														   outputLevel,
														   outputLocalIndexVector,
														   outputGlobalIndexVector);
	}




	/////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INNER_BIT (0x0)
#define LEAF_BIT (0x1)
#define EMPTY_BIT (0x2)

	__global__
	void createNodes(const unsigned int level, 
	                 const bool preserveBottomLevel,
	                 const unsigned int lowestlevel,
		             const size_t levelsize,
					 const uint32_t* levels,
					 const uint32_t* lidx,
					 const int32_t* gidx,
  					 const uint32_t* exists,
					 cudaTextureObject_t values,
					 cudaTextureObject_t mins,
					 cudaTextureObject_t maxs,
					 const size_t numnodes,
					 const uint32_t root_idx,
					 const uint32_t* bottomLevelPtrs,
					 gpu_node_t* nodes)
	{
			const int i = blockIdx.x * blockDim.x + threadIdx.x;
			if (i >= numnodes)
				return;

			// Get level and check if it matches the node associated with the thread
			const uint32_t tlevel = levels[i];
			if (level != tlevel)
				return;

			// Get local index
			const uint32_t idx = lidx[i];

			// Fill in node information
			gpu_node_t node;
			// node.value = tex1Dfetch<float>(values, idx);

			const uint32_t yidx = idx / levelsize;
			const uint32_t xidx = idx - yidx * levelsize;
			node.value = tex2D<float>(values, xidx, yidx);

//!! Only allow if we need min and max
#if 1
			node.min = tex2D<float>(mins, xidx, yidx);
			node.max = tex2D<float>(maxs, xidx, yidx);
#endif

			bool isleaf = true;

			if (preserveBottomLevel && level == lowestlevel + 1)
			{
				unsigned int childofs[4][2] = { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } };
				for (size_t k = 0; k < 4; ++k)
				{
					const uint32_t y = idx / levelsize;
					const uint32_t x = idx - y * levelsize;
					const uint32_t childidx = (2 * y + childofs[k][1]) * levelsize * 2 + (2 * x + childofs[k][0]);
					node.childrenList[k] = exists[childidx] ? int32_t(bottomLevelPtrs[childidx]) : -1;
					if (node.childrenList[k] != -1)
					{
						isleaf = false;
					}
				}
			} 
			else if (level > lowestlevel)
			{
				unsigned int childofs[4][2] = { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } };
				for (size_t k = 0; k < 4; ++k)
				{
					const uint32_t y = idx / levelsize;
					const uint32_t x = idx - y * levelsize;
					const uint32_t childidx = (2 * y + childofs[k][1]) * levelsize * 2 + (2 * x + childofs[k][0]);
					node.childrenList[k] = gidx[childidx];
					
					if (node.childrenList[k] != -1)
					{
						node.childrenList[k] += root_idx;
						isleaf = false;
					}
				}
			}
			else 
			{
				for (size_t k = 0; k < 4; ++k)
					node.childrenList[k] = -1;
			}

			if (isleaf)
				node.flags = LEAF_BIT;
			else
				node.flags = (node.value == -1.0f) ? EMPTY_BIT : INNER_BIT;

			// Save node
			nodes[i] = node;
		}

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
						gpu_node_t* d_nodes)
	{
		const unsigned int numThreads = 1024;
		dim3 thread(numThreads, 1, 1);
		const int extra = numnodes % thread.x ? 1 : 0;
		dim3 block(numnodes / thread.x + extra, 1, 1);
		createNodes <<< block, thread >>> (level, preserveBottomLevel, lowestlevel, levelsize, d_levels, d_lidx, d_gidx, d_exists, values, mins, maxs, numnodes, root_idx, bottomLevelPtrs, d_nodes);
	}

}