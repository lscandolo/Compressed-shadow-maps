#ifndef __CUDACC__
	#define __CUDACC__
#endif

#include <iostream> //!!

#include "cuda/csm/QuadtreeEvaluation-Relative.cuh"
#include "cuda/csm/ErrorCheck.cuh"
#include "cuda/csm/HelperFunctions.cuh"
#include "cuda/csm/QuadtreeEvaluation-Common.cuh"

#include <cuda.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <surface_functions.h>
#include <texture_fetch_functions.h>
#include <vector_types.h>

#include "csm/CudaHelpers.h"
#include "csm/QuadtreeTypes.h"

namespace Quadtree {

	struct partial_search
	{
		uint32_t nodeptr;
		level_t  level;
		uint8_t  flags;
		float    value;
	};

///////////////////////////////////////////////////////////

	static
	__device__
	float getDepthValue(const cuda_quadtree_data& qdata,
				 	    const int2 queryPoint,
					    const uint8_t lodlevel)
	{
		partial_search search;
		search.level = 0;
		search.nodeptr = 0;
		search.value = getQuadtreeValue(qdata, 1);
		search.flags = NODE_INNER;

		while (true)
		{
			// Read flags and pointer
			uint32_t flagsAndOffset = getQuadtreeFlagsAndOffet(qdata, search.nodeptr);
			uint8_t flags = flagsAndOffset; // & 0xff (but this is not needed)
			uint32_t offset = (flagsAndOffset >> 8) & 0xffffff;

			// Apply scaling to offset
			scaleOffset(offset, search.level, qdata);

			// Get next child index
			uint8_t nextchild = getNextChild(queryPoint, search.level, qdata.maxTreeLevel);

			// Try to traverse the next child (if it exists)
			uint8_t childflags = nextchild ? (flags >> (nextchild << 1)) & 3 : flags & 3;

			if (is_nonexistantnode(childflags) || (qdata.maxTreeLevel - search.level) <= lodlevel)
			{
				return search.value;
			}

			// Count memory up to that child using bit counting
			//if (nextchild) {
				flags &= (1 << (nextchild << 1)) - 1;	// Mask away all bits for the next child and the children after (e.g. if nextchild is 3rd child (=2), perform and with 0b00001111)
				offset += numberOfSetBits(flags);	// Simply add the number of bits to the offset (1 for leafs and empty nodes, 2 for inner nodes)
			//}

			// Setup traversal for child node
			search.level++;
			search.nodeptr += offset;
			search.flags = childflags;

			// If this is the leaf, just read the value and return
			if (is_leafnode(search.flags))
			{
				search.value += getQuadtreeValue(qdata, search.nodeptr);

				return search.value;
			}

			// If the node is not empty, store the position of the value
			if (is_innernode(search.flags)) {
				search.value += getQuadtreeValue(qdata, search.nodeptr + 1);
			}
			
		}

		return;	// Should never be reached
	}



////////////////////////////////////////////////////////////////////////////////////////////////////

	static
	__device__
	bool getCommonDepthValue(const cuda_quadtree_data& qdata,
						     partial_search& search,
							 const int2 queryPoint,
							 const level_t lastCommonLevel)
	{

		while(search.level < lastCommonLevel) 
		{

			// Read flags and pointer
			uint32_t flagsAndOffset = getQuadtreeFlagsAndOffet(qdata, search.nodeptr);
			uint8_t flags = flagsAndOffset; // & 0xff (but this is not needed)
			uint32_t offset = (flagsAndOffset >> 8) & 0xffffff;
			
			// Apply scaling to offset
			scaleOffset(offset, search.level, qdata);
		
			// Get next child index
			uint8_t nextchild = getNextChild(queryPoint, search.level, qdata.maxTreeLevel);

			// Try to traverse the next child (if it exists)
			uint8_t childflags = nextchild ? (flags >> (nextchild << 1)) & 3 : flags & 3;
			if (is_nonexistantnode(childflags))
			{
				// Return value from the last stored position
				return true;
			}
			
			// Count memory up to that child using bit counting
			//if (nextchild) {
				flags &= (1 << (nextchild << 1)) - 1;	// Mask away all bits for the next child and the children after (e.g. if nextchild is 3rd child (=2), perform and with 0b00001111)
				offset += numberOfSetBits(flags);	// Simply add the number of bits to the offset (1 for leafs and empty nodes, 2 for inner nodes)
			//}

			// Setup traversal for child node
			search.level++;
			search.nodeptr += offset;
			search.flags = childflags;

			bool isleaf  = is_leafnode(childflags);
			bool isinner = is_innernode(childflags);

			if (isleaf) {

				search.value += getQuadtreeValue(qdata, search.nodeptr);

				return true;
			}
			
			if (isinner) {

				search.value += getQuadtreeValue(qdata, search.nodeptr + 1);
			}

		}
	
		return false;	// We reached the last common level

	}

////////////////////////////////////////////////////////////////////////////////////////////////////
	static
	__device__
	float2 getDepthValueHierarchical(const cuda_quadtree_data& qdata,
									 partial_search& search,
									 const int2 queryPoint,
									 const uint8_t lodlevel)
	{

		float ceilVal;

		while(true) 
		{
			bool floorLevel = qdata.maxTreeLevel - search.level + 2 == lodlevel;
			bool ceilLevel  = qdata.maxTreeLevel - search.level + 1 == lodlevel;
			bool isempty = is_emptynode(search.flags);
	
			if (ceilLevel) { 
				ceilVal = search.value;
			} else if (floorLevel) {
				float floorVal = isempty ? search.value : ceilVal;
				
				return make_float2(ceilVal, floorVal);
			}

			// Read flags and offset
			uint32_t flagsAndOffset = getQuadtreeFlagsAndOffet(qdata, search.nodeptr);
			uint8_t flags = flagsAndOffset; // & 0xff (but this is not needed)
			uint32_t offset = (flagsAndOffset >> 8) & 0xffffff;
			
			// Apply scaling to offset
			scaleOffset(offset, search.level, qdata);
		
			// Get next child index
			uint8_t nextchild = getNextChild(queryPoint, search.level, qdata.maxTreeLevel);

			// Try to traverse the next child (if it exists)
			uint8_t childflags = nextchild ? (flags >> (nextchild << 1)) & 3 : flags & 3;
		
			if (is_nonexistantnode(childflags))
			{
				// Return value from the last stored position
				float floorVal = search.value;
			
				if (!floorLevel)
					ceilVal = floorVal;

				return make_float2(ceilVal, floorVal);
			}
			
			// Count memory up to that child using bit counting (if it's not the first node)
			if (nextchild) {
				flags &= (1 << (nextchild << 1)) - 1;	// Mask away all bits for the next child and the children after (e.g. if nextchild is 3rd child (=2), perform and with 0b00001111)
				offset += numberOfSetBits(flags);	// Simply add the number of bits to the offset (1 for leafs and empty nodes, 2 for inner nodes)
			}

			// Setup traversal for child node
			search.level++;
			search.nodeptr += offset;
			search.flags    = childflags;

			// If this is the leaf, just read the value and return
			if (is_leafnode(childflags))
			{

				float floorVal = search.value + getQuadtreeValue(qdata, search.nodeptr);

				if (!floorLevel)
					ceilVal = floorVal;

				return make_float2(ceilVal, floorVal);
			} 
			
			if (is_innernode(childflags)) 
			{
				// If the node is not empty, store the position of the value
				search.value += getQuadtreeValue(qdata, search.nodeptr + 1);
			}
			
		}
	
		return make_float2(0.f, 0.f);	// Should never be reached


	}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	static
	__device__
	float evaluateQuadtreeHierarchicalPCF(float4 coord,
										  const cuda_quadtree_data& qdata,
										  const float bias,  uint8_t pcf, 
										  float lodlevel, const float pixelsize)
	{

		// Compute query coordinate in depth map space and the get depth value of scene point point
		const uint32_t resolution = 1 << qdata.maxTreeLevel;
		const float2  queryCoords = make_float2(coord.x * resolution, coord.y * resolution);
		const int2   iQueryCoords = make_int2(queryCoords.x, queryCoords.y);
		const float pointDepth = coord.z;


		// Compute minimum and maximum query point of the kernel
		// Ksize is the size of a pixel, clamped between 1 and pow(2, flodlevel) 
		///  because flodlevel was clamped but fpixelsize was not
		const float ksize = min( max(pixelsize, 1.f), float(1 << __float2int_ru(lodlevel)));
		const int pcf_ksize = pcf * ksize;
		
		const int2 mincoords = iQueryCoords - make_int2(pcf_ksize, pcf_ksize);
		const int2 maxcoords = iQueryCoords + make_int2(pcf_ksize, pcf_ksize);
		
		/// Return 0 if kernel size goes out of bounds
		if (mincoords.x < 0 || mincoords.y < 0 || maxcoords.x >= resolution || maxcoords.y >= resolution)
			return 0.0;
	 
		/// Get last level for which traversal is the same for all points in the kernel
		const uint8_t minPossibleLevel = 32 - qdata.maxTreeLevel;
		level_t lastCommonLevel = getLastCommonLevel(mincoords, maxcoords, minPossibleLevel);

		// Traverse down the tree as long as the PCF kernel is still covered by a single node
		partial_search search;
		search.level = 0;
		search.nodeptr = 0;
		search.value = getQuadtreeValue(qdata, 1);
		search.flags = NODE_INNER;

		bool hasOnlyOneValue = getCommonDepthValue(qdata, search, iQueryCoords, lastCommonLevel);
		if (hasOnlyOneValue)
		{
			if (pointDepth >= search.value + bias)
				return 1.f;
			else 
				return 0.f;
		}

		float shadow = 0.f;

		int2 prevcoords = iQueryCoords + make_int2(-pcf, -pcf) * ksize;
		float2 lightDepth;
		partial_search pointsearch;

		/// Fractional part of the computed lod level
		const float fractLodLevel = (lodlevel - floorf(lodlevel));

		float stdMult = 1.f / float(2*pcf+1);
		float pcfMult = 1.f / (pcf*pcf);
		pointsearch.level = qdata.maxTreeLevel + 1;

		for (int8_t x = -pcf; x <= pcf; ++x) {
			for (int8_t y = -pcf; y <= pcf; ++y) {
				
				float2 nextcoords =  queryCoords +  make_float2(x, y) * float(ksize);
				int2   iNextcoords = make_int2(nextcoords.x, nextcoords.y);
				iNextcoords =  iQueryCoords +  make_int2(x, y) * ksize;

				float xVal = stdMult;
				float yVal = stdMult;
				if (lodlevel == 0) {
					xVal = pcfMult * max(0.f, pcf * ksize - fabsf(queryCoords.x - iNextcoords.x));
					yVal = pcfMult * max(0.f, pcf * ksize - fabsf(queryCoords.y - iNextcoords.y));
				}
				
				lastCommonLevel = getLastCommonLevel(prevcoords, iNextcoords, minPossibleLevel);

				// It's not '>' because getDepthValueHierarchical also looks at the previous level to do trilinear interpolation
				if (pointsearch.level >= lastCommonLevel) {
					pointsearch = search;
					prevcoords = iNextcoords;
					lightDepth = getDepthValueHierarchical(qdata, pointsearch, iNextcoords,int(lodlevel));

					if (pointDepth >= lightDepth.x + bias) {
						lightDepth.x = fractLodLevel;
					} else {
						lightDepth.x = 0;
					}

					if (pointDepth >= lightDepth.y + bias) {
						lightDepth.x += 1.f - fractLodLevel;
					}
				}
				
				shadow += lightDepth.x * xVal * yVal; 
			}
		}
		//shadow /= float((2*pcf+1) * (2*pcf+1));

		return shadow;

	}

////////////////////////////////////////////////////////////////////////////////////////////////

	static
	__device__
	float
	evaluateQuadtree(float4 coord,
	                 const float bias,
					 const cuda_quadtree_data& qdata,
					 float lodlevel)
	{

		float shadow = 0.f;
		uint32_t resolution = 1 << qdata.maxTreeLevel;
		int2   queryCoords = make_int2(coord.x * resolution, coord.y * resolution);
		float pointDepth = coord.z;
	
		float depthValue = getDepthValue(qdata, queryCoords, lodlevel);

		if (pointDepth >= depthValue + bias)
			shadow = 1.f;

		return shadow;
	}


	template<int PCFSIZE>
	static
	__global__
	void
	evaluateShadowKernel(const dim3                size,
						 const cuda_quadtree_data  qdata,
						 const cuda_scene_data     sdata,
						 const cudaTextureObject_t worldPosTex,
    					 const cudaTextureObject_t worldDPosTex,
						 const cudaTextureObject_t worldNorTex,
						 const cudaSurfaceObject_t resultSurf)
	{
		const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
		const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
		
		if (x >= size.x || y >= size.y) {
			return;
		} 
		
		float4 worldPos = tex2D<float4>(worldPosTex, float(x), float(y)); 
		if (worldPos.w == 0.f)
			return;

		worldPos.w = 1.f;

		float4 worldNor = tex2D<float4>(worldNorTex, float(x), float(y)); 
		float3 lightDir = sdata.isDirectional ? sdata.lightDir :  make_float3(worldPos) - sdata.lightPos;


		float costheta = dot(lightDir, make_float3(worldNor));
		
		if (costheta >= 0.f) return;

		float4 lCoords = computeNormalizedCoordinates(worldPos, sdata.nearplane, sdata.farplane, sdata.mvp, sdata.isDirectional);
		if (lCoords.x < 0.f || lCoords.x >= 1.f ||
			lCoords.y < 0.f || lCoords.y >= 1.f ||
			lCoords.z < 0.f || lCoords.z >= 1.f) {
			return;
		}
		 
		float4 worldDPos = tex2D<float4>(worldDPosTex, float(x), float(y));
		float4 deltaPos = make_float4(worldPos.x + worldDPos.x, worldPos.y + worldDPos.y, worldPos.z + worldDPos.z, 1.f);
		float4 deltaCoords = computeNormalizedCoordinates(deltaPos, sdata.nearplane, sdata.farplane, sdata.mvp, sdata.isDirectional);

		deltaCoords.x -= lCoords.x;
		deltaCoords.y -= lCoords.y;

		uint32_t resolution = 1 << qdata.maxTreeLevel;
		float fpixelsize = resolution * hypotf(deltaCoords.x, deltaCoords.y) * 0.33f;//!!
		float flodlevel = __log2f(fpixelsize);
		const float maxLod = 5.f;
		flodlevel = max(0.f, min(flodlevel, maxLod));
		float bias = (flodlevel*flodlevel + 1.f) * sdata.bias;
		
		if (!sdata.hierarchical) flodlevel = 0;

		float shadowValue;
		if (PCFSIZE > 0) {
			shadowValue = evaluateQuadtreeHierarchicalPCF(lCoords, qdata, bias, PCFSIZE, flodlevel, fpixelsize);
		} else {
			shadowValue = evaluateQuadtree(lCoords, bias, qdata, flodlevel);
		}

		uint8_t val = (1.f - shadowValue) * 255.f;

		if (val) { surf2Dwrite<uint8_t>(val, resultSurf, x * sizeof(uint8_t), y); }
	}
	 
	void evaluateShadow_Relative(const size_t width,
		                         const size_t height,
								 const uint32_t pcf_size,
								 const cuda_quadtree_data& qdata,
								 const cuda_scene_data& sdata,
								 const cudaTextureObject_t worldPosTex,
								 const cudaTextureObject_t worldDPosTex,
								 const cudaTextureObject_t worldNorTex,
								 const cudaSurfaceObject_t resultSurf)
	{
		dim3 size(width, height);
		dim3 block(16, 16);
		cuda_kernel_launch_size_t kernel = computeLaunchSize2D(size, block);

		//!!
		//cudaFuncSetCacheConfig(evaluateShadowKernel, cudaFuncCachePreferL1);
		
		switch (pcf_size) {
		default:
		case 0:
			evaluateShadowKernel<0> <<< kernel.block, kernel.thread >>> (size,
			                                                               qdata,
					    												   sdata,
						    											   worldPosTex,
							    										   worldDPosTex,
								    									   worldNorTex,
									    								   resultSurf);
			break;
		case 1:
			evaluateShadowKernel<1> <<< kernel.block, kernel.thread >>> (size,
			                                                               qdata,
					    												   sdata,
						    											   worldPosTex,
							    										   worldDPosTex,
								    									   worldNorTex,
									    								   resultSurf);
			break;
		case 2:
			evaluateShadowKernel<2> <<< kernel.block, kernel.thread >>> (size,
			                                                               qdata,
					    												   sdata,
						    											   worldPosTex,
							    										   worldDPosTex,
								    									   worldNorTex,
									    								   resultSurf);
			break;
		case 3:
			evaluateShadowKernel<3> <<< kernel.block, kernel.thread >>> (size,
			                                                               qdata,
					    												   sdata,
																		   worldPosTex,
							    										   worldDPosTex,
								    									   worldNorTex,
									    								   resultSurf);
			break;
		case 4:
			evaluateShadowKernel<4> <<< kernel.block, kernel.thread >>> (size,
			                                                               qdata,
					    												   sdata,
						    											   worldPosTex,
							    										   worldDPosTex,
								    									   worldNorTex,
									    								   resultSurf);
			break;
		case 5:
			evaluateShadowKernel<5> <<< kernel.block, kernel.thread >>> (size,
			                                                               qdata,
					    												   sdata,
						    											   worldPosTex,
							    										   worldDPosTex,
								    									   worldNorTex,
									    								   resultSurf);
			break;
		}
		gpuVerify();
	} 

}

