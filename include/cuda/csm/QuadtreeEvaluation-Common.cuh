#ifndef __CUDACC__
	#define __CUDACC__
#endif

#include <cuda.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <surface_functions.h>
#include <texture_fetch_functions.h>
#include <vector_types.h>

#include "csm/QuadtreeTypes.h"

namespace Quadtree {

#define NODE_NONEXIST 	(uint8_t(0))
#define NODE_LEAF 		(uint8_t(1))
#define NODE_EMPTY 		(uint8_t(2))
#define NODE_INNER 		(uint8_t(3))

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	static
	__device__
	inline
	float4 matrixmul(const float4& v, const matrix& m)
	{
			return make_float4(dot(v, m.r0), dot(v, m.r1), dot(v, m.r2), dot(v, m.r3));

			/*
			float4 V;
			V.x = m.c0.x * v.x + m.c1.x * v.y + m.c2.x * v.z + m.c3.x * v.w;
			V.y = m.c0.y * v.x + m.c1.y * v.y + m.c2.y * v.z + m.c3.y * v.w;
			V.z = m.c0.z * v.x + m.c1.z * v.y + m.c2.z * v.z + m.c3.z * v.w;
			V.w = m.c0.w * v.x + m.c1.w * v.y + m.c2.w * v.z + m.c3.w * v.w;
			return V;
			*/
		}

	static
	__device__
	inline
	bool is_emptynode(const uint8_t flags)
	{
			return (flags == NODE_EMPTY);
		}

	static
	__device__
	inline
	bool is_leafnode(const uint8_t flags)
	{
			return (flags == NODE_LEAF);
		}

	static
	__device__
	inline
	bool is_innernode(const uint8_t flags)
	{
			return (flags == NODE_INNER);
		}

	static
	__device__
	inline
	bool is_nonexistantnode(const uint8_t flags)
	{
			return (flags == NODE_NONEXIST);
		}

	static
	__device__
	inline
	uint32_t numberOfSetBits(const uint32_t i)
	{
			return __popc(i);
			/*
			i = i - ((i >> 1) & 0x55555555);
			i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
			return int((((i + (i >> 4)) & 0x0f0f0f0f) * 0x01010101) >> 24);
			*/
		}

	static
	__device__
	inline
	level_t getLastCommonLevel(const int2 c1, const int2 c2, const level_t lastCommonLevel)
	{
			return __clz((c1.x^c2.x) | (c1.y^c2.y)) - lastCommonLevel;
		}


	static
	__device__
	inline
	void scaleOffset(uint32_t& offset, const level_t level, const cuda_quadtree_data& qdata)
	{
		scale_t scale = tex2D<uint16_t>(qdata.scalesTexture, level, 0);
		if (scale > 1)
			offset *= scale;
	}

	static
	__device__
	inline
	uint8_t getNextChild(const int2 queryPoint, const level_t level, const level_t maxTreeLevel)
	{
		// Get next child index
		uint8_t bitidx = (maxTreeLevel - level) - 1;
		uint32_t bit = 1 << bitidx;
		uint8_t nextchild = ((queryPoint.x & bit) | ((queryPoint.y & bit) << 1)) >> bitidx;

		return nextchild;

		//uint8_t bitidx = (maxTreeLevel - search.level) - 1;
		//uint8_t nextchild = (queryPoint.x >> bitidx)     & 1;
		//nextchild        |= (queryPoint.y >> (bitidx-1)) & 2;
	}

	static
	__device__
	inline
	uint32_t getQuadtreeFlagsAndOffet(const cuda_quadtree_data& qdata, const uint32_t ptr)
	{
		return qdata.quadtree[ptr];
	}

	static
	__device__
	inline
	float getQuadtreeValue(const cuda_quadtree_data& qdata, const uint32_t ptr)
	{
		return ((float*)qdata.quadtree)[ptr];
	}


	static
	__device__
	float4
	computeNormalizedCoordinates(float4 pos, float nearplane, float farplane, const matrix& mvp, bool isDirectional)
	{
			float4 lCoords = matrixmul(pos, mvp);
			lCoords *= 1.f / lCoords.w;
			lCoords = lCoords * 0.5f + make_float4(0.5f, 0.5f, 0.5f, 0.5f);
			if (!isDirectional)
			{
				lCoords.z = (2.f * nearplane) / (farplane + nearplane - lCoords.z * (farplane - nearplane));
			}
			return lCoords;
		}


}
