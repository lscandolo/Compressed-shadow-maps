#pragma once

#include <cstdint>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <cuda_runtime.h>

#define TIGHTEN          1
#define REORDER_TOP      0
#define MERGE_TESTS      1024
#define TAKE_STATS       0

#define USE_OMP_HASHES   1

#if (USE_OMP_HASHES && TAKE_STATS)

#error Cannot take stats and run in parallel at the same time

#endif

namespace Quadtree {

	enum ValueType {
		VT_ABSOLUTE = 0,
		VT_RELATIVE = 1
	};

	struct children_t
	{
		int32_t tl;
		int32_t tr;
		int32_t bl;
		int32_t br;
	};

	struct node_t
	{
		union {
			int32_t    childrenList[4];
			children_t children;
		};

		float    value;
		float    min, max;
		uint32_t value_index;
		uint32_t children_offset;

		uint8_t  flags; 
	};

	struct root_extra_t
	{
		int32_t  inode;
		float    max;
		float    min;
		uint32_t count;
	};

	struct node_extra_t
	{
		node_t node;
		float  max;
		float  min;
	};


	struct quadtree_config_t
	{
	private:
		quadtree_config_t();
	public:
		quadtree_config_t(size_t levels, uint32_t default_flagbytes, uint32_t default_ptrbytes, uint32_t default_valuebytes, uint32_t default_alignment)
		{
			flagbytes = default_flagbytes;
			ptrbytes.resize(levels, default_ptrbytes);
			valuebytes = default_valuebytes;
			alignment = default_alignment;
		}

		uint8_t alignment;
		uint8_t flagbytes;
		uint8_t valuebytes;
		std::vector<uint8_t> ptrbytes;
	};

	typedef std::vector<float> value_palette_t;

	/////////////////////////////////

	typedef uint16_t scale_t;
	typedef int8_t level_t;


	struct matrix { 
		//float4 c0, c1, c2, c3; 
		float4 r0, r1, r2, r3; 
	};

	struct cuda_quadtree_data { 
		uint32_t* quadtree;
		cudaTextureObject_t    scalesTexture;

		level_t  maxTreeLevel;

	};

	struct cuda_scene_data {
		matrix mvp;
		float nearplane;
		float farplane;
		float3 lightPos;
		float3 lightDir;
		float  bias;
		bool   hierarchical;
		bool   isDirectional;
	};

	typedef node_t gpu_node_t;


	
};