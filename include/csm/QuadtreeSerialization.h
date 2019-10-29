#pragma once

#include "QuadtreeTypes.h"

namespace Quadtree
{


	static const uint32_t ptrbits = 24;

	struct serializationStatistics
	{
		size_t node_qty;
		size_t leaf_qty;
		size_t full_inner_qty;
		size_t empty_inner_qty;
		
		uint64_t  total_time;
		uint64_t  offset_compute_time;
		uint64_t  node_output_time;

		serializationStatistics() 
			: node_qty(0)
			, leaf_qty(0)
			, full_inner_qty(0)
			, empty_inner_qty(0)
		{}
	};

	struct SerializationInput
	{
		SerializationInput(std::vector<node_t>& _nodes) : nodes(_nodes){}

		std::vector<node_t>& nodes;
		uint32_t levels;
		uint32_t root_index;
		Quadtree::ValueType value_types;
	};

	struct SerializationOutput
	{
		std::vector<uint32_t> levelscales;
		std::vector<uint8_t>   nodebuffer;
		serializationStatistics stats;
	};

// Optimized version

void     setInitialNodeOffsets(SerializationInput& input, const uint32_t index, uint32_t& currentSize, float ancestor_depth, uint32_t currentPos);
void     getLevelOffsets(const std::vector<node_t>& nodes, std::vector<uint32_t>& offsets, uint32_t level, uint32_t index) ;
uint32_t computeBestLevelScale(const std::vector<uint32_t>& offsets) ;
uint32_t padLevel(std::vector<node_t>& nodes, uint32_t best_scale, uint32_t level, uint32_t index, uint32_t& total_padding, uint32_t padding = 0);
void     getNodeArraySize(std::vector<node_t>& nodes, uint32_t index, uint32_t& size, uint32_t current_pos = 0, uint32_t level = 0);
void     fillINode(const node_t& node, std::vector<uint8_t>& nodebuffer, uint32_t inode, const uint8_t flags, const uint32_t childptr);
void     fillSerializedNodeBuffer(std::vector<node_t>& nodes, std::vector<uint8_t>& nodebuffer, const std::vector<uint32_t>& scales, uint32_t index, uint32_t inode = 0, uint32_t level = 0);

//void     traverseSerializedTree(const std::vector<uint8_t>& nodes, const std::vector<uint32_t>& scales, const uint32_t inode, const bool isleaf, const bool isempty, uint32_t level, size_t& count);
//void     traverseOriginalTree(std::vector<node_t>& nodes, uint32_t index, serializationStatistics& stats, uint32_t level, bool markUsed = false);

void     createSerializedNodeBuffer(SerializationInput& input, SerializationOutput& output);

};