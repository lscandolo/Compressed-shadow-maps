#pragma once

#include "QuadtreeTypes.h"
#include "QuadtreeSerialization.h"

namespace Quadtree
{

	typedef uint64_t shape_hash_t;

	struct merge_info_t
	{
		float   subtree_all_max, subtree_all_min;
		float   subtree_any_max, subtree_any_min;
		int32_t subtree_size;
		uint32_t serialized_position;
		uint32_t merge_node;
		shape_hash_t  shape_hash;
		uint8_t level;
		uint8_t subtree_height;
		bool merged;
		bool merger;
		bool main_merger;
	};

	typedef std::unordered_map<shape_hash_t, std::vector<uint32_t> > mergeable_hash;

	struct merge_structs_t
	{
		std::vector< uint32_t > max_level_offsets;
		std::vector< uint32_t > order;
		std::vector< mergeable_hash > hash_tables;
		std::vector< std::set<shape_hash_t> > hash_keys;
		std::vector< std::unordered_map<shape_hash_t, bool> > existing_keys;
		size_t node_qty;
		size_t merged_size;
		size_t serialized_size;
	};


	void     createSerializedMergedNodeBuffer(SerializationInput& input, SerializationOutput& output);

	void     collectMergeStats(SerializationInput& input, std::vector<merge_info_t>& nodes_h, merge_structs_t& hs);
	void     collectMergeStatsRec(SerializationInput& input, std::vector<merge_info_t>& nodes_h, uint32_t current_index, uint32_t current_level, merge_structs_t& hs, float ancestor_depth);

	size_t   markMergedNodes(SerializationInput& input,  std::vector<merge_info_t>& nodes_h, SerializationOutput& output);
	size_t   matchNodes(std::vector<node_t>& nodes, std::vector<merge_info_t>& nodes_h, uint32_t levels, merge_structs_t& hs);
	bool     testEqual(const std::vector<node_t>& nodes, const std::vector<merge_info_t>& nodes_h, const node_t& n1, const node_t& n2, const bool topNode);

	void     mergeNode(std::vector<node_t>& nodes, std::vector<merge_info_t>& nodes_h, uint32_t merger, uint32_t merged);

	void     computeMergedSerializedOffsets(std::vector<node_t>& nodes, std::vector<merge_info_t>& nodes_h, uint32_t current_index, uint32_t current_serialized_pos, uint32_t& current_serialization_size);
	void     getMergedNodeArraySize(const std::vector<node_t>& nodes, std::vector<merge_info_t>& nodes_h, uint32_t index, uint32_t& serialized_size, uint32_t& merged_size, serializationStatistics& stats, uint32_t current_pos = 0, uint32_t level = 0);
	void     getMergedLevelOffsets(const std::vector<node_t>& nodes, const std::vector<merge_info_t>& nodes_h, std::vector<uint32_t>& offsets, uint32_t level, uint32_t index) ;
	uint32_t padMergedLevel(std::vector<node_t>& nodes, const std::vector<merge_info_t>& nodes_h, const uint32_t best_scale, const uint32_t level, const uint32_t index, uint32_t& total_padding, uint32_t padding = 0);
	void     fillSerializedMergedNodeBuffer(const std::vector<node_t>& nodes, const std::vector<merge_info_t>& nodes_h, std::vector<uint8_t>& nodebuffer, const std::vector<uint32_t>& scales, const uint32_t index, const uint32_t inode = 0, const uint32_t level = 0);


	void     traverseMergedSerializedTree(const std::vector<uint8_t>& nodes, const std::vector<uint32_t>& scales, const uint32_t inode, const bool isleaf, const bool isempty, uint32_t level, size_t& count);


};