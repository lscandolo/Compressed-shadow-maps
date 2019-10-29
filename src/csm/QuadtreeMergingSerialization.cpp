#include "common.h"
#include "csm/QuadtreeTypes.h"
#include "csm/QuadtreeSerialization.h"
#include "csm/QuadtreeMergingSerialization.h"
#include "csm/QuadtreePrivate.h"
#include "helpers/ScopeTimer.h"
#include <iostream>
#include <algorithm>

// Stupid windows
#include <Windows.h> 
#undef min
#undef max

namespace Quadtree
{

	static const uint32_t mergelevels = 12;

	static inline uint32_t getSerializedNodeSize(const uint8_t flags)
	{
		switch (flags)
		{
		case NODE_INNER: return 8;
		case NODE_EMPTY: return 4;
		case NODE_LEAF:  return 4;
		default: return 0;
		};
	}

	static inline uint32_t getNodeSize(const uint8_t flags)
	{
		if (flags & LEAF_BIT)
			return 4;
		if (flags & EMPTY_BIT)
			return 4;

		return 8;
	}

	static inline bool isLeaf(const node_t& node)
	{
		return (node.flags & EMPTY_BIT);
	}

	static inline bool isEmpty(const node_t& node)
	{
		return (node.flags & EMPTY_BIT);
	}

	static inline bool isFull(const node_t& node)
	{
		return !(node.flags);
	}

	//static std::vector<merge_structs_t> _hs;

	size_t
	markMergedNodes(SerializationInput& input, std::vector<merge_info_t>& nodes_h, SerializationOutput& output)
	{

		merge_structs_t _hs;

		// Collect stats from the non-serialized tree to do the mergeing
		{ PROFILE_SCOPE("Mergeing - Create hash tables");
			collectMergeStats(input, nodes_h, _hs);
		}

		// Find the mergeable nodes and mark them 
		{PROFILE_SCOPE("Mergeing - Merge nodes")
			size_t merged_size = matchNodes(input.nodes, nodes_h, input.levels, _hs);
		}

	return 0;
	}
    
	size_t
	matchNodes(std::vector<node_t>& nodes, 
	           std::vector<merge_info_t>& nodes_h, 
			   uint32_t levels, 
			   merge_structs_t& hs)
	{
		size_t merged_size = 0;

		//
		std::vector <size_t> hash_max_size;
		if (TAKE_STATS)
			hash_max_size.resize(hs.hash_keys.size(), 0);
		//

		size_t minmax_rejections = 0;
		size_t merger_rejections = 0;
		size_t merged_rejections = 0;
		size_t test_rejections = 0;
		size_t size_rejections = 0;
		size_t offset_rejections = 0;
		size_t total_tests = 0;

		for (size_t level =  hs.hash_keys.size() - 1; level > 0; --level)
		{
			mergeable_hash& hash_table = hs.hash_tables[level];
			std::set<shape_hash_t>& hash_keys = hs.hash_keys[level];
			


#if USE_OMP_HASHES
			std::vector<shape_hash_t> hash_vector;
			hash_vector.reserve(hash_keys.size());
			for (auto key_it = hash_keys.begin(); key_it != hash_keys.end(); key_it++)
				hash_vector.push_back(*key_it);
			//std::set<shape_hash_t>().swap(hash_keys);
            #pragma omp parallel for
			for (int k = 0;  k < hash_vector.size(); k++)
			{
				if (hash_table.count(hash_vector[k]) < 1)
					continue;
				std::vector<uint32_t>& collision_list = hash_table[hash_vector[k]];
#else
			for (auto key_it = hash_keys.begin(); key_it != hash_keys.end(); key_it++) {
				std::vector<uint32_t>& collision_list = hash_table[*key_it];
#endif

				
				// If there is no collision, skip it
				if (collision_list.size() <= 1)
					continue;

				// Save maximum collision list size for this level
				if (TAKE_STATS && collision_list.size() > hash_max_size[level])
					hash_max_size[level] = collision_list.size();
				
				
				// last_idx points to the last known not merged index 
				// (we swap merged nodes to this value and decrement it, so merged nodes are sent to the end)
				uint32_t last_idx = uint32_t(collision_list.size()) - 1;

				// index1 : merged
				// index2 : merger
				//for (uint32_t index1 = 0; index1 < collision_list.size(); ++index1) {
				for (uint32_t index1 = uint32_t(collision_list.size()) - 1; index1 != (uint32_t)-1; --index1) {

					// Backward search in increasing steps
					//size_t test_qty = std::min(size_t(MERGE_TESTS), collision_list.size() - index1 - 1);
					//const size_t STEP_SIZE = 1000;
					//for (size_t limit = 0; limit < test_qty; limit += STEP_SIZE) 
					//for (size_t index2 = std::min(index1 + limit + STEP_SIZE, last_idx); index2 > index1 + limit; --index2) {

					// Backward search: More memory save but mergeing can take a node very far
					//for (uint32_t index2 = std::min(index1 + MERGE_TESTS, uint32_t(last_idx)); index2 < collision_list.size() && index2 > index1; --index2) {

					// Forward search: Closer mergeing but less memory save
					for (size_t index2 = index1 + 1; index2 <= std::min(index1 + MERGE_TESTS, last_idx); ++index2) {
					//for (size_t index2 = index1 + 1; index2 < last_idx; ++index2) { //!! This is for testing with no bounds
					

						//uint32_t i1 = collision_list[index1];
						uint32_t i1 = collision_list[index1];
						uint32_t i2 = collision_list[index2];

						node_t& n1 = nodes[i1];
						node_t& n2 = nodes[i2];

						merge_info_t& n1_h = nodes_h[i1];
						merge_info_t& n2_h = nodes_h[i2];

						// Check if node 2 is already merged
						if (n2_h.merged) {

							// Send node 2 to the back, because since it's already merged we don't need it anymore
							std::swap(collision_list[index2], collision_list[last_idx--]);

							if (TAKE_STATS)
								merged_rejections++;
							continue;
						}

						if (TAKE_STATS)
							total_tests++;

						// Check if aggregated intervals are compatible
						if (n1_h.subtree_any_max < n2_h.subtree_all_min || n1_h.subtree_any_min > n2_h.subtree_all_max) {
							if (TAKE_STATS)
								minmax_rejections++;
							continue;
						}

						// Check if offset is too large
						if (n2_h.serialized_position - n1_h.serialized_position > 1 << (ptrbits-1)) {
							if (TAKE_STATS)
								offset_rejections++;
							
							continue;
						}

						// Full test
						if (!testEqual(nodes, nodes_h, n1, n2, true)) {
							if (TAKE_STATS)
								test_rejections++;
							continue;
						}

						mergeNode(nodes, nodes_h, i1, i2);
						// Send node 1 to the back, because since it's already merged we don't need it anymore
						for (uint32_t i = 0; i < 4; i++)
						{
							nodes[i1].childrenList[i] = nodes[i2].childrenList[i];
						}
						
						std::swap(collision_list[index1], collision_list[last_idx--]);

						merged_size += n1_h.subtree_size;
						break;

					}
				}
			}
		}

		if (TAKE_STATS) {

			size_t mergex_rejections = merged_rejections + merger_rejections;

			double merger_perc = 100. * (merger_rejections / double(total_tests));
			double merged_perc = 100. * (merged_rejections / double(total_tests));
			double mergex_perc = 100. * (mergex_rejections / double(total_tests));
			double interval_perc = 100. * (minmax_rejections / double(total_tests));
			double size_perc = 100. * (size_rejections / double(total_tests));
			double offset_perc = 100. * (offset_rejections / double(total_tests));
			double test_perc = 100. * (test_rejections / double(total_tests));

			std::cout << "Mergeing stats: " << std::endl;
			std::cout << "Total tests: " << total_tests << std::endl;
			std::cout << "Merger rejections: " << merger_rejections << " ( " << merger_perc << "% )" << std::endl;
			std::cout << "Merged rejections: " << merged_rejections << " ( " << merged_perc << "% )" << std::endl;
			std::cout << "Mergex rejections: " << mergex_rejections << " ( " << mergex_perc << "% )" << std::endl;
			std::cout << "Interval rejections: " << minmax_rejections << " ( " << interval_perc << "% )" << std::endl;
			std::cout << "Size     rejections: " << size_rejections << " ( " << size_perc << "% )" << std::endl;
			std::cout << "Offset   rejections: " << offset_rejections << " ( " << offset_perc << "% )" << std::endl;
			std::cout << "Test     rejections: " << test_rejections << " ( " << test_perc << "% )" << std::endl;
		}


		return merged_size;

	}

	bool     
	testEqual(const std::vector<node_t>& nodes, 
	          const std::vector<merge_info_t>& nodes_h,
			  const node_t& n1, 
			  const node_t& n2, 
			  const bool topNode)
	{

		// n1: potential merged
		// n2: potential merger

		if (topNode && REORDER_TOP) 
		{
			uint32_t i1_start = 0, i2_start = 0;
			uint32_t i1_end = 3;


			while (n1.childrenList[i1_start] == -1) i1_start++;
			while (n1.childrenList[i1_end] == -1) i1_end--;
			while (n2.childrenList[i2_start] == -1) i2_start++;

			for (uint32_t i1 = i1_start, i2 = i2_start; i1 <= i1_end; ++i1, ++i2)
			{
				if (i2 > 3)
					return false;

				int32_t idx1 = n1.childrenList[i1];
				int32_t idx2 = n2.childrenList[i2];

				if (idx1 == -1)
				{
					if (idx2 != -1)
						return false;
					else
						continue;
				} else {
					if (idx2 == -1)
						return false;
				}


				const node_t& c1 = nodes[idx1];
				const node_t& c2 = nodes[idx2];
				const merge_info_t& c1_h = nodes_h[idx1];
				const merge_info_t& c2_h = nodes_h[idx2];

				if (c1.flags != c2.flags || c1_h.shape_hash != c2_h.shape_hash || c1_h.merger)
					return false;

				if ((c1.flags & EMPTY_BIT) == 0) 
				{
					if (TIGHTEN)
					{
						// Assuming we will shorten mergers min/max
						if (c1.max < c2.min || c1.min > c2.max)
						{
							return false;
						}
					} else {
					// Assuming we will NOT shorten mergers min/max
						if (c1.min > c2.value || c1.max < c2.value)
						{
							return false;
						}
					}
				}

				if ( (c1.flags & LEAF_BIT) == 0)
				{
					if (!testEqual(nodes, nodes_h, c1, c2, false))
						return false;
				}
	
			}
		}
		else 
		{
			for (uint32_t i = 0; i < 4; ++i)
			{
				int32_t idx1 = n1.childrenList[i];
				int32_t idx2 = n2.childrenList[i];

				if (idx1 == -1 || idx2 == -1)
				{
					if (idx1 != idx2) 
						return false;
					else
						continue;
				}

				const node_t& c1 = nodes[idx1];
				const node_t& c2 = nodes[idx2];
				const merge_info_t& c1_h = nodes_h[idx1];
				const merge_info_t& c2_h = nodes_h[idx2];

				if (c1.flags != c2.flags || c1_h.shape_hash != c2_h.shape_hash)
					return false;

				if ((c1.flags & EMPTY_BIT) == 0) 
				{
					if (TIGHTEN)
					{
						// Assuming we will shorten mergers min/max
						if (c1.max < c2.min || c1.min > c2.max)
						{
							return false;
						}
					} else {
					// Assuming we will NOT shorten mergers min/max
						if (c1.min > c2.value || c1.max < c2.value)
						{
							return false;
						}
					}
				}

				if ( (c1.flags & LEAF_BIT) == 0)
				{
					if (!testEqual(nodes, nodes_h, c1, c2, false))
						return false;
				}
	
			}
		}

		return true;		

    }
	
	static void    
	markSubtreeAsMerged(std::vector<node_t>& nodes,
                          std::vector<merge_info_t>& nodes_h,
	                      uint32_t merged)
	{
		node_t& node = nodes[merged];
		merge_info_t& node_h = nodes_h[merged];
		
		node_h.merged = true;
		
		if (node.flags & LEAF_BIT)
			return;

		for (uint32_t i = 0; i < 4; ++i)
		{
			int32_t c = node.childrenList[i];
			if (c < 0)
				continue;
			markSubtreeAsMerged(nodes, nodes_h, c);
		}
	}

	static void     
	tightenMergerTree(std::vector<node_t>& nodes, 
						uint32_t merger, 
						uint32_t merged, 
						bool topNode)
	{
		node_t& n1 = nodes[merged];
		node_t& n2 = nodes[merger];

		if (topNode && REORDER_TOP)
		{
			uint32_t i1_start = 0, i2_start = 0;

			while (n1.childrenList[i1_start] == -1) i1_start++;
			while (n2.childrenList[i2_start] == -1) i2_start++;

			for (uint32_t i1 = i1_start, i2 = i2_start; i1 < 4; ++i1, ++i2)
			{
				int32_t idx1 = n1.childrenList[i1];
				int32_t idx2 = n2.childrenList[i2];

				if (idx1 == -1)
				{
					continue;
				}

				node_t& c1 = nodes[idx1];
				node_t& c2 = nodes[idx2];

				if ((c2.flags & EMPTY_BIT) == 0)
				{
					c2.min = std::max(c2.min, c1.min);
					c2.max = std::min(c2.max, c1.max);
					c2.value = (c2.max + c2.min) * 0.5f;
				}

				if ((c1.flags & LEAF_BIT) == 0)
				{
					tightenMergerTree(nodes, n2.childrenList[i2], n1.childrenList[i1], false);
				}
			}
		}
		else 
		{
			for (uint32_t i = 0; i < 4; ++i)
			{
				int32_t idx1 = n1.childrenList[i];

				if (idx1 == -1)
				{
					continue;
				}

				int32_t idx2 = n2.childrenList[i];

				node_t& c1 = nodes[idx1];
				node_t& c2 = nodes[idx2];

				if ((c2.flags & EMPTY_BIT) == 0)
				{
					c2.min = std::max(c2.min, c1.min);
					c2.max = std::min(c2.max, c1.max);
					c2.value = (c2.max + c2.min) * 0.5f;
				}

				if ((c2.flags & LEAF_BIT) == 0)
				{
					tightenMergerTree(nodes, n2.childrenList[i], n1.childrenList[i], false);
				}
			}
		}
	}

	void     
	mergeNode(std::vector<node_t>& nodes, 
	           std::vector<merge_info_t>& nodes_h, 
			   uint32_t merged, 
			   uint32_t merger)
	{
		markSubtreeAsMerged(nodes, nodes_h, merged);
		nodes_h[merged].merge_node = merger;
		nodes_h[merged].main_merger = true;
		nodes_h[merger].merger = true;

		if (TIGHTEN) 
		{
			tightenMergerTree(nodes, merger, merged, true);
		}
	}


	
	void
	collectMergeStats(SerializationInput& input, 
	                   std::vector<merge_info_t>& nodes_h, 
					   merge_structs_t& hs)
	{
		
		hs.node_qty = 0;
		hs.order.clear();
		hs.order.reserve(input.nodes.size());
		hs.merged_size = 0;
		hs.serialized_size = 0;
		hs.order.push_back(input.root_index);
		hs.hash_keys.resize(input.levels);
		hs.existing_keys.resize(input.levels);
		hs.hash_tables.resize(input.levels);
		hs.max_level_offsets.resize(input.levels);


		// Reserve size so insertions are faster
		//for (uint32_t i = 0; i < levels; ++i)
		//{
		//	hs.hash_tables[i].reserve(8192);
		//}

		// Collect stats recursively
		{// PROFILE_SCOPE("Mergeing - Collect stats");
		    collectMergeStatsRec(input, nodes_h, input.root_index, 0, hs, 0.f);
		    nodes_h[input.root_index].serialized_position = 0;
		}

		{// PROFILE_SCOPE("Mergeing - Create hash tables");
			// Put all nodes in the hash table 
			for (uint32_t i = 0; i < hs.order.size(); ++i) {
				uint32_t node_index = hs.order[i];
				const node_t& node = input.nodes[node_index];
				const merge_info_t& node_h = nodes_h[node_index];

				// Disregard leaves and nodes above the mergeable levels
				if (node_h.subtree_height >= 1 && input.levels - node_h.level <= mergelevels) {
					if (hs.existing_keys[node_h.subtree_height].find(node_h.shape_hash) != hs.existing_keys[node_h.subtree_height].end())
					{
						if (hs.hash_keys[node_h.subtree_height].find(node_h.shape_hash) != hs.hash_keys[node_h.subtree_height].end()) {
							std::vector<uint32_t>& collision_list = hs.hash_tables[node_h.subtree_height][node_h.shape_hash];
							collision_list.push_back(node_index);
						}
						hs.hash_keys[node_h.subtree_height].insert(node_h.shape_hash);
					}
					hs.existing_keys[node_h.subtree_height][node_h.shape_hash] = true;
				}
			}
		}
		// hs.order.clear();
	}

	void
	collectMergeStatsRec(SerializationInput& input, 
	                      std::vector<merge_info_t>& nodes_h, 
						  uint32_t current_index, 
						  uint32_t current_level, 
						  merge_structs_t& hs, 
						  float ancestor_depth)
	{

		// Save this index in the ordered list (depth first order)
		hs.node_qty++;

		node_t& node = input.nodes[current_index];
		merge_info_t& node_h = nodes_h[current_index];

		// Initialize node merge stats
		node_h.subtree_size = 0;
		node_h.subtree_height = 0;
		node_h.shape_hash = 1;
		node_h.merged = false;
		node_h.merger = false;
		node_h.main_merger = false;
		node_h.level = current_level;

		node_h.subtree_all_max = 0.f;
		node_h.subtree_all_min = 1.f;

		node_h.subtree_any_max = 1.f;
		node_h.subtree_any_min = 0.f;

		uint32_t current_serialized_size = uint32_t(hs.serialized_size);

		float absolute_node_value = node.value;

		if (input.value_types == VT_RELATIVE) 
		{
		// For all relative nodes
			node.max -= ancestor_depth;
			node.min -= ancestor_depth;
			node.value -= ancestor_depth;
		}
		
		if (node.flags & LEAF_BIT) {

			return;
		}

		if (!(node.flags & EMPTY_BIT))
		{
			ancestor_depth = absolute_node_value;
		}


		uint32_t childrenpos = current_serialized_size;
		hs.max_level_offsets[current_level] = std::max(childrenpos - node_h.serialized_position, hs.max_level_offsets[current_level]);

		for (int i = 0; i < 4; ++i)
		{
			node_h.shape_hash <<= 2;

			if (node.childrenList[i] < 0)
				continue;

			// Add children to order list (this is done before the recursion because all these 
			//  children go together)
			hs.order.push_back(node.childrenList[i]);

			node_t& child = input.nodes[node.childrenList[i]];
			merge_info_t& child_h = nodes_h[node.childrenList[i]];

			// Save shape_hash initially from children flags
			if (child.flags & LEAF_BIT) {
				node_h.shape_hash |= NODE_LEAF;
			} else if (child.flags & EMPTY_BIT) {
				node_h.shape_hash |= NODE_EMPTY;
			} else {
				node_h.shape_hash |= NODE_INNER;
			}			

			// Save final serialized position of this node
			child_h.serialized_position = childrenpos;
			childrenpos += getNodeSize(child.flags);
		}
		hs.serialized_size = childrenpos;

		// Collect stats from children and update stats
		for (int i = 0; i < 4; ++i)
		{
			if (node.childrenList[i] < 0)
				continue;

			node_t& child = input.nodes[node.childrenList[i]];
			merge_info_t& child_h = nodes_h[node.childrenList[i]];

			collectMergeStatsRec(input, nodes_h, node.childrenList[i], current_level + 1, hs, ancestor_depth);

			node_h.subtree_size += getSerializedNodeSize(child.flags) + child_h.subtree_size;

			//uint32_t shift_qty = ((child.hstats.subtree_height * 8) + i) % (8*(sizeof(shape_hash_t)-1));
			//node.hstats.shape_hash ^= (child.hstats.shape_hash + child.hstats.subtree_size) << (shift_qty);
			//node.hstats.shape_hash += child.hstats.subtree_size << (i + shift_qty);

			//node.hstats.shape_hash += child.flags << i;
			node_h.shape_hash += child_h.subtree_size << i;
			node_h.shape_hash *= child_h.shape_hash << i;

			node_h.subtree_height = std::max(node_h.subtree_height, uint8_t(child_h.subtree_height + 1));

			node_h.subtree_all_max = std::max(node_h.subtree_all_max, child_h.subtree_all_max);
			node_h.subtree_all_min = std::min(node_h.subtree_all_min, child_h.subtree_all_min);
			if (!(child.flags & EMPTY_BIT)) {
				node_h.subtree_all_max = std::max(node_h.subtree_all_max, child.max);
				node_h.subtree_all_min = std::min(node_h.subtree_all_min, child.min);
			}

			node_h.subtree_any_max = std::min(node_h.subtree_any_max, child_h.subtree_any_max);
			node_h.subtree_any_min = std::max(node_h.subtree_any_min, child_h.subtree_any_min);
			if (!(child.flags & EMPTY_BIT)) {
				node_h.subtree_any_max = std::min(node_h.subtree_any_max, child.max);
				node_h.subtree_any_min = std::max(node_h.subtree_any_min, child.min);
			}

		}

	}

	void
	computeMergedSerializedOffsets(std::vector<node_t>& nodes, 
	                                 std::vector<merge_info_t>& nodes_h,
									 uint32_t current_index,
									 uint32_t current_serialized_pos,
									 uint32_t& current_serialization_size)
	{
		node_t& node = nodes[current_index];
		merge_info_t& node_h = nodes_h[current_index];
		uint8_t node_size = getNodeSize(node.flags);

		const uint32_t starting_serialization_size = current_serialization_size;

		if ((node.flags & LEAF_BIT) || node_h.merged)
			return;

		node.children_offset = (current_serialization_size - current_serialized_pos) / 4 /*alignment*/;

		uint32_t childrensize = 0;
		uint32_t childpos[4];

		for (int i = 0; i < 4; ++i)
		{
			int32_t childindex = node.childrenList[i];
			childpos[i] = childrensize;

			if (childindex < 0)
				continue;

			const node_t& child = nodes[childindex];
			childrensize += getNodeSize(child.flags);
		}
		current_serialization_size += childrensize;

		for (int i = 0; i < 4; ++i)
		{
			int32_t childIndex = node.childrenList[i];

			if (childIndex < 0)
				continue;

			computeMergedSerializedOffsets(nodes, nodes_h, childIndex, starting_serialization_size + childpos[i], current_serialization_size);
		}
	}

	void
	fillSerializedMergedNodeBuffer(const std::vector<node_t>& nodes, 
	                                 const std::vector<merge_info_t>& nodes_h,
									 std::vector<uint8_t>& nodebuffer, 
									 const std::vector<uint32_t>& scales, 
									 const uint32_t index,
									 const uint32_t inode, 
									 const uint32_t level)
	{

		const node_t& node = nodes[index];
		const merge_info_t& node_h = nodes_h[index];

		uint8_t flags = 0;
		for (uint32_t i = 0; i < 4; ++i)
		{
			uint8_t childflags = 0;
			if (node.childrenList[i] != -1)
			{
				const node_t& child = nodes[node.childrenList[i]];
				if (child.flags & LEAF_BIT) {
					childflags = NODE_LEAF;
				} else if (child.flags & EMPTY_BIT) {
					childflags = NODE_EMPTY;
				} else {
					childflags = NODE_INNER;
				}
			}
			flags |= childflags << (2 * i);
		}

		// Save the current end of the output byte buffer as the new location for any children, then allocate children memory
		uint32_t childptr = node.children_offset;

		if (node_h.merged)
		{
			const node_t& merger = nodes[node_h.merge_node];
			const merge_info_t& merger_h = nodes_h[node_h.merge_node];

			uint32_t merger_offset = (merger_h.serialized_position - node_h.serialized_position) / 4 /*alignment*/;
			childptr = merger_offset + merger.children_offset;

			Verify(("Error: child offset generated is too big", childptr < 1 << (ptrbits-1)));

			if (merger_h.serialized_position <= node_h.serialized_position)
			{
				std::cout << "Error! merger before merged" << index << std::endl;
			}

		}

		uint32_t scaledptr = childptr / scales[level];
		fillINode(node, nodebuffer, inode, flags, scaledptr);

		// If the current node is not a leaf node or merged, recursively process children
		if (!( (node.flags & LEAF_BIT) || node_h.merged) )
		{
			uint32_t offset = childptr * 4 /*alignment*/;
			for (uint32_t i = 0; i < 4; ++i)
			{
				uint8_t childflags = (flags >> (i * 2)) & 3;
				if (childflags > 0)
				{
					fillSerializedMergedNodeBuffer(nodes, nodes_h, nodebuffer, scales, node.childrenList[i], inode + offset, level + 1);
					offset += (childflags == NODE_INNER ? 8 : 4);
				}
			}
		}
	}

	void
	createSerializedMergedNodeBuffer(SerializationInput& input,
									 SerializationOutput& output)
	{
		std::vector<merge_info_t> nodes_h(input.nodes.size());
		
		uint32_t serialized_size = getNodeSize(input.nodes[input.root_index].flags);

		// Mark merged nodes
		{ 
			markMergedNodes(input, nodes_h, output);
		}

		{ PROFILE_SCOPE("Mergeing - Serialization");
			{ //PROFILE_SCOPE("Mergeing - Computing final offsets");
				// Go through the tree and compute the original offsets
				computeMergedSerializedOffsets(input.nodes, nodes_h, input.root_index, 0, serialized_size);
			}

			{ //PROFILE_SCOPE("Mergeing - Computing scales");
				output.levelscales = std::vector<uint32_t>(input.levels, 1);
				// Identify scales and add padding
				{
					std::vector<uint32_t> offsets;
					std::vector<uint32_t>& bestscales = output.levelscales;
					bestscales.resize(input.levels, 1);
					for (int i = input.levels - 2; i >= 0; --i)
					{
						//
						//std::cout << std::endl << "/// Processing level " << i << " ///" << std::endl;
						//

						offsets.clear();
						getMergedLevelOffsets(input.nodes, nodes_h, offsets, i, input.root_index);

						uint32_t best_scale;
						best_scale = computeBestLevelScale(offsets);
						bestscales[i] = best_scale;
						// std::cout << "Best scale: " << best_scale << std::endl;
						if (best_scale == 1)
							continue;

						uint32_t new_padding = 0;

						if (input.levels - i <= mergelevels)
						{
							std::cout << "Error: Scaling in mergeable level!" << std::endl;
						}

						padMergedLevel(input.nodes, nodes_h, best_scale, i, input.root_index, new_padding);
						//
						// std::cout << "Created padding: " << new_padding << std::endl;
						//
					}
				}
			}

			serialized_size = getNodeSize(input.nodes[input.root_index].flags);
			uint32_t merged_size = 0;
			{ // PROFILE_SCOPE("Mergeing - Compute Size");
				getMergedNodeArraySize(input.nodes, nodes_h, input.root_index, serialized_size, merged_size, output.stats);
			}

			output.nodebuffer.resize(serialized_size, 0);

			// Create serialized structure
			{ // PROFILE_SCOPE("Mergeing - Serialize");
				fillSerializedMergedNodeBuffer(input.nodes, nodes_h, output.nodebuffer, output.levelscales, input.root_index);
			}
		}

		// Traverse tree to check integrity
		if (false)
		{ //PROFILE_SCOPE("Mergeing - Integrity");
			size_t count = 0;
			traverseMergedSerializedTree(output.nodebuffer, output.levelscales, 0, false, false, 0, count);
			std::cout << "Count: " << count << std::endl;
		}

	}



	uint32_t
	padMergedLevel(std::vector<node_t>& nodes, 
	                 const std::vector<merge_info_t>& nodes_h,
	                 const uint32_t best_scale, 
					 const uint32_t level, 
					 const uint32_t index, 
					 uint32_t& total_padding,
					 uint32_t  padding)
	{
		node_t& node = nodes[index];
		const merge_info_t& node_h = nodes_h[index]; 

		if ((node.flags & LEAF_BIT) || node_h.merged) {
			return padding;
		}

		node.children_offset += padding;

		uint32_t children_padding = 0;
		if (level) {
			for (int i = 0; i < 4; ++i)
			{
				int32_t childIndex = node.childrenList[i];

				if (childIndex < 0)
				{
					continue;
				}

				children_padding = padMergedLevel(nodes, nodes_h, best_scale, level - 1, childIndex, total_padding, children_padding);
			}
			return children_padding + padding;
		} else {
			uint32_t offset = node.children_offset;
			uint32_t new_offset = uint32_t(ceil(offset / (float)best_scale));
			uint32_t new_padding = new_offset * best_scale - offset;
			padding += new_padding;
			total_padding += new_padding;

			node.children_offset = best_scale * new_offset;

			return padding;
		}

	}

	void
	getMergedLevelOffsets(const std::vector<node_t>& nodes,
	                        const std::vector<merge_info_t>& nodes_h,
							std::vector<uint32_t>& offsets,
							uint32_t level,
							uint32_t index) 
	{
		const node_t& node = nodes[index];
		const merge_info_t& node_h = nodes_h[index];


		if (node_h.merged)
			return;

		//////// Cannot be a leaf!

		if (level == 0)
		{
			// If this is the desired level, output the offset
			offsets.push_back(node.children_offset);
		}
		else {
			// Else go to the children recursively
			for (int i = 0; i < 4; ++i)
			{
				int32_t childIndex = node.childrenList[i];

				if (childIndex < 0 || (nodes[childIndex].flags & LEAF_BIT) || (nodes_h[childIndex].merged)){
					// If next level would be the last, it has offset 0 to children (no children)
					if (level == 1)
						offsets.push_back(0);

					continue;
				}

				getMergedLevelOffsets(nodes, nodes_h, offsets, level - 1, childIndex);
			}
		}
	}

	void
	getMergedNodeArraySize(const std::vector<node_t>& nodes, 
	                         std::vector<merge_info_t>& nodes_h,
							 uint32_t index, 
							 uint32_t& serialized_size, 
							 uint32_t& merged_size, 
							 serializationStatistics& stats, 
							 uint32_t current_pos, 
							 uint32_t level)
	{
		const node_t& node = nodes[index];
		merge_info_t& node_h = nodes_h[index];
		node_h.serialized_position = current_pos;

		serialized_size = std::max(serialized_size, current_pos + getNodeSize(node.flags));

		stats.node_qty++;

		if (node.flags & EMPTY_BIT) {
			stats.empty_inner_qty++;
		} else if (node.flags & LEAF_BIT) {
			stats.leaf_qty++;
		} else {
			stats.full_inner_qty++;
		}


		if ((node.flags & LEAF_BIT) || node_h.merged) {
			if (node_h.merged)
				merged_size += node_h.subtree_size;
			return;
		}

		uint32_t childpos = current_pos + node.children_offset * 4 /*alignment*/;
		for (int i = 0; i < 4; ++i)
		{
			int32_t childindex = node.childrenList[i];

			if (childindex < 0)
				continue;

			uint32_t childFlags = nodes[childindex].flags;

			getMergedNodeArraySize(nodes, nodes_h, childindex, serialized_size, merged_size, stats, childpos, level + 1);
			childpos += getNodeSize(childFlags);
		}
	}

	void
	traverseMergedSerializedTree(const std::vector<uint8_t>& nodebuffer, const std::vector<uint32_t>& scales, const uint32_t inode, const bool isleaf, const bool isempty, uint32_t level, size_t& count)
	{
		count++;

		// If this is the leaf, just return
		if (isleaf)
		{
			return;
		}

		// Read flags and pointer
		const uint8_t flags = nodebuffer[inode];

		union pointerUnion
		{
			uint32_t ptr;
			uint8_t bytes[3];
		} pointer;

		pointer.ptr = *((uint32_t*)(nodebuffer.data() + inode));
		pointer.ptr = (pointer.ptr >> 8) & 0xffffff;

		// Process children
		uint32_t childnode = inode + (pointer.ptr * scales[level]) * 4 /*alignment*/;
		for (uint32_t i = 0; i < 4; ++i)
		{
			uint8_t childflags = (flags >> (i * 2)) & 3;
			if (childflags == NODE_INNER)
				traverseMergedSerializedTree(nodebuffer, scales, childnode, false, false, level + 1, count);
			else if (childflags == NODE_EMPTY)
				traverseMergedSerializedTree(nodebuffer, scales, childnode, false, true, level + 1, count);
			else if (childflags == NODE_LEAF)
				traverseMergedSerializedTree(nodebuffer, scales, childnode, true, false, level + 1, count);

			childnode += getSerializedNodeSize(childflags);
		}
	}


};


