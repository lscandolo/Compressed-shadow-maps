#include "common.h"


#include "csm/QuadtreeTypes.h"
#include "csm/QuadtreeSerialization.h"
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
	
	void
	setInitialNodeOffsets(SerializationInput& input, const uint32_t index, uint32_t& currentSize, float ancestor_depth, uint32_t currentPos)
	{
		node_t& node = input.nodes[index];


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

		if (node.flags & LEAF_BIT)
			return;

		if (!(node.flags & EMPTY_BIT))
		{
			ancestor_depth = absolute_node_value;
		}

		const uint32_t startsize = currentSize;
			node.children_offset = (currentSize - currentPos) / 4 /*alignment*/;

			uint32_t childrensize = 0;
			uint32_t childpos[4];

			for (int i = 0; i < 4; ++i)
			{
				int32_t childindex = node.childrenList[i];
				childpos[i] = childrensize;

				if (childindex < 0)
					continue;

				const node_t& child = input.nodes[childindex];
				childrensize += getNodeSize(child.flags);
			}
			currentSize += childrensize;

			for (int i = 0; i < 4; ++i)
			{
				int32_t childIndex = node.childrenList[i];

				if (childIndex < 0)
					continue;

				setInitialNodeOffsets(input, childIndex, currentSize, ancestor_depth, startsize + childpos[i]);
			}
	}


	void
	getLevelOffsets(const std::vector<node_t>& nodes, std::vector<uint32_t>& offsets, uint32_t level, uint32_t index) 
	{
		const node_t& node = nodes[index];


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

				if (childIndex < 0 || (nodes[childIndex].flags & LEAF_BIT)){
					// If next level would be the last, it has offset 0 to children (no children)
					if (level == 1)
						offsets.push_back(0);

					continue;
				}

				getLevelOffsets(nodes, offsets, level - 1, childIndex);
			}
		}
	}

	uint32_t
	computeBestLevelScale(const std::vector<uint32_t>& offsets) 
	{
			// Compute maximum difference
			uint32_t maxoff = 0;
			for (size_t i = 1; i < offsets.size(); ++i)
				maxoff = std::max(maxoff, offsets[i]);

			// Compute range 
			const uint32_t range = (1 << ptrbits) - 1;

			if (maxoff < range) {
				//std::cout << "bestscale = 1 (padding = 0)" << std::endl;
				return 1;
			}

			uint32_t bestscale = 0;
			uint32_t smallest_padding = std::numeric_limits<uint32_t>::max();

			// Find the smallest scale that is able to reach the maximum offset
			uint32_t minscale = uint32_t(ceil(maxoff / float(range)));
			const uint32_t maxscale = std::min(maxoff, minscale + 15);
			//const uint32_t maxscale = maxoff;
			//std::cout << "minscale=" << minscale << ", maxscale=" << maxscale << std::endl;
			//std::cout << "offsets.size(): " << offsets.size() << std::endl;

			// Try all scales
			for (uint32_t scale = minscale; scale <= maxscale; scale += 1)
			{
				uint32_t padding = 0;
				for (size_t i = 0; i < offsets.size(); i += 4)
				{
					uint32_t local_padding = 0;
					for (size_t j = 0; j < 4; ++j)
					{


						if (!offsets[i + j]) {
							continue;
						}


						// Check if the offset can be represented, i.e. it is a integral multiple in of the scale (in the range [0,255])
						// (note that it is ensured that the maximum offset can be reached)
						const uint32_t off = offsets[i + j] + local_padding;

						// If the padding makes it unreachable, then bail
						if (off > scale * range)
						{
							padding = std::numeric_limits<uint32_t>::max();
							i = offsets.size();
							break;
						}

						uint32_t newoff = uint32_t(ceil(off / (float)scale));
						uint32_t new_padding = newoff * scale - off;
						local_padding += new_padding;
						padding += new_padding;

					}

					if (padding > smallest_padding)
					{
						break;
					}

				}

				if (padding < smallest_padding)
				{
					smallest_padding = padding;
					bestscale = scale;
				}
			}

			Verify(("Error: scale type isn't big enough to hold best scale during serialization", pow(2,sizeof(scale_t) * 8) > bestscale));
			
			return bestscale;
		}


	uint32_t
	padLevel(std::vector<node_t>& nodes, uint32_t best_scale, uint32_t level, uint32_t index, uint32_t& total_padding, uint32_t padding)
	{
		node_t& node = nodes[index];

		if (node.flags & LEAF_BIT) {
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

				children_padding = padLevel(nodes, best_scale, level - 1, childIndex, total_padding, children_padding);
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
	getNodeArraySize(std::vector<node_t>& nodes, uint32_t index, uint32_t& size, uint32_t current_pos, uint32_t level)
	{
		node_t& node = nodes[index];

		size = std::max(size, current_pos + getNodeSize(node.flags));


		if (node.flags & LEAF_BIT)
			return;

		uint32_t childpos = current_pos + node.children_offset * 4 /*alignment*/;
		for (int i = 0; i < 4; ++i)
		{
			int32_t childindex = node.childrenList[i];

			if (childindex < 0)
				continue;

			uint32_t childFlags = nodes[childindex].flags;

			getNodeArraySize(nodes, childindex, size, childpos, level + 1);
			childpos += getNodeSize(childFlags);
		}
	}

	void
	fillINode(const node_t& node, std::vector<uint8_t>& nodebuffer, uint32_t inode, const uint8_t flags, const uint32_t childptr)
	{
#pragma pack(push, 1)
		struct LeafStruct
		{
			float   value;
		} leaf;

		struct EmptyStruct
		{
			uint8_t flags;
			uint16_t childrenoffset;
			uint8_t  pad;
		} empty;

		struct FullStruct
		{
			uint8_t flags;
			uint16_t childrenoffset;
			uint8_t pad;
			float   value;
		} full;
#pragma pack(pop)

		if (node.flags & LEAF_BIT)
		{
			leaf.value = node.value;
			LeafStruct* ptr = (LeafStruct*)(nodebuffer.data() + inode);
			*ptr = leaf;
		} else if (node.flags & EMPTY_BIT) {
			empty.flags = flags;
			empty.childrenoffset = childptr;
			empty.pad = childptr >> 16;
			EmptyStruct* ptr = (EmptyStruct*)(nodebuffer.data() + inode);
			*ptr = empty;
		} else {
			full.flags = flags;
			full.childrenoffset = childptr;
			full.pad = childptr >> 16;
			full.value = node.value;
			FullStruct* ptr = (FullStruct*)(nodebuffer.data() + inode);
			*ptr = full;
		}

	}

	void
	fillSerializedNodeBuffer(std::vector<node_t>& nodes, std::vector<uint8_t>& nodebuffer, const std::vector<uint32_t>& scales, uint32_t index, uint32_t inode, uint32_t level)
	{
		node_t& node = nodes[index];


		/*
			 * Create new flags defining the types of the children. A node can have 3 different states
			 * or it does not exist. This can be encoded efficiently for all for children using 2 bits
			 * per children leading to a overall 8 bit mask for each inner node.
			 *
			 *		00 - Child node does not exist
			 *		01 - Inner node
			 *		10 - Empty inner node
			 *		11 - Leaf node
			 *
			 */

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
		uint32_t scaledptr = childptr / scales[level];

		fillINode(node, nodebuffer, inode, flags, scaledptr);

		// If the current node is not a leaf node, recursively process children
		if (!(node.flags & LEAF_BIT))
		{
			uint32_t offset = childptr * 4 /*alignment*/;
			for (uint32_t i = 0; i < 4; ++i)
			{
				uint8_t childflags = (flags >> (i * 2)) & 3;
				if (childflags > 0)
				{
					fillSerializedNodeBuffer(nodes, nodebuffer, scales, node.childrenList[i], inode + offset, level + 1);
					offset += (childflags == NODE_INNER ? 8 : 4);
				}
			}
		}
	}




	static void
	traverseSerializedTree(const std::vector<uint8_t>& nodebuffer, const std::vector<uint32_t>& scales, const uint32_t inode, const bool isleaf, const bool isempty, uint32_t level, size_t& count)
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
				traverseSerializedTree(nodebuffer, scales, childnode, false, false, level + 1, count);
			else if (childflags == NODE_EMPTY)
				traverseSerializedTree(nodebuffer, scales, childnode, false, true, level + 1, count);
			else if (childflags == NODE_LEAF)
				traverseSerializedTree(nodebuffer, scales, childnode, true, false, level + 1, count);

			childnode += getSerializedNodeSize(childflags);
		}
	}

	static void
	traverseOriginalTree(std::vector<node_t>& nodes, uint32_t index, serializationStatistics& stats, uint32_t level, bool markUsed)
	{
		node_t& node = nodes[index];

		if (markUsed)
			node.value_index = 1;

		stats.node_qty++;

		if (node.flags & LEAF_BIT) {
			stats.leaf_qty++;
			return;
		}

		if (node.flags & EMPTY_BIT) {
			stats.empty_inner_qty++;
		} else {
			stats.full_inner_qty++;
		}


		for (int i = 0; i < 4; ++i)
		{
			if (node.childrenList[i] != -1)
				traverseOriginalTree(nodes, node.childrenList[i], stats, level - 1, markUsed);
		}

	}
	
	void
	createSerializedNodeBuffer(SerializationInput& input, SerializationOutput& output)
	{
		uint32_t size = getNodeSize(0); // Root size

		{PROFILE_SCOPE("Serialization-setInitialNodeOffsets");
		    setInitialNodeOffsets(input, input.root_index, size, 0, 0);
		}
		std::cout << "Initial size: " << size << std::endl;

		std::vector<uint32_t> offsets;
		std::vector<uint32_t>& bestscales = output.levelscales;
		bestscales.resize(input.levels, 1);
		for (int i = input.levels - 2; i >= 0; --i)
		{

			//
			//std::cout << std::endl << "/// Processing level " << i << " ///" << std::endl;
			//

			offsets.clear();
			{PROFILE_SCOPE("Serialization-getLevelOffsets");
			getLevelOffsets(input.nodes, offsets, i, input.root_index);
			}

			uint32_t best_scale;
			{PROFILE_SCOPE("Serialization-computeBestLevelScale");
			best_scale = computeBestLevelScale(offsets);
			}
			bestscales[i] = best_scale;
			// std::cout << "Best scale: " << best_scale << std::endl;
			if (best_scale == 1)
				continue;
			uint32_t new_padding = 0;


			{PROFILE_SCOPE("Serialization-padLevel");
			padLevel(input.nodes, best_scale, i, input.root_index, new_padding);
			}

			//
			std::cout << "Created padding: " << new_padding << std::endl;
			//
		}

		uint32_t new_size = 0;
		{PROFILE_SCOPE("Serialization-getNodeArraySize");
		getNodeArraySize(input.nodes, input.root_index, new_size);
		}
		std::cout << "Final padded size : " << new_size << std::endl;

		output.nodebuffer.resize(new_size, 0);

		{PROFILE_SCOPE("Serialization-fillSerializedNodeBuffer");
			fillSerializedNodeBuffer(input.nodes, output.nodebuffer, bestscales, input.root_index);
		}


		/////////////// Only for stats

		output.stats.total_time = 0;
		output.stats.offset_compute_time = 0;
		output.stats.node_output_time = 0;

		{ PROFILE_SCOPE("Serialization-traverseOriginalTree (stats)");
			traverseOriginalTree(input.nodes, input.root_index, output.stats, input.levels - 1, false);
		}

	}


};


