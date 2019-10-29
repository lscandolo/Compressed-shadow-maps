struct partial_search
{
	int32_t nodeptr;
	float    value;
	level_t  level;
	uint8_t  flags;
};

////////////////////// ALIGNED TO 4

///////////////////////////////////////////////////////////

float getDepthValue(const ivec2 queryPoint,
				    const level_t lodlevel)
	{
		partial_search search;
		search.level = level_t(0);
		search.nodeptr = int32_t(0);
		search.value = getQuadtreeValue(1);
		search.flags = NODE_INNER;

		while (true)
		{
			// Read flags and pointer
			uint32_t flagsAndOffset = getQuadtreeFlagsAndOffet(search.nodeptr);
			uint8_t flags = uint8_t(flagsAndOffset&0xff); // & 0xff (but this is not needed)
			uint32_t offset = (flagsAndOffset >> 8) & 0xffffff;

			// Apply scaling to offset
			scaleOffset(offset, search.level);

			// Get next child index
			uint8_t nextchild = getNextChild(queryPoint, search.level);

			// Try to traverse the next child (if it exists)
			uint8_t childflags = (flags >> (nextchild << uint8_t(1))) & uint8_t(3);

			if (is_nonexistantnode(childflags) || (maxTreeLevel_8 - search.level) <= lodlevel)
			{
				return search.value;
			}

			// Count memory up to that child using bit counting
			if (nextchild != uint8_t(0)) {
				flags &= (uint8_t(1) << (nextchild << uint8_t(1))) - uint8_t(1);	// Mask away all bits for the next child and the children after (e.g. if nextchild is 3rd child (=2), perform and with 0b00001111)
				offset += uint32_t(numberOfSetBits(flags));	// Simply add the number of bits to the offset (1 for leafs and empty nodes, 2 for inner nodes)
			}

			// Setup traversal for child node
			search.level++;
			search.nodeptr += int32_t(offset);
			search.flags = childflags;

			// If this is the leaf, just read the value and return
			if (is_leafnode(search.flags))
			{
				search.value += getQuadtreeValue(search.nodeptr);
				return search.value;
			}

			// If the node is not empty, store the position of the value
			if (is_innernode(search.flags)) {
				search.value += getQuadtreeValue(search.nodeptr + int32_t(1));
			}
			
		}

		return -1;	// Should never be reached
	}



////////////////////////////////////////////////////////////////////////////////////////////////////

	bool getCommonDepthValue(inout partial_search search,
							 const ivec2 queryPoint,
							 const level_t lastCommonLevel)
	{

		while(search.level < lastCommonLevel) 
		{

			// Read flags and pointer
			uint32_t flagsAndOffset = getQuadtreeFlagsAndOffet(search.nodeptr);
			uint8_t flags = uint8_t(flagsAndOffset&0xff); // & 0xff (but this is not needed)
			uint32_t offset = (flagsAndOffset >> 8) & 0xffffff;
			
			// Apply scaling to offset
			scaleOffset(offset, search.level);
		
			// Get next child index
			uint8_t nextchild = getNextChild(queryPoint, search.level);

			// Try to traverse the next child (if it exists)
			uint8_t childflags = (flags >> (nextchild << uint8_t(1))) & uint8_t(3);
			if (is_nonexistantnode(childflags))
			{
				// Return value from the last stored position
				return true;
			}
			
			// Count memory up to that child using bit counting
			if (nextchild != uint8_t(0)) {
				flags &= (uint8_t(1) << (nextchild << uint8_t(1))) - uint8_t(1);	// Mask away all bits for the next child and the children after (e.g. if nextchild is 3rd child (=2), perform and with 0b00001111)
				offset += uint32_t(numberOfSetBits(flags));	// Simply add the number of bits to the offset (1 for leafs and empty nodes, 2 for inner nodes)
			}

			// Setup traversal for child node
			search.level++;
			search.nodeptr += int32_t(offset);
			search.flags = childflags;

			bool isleaf  = is_leafnode(childflags);
			bool isinner = is_innernode(childflags);

			if (isleaf) {
				search.value += getQuadtreeValue(search.nodeptr);
				return true;
			}
			
			if (isinner) {
				search.value += getQuadtreeValue(search.nodeptr + 1);
			}

		}
	
		return false;	// We reached the last common level

	}

////////////////////////////////////////////////////////////////////////////////////////////////////
vec2 getDepthValueHierarchical(inout partial_search search,
							   const ivec2 queryPoint,
							   const uint8_t lodlevel)
	{

		float ceilVal;

		while(true) 
		{
		bool floorLevel = maxTreeLevel_8 - search.level + uint8_t(2) == lodlevel;
		bool ceilLevel  = maxTreeLevel_8 - search.level + uint8_t(1) == lodlevel;
		bool isempty = is_emptynode(search.flags);
	
		if (ceilLevel) {
			ceilVal = search.value;
		} else if (floorLevel) {
			float floorVal = isempty ? search.value : ceilVal;
			return vec2(ceilVal, floorVal);
		}

		// Read flags and offset
		uint32_t flagsAndOffset = getQuadtreeFlagsAndOffet(search.nodeptr);
		uint8_t flags           = uint8_t(flagsAndOffset&0xff); // & 0xff (but this is not needed)
		uint32_t offset         = (flagsAndOffset >> 8) & 0xffffff; 
			
		// Apply scaling to offset
		scaleOffset(offset, search.level);
		
		// Get next child index
		uint8_t nextchild = getNextChild(queryPoint, search.level);
		
		// Try to traverse the next child (if it exists)
		uint8_t childflags = nextchild!=uint8_t(0) ? (flags >> (nextchild << uint8_t(1))) & uint8_t(3) : flags & uint8_t(3);
		
		if (is_nonexistantnode(childflags))
		{
			// Return value from the last stored position
			float floorVal = search.value;
			
			if (!floorLevel)
				ceilVal = floorVal;

			return vec2(ceilVal, floorVal);
		}
			
		// Count memory up to that child using bit counting (if it's not the first node)
		if (nextchild!=uint8_t(0)) {
			flags &= (uint8_t(1) << (nextchild << uint8_t(1))) - uint8_t(1);	// Mask away all bits for the next child and the children after (e.g. if nextchild is 3rd child (=2), perform and with 0b00001111)
			offset += uint32_t(numberOfSetBits(flags));	// Simply add the number of bits to the offset (1 for leafs and empty nodes, 2 for inner nodes)
		}

		// Setup traversal for child node
		search.level++;
		search.nodeptr += int32_t(offset);
		search.flags    = childflags;

		// If this is the leaf, just read the value and return
		if (is_leafnode(childflags))
		{
			float floorVal = search.value + getQuadtreeValue(search.nodeptr);

			if (!floorLevel)
				ceilVal = floorVal;

			return vec2(ceilVal, floorVal);
		} 
			
		if (is_innernode(childflags)) 
		{
			// If the node is not empty, store the position of the value
			search.value += getQuadtreeValue(search.nodeptr + 1);
		}
			
	}
	
	return vec2(0.f, 0.f);	// Should never be reached


}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float evaluateQuadtreeHierarchicalPCF(vec4 coord,
									  const float bias,  uint8_t pcf, 
									  float lodlevel, const float pixelsize)
{

	// Compute query coordinate in depth map space and the get depth value of scene point point
	const uint32_t resolution = 1 << maxTreeLevel;
	const vec2  queryCoords = vec2(coord.x * resolution, coord.y * resolution);
	const ivec2   iQueryCoords = ivec2(queryCoords.x, queryCoords.y);
	const float pointDepth = coord.z;


	// Compute minimum and maximum query point of the kernel
	// Ksize is the size of a pixel, clamped between 1 and pow(2, flodlevel) 
	//  because flodlevel was clamped but fpixelsize was not
	const float ksize = min( max(pixelsize, 1.f), float(1 << int(lodlevel+1)));
	const int pcf_ksize = int(pcf * ksize);
		
	const ivec2 mincoords = iQueryCoords - ivec2(pcf_ksize, pcf_ksize);
	const ivec2 maxcoords = iQueryCoords + ivec2(pcf_ksize, pcf_ksize);
		
	// Return 0 if kernel size goes out of bounds
	if (mincoords.x < 0 || mincoords.y < 0 || maxcoords.x >= depthMapResolution || maxcoords.y >= depthMapResolution)
		return 0.0;
	 
	// Get last level for which traversal is the same for all points in the kernel
	const level_t minPossibleLevel = uint8_t(32) - maxTreeLevel_8;
	level_t lastCommonLevel = getLastCommonLevel(mincoords, maxcoords, minPossibleLevel);


	// Traverse down the tree as long as the PCF kernel is still covered by a single node
	partial_search search;
	search.level = uint8_t(0);
	search.nodeptr = 0;
	search.value = getQuadtreeValue(1);
	search.flags = NODE_INNER;

	bool hasOnlyOneValue = getCommonDepthValue(search, iQueryCoords, lastCommonLevel);
	if (hasOnlyOneValue)
	{
		if (pointDepth >= search.value + bias)
			return 1.f;
		else 
			return 0.f;
	}

	float shadow = 0.f;

	ivec2 prevcoords = iQueryCoords + ivec2(-pcf * ksize, -pcf * ksize);
	vec2 lightDepth;
	partial_search pointsearch;

	// Fractional part of the computed lod level
	const float fractLodLevel = (lodlevel - floor(lodlevel));

	float stdMult = 1.f / float(2*float(pcf)+1);
	float pcfMult = 1.f / float(pcf*pcf);
	pointsearch.level = maxTreeLevel_8 + uint8_t(1);

	for (int8_t x = -int8_t(pcf); x <= int8_t(pcf); ++x) {
		for (int8_t y = -int8_t(pcf); y <= int8_t(pcf); ++y) {
				
			vec2 nextcoords =  queryCoords +  vec2(x, y) * float(ksize);
			ivec2   iNextcoords = ivec2(nextcoords.x, nextcoords.y);
			iNextcoords =  ivec2(iQueryCoords +  vec2(x, y) * ksize);

			float xVal = stdMult;
			float yVal = stdMult;
			if (lodlevel == 0) {
				xVal = pcfMult * max(0.f, pcf * ksize - abs(queryCoords.x - iNextcoords.x));
				yVal = pcfMult * max(0.f, pcf * ksize - abs(queryCoords.y - iNextcoords.y));
			}
				
			lastCommonLevel = getLastCommonLevel(prevcoords, iNextcoords, minPossibleLevel);

			// It's not '>' because getDepthValueHierarchical also looks at the previous level to do trilinear interpolation
			if (pointsearch.level >= lastCommonLevel) {
				pointsearch = search;
				prevcoords = iNextcoords;
				lightDepth = getDepthValueHierarchical(pointsearch, iNextcoords,uint8_t(lodlevel));

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

float evaluateShadow(vec4 lightCoords, float bias, uint8_t lodlevel)
{
	float shadow = 0.0;
	ivec2 queryCoords = ivec2(lightCoords.xy * float(depthMapResolution) + vec2(0.5)) ;
	float pointDepth = lightCoords.z;
	
	// Lastvalue points to 1 initially which is the value position of the root node at pos=0
	float lightDepth = getDepthValue(queryCoords, lodlevel);
	if (pointDepth < lightDepth + bias)
		shadow = 1.0;

	return 1.0-shadow;
}
