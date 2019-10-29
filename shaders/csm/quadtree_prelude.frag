uniform usamplerBuffer  nodesTex32;
uniform usamplerBuffer  scalesTex;
uniform mat4            light_vp;


uniform int             maxTreeLevel;
uniform int             depthMapResolution;

#ifdef GL_NV_gpu_shader5
#else
	#define uint32_t uint
	#define uint16_t uint
	#define uint8_t  uint

	#define int32_t int
	#define int16_t int
	#define int8_t  int
#endif

#define level_t uint8_t

const uint8_t maxTreeLevel_8 = uint8_t(maxTreeLevel);


#define NODE_NONEXIST 	(uint8_t(0))
#define NODE_LEAF 		(uint8_t(1))
#define NODE_EMPTY 		(uint8_t(2))
#define NODE_INNER 		(uint8_t(3))


bool is_nonexistantnode(const uint8_t flags)
{
	return (flags == NODE_NONEXIST);
}

bool is_emptynode(const uint8_t flags)
{
	return (flags == NODE_EMPTY);
}

bool is_leafnode(const uint8_t flags)
{
	return (flags == NODE_LEAF);
}

bool is_innernode(const uint8_t flags)
{
	return (flags == NODE_INNER);
}

uint numberOfSetBits(uint8_t i)
{
	return bitCount(i);
}

vec4 computeShadowNormalizedCoordinates(vec3 pos)
{
	vec4 lCoords = light_vp * vec4(pos, 1.0);
	lCoords *= 1.0 / lCoords.w;
	lCoords.xyz = lCoords.xyz * 0.5 + vec3(0.5);

	if (!light_is_dir) {
		lCoords.z = (2.0 * light_near) / (light_far + light_near - lCoords.z * (light_far - light_near));
	} 

	return lCoords;
}

vec2 computeShadowPixelCoordinates(vec3 pos)
{
	return computeShadowNormalizedCoordinates(pos).xy * depthMapResolution;
}

vec3 computelight_dir(vec3 pos)
{
	if (light_is_dir)
	{
		return normalize(light_dir);
	} else {
		return normalize(pos - light_spotpos);
	}
}

void scaleOffset(inout uint32_t offset, const level_t level)
{
	uint32_t scale = uint32_t(texelFetch(scalesTex, int(level)).x);
	if (scale > 1) offset *= scale;
}

uint8_t getNextChild(const ivec2 queryPoint, const uint8_t level)
{
	// Get next child index
	uint bitidx = uint((maxTreeLevel - uint(level)) - 1);
	uint bit = 1 << bitidx;
	uint nextchild = ((queryPoint.x & bit) | ((queryPoint.y & bit) << 1)) >> bitidx;
	return uint8_t(nextchild);
}

uint8_t clz(uint32_t x)
{
	if (x==0) return uint8_t(32);
	
	
	return uint8_t( 31 - findMSB(x) );
}

uint8_t getLastCommonLevel(const ivec2 c1, const ivec2 c2, const uint8_t minPossibleLevel)
{
	return max(uint8_t(minPossibleLevel), clz((c1.x^c2.x) | (c1.y^c2.y))) - minPossibleLevel;
}

float getQuadtreeValue(const int ptr)
{
	return uintBitsToFloat(texelFetch(nodesTex32, ptr).x);
}

uint32_t getQuadtreeFlagsAndOffet(const int ptr)
{
	return uint32_t(texelFetch(nodesTex32, ptr).x);
}