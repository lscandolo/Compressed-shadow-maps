#pragma once

#include "common.h"

#include "helpers/OpenGLHelpers.h"

#include <algorithm>
#include <map>
#include "csm/QuadtreeTypes.h"
#include "csm/Tiler.h"
#include "csm/CudaOpenGLResource.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <thrust/device_vector.h>


struct ShadowPrecomputeInput
{
	size_t width, height;
	CudaOpenGLResource* shadowTexResource;
	CudaOpenGLResource* worldPosTexResource;
	CudaOpenGLResource* worldDPosTexResource;
	CudaOpenGLResource* worldNorTexResource;
	glm::mat4   lightVPMatrix;
	glm::vec3   lightDir;
	glm::vec3   lightPos;
	float lightnearplane;
	float lightfarplane;
	bool  lightisdirectional;
	bool    pcf_hierarchical;
	uint8_t pcf_size;
	float   depth_bias;
};


struct CompressionConfig
{
	glm::ivec3          tile_qty;
	uint32_t            resolution;
	uint32_t            tile_resolution;
	Quadtree::ValueType values_type;
	bool                merged;
};

struct TileCompressionInput
{
	Tiler::TileParameters      tile_params;
	GLHelpers::TextureObject2D front;
	GLHelpers::TextureObject2D back;
};

struct CompressionStats
{
	size_t orig_size;
	size_t compressed_size;
	size_t leaf_node_qty;
	size_t inner_node_qty;
	size_t empty_node_qty;
};

using  tile_coord_t = glm::uvec2;
struct tile_comp { bool operator() (const tile_coord_t& lhs, const tile_coord_t& rhs) const { return lhs.x == rhs.x ? lhs.y < rhs.y : lhs.x < rhs.x; } };
using  tiles_t = std::map<tile_coord_t, Quadtree::root_extra_t, tile_comp>;


class QuadtreeGPUCompression
{
public:
	///////////////////////////////////////////////////////////////
	#define INNER_BIT (0x0)
	#define LEAF_BIT (0x1)
	#define EMPTY_BIT (0x2)

	typedef Quadtree::children_t children_t;
	typedef Quadtree::node_t node_t;
	typedef Quadtree::node_extra_t node_extra_t;
	typedef Quadtree::root_extra_t root_extra_t;

	#define MAX_LEVELS 32

	///////////////////////////////////////////////////////////////
public:

	QuadtreeGPUCompression();
	~QuadtreeGPUCompression();

	virtual void processSamplesPrelude(CompressionConfig config);
	virtual void processTile(const TileCompressionInput& input);
	virtual CompressionStats processSamplesEpilogue();
	virtual void reset();
	virtual bool isPrecomputed();
	virtual int  computeShadows(ShadowPrecomputeInput& input);

protected:
	void _createOpenGLResources();
private:
	/////////////////// GPU creation
	void _createFloatTexture(const size_t size, GLuint& id, cudaGraphicsResource_t& resource);
	void _createResources(size_t size);
	void _createTemporaryResources(size_t size);
	void _createMapping(const uint32_t mipmapLevels, cudaGraphicsResource_t& resource, std::vector<cudaArray_t>& arr, std::vector<cudaTextureObject_t>& textures, std::vector<cudaSurfaceObject_t>& surfaces);
	void _createMappings(const uint32_t mipmapLevels);
	void _destroyMappings(const uint32_t mipmapLevels);
	void _freeOpenGLResources();
	void _freeTemporaryResources();
	void _copyDepthBounds(GLHelpers::TextureObject2D dmax, GLHelpers::TextureObject2D dmin, size_t size);

	uint32_t _createLevels(const size_t size, const bool preserveBottomLevel = false);
	root_extra_t _convertQuadtree(const size_t size, const uint32_t numexists, const bool preserveBottomLevel = false, const uint32_t* bottomLevelPtrs = NULL);

	/////////////////// Top level construction (copied from CPU compression - TODO: Perform this on the GPU and merge tile and top-level construction)
	uint32_t _createTopLevelTreeStructure();
	/////////////////// 
private:


	// Texture IDs and CUDA data for the buffers used during construction
public: //!!
	GLHelpers::TextureObject2D min_mipmap_tex;
	GLHelpers::TextureObject2D max_mipmap_tex;
	GLHelpers::TextureObject2D val_mipmap_tex;

	//GLuint _min_mipmap_id;
	//GLuint _max_mipmap_id;
	//GLuint _value_mipmap_id;

	cudaGraphicsResource_t min_mipmap_cuda_id;
	cudaGraphicsResource_t max_mipmap_cuda_id;
	cudaGraphicsResource_t val_mipmap_cuda_id;

	std::vector<cudaArray_t> _min_mipmap_levels_arrays;
	std::vector<cudaArray_t> _max_mipmap_levels_arrays;
	std::vector<cudaArray_t> _value_mipmap_levels_arrays;

	std::vector<cudaTextureObject_t> _min_textures;
	std::vector<cudaTextureObject_t> _max_textures;
	std::vector<cudaTextureObject_t> _value_textures;

	std::vector<cudaSurfaceObject_t> _min_surfaces;
	std::vector<cudaSurfaceObject_t> _max_surfaces;
	std::vector<cudaSurfaceObject_t> _value_surfaces;

	// Generated node and tile data
	std::vector<node_t> _nodes;		// Contains the final node array
	
	tiles_t _tiles;				// Contain the index of each tile's root node relative to the global nodes array
	uint32_t          _root_index;

	CompressionConfig config;

	// Stores the final OpenGL buffers and corresponding textures for rendering
public:
	GLHelpers::BufferObject node_children_buffer;
	GLHelpers::BufferObject node_value_buffer;
	GLHelpers::BufferObject node_flags_buffer;

	GLHelpers::TextureObjectBuffer node_children_tex;
	GLHelpers::TextureObjectBuffer node_value_tex;
	GLHelpers::TextureObjectBuffer node_flags_tex;

	//GLuint _nodeChildrenTexture;
	//GLuint _nodeValueTexture;
	//GLuint _nodeFlagsTexture;
	CompressionStats compressionStats;

	////////// Serialized nodes buffers
	GLHelpers::BufferObject orig_nodes_buffer;
	GLHelpers::BufferObject nodes_32_buffer;
	GLHelpers::BufferObject scales_buffer;
	GLHelpers::BufferObject ptrSizes_buffer;

	GLHelpers::TextureObjectBuffer nodes_32_tex;
	GLHelpers::TextureObjectBuffer scales_tex;
	GLHelpers::TextureObjectBuffer ptrSizes_tex;

	GLHelpers::TextureObject2D scales_2dtex;

	//GLuint _nodesTexture;
	//GLuint _nodesTexture32;
	//GLuint _scalesTexture;
	//GLuint _ptrSizesTexture;

	std::vector<uint32_t>	_scales;


#if USE_TOP_LEVEL_TEXTURE
	/////////// Top Level Structure
	GLuint _tlnValuesTexture;
	GLuint _tlnPointersTexture;
	CudaOpenGLResource   cuda_tlnValues_resource;
	CudaOpenGLResource   cuda_tlnPointers_resource;
#endif



	//////////// Precomputed evaluation data
	uint32_t*  cuda_quadtree_buffer;
	cudaTextureObject_t  cuda_scales_texture;
	CudaOpenGLResource   cuda_scales_resource;

	/////////////// Temporary cuda variables

	uint32_t* tempCount;
	uint8_t*  tempLevelExists;
	std::vector<uint32_t*> tempExistsVector;
	std::vector<thrust::device_vector<int32_t>> tempGidx;
	thrust::device_vector<uint32_t> tempLevels;
	thrust::device_vector<uint32_t> tempLidx;
	node_t*     tempDNodes;
	size_t      tempDNodes_size;
	GLuint      tempFlagClearFB;



};



