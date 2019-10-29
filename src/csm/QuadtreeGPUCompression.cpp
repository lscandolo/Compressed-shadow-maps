#include "csm/CudaHelpers.h"
#include "csm/QuadtreeGPUCompression.h"
//#include "Directories.h"
//#include "Exr.h"
#include "helpers/ScopeTimer.h"


#include "cuda/csm/MipmapCreation.cuh"
#include "cuda/csm/QuadtreeCreation.cuh"
#include "cuda/csm/QuadtreeCreationHelpers.cuh"
#include "cuda/csm/ErrorCheck.cuh"
#include "cuda/csm/QuadtreeEvaluation-Absolute.cuh"
#include "cuda/csm/QuadtreeEvaluation-Relative.cuh"
#include "csm/QuadtreeSerialization.h"
#include "csm/QuadtreeMergingSerialization.h"
#include "csm/QuadtreePrivate.h"

#include <glm/gtc/type_ptr.hpp>

using namespace Quadtree;
using namespace GLHelpers;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

QuadtreeGPUCompression::QuadtreeGPUCompression()  :
	
	node_children_buffer(create_object<BufferObject>()),
	node_value_buffer(create_object<BufferObject>()),
	node_flags_buffer(create_object<BufferObject>()),
	orig_nodes_buffer(create_object<BufferObject>()),
	nodes_32_buffer(create_object<BufferObject>()),
	scales_buffer(create_object<BufferObject>()),
	ptrSizes_buffer(create_object<BufferObject>()),

	node_children_tex(create_object<TextureObjectBuffer>()),
	node_value_tex(create_object<TextureObjectBuffer>()),
	node_flags_tex(create_object<TextureObjectBuffer>()),
	nodes_32_tex(create_object<TextureObjectBuffer>()),
	scales_tex(create_object<TextureObjectBuffer>()),
	ptrSizes_tex(create_object<TextureObjectBuffer>()),

	min_mipmap_tex(create_object<TextureObject2D>()),
	max_mipmap_tex(create_object<TextureObject2D>()),
	val_mipmap_tex(create_object<TextureObject2D>()),
	
	scales_2dtex(create_object<TextureObject2D>())

{
	cuda_quadtree_buffer = 0;
	reset();
}

QuadtreeGPUCompression::~QuadtreeGPUCompression()
{
	_freeOpenGLResources();
}


void
QuadtreeGPUCompression::reset()
{
	_tiles.clear();
	_nodes.clear();

	if (min_mipmap_tex && min_mipmap_tex->id) {
		cudaGraphicsUnregisterResource(min_mipmap_cuda_id);
		min_mipmap_tex->Release();
		check_opengl();
	}

	if (max_mipmap_tex && max_mipmap_tex->id) {
		cudaGraphicsUnregisterResource(max_mipmap_cuda_id);
		max_mipmap_tex->Release();
		check_opengl();
	}

	if (val_mipmap_tex && val_mipmap_tex->id) {
		cudaGraphicsUnregisterResource(val_mipmap_cuda_id);
		val_mipmap_tex->Release();
		check_opengl();
	}

	if (scales_tex && scales_tex->id) {
		cuda_scales_resource.unmapBuffer();
		cuda_scales_resource.unregisterResource();
	}
}

bool 
QuadtreeGPUCompression::isPrecomputed()
{
	return true;
}

int
QuadtreeGPUCompression::computeShadows(ShadowPrecomputeInput& input)
{
	float* vals = glm::value_ptr(input.lightVPMatrix);
	Quadtree::matrix mvp;

	
	//mvp.r0 = make_float4(vals[0],  vals[1],  vals[2],  vals[3]);
	//mvp.r1 = make_float4(vals[4],  vals[5],  vals[6],  vals[7]);
	//mvp.r2 = make_float4(vals[8],  vals[9],  vals[10], vals[11]);
	//mvp.r3 = make_float4(vals[12], vals[13], vals[14], vals[15]);
	

	mvp.r0 = make_float4(vals[0], vals[4], vals[8], vals[12]);
	mvp.r1 = make_float4(vals[1], vals[5], vals[9], vals[13]);
	mvp.r2 = make_float4(vals[2], vals[6], vals[10], vals[14]);
	mvp.r3 = make_float4(vals[3], vals[7], vals[11], vals[15]);

	Quadtree::cuda_quadtree_data qdata;

	qdata.quadtree      = cuda_quadtree_buffer; 
	qdata.scalesTexture = cuda_scales_texture;

	qdata.maxTreeLevel = level_t(log2(config.resolution));

	Quadtree::cuda_scene_data sdata;
	sdata.nearplane = input.lightnearplane;
	sdata.farplane = input.lightfarplane;
	sdata.isDirectional = input.lightisdirectional;
	sdata.lightDir = make_float3(input.lightDir.x, input.lightDir.y, input.lightDir.z);
	sdata.lightPos = make_float3(input.lightPos.x, input.lightPos.y, input.lightPos.z);
	sdata.mvp = mvp;
	sdata.bias = input.depth_bias;
	sdata.hierarchical = input.pcf_hierarchical;

	switch (config.values_type)
	{
	default:
	case VT_ABSOLUTE:

		Quadtree::evaluateShadow_Absolute(input.width, 
			                              input.height,
										  input.pcf_size,
										  qdata,
										  sdata,
										  input.worldPosTexResource->getCudaTexture(0),
										  input.worldDPosTexResource->getCudaTexture(0),
										  input.worldNorTexResource->getCudaTexture(0),
										  input.shadowTexResource->getCudaSurface(0));
		break;
	case VT_RELATIVE:
		Quadtree::evaluateShadow_Relative(input.width, 
			                              input.height,
										  input.pcf_size,
										  qdata,
										  sdata,
										  input.worldPosTexResource->getCudaTexture(0),
										  input.worldDPosTexResource->getCudaTexture(0),
										  input.worldNorTexResource->getCudaTexture(0),
										  input.shadowTexResource->getCudaSurface(0));
		break;
	}


	check_opengl();

	return 0;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void 
QuadtreeGPUCompression::processSamplesPrelude(CompressionConfig config)
{
	//PROFILE_SCOPE("Create resources");

	this->config = config;

	// Create resources used for compression
	_createResources(config.tile_resolution);
	
	// Create resources used temporarily only during tree creation
	_createTemporaryResources(config.tile_resolution);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CompressionStats 
QuadtreeGPUCompression::processSamplesEpilogue()
{

	std::cout << std::endl;

	_root_index = _createTopLevelTreeStructure();
	_createOpenGLResources();


	{ //PROFILE_SCOPE("Free temp resources");
	  _freeTemporaryResources();
	}

	return compressionStats;

}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
QuadtreeGPUCompression::processTile(const TileCompressionInput& input)
{
	// PROFILE_SCOPE("MH Quadtree Creation Total");

	auto sampleCoord = input.tile_params.tile_coord;
	auto tileCoord = tile_coord_t(sampleCoord.x, sampleCoord.y);

	glPushAttrib(GL_ENABLE_BIT);	// Save attributes just in case
	const uint32_t tile_size = config.tile_resolution;
	const size_t pixel_count = tile_size * tile_size;
	const uint32_t mipmapLevels = uint32_t(log2(tile_size)) + 1;
	uint32_t numexists = 0;
	/////////////////////////////////////////////


	{ // PROFILE_SCOPE("MH Creation Total");

	    // 1. Fill the lowest level of the min/max level with the min/max depth values
	    _copyDepthBounds(input.back, input.front, tile_size);

		// 2. Map LOD levels of level texture buffers to CUDA texture and surface objects
		_createMappings(mipmapLevels);

		// 3. Create the levels of the tree as a mip-map
		numexists = _createLevels(tile_size);

		CudaVerifyErr();
	}

	{// PROFILE_SCOPE("Node Creation Total");

	    // 4. Convert the levels to a quad-tree structure
	    root_extra_t tileRoot = _convertQuadtree(tile_size, numexists);
		_tiles[tileCoord] = tileRoot;

	    CudaVerifyErr();
	}
	//	std::cout << "Quadtree conversion in " << (endtime - starttime) << " milliseconds." << std::endl;

	// 5. Unmap and unregister resources
	_destroyMappings(mipmapLevels);

	////////////////////////////////


	if (sampleCoord.y == 0) { std::printf("\rProcessed tiles up to %d,%d", sampleCoord.x, sampleCoord.y); }

	// Reestablish bindings
	glPopAttrib();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
QuadtreeGPUCompression::_createResources(size_t size)
{
	
}

void
QuadtreeGPUCompression::_freeOpenGLResources()
{
	node_children_buffer->Release();
	node_value_buffer->Release();
	node_flags_buffer->Release();
	orig_nodes_buffer->Release();
	nodes_32_buffer->Release();
	scales_buffer->Release();
	ptrSizes_buffer->Release();

	node_children_tex->Release();
	node_value_tex->Release();
	node_flags_tex->Release();
	nodes_32_tex->Release();
	scales_tex->Release();
	ptrSizes_tex->Release();

	min_mipmap_tex->Release();
	max_mipmap_tex->Release();
	val_mipmap_tex->Release();
}

void
QuadtreeGPUCompression::_createTemporaryResources(size_t size)
{
	const uint32_t mipmapLevels = uint32_t(log2(size)) + 1;

	if (!min_mipmap_tex->id) min_mipmap_tex->create(glm::ivec2(size, size), GL_R32F, nullptr, true);
	if (!max_mipmap_tex->id) max_mipmap_tex->create(glm::ivec2(size, size), GL_R32F, nullptr, true);
	if (!val_mipmap_tex->id) val_mipmap_tex->create(glm::ivec2(size, size), GL_R32F, nullptr, true);

	min_mipmap_tex->setInterpolationMethod(GL_NEAREST);
	max_mipmap_tex->setInterpolationMethod(GL_NEAREST);
	val_mipmap_tex->setInterpolationMethod(GL_NEAREST);

	min_mipmap_tex->setWrapMethod(GL_CLAMP_TO_EDGE);
	max_mipmap_tex->setWrapMethod(GL_CLAMP_TO_EDGE);
	val_mipmap_tex->setWrapMethod(GL_CLAMP_TO_EDGE);

	cudaGraphicsGLRegisterImage(&min_mipmap_cuda_id, min_mipmap_tex->id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
	cudaGraphicsGLRegisterImage(&max_mipmap_cuda_id, max_mipmap_tex->id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
	cudaGraphicsGLRegisterImage(&val_mipmap_cuda_id, val_mipmap_tex->id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);

	tempExistsVector.resize(mipmapLevels);

	for (uint32_t i = 0; i < mipmapLevels; ++i)
	{
		// Allocate memory for binary flags
		uint32_t current_size = uint32_t(size >> i);
		cudaMalloc(&tempExistsVector[i], sizeof(uint32_t)* current_size * current_size);
	}

	cudaMalloc(&tempCount, sizeof(uint32_t));

	cudaMalloc(&tempLevelExists, mipmapLevels * sizeof(uint8_t));

	size_t initialNodeCreationResourcesSize = 65536 * 128;
	tempDNodes_size = initialNodeCreationResourcesSize;
	cudaMalloc(&tempDNodes, sizeof(gpu_node_t) * tempDNodes_size);

	createThrustTemporaryResources(size, tempGidx, tempLevels, tempLidx, initialNodeCreationResourcesSize);

	glGenFramebuffers(1, &tempFlagClearFB);
}

void
QuadtreeGPUCompression::_freeTemporaryResources()
{
	// Delete tile helpers
	cudaGraphicsUnregisterResource(min_mipmap_cuda_id);
	cudaGraphicsUnregisterResource(max_mipmap_cuda_id);
	cudaGraphicsUnregisterResource(val_mipmap_cuda_id);

	min_mipmap_tex->Release();
	max_mipmap_tex->Release();
	val_mipmap_tex->Release();

	node_children_buffer->Release();
	node_value_buffer->Release();
	node_flags_buffer->Release();

	for (uint32_t i = 0; i < tempExistsVector.size(); ++i)
	{
		cudaFree(tempExistsVector[i]);
	}
	tempExistsVector.clear();

	cudaFree(tempCount);
	cudaFree(tempLevelExists);

	cudaFree(tempDNodes);
	tempDNodes_size = 0;

	freeThrustTemporaryResources(tempGidx, tempLevels, tempLidx);

	glDeleteFramebuffers(1, &tempFlagClearFB);
}



void 
QuadtreeGPUCompression::_createMapping(const uint32_t mipmapLevels, cudaGraphicsResource_t& resource, std::vector<cudaArray_t>& arr, std::vector<cudaTextureObject_t>& textures, std::vector<cudaSurfaceObject_t>& surfaces)
{
	// Map CUDA resource
	cudaGraphicsMapResources(1, &resource);
	CudaVerifyErr();

	// Resize return vectors
	arr.resize(mipmapLevels);
	textures.resize(mipmapLevels);
	surfaces.resize(mipmapLevels);

	// Create texture descriptor
	struct cudaTextureDesc textureDesc;
	textureDesc.addressMode[0] = cudaAddressModeWrap;
	textureDesc.addressMode[1] = cudaAddressModeWrap;
	textureDesc.filterMode = cudaFilterModePoint;
	textureDesc.readMode = cudaReadModeElementType;
	textureDesc.sRGB = false;
	textureDesc.normalizedCoords = false;
	textureDesc.filterMode = cudaFilterModePoint;
	textureDesc.maxAnisotropy = 16;
	textureDesc.mipmapFilterMode = cudaFilterModePoint;
	textureDesc.mipmapLevelBias = 0;

	for (size_t i = 0; i < mipmapLevels; ++i)
	{
		// Get CUDA arrays for current mipmap level
		cudaGraphicsSubResourceGetMappedArray(&arr[i], resource, 0, unsigned int(i));
		CudaVerifyErr();

		// Create CUDA texture object for current mipmap level
		cudaResourceDesc desc;
		desc.resType = cudaResourceTypeArray;
		desc.res.array.array = arr[i];
		textureDesc.minMipmapLevelClamp = float(i);
		textureDesc.maxMipmapLevelClamp = float(i);
		cudaCreateTextureObject(&textures[i], &desc, &textureDesc, nullptr);
		CudaVerifyErr();

		// Create CUDA surface objects for current mipmap level
		cudaCreateSurfaceObject(&surfaces[i], &desc);
		CudaVerifyErr();
	}
}

void 
QuadtreeGPUCompression::_createMappings(const uint32_t mipmapLevels)
{
	_createMapping(mipmapLevels, min_mipmap_cuda_id, _min_mipmap_levels_arrays, _min_textures, _min_surfaces);
	_createMapping(mipmapLevels, max_mipmap_cuda_id, _max_mipmap_levels_arrays, _max_textures, _max_surfaces);
	_createMapping(mipmapLevels, val_mipmap_cuda_id, _value_mipmap_levels_arrays, _value_textures, _value_surfaces);

}

void 
QuadtreeGPUCompression::_destroyMappings(const uint32_t mipmapLevels)
{
	// Destroy surface objects
	for (size_t i = 0; i < mipmapLevels; ++i)
	{
		cudaDestroySurfaceObject(_min_surfaces[i]);
		cudaDestroySurfaceObject(_max_surfaces[i]);
		cudaDestroySurfaceObject(_value_surfaces[i]);
	}
	CudaVerifyErr();

	// Destroy texture objects
	for (size_t i = 0; i < mipmapLevels; ++i)
	{
		cudaDestroyTextureObject(_min_textures[i]);
		cudaDestroyTextureObject(_max_textures[i]);
		cudaDestroyTextureObject(_value_textures[i]);
	}
	CudaVerifyErr();

	// Unmap CUDA arrays
	cudaGraphicsUnmapResources(1, &min_mipmap_cuda_id);
	cudaGraphicsUnmapResources(1, &max_mipmap_cuda_id);
	cudaGraphicsUnmapResources(1, &val_mipmap_cuda_id);
	CudaVerifyErr();
}


uint32_t
QuadtreeGPUCompression::_createLevels(const size_t size, const bool preserveBottomLevel)
{

	// Create levels of quadtree
    const uint32_t mipmapLevels = uint32_t(log2(size)) + 1;
    const uint32_t textureLevels = uint32_t(log2(config.tile_resolution)) + 1;
	const uint32_t topLevel = textureLevels - 1;
	const uint32_t bottomLevel = (topLevel + 1) - mipmapLevels;

	{ PROFILE_SCOPE("MH Creation-createQuatreeLevels")
	    cudaMemset(tempLevelExists, 0, mipmapLevels * sizeof(uint8_t));

	    for (size_t i = bottomLevel + 1; i <= topLevel; ++i)
		{
			size_t current_size = size >> (i - bottomLevel);
			createQuadtreeLevel((unsigned int)i, 
				preserveBottomLevel,
				_min_textures[i - 1],
				_max_textures[i - 1], 
				_value_surfaces[i - 1], 
				tempExistsVector[i - 1],
				_min_surfaces[i], 
				_max_surfaces[i], 
				_value_surfaces[i], 
				tempExistsVector[i],
				tempCount,
				tempLevelExists,
				dim3( uint32_t(current_size), uint32_t(current_size) ));
			CudaVerifyErr();
		}
    }

	uint32_t numexists;
	{ PROFILE_SCOPE("MH Creation-readNumexists");
	    cudaMemcpy(&numexists, tempCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
	}

	return numexists;

}

void 
QuadtreeGPUCompression::_copyDepthBounds(GLHelpers::TextureObject2D dmax, GLHelpers::TextureObject2D dmin, size_t size)
{
	{ PROFILE_SCOPE("MH Creation-copyDepth")
		dmin->copy(min_mipmap_tex);
		dmax->copy(max_mipmap_tex);
	}
}


root_extra_t 
QuadtreeGPUCompression::_convertQuadtree(const size_t size, 
										 const uint32_t numexists,
                                         const bool preserveBottomLevel,
										 const uint32_t* bottomLevelPtrs)
{
    const uint32_t mipmapLevels = uint32_t(log2(size)) + 1;
    const uint32_t textureLevels = uint32_t(log2(config.tile_resolution)) + 1;
	const uint32_t topLevel = textureLevels - 1;
	const uint32_t bottomLevel = (topLevel  + 1) - mipmapLevels;

	if (numexists == 1 && !preserveBottomLevel)
	{
		/// If numexists == 1, we have only the root node!
		root_extra_t root;
		cudaMemcpyFromArray(&root.min, _min_mipmap_levels_arrays[topLevel], 0, 0, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpyFromArray(&root.max, _max_mipmap_levels_arrays[topLevel], 0, 0, sizeof(float), cudaMemcpyDeviceToHost);
		root.inode = int32_t(_nodes.size());
		root.count = numexists;

		std::vector<int32_t> bottomPtrs(4, -1);
		if (bottomLevelPtrs != NULL)
			cudaMemcpy(bottomPtrs.data(), bottomLevelPtrs, 4 * sizeof(int32_t), cudaMemcpyDeviceToHost);

		node_t root_node;
		root_node.children.tl = bottomPtrs[0];
		root_node.children.tr = bottomPtrs[1];
		root_node.children.bl = bottomPtrs[2];
		root_node.children.br = bottomPtrs[3];
		root_node.flags = bottomLevelPtrs == NULL ? LEAF_BIT : 0;
		root_node.min = root.min;
		root_node.max = root.max;
		root_node.value = 0.5f * (root_node.min + root_node.max); 
		_nodes.push_back(root_node);

		return root;
	}

	std::vector<uint8_t> levelExists(mipmapLevels);
	cudaMemcpy(levelExists.data(), tempLevelExists, mipmapLevels*sizeof(uint8_t), cudaMemcpyDeviceToHost);
	uint8_t lowestlevel = bottomLevel;
	for (uint8_t i = 0; i < mipmapLevels; ++i)
	{
		if (levelExists[i])
			break;
		lowestlevel++;
	}
	lowestlevel = bottomLevel;


	// Create nodes on the GPU
	size_t root_idx = _nodes.size();
	{// PROFILE_SCOPE("Node Creation-createNodes");
		
		{//PROFILE_SCOPE("Node Creation-cudaMalloc");
		    resizeTemporaryStorage(numexists, tempDNodes_size, tempDNodes, tempLevels, tempLidx);
		}

		{//PROFILE_SCOPE("Node Creation-createNodeIndices");
		    // Fill in the levels, lidx and gidx arrays, 
		    // which respectively contain the level, local index in the level and global index in the node array
		    // of each new node created
		    createNodeIndices(size, preserveBottomLevel, numexists, lowestlevel, textureLevels, tempExistsVector, tempLevels, tempLidx, tempGidx);
			CudaVerifyErr();
		}

		// Create node array
		{//PROFILE_SCOPE("Node Creation-createNodeArrays");
		    // Use the node indices to create the nodes and fill the temporary node array
		    createNodeArrays(size, preserveBottomLevel, numexists, lowestlevel, textureLevels, tempLevels, tempLidx, tempGidx, tempExistsVector, _value_textures, _min_textures, _max_textures, tempDNodes, (uint32_t)root_idx, bottomLevelPtrs);
			CudaVerifyErr();
		}
	
		// Download the nodes to the CPU
		if (sizeof(gpu_node_t) != sizeof(node_t))
			std::cout << "CRITICAL ERROR! GPU and CPU node size does not match!" << std::endl;
	}
	
	{//PROFILE_SCOPE("Node Creation-node copy");
		std::vector<node_t> newnodes(numexists);
		cudaMemcpy(newnodes.data(), tempDNodes, sizeof(node_t)* numexists, cudaMemcpyDeviceToHost);

		// Append the nodes of this tile to the global list
		_nodes.insert(_nodes.end(), newnodes.begin(), newnodes.end());
		// std::cout << "_nodes.size=" << _nodes.size() << ", root_idx=" << root_idx << std::endl;

		// Store the root node with extra information in the tile map
		root_extra_t root;
		root.inode = int32_t(root_idx);
		root.count = numexists;
		cudaMemcpyFromArray(&root.min, _min_mipmap_levels_arrays[topLevel], 0, 0, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpyFromArray(&root.max, _max_mipmap_levels_arrays[topLevel], 0, 0, sizeof(float), cudaMemcpyDeviceToHost);
		return root;
	}

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

uint32_t
QuadtreeGPUCompression::_createTopLevelTreeStructure()
{
	// PROFILE_SCOPE("Top structure creation");

	// If there is only a single tile, simply returns its index in the global node array
	int num_tiles = config.tile_qty.x;

	if (num_tiles == 1)
	{
		const root_extra_t& root = _tiles[tile_coord_t(0, 0)];
		return root.inode;
	}

	const uint32_t topLevelsQty = uint32_t(log2(num_tiles)) + 1;
	std::cout << "Top levels: " << topLevelsQty << std::endl;


	uint32_t topLevel = uint32_t(log2(config.tile_resolution));
	uint32_t bottomLevel = topLevel - topLevelsQty + 1;
	const uint32_t bottomLevelElems = num_tiles * num_tiles;

	std::vector<float>    imins(bottomLevelElems);
	std::vector<float>    imaxs(bottomLevelElems);
	std::vector<uint32_t> counts(bottomLevelElems);
	std::vector<uint32_t> inodes(bottomLevelElems);

	for (size_t y = 0; y < num_tiles; ++y) {
		for (size_t x = 0; x < num_tiles; ++x) {
			const root_extra_t& troot = _tiles[tile_coord_t(x, y)];
			const uint32_t idx = uint32_t(y * num_tiles + x);
			imins[idx] = troot.min;
			imaxs[idx] = troot.max;
			counts[idx] = troot.count;
			inodes[idx] = troot.inode;

		}
	}

	cudaMemcpy(tempExistsVector[bottomLevel], counts.data(), counts.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

	uint32_t* tempBottomPtrs;
	cudaMalloc(&tempBottomPtrs, bottomLevelElems * sizeof(uint32_t));
	cudaMemcpy(tempBottomPtrs, inodes.data(), inodes.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

	min_mipmap_tex->uploadTextureData(imins.data(), bottomLevel);
	max_mipmap_tex->uploadTextureData(imaxs.data(), bottomLevel);

	// 2. Map LOD levels of level texture buffers to CUDA texture and surface objects
	uint32_t totalTextureLevels = uint32_t(log2(config.tile_resolution)) + 1;
	_createMappings(totalTextureLevels);

	// 3. Create the levels of the tree as a mip-map
	uint32_t newnumexists = _createLevels(num_tiles, true);
	CudaVerifyErr();

	// Add the number of nodes that are not counted (because they don't exist) but are in the nodes vector
	std::vector<uint32_t> bottomLevelExists(bottomLevelElems);
	cudaMemcpy(bottomLevelExists.data(), tempExistsVector[bottomLevel], counts.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	for (uint32_t i = 0; i < bottomLevelElems; ++i)
	{
		if (!bottomLevelExists[i])
			newnumexists++;
	}

	uint32_t oldnumexists = uint32_t(_nodes.size());
	uint32_t numexists = newnumexists - oldnumexists;


    // 4. Convert the levels to a quad-tree structure
	root_extra_t root = _convertQuadtree(num_tiles, numexists, true, tempBottomPtrs);
    CudaVerifyErr();

	// 5. Unmap and unregister resources
	_destroyMappings(totalTextureLevels);

	cudaFree(tempBottomPtrs);

	return root.inode;

}


//////////////////////////////////////////////////////////////////////////////////////////////////////


void
QuadtreeGPUCompression::_createOpenGLResources()
{

	/// Compact values
	Quadtree::SerializationOutput nodeSerialization;
	const uint32_t mipmapLevels = uint32_t(log2(config.resolution)) + 1; 

	Quadtree::SerializationInput serializationInput(_nodes);
	serializationInput.levels = mipmapLevels;
	serializationInput.root_index  = _root_index;
	serializationInput.value_types = config.values_type; 

	if (config.merged) {
		{ PROFILE_SCOPE("Compression serialization (w/hj)");
		    createSerializedMergedNodeBuffer(serializationInput, nodeSerialization);
		}
	} else {
		{PROFILE_SCOPE("Compression serialization");
		    createSerializedNodeBuffer(serializationInput, nodeSerialization);
		}
	}


	{ //PROFILE_SCOPE("Compression upload");

		if (isPrecomputed()) {
			///////////////////// Copy data to the gpu
			cudaFree(cuda_quadtree_buffer);
			cudaMalloc(&cuda_quadtree_buffer, nodeSerialization.nodebuffer.size() * sizeof(uint8_t));
			cudaMemcpy(cuda_quadtree_buffer, nodeSerialization.nodebuffer.data(), nodeSerialization.nodebuffer.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);

			scales_2dtex->create(glm::ivec2(mipmapLevels, 1), GL_R16UI, nodeSerialization.levelscales.data(), false);
			scales_2dtex->Unbind();
			cuda_scales_resource.registerResource(scales_2dtex->id, GL_TEXTURE_2D);
			cuda_scales_resource.mapResource();
			cuda_scales_texture = cuda_scales_resource.getCudaTexture(0);

			check_opengl();

		}




		/////////////////////////////////////////////////

		/////////////////////////////////////////////////////////////////
		if (!isPrecomputed() || true) { //!!
			/////// Upload converted node buffer

			orig_nodes_buffer->Create(GL_SHADER_STORAGE_BUFFER);
			orig_nodes_buffer->UploadData(_nodes.size() * sizeof(Quadtree::node_t), GL_STATIC_DRAW, _nodes.data());

			nodes_32_buffer->Create(GL_TEXTURE_BUFFER);
			nodes_32_buffer->UploadData(nodeSerialization.nodebuffer.size() * sizeof(uint8_t), GL_STATIC_DRAW, nodeSerialization.nodebuffer.data());
			nodes_32_tex->attachBufferData(nodes_32_buffer, GL_R32UI);

			scales_buffer->Create(GL_TEXTURE_BUFFER);
			scales_buffer->UploadData(nodeSerialization.levelscales.size() * sizeof(uint32_t), GL_STATIC_DRAW, nodeSerialization.levelscales.data());
			scales_tex->attachBufferData(scales_buffer, GL_R32UI);

			_scales.resize(MAX_LEVELS);
			std::fill(_scales.begin(), _scales.end(), 0);
			_scales.assign(nodeSerialization.levelscales.begin(), nodeSerialization.levelscales.begin() + std::min(_scales.size(), nodeSerialization.levelscales.size()) );

		}
	
	}

	std::cout << std::endl  << "Compressed shadow map stats:" << std::endl;
	const size_t orgmem = size_t(config.resolution) * size_t(config.resolution) * 4;
	const size_t compmem = size_t(nodeSerialization.nodebuffer.size() * sizeof(uint8_t));
	const double compratio = (double)compmem / (double)orgmem;
	std::cout << "original memory: " << orgmem << " bytes (" << ((float)orgmem / (1024.0f*1024.0f)) << " MB)" << std::endl;
	std::cout << "compressed memory: " << compmem << " bytes (" << ((float)compmem / (1024.0f*1024.0f)) << " MB)" << std::endl;
	std::cout << "compression ratio: " << compratio * 100.0 << "%" << std::endl;
	std::cout << "nodes : " << nodeSerialization.stats.node_qty 
		      << " \t leaf/full/empty: " 
			  << nodeSerialization.stats.leaf_qty << "/"
			  << nodeSerialization.stats.full_inner_qty << "/"
			  << nodeSerialization.stats.empty_inner_qty << std::endl;

	compressionStats.orig_size = orgmem;
	compressionStats.compressed_size = compmem;
	compressionStats.leaf_node_qty = nodeSerialization.stats.leaf_qty;
	compressionStats.inner_node_qty = nodeSerialization.stats.full_inner_qty;
	compressionStats.empty_node_qty = nodeSerialization.stats.empty_inner_qty;


}

