#include "common.h"

#include "cuda/csm/ErrorCheck.cuh"
#include "csm/CudaOpenGLResource.h"
#include <cuda.h>
#include <cuda_gl_interop.h>

CudaOpenGLResource::CudaOpenGLResource()
: _registered(false)
, _mapped(false)
, _surfaces_ready(false)
, _textures_ready(false)
{}

CudaOpenGLResource::~CudaOpenGLResource()
{
	if (_mapped)
		unmapResource(); 

	if (_registered)
		unregisterResource();
}


int
CudaOpenGLResource::registerResource(GLint tex, GLenum type, int levels, unsigned int flags)
{
	if (_registered)
		return -1;

	_type = type;

	gpuVerify();
	if (_type == GL_TEXTURE_BUFFER) {
		cudaGraphicsGLRegisterBuffer(&_cuda_id, tex, cudaGraphicsMapFlagsReadOnly);
	} else {
		cudaGraphicsGLRegisterImage(&_cuda_id, tex, _type, flags);
	}
	gpuVerify();

	_registered = true;
	_mipmapLevels = levels;

	return 0;
}

int
CudaOpenGLResource::unregisterResource()
{
	if (!_registered || _mapped)
		return -1;

	cudaGraphicsUnregisterResource(_cuda_id);
	gpuVerify();

	_registered = false;

	return 0;
}

int
CudaOpenGLResource::mapResource()
{
	if (!_registered || _mapped)
		return -1;

	cudaGraphicsMapResources(1, &_cuda_id);
	gpuVerify();

	_cudaArrays.resize(_mipmapLevels);
	_resource_descs.resize(_mipmapLevels);

	if (_type == GL_TEXTURE_BUFFER) 
	{
	
	}

	for (int level_index = 0; level_index < _mipmapLevels; ++level_index)
	{
		cudaGraphicsSubResourceGetMappedArray(&_cudaArrays[level_index], _cuda_id, 0, unsigned int(level_index));
		gpuVerify();

		memset(&_resource_descs[level_index], 0, sizeof(cudaResourceDesc));
		_resource_descs[level_index].resType = cudaResourceTypeArray;
		_resource_descs[level_index].res.array.array = _cudaArrays[level_index];
	}
	
	_mapped = true;

	return 0;
}

int
CudaOpenGLResource::unmapResource()
{
	if (!_registered || !_mapped)
		return -1;

	if (_surfaces_ready) {
		for (int level_index = 0; level_index < _mipmapLevels; ++level_index)
		{
			cudaDestroySurfaceObject(_surface_objs[level_index]);
			gpuVerify();
		}
	}

	if (_textures_ready) {
		for (int level_index = 0; level_index < _mipmapLevels; ++level_index)
		{
			cudaDestroyTextureObject(_texture_objs[level_index]);
			gpuVerify();
		}
	}


	cudaGraphicsUnmapResources(1, &_cuda_id);
	gpuVerify();

	_cudaArrays.clear();
	_texture_objs.clear();
	_resource_descs.clear();
	_surface_objs.clear();

	_mapped = false;
	_surfaces_ready = false;
	_textures_ready = false;
	
	return 0;

}

void*
CudaOpenGLResource::mapBuffer()
{
	if (!_registered || _mapped)
		return nullptr;

	cudaGraphicsMapResources(1, &_cuda_id);
	gpuVerify();

	if (_type != GL_TEXTURE_BUFFER) 
	{
		return nullptr;
	}

	void* ptr;
	size_t size;

	cudaGraphicsResourceGetMappedPointer(&ptr, &size, _cuda_id);
	gpuVerify();
	
	_mapped = true;

	return ptr;
}

int
CudaOpenGLResource::unmapBuffer()
{
	if (!_registered || !_mapped)
		return -1;

	if (_type != GL_TEXTURE_BUFFER)
		return -1;

	cudaGraphicsUnmapResources(1, &_cuda_id);
	gpuVerify();

	_mapped = false;

	return 0;
}


cudaArray_t
CudaOpenGLResource::getCudaArray(int level)
{
	Assert(_registered || _mapped || level < _mipmapLevels, "Error retrieving opengl cuda array.");

	return _cudaArrays[level];

}

cudaSurfaceObject_t
CudaOpenGLResource::getCudaSurface(int level)
{
	Assert(_registered || _mapped || level < _mipmapLevels, "Error retrieving opengl cuda surface.");

	if (!_surfaces_ready)
		_createSurfaces();

	return _surface_objs[level];
}

cudaTextureObject_t
CudaOpenGLResource::getCudaTexture(int level)
{
	Assert(_registered || _mapped || level < _mipmapLevels, "Error retrieving opengl cuda texture.");

	if (!_textures_ready)
		_createTextures();

	return _texture_objs[level];
}

const std::vector<cudaArray_t>&
CudaOpenGLResource::getCudaArrays()
{
	Assert(_registered || _mapped, "Error retrieving opengl cuda texture.");

	return _cudaArrays;

}

const std::vector<cudaSurfaceObject_t>&
CudaOpenGLResource::getCudaSurfaces()
{
	Assert(_registered || _mapped, "Error retrieving opengl cuda surfaces.");

	if (!_surfaces_ready)
		_createSurfaces();

	return _surface_objs;
}

const std::vector<cudaTextureObject_t>&
CudaOpenGLResource::getCudaTextures()
{
	Assert(_registered || _mapped, "Error retrieving opengl cuda textures.");

	if (!_textures_ready)
		_createTextures();

	return _texture_objs;
}

void
CudaOpenGLResource::_createSurfaces()
{
	_surface_objs.resize(_mipmapLevels);
	for (int level_index = 0; level_index < _mipmapLevels; ++level_index)
	{
		cudaCreateSurfaceObject(&_surface_objs[level_index], &_resource_descs[level_index]);
		gpuVerify();
	}
	_surfaces_ready = true;
}

void
CudaOpenGLResource::_createTextures()
{
	struct cudaTextureDesc textureDesc;
	memset(&textureDesc, 0, sizeof(cudaTextureDesc));
	textureDesc.addressMode[0] = cudaAddressModeClamp;
	textureDesc.addressMode[1] = cudaAddressModeClamp;
	textureDesc.addressMode[2] = cudaAddressModeClamp;
	textureDesc.filterMode = cudaFilterModePoint;
	textureDesc.readMode = cudaReadModeElementType;
	textureDesc.sRGB = false;
	textureDesc.normalizedCoords = false;
	textureDesc.maxAnisotropy = 0;
	textureDesc.mipmapFilterMode = cudaFilterModePoint;
	textureDesc.mipmapLevelBias = 0;

	_texture_objs.resize(_mipmapLevels);
	for (int level_index = 0; level_index < _mipmapLevels; ++level_index)
	{
		textureDesc.minMipmapLevelClamp = float(level_index);
		textureDesc.maxMipmapLevelClamp = float(level_index);

		cudaCreateTextureObject(&_texture_objs[level_index], &_resource_descs[level_index], &textureDesc, nullptr);
		gpuVerify();
	}
	_textures_ready = true;
}
