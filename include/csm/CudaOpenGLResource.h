#pragma once

#include "CudaHelpers.h"
#include <cuda_runtime.h>
#include <vector>


//#include "CudaHelpers.h"

class CudaOpenGLResource
{
public:

	CudaOpenGLResource();
	~CudaOpenGLResource();

	int registerResource(GLint tex, GLenum type, int levels = 1, unsigned int flags = cudaGraphicsRegisterFlagsSurfaceLoadStore);
	int unregisterResource();

	int mapResource();
	int unmapResource();

	void* mapBuffer();
	int   unmapBuffer();


	cudaArray_t getCudaArray(int level = 0);
	cudaSurfaceObject_t getCudaSurface(int level = 0);
	cudaTextureObject_t getCudaTexture(int level = 0);
	
	const std::vector<cudaArray_t>& getCudaArrays();
	const std::vector<cudaSurfaceObject_t>& getCudaSurfaces();
	const std::vector<cudaTextureObject_t>& getCudaTextures();


private:

	bool _registered;
	bool _mapped;
	int  _mipmapLevels;
	GLenum _type;

	cudaGraphicsResource_t _cuda_id;
	cudaArray_t            _cuda_array;
	bool                   _surfaces_ready;
	bool                   _textures_ready;

	void _createSurfaces();
	void _createTextures();

	std::vector<cudaArray_t>            _cudaArrays;
	std::vector<cudaResourceDesc>       _resource_descs;
	std::vector<cudaSurfaceObject_t>    _surface_objs;
	std::vector<cudaTextureObject_t>    _texture_objs;
};