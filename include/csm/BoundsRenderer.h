#pragma once

#include "common.h"
#include "helpers/MeshData.h"
#include "helpers/OpenGLHelpers.h"
#include "csm/Tiler.h"
#include "csm/QuadtreeGPUCompression.h"

#include <string>

class BoundsRenderer 
{

public:

	////////////// Creation/Destruction methods
	BoundsRenderer();

	void initialize(int32_t tile_resolution);

	TileCompressionInput
	computeBounds(const MeshData& mesh,
				  const Tiler::TileParameters& sampleParameters,
				  const float nearPlane,
				  const float farPlane,
				  bool directional);

	void renderEpilogue();

private:
public: //

	void loadPrograms();

	int32_t tile_resolution;

	//SamplerConfig _config;
	GLHelpers::BufferObject                 pos_buffer;
	GLHelpers::BufferObject                 index_buffer;

	GLHelpers::TextureObject2D              _frontDepthTex;
	GLHelpers::TextureObject2D              _backDepthTex;
		
	GLHelpers::FramebufferObject            _renderDepthFramebuffer;
	GLHelpers::ProgramObject                _renderDepthProgram;

	GLHelpers::ProgramObject                _depthPeelRenderProgram;

	GLHelpers::ProgramObject                _depthPeelCoalesceProgram;

	GLHelpers::FramebufferObject            _depthPeelFramebuffer;
	GLHelpers::TextureObject2D              _depthPeelLayerTex;
	GLHelpers::TextureObject2D              _depthPeelSideTex;
	GLHelpers::TextureObject2D              _depthPeelMinTex;

	GLHelpers::TextureObject2D              _depthPeelStatusTex;

	GLHelpers::BufferObject                 _depthPeelShaderStorageBuffer;
};
