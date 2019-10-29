#include "managers/GLStateManager.h"
#include "helpers/ScopeTimer.h"
#include "csm/CudaHelpers.h"
#include "csm/BoundsRenderer.h"
#include <glm/gtc/type_ptr.hpp>
using namespace GLHelpers;

//s

BoundsRenderer::BoundsRenderer() 
{
	/// Front bound ///
	_renderDepthFramebuffer = create_object<FramebufferObject>();
	_renderDepthProgram     = create_object<ProgramObject>();
	_frontDepthTex          = create_object<TextureObject2D>();

	/// Back bound ///
    _backDepthTex             = create_object<TextureObject2D>();
	_depthPeelRenderProgram   = create_object<ProgramObject>();
	_depthPeelCoalesceProgram = create_object<ProgramObject>();

	_depthPeelFramebuffer     = create_object<FramebufferObject>();
	_depthPeelLayerTex        = create_object<TextureObject2D>();
	_depthPeelSideTex         = create_object<TextureObject2D>();
	_depthPeelStatusTex       = create_object<TextureObject2D>();
	_depthPeelMinTex          = create_object<TextureObject2D>();

	_depthPeelShaderStorageBuffer = create_object<BufferObject>();
}

void 
BoundsRenderer::initialize(int32_t tile_resolution)
{
	this->tile_resolution = tile_resolution;

	//// Front bound ///
	_renderDepthFramebuffer->Create();

	_frontDepthTex->create(glm::ivec2(tile_resolution, tile_resolution), GL_DEPTH_COMPONENT32F, nullptr);
	_backDepthTex->create(glm::ivec2(tile_resolution, tile_resolution), GL_DEPTH_COMPONENT32F, nullptr);

	_depthPeelMinTex->create(glm::ivec2(tile_resolution, tile_resolution), GL_R32F, nullptr);

	//// Create depth peel framebuffers
	_depthPeelFramebuffer->Create();
		
	_depthPeelLayerTex->create(glm::ivec2(tile_resolution, tile_resolution), GL_DEPTH_COMPONENT32F, nullptr);
	_depthPeelSideTex->create(glm::ivec2(tile_resolution, tile_resolution), GL_R8UI, nullptr);

	_depthPeelStatusTex->create(glm::ivec2(tile_resolution, tile_resolution), GL_R8I, nullptr);

	_depthPeelShaderStorageBuffer->Create(GL_SHADER_STORAGE_BUFFER);
	_depthPeelShaderStorageBuffer->UploadData(sizeof(GLuint), GL_DYNAMIC_READ, nullptr);

	loadPrograms();
}

void 
BoundsRenderer::loadPrograms()
{

	////////////////////////////////////////////// Front render program
	//////////////////////////////////////////////

	{
		ProgramShaderPaths spaths;
		spaths.commonPath = "shaders/csm/";
		spaths.vertexShaderFilename = "render_repth.vert";
		spaths.fragmentShaderFilename = "render_depth.frag";
		_renderDepthProgram->CompileProgram(spaths, nullptr);
	}

	////////////////////////////////////////////// Depth peel render program
	//////////////////////////////////////////////

	{
		ProgramShaderPaths spaths;
		spaths.commonPath = "shaders/csm/";
		spaths.vertexShaderFilename = "depth_peel_render.vert";
		spaths.fragmentShaderFilename = "depth_peel_render.frag";
		_depthPeelRenderProgram->CompileProgram(spaths, nullptr);
	}

	////////////////////////////////////////////// Depth peel coalesce program
	//////////////////////////////////////////////

	_depthPeelCoalesceProgram->CompileComputeProgram(nullptr, "shaders/csm/depth_peel_coalesce.comp");

}

void 
BoundsRenderer::renderEpilogue()
{
	_frontDepthTex->Release();
	_backDepthTex->Release();
	_renderDepthFramebuffer->Delete();
	_renderDepthProgram->Delete();

	_depthPeelFramebuffer->Delete();
	_depthPeelLayerTex->Release();
	_depthPeelSideTex->Release();
	_depthPeelMinTex->Release();
	_depthPeelStatusTex->Release();

	_depthPeelRenderProgram->Delete();
	_depthPeelCoalesceProgram->Delete();
	_depthPeelShaderStorageBuffer->Release();

	check_opengl();
}

TileCompressionInput
BoundsRenderer::computeBounds(const MeshData& mesh,
											 const Tiler::TileParameters& tile_params,
											 const float nearPlane,
											 const float farPlane,
											 bool directional)
{
	
	int MAX_LAYERS = 128;

	check_opengl();

	int  back_result_texture_index = 0;
	GLint isDirectional = directional;
	
	const int bytesPerElement = mesh.gl.indexbuffertype == GL_UNSIGNED_INT ? 4 : 2;
	const GLsizei total_elements = GLsizei(mesh.gl.indexbuffer->bytesize / bytesPerElement);


	const glm::mat4& mvpMatrix = tile_params.tile_vpmatrix;

	{  // Render front face
		
		{ //PROFILE_SCOPE("Front render")

			// Render the front faces

				// Set program and uniforms
			_renderDepthProgram->Use();
			_renderDepthFramebuffer->Bind();

			// Set GL viewport
			_renderDepthFramebuffer->AttachDepthTexture(_frontDepthTex);
			_renderDepthFramebuffer->EnableConsecutiveDrawbuffers(0);

			_renderDepthProgram->SetUniform("mvp", mvpMatrix);
			_renderDepthProgram->SetUniform("near", tile_params.near_plane);
			_renderDepthProgram->SetUniform("far", tile_params.far_plane);
			_renderDepthProgram->SetUniform("isDirectional", isDirectional);

			GLState glstate;
			glstate.setViewportBox(0, 0, tile_resolution, tile_resolution);
			glstate.enable_depth_write = true;
			glstate.enable_depth_test = true;
			glstate.enable_cull_face = true;
			glstate.cull_face = GL_BACK;
			glstate.depth_test_func = GL_LESS;
			GLStateManager::instance().setState(glstate);

			glClearDepth(1.0f);
			glClear(GL_DEPTH_BUFFER_BIT);

			mesh.gl.vao->Bind();
			mesh.gl.indexbuffer->Bind();

			glDrawElements(GL_TRIANGLES, total_elements, mesh.gl.indexbuffertype, OFFSET_POINTER(0));
			check_opengl();

		}

		{ // PROFILE_SCOPE("Back render")
			// Render the back faces

			//// Setup depth peel program
			_depthPeelRenderProgram->Use();

			// Set program and uniforms
			_depthPeelRenderProgram->SetUniform("near", tile_params.near_plane);
			_depthPeelRenderProgram->SetUniform("far", tile_params.far_plane);
			_depthPeelRenderProgram->SetUniform("isDirectional", isDirectional);

			mesh.gl.vao->Bind();
			mesh.gl.indexbuffer->Bind();

			// Attach render targets and clear
			{
				/// Special binding just to clear status and depth buffers
				_depthPeelFramebuffer->Bind();
				glViewport(0, 0, tile_resolution, tile_resolution);

				_depthPeelFramebuffer->AttachDepthTexture(_depthPeelLayerTex);
				_depthPeelFramebuffer->AttachColorTexture(_depthPeelStatusTex, 0);
				_depthPeelFramebuffer->EnableConsecutiveDrawbuffers(1);

				GLint   clearValues[] = { 1, 1, 1, 1 };
				glClearBufferiv(GL_COLOR, 0, clearValues);
			}

			GLuint samplesPassedQuery;
			glGenQueries(1, &samplesPassedQuery);

			////////////////// Copy depth from frontDepthTex to 
			////////////////////// frontDepthTextures[1] and backDepthTextures[1]
			////////////////////// so the starting peel starts from that depth
			_frontDepthTex->copy(_depthPeelLayerTex);

			int pass = 0;
			GLuint shaderStorageBufferContents = 1;
			bool last_pass;
			do {

				last_pass = shaderStorageBufferContents == 0;

				glBeginQuery(GL_SAMPLES_PASSED, samplesPassedQuery); /// TODO: relocate

				////////////// Draw next layer of front facing geometry
				if (!last_pass)
				{
					_depthPeelRenderProgram->Use();
					_depthPeelFramebuffer->Bind();
					
					_depthPeelFramebuffer->AttachDepthTexture(_depthPeelLayerTex);
					_depthPeelFramebuffer->AttachColorTexture(_depthPeelSideTex, 0);
					_depthPeelFramebuffer->EnableConsecutiveDrawbuffers(1);

					GLState glstate;
					glstate.setViewportBox(0, 0, tile_resolution, tile_resolution);
					glstate.enable_blend       = false;
					glstate.enable_cull_face   = false;
					glstate.depth_test_func    = GL_LESS;
					glstate.enable_depth_test  = true;
					glstate.enable_depth_write = true;
					GLStateManager::instance().setState(glstate);

					_depthPeelRenderProgram->SetUniform("mvp", mvpMatrix);
					_depthPeelRenderProgram->SetTexture("minDepthTex", pass == 0 ? _frontDepthTex : _depthPeelMinTex, 0);
					_depthPeelRenderProgram->SetTexture("statusTex", _depthPeelStatusTex, 1);

					glClearDepth(1.0f);
					glClear(GL_DEPTH_BUFFER_BIT);
					 
					glDrawElements(GL_TRIANGLES, total_elements, mesh.gl.indexbuffertype, OFFSET_POINTER(0));
					check_opengl();
				} else { // Final step: we simulate a final back face at max depth
					_depthPeelFramebuffer->Bind();
					glViewport(0, 0, tile_resolution, tile_resolution);

					_depthPeelFramebuffer->AttachDepthTexture(_depthPeelLayerTex);
					_depthPeelFramebuffer->AttachColorTexture(_depthPeelSideTex, 0);
					_depthPeelFramebuffer->EnableConsecutiveDrawbuffers(1);

					GLint   clearValues[] = { 0, 0, 0, 0 };
					glClearBufferiv(GL_COLOR, 0, clearValues);

					glClearDepth(1.0f);
					glClear(GL_DEPTH_BUFFER_BIT);
				}

				glEndQuery(GL_SAMPLES_PASSED);

				{ /// Coalesce program execution
					_depthPeelCoalesceProgram->Use();
					_depthPeelCoalesceProgram->SetImageTexture(_depthPeelStatusTex, 0, GL_WRITE_ONLY);
					_depthPeelCoalesceProgram->SetImageTexture(_depthPeelMinTex, 1, GL_WRITE_ONLY);
					_depthPeelCoalesceProgram->SetTexture("layerTex", _depthPeelLayerTex, 2);
					_depthPeelCoalesceProgram->SetTexture("sideTex", _depthPeelSideTex, 3);
					_depthPeelCoalesceProgram->SetUniform("firstPass", pass == 0);

					shaderStorageBufferContents = 0;
					_depthPeelShaderStorageBuffer->Bind();
					_depthPeelCoalesceProgram->SetSSBO("globalStatus", _depthPeelShaderStorageBuffer, 0);
					glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &shaderStorageBufferContents);

					glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

					_depthPeelCoalesceProgram->DispatchCompute(glm::ivec2(tile_resolution, tile_resolution), glm::ivec2(32, 32));

					glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT); 

					//glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R8I);
					//glBindImageTexture(1, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

					_depthPeelShaderStorageBuffer->DownloadData(&shaderStorageBufferContents, sizeof(GLuint));
				}

				pass ++;
			} while (!last_pass && pass < MAX_LAYERS);
			
		}
	}

	// Output to be fed to compressor
	TileCompressionInput compressionInput;
	compressionInput.tile_params = tile_params;
	compressionInput.front       = _frontDepthTex;
	compressionInput.back        = _depthPeelMinTex;

	return compressionInput;
}
