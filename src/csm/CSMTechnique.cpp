#include "common.h"
#include "csm/CSMTechnique.h"
#include "managers/GLStateManager.h"
#include "managers/TextureManager.h"
#include "helpers/RenderOptions.h"
#include "helpers/ScopeTimer.h"

#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>
#include <iostream>

using namespace GLHelpers;

#define EVAL_SCOPE_NAME_CUDA "Shadow evaluation cuda"
#define EVAL_SCOPE_NAME_GL   "Shadow evaluation gl"

CSMTechnique::CSMTechnique() :
	fbo(create_object<FramebufferObject>()),
	colortex(create_object<TextureObject2D>()),
	depthtex(create_object<TextureObject2D>()),

	render_depth_program(create_object<ProgramObject>()),
	gbuffer_program(create_object<ProgramObject>()),
	shade_program(create_object<ProgramObject>()),
	eval_shadow_program(create_object<ProgramObject>()),

	shadowtex(create_object<TextureObject2D>()),
	wpostex(create_object<TextureObject2D>()),
	wdpostex(create_object<TextureObject2D>()),
	wnortex(create_object<TextureObject2D>()),
	tcmattex(create_object<TextureObject2D>())

{
	dirty_programs = false;
	dirty_csm = true;
	compression_option = 0;

	tilesize = 4096;
	csmsize  = 65536;
	
	merge_csm = true; 
	relative_csm = true;

	light_near = 0.01f;
	light_far = 1.f;
	light_is_dir = true;
	light_dir = glm::vec3(-1.f, -1.f, -1.f);
	light_spotpos = glm::vec3(1.f, 1.f, 1.f);
	light_spotangle = 45.f;
	light_color = glm::vec3(1, 1, 1);
	light_intensity = 1.f;

	eval_hierarchical = true;
	eval_pcf_size = 1;
	eval_cuda = false;
}

int  CSMTechnique::initial_setup(const SetupData& ts)
{
	dirty_csm = true;

	mesh = ts.mesh;
	cam  = ts.camera;

	fbo->Create();
	reload_programs();
	recreate_textures(ts.output_size);
	backgroundrenderer.setup();

	return SUCCESS;
}

void CSMTechnique::output_resize_callback(size2D output_size)
{
	recreate_textures(output_size);
}

void CSMTechnique::reload_programs()
{

	{
		ProgramShaderPaths p;
		p.commonPath = "shaders/";
		p.vertexShaderFilename = "csm/render_repth.vert";
		try {
			render_depth_program->CompileProgram(p, nullptr);
		} catch (std::runtime_error(e)) { std::cout << "Error creating opengl program" << std::endl; }
	}

	{
		ProgramShaderPaths p;
		p.commonPath = "shaders/";
		p.vertexShaderFilename = "csm/gbuffer.vert";
		p.geometryShaderFilename = "csm/gbuffer.geom";
		p.fragmentShaderFilename = "csm/gbuffer.frag";
		try {
			gbuffer_program->CompileProgram(p, nullptr);
		}
		catch (std::runtime_error(e)) { std::cout << "Error creating opengl program" << std::endl; }
	}

	{
		ProgramShaderPaths p;
		p.commonPath = "shaders/";
		p.vertexShaderFilename = "csm/shade.vert";
		p.geometryShaderFilename = "csm/shade.geom";
		p.fragmentShaderFilename = "csm/shade.frag";
		try {
			shade_program->CompileProgram(p, nullptr);
		}
		catch (std::runtime_error(e)) { std::cout << "Error creating opengl program" << std::endl; }
	}

	{
		GLHelpers::Detail::ShaderCommandLine cmd;
		cmd["RELATIVE_VALUES"] = relative_csm ? "1" : "0";

		try {
			eval_shadow_program->CompileComputeProgram(nullptr, "shaders/csm/eval_shadow.comp", cmd);
		}
		catch (std::runtime_error(e)) { std::cout << "Error creating opengl program" << std::endl; }
	}

	

	copier.reload_programs();
	backgroundrenderer.reload_programs();
	boundsrenderer.loadPrograms();

	dirty_programs = false;
}

std::string CSMTechnique::name()
{
	return "Compressed shadow maps";
}


void CSMTechnique::recreate_textures(size2D output_size)
{
	size.width  = output_size.width;
	size.height = output_size.height;

	GLenum depthformat    = GL_DEPTH_COMPONENT32F;
	GLenum colorformat    = GL_RGBA8;
	GLenum shadowformat   = GL_R8;
	GLenum wposformat     = GL_RGBA32F;
	GLenum wdposformat    = GL_RGB32F;
	GLenum wnortexformat  = GL_RGBA32F;
	GLenum tcmattexformat = GL_RGB32F;

	shadowtex_cuda.unregisterResource();
	wpostex_cuda.unregisterResource();
	wdpostex_cuda.unregisterResource();
	wnortex_cuda.unregisterResource();

	if (depthtex->width != size.width || depthtex->height != size.height || depthtex->internalFormat != depthformat) {
		depthtex->create(glm::ivec2(size.width, size.height), depthformat, nullptr);
		depthtex->setParameteri(GL_TEXTURE_COMPARE_MODE, GL_NONE);
		fbo->AttachDepthTexture(depthtex);
	}

	if (colortex->width != size.width || colortex->height != size.height || colortex->internalFormat != colorformat) {
		colortex->create(glm::ivec2(size.width, size.height), colorformat, nullptr);
		fbo->AttachColorTexture(colortex, 0);
		fbo->EnableConsecutiveDrawbuffers(1, 0);
	}

	if (shadowtex->width != size.width || shadowtex->height != size.height || shadowtex->internalFormat != shadowformat) {
		shadowtex->create(glm::ivec2(size.width, size.height), shadowformat, nullptr);
		shadowtex->setInterpolationMethod(GL_NEAREST);
		shadowtex->setWrapMethod(GL_CLAMP_TO_EDGE);
	}

	if (wpostex->width != size.width || wpostex->height != size.height || wpostex->internalFormat != wposformat) {
		wpostex->create(glm::ivec2(size.width, size.height), wposformat, nullptr);
	}

	if (wdpostex->width != size.width || wdpostex->height != size.height || wdpostex->internalFormat != wdposformat) {
		wdpostex->create(glm::ivec2(size.width, size.height), wdposformat, nullptr);
	}

	if (wnortex->width != size.width || wnortex->height != size.height || wnortex->internalFormat != wnortexformat) {
		wnortex->create(glm::ivec2(size.width, size.height), wnortexformat, nullptr);
	}

	if (tcmattex->width != size.width || tcmattex->height != size.height || tcmattex->internalFormat != tcmattexformat) {
		tcmattex->create(glm::ivec2(size.width, size.height), tcmattexformat, nullptr);
	}

	shadowtex_cuda.registerResource(shadowtex->id, GL_TEXTURE_2D, shadowtex->levels, cudaGraphicsRegisterFlagsWriteDiscard);
	wpostex_cuda.registerResource  (wpostex->id,   GL_TEXTURE_2D, wpostex->levels,   cudaGraphicsRegisterFlagsTextureGather);
	wdpostex_cuda.registerResource (wdpostex->id,  GL_TEXTURE_2D, wdpostex->levels,  cudaGraphicsRegisterFlagsTextureGather);
	wnortex_cuda.registerResource  (wnortex->id,   GL_TEXTURE_2D, wnortex->levels,   cudaGraphicsRegisterFlagsTextureGather);
}

void CSMTechnique::destroy()
{
	fbo->Delete();
	shadowtex->Release();
	colortex->Release();
	depthtex->Release();
}

void CSMTechnique::create_csm()
{ 
	csmconfig.merged = merge_csm;
	csmconfig.tile_qty = glm::ivec3(csmsize / tilesize, csmsize / tilesize, 1);
	csmconfig.tile_resolution = tilesize;
	csmconfig.resolution = csmsize;
	csmconfig.values_type = relative_csm ? Quadtree::VT_RELATIVE : Quadtree::VT_ABSOLUTE;
	csm.reset();
	csm.processSamplesPrelude(csmconfig);

	boundsrenderer.initialize(tilesize);

	tinput.bbox    = mesh->bbox;
	tinput.bsphere = mesh->bsphere;
	tinput.is_dir_light = light_is_dir;
	tinput.light_dir = glm::normalize(light_dir);
	tinput.light_near_plane = 0.01f;
	tinput.light_far_plane  = 1.5f;
	tinput.light_spotangle = light_spotangle;
	tinput.light_spotpos = light_spotpos;
	tinput.tile_qty.x = csmsize / tilesize;
	tinput.tile_qty.y = csmsize / tilesize;
	tinput.tile_qty.z = 1;

	Tiler tiler;
	for (int x = 0; x < tinput.tile_qty.x; ++x) {
		for (int y = 0; y < tinput.tile_qty.y; ++y) {
			tparam = tiler.computeTileParameters(tinput, glm::ivec3(x, y, 0));

			TileCompressionInput tci = boundsrenderer.computeBounds(*mesh, tparam, tparam.near_plane, tparam.far_plane, light_is_dir);
			csm.processTile(tci);
		}
	}
	
	cstats = csm.processSamplesEpilogue();



	dirty_csm = false;
}

void CSMTechnique::evaluate_shadow_cuda()
{
	PROFILE_SCOPE(EVAL_SCOPE_NAME_CUDA)

	ShadowPrecomputeInput spi;
	spi.width = size.width;
	spi.height = size.height;
	spi.depth_bias = 0.00005f;
	spi.lightDir = glm::normalize(tinput.light_dir);
	spi.lightnearplane = tparam.near_plane;
	spi.lightfarplane = tparam.far_plane;
	spi.lightisdirectional = tinput.is_dir_light;
	spi.lightPos = tinput.light_spotpos;
	spi.lightVPMatrix = tparam.untiled_vpmatrix;
	spi.pcf_hierarchical = eval_hierarchical;
	spi.pcf_size = eval_pcf_size;
	spi.shadowTexResource = &shadowtex_cuda;
	spi.worldPosTexResource = &wpostex_cuda;
	spi.worldDPosTexResource = &wdpostex_cuda;
	spi.worldNorTexResource = &wnortex_cuda;

	shadowtex_cuda.mapResource();
	wpostex_cuda.mapResource();
	wdpostex_cuda.mapResource();
	wnortex_cuda.mapResource();

	csm.computeShadows(spi);

	wnortex_cuda.unmapResource();
	wdpostex_cuda.unmapResource();
	wpostex_cuda.unmapResource();
	shadowtex_cuda.unmapResource();
}

void CSMTechnique::evaluate_shadow_opengl()
{
	PROFILE_SCOPE(EVAL_SCOPE_NAME_GL)
	eval_shadow_program->Use();

	eval_shadow_program->SetTexture("nodesTex32", csm.nodes_32_tex, 0);
	eval_shadow_program->SetTexture("scalesTex", csm.scales_tex, 1);
	eval_shadow_program->SetUniform("maxTreeLevel", GLint(log2(csmsize)));
	eval_shadow_program->SetUniform("depthMapResolution", GLint(csmsize));
	eval_shadow_program->SetUniform("light_vp", tparam.untiled_vpmatrix);

	eval_shadow_program->SetUniform("light_is_dir", tinput.is_dir_light);
	eval_shadow_program->SetUniform("light_dir", glm::normalize(tinput.light_dir));
	eval_shadow_program->SetUniform("light_spotpos", tinput.light_spotpos);
	eval_shadow_program->SetUniform("light_spotangle", float(tinput.light_spotangle * M_PI / 180.f));
	eval_shadow_program->SetUniform("light_intensity", light_intensity);
	eval_shadow_program->SetUniform("light_color", light_color);
	eval_shadow_program->SetUniform("light_near", tparam.near_plane);
	eval_shadow_program->SetUniform("light_far", tparam.far_plane);
	eval_shadow_program->SetTexture("wpostex", wpostex, 2);
	eval_shadow_program->SetTexture("wdpostex", wdpostex, 3);
	eval_shadow_program->SetTexture("wnortex", wnortex, 4);

	eval_shadow_program->SetUniform("hierarchical_eval", eval_hierarchical);
	eval_shadow_program->SetUniform("PCFSIZE", eval_pcf_size);

	eval_shadow_program->SetImageTexture(shadowtex, 0, GL_WRITE_ONLY);

	eval_shadow_program->DispatchCompute(glm::ivec2(size.width, size.height), glm::ivec2(16, 16));
}

void CSMTechnique::frame_prologue()
{
	if (dirty_programs) reload_programs();
	if (dirty_csm) create_csm();

	// Render background
	backgroundrenderer.renderColors(fbo, size, cam, glm::vec3(0, 1, 0), glm::vec3(0.59, 0.79, 1.0), glm::vec3(0.8, 0.59, 0.49));

	GLState state;
	state.setViewportBox(0, 0, size.width, size.height);
	state.enable_cull_face = true;
	state.cull_face = GL_BACK;
	state.enable_depth_test = true;
	state.enable_depth_write = true;
	state.depth_test_func = GL_LEQUAL;
	state.enable_blend = false;
	GLStateManager::instance().setState(state);

	mesh->gl.vao->Bind();
	mesh->gl.indexbuffer->Bind();
	const int bytesPerElement = mesh->gl.indexbuffertype == GL_UNSIGNED_INT ? 4 : 2;
	GLsizei total_elements = GLsizei(mesh->gl.indexbuffer->bytesize / bytesPerElement);

	// Clear depth
	fbo->Bind();
	fbo->AttachDepthTexture(depthtex);
	float clearDepth = 1.f;
	glClearNamedFramebufferfv(fbo->id, GL_DEPTH, 0, &clearDepth);

	//// Depth prepass
	render_depth_program->Use();
	render_depth_program->SetUniform("mvp", cam->proj_matrix() * cam->view_matrix());
	glDrawElements(GL_TRIANGLES, total_elements, mesh->gl.indexbuffertype, OFFSET_POINTER(0));

	// Clear
	fbo->AttachColorTexture(shadowtex, 0);
	fbo->EnableConsecutiveDrawbuffers(1);
	glClearNamedFramebufferfv(fbo->id, GL_COLOR, 0, std::array<float, 4>{0, 0, 1, 1}.data());

	fbo->AttachColorTexture(wpostex,  0);
	fbo->AttachColorTexture(wdpostex, 1);
	fbo->AttachColorTexture(wnortex,  2);
	fbo->AttachColorTexture(tcmattex, 3);
	fbo->EnableConsecutiveDrawbuffers(4);
	glClearNamedFramebufferfv(fbo->id, GL_COLOR, 0, std::array<float, 4>{0, 0, 0, 0}.data());
	glClearNamedFramebufferfv(fbo->id, GL_COLOR, 1, std::array<float, 4>{0, 0, 0, 0}.data());
	glClearNamedFramebufferfv(fbo->id, GL_COLOR, 2, std::array<float, 4>{0, 0, 0, 0}.data());
	glClearNamedFramebufferfv(fbo->id, GL_COLOR, 3, std::array<float, 4>{0, 0, 0, 0}.data());

	gbuffer_program->Use();
	gbuffer_program->SetUniform("mMatrix", glm::identity<glm::mat4>());
	gbuffer_program->SetUniform("vMatrix", cam->view_matrix());
	gbuffer_program->SetUniform("pMatrix", cam->proj_matrix());
	gbuffer_program->SetUniform("mvpMatrix", cam->proj_matrix() * cam->view_matrix());
	gbuffer_program->SetUniform("normalMatrix", glm::identity<glm::mat3>());

	state.setViewportBox(0, 0, size.width, size.height);
	state.enable_cull_face = true;
	state.cull_face = GL_BACK;
	state.enable_depth_test = true;
	state.enable_depth_write = false;
	state.depth_test_func = GL_EQUAL;
	state.enable_blend = false;
	GLStateManager::instance().setState(state);

	glDrawElements(GL_TRIANGLES, total_elements, mesh->gl.indexbuffertype, OFFSET_POINTER(0));
	mesh->gl.vao->Unbind();
	mesh->gl.indexbuffer->Unbind();
	fbo->DetachColorTexture(0);
	fbo->DetachColorTexture(1);
	fbo->DetachColorTexture(2);
	fbo->DetachColorTexture(3);

	if (eval_cuda) {
		evaluate_shadow_cuda();
	} else {
		evaluate_shadow_opengl();
	}

}



void CSMTechnique::frame_render()
{

	//// Shade gbuffer
	shade_program->Use();
	fbo->Bind();
	fbo->AttachColorTexture(colortex, 0);
	fbo->EnableConsecutiveDrawbuffers(1);

	GLState state;
	state.setViewportBox(0, 0, size.width, size.height);
	state.enable_cull_face   = false;
	state.enable_depth_test  = false;
	state.enable_depth_write = false;
	state.enable_blend       = false;
	GLStateManager::instance().setState(state);

	shade_program->SetTexture("shadowtex", shadowtex, 0);
	shade_program->SetTexture("wpostex",   wpostex,   1);
	shade_program->SetTexture("wdpostex",  wdpostex,  2);
	shade_program->SetTexture("wnortex",   wnortex,   3);
	shade_program->SetTexture("tcmattex",  tcmattex,  4);

	shade_program->SetUniform("ws_campos", cam->position());
	//shade_program->SetUniform("enableAlphaTest", alpha_test_enabled);
	shade_program->SetUniform("cam_nearfar", glm::vec2(cam->nearplane(), cam->farplane()));
	shade_program->SetSSBO("material_ssbo", mesh->materialdata.materialListBuffer, 0);
	shade_program->SetUniform("ambient_light", glm::vec3(0.6, 0.6, 0.6));
	shade_program->SetUniform("light_is_dir", tinput.is_dir_light);
	shade_program->SetUniform("light_dir", glm::normalize(tinput.light_dir));
	shade_program->SetUniform("light_spotpos", tinput.light_spotpos);
	shade_program->SetUniform("light_spotangle", float(tinput.light_spotangle * M_PI / 180.f));
	shade_program->SetUniform("light_intensity", light_intensity);
	shade_program->SetUniform("light_color", light_color);
	shade_program->SetUniform("light_near", tparam.near_plane);
	shade_program->SetUniform("light_far", tparam.far_plane);


	glDrawArrays(GL_POINTS, 0, 4);
	shade_program->Use(false);

	check_opengl();
}


void CSMTechnique::frame_epilogue()
{
	size2D& window_size = DefaultRenderOptions().window_size;
	fbo->AttachColorTexture(colortex, 0);
	fbo->EnableConsecutiveDrawbuffers(1, 0);

	fbo->Unbind();

	glBlitNamedFramebuffer(fbo->id, 0, 0, 0, size.width, size.height, 0, 0, window_size.width, window_size.height, GL_COLOR_BUFFER_BIT, size == window_size ? GL_NEAREST : GL_LINEAR);
}

void CSMTechnique::draw_gui(int order)
{
	if (ImGui::TreeNode("Shadow map creation"))
	{
		if (ImGui::Button("Update compressed shadow map"))
		{
			dirty_csm = true;
			dirty_programs = true;
		}

		outdatedCSM = false;
		outdatedCSM |= (relative_csm ? Quadtree::VT_RELATIVE : Quadtree::VT_ABSOLUTE) != csmconfig.values_type;
		outdatedCSM |= merge_csm != csmconfig.merged;
		outdatedCSM |= tilesize != csmconfig.tile_resolution;
		outdatedCSM |= csmsize != csmconfig.resolution;
		outdatedCSM |= light_is_dir != tinput.is_dir_light;
		outdatedCSM |= glm::distance(glm::normalize(light_dir), glm::normalize(tinput.light_dir)) > 0.00001f;
		outdatedCSM |= !tinput.is_dir_light && (tinput.light_spotpos != light_spotpos);
		outdatedCSM |= !tinput.is_dir_light && (tinput.light_spotangle != light_spotangle);


		if (outdatedCSM) {
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(1, 0, 0, 1), "Shadow map configuration changed!");
		}


		if (ImGui::BeginCombo("Tile size", std::to_string(tilesize).c_str()))
		{
			bool _b = true;
			for (int sz = 256; sz <= 4096; sz *= 2) {
				if (ImGui::Selectable(std::to_string(sz).c_str(), _b)) { tilesize = sz; }
			}
			ImGui::EndCombo();
		}

		if (ImGui::BeginCombo("Shadow map size", std::to_string(csmsize).c_str()))
		{
			bool _b = true;
			for (int sz = 4096; sz <= 262144; sz *= 2) {
				if (ImGui::Selectable(std::to_string(sz).c_str(), _b)) { csmsize = sz; }
			}
			ImGui::EndCombo();
		}



		int merge_csm_temp = merge_csm;
		ImGui::Text("Compression type");
		ImGui::RadioButton("Original", &merge_csm_temp, 0); ImGui::SameLine();
		ImGui::RadioButton("Merged", &merge_csm_temp, 1);
		merge_csm = merge_csm_temp;

		int relative_csm_temp = relative_csm;
		ImGui::Text("Node values");
		ImGui::RadioButton("Absolute", &relative_csm_temp, 0); ImGui::SameLine();
		ImGui::RadioButton("Relative", &relative_csm_temp, 1);
		relative_csm = relative_csm_temp;


		int light_is_dir_temp = light_is_dir;
		ImGui::Text("Light type");
		ImGui::RadioButton("Directional", &light_is_dir_temp, 1); ImGui::SameLine();
		ImGui::RadioButton("Spotlight", &light_is_dir_temp, 0);
		light_is_dir = light_is_dir_temp;

		ImGui::DragFloat3("Direction", glm::value_ptr(light_dir), 0.01f);

		if (!light_is_dir) {
			ImGui::DragFloat3("Position", glm::value_ptr(light_spotpos), 0.01f);
			ImGui::DragFloat("Angle", &light_spotangle, 0.1f, 0.001f, 180.f);
		}
		
		ImGui::Text("Shadow map size: %f MB", cstats.compressed_size/(1024.f*1024)); 
		ImGui::Text("Uncompressed size: %zu MB", cstats.orig_size / (1024*1024));
		ImGui::Text("Compression ratio: %f%%", cstats.compressed_size/float(cstats.orig_size));
		ImGui::Text("Nodes (leaf/inner/empty): %zu/%zu/%zu", cstats.leaf_node_qty, cstats.inner_node_qty, cstats.empty_node_qty);

		ImGui::TreePop();
	}


	if (ImGui::TreeNode("Shadow map evaluation"))
	{
		int eval_cuda_temp = eval_cuda ? 1 : 0;
		ImGui::RadioButton("OpenGL", &eval_cuda_temp, 0); ImGui::SameLine();
		ImGui::RadioButton("CUDA", &eval_cuda_temp, 1); 
		eval_cuda = eval_cuda_temp;

		if (eval_cuda) {
			TimingStats eval_time_cuda = TimingManager::instance().getTimingStats(EVAL_SCOPE_NAME_CUDA);
			ImGui::Text("Shadow evaluation time (cuda): %.3fms", eval_time_cuda.cuda_time);
		}
		else {
			TimingStats eval_time_gl = TimingManager::instance().getTimingStats(EVAL_SCOPE_NAME_GL);
			ImGui::Text("Shadow evaluation time (opengl): %.3fms", eval_time_gl.gl_time);
		}

		ImGui::Text("PCF kernel size");

		ImVec4 onColor = ImVec4(0.3, 0.5, 1.0, 1);
		ImVec4 offColor = ImVec4(0.15, 0.25, 0.5, 1);
		ImVec4 hoverColor = ImVec4(0.2, 0.4, 0.75, 1);
		std::vector<std::string> pcf_labels{ "Single" , "3x3", "5x5", "7x7", "9x9", "11x11" };
		for (int i = 0; i <= 5; ++i)
		{
			if (i > 0) ImGui::SameLine();
			ImGui::PushID(i);
			ImGui::PushStyleColor(ImGuiCol_Button, eval_pcf_size == i ? onColor : offColor);
			ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoverColor);
			ImGui::PushStyleColor(ImGuiCol_ButtonActive, onColor);
			if (ImGui::Button(pcf_labels[i].c_str())) eval_pcf_size = i;
			ImGui::PopStyleColor(3);
			ImGui::PopID();
		}

		ImGui::Checkbox("Hierarchical evaluation", &eval_hierarchical);
		ImGui::TreePop();
	}

	if (ImGui::TreeNode("Misc. options"))
	{
		ImGui::ColorEdit3("Color", glm::value_ptr(light_color));
		ImGui::DragFloat("Intensity", &light_intensity, 0.01f, 0.000f, 10.f);
		dirty_programs = ImGui::Button("Reload OpenGL shaders");
		ImGui::TreePop();
	}
}
		
