#include "techniques/rendersteps/OpenGLBackgroundRenderer.h"
#include "managers/GLStateManager.h"
#include <iostream>

using namespace GLHelpers;

OpenGLBackgroundRenderer::OpenGLBackgroundRenderer() :
	program(create_object<ProgramObject>())
{}

void OpenGLBackgroundRenderer::setup()
{
	reload_programs();
}

void OpenGLBackgroundRenderer::reload_programs()
{
	GLHelpers::Detail::ShaderCommandLine cmd;

	ProgramShaderPaths p;
	p.commonPath = "shaders/";
	p.vertexShaderFilename = "quad_dir.vert";
	p.fragmentShaderFilename = "background.frag";

	try {
		program->CompileProgram(p, nullptr, cmd);
	}
	catch (std::runtime_error(e)) {
		std::cout << "Error creating opengl program" << std::endl;
	}
}

void OpenGLBackgroundRenderer::renderColors(GLHelpers::FramebufferObject& fbo, size2D size, Camera* cam, glm::vec3 zenith_dir, glm::vec3 zenith_color, glm::vec3 nadir_color)
{
	renderColors(fbo, size, cam, zenith_dir, zenith_color, nadir_color, 0.5f*(zenith_color + nadir_color));
}
void OpenGLBackgroundRenderer::renderColors(GLHelpers::FramebufferObject& fbo, size2D size, Camera* cam, glm::vec3 zenith_dir, glm::vec3 zenith_color, glm::vec3 nadir_color, glm::vec3 horizon_color)
{
	program->SetUniform("zenith_color", zenith_color);
	program->SetUniform("horizon_color", horizon_color);
	program->SetUniform("nadir_color", nadir_color);
	program->SetUniform("zenith_dir", glm::normalize(zenith_dir));
	set_cam(cam);

	draw(fbo, size);
}


void OpenGLBackgroundRenderer::set_cam(Camera* cam)
{
	Assert(cam);

	glm::mat4 vMatrix = cam->view_matrix();
	glm::mat4 pMatrix = cam->proj_matrix();
	program->SetUniform("vpMatrixInv", glm::inverse(pMatrix * vMatrix));
	program->SetUniform("wsCamPos", cam->position());
}


void OpenGLBackgroundRenderer::draw(FramebufferObject& fbo, size2D size)
{
	fbo->Bind();

	GLState state;
	state.setViewportBox(0, 0, size.width, size.height);
	state.enable_depth_test = false;
	state.enable_depth_write = false;
	state.enable_blend = false;
	state.enable_cull_face = false;
	GLStateManager::instance().setState(state);

	program->Use();

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

	program->Use(false);
}

void OpenGLBackgroundRenderer::destroy()
{
	program->Delete();
}
