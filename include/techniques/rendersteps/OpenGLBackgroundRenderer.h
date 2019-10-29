#pragma once

#include "common.h"
#include "helpers/OpenGLHelpers.h"
#include "helpers/Camera.h"

class OpenGLBackgroundRenderer
{
public: 

	OpenGLBackgroundRenderer();
	void setup();
	void renderColors(GLHelpers::FramebufferObject& fbo, size2D size, Camera* cam, glm::vec3 zenith_dir, glm::vec3 zenith_color, glm::vec3 nadir_color);
	void renderColors(GLHelpers::FramebufferObject& fbo, size2D size, Camera* cam, glm::vec3 zenith_dir, glm::vec3 zenith_color, glm::vec3 nadir_color, glm::vec3 horizon_color);
	void destroy();

	void reload_programs();
private:

	void set_cam(Camera* cam);
	void draw(GLHelpers::FramebufferObject& fbo, size2D size);
	GLHelpers::ProgramObject program;

};