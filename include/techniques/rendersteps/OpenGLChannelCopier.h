#pragma once

#include "common.h"
#include "helpers/OpenGLHelpers.h"

class OpenGLChannelCopier
{
public: 

	OpenGLChannelCopier();
	void setup();
	void copy(GLHelpers::TextureObject2D in, GLHelpers::TextureObject2D out, glm::ivec4 out_channels = glm::ivec4(0,1,2,3), int in_mipmap = 0, int out_mipmap = 0);
	void destroy();

	void reload_programs();
private:


	GLHelpers::ProgramObject program;

};