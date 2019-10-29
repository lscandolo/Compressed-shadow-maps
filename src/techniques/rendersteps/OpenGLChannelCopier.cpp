#include "techniques/rendersteps/OpenGLChannelCopier.h"
#include "managers/GLStateManager.h"
#include <iostream>

using namespace GLHelpers;
using namespace glm;

OpenGLChannelCopier::OpenGLChannelCopier() :
	program(create_object<ProgramObject>())
{}

void OpenGLChannelCopier::setup()
{
	reload_programs();
}

void OpenGLChannelCopier::reload_programs()
{
	try {
		program->CompileComputeProgram(nullptr, "shaders/copy_channels.comp");
	}
	catch (std::runtime_error(e)) {
		std::cout << "Error creating opengl program" << std::endl;
	}
}

void OpenGLChannelCopier::copy(GLHelpers::TextureObject2D in, GLHelpers::TextureObject2D out, glm::ivec4 out_channels, int in_mipmap, int out_mipmap)
{

	if (!program->programId) reload_programs();

	out_channels = clamp(out_channels, ivec4(0), ivec4(3));

	in_mipmap  = clamp(in_mipmap,  0, int(in->levels  - 1));
	out_mipmap = clamp(out_mipmap, 0, int(out->levels - 1));

	program->SetUniform("in_mipmap", in_mipmap);
	program->SetTexture("in_tex", in, 0);

	program->SetUniform("out_mipmap", out_mipmap);
	program->SetUniform("out_channels", out_channels);
	program->SetImageTexture(out, 1, GL_WRITE_ONLY, out_mipmap);

	if (out_mipmap == 0) {
		program->DispatchCompute(glm::ivec2(out->width, out->height), glm::ivec2(32, 32), true);
	} else {
		vec2 size = vec2(out->width, out->height);
		size = ceil(size / vec2(1<<out_mipmap, 1 << out_mipmap) );
		program->DispatchCompute(glm::ivec2(size.x, size.y), glm::ivec2(32, 32), true);
	}
	program->Use(false);
}

void OpenGLChannelCopier::destroy()
{
	program->Delete();
}
