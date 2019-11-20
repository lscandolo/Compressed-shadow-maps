#pragma once

#include "common.h"

#include "OpenGLShaderParser.h"
#include "OpenGLShaderCompiler.h"

#include "OpenGLBufferObject.h"
#include "OpenGLFrameBufferObject.h"
#include "OpenGLProgramObject.h"
#include "OpenGLSamplerObject.h"
#include "OpenGLShaderObject.h"
#include "OpenGLTextureObject.h"

namespace GLHelpers 
{

	template<typename T> T create_object() { return std::make_shared<T::element_type>(); }

}