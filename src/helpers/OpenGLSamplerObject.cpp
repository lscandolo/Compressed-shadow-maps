#include "helpers/OpenGLSamplerObject.h"

namespace GLHelpers {

	SamplerObjectImpl::SamplerObjectImpl()
		: id(0)
	{}

	SamplerObjectImpl::~SamplerObjectImpl()
	{}

	void SamplerObjectImpl::Release()
	{
		if (id) glDeleteSamplers(1, &id);
	}

	void SamplerObjectImpl::Bind(GLuint textureUnit)
	{
		glBindSampler(textureUnit, id);
	}

	void SamplerObjectImpl::Unbind(GLuint textureUnit)
	{
		glBindSampler(textureUnit, 0);
	}

	void SamplerObjectImpl::create()
	{
		Release();
		glCreateSamplers(1, &id);
	}

	void SamplerObjectImpl::setInterpolationMethod(GLenum method)
	{
		glSamplerParameteri(id, GL_TEXTURE_MIN_FILTER, method);
		glSamplerParameteri(id, GL_TEXTURE_MAG_FILTER, method);
	}

	void SamplerObjectImpl::setInterpolationMethod(GLenum minMethod, GLenum magMethod)
	{
		glSamplerParameteri(id, GL_TEXTURE_MIN_FILTER, minMethod);
		glSamplerParameteri(id, GL_TEXTURE_MAG_FILTER, magMethod);
	}

	void SamplerObjectImpl::setWrapMethod(GLenum method)
	{
		glSamplerParameteri(id, GL_TEXTURE_WRAP_S, method);
		glSamplerParameteri(id, GL_TEXTURE_WRAP_T, method);
		glSamplerParameteri(id, GL_TEXTURE_WRAP_R, method);
	}

	void SamplerObjectImpl::setWrapMethod(GLenum methodS, GLenum methodT, GLenum methodR)
	{
		glSamplerParameteri(id, GL_TEXTURE_WRAP_S, methodS);
		glSamplerParameteri(id, GL_TEXTURE_WRAP_T, methodT);
		glSamplerParameteri(id, GL_TEXTURE_WRAP_R, methodR);
	}

	void SamplerObjectImpl::setParameter(GLenum paramName, GLint param)
	{
		glSamplerParameteri(id, paramName, param);
	}

	void SamplerObjectImpl::setParameterf(GLenum paramName, GLfloat param)
	{
		glSamplerParameterf(id, paramName, param);
	}

	GLint SamplerObjectImpl::getParameter(GLenum paramName)
	{
		GLint param;
		glGetSamplerParameteriv(id, paramName, &param);
		return param;
	}

	GLfloat SamplerObjectImpl::getParameterf(GLenum paramName)
	{
		GLfloat param;
		glGetSamplerParameterfv(id, paramName, &param);
		return param;
	}

}
