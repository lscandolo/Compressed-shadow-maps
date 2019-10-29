#pragma once

#include "common.h"
#include "helpers/OpenGLTextureObject.h"

namespace GLHelpers {

	/// @brief Check the status of the framebuffer
	/// @return true if everything is alright, false otherwise
	bool CheckFrameBufferStatus( GLuint uiFBO, LoggerInterface* pLogger );

	class FramebufferObjectImpl
	{
	public:

		FramebufferObjectImpl();
		~FramebufferObjectImpl();

		void Create();
		void Delete();
		
		void Bind(GLenum target = GL_FRAMEBUFFER) const;
		static void Unbind(GLenum target = GL_FRAMEBUFFER);
		
		void AttachTexture            (TextureObject tex, GLenum framebufferAttachment, GLint texlevel = 0, GLenum framebufferTarget = GL_FRAMEBUFFER) const;
		void AttachStencilTexture     (TextureObject tex, GLint texlevel = 0, GLenum framebufferTarget = GL_FRAMEBUFFER) const;
		void AttachDepthStencilTexture(TextureObject tex, GLint texlevel = 0, GLenum framebufferTarget = GL_FRAMEBUFFER) const;
		void AttachDepthTexture       (TextureObject tex, GLint texlevel = 0, GLenum framebufferTarget = GL_FRAMEBUFFER) const;
		void AttachColorTexture       (TextureObject tex, int colorAttachmentIndex, GLint texlevel = 0, GLenum target = GL_FRAMEBUFFER) const;

		void AttachCubemapFace        (TextureObjectCube tex, GLenum texFaceTarget, GLenum framebufferAttachment, GLint texlevel = 0, GLenum target = GL_FRAMEBUFFER) const;
		void AttachTexture2DLayer     (TextureObject2DArray tex, GLenum framebufferAttachment, GLint texLayer, GLint texlevel = 0, GLenum framebufferTarget = GL_FRAMEBUFFER) const;
		void AttachTexture2DLayer     (TextureObject3D tex, GLenum framebufferAttachment, GLint texLayer, GLint texlevel = 0, GLenum framebufferTarget = GL_FRAMEBUFFER) const;
		void AttachTextureCubeLayer   (TextureObjectCubeArray tex, GLenum framebufferAttachment, GLint texLayer, GLint texlevel = 0, GLenum framebufferTarget = GL_FRAMEBUFFER) const;

		void DetachColorTexture(GLenum framebufferAttachment, GLenum framebufferTarget = GL_FRAMEBUFFER) const;

		void EnableConsecutiveDrawbuffers(GLuint drawbufferQty, GLuint startAttachmentIndex = 0, GLenum framebufferTarget = GL_FRAMEBUFFER) const;

		GLuint id;
	};

	using FramebufferObject = std::shared_ptr<FramebufferObjectImpl>;
}
