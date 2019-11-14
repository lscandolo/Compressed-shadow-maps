/**
 * @author Matthias Holländer and Leonardo Scandolo
 * The character codes in this file are Copyright(c)
 * 2009 - 2013 Matthias Holländer.
 * 2015 - 2019 Leonardo Scandolo.
 * All rights reserved.
 *
 * No claims are made as to fitness for any particular purpose.
 * No warranties of any kind are expressed or implied.
 * The recipient agrees to determine applicability of information provided.
 *
 * The origin of this software, code, or partial code must not be misrepresented;
 * you must not claim that you wrote the original software.
 * Further, the author's name must be stated in any code or partial code.
 * If you use this software, code or partial code (including in the context
 * of a software product), an acknowledgment in the product documentation
 * is required.
 *
 * Altered source versions must be plainly marked as such, and must not be
 * misrepresented as being the original software.
 *
 * In no event shall the author or copyright holders be liable for any claim,
 * damages or other liability, whether in an action of contract, tort or otherwise,
 * arising from, out of or in connection with the software or the use or other
 * dealings in the software. Further, in no event shall the author be liable
 * for any direct, indirect, incidental, special, exemplary, or consequential
 * damages (including, but not limited to, procurement of substitute goods or
 * service; loss of use, data, or profits; or business interruption).
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies, portions or substantial portions of the Software.
 */

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
