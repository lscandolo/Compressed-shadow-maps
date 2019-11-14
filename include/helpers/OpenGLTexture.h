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
#include "helpers/ImageData.h"

#include <glm/glm.hpp>

namespace GLHelpers {

	// In case you want roll your own texture loader
	GLenum						GFXImageFormatToOpenGL( const ImageFormat& eFormat );
	
	// To retrieve the correct channel format it is not enough to use the GFX::ImageChannelFormat only.
	// 2 weird cases are combined here:
	// 1) OpenGL differentiates between integer and non-integer formats, i.e. there's a difference between
	//    GL_RED and GL_RED_INTEGER
	// 2) just based on the internal format we cannot differentiate between RGB and BGR formats
	GLenum						GFXChannelFormatToOpenGL( ImageChannelFormat eChannelFormat, GLenum eInternalFormat );

	// Potential because of actual channel format might differ (rgb/bgr issues)
	GLenum						GetPotentialChannelFormat( GLenum eInternalFormat );

	// This can be used in case nothing needs to be enforced
	// Should in general only be used if the actual supplied pixel data later
	// on will be nullptr
	GLenum						GetPotentialPixelDataTransferType( GLenum eInternalFormat );
	
	GLenum						GFXImageTypeToOpenGL( const ImageType& eType );


	int							GetImageSize1D( GLuint uiTexture );
	glm::ivec2					GetImageSize2D( GLuint uiTexture );
	glm::ivec3                  GetImageSize2DArray( GLuint uiTexture );
	glm::ivec3					GetImageSize3D( GLuint uiTexture );
	glm::ivec2					GetImageSizeCubeMap( GLuint uiTexture );
	

	GLenum						GetInternalFormat1D( GLuint uiTexture );
	GLenum						GetInternalFormat2D( GLuint uiTexture );
	GLenum						GetInternalFormat3D( GLuint uiTexture );
	GLenum						GetInternalFormatCubeMap( GLuint uiTexture );
	GLenum						GetInternalFormat( GLenum eTarget, GLuint uiTexture );


	size_t						GetBpp( GLenum eInternalFormat );
	
	int							GetNumberOfComponents( GLenum eFormat );

	std::string   				GetImageLayoutString( GLenum eFormat );

	bool   				        IsImageFormatInteger( GLenum eFormat );
	bool   				        IsImageFormatUnsignedInteger( GLenum eFormat );

	GLenum						GetCompatibleImageFormat( GLenum eInternalFormat );

	GLuint						CreateDefaultSamplerState();

	/// @brief will create a texture that has same size and format
	/// @param eTarget such as GL_TEXTURE_2D, GL_TEXTURE_1D
	GLuint						CreateSimilarTexture( GLenum eTarget, GLuint uiTexture, void* pPixelData = nullptr );


}