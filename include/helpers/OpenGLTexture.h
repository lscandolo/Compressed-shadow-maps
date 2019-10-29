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