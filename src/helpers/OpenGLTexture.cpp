#include "common.h"
#include "helpers/OpenGLTexture.h"
#include "helpers/ImageData.h"

// helper functions might need to use textureobject/texturearrayobject related helper functions

namespace GLHelpers {

	GLenum GFXImageFormatToOpenGL( const ImageFormat& eFormat )
	{
		const GLenum desc[] = {
			GL_INVALID_ENUM,						// FORMAT_UNKNOWN,

			// unsigned integer
			GL_R8UI,								// R8UI,
			GL_RG8UI,								// RG8UI,
			GL_RGB8UI,								// RGB8UI,
			GL_RGBA8UI,								// RGBA8UI,

			GL_R16UI,								// R16UI,
			GL_RG16UI,								// RG16UI,
			GL_RGB16UI,								// RGB16UI,
			GL_RGBA16UI,							// RGBA16UI,

			GL_R32UI,								// R32UI,
			GL_RG32UI,								// RG32UI, 
			GL_RGB32UI,								// RGB32UI,
			GL_RGBA32UI,							// RGBA32UI,

			// signed integer
			GL_R8I,									// R8I,
			GL_RG8I,								// RG8I,
			GL_RGB8I,								// RGB8I,
			GL_RGBA8I,								// RGBA8I,

			GL_R16I,								// R16I,
			GL_RG16I,								// RG16I,
			GL_RGB16I,								// RGB16I,
			GL_RGBA16I,								// RGBA16I,

			GL_R32I,								// R32I,
			GL_RG32I,								// RG32I,
			GL_RGB32I,								// RGB32I,
			GL_RGBA32I,								// RGBA32I,

			// floating point formats
			GL_R16F,								// R16F,
			GL_RG16F,								// RG16F,
			GL_RGB16F,								// RGB16F,
			GL_RGBA16F,								// RGBA16F

			GL_R32F,								// R32F,
			GL_RG32F,								// RG32F,
			GL_RGB32F,								// RGB32F,
			GL_RGBA32F,								// RGBA32F

			// floating point formats
			GL_R32F,								// R64F,
			GL_RG32F,								// RG64F,
			GL_RGB32F,								// RGB64F,
			GL_RGBA32F,								// RGBA64F

			GL_R8,									// R8_UNORM,
			GL_RG8,									// RG8_UNORM,
			GL_RGB8,								// RGB8_UNORM,
			GL_RGBA8,								// RGBA8_UNORM

			GL_COMPRESSED_RED_RGTC1,				// R_ATI_1N_UNORM
			GL_COMPRESSED_RG_RGTC2,					// COMPRESSED_RG_BC5_UNORM
			GL_COMPRESSED_RGBA_BPTC_UNORM,			// COMPRESSED_RG_BC7_UNORM

			GL_COMPRESSED_RGB_S3TC_DXT1_EXT,		// COMPRESSED_RGB_S3TC_DXT1_EXT
			GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,		// COMPRESSED_RGBA_S3TC_DXT1_EXT,
			GL_COMPRESSED_RGBA_S3TC_DXT3_EXT,		// COMPRESSED_RGBA_S3TC_DXT3_EXT,
			GL_COMPRESSED_RGBA_S3TC_DXT5_EXT,		// COMPRESSED_RGBA_S3TC_DXT5_EXT,

			GL_INVALID_ENUM,						// FORMAT_MAX
		};
		static_assert( ARRAY_SIZE( desc ) == ( int )ImageFormat::FORMAT_MAX+1, "arrays should have the same size" );
		return desc[ ( int )eFormat ];
	}

	GLenum GFXChannelFormatToOpenGL( ImageChannelFormat eChannelFormat, GLenum eInternalFormat )
	{
		// http://www.khronos.org/opengles/sdk/docs/man3/xhtml/glTexImage2D.xml

		switch( eInternalFormat ) {
		case GL_R8:
		case GL_R8_SNORM:
		case GL_R16F:
		case GL_R32F:
			return GL_RED;

		case GL_R8UI:
		case GL_R8I:
		case GL_R16UI:
		case GL_R16I:
		case GL_R32UI:
		case GL_R32I:
			return GL_RED_INTEGER;

		case GL_RG8UI:
		case GL_RG8I:
		case GL_RG16UI:
		case GL_RG16I:
		case GL_RG32UI:
		case GL_RG32I:
			return GL_RG_INTEGER;

		case GL_RGB8UI:
		case GL_RGB8I:
		case GL_RGB16UI:
		case GL_RGB16I:
		case GL_RGB32UI:
		case GL_RGB32I:
			return GL_RGB_INTEGER;

		case GL_RGBA8UI:
		case GL_RGBA8I:
		case GL_RGB10_A2UI:
		case GL_RGBA16UI:
		case GL_RGBA16I:
		case GL_RGBA32UI:
		case GL_RGBA32I:
			return GL_RGBA_INTEGER;

		case GL_RG8:
		case GL_RG8_SNORM:
		case GL_RG16F:
		case GL_RG32F:
			return GL_RG;

		case GL_RGB8:
		case GL_SRGB8:
		case GL_RGB565:
		case GL_RGB8_SNORM:
		case GL_R11F_G11F_B10F:
		case GL_RGB9_E5:
		case GL_RGB16F:
		case GL_RGB32F:
			if ( eChannelFormat == ImageChannelFormat::BGR ) {
				return GL_BGR;
			}
			else if ( eChannelFormat == ImageChannelFormat::RGB ) {
				return GL_RGB;
			}
			break;
		case GL_RGBA8:
		case GL_SRGB_ALPHA:	// == GL_SRGB8_ALPHA8 ???
		case GL_RGBA8_SNORM:
		case GL_RGB5_A1:
		case GL_RGBA4:
		case GL_RGB10_A2:
		case GL_RGBA16F:
		case GL_RGBA32F:
			if ( eChannelFormat == ImageChannelFormat::BGRA ) {
				return GL_BGRA;
			}
			else if ( eChannelFormat == ImageChannelFormat::RGBA ) {
				return GL_RGBA;
			}
			break;
		default:
			throw( std::runtime_error( "Invalid OpenGL enum" ) );
			break;
		}

		// reaches only here, when rgb/bgr mismatching
		throw( std::runtime_error( "Parameter mismatch" ) );
	}

	GLenum GetPotentialChannelFormat( GLenum eInternalFormat )
	{
		// http://www.khronos.org/opengles/sdk/docs/man3/xhtml/glTexImage2D.xml
		
		switch( eInternalFormat ) {
			case GL_R8:
			case GL_R8_SNORM:
			case GL_R16F:
			case GL_R32F:
				return GL_RED;

			case GL_R8UI:
			case GL_R8I:
			case GL_R16UI:
			case GL_R16I:
			case GL_R32UI:
			case GL_R32I:
				return GL_RED_INTEGER;

			case GL_RG8UI:
			case GL_RG8I:
			case GL_RG16UI:
			case GL_RG16I:
			case GL_RG32UI:
			case GL_RG32I:
				return GL_RG_INTEGER;

			case GL_RGB8UI:
			case GL_RGB8I:
			case GL_RGB16UI:
			case GL_RGB16I:
			case GL_RGB32UI:
			case GL_RGB32I:
				return GL_RGB_INTEGER;

			case GL_RGBA8UI:
			case GL_RGBA8I:
			case GL_RGB10_A2UI:
			case GL_RGBA16UI:
			case GL_RGBA16I:
			case GL_RGBA32UI:
			case GL_RGBA32I:
				return GL_RGBA_INTEGER;

			case GL_RG8:
			case GL_RG8_SNORM:
			case GL_RG16F:
			case GL_RG32F:
				return GL_RG;

			case GL_RGB8:
			case GL_SRGB8:
			case GL_RGB565:
			case GL_RGB8_SNORM:
			case GL_R11F_G11F_B10F:
			case GL_RGB9_E5:
			case GL_RGB16F:
			case GL_RGB32F:
				return GL_RGB;


			case GL_RGBA8:
			case GL_SRGB_ALPHA:	// == GL_SRGB8_ALPHA8 ???
			case GL_RGBA8_SNORM:
			case GL_RGB5_A1:
			case GL_RGBA4:
			case GL_RGB10_A2:
			case GL_RGBA16F:
			case GL_RGBA32F:
				return GL_RGBA;
			default:
				throw( std::runtime_error( "Invalid OpenGL enum" ) );
				break;
		}
	}

	GLenum GFXImageTypeToOpenGL( const ImageType& eType )
	{
		const GLenum desc[] = {
			GL_INVALID_ENUM,		// IMAGETYPE_UNKNOWN,

			GL_BYTE,				// BYTE,
			GL_UNSIGNED_BYTE,		// UNSIGNED_BYTE,

			GL_SHORT,				// SHORT,
			GL_UNSIGNED_SHORT,		// UNSIGNED_SHORT,

			GL_INT,					// INT,
			GL_UNSIGNED_INT,		// UNSIGNED_INT,

			// OpenGL has no long types
			GL_INT,					// LONG
			GL_UNSIGNED_INT,		// UNSIGNED_LONG

			GL_FLOAT,				// FLOAT,
			GL_DOUBLE,				// DOUBLE

			GL_INVALID_ENUM,		// IMAGETYPE_MAX,
		};
		static_assert( ARRAY_SIZE( desc ) == ( int )ImageType::IMAGETYPE_MAX+1, "arrays should have the same size" );
		return desc[ ( int )eType ];
	}

	int GetImageSize1D( GLuint uiTexture )
	{
		int iReturn = 0;
		glBindTexture( GL_TEXTURE_1D, uiTexture );
		glGetTexLevelParameteriv( GL_TEXTURE_1D, 0, GL_TEXTURE_WIDTH,  &iReturn );
		glBindTexture( GL_TEXTURE_1D, 0 );

		check_opengl();
		return iReturn;
	}

	glm::ivec2 GetImageSize2D( GLuint uiTexture )
	{
		glm::ivec2 vReturn( true );
		glBindTexture( GL_TEXTURE_2D, uiTexture );
		glGetTexLevelParameteriv( GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH,  &vReturn[ 0 ] );
		glGetTexLevelParameteriv( GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &vReturn[ 1 ] );
		glBindTexture( GL_TEXTURE_2D, 0 );

		check_opengl();
		return vReturn;
	}

    glm::ivec3 GetImageSize2DArray( GLuint uiTexture )
    {
        glm::ivec3 vReturn(true);
        glBindTexture(GL_TEXTURE_2D_ARRAY, uiTexture);
        glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_WIDTH,  &vReturn[0]);
        glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_HEIGHT, &vReturn[1]);
        glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_DEPTH,  &vReturn[2]);
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

        check_opengl();
        return vReturn;
    }

	glm::ivec3 GetImageSize3D( GLuint uiTexture )
	{
		glm::ivec3 vReturn( true );
		glBindTexture( GL_TEXTURE_3D, uiTexture );

		glGetTexLevelParameteriv( GL_TEXTURE_3D, 0, GL_TEXTURE_WIDTH,  &vReturn[ 0 ] );
		glGetTexLevelParameteriv( GL_TEXTURE_3D, 0, GL_TEXTURE_HEIGHT, &vReturn[ 1 ] );
		glGetTexLevelParameteriv( GL_TEXTURE_3D, 0, GL_TEXTURE_DEPTH,  &vReturn[ 2 ] );
		
		glBindTexture( GL_TEXTURE_3D, 0 );

		check_opengl();

		return vReturn;
	}

	glm::ivec2 GetImageSizeCubeMap( GLuint uiTexture )
	{
		glm::ivec2 vReturn( true );
		glBindTexture( GL_TEXTURE_CUBE_MAP, uiTexture );
		glGetTexLevelParameteriv( GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_TEXTURE_WIDTH,  &vReturn[ 0 ] );
		glGetTexLevelParameteriv( GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_TEXTURE_HEIGHT, &vReturn[ 1 ] );
		glBindTexture( GL_TEXTURE_CUBE_MAP, 0 );

		check_opengl();
		return vReturn;
	}

	GLenum GetInternalFormat( GLenum eTarget, GLuint uiTexture )
	{
		glBindTexture( eTarget, uiTexture );
		GLint iTemp = 0;
		glGetTexLevelParameteriv( eTarget, 0, GL_TEXTURE_INTERNAL_FORMAT, &iTemp );
		glBindTexture( eTarget, 0 );

		check_opengl();
		return iTemp;
	}

	GLenum GetInternalFormat1D( GLuint uiTexture )
	{
		return GetInternalFormat( GL_TEXTURE_1D, uiTexture );
	}

	GLenum GetInternalFormat2D( GLuint uiTexture )
	{
		return GetInternalFormat( GL_TEXTURE_2D, uiTexture );
	}

	GLenum GetInternalFormat3D( GLuint uiTexture )
	{
		return GetInternalFormat( GL_TEXTURE_3D, uiTexture );
	}

	GLenum GetInternalFormatCubeMap( GLuint uiTexture )
	{
		return GetInternalFormat( GL_TEXTURE_CUBE_MAP, uiTexture );
	}

	int GetNumberOfComponents( GLenum eFormat )
	{
		switch( eFormat )
		{
		case GL_R8UI:
		case GL_R16UI:
		case GL_R32UI:
		case GL_R8I:
		case GL_R16I:
		case GL_R32I:
		case GL_R32F:
		case GL_R8:
		case GL_COMPRESSED_RED_RGTC1:
		case GL_COMPRESSED_SIGNED_RED_RGTC1:
			return 1;

		case GL_RG8UI:
		case GL_RG16UI:
		case GL_RG32UI:
		case GL_RG8I:
		case GL_RG16I:
		case GL_RG32I:
		case GL_RG32F:
		case GL_RG8:
		case GL_COMPRESSED_RG_RGTC2:
		case GL_COMPRESSED_SIGNED_RG_RGTC2:
			return 2;

		case GL_RGB8UI:
		case GL_RGB16UI:
		case GL_RGB32UI:
		case GL_RGB8I:
		case GL_RGB16I:
		case GL_RGB32I:
		case GL_RGB32F:
		case GL_RGB8:
		case GL_COMPRESSED_RGB_S3TC_DXT1_EXT:
		case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT:
		case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:
			return 3;

		case GL_RGBA8UI:
		case GL_RGBA16UI:
		case GL_RGBA32UI:
		case GL_RGBA8I:
		case GL_RGBA16I:
		case GL_RGBA32I:
		case GL_RGBA32F:
		case GL_RGBA8:
		case GL_COMPRESSED_RGBA_BPTC_UNORM:
		case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT:
		case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT:
		case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT:
		//case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM​:
			return 4;

		default:
			// This is NOT a complete list!


			throw( std::runtime_error( "Unknown GLenum" ) );
			break;
		}
	}

	std::string GetImageLayoutString(GLenum eFormat)
	{
		switch( eFormat )
		{
		case GL_R8UI:  return "r8ui";
		case GL_R16UI: return "r16ui";
		case GL_R32UI: return "r32ui";
		case GL_R8I:   return "r8i";
		case GL_R16I:  return "r16i";
		case GL_R32I:  return "r32i";
		case GL_R32F:  return "r32f";
		case GL_R8:    return "r8";

		case GL_RG8UI:  return "rg8ui";
		case GL_RG16UI: return "rg16ui";
		case GL_RG32UI: return "rg32ui";
		case GL_RG8I:   return "rg8i";
		case GL_RG16I:  return "rg16i";
		case GL_RG32I:  return "rg32i";
		case GL_RG32F:  return "rg32f";
		case GL_RG8:    return "rg8";

		case GL_RGB8UI:  return "rgb8ui";
		case GL_RGB16UI: return "rgb16ui";
		case GL_RGB32UI: return "rgb32ui";
		case GL_RGB8I:   return "rgb8i";
		case GL_RGB16I:  return "rgb16i";
		case GL_RGB32I:  return "rgb32i";
		case GL_RGB32F:  return "rgb32f";
		case GL_RGB8:    return "rgb8";

		case GL_RGBA8UI:  return "rgba8ui";
		case GL_RGBA16UI: return "rgba16ui";
		case GL_RGBA32UI: return "rgba32ui";
		case GL_RGBA8I:   return "rgba8i";
		case GL_RGBA16I:  return "rgba16i";
		case GL_RGBA32I:  return "rgba32i";
		case GL_RGBA32F:  return "rgba32f";
		case GL_RGBA8:    return "rgba8";

		case GL_R11F_G11F_B10F: return "r11f_g11f_b10f";
		case GL_RGB10_A2UI:     return "rgb10_a2ui";
		case GL_RGB10_A2:       return "rgb10_a2";

		case GL_RGBA16_SNORM: return "rgba16_snorm";
		case GL_RGBA8_SNORM:  return "rgba8_snorm";
		case GL_RG16_SNORM:   return "rg16_snorm";
		case GL_RG8_SNORM:    return "rg8_snorm";
		case GL_R16_SNORM:    return "r16_snorm";
		case GL_R8_SNORM:     return "r8_snorm";


		case GL_COMPRESSED_RED_RGTC1:
		case GL_COMPRESSED_SIGNED_RED_RGTC1:

		case GL_COMPRESSED_RG_RGTC2:
		case GL_COMPRESSED_SIGNED_RG_RGTC2:

		case GL_COMPRESSED_RGB_S3TC_DXT1_EXT:
		case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT:
		case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:

		case GL_COMPRESSED_RGBA_BPTC_UNORM:
		case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT:
		case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT:
		case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT:

			throw( std::runtime_error( "Unsupported format" ) );
			break;

		default:
			// ^^This is NOT a complete list!


			throw( std::runtime_error( "Unknown GLenum" ) );
			break;
		}
	}

	bool IsImageFormatInteger(GLenum eFormat)
	{
		switch( eFormat )
		{
		case GL_R8I:   
		case GL_R16I:  
		case GL_R32I:  

		case GL_RG8I:   
		case GL_RG16I:  
		case GL_RG32I:  

		case GL_RGB8I:   
		case GL_RGB16I:  
		case GL_RGB32I:  

		case GL_RGBA8I:   
		case GL_RGBA16I:  
		case GL_RGBA32I:  
			return true;

		default:
			return false;
		}
	}

	bool IsImageFormatUnsignedInteger(GLenum eFormat)
	{
		switch( eFormat )
		{
		case GL_R8UI:   
		case GL_R16UI:  
		case GL_R32UI:  

		case GL_RG8UI:   
		case GL_RG16UI:  
		case GL_RG32UI:  

		case GL_RGB8UI:   
		case GL_RGB16UI:  
		case GL_RGB32UI:  

		case GL_RGBA8UI:   
		case GL_RGBA16UI:  
		case GL_RGBA32UI:  
			return true;

		default:
			return false;
		}

	}

	GLenum GetCompatibleImageFormat(GLenum eInternalFormat)
	{
		switch (eInternalFormat)
		{
		case GL_RGB8UI:   return GL_RGBA8UI;

		case GL_RGB8I:    return GL_RGBA8I;
		case GL_RGB16UI:  return GL_RGBA16UI;
		case GL_RGB16I:   return GL_RGBA16I;
		case GL_RGB32UI:  return GL_RGBA32UI;
		case GL_RGB32I:   return GL_RGBA32I;

		case GL_RGB8:   return GL_RGBA8;
		case GL_SRGB8:  return GL_INVALID_ENUM;
		case GL_RGB565:  return GL_INVALID_ENUM;
		case GL_RGB8_SNORM:  return GL_RGBA8_SNORM;
		case GL_R11F_G11F_B10F:  return GL_INVALID_ENUM;
		case GL_RGB9_E5:  return GL_INVALID_ENUM;
		case GL_RGB16F:  return GL_RGBA16F;
		case GL_RGB32F:  return GL_RGBA32F;
		
		default:
			return eInternalFormat;
		}

		return eInternalFormat;
	}


	GLuint CreateDefaultSamplerState()
	{
		// Create a default sampler state based on OpenGL specifications
		// http://www.opengl.org/sdk/docs/man/xhtml/glSamplerParameter.xml
		GLuint uiSamplerState = 0;
		glGenSamplers( 1, &uiSamplerState );

		glSamplerParameteri( uiSamplerState, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR );
		glSamplerParameteri( uiSamplerState, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

		glSamplerParameterf( uiSamplerState, GL_TEXTURE_MIN_LOD, -1000.f );
		glSamplerParameterf( uiSamplerState, GL_TEXTURE_MAX_LOD,  1000.f );

		glSamplerParameteri( uiSamplerState, GL_TEXTURE_WRAP_S, GL_REPEAT );
		glSamplerParameteri( uiSamplerState, GL_TEXTURE_WRAP_T, GL_REPEAT );
		glSamplerParameteri( uiSamplerState, GL_TEXTURE_WRAP_R, GL_REPEAT );

		const glm::vec4 vBorderColor( 0.f, 0.f, 0.f, 0.f );
		glSamplerParameterfv( uiSamplerState, GL_TEXTURE_BORDER_COLOR, (float*)(&vBorderColor.data) );

		glSamplerParameteri( uiSamplerState, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL );
		glSamplerParameteri( uiSamplerState, GL_TEXTURE_COMPARE_MODE, GL_NONE );

		glSamplerParameterf( uiSamplerState, GL_TEXTURE_LOD_BIAS, 0.f );
		glSamplerParameteri( uiSamplerState, GL_TEXTURE_MAX_LOD, 1000 );

		check_opengl();

		return uiSamplerState;
	}

	size_t GetBpp( GLenum eInternalFormat )
	{
		switch( eInternalFormat )
		{
		case GL_R8:
		case GL_R8I:
		case GL_R8UI:
			return 8;

		case GL_RG8:
		case GL_RG8UI:
		case GL_RG8I:
		case GL_R16I:
		case GL_R16UI:
			return 16;

		case GL_RGB8:
		case GL_RGB8I:
		case GL_RGB8UI:
			return 24;

		case GL_R32I:
		case GL_R32F:
		case GL_R32UI:
		case GL_RG16UI:
		case GL_RG16I:
		case GL_RGBA8:
		case GL_RGBA8I:
		case GL_RGBA8UI:
			return 32;

		case GL_RGB16I:
		case GL_RGB16UI:
		
			return 48;
		
		case GL_RG32UI:
		case GL_RG32I:
		case GL_RG32F:
		case GL_RGBA16I:
		case GL_RGBA16UI:
			return 64;
		
		
		case GL_RGB32I:
		case GL_RGB32UI:
		case GL_RGB32F:
			return 96;

		

		
		case GL_RGBA32I:
		case GL_RGBA32F:
		case GL_RGBA32UI:
			return 128;

		//case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM​:
		case GL_COMPRESSED_RGBA_BPTC_UNORM:
		case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT:
		case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT:
		case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT:
		case GL_COMPRESSED_RGB_S3TC_DXT1_EXT:
		case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT:
		case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:
		case GL_COMPRESSED_RG_RGTC2:
		case GL_COMPRESSED_SIGNED_RG_RGTC2:
		case GL_COMPRESSED_RED_RGTC1:
		case GL_COMPRESSED_SIGNED_RED_RGTC1:

		default:
			// This is NOT a complete list!
			throw( std::runtime_error( "Unknown size of internal Format" ) );
			break;
		}
	}

	GLenum GetPotentialPixelDataTransferType( GLenum eInternalFormat )
	{
		switch( eInternalFormat ) {
		case GL_R8:
		case GL_RG8:
		case GL_RGB8:
		case GL_RGBA8:
			return GL_UNSIGNED_BYTE;

		case GL_R8I:
		case GL_RG8I:
		case GL_RGB8I:
		case GL_RGBA8I:
			Assert( !"Untested - 8 bit version" );
			return GL_INT;


		case GL_R8UI:
		case GL_RG8UI:
		case GL_RGB8UI:
		case GL_RGBA8UI:
			return GL_UNSIGNED_INT;
			
		case GL_R16I:
		case GL_RG16I:
		case GL_RGB16I:
			Assert( !"Untested" );
			return GL_INT;
		
		case GL_R16UI:
		case GL_RG16UI:
		case GL_RGB16UI:
			Assert( !"Untested" );
			return GL_UNSIGNED_INT;

		case GL_R32I:
		case GL_RG32I:
		case GL_RGB32I:
		case GL_RGBA32I:
			Assert( !"Untested" );
			return GL_INT;

		case GL_R32UI:
		case GL_RG32UI:
		case GL_RGB32UI:
		case GL_RGBA32UI:
			Assert( !"Untested" );
			return GL_UNSIGNED_INT;
		
		case GL_R32F:
		case GL_RG32F:
		case GL_RGB32F:
		case GL_RGBA32F:
			Assert( !"Untested" );
			return GL_FLOAT;

		case GL_RGBA16I:
		case GL_RGBA16UI:
			Assert( !"Untested, maybe use GL_BYTES" );
			return GL_INT;



		//case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM​:
		case GL_COMPRESSED_RGBA_BPTC_UNORM:
		case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT:
		case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT:
		case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT:
		case GL_COMPRESSED_RGB_S3TC_DXT1_EXT:
		case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT:
		case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:
		case GL_COMPRESSED_RG_RGTC2:
		case GL_COMPRESSED_SIGNED_RG_RGTC2:
		case GL_COMPRESSED_RED_RGTC1:
		case GL_COMPRESSED_SIGNED_RED_RGTC1:
			Assert( !"Untested, maybe use GL_BYTES" );
			return GL_INT;
		default:
			{
				throw( std::runtime_error( "Unknown GLenum" ) );
			}
			break;
		}
	}

	GLuint CreateSimilarTexture( GLenum eTarget, GLuint uiTexture, void* pPixelData )
	{
		Assert( glIsTexture( uiTexture ) == GL_TRUE );

		GLuint uiNewTexture = 0;
		glGenTextures( 1, &uiNewTexture );

		switch( eTarget ) {
		case GL_TEXTURE_1D:
			{
				auto iSize = GetImageSize1D( uiTexture );

				GLenum eInternalFormat = GetInternalFormat1D( uiTexture );
				GLenum eChannelFormat = GetPotentialChannelFormat( eInternalFormat );
				GLenum eOGLType = GetPotentialPixelDataTransferType( eInternalFormat );

				glBindTexture( GL_TEXTURE_1D, uiNewTexture );
				glTexImage1D( GL_TEXTURE_1D,
							  0,							// mipmap-level
							  eInternalFormat,
							  iSize,
							  0,							// border
							  eChannelFormat,				// such as RGB, RGBA, BGRA, etc...
							  eOGLType,
							  pPixelData );
			}
			break;
		case GL_TEXTURE_2D:
			{
				auto iSize = GetImageSize2D( uiTexture );

				GLenum eInternalFormat = GetInternalFormat2D( uiTexture );
				GLenum eChannelFormat = GetPotentialChannelFormat( eInternalFormat );
				GLenum eOGLType = GetPotentialPixelDataTransferType( eInternalFormat );

				glBindTexture( GL_TEXTURE_2D, uiNewTexture );
				glTexImage2D( GL_TEXTURE_2D,
							  0,							// mipmap-level
							  eInternalFormat,
							  iSize[ 0 ],
							  iSize[ 1 ],
							  0,							// border
							  eChannelFormat,				// such as RGB, RGBA, BGRA, etc...
							  eOGLType,
							  pPixelData );
			}
			break;
		case GL_TEXTURE_3D:
			{
				auto iSize = GetImageSize3D( uiTexture );

				GLenum eInternalFormat = GetInternalFormat3D( uiTexture );
				GLenum eChannelFormat = GetPotentialChannelFormat( eInternalFormat );
				GLenum eOGLType = GetPotentialPixelDataTransferType( eInternalFormat );

				glBindTexture( GL_TEXTURE_3D, uiNewTexture );
				glTexImage3D( GL_TEXTURE_3D,
							  0,							// mipmap-level
							  eInternalFormat,
							  iSize[ 0 ],
							  iSize[ 1 ],
							  iSize[ 2 ],
							  0,							// border
							  eChannelFormat,				// such as RGB, RGBA, BGRA, etc...
							  eOGLType,
							  pPixelData );
			}
			break;
		default:
			throw( std::runtime_error( "Invalid Parameter" ) );
			break;
		}
		check_opengl();
		return uiNewTexture;
	}


	// generating mip maps with DSA
	// glGenerateTextureMipmapEXT( texobject, GL_TEXTURE_2D );

}
