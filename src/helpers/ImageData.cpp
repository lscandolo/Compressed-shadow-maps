#include "common.h"
#include "helpers/ImageData.h"

#include <glm/glm.hpp>
#include <set>

ImageFormatDescriptor GetImageFormatDescriptor( const ImageFormat& eFormat )
{
	const ImageFormatDescriptor desc[] =
	{
		//size_t		m_sBlockSize;
		//glm::ivec3	m_BlockDimensions;
		//bool			m_bIsCompressed;
		//size_t		m_sNumberOfComponents;
		{ 0, glm::ivec3( 0, 0, 0 ), false, 0},	// FORMAT_UNKNOWN,

		// unsigned
		{ 1, glm::ivec3( 1, 1, 1 ), false, 1},	// R8UI,
		{ 2, glm::ivec3( 1, 1, 1 ), false, 2},	// RG8UI,
		{ 3, glm::ivec3( 1, 1, 1 ), false, 3},	// RGB8UI,
		{ 4, glm::ivec3( 1, 1, 1 ), false, 4},	// RGBA8UI,

		{ 2, glm::ivec3( 1, 1, 1 ), false, 1},	// R16UI,
		{ 4, glm::ivec3( 1, 1, 1 ), false, 2},	// RG16UI,
		{ 6, glm::ivec3( 1, 1, 1 ), false, 3},	// RGB16UI,
		{ 8, glm::ivec3( 1, 1, 1 ), false, 4},	// RGBA16UI,

		{ 4, glm::ivec3( 1, 1, 1 ), false, 1},	// R32UI,
		{ 8, glm::ivec3( 1, 1, 1 ), false, 2},	// RG32UI,
		{12, glm::ivec3( 1, 1, 1 ), false, 3},	// RGB32UI,
		{16, glm::ivec3( 1, 1, 1 ), false, 4},	// RGBA32UI,

		// signed
		{ 1, glm::ivec3( 1, 1, 1 ), false, 1},	// R8I,
		{ 2, glm::ivec3( 1, 1, 1 ), false, 2},	// RG8I,
		{ 3, glm::ivec3( 1, 1, 1 ), false, 3},	// RGB8I,
		{ 4, glm::ivec3( 1, 1, 1 ), false, 4},	// RGBA8I,

		{ 2, glm::ivec3( 1, 1, 1 ), false, 1},	// R16I,
		{ 4, glm::ivec3( 1, 1, 1 ), false, 2},	// RG16I,
		{ 6, glm::ivec3( 1, 1, 1 ), false, 3},	// RGB16I,
		{ 8, glm::ivec3( 1, 1, 1 ), false, 4},	// RGBA16I,

		{ 4, glm::ivec3( 1, 1, 1 ), false, 1},	// R32I,
		{ 8, glm::ivec3( 1, 1, 1 ), false, 2},	// RG32I,
		{12, glm::ivec3( 1, 1, 1 ), false, 3},	// RGB32I,
		{16, glm::ivec3( 1, 1, 1 ), false, 4},	// RGBA32I,

		// before was block size 2,4,6,8:
		{ 2, glm::ivec3(1, 1, 1), false, 1},	// R16F,
		{ 4, glm::ivec3(1, 1, 1), false, 2},	// RG16F,
		{ 6, glm::ivec3(1, 1, 1), false, 3},	// RGB16F,
		{ 8, glm::ivec3(1, 1, 1), false, 4},	// RGBA16F,

		// before was block size 2,4,6,8:
		{ 4, glm::ivec3( 1, 1, 1 ), false, 1},	// R32F,
		{ 8, glm::ivec3( 1, 1, 1 ), false, 2},	// RG32F,
		{12, glm::ivec3( 1, 1, 1 ), false, 3},	// RGB32F,
		{16, glm::ivec3( 1, 1, 1 ), false, 4},	// RGBA32F,

		// before was block size 2,4,6,8:
		{ 8, glm::ivec3(1, 1, 1), false, 1},	// R64F,
		{16, glm::ivec3(1, 1, 1), false, 2},	// RG64F,
		{24, glm::ivec3(1, 1, 1), false, 3},	// RGB64F,
		{32, glm::ivec3(1, 1, 1), false, 4},	// RGBA64F,

		{ 1, glm::ivec3( 1, 1, 1 ), false, 1},	// R8_UNORM,
		{ 2, glm::ivec3( 1, 1, 1 ), false, 2},	// RG8_UNORM,
		{ 3, glm::ivec3( 1, 1, 1 ), false, 3},	// RGB8_UNORM,
		{ 4, glm::ivec3( 1, 1, 1 ), false, 4},	// RGBA8_UNORM,

		// DXT
		{ 8, glm::ivec3( 4, 4, 1 ), true, 1},		// R_ATI_1N_UNORM
		{16, glm::ivec3( 4, 4, 1 ), true, 2},		// COMPRESSED_RG_BC5_UNORM
		{16, glm::ivec3( 4, 4, 1 ), true, 3},		// COMPRESSED_RG_BC7_UNORM


		{ 8, glm::ivec3( 4, 4, 1 ), true, 3},		// COMPRESSED_RGB_S3TC_DXT1_EXT
		{ 8, glm::ivec3( 4, 4, 1 ), true, 4},		// COMPRESSED_RGBA_S3TC_DXT1_EXT,
		{16, glm::ivec3( 4, 4, 1 ), true, 4},		// COMPRESSED_RGBA_S3TC_DXT3_EXT,
		{16, glm::ivec3( 4, 4, 1 ), true, 4},		// COMPRESSED_RGBA_S3TC_DXT5_EXT,

		{ 0, glm::ivec3( 1, 1, 1 ), false, 0},	// FORMAT_MAX
	};

	static_assert( ARRAY_SIZE( desc ) == ( int )ImageFormat::FORMAT_MAX+1, "array sizes should match" );
	return desc[ ( int ) eFormat ];
}

bool IsIntegerFormat(ImageFormat eImgFormat)
{
	static const std::set<ImageFormat> integerFormats{
		// unsigned int
		ImageFormat::R8UI, ImageFormat::RG8UI, ImageFormat::RGB8UI, ImageFormat::RGBA8UI,

		// short formats (16 bit)
		ImageFormat::R16UI, ImageFormat::RG16UI, ImageFormat::RGB16UI, ImageFormat::RGBA16UI,

		// (32 bit)
		ImageFormat::R32UI, ImageFormat::RG32UI, ImageFormat::RGB32UI, ImageFormat::RGBA32UI,

		// signed integer (8 bit)
		ImageFormat::R8I, ImageFormat::RG8I, ImageFormat::RGB8I, ImageFormat::RGBA8I,

		// signed integer (16 bit)
		ImageFormat::R16I, ImageFormat::RG16I, ImageFormat::RGB16I, ImageFormat::RGBA16I,

		// signed integer (32 bit)
		ImageFormat::R32I, ImageFormat::RG32I,  ImageFormat::RGB32I, ImageFormat::RGBA32I,

		// unorm formats
		ImageFormat::R8_UNORM, ImageFormat::RG8_UNORM, ImageFormat::RGB8_UNORM, ImageFormat::RGBA8_UNORM,
	};

	return integerFormats.find(eImgFormat) != integerFormats.end();

}


//////////////////////////////////////////////////////////////////////////
// ImageData
//////////////////////////////////////////////////////////////////////////
ImageData::ImageData() :
	m_eFormat( ImageFormat::FORMAT_UNKNOWN ),
	m_eChannelFormat( ImageChannelFormat::FORMAT_UNKNOWN ),
	m_eType( ImageType::IMAGETYPE_UNKNOWN ),
	m_LayerSize( true ),
	m_sNumberOfLayers( 0 ),
	m_sNumberOfFaces( 0 ),
	m_sNumberOfMipMapLevels( 0 ),
	m_sBytePerPixel( 0 )
{}

size_t ImageData::GetLevelSize( size_t sLevel ) const
{
	auto form = GetImageFormatDescriptor( m_eFormat );

	// with each mipmap level the width/height decreases by half
	glm::uvec2 dim( m_LayerSize );
	dim[ 0 ] >>= sLevel;
	dim[ 1 ] >>= sLevel;

	size_t sModX = dim[ 0 ] % form.m_BlockDimensions[ 0 ];
	size_t sModY = dim[ 1 ] % form.m_BlockDimensions[ 1 ];
	//size_t sModY = dim[ 2 ] % form.m_BlockDimensions[ 2 ];

	size_t sReturn = form.m_sBlockSize;

	// In case the width/height is not a multiple of the block dimension
	sReturn *= ( sModX != 0 ? dim[ 0 ] + form.m_BlockDimensions[ 0 ] - sModX : dim[ 0 ] );
	sReturn *= ( sModY != 0 ? dim[ 1 ] + form.m_BlockDimensions[ 1 ] - sModY : dim[ 1 ] );

	sReturn /= form.m_BlockDimensions[ 0 ];
	sReturn /= form.m_BlockDimensions[ 1 ];

	return sReturn;
}
