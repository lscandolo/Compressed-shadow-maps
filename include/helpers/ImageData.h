#pragma once

#include "common.h"

#include <glm/glm.hpp>

class ImageFormatDescriptor
{
public:
	size_t		m_sBlockSize;
	glm::ivec3	m_BlockDimensions;
	bool		m_bIsCompressed;
	size_t		m_sNumberOfComponents;
};
 
enum class ImageFormat
{
	FORMAT_UNKNOWN,

	// unsigned int
	R8UI,
	RG8UI,
	RGB8UI,
	RGBA8UI,

	// short formats (16 bit)
	R16UI,
	RG16UI,
	RGB16UI,
	RGBA16UI,

	// (32 bit)
	R32UI,
	RG32UI,
	RGB32UI,
	RGBA32UI,

	// signed integer (8 bit)
	R8I,
	RG8I,
	RGB8I,
	RGBA8I,

	// signed integer (16 bit)
	R16I,
	RG16I,
	RGB16I,
	RGBA16I,

	// signed integer (32 bit)
	R32I,
	RG32I,
	RGB32I,
	RGBA32I,

	// floating point formats (16 bit)
	R16F,
	RG16F,
	RGB16F,
	RGBA16F,

	// floating point formats (32 bit)
	R32F,
	RG32F,
	RGB32F,
	RGBA32F,

	// double precision floating point formats
	R64F,
	RG64F,
	RGB64F,
	RGBA64F,

	// unorm formats
	R8_UNORM,
	RG8_UNORM,
	RGB8_UNORM,
	RGBA8_UNORM,

	COMPRESSED_BC4_UNORM,		///< ATI Red Channel only (e.g. DXT10 format : DXGI_FORMAT_BC4_UNORM) 
	COMPRESSED_RG_BC5_UNORM,	///< (e.g. DXT10 format : DXGI_FORMAT_BC5_UNORM)
	COMPRESSED_RGBA_BC7_UNORM,	///< (e.g. DXT10 format : DXGI_FORMAT_BC7_UNORM)		

	COMPRESSED_RGB_S3TC_DXT1,
	COMPRESSED_RGBA_S3TC_DXT1,
	COMPRESSED_RGBA_S3TC_DXT3,
	COMPRESSED_RGBA_S3TC_DXT5,

	FORMAT_MAX
};

enum class ImageChannelFormat
{
	FORMAT_UNKNOWN,

	RED,
	GREEN,
	BLUE,
	ALPHA,

	RG,

	RGB,
	RGBA,

	BGR,
	BGRA,

	FORMAT_MAX,
};

enum class ImageType
{
	IMAGETYPE_UNKNOWN,

	BYTE,
	UNSIGNED_BYTE,

	SHORT,
	UNSIGNED_SHORT,

	INT,
	UNSIGNED_INT,

	LONG,
	UNSIGNED_LONG,

	FLOAT,
	DOUBLE,

	IMAGETYPE_MAX,
};


ImageFormatDescriptor	GetImageFormatDescriptor(const ImageFormat& eFormat);
bool					IsIntegerFormat(ImageFormat eImgFormat);

class ImageData
{
public:
	ImageData();

	ImageFormat							m_eFormat;
	ImageChannelFormat					m_eChannelFormat;				///< format of the pixel data in m_Data (basically the order of components)
	ImageType							m_eType;						///< type of the pixel data in m_Data (basically the value of each channel)

	glm::uvec2  						m_LayerSize;					///< The 2D size of one layer
	std::size_t							m_sNumberOfLayers;				///< only relevant for 3D Textures
	std::size_t							m_sNumberOfFaces;				///< only relevant for cube maps
	std::size_t							m_sNumberOfMipMapLevels;
	std::size_t							m_sBytePerPixel;
	std::vector< byte >				    m_Data;							///< All image data in one block


	byte*						    	data()
	{
		return m_Data.data();
	}

	const byte*						    data() const
	{
		return m_Data.data();
	}

	size_t								GetLevelSize(size_t sLevel) const;
};
