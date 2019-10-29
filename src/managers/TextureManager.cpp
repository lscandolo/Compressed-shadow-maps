#include "managers/TextureManager.h"
#include "helpers/OpenGLHelpers.h"

#include <FreeImage.h>
#include <algorithm>

using namespace GLHelpers;

TextureData::TextureData() :
	gltex(nullptr)
{}

TextureManagerInstance::TextureManagerInstance()
{
	
}

TextureManagerInstance::~TextureManagerInstance()
{
	clear();
}

bool TextureManagerInstance::clear()
{
	textures.clear();
	tex1d.clear();
	tex2d.clear();
	tex3d.clear();

	return SUCCESS;
}

bool TextureManagerInstance::deleteTexture(std::string name)
{
	if (!hasTexture(name)) return ERROR_RESOURCE_NOT_FOUND;

	TextureData& td = getTexture(name);
	textures.erase(name);

	return SUCCESS;
}

int TextureManagerInstance::loadImageData(std::string filename, bool RGBtoRGBA)
{
	if (hasTexture(filename)) return SUCCESS;

	if (filename.size() < 4) return ERROR_READING_FILE;

	std::string suffix_4 = filename.substr(filename.size() - 4, 4);
	std::transform(suffix_4.begin(), suffix_4.end(), suffix_4.begin(), ::tolower);

	std::string suffix_5 = filename.substr(filename.size() - 5, 5);
	std::transform(suffix_5.begin(), suffix_5.end(), suffix_5.begin(), ::tolower);

	bool loaded_success = false;
	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;

	if (suffix_5 == ".jpeg" || suffix_4 == ".jpg") {
		fif = FIF_JPEG;
	}
	else if (suffix_4 == ".png") {
		fif = FIF_PNG;
	}
	else if (suffix_4 == ".tif" || suffix_5 == ".tiff") {
		fif = FIF_TIFF;
	}
	else if (suffix_4 == ".tga") {
		fif = FIF_TARGA;
	}
	else if (suffix_4 == ".bmp") {
		fif = FIF_BMP;
	}
	else if (suffix_4 == ".hdr") {
		fif = FIF_HDR;
	}
	else if (suffix_4 == ".exr") {
		fif = FIF_EXR;
	}

	FIBITMAP* fib = FreeImage_Load(fif, filename.c_str(), 0);

	TextureData td;
	if (!fib) return ERROR_LOADING_RESOURCE;

	BITMAPINFOHEADER* infoheader = FreeImage_GetInfoHeader(fib);
	BITMAPINFO* info = FreeImage_GetInfo(fib);

	td.data.m_LayerSize.x = FreeImage_GetWidth(fib);
	td.data.m_LayerSize.y = FreeImage_GetHeight(fib);
	td.data.m_sNumberOfLayers = 1;
	td.data.m_sNumberOfMipMapLevels = 1;
	td.data.m_sNumberOfFaces = 1;
	td.data.m_sBytePerPixel = FreeImage_GetBPP(fib) / 8;

	FREE_IMAGE_TYPE type = FreeImage_GetImageType(fib);

	bool addAlpha = false;

	switch (type) {

	case FIT_BITMAP:	//! standard image			: 1-, 4-, 8-, 16-, 24-, 32-bit
		td.data.m_eType = ImageType::BYTE;
		if (td.data.m_sBytePerPixel == 1) {
			td.data.m_eChannelFormat = ImageChannelFormat::RED;
			td.data.m_eFormat = ImageFormat::R8_UNORM;
		}
		else if (td.data.m_sBytePerPixel == 2) {
			td.data.m_eChannelFormat = ImageChannelFormat::RG;
			td.data.m_eFormat = ImageFormat::RG8_UNORM;
		}
		else if (td.data.m_sBytePerPixel == 3) {
			if (RGBtoRGBA) {
				addAlpha = true;
				td.data.m_sBytePerPixel = 4;
				td.data.m_eChannelFormat = ImageChannelFormat::RGBA;
				td.data.m_eFormat = ImageFormat::RGBA8_UNORM;
			}
			else {
				td.data.m_eChannelFormat = ImageChannelFormat::RGB;
				td.data.m_eFormat = ImageFormat::RGB8_UNORM;
			}
		}
		else if (td.data.m_sBytePerPixel == 4 && !addAlpha) {
			td.data.m_eChannelFormat = ImageChannelFormat::RGBA;
			td.data.m_eFormat = ImageFormat::RGBA8_UNORM;
		}
		else {
			goto jpg_load_end;
		}

		break;
	case FIT_UINT16:	//! array of unsigned short	: unsigned 16-bit
		td.data.m_eChannelFormat = ImageChannelFormat::RED;
		td.data.m_eType = ImageType::UNSIGNED_SHORT;
		td.data.m_eFormat = ImageFormat::R16UI;

		break;
	case FIT_INT16:     //! array of short			: signed 16-bit
		td.data.m_eChannelFormat = ImageChannelFormat::RED;
		td.data.m_eType = ImageType::SHORT;
		td.data.m_eFormat = ImageFormat::R16I;

		break;
	case FIT_UINT32:	//! array of unsigned long	: unsigned 32-bit
		td.data.m_eChannelFormat = ImageChannelFormat::RED;
		td.data.m_eType = ImageType::UNSIGNED_INT;
		td.data.m_eFormat = ImageFormat::R32UI;

		break;
	case FIT_INT32:	    //! array of long			: signed 32-bit
		td.data.m_eChannelFormat = ImageChannelFormat::RED;
		td.data.m_eType = ImageType::INT;
		td.data.m_eFormat = ImageFormat::R32I;

		break;
	case FIT_FLOAT:	    //! array of float			: 32-bit IEEE floating point
		td.data.m_eChannelFormat = ImageChannelFormat::RED;
		td.data.m_eType = ImageType::FLOAT;
		td.data.m_eFormat = ImageFormat::R32F;

		break;
	case FIT_DOUBLE:	//! array of double			: 64-bit IEEE floating point
		td.data.m_eChannelFormat = ImageChannelFormat::RED;
		td.data.m_eType = ImageType::DOUBLE;
		td.data.m_eFormat = ImageFormat::R64F;

		break;
	case FIT_RGB16:	    //! 48-bit RGB image			: 3 x 16-bit // THIS IS UPGRADED TO FLOAT
		if (RGBtoRGBA) {
			addAlpha = true;
			td.data.m_sBytePerPixel = 16;
			td.data.m_eChannelFormat = ImageChannelFormat::RGBA;
			td.data.m_eType = ImageType::FLOAT;
			td.data.m_eFormat = ImageFormat::RGBA16F;
		} else {
			td.data.m_sBytePerPixel = 12;
			td.data.m_eChannelFormat = ImageChannelFormat::RGB;
			td.data.m_eType = ImageType::FLOAT;
			td.data.m_eFormat = ImageFormat::RGB16F;
		}

		break;
	case FIT_RGBA16:	//! 64-bit RGBA image		: 4 x 16-bit
		td.data.m_sBytePerPixel = 16;
		td.data.m_eChannelFormat = ImageChannelFormat::RGBA;
		td.data.m_eType = ImageType::FLOAT;
		td.data.m_eFormat = ImageFormat::RGBA16F;

		break;
	case FIT_RGBF:	    //! 96-bit RGB float image	: 3 x 32-bit IEEE floating point
		if (RGBtoRGBA) {
			addAlpha = true;
			td.data.m_sBytePerPixel = 16;
			td.data.m_eChannelFormat = ImageChannelFormat::RGBA;
			td.data.m_eType = ImageType::FLOAT;
			td.data.m_eFormat = ImageFormat::RGBA32F;
		}
		else {
			td.data.m_eChannelFormat = ImageChannelFormat::RGB;
			td.data.m_eType = ImageType::FLOAT;
			td.data.m_eFormat = ImageFormat::RGB32F;
		}
		break;
	case FIT_RGBAF:	    //! 128-bit RGBA float image	: 4 x 32-bit IEEE floating point
		td.data.m_eChannelFormat = ImageChannelFormat::RGBA;
		td.data.m_eType = ImageType::FLOAT;
		td.data.m_eFormat = ImageFormat::RGBA32F;

		break;

	default:
	case FIT_COMPLEX:	//! array of FICOMPLEX		: 2 x 64-bit IEEE floating point
	case FIT_UNKNOWN:
		goto jpg_load_end;
	}

	if (td.data.m_LayerSize.x == 0 || td.data.m_LayerSize.y == 0) goto jpg_load_end;

	int channelcount = 0;
	switch (td.data.m_eChannelFormat) {
	case ImageChannelFormat::RED:
	case ImageChannelFormat::GREEN:
	case ImageChannelFormat::BLUE:
	case ImageChannelFormat::ALPHA:
		channelcount = 1; break;
	case ImageChannelFormat::RG:
		channelcount = 2; break;
	case ImageChannelFormat::BGR:
	case ImageChannelFormat::RGB:
		channelcount = 3; break;
	case ImageChannelFormat::BGRA:
	case ImageChannelFormat::RGBA:
		channelcount = 4; break;
	default: break;
	}


	size_t datasize = td.data.m_sBytePerPixel * td.data.m_LayerSize.x * td.data.m_LayerSize.y * td.data.m_sNumberOfLayers * td.data.m_sNumberOfFaces;
	td.data.m_Data.resize(datasize);

	//	RGBQUAD color;

	size_t scanline_size = td.data.m_LayerSize.x * td.data.m_sBytePerPixel;

	for (unsigned int y = 0; y < td.data.m_LayerSize.y; ++y) {

		byte       *i_bytes   = (byte     *)(FreeImage_GetScanLine(fib, y));
		int16_t    *i_shorts  = (int16_t  *)(FreeImage_GetScanLine(fib, y));
		uint16_t   *i_ushorts = (uint16_t *)(FreeImage_GetScanLine(fib, y));
		int32_t    *i_ints    = (int32_t  *)(FreeImage_GetScanLine(fib, y));
		uint32_t   *i_uints   = (uint32_t *)(FreeImage_GetScanLine(fib, y));
		float      *i_floats  = (float    *)(FreeImage_GetScanLine(fib, y));
		double     *i_doubles = (double   *)(FreeImage_GetScanLine(fib, y));

		byte*    o_bytes   = (byte    *)(td.data.m_Data.data() + y * scanline_size);
		int16_t* o_shorts  = (int16_t *)(td.data.m_Data.data() + y * scanline_size);
		int32_t* o_ints    = (int32_t *)(td.data.m_Data.data() + y * scanline_size);
		float*   o_floats  = (float   *)(td.data.m_Data.data() + y * scanline_size);
		double*  o_doubles = (double  *)(td.data.m_Data.data() + y * scanline_size);;

		switch (td.data.m_eFormat) {
		case ImageFormat::R8_UNORM:
		case ImageFormat::RG8_UNORM:
		case ImageFormat::RGB8_UNORM:
		case ImageFormat::RGBA8_UNORM:
		{
			RGBQUAD c;
			for (unsigned int x = 0; x < td.data.m_LayerSize.x; ++x) {
				FreeImage_GetPixelColor(fib, x, y, &c);
				if (channelcount >= 1) o_bytes[channelcount * x + 0] = c.rgbRed;
				if (channelcount >= 2) o_bytes[channelcount * x + 1] = c.rgbGreen;
				if (channelcount >= 3) o_bytes[channelcount * x + 2] = c.rgbBlue;
				if (channelcount >= 4) o_bytes[channelcount * x + 3] = addAlpha ? std::numeric_limits<byte>::max() : c.rgbReserved;
			}
		} break;

		case ImageFormat::R16UI:
		case ImageFormat::R16I:
		case ImageFormat::RGB16UI:
		case ImageFormat::RGB16I:
		case ImageFormat::RGBA16UI:
		case ImageFormat::RGBA16I: 
		{
			if (addAlpha && (td.data.m_eFormat == ImageFormat::RGBA16UI || td.data.m_eFormat == ImageFormat::RGBA16I)) {
				for (unsigned int x = 0; x < td.data.m_LayerSize.x; ++x) {
					o_shorts[4 * x + 0] = i_shorts[3 * x + 0];
					o_shorts[4 * x + 1] = i_shorts[3 * x + 1];
					o_shorts[4 * x + 2] = i_shorts[3 * x + 2];
					o_shorts[4 * x + 3] = 1;
				}
			} else {
				std::memcpy(o_shorts, i_shorts, scanline_size);
			}
		} break;

		case ImageFormat::RGBA16F:
		{
			if (addAlpha && td.data.m_eFormat == ImageFormat::RGBA16F) 
			{
				for (unsigned int x = 0; x < td.data.m_LayerSize.x; ++x) {
					o_floats[4 * x + 0] = i_ushorts[3 * x + 0] / float(0xffff);
					o_floats[4 * x + 1] = i_ushorts[3 * x + 1] / float(0xffff);
					o_floats[4 * x + 2] = i_ushorts[3 * x + 2] / float(0xffff);
					o_floats[4 * x + 3] = 1;
				}
			} else {
				for (unsigned int x = 0; x < td.data.m_LayerSize.x; ++x) {
					o_floats[4 * x + 0] = i_ushorts[4 * x + 0] / float(0xffff);
					o_floats[4 * x + 1] = i_ushorts[4 * x + 1] / float(0xffff);
					o_floats[4 * x + 2] = i_ushorts[4 * x + 2] / float(0xffff);
					o_floats[4 * x + 3] = i_ushorts[4 * x + 3] / float(0xffff);
				}
			}

		} break;
		case ImageFormat::RGB16F:
		{
			for (unsigned int x = 0; x < td.data.m_LayerSize.x; ++x) {
				o_floats[3 * x + 0] = i_ushorts[3 * x + 0] / float(0xffff);
				o_floats[3 * x + 1] = i_ushorts[3 * x + 1] / float(0xffff);
				o_floats[3 * x + 2] = i_ushorts[3 * x + 2] / float(0xffff);
			}
		} break;


		case ImageFormat::R32UI:
		case ImageFormat::R32I:
		{
			std::memcpy(o_ints, i_ints, scanline_size);
		} break;

		case ImageFormat::R32F:
		case ImageFormat::RGB32F:
		case ImageFormat::RGBA32F:
		{
			if (td.data.m_eFormat == ImageFormat::RGBA32F && addAlpha) {
				for (unsigned int x = 0; x < td.data.m_LayerSize.x; ++x) {
					o_floats[4 * x + 0] = i_floats[3 * x + 0];
					o_floats[4 * x + 1] = i_floats[3 * x + 1];
					o_floats[4 * x + 2] = i_floats[3 * x + 2];
					o_floats[4 * x + 3] = 1.f;
				}
			} else {
				std::memcpy(o_floats, i_floats, scanline_size);
			}
		} break;

		case ImageFormat::R64F:
		{
			std::memcpy(o_doubles, i_doubles, scanline_size);
		} break;

		default:
			goto jpg_load_end;
		}
	}

	loaded_success = true;

jpg_load_end:
	FreeImage_Unload(fib);

	if (!loaded_success) return ERROR_LOADING_RESOURCE;

	textures[filename] = td;
	td.name = filename;

	return SUCCESS;
}

int TextureManagerInstance::loadTexture(std::string filename, bool generate_mipmap, bool RGBtoRGBA)
{
	if (hasTexture(filename) && getTexture(filename).gltex) return SUCCESS;

	if (!hasTexture(filename)) {
		int result = loadImageData(filename, RGBtoRGBA);
		if (result != SUCCESS) return result;
	}

	TextureData& td = getTexture(filename);

	tex2d.push_back(create_object<TextureObject2D>());
	TextureObject2D& newtex = tex2d.back();

	newtex->loadFromImageData(td.data, false, true);
	if (generate_mipmap) {
		newtex->generateMipmap();
		newtex->setParameterf(GL_TEXTURE_MAX_ANISOTROPY, 8.f);
	}
	td.gltex = newtex;

	return SUCCESS;
}
 
int TextureManagerInstance::saveTexture(ImageFileFormat format, TextureObject2D tex, std::string filename)
{
	if (!tex->id || !tex->width || !tex->height) return ERROR_INVALID_PARAMETER;

	size_t pitch = tex->bpp / 8 * tex->width;
	size_t texdatasize = pitch * tex->height;

	// Fix for half precision (converted to full precision)
	switch (tex->internalFormat) {
	case GL_R16F:
	case GL_RG16F:
	case GL_RGB16F:
	case GL_RGBA16F:
		texdatasize*= 2;
		break;
	}

	FREE_IMAGE_FORMAT fiformat;
	switch (format)
	{
	default:
	case ImageFileFormat::EXR:
		fiformat = FREE_IMAGE_FORMAT::FIF_EXR; break;
	case ImageFileFormat::TIFF:
		fiformat = FREE_IMAGE_FORMAT::FIF_TIFF; break;
	}

	std::vector<uint8_t> texdata(texdatasize);
	tex->downloadTextureData(texdata.data(), 0, GLsizei(texdatasize));

	return _saveTexture32Bits(fiformat, filename, texdata, size2D(tex->width, tex->height), tex->internalFormat, tex->channels, tex->bpp);
}

int TextureManagerInstance::saveTexture(ImageFileFormat format, TextureObject3D tex, int layer, std::string filename)
{
	if (!tex->id || !tex->width || !tex->height) return ERROR_INVALID_PARAMETER;

	size_t pitch = tex->bpp / 8 * tex->width;
	size_t texdatasize = pitch * tex->height;

	// Fix for half precision (converted to full precision)
	switch (tex->internalFormat) {
	case GL_R16F:
	case GL_RG16F:
	case GL_RGB16F:
	case GL_RGBA16F:
		texdatasize *= 2;
		break;
	}

	std::vector<uint8_t> texdata(texdatasize);
	tex->downloadTextureLayerData(texdata.data(), layer, 0, GLsizei(texdatasize));

	FREE_IMAGE_FORMAT fiformat;
	switch (format) 
	{
	default:
	case ImageFileFormat::EXR:
		fiformat = FREE_IMAGE_FORMAT::FIF_EXR; break;
	case ImageFileFormat::TIFF:
		fiformat = FREE_IMAGE_FORMAT::FIF_TIFF; break;
	}

	return _saveTexture32Bits(fiformat, filename, texdata, size2D(tex->width, tex->height), tex->internalFormat, tex->channels, tex->bpp);
}

int TextureManagerInstance::_saveTexture32Bits(enum FREE_IMAGE_FORMAT format, const std::string& filename, std::vector<uint8_t>& texdata, size2D texsize, GLenum internalformat, GLuint channels, GLuint bpp)
{

	// Fix for half precision (converted to full precision)
	switch (internalformat) {
	case GL_R16F:
	case GL_RG16F:
	case GL_RGB16F:
	case GL_RGBA16F:
		bpp *= 2;
		break;
	}
	
	size_t pitch = bpp / 8 * texsize.width;

	size_t texdatasize = texdata.size();
	uint8_t* imgdata   = texdata.data();

	bool save_float_data = false;

	std::vector<float> floatdata;
	FREE_IMAGE_TYPE image_type;

	switch (internalformat) {
	case GL_RGBA8:
	case GL_RGB8:
	case GL_RG8:
	case GL_R8:
	{
		// Convert to floating point
		channels = channels == 2 ? 3 : channels; // Can only handle R, RGB and RGBA
		bpp = channels * 8 * sizeof(float);
		pitch = bpp / 8 * texsize.width;
		texdatasize = pitch * texsize.height;
		int float_count = channels * texsize.width * texsize.height;
		floatdata.resize(float_count);

		if (internalformat == GL_R8)    image_type = FIT_FLOAT;
		if (internalformat == GL_RG8)   image_type = FIT_RGBF;
		if (internalformat == GL_RGB8)  image_type = FIT_RGBF;
		if (internalformat == GL_RGBA8) image_type = FIT_RGBAF;

		int out_index = 0;
		int in_index = 0;

		for (unsigned int j = 0; j < texsize.height; j++) {
			for (unsigned int i = 0; i < texsize.width; i++) {

				if (true) {
					floatdata[out_index++] = texdata[in_index++] / 255.f;
				}

				if (image_type == FIT_RGBF || image_type == FIT_RGBAF) {
					floatdata[out_index++] = texdata[in_index++] / 255.f;
				}

				if (image_type == FIT_RGBF || image_type == FIT_RGBAF) {
					if (internalformat != GL_RG8) floatdata[out_index++] = texdata[in_index++] / 255.f;
					if (internalformat == GL_RG8) floatdata[out_index++] = 0.f;
				}

				if (image_type == FIT_RGBAF) {
					floatdata[out_index++] = texdata[in_index++] / 255.f;
				}
			}
			imgdata = reinterpret_cast<uint8_t*>(floatdata.data());

		}
		save_float_data = false;
		break;
	}

	//case GL_R16UI:
	//	image_type = FIT_UINT16;  //! array of unsigned short	: unsigned 16-bit
	//	break;

	//case GL_R16I:
	//	image_type = FIT_INT16;   //! array of short			: signed 16-bit
	//	break;

	//case GL_R32UI:
	//	image_type = FIT_UINT32;  //! array of unsigned long	: unsigned 32-bit
	//	break;

	//case GL_R32I:
	//	image_type = FIT_INT32;   //! array of long			    : signed 32-bit
	//	break;

	case GL_R16F:
		save_float_data = false;
		image_type = FIT_FLOAT;   //! array of float			: 32-bit IEEE floating point
		break;

	case GL_R32F:
		save_float_data = true;
		image_type = FIT_FLOAT;   //! array of float			: 32-bit IEEE floating point
		break;

	case GL_DEPTH_COMPONENT32F:
		save_float_data = true;
		image_type = FIT_FLOAT;   //! array of float			: 32-bit IEEE floating point
		break;

	case GL_RG16F:
	case GL_RG32F:
	{
		int channels = 3;
		int float_count = channels * texsize.width * texsize.height;
		bpp = channels * 8 * sizeof(float);
		pitch = bpp / 8 * texsize.width;
		texdatasize = pitch * texsize.height;
		floatdata.resize(float_count);
		const float* input_floats = reinterpret_cast<const float*>(texdata.data());
		for (unsigned int i = 0; i < texsize.width * texsize.height; ++i)
		{
			floatdata[3 * i + 0] = input_floats[2 * i + 0];
			floatdata[3 * i + 1] = input_floats[2 * i + 1];
			floatdata[3 * i + 2] = 0.f;
		}

		imgdata = reinterpret_cast<uint8_t*>(floatdata.data());
		save_float_data = internalformat == GL_RG32F ? true : false;
		image_type = FIT_RGBF;    //! 96-bit RGB float image	: 3 x 32-bit IEEE floating point

		break;
	}

	case GL_RGB16F:
		save_float_data = false;
		image_type = FIT_RGBF;    //! 96-bit RGB float image	: 3 x 32-bit IEEE floating point
		break;

	case GL_RGB32F:
		save_float_data = true;
		image_type = FIT_RGBF;    //! 96-bit RGB float image	: 3 x 32-bit IEEE floating point
		break;

	case GL_RGBA16F:
		save_float_data = false;
		image_type = FIT_RGBAF;   //! 128-bit RGBA float image	: 4 x 32-bit IEEE floating point
		break;

	case GL_RGBA32F:
		save_float_data = true;
		image_type = FIT_RGBAF;   //! 128-bit RGBA float image	: 4 x 32-bit IEEE floating point
		break;

	default:
		return ERROR_INVALID_PARAMETER;

	}

	const unsigned int red_mask = 0, blue_mask = 0, green_mask = 0;

	FIBITMAP* fib = FreeImage_ConvertFromRawBitsEx(false, imgdata, image_type, int(texsize.width), int(texsize.height), int(pitch), unsigned int(bpp), red_mask, green_mask, blue_mask, false);
	
	int save_flags = 0; //Default for all types
	switch (format)
	{
	case FREE_IMAGE_FORMAT::FIF_EXR:
		save_flags = save_float_data ? EXR_FLOAT | EXR_PIZ : EXR_DEFAULT;
		break;
	case FREE_IMAGE_FORMAT::FIF_TIFF:
		save_flags = TIFF_DEFLATE;
	}


	bool result = FreeImage_Save(format, fib, filename.c_str(), save_flags);

	FreeImage_Unload(fib);

	return result ? SUCCESS : ERROR_EXTERNAL_LIB;
}


TextureData& TextureManagerInstance::getTexture(std::string name)
{
#ifdef _DEBUG
	if (!hasTexture(name)) throw(std::runtime_error("Texture manager error: requesting non-existant texture."));
#endif

	return textures.find(name)->second;
}

const TextureData& TextureManagerInstance::getTexture(std::string name) const
{
#ifdef _DEBUG
	if (!hasTexture(name)) throw(std::runtime_error("Texture manager error: requesting non-existant texture."));
#endif

	return textures.find(name)->second;
}

bool TextureManagerInstance::hasTexture(std::string name) const
{
	return textures.find(name) != textures.end();
}
