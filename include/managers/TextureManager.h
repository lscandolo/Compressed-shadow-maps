#pragma once

#include "common.h"
#include "helpers/ImageData.h"
#include "helpers/OpenGLTextureObject.h"
#include "helpers/Singleton.h"

#include <glm/glm.hpp>
#include <map>

struct TextureData
{
	std::string name;

	ImageData data;
	GLHelpers::TextureObject gltex;
	TextureData();
};

enum class ImageFileFormat
{
	EXR,
	TIFF
}; 

class TextureManagerInstance
{

public:

	// RGBtoRGBA means load RGB images as RGBA with 1 in the alpha channel
	int                 loadImageData(std::string filename, bool RGBtoRGBA = true);
	int                 loadTexture(std::string filename, bool generate_mipmap = true, bool RGBtoRGBA = true);
	TextureData&        getTexture(std::string name);
	const TextureData&  getTexture(std::string name) const;
	bool                hasTexture(std::string name) const;
	bool                deleteTexture(std::string name);
	bool                clear();
	int                 saveTexture(ImageFileFormat format, GLHelpers::TextureObject2D tex, std::string filename);
	int                 saveTexture(ImageFileFormat format, GLHelpers::TextureObject3D tex, int layer, std::string filename);

protected:
	TextureManagerInstance();
	~TextureManagerInstance();

private:

	int _saveTexture32Bits(enum FREE_IMAGE_FORMAT format, const std::string& filename, std::vector<uint8_t>& texdata, size2D texsize, GLenum internalformat, GLuint channels, GLuint bpp);
	std::map<std::string, TextureData> textures;
	std::list<GLHelpers::TextureObject1D> tex1d;
	std::list<GLHelpers::TextureObject2D> tex2d;
	std::list<GLHelpers::TextureObject3D> tex3d;

	friend Singleton<TextureManagerInstance>;
};

typedef Singleton<TextureManagerInstance> TextureManager;
