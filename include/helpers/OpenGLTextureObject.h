#pragma once

#include "common.h"
#include "helpers/ImageData.h"
#include "helpers/OpenGLBufferObject.h"

#include <glm/glm.hpp>

namespace GLHelpers {

	class TextureObjectImpl
	{
	public:

		TextureObjectImpl();
		virtual ~TextureObjectImpl();

		// keep this in mind for state changes
		// http://www.opengl.org/wiki/Common_Mistakes#The_Object_Oriented_Language_Problem

		GLuint					id;                    // texture id
		GLenum                  internalFormat;        // internal OpenGL format
		GLenum                  dataType;              // data type
		GLenum                  dataFormat;            // data format
		GLuint                  width, height, depth;  // texture resolution
		GLuint                  channels;              // number of texture channels
		GLuint                  bpp;                   // pixel resolution = bits per pixel
		GLenum                  target;                // default target
		GLuint                  levels;                // mipmap levels

		void                    create(glm::ivec2 size, GLenum internalFormat, void* data, bool createMipmapStorage = false, bool mutableStorage = false);
		void                    create(glm::ivec3 size, GLenum internalFormat, void* data, bool createMipmapStorage = false, bool mutableStorage = false);

		virtual void			Release();
		virtual void            Bind(GLint textureUnit = -1) const;
		virtual void            Unbind(GLint textureUnit = -1) const;

		void                    setupTextureFormat(GLenum textureFormat);
		virtual void            initializeTextureMemory(const void* data, bool createMipmapStorage, bool mutableStorage) = 0;

		virtual void  setInterpolationMethod(GLenum method);
		virtual void  setInterpolationMethod(GLenum minMethod, GLenum magMethod);
		virtual void  setWrapMethod(GLenum method);
		virtual void  setWrapMethod(GLenum methodS, GLenum methodT, GLenum methodR);
		virtual void  setParameteri(GLenum paramName, GLint param);
		virtual void  setParameterf(GLenum paramName, GLfloat param);
		virtual GLint getParameter(GLenum paramName);

		virtual GLuint64 getHandle();
		virtual bool     isResident();
		virtual void     makeResident();
		virtual void     makeNotResident();

		void generateMipmap();
		void generateMipmap(GLint maxLevel, GLenum minMethod = GL_LINEAR_MIPMAP_LINEAR, GLenum magMethod = GL_LINEAR);

		glm::ivec2 size();
		glm::ivec3 size3();
	};

	class TextureObject1DImpl : public TextureObjectImpl
	{
	public:
		TextureObject1DImpl();
		~TextureObject1DImpl();

		virtual void            initializeTextureMemory(const void* data, bool createMipmapStorage = false, bool mutableStorage = false);
		virtual void            initializeTextureMemoryCompressed(const void* data, GLsizei dataSize, bool createMipmapStorage = false);
		virtual void            uploadTextureData(const void* data, GLint lod = 0);
//		virtual void            downloadTextureData(void* outputData); // TODO
		virtual void            loadFromImageData(const ImageData& img, bool bMutable, bool bCreateMipMapStorage);
	};

	class TextureObject2DImpl : public TextureObjectImpl
	{
	public:
		TextureObject2DImpl();
		~TextureObject2DImpl();

		virtual void            initializeTextureMemory(const void* data, bool createMipmapStorage = false, bool mutableStorage = false);
		virtual void            initializeTextureMemoryCompressed(const void* data, GLsizei dataSize, bool createMipmapStorage = false);
		virtual void            uploadTextureData(const void* data, GLint lod = 0);
		virtual void            downloadTextureData(void* outputData, GLint  lod = 0, GLsizei bufsize = -1) const; 
//		virtual void            getTextureDataSize(void* outputData, GLint  lod = 0); // TODO
		virtual void            loadFromImageData(const ImageData& img, bool bMutable, bool bCreateMipMapStorage);
		virtual void            setSubImage(glm::ivec2 offset, glm::ivec2 size, const void* data, GLuint mipmaplevel = 0);
		virtual void            copy(std::shared_ptr<TextureObject2DImpl> dst, glm::ivec2 copysize = glm::ivec2(-1, -1), glm::ivec2 srcoffset = glm::ivec2(0, 0), glm::ivec2 dstoffset = glm::ivec2(0, 0), GLuint src_mipmap = 0, GLuint dst_mipmap = 0);
	};

	class TextureObject2DArrayImpl : public TextureObjectImpl
	{
	public:
		TextureObject2DArrayImpl();
		~TextureObject2DArrayImpl();

		virtual void            initializeTextureMemory(const void* data, bool createMipmapStorage = false, bool mutableStorage = false);
		virtual void            initializeTextureMemoryCompressed(const void* data, GLsizei dataSize, bool createMipmapStorage = false);
		virtual void            uploadTextureData(const void* data, GLint lod = 0);
		virtual void            downloadTextureData(void* outputData, GLint  lod = 0, GLsizei bufsize = -1) const; 
//		virtual void            getTextureDataSize(void* outputData, GLint  lod = 0); // TODO
		virtual void            loadFromImageData(const ImageData& img, bool bMutable, bool bCreateMipMapStorage);
		virtual void            setSubImage(glm::ivec3 offset, glm::ivec3 size, const void* data, GLuint mipmaplevel = 0);
	};

	class TextureObject2DMultisampleImpl : public TextureObjectImpl
	{
	public:
		TextureObject2DMultisampleImpl();
		~TextureObject2DMultisampleImpl();

		GLuint                  samples;
		GLboolean               fixedSampleLocations;

		virtual void            create(glm::ivec3 size, GLenum internalFormat, void* data);
		virtual void            create(glm::ivec2 size, int samples, GLenum internalFormat, GLboolean fixedSampleLocations = true);
		virtual void            initializeTextureMemory(const void* data, bool createMipmapStorage = false, bool mutableStorage = false);
		virtual void            uploadTextureData(const void* data, GLint lod = 0);
	};

	class TextureObject3DImpl : public TextureObjectImpl
	{
	public:
		TextureObject3DImpl();
		~TextureObject3DImpl();

		virtual void            initializeTextureMemory(const void* data, bool createMipmapStorage = false, bool mutableStorage = false);
		virtual void            initializeTextureMemoryCompressed(const void* data, GLsizei dataSize, bool createMipmapStorage = false);
		virtual void            uploadTextureData(const void* data, GLint lod = 0);
		virtual void            downloadTextureData(void* outputData, GLint  lod = 0, GLsizei bufsize = -1); // TODO
		virtual void            downloadTextureLayerData(void* outputData, GLuint layer, GLint  lod = 0, GLsizei bufsize = -1); // TODO
		virtual void            loadFromImageData(const ImageData& img, bool bMutable, bool bCreateMipMapStorage);
	};

	class TextureObjectBufferImpl : public TextureObjectImpl
	{
	public:
		TextureObjectBufferImpl();
		~TextureObjectBufferImpl();

		virtual void            initializeTextureMemory(const void* data, bool createMipmapStorage = false, bool mutableStorage = false);
		virtual void            uploadTextureData(const void* data, GLint lod = 0);
		virtual void            create();
		virtual void            attachBufferData(BufferObject buffer, GLenum internalFormat);
		virtual void            detachBufferData();
	};

	class TextureObjectCubeImpl : public TextureObjectImpl
	{
	public:
		TextureObjectCubeImpl();
		~TextureObjectCubeImpl();

		virtual void            initializeTextureMemory(const void* data, bool createMipmapStorage = false, bool mutableStorage = false);
		virtual void            uploadTextureData(GLenum face, const void* data, GLint lod = 0);
		virtual void            uploadTextureData(const void* data[6], GLint lod = 0);
		virtual void            loadFromHorizontalCrossImageData(const ImageData& img, bool bMutable, bool bCreateMipMapStorage);
	};

	class TextureObjectCubeArrayImpl : public TextureObjectImpl
	{
	public:
		TextureObjectCubeArrayImpl();
		~TextureObjectCubeArrayImpl();

		virtual void            initializeTextureMemory(const void* data, bool createMipmapStorage = false, bool mutableStorage = false);
	};

	using TextureObject              = std::shared_ptr<TextureObjectImpl>;
	using TextureObject1D            = std::shared_ptr<TextureObject1DImpl>;
	using TextureObject2D            = std::shared_ptr<TextureObject2DImpl>;
	using TextureObject2DArray       = std::shared_ptr<TextureObject2DArrayImpl>;
	using TextureObject2DMultisample = std::shared_ptr<TextureObject2DMultisampleImpl>;
	using TextureObject3D            = std::shared_ptr<TextureObject3DImpl>;
	using TextureObjectBuffer        = std::shared_ptr<TextureObjectBufferImpl>;
	using TextureObjectCube          = std::shared_ptr<TextureObjectCubeImpl>;
	using TextureObjectCubeArray     = std::shared_ptr<TextureObjectCubeArrayImpl>;


}

