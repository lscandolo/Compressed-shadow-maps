#pragma once

#define _USE_MATH_DEFINES

#include "error_codes.h"

#include <gl/glew.h>
#include <gl/gl.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <filesystem>

#if _HAS_CXX17
	typedef std::filesystem::path Path;
#else
	typedef std::experimental::filesystem::path Path;
#endif

#include <iostream>



#define OPENGL_BINDLESS 1

#define ARRAY_SIZE(a) (sizeof(a) / sizeof(a[0]))

#define OFFSET_POINTER(a) (reinterpret_cast<void*>(a))

#define LINEARIZE_DEPTH 1

typedef uint8_t byte;

void check_opengl();

void Assert(bool flag, char* text);
void Assert(bool flag);
void Verify(bool flag);

typedef std::string String;
String MakeString(std::string s);
std::string ToStdString(String s);

std::vector< byte > LoadFile(const String& filename, bool removeUTFheader = true);

void GLAPIENTRY DebugCallbackGL(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam);
void errorCallbackGLFW(int error, const char* description);

struct size2D
{
	union {
		unsigned int x;
		unsigned int width;
	};

	union {
		unsigned int y;
		unsigned int height;
	};

	size2D() : width(0), height(0) {}
	explicit size2D(unsigned int v) : width(v), height(v) {}
	size2D(unsigned int w, unsigned int h) : width(w), height(h) {}
	bool operator==( const size2D& rhs) { return x==rhs.x&&y==rhs.y;}
	bool operator!=(const size2D& rhs) { return x!=rhs.x||y!=rhs.y;}
};

struct size3D
{
	union {
		unsigned int x;
		unsigned int width;
	};

	union {
		unsigned int y;
		unsigned int height;
	};

	union {
		unsigned int z;
		unsigned int depth;
	};

	size3D() : width(0), height(0), depth(0) {}
	explicit size3D(unsigned int v) : width(v), height(v), depth(v) {}
	size3D(unsigned int w, unsigned int h, unsigned int d) : width(w), height(h), depth(d) {}
	bool operator==( const size3D& rhs) { return x==rhs.x&&y==rhs.y&&z==rhs.z;}
	bool operator!=(const size3D& rhs) { return x!=rhs.x||y!=rhs.y||z!=rhs.z;}
};

class LoggerInterface
{
public:
	virtual void LogError(std::string s);
	virtual void LogWarning(std::string s);
};

class LogToFile : public LoggerInterface
{
	std::string filename;
public:

	LogToFile(std::string filename);
	virtual void LogError(std::string s) override;
	virtual void LogWarning(std::string s) override;
};

typedef std::runtime_error ExceptionShader;
typedef std::runtime_error ExceptionOpenGL;


