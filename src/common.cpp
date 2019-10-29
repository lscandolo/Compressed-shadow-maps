#include <common.h>
#include <iostream>
#include <fstream>

void check_opengl()
{
#ifdef _DEBUG
	GLenum error = glGetError();
	if (error == GL_NO_ERROR) return;

	std::cerr << "GL Error: " << glewGetErrorString(error) << std::endl;

	throw(ExceptionOpenGL("OpenGL Error."));
#endif
}

void Assert(bool flag, char* text)  //!!
{
#ifdef _DEBUG
	if (!flag)
	{
		std::cerr << text << std::endl;
		throw(std::runtime_error(text));
}
#endif
}

void Assert(bool flag)  //!!
{
	Assert(flag, "Assert error");
}


void Verify(bool flag)  //!!
{
	if (!flag)
	{
		throw(std::runtime_error("Verify error"));
	}
}

void GLAPIENTRY DebugCallbackGL(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam)
{

	// Ignore non-significant error/warning codes
	if (id == 131169  ||
		id == 0x20071 || // Performance notification about usage hints for buffers
		id == 131218  ||
		id == 131204  ||
		id == 0x20052||// Performance warning about same thread pixel transfer / render
		id == 0x20072) // Performance notification about usage hints for buffers
	{
		return;
	}

	std::string sourcestring;
	switch (source) {
	case GL_DEBUG_SOURCE_API:             sourcestring = "API"; break;
	case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   sourcestring = "Window System"; break;
	case GL_DEBUG_SOURCE_SHADER_COMPILER: sourcestring = "Shader Compiler"; break;
	case GL_DEBUG_SOURCE_THIRD_PARTY:     sourcestring = "Third Party"; break;
	case GL_DEBUG_SOURCE_APPLICATION:     sourcestring = "Application"; break;
	case GL_DEBUG_SOURCE_OTHER:           sourcestring = "Other"; break;
	default:                              sourcestring = "Unknown(" + std::to_string(source) + ")"; break;
	}

	std::string typestring;
	switch (type) {
	case GL_DEBUG_TYPE_ERROR:               typestring = "Error"; break;
	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: typestring = "Deprecated Behaviour"; break;
	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  typestring = "Undefined Behaviour"; break;
	case GL_DEBUG_TYPE_PORTABILITY:         typestring = "Portability"; break;
	case GL_DEBUG_TYPE_PERFORMANCE:         typestring = "Performance"; break;
	case GL_DEBUG_TYPE_MARKER:              typestring = "Marker"; break;
	case GL_DEBUG_TYPE_PUSH_GROUP:          typestring = "Push Group"; break;
	case GL_DEBUG_TYPE_POP_GROUP:           typestring = "Pop Group"; break;
	case GL_DEBUG_TYPE_OTHER:               typestring = "Other"; break;
	default:                                typestring = "Unknown(" + std::to_string(type) + ")"; break;
	}

	std::string severitystring;
	switch (severity) {
	case GL_DEBUG_SEVERITY_NOTIFICATION: severitystring = "Notification"; break;
	case GL_DEBUG_SEVERITY_LOW:          severitystring = "Low"; break;
	case GL_DEBUG_SEVERITY_MEDIUM:       severitystring = "Medium"; break;
	case GL_DEBUG_SEVERITY_HIGH:         severitystring = "High"; break;
	default:                             severitystring = "Unknown(" + std::to_string(severity) + ")"; break;
	}
	static int count = 0;
	count++;
	fprintf(stderr, "(%000d) GL DEBUG CALLBACK : %s type = %s, severity = %s, source = %s, message = %s (0x%x)\n",
		count,
		(type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""),
		typestring.c_str(), severitystring.c_str(), sourcestring.c_str(), message, id);
}

void errorCallbackGLFW(int error, const char* description)
{
	std::cerr << "GLFW Error: " << description << std::endl;
}


String MakeString(std::string s)
{
	return String(s);
}

std::string ToStdString(String s)
{
	return std::string(s);
}

std::vector< byte > LoadFile(const String& filename, bool removeUTFheader)
{
	std::vector< byte > contents;
	std::ifstream f;
	f.open(ToStdString(filename).c_str(), std::ios::in | std::ios::binary | std::ios::ate);
	if (f.is_open())
	{
		size_t size = f.tellg();

		f.seekg(0, std::ios::beg);

		if (removeUTFheader && size >= 3) 
		{
			char header[3];
			f.read(header, 3);
			if (header[0] == char(0xEF) && header[1] == char(0xBB) && header[2] == char(0xBF)) {
				size -= 3;
			} else {
				f.seekg(0, std::ios::beg);
			}
		}

		contents.resize(size);
		f.read((char*)(contents.data()), size);

		f.close();
	}
	return contents;
}

std::vector< byte > LoadTextFile(const String& filename)
{
	std::vector< byte > contents;
	std::ifstream f;
	f.open(ToStdString(filename).c_str(), std::ios::in );
	if (f.is_open())
	{
		std::string fstring((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

		contents.resize(fstring.size());

		std::memcpy(contents.data(), fstring.c_str(), fstring.size());
		f.close();
	}
	return contents;
}

void LoggerInterface::LogError(std::string s)
{
	std::cerr << "Error: " << s << std::endl;
}

void LoggerInterface::LogWarning(std::string s)
{
	std::cout << "Error: " << s << std::endl;
}

LogToFile::LogToFile(std::string _filename)
	: filename(_filename)
{}

void LogToFile::LogError(std::string s)
{
	std::cerr << "Error: " << s << std::endl;
}

void LogToFile::LogWarning(std::string s)
{
	std::cout << "Error: " << s << std::endl;
}