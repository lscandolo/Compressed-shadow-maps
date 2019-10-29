#pragma once

#include "common.h"
#include "helpers/OpenGLShaderParser.h"

#include <map>

namespace GLHelpers {

	class ShaderCompiler
	{
	public:
		ShaderCompiler( LoggerInterface* pLogger = nullptr );

		GLuint											Create( GLenum eShaderType, const String& strFilename, const String& strArguments = String( "" ) );
		GLuint											CreateFromMemory( GLenum eShaderType, const std::string& strSource, const String& strArguments = String(""));

		///// Explicit Detail::ShaderCommandLine functions (because the string->ShaderCommandLine parsing is still a TODO
		GLuint											Create( GLenum eShaderType, const String& strFilename, const Detail::ShaderCommandLine& cmd );
		GLuint											CreateFromMemory( GLenum eShaderType, const std::string& strSource, const Detail::ShaderCommandLine& cmd);

		bool											Check();

	private:
		LoggerInterface*								m_pLogger;
		std::vector< GLuint >							m_PendingCompiles;
		std::map< GLuint, String >						m_ShaderToFile;
		std::map< GLuint, Detail::ShaderParserOutput >	m_ShaderToData;

		std::string										GetModifiedErrorMessage( GLuint uiShader );
		std::string										ModifyErrorMessage( Detail::ShaderParserOutput& shaderData, const std::string& strOrgMessage );

		GLuint											CreateFromMemoryTagged( GLenum eShaderType, const std::string& strSource, const Detail::ShaderCommandLine& cmd, const String& strSourceName );



	};

}
