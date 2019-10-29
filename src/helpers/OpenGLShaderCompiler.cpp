#include "common.h"
#include "helpers/OpenGLShaderCompiler.h"
#include "helpers/OpenGLShaderParser.h"

#include <sstream>
#include <iostream>

namespace GLHelpers {

	GLuint ShaderCompiler::CreateFromMemoryTagged(GLenum eShaderType, const std::string& strSource, const Detail::ShaderCommandLine& cmd, const String& strSourceName)
	{
		Detail::ShaderParserOutput parsed = ShaderParser().ParseGeneric( cmd, strSource, strSourceName );

		GLuint uiShader = glCreateShader( eShaderType );
		Verify( uiShader != 0 );

		// signature: (GLuint shader, GLsizei count, const GLchar** strings, const GLint* lengths);
		const GLchar* szSource = parsed.m_strPreprocessedSource.c_str();
		glShaderSource( uiShader, 1, &szSource, NULL);
		glCompileShader( uiShader );

		check_opengl();

		m_PendingCompiles.push_back( uiShader );
		m_ShaderToData[ uiShader ] = std::move( parsed );
		m_ShaderToFile[ uiShader ] = strSourceName;

		return uiShader;
	}

	GLuint ShaderCompiler::Create(GLenum eShaderType, const String& strFilename, const String& strArguments)
	{
		return Create(eShaderType, strFilename, Detail::ShaderCommandLine(strArguments));
	}

	GLuint ShaderCompiler::Create( GLenum eShaderType, const String& strFilename, const Detail::ShaderCommandLine& cmd )
	{
		std::vector< byte > fileData = LoadFile(strFilename);

		const std::string strOrgSourceCode( fileData.begin(), fileData.end() );
		return CreateFromMemoryTagged( eShaderType, strOrgSourceCode, cmd, strFilename );
	}

	GLuint ShaderCompiler::CreateFromMemory( GLenum eShaderType, const std::string& strSource, const String& strArguments )
	{
		return CreateFromMemoryTagged( eShaderType, strSource, Detail::ShaderCommandLine(strArguments), String( "<From Memory>" ) );
	}

	GLuint	ShaderCompiler::CreateFromMemory(GLenum eShaderType, const std::string& strSource, const Detail::ShaderCommandLine& cmd)
	{
		return CreateFromMemoryTagged(eShaderType, strSource, cmd, String("<From Memory>"));
	}

	std::string ShaderCompiler::GetModifiedErrorMessage( GLuint uiShader )
	{
		// should only be called when an error message exists
		if ( true /*Mx::DebugTag::value*/ ) {
			GLint iResult;
			glGetShaderiv( uiShader, GL_COMPILE_STATUS, &iResult );
			Assert( iResult == GL_FALSE );
		}

		int iInfoLogLength;
		glGetShaderiv( uiShader, GL_INFO_LOG_LENGTH, &iInfoLogLength );

		if( iInfoLogLength <= 0 ) {
			Assert( !"OpenGL complains about an error but there's no log?" );
			return std::string();
		}

		std::string buffer;
		buffer.resize( iInfoLogLength );
		glGetShaderInfoLog( uiShader, iInfoLogLength, NULL, &buffer[ 0 ] );

		auto& shaderData = m_ShaderToData[ uiShader ];

		// replace the file/line with the actual data
		//return ModifyErrorMessage( shaderData, buffer ); //!! Disabled because not all drivers return error lines between parenthesis
		return buffer;
	}

	std::string ShaderCompiler::ModifyErrorMessage( Detail::ShaderParserOutput& shaderData, const std::string& strOrgMessage )
	{
		// each GLSL warning/error produces "code(linenumber): warning/error Code: ErrorMessage", so look twice for ":"
		// otherwise there might be a mismatch with the error message
		const char cFindStart = '(';
		const char cToFindEnd = ')';
		const char cTokenInBetween = ':';

		std::istringstream ssIn( strOrgMessage );
		std::string strLine;
		std::stringstream ssOut;
		std::stringstream ssTokenizer;
		int iOrgLineNumber = -1;

		while( std::getline( ssIn, strLine ) ) {
			std::string::size_type sBegin = strLine.find( cFindStart );	++sBegin;
			std::string::size_type sEnd   = strLine.find( cToFindEnd, sBegin );
			std::string::size_type sBegin0 = strLine.find( cTokenInBetween, sEnd ); ++sBegin0;
			std::string::size_type sBegin1 = strLine.find( cTokenInBetween, sBegin0 );

			bool bModify = false;
			if ( sBegin  != std::string::npos &&
				 sEnd    != std::string::npos &&
				 sBegin0 != std::string::npos &&
				 sBegin1 != std::string::npos )
			{
				// we need all of them... :)
				std::string strOrgLineNumber = strLine.substr( sBegin, sEnd - sBegin );
				iOrgLineNumber = std::stoi( strOrgLineNumber );
				bModify = ( iOrgLineNumber < ( int )shaderData.m_LineTrack.size() );
			}
			if ( bModify ) {
				// extract before and after strings
				//const String* pOrigin = shaderData.m_LineTrack[ iOrgLineNumber ];
				auto& origin = shaderData.m_LineTrack[ iOrgLineNumber ];

				ssOut << strLine.substr( 0, sBegin );
				ssOut << ToStdString( *origin.first ) << ":" << origin.second;
				ssOut << strLine.substr( sEnd ) << std::endl;
			}
			else {
				// leave untouched
				ssOut << strLine << std::endl;
			}
		}
		return ssOut.str();
	}

		bool ShaderCompiler::Check()
	{
		bool bReturn = true;
		// @todo: #pragma omp
		// should be possible, see OpenGL Insights
		for( auto& shader : m_PendingCompiles ) {

			GLint iResult = GL_FALSE;
			glGetShaderiv( shader, GL_COMPILE_STATUS, &iResult );

			if( iResult == GL_FALSE ) {
				// log somewhere
				std::string strError = GetModifiedErrorMessage( shader );
				std::cout << "ShaderCompiler::Check:\n"
									<< strError << std::endl;
				if ( m_pLogger ) {

					std::stringstream ss;
					ss << "Failed to compile shader ";
					ss << strError;
					m_pLogger->LogError( ss.str() );
				}
				else {
					std::stringstream ss;
					ss << "Failed to compile shader ";
					ss << strError;
					std::cerr << ( ss.str() ) << "\n";
				}
			}
			bReturn &= ( iResult == GL_TRUE );
		}
		m_PendingCompiles.clear();
		return bReturn;
	}

	ShaderCompiler::ShaderCompiler( LoggerInterface* pLogger ) :
		m_pLogger( pLogger )
	{}

}
