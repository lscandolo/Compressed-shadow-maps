#include "common.h"
#include "helpers/OpenGLShaderObject.h"

#include <sstream>
#include <vector>

namespace GLHelpers
{

	// static method
	void ShaderObjectImpl::ReleaseShaderCompiler()
	{
		glReleaseShaderCompiler();
	}

		ShaderObjectImpl::ShaderObjectImpl( /*EShaderType::E eType*/ ) :
		//ShaderObjectImpl( eType ),
		m_uiShaderHandle( 0 )
	{
		//GLenum eOpenGLType = ShaderTypeToOpenGL( eType );

		//Assert( eOpenGLType != GL_INVALID_ENUM );

		//m_uiShaderHandle = glCreateShader( eOpenGLType );

		check_opengl();
	}

	ShaderObjectImpl::~ShaderObjectImpl()
	{
		if ( m_uiShaderHandle != 0 ) {
			glDeleteShader( m_uiShaderHandle );
		}

		check_opengl();
	}

	void ShaderObjectImpl::UploadTextCode( const char* szCode, size_t sNumCharacters )
	{
		Assert( szCode != nullptr );

		const GLchar* pSource = ( const GLchar* ) szCode;
		if ( sNumCharacters == 0 ) {
			glShaderSource( m_uiShaderHandle, 1, &pSource, 0 );
		}
		else {
			GLint iLength = static_cast< GLint >( sNumCharacters );
			// signature: (GLuint shader, GLsizei count, const GLchar** strings, const GLint* lengths);
			glShaderSource( m_uiShaderHandle, 1, &pSource, &iLength );
		}

		check_opengl();
	}

	void ShaderObjectImpl::UploadTextCode( std::vector< const GLchar* >& listCode )
	{
		// Signature (GLuint shader, GLsizei count, const GLchar** strings, const GLint* lengths);
		const GLchar* pSource = ( const GLchar* )listCode[ 0 ];
		glShaderSource( m_uiShaderHandle, ( GLsizei )listCode.size(), &pSource, 0 );

		check_opengl();
	}

	bool ShaderObjectImpl::Compile() const
	{
		glCompileShader( m_uiShaderHandle );

		GLint iTemp;

		// signature: (GLuint shader, GLenum pname, GLint* param)
		glGetShaderiv( m_uiShaderHandle, GL_COMPILE_STATUS, &iTemp );

		check_opengl();

		bool bCompiled = ( iTemp == GL_TRUE );

		// Check the output parameters of the shader

		return bCompiled;
	}

	std::string ShaderObjectImpl::GetError() const
	{
		std::string strInfoLog = GetInfoLog();

		// should be parsed for warnings and stuff

		return strInfoLog;
	}

	std::string ShaderObjectImpl::GetInfoLog() const
	{
		GLint iLogLength;
		glGetShaderiv( m_uiShaderHandle, GL_INFO_LOG_LENGTH, &iLogLength );

		if ( iLogLength == 0 ) {
			return std::string();
		}

		std::unique_ptr< GLchar[] > infoLog( new GLchar[ iLogLength ] );
		GLchar* pszInfoLog = infoLog.get();
		GLsizei iWritten;

		glGetShaderInfoLog( m_uiShaderHandle, iLogLength, &iWritten, pszInfoLog );

		check_opengl();

		std::stringstream ss;
		ss << pszInfoLog;
		
		return ss.str();
	}

}
