#include "common.h"
#include "helpers/OpenGLProgramObject.h"
#include "helpers/OpenGLShaderCompiler.h"

#include <iostream>
#include <glm/gtc/type_ptr.hpp>

namespace GLHelpers {

	bool CheckProgram( GLuint uiProgramName, LoggerInterface* pLogger )
	{
		if( !uiProgramName ) return false;

		GLint iResult = GL_FALSE;
		glGetProgramiv( uiProgramName, GL_LINK_STATUS, &iResult );

		if ( !iResult ) {
			int iLogLength;
			glGetProgramiv( uiProgramName, GL_INFO_LOG_LENGTH, &iLogLength );

			if( iLogLength > 0 ) {
				std::string buffer;				
				buffer.resize( iLogLength );
				glGetProgramInfoLog( uiProgramName, iLogLength, nullptr, &buffer[ 0 ] );

				if ( pLogger ) {
					if ( iResult == GL_TRUE ) {
						pLogger->LogWarning( buffer );
					}
					else {
						pLogger->LogError( buffer );
					}
				}
				else {
					std::cout << buffer << "\n";
				}
			}
		}

		return iResult == GL_TRUE;
	}

	bool ValidateProgram( GLuint uiProgramName, LoggerInterface* pLogger )
	{
		if( !uiProgramName) return false;

		Assert( !"Untested" );

		glValidateProgram( uiProgramName );
		GLint iResult = GL_FALSE;
		glGetProgramiv( uiProgramName, GL_VALIDATE_STATUS, &iResult );

		if( pLogger && ( iResult == GL_FALSE ) ) {
			int InfoLogLength;
			glGetProgramiv( uiProgramName, GL_INFO_LOG_LENGTH, &InfoLogLength );
			if( InfoLogLength > 0) {
				std::vector< char > buf( InfoLogLength );
				glGetProgramInfoLog( uiProgramName, InfoLogLength, nullptr, &buf[ 0 ] );

				if ( iResult == GL_TRUE ) {
					pLogger->LogWarning( &buf[ 0 ] );
				}
				else {
					pLogger->LogError( &buf[ 0 ] );
				}
			}
		}

		return iResult == GL_TRUE;
	}

	static void GetOpenGLError(bool ignore = false)
	{
	#ifndef NDEBUG

		bool got_error = false;
		GLenum error = 0;
		error = glGetError();
		std::string errorstring = "";

		while (error != GL_NO_ERROR)
		{
			if (error == GL_INVALID_ENUM)
			{
				//An unacceptable value is specified for an enumerated argument. The offending command is ignored and has no other side effect than to set the error flag.
				errorstring += "invalid enum...\n";
				got_error = true;
			}

			if (error == GL_INVALID_VALUE)
			{
				//A numeric argument is out of range. The offending command is ignored and has no other side effect than to set the error flag.
				errorstring += "invalid value...\n";
				got_error = true;
			}

			if (error == GL_INVALID_OPERATION)
			{
				//The specified operation is not allowed in the current state. The offending command is ignored and has no other side effect than to set the error flag.
				errorstring += "invalid operation...\n";
				got_error = true;
			}

			if (error == GL_STACK_OVERFLOW)
			{
				//This command would cause a stack overflow. The offending command is ignored and has no other side effect than to set the error flag.
				errorstring += "stack overflow...\n";
				got_error = true;
			}

			if (error == GL_STACK_UNDERFLOW)
			{
				//This command would cause a stack underflow. The offending command is ignored and has no other side effect than to set the error flag.
				errorstring += "stack underflow...\n";
				got_error = true;
			}

			if (error == GL_OUT_OF_MEMORY)
			{
				//There is not enough memory left to execute the command. The state of the GL is undefined, except for the state of the error flags, after this error is recorded.
				errorstring += "out of memory...\n";
				got_error = true;
			}

			if (error == GL_TABLE_TOO_LARGE)
			{
				//The specified table exceeds the implementation's maximum supported table size.  The offending command is ignored and has no other side effect than to set the error flag.
				errorstring += "table too large...\n";
				got_error = true;
			}

			error = glGetError();
		}

		if (got_error && !ignore)
		{
			Assert(false);
			//LOGERROR("OpenGL error : %s\n", errorstring.c_str());
		}

	#endif
	}



	// static method
	size_t ProgramObjectImpl::GetNumberOfBinaryFormats()
	{
		GLint iTemp;
		glGetIntegerv( GL_NUM_PROGRAM_BINARY_FORMATS, &iTemp );

		Assert( !"Check what's going on here..." );
		glGetIntegerv( GL_PROGRAM_BINARY_FORMATS, &iTemp );

		return iTemp;
	}

	// static method
	std::vector< GLint > ProgramObjectImpl::GetBinaryFormats()
	{
		std::vector< GLint > retVector;
		size_t sNumberOfBinaryFormats = GetNumberOfBinaryFormats();

		//Mx::UniquePtrT< GLint, Memory::ArrayDelete > binaryFormats( new GLint[ sNumberOfBinaryFormats ] );
		std::unique_ptr< GLint[] > binaryFormats( new GLint[ sNumberOfBinaryFormats ] );

		GLint* pBinary = binaryFormats.get();
		glGetIntegerv( GL_PROGRAM_BINARY_FORMATS, pBinary );
		retVector.assign( pBinary, pBinary + sNumberOfBinaryFormats );

		return retVector;
	}

	ProgramObjectImpl::ProgramObjectImpl()
	{
		programId = 0;
	}

	ProgramObjectImpl::ProgramObjectImpl( GLenum eBinaryFormat, const char* pData, size_t sSize )
	{
		programId = glCreateProgram();
		if ( programId == 0 ) {
			throw( ExceptionShader( "glCreateProgram failed" ) );
		}
		check_opengl();
		UploadBinary( eBinaryFormat, pData, sSize );
	}

	ProgramObjectImpl::~ProgramObjectImpl()
	{
		Delete();
	}

	bool ProgramObjectImpl::Link()
	{
		glLinkProgram( programId );

		// check
		GLint iTemp;
		glGetProgramiv( programId, GL_LINK_STATUS, &iTemp );
		bool bReturn = ( iTemp != GL_FALSE );

		// Validate as well:
		bReturn &= IsValid();

		check_opengl();
		return bReturn;
	}

	bool ProgramObjectImpl::IsValid() const
	{
		if (!programId) return false;

		GLint iTemp;
		glValidateProgram( programId );
		glGetProgramiv( programId, GL_VALIDATE_STATUS, &iTemp );

		check_opengl();
		return ( iTemp != GL_FALSE );
	}

	void ProgramObjectImpl::Delete()
	{
		if (programId != 0) {
			glDeleteProgram(programId);
			programId = 0;
		}
	}

	std::string ProgramObjectImpl::GetLinkError() const
	{
		GLint iInfoLogLength = 0;
		glGetObjectParameterivARB( programId, GL_OBJECT_INFO_LOG_LENGTH_ARB, &iInfoLogLength );

		if( iInfoLogLength > 1 ) {
			int iCharsWritten  = 0;
			//Mx::UniquePtrT< GLchar, Memory::ArrayDelete > infoLog( new GLchar[ iInfoLogLength ] );
			std::unique_ptr< GLchar[] > infoLog( new GLchar[ iInfoLogLength ] );
			glGetInfoLogARB( programId, iInfoLogLength, &iCharsWritten, infoLog.get() );
			return std::string( infoLog.get() );
		}
		return std::string( "" );
	}

	std::string ProgramObjectImpl::GetError() const
	{
		std::string strInfoLog = GetInfoLog();

		// @todo: I guess there must be some parsing going on.
		// The info log might have some warnings only
		Assert( 0 );

		return strInfoLog;
	}

	std::string ProgramObjectImpl::GetInfoLog() const
	{
		GLint iLogLength;
		glGetProgramiv( programId, GL_INFO_LOG_LENGTH, &iLogLength );
		
		if ( iLogLength == 0 ) {
			return std::string();
		}

		//Mx::UniquePtrT< GLchar, Memory::ArrayDelete > infoLog( new GLchar[ iLogLength ] );
		std::unique_ptr< GLchar[] > infoLog( new GLchar[ iLogLength ] );
		GLchar* pszInfoLog = infoLog.get();
		GLsizei iWritten;

		glGetProgramInfoLog( programId, iLogLength, &iWritten, pszInfoLog );

		check_opengl();

		std::string strReturn( pszInfoLog );

		return strReturn;
	}

	GLint ProgramObjectImpl::GetUniformLocation( const char* szUniformName ) const
	{
		return glGetUniformLocation( programId, szUniformName );
	}

	GLint ProgramObjectImpl::GetUniformLocation( const std::string strUniformName ) const
	{
		return glGetUniformLocation( programId, strUniformName.c_str() );
	}

	GLint ProgramObjectImpl::GetUniformBufferSize( const char* szBufferName ) const
	{
		return glGetUniformBufferSizeEXT ? glGetUniformBufferSizeEXT( programId, GetUniformLocation( szBufferName ) ) : 0;
	}

	void ProgramObjectImpl::Use(bool enable) const
	{
		glUseProgram( enable ? programId : 0);


		check_opengl();
	}

	void ProgramObjectImpl::AttachShader( ShaderObject shaderObject ) const
	{
		glAttachShader( programId, shaderObject->GetOpenGLID() );

		check_opengl();
	}

	size_t ProgramObjectImpl::GetNumberOfAttachedShaders() const
	{
		GLint iAttached;
		glGetProgramiv( programId, GL_ATTACHED_SHADERS, &iAttached );

		check_opengl();
		return iAttached;
	}

	void ProgramObjectImpl::SetAttributeLocation(const char* attributeName, GLuint index) const
	{
		glBindAttribLocation( programId, index, attributeName);
	}

	GLint ProgramObjectImpl::GetAttribLocation( const char* szAttribLocationName ) const
	{	
		return glGetAttribLocation( programId, szAttribLocationName );
	}

	size_t ProgramObjectImpl::GetNumberOfActiveAttributes() const
	{
		GLint iAttached;
		glGetProgramiv( programId, GL_ACTIVE_ATTRIBUTES, &iAttached );

		check_opengl();
		return iAttached;
	}

	size_t ProgramObjectImpl::GetLengthLongestAttributeName() const
	{
		GLint iAttached;
		glGetProgramiv( programId, GL_ACTIVE_ATTRIBUTE_MAX_LENGTH, &iAttached );

		check_opengl();
		return iAttached;
	}

	void ProgramObjectImpl::GetAttribute( GLuint uiAttribIndex, std::string& strAttributeName, GLint& iSize, GLenum& eType ) const
	{
		Assert( uiAttribIndex < GetNumberOfActiveAttributes() );

		size_t sLongest = GetLengthLongestAttributeName();

		//Mx::UniquePtrT< GLchar, Memory::ArrayDelete > szBuffer( new GLchar[ sLongest ] );
		std::unique_ptr< GLchar[] > szBuffer( new GLchar[ sLongest ] );

		// signature: (GLuint program, GLuint index, GLsizei maxLength, GLsizei* length, GLint* size, GLenum* type, GLchar* name);
		glGetActiveAttrib( programId, uiAttribIndex, ( GLsizei )sLongest, 0, &iSize, &eType, &( szBuffer.get()[ 0 ] ) );
		strAttributeName = ( const char* )szBuffer.get();

		check_opengl();
	}

	void ProgramObjectImpl::GetAttribute( GLuint uiAttribIndex, std::string& strAttributeName, GLint& iSize, GLenum& eType, GLint& iLocation ) const
	{
		GetAttribute( uiAttribIndex, strAttributeName, iSize, eType );
		iLocation = GetAttribLocation( strAttributeName.c_str() );

		check_opengl();
	}

	void ProgramObjectImpl::GetAttribute( GLuint uiAttribIndex, std::string& strAttributeName, GLenum& eType ) const
	{
		GLint iSize;
		GLenum eOGLType;
		GetAttribute( uiAttribIndex, strAttributeName, iSize, eOGLType );

		Assert( 0 );
		//eType = OpenGL::OpenGLFormatToMx( eOGLType );
	}

	void ProgramObjectImpl::GetAttribute( GLuint uiAttribIndex, std::string& strAttributeName, GLenum& eType, GLint& iLocation ) const
	{
		GetAttribute( uiAttribIndex, strAttributeName, eType );
		iLocation = GetAttribLocation( strAttributeName.c_str() );

		check_opengl();
	}

	size_t ProgramObjectImpl::GetNumberOfActiveUniforms() const
	{
		GLint iAttached;
		glGetProgramiv( programId, GL_ACTIVE_UNIFORMS, &iAttached );

		check_opengl();
		return iAttached;
	}

	GLsizei ProgramObjectImpl::GetLengthLongestUniformName() const
	{
		GLsizei iAttached;
		glGetProgramiv( programId, GL_ACTIVE_UNIFORM_MAX_LENGTH, &iAttached );

		check_opengl();
		return iAttached;
	}

	void ProgramObjectImpl::GetUniform( GLuint uiUniformIndex, std::string& strUniformName, GLint& iSize, GLenum& eType ) const
	{
		Assert( uiUniformIndex < GetNumberOfActiveUniforms() );

		size_t sLongest = GetLengthLongestUniformName();
		//Mx::UniquePtrT< GLchar, Memory::ArrayDelete > buffer( new GLchar[ sLongest ] );
		std::unique_ptr< GLchar[] > buffer( new GLchar[ sLongest ] );

		// signature: (GLuint program, GLuint index, GLsizei maxLength, GLsizei* length, GLint* size, GLenum* type, GLchar* name);
		glGetActiveUniform( programId, uiUniformIndex, ( GLsizei )sLongest, 0, &iSize, &eType, buffer.get() );

		strUniformName = std::string( buffer.get() );

		check_opengl();
	}

	void ProgramObjectImpl::GetUniform( GLuint uiUniformIndex, std::string& strUniformName, GLint& iSize, GLenum& eType, GLint& iLocation ) const
	{
		GetUniform( uiUniformIndex, strUniformName, iSize, eType );
		iLocation = GetUniformLocation( strUniformName.c_str() );
		check_opengl();
	}

	GLuint ProgramObjectImpl::GetUniformBlockIndex( const char* szUniformBlockName ) const
	{
		return glGetUniformBlockIndex( programId, szUniformBlockName );
	}

	void ProgramObjectImpl::SetUniformBlockBinding(GLuint uniformBlockIndex, GLuint uniformBlockBinding) const
	{
		glUniformBlockBinding(programId, uniformBlockIndex, uniformBlockBinding);
	}

	void ProgramObjectImpl::SetUniformBlockBinding(const char* szUniformBlockName, GLuint uniformBlockBinding) const
	{
		GLuint blockIndex = GetUniformBlockIndex(szUniformBlockName);
		if (blockIndex != GL_INVALID_INDEX) {
			glUniformBlockBinding(programId, blockIndex, uniformBlockBinding);
		} else { 
			Assert(!"Incorrect uniform block name");
		}
	}

	std::string ProgramObjectImpl::GetActiveUniformName( GLuint uiUniformIndex ) const
	{
		GLsizei sLongest = GetLengthLongestUniformName();
		//Mx::UniquePtrT< char, Memory::ArrayDelete > buffer( new char[ sLongest ] );
		std::unique_ptr< GLchar[] > buffer( new GLchar[ sLongest ] );
		
		GLsizei iWritten = 0;

		// signature: (GLuint program, GLuint uniformIndex, GLsizei bufSize, GLsizei* length, GLchar* uniformName);
		glGetActiveUniformName( programId, uiUniformIndex, sLongest, &iWritten, buffer.get() );

		Assert( !"Check buffer for zero-termination!" );

		check_opengl();
		return std::string( buffer.get() );
	}

	size_t ProgramObjectImpl::GetNumberOfActiveUniformBlocks() const
	{
		GLint iAttached;
		glGetProgramiv( programId, GL_ACTIVE_UNIFORM_BLOCKS, &iAttached );

		check_opengl();
		return iAttached;
	}

	size_t ProgramObjectImpl::GetLengthLongestUniformBlock() const
	{
		GLint iAttached;
		glGetProgramiv( programId, GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH, &iAttached );

		check_opengl();
		return iAttached;
	}

	size_t ProgramObjectImpl::GetUniformBlockDataSize( GLuint uiUniformBlockIndex ) const
	{
		Assert( !"Untested!" );

		Assert( uiUniformBlockIndex < GetNumberOfActiveUniformBlocks() && "Not a valid block index" );
		GLint iBlockSize;

		// signature: (GLuint program, GLuint uniformBlockIndex, GLenum pname, GLint* params);
		glGetActiveUniformBlockiv( programId, uiUniformBlockIndex, GL_UNIFORM_BLOCK_DATA_SIZE, &iBlockSize );
		
		check_opengl();

		return iBlockSize;
	}

	size_t ProgramObjectImpl::GetUniformBlockActiveUniforms( GLuint uiUniformBlockIndex ) const
	{
		Assert( !"Untested!" );

		Assert( uiUniformBlockIndex < GetNumberOfActiveUniformBlocks() && "Not a valid block index" );
		GLint iActiveUniformsInBlock;

		// signature: (GLuint program, GLuint uniformBlockIndex, GLenum pname, GLint* params);
		glGetActiveUniformBlockiv( programId, uiUniformBlockIndex, GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS, &iActiveUniformsInBlock );

		check_opengl();

		return iActiveUniformsInBlock;
	}

	size_t ProgramObjectImpl::GetUniformBlockNameLength( GLuint uiUniformBlockIndex ) const
	{

		Assert( uiUniformBlockIndex < GetNumberOfActiveUniformBlocks() && "Not a valid block index" );
		GLint iNameLength;

		// signature: (GLuint program, GLuint uniformBlockIndex, GLenum pname, GLint* params);
		glGetActiveUniformBlockiv( programId, uiUniformBlockIndex, GL_UNIFORM_BLOCK_NAME_LENGTH, &iNameLength );

		check_opengl();

		return iNameLength;
	}

	void   
	ProgramObjectImpl::SetUniformBlock(const char* szUniformBlockName, const BufferObject& buffer, GLuint uniformBlockBinding, GLintptr offset, GLsizeiptr size) const
	{
		SetUniformBlockBinding(szUniformBlockName, uniformBlockBinding);

		if (size == 0) size = buffer->bytesize;
		glBindBufferRange(GL_UNIFORM_BUFFER, uniformBlockBinding, buffer->id , 0, size);

		check_opengl();
	}

	GLuint 
	ProgramObjectImpl::GetSSBOResourceIndex(const char* szUniformBlockName) const
	{
		GLuint resourceIndex = glGetProgramResourceIndex(programId, GL_SHADER_STORAGE_BLOCK, szUniformBlockName);

		check_opengl();

		return resourceIndex;
	}

	void 
	ProgramObjectImpl::SetSSBO(const char* szUniformBlockName, const BufferObject& buffer, GLuint bindingPointIndex, GLintptr offset, GLsizeiptr size) const
	{
		GLuint resourceIndex = GetSSBOResourceIndex(szUniformBlockName);

		if (resourceIndex == GL_INVALID_ENUM) return;

		glShaderStorageBlockBinding(programId, resourceIndex, bindingPointIndex);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bindingPointIndex, buffer->id);

		check_opengl();

	}

	std::string ProgramObjectImpl::GetUniformBlockName( GLuint uiUniformBlockIndex ) const
	{
		Assert( !"Todo" );
		return std::string( "blub" );
	}

	void ProgramObjectImpl::SetGeometryShaderNumVerticesOut( int32_t iNumVerticesOut ) const
	{
	#if defined( _DEBUG ) || defined( DEBUG )
		
		GLint iMaxHardwareValue = 0;
		glGetIntegerv( GL_MAX_GEOMETRY_OUTPUT_VERTICES_EXT, &iMaxHardwareValue );

		Assert( ( GLint )iNumVerticesOut <= iMaxHardwareValue );

	#endif // DEBUG

		glProgramParameteri( programId, GL_GEOMETRY_VERTICES_OUT_EXT, ( GLint )iNumVerticesOut );

		check_opengl();
	}

	void ProgramObjectImpl::SetGeometryShaderInputType( GLenum eType ) const
	{
		Assert( ( ( eType == GL_POINTS ) ||
					( eType == GL_LINES ) ||
					( eType == GL_LINES_ADJACENCY ) ||
					( eType == GL_TRIANGLES ) ||
					( eType == GL_TRIANGLES_ADJACENCY ) ) && ( "Invalid Enum Type!" ) );
		glProgramParameteri( programId, GL_GEOMETRY_INPUT_TYPE, eType );

		check_opengl();
	}

	void ProgramObjectImpl::SetGeometryShaderOutputType( GLenum eType ) const
	{
		Assert( ( ( eType == GL_POINTS ) ||
					( eType == GL_LINE_STRIP ) ||
					( eType == GL_TRIANGLE_STRIP ) ) && ( "Invalid Enum Type!" ) );
		glProgramParameteri( programId, GL_GEOMETRY_OUTPUT_TYPE, eType );

		check_opengl();
	}

	GLint ProgramObjectImpl::GetTessellationControlOutputVertices() const
	{
		Assert( !"Untested" );
		GLint iReturn = 0;
		// signature: (GLuint program, GLenum pname, GLint* param);
		glGetProgramiv( programId, GL_TESS_CONTROL_OUTPUT_VERTICES, &iReturn );

		check_opengl();
		return iReturn;
	}

	GLint ProgramObjectImpl::GetTessellationGenerationMode() const
	{
		Assert( !"Untested" );
		GLint iReturn = 0;
		// signature: (GLuint program, GLenum pname, GLint* param);
		glGetProgramiv( programId, GL_TESS_GEN_MODE, &iReturn );

		check_opengl();
		return iReturn;
	}

	GLint ProgramObjectImpl::GetTessellationGenerationSpacing() const
	{
		Assert( !"Untested" );
		GLint iReturn = 0;
		// signature: (GLuint program, GLenum pname, GLint* param);
		glGetProgramiv( programId, GL_TESS_GEN_SPACING, &iReturn );

		check_opengl();
		return iReturn;
	}

	GLint ProgramObjectImpl::GetTessellationGenerationVertexOrder() const
	{
		Assert( !"Untested" );
		GLint iReturn = 0;
		// signature: (GLuint program, GLenum pname, GLint* param);
		glGetProgramiv( programId, GL_TESS_GEN_VERTEX_ORDER, &iReturn );
		return iReturn;
	}

	GLint ProgramObjectImpl::GetTessellationGenerationPointMode() const
	{
		Assert( !"Untested" );
		GLint iReturn = 0;
		// signature: (GLuint program, GLenum pname, GLint* param);
		glGetProgramiv( programId, GL_TESS_GEN_POINT_MODE, &iReturn );

		check_opengl();
		return iReturn;
	}

	void ProgramObjectImpl::SetBinaryHint( bool bHint ) const
	{
		glProgramParameteri( programId, GL_PROGRAM_BINARY_RETRIEVABLE_HINT, ( bHint ? GL_TRUE : GL_FALSE ) );

		check_opengl();
	}

	std::unique_ptr< char[] > ProgramObjectImpl::GetBinary( GLenum& refBinaryFormat, size_t& refSize ) const
	{
		GLint iProgramLength;
		glGetProgramiv( programId, GL_PROGRAM_BINARY_LENGTH, &iProgramLength );

	#if defined( _DEBUG ) || defined( DEBUG )
		// try again (maybe the hint was not set)
		if ( iProgramLength == 0 ) {
			SetBinaryHint( true );
			ProgramObjectImpl* pProgramNonConst = const_cast< ProgramObjectImpl* >( this );
			Verify( pProgramNonConst->Link() );
			glGetProgramiv( programId, GL_PROGRAM_BINARY_LENGTH, &iProgramLength );
			if ( iProgramLength != 0 ) {
				throw( ExceptionShader( "You forgot to call SetBinaryHint for this shader. This will crash in release mode!" ) );
			}
		}
	#endif // DEBUG

		if ( iProgramLength == 0 ) {
			// @todo throw exception
			Assert( 0 );
			refSize = 0;
			refBinaryFormat = GL_INVALID_ENUM;
			
			throw( ExceptionOpenGL( "Could not retrieve binary from shader!" ) );
			return nullptr;
		}


		std::unique_ptr< char[] > data( new char[ iProgramLength ] );
		GLsizei iWritten = 0;

		// signature: GLuint program, GLsizei bufSize, GLsizei* length, GLenum* binaryFormat, GLvoid* binary
		glGetProgramBinary( programId, ( GLsizei ) iProgramLength, &iWritten, &refBinaryFormat, static_cast< GLvoid* >( data.get() ) );

		refSize = iWritten;

		check_opengl();

		return data;
	}

	void ProgramObjectImpl::UploadBinary( GLenum eBinaryFormat, const char* szData, size_t sSize )
	{
		glProgramBinary( programId, eBinaryFormat, szData, ( GLsizei ) sSize );

		check_opengl();
	}


#pragma push_macro("check_is_member")
#pragma push_macro("check_type")
//#ifdef _DEBUG
#if 0
#define check_is_member(uName) { if (!hasUniform(uName)) { LOGERROR("shader `%s` has no uniform `%s`\n", name.c_str(), uName.c_str()); return; } } (void)(0)
#define check_type(type) { if (getType(uniformName) != type) { LOGERROR("type of uniform `%s` in shader `%s` is not " #type "\n", uniformName.c_str(), name.c_str()); return; } } (void)(0)
#else
#define check_is_member(uName) (void)(0)
#define check_type(type) (void)(0)
#endif

	void ProgramObjectImpl::SetUniform(const std::string& uniformName, GLboolean value)
	{
		SetUniform(uniformName, GLint(value));
	}

	void ProgramObjectImpl::SetUniform(const std::string& uniformName, GLint value) {
		check_is_member(uniformName);
		GetOpenGLError();
		glProgramUniform1i(programId, GetUniformLocation(uniformName), value);
		GetOpenGLError();
	}

	void ProgramObjectImpl::SetUniform(const std::string& uniformName, const std::vector<GLint>& values, GLint count) {
		check_is_member(uniformName);
		check_type(GL_FLOAT);
		GLsizei uniform_count = count >= 0 ? (GLsizei) count : (GLsizei) values.size();
		glProgramUniform1iv(programId, GetUniformLocation(uniformName), uniform_count, values.data());
	}

	void ProgramObjectImpl::SetUniform(const std::string& uniformName, GLuint value) {
		check_is_member(uniformName);
		check_type(GL_UNSIGNED_INT);
		glProgramUniform1ui(programId, GetUniformLocation(uniformName), value);
		check_opengl();
	}

	void ProgramObjectImpl::SetUniform(const std::string& uniformName, const std::vector<GLuint>& values, GLint count) {
		check_is_member(uniformName);
		check_type(GL_FLOAT);
		GLsizei uniform_count = count >= 0 ? (GLsizei) count : (GLsizei) values.size();
		glProgramUniform1uiv(programId, GetUniformLocation(uniformName), uniform_count, values.data());
	}

	void ProgramObjectImpl::SetUniform(const std::string& uniformName, GLfloat value) {
		check_is_member(uniformName);
		check_type(GL_FLOAT);
		glProgramUniform1f(programId, GetUniformLocation(uniformName), value);
	}

	void ProgramObjectImpl::SetUniform(const std::string& uniformName, const std::vector<GLfloat>& values, GLint count) {
		check_is_member(uniformName);
		check_type(GL_FLOAT);
		GLsizei uniform_count = count >= 0 ? (GLsizei) count : (GLsizei) values.size();
		glProgramUniform1fv(programId, GetUniformLocation(uniformName), uniform_count, values.data());
	}

	void ProgramObjectImpl::SetUniform(const std::string& uniformName, const glm::ivec2& value) {
		check_is_member(uniformName);
		check_type(GL_INT_VEC2);
		glProgramUniform2iv(programId, GetUniformLocation(uniformName), 1, glm::value_ptr(value));
	}

	void ProgramObjectImpl::SetUniform(const std::string& uniformName, const glm::vec2& value) {
		check_is_member(uniformName);
		check_type(GL_FLOAT_VEC2);
		glProgramUniform2fv(programId, GetUniformLocation(uniformName), 1, glm::value_ptr(value));
	}

	void ProgramObjectImpl::SetUniform(const std::string& uniformName, const std::vector<glm::vec2>& values, GLint count) {
		check_is_member(uniformName);
		check_type(GL_FLOAT_VEC2);
		GLsizei uniform_count = count >= 0 ? (GLsizei) count : (GLsizei) values.size();
		glProgramUniform2fv(programId, GetUniformLocation(uniformName), uniform_count, (float *)values.data());
	}

	void ProgramObjectImpl::SetUniform(const std::string& uniformName, const glm::ivec3& value) {
		check_is_member(uniformName);
		check_type(GL_INT_VEC3);
		glProgramUniform3iv(programId, GetUniformLocation(uniformName), 1, glm::value_ptr(value) );
	}

	void ProgramObjectImpl::SetUniform(const std::string& uniformName, const glm::vec3& value) {
		check_is_member(uniformName);
		check_type(GL_FLOAT_VEC3);
		glProgramUniform3fv(programId, GetUniformLocation(uniformName), 1, glm::value_ptr(value));
	}

	void ProgramObjectImpl::SetUniform(const std::string& uniformName, const std::vector<glm::vec3>& values, GLint count) {
		check_is_member(uniformName);
		check_type(GL_FLOAT_VEC3);
		GLsizei uniform_count = count >= 0 ? (GLsizei) count : (GLsizei) values.size();
		glProgramUniform3fv(programId, GetUniformLocation(uniformName), uniform_count, (float *)values.data());
	}

	void ProgramObjectImpl::SetUniform(const std::string& uniformName, const glm::ivec4& value) {
		check_is_member(uniformName);
		check_type(GL_INT_VEC4);
		glProgramUniform4iv(programId, GetUniformLocation(uniformName), 1, glm::value_ptr(value));
	}

	void ProgramObjectImpl::SetUniform(const std::string& uniformName, const glm::vec4& value) {
		check_is_member(uniformName);
		check_type(GL_FLOAT_VEC4);
		glProgramUniform4fv(programId, GetUniformLocation(uniformName), 1, glm::value_ptr(value));
	}

	void ProgramObjectImpl::SetUniform(const std::string& uniformName, const glm::quat& value) {
		check_is_member(uniformName);
		check_type(GL_FLOAT_VEC4);
		glProgramUniform4fv(programId, GetUniformLocation(uniformName), 1, glm::value_ptr(value));
	}

	void ProgramObjectImpl::SetUniform(const std::string& uniformName, const std::vector<glm::vec4>& values, GLint count) {
		check_is_member(uniformName);
		check_type(GL_FLOAT_VEC4);
		GLsizei uniform_count = count >= 0 ? (GLsizei) count : (GLsizei) values.size();
		glProgramUniform4fv(programId, GetUniformLocation(uniformName), uniform_count, (float *)values.data());
	}

	void ProgramObjectImpl::SetUniform(const std::string& uniformName, const std::vector<glm::quat>& values, GLint count) {
		check_is_member(uniformName);
		check_type(GL_FLOAT_VEC4);
		GLsizei uniform_count = count >= 0 ? (GLsizei) count : (GLsizei) values.size();
		glProgramUniform4fv(programId, GetUniformLocation(uniformName), uniform_count, (float *)values.data());
	}

	void ProgramObjectImpl::SetUniform(const std::string& uniformName, const glm::mat3x3& value, GLboolean transpose) {
		check_is_member(uniformName);
		check_type(GL_FLOAT_MAT3);
		glProgramUniformMatrix3fv(programId, GetUniformLocation(uniformName), 1, transpose, glm::value_ptr(value));
	}

	void ProgramObjectImpl::SetUniform(const std::string& uniformName, const std::vector<glm::mat3x3>& values, GLboolean transpose, GLint count) {
		check_is_member(uniformName);
		check_type(GL_FLOAT_MAT3);
		GLsizei matrix_count = count >= 0 ? (GLsizei) count : (GLsizei) values.size();
		glProgramUniformMatrix3fv(programId, GetUniformLocation(uniformName), matrix_count, transpose, (float *)values.data());
	}

	void ProgramObjectImpl::SetUniform(const std::string& uniformName, const glm::mat4x4& value, GLboolean transpose) {
		check_is_member(uniformName);
		check_type(GL_FLOAT_MAT4);
		glProgramUniformMatrix4fv(programId, GetUniformLocation(uniformName), 1, transpose, glm::value_ptr(value));
	}

	void ProgramObjectImpl::SetUniform(const std::string& uniformName, const std::vector<glm::mat4x4>& values, GLboolean transpose, GLint count) {
		check_is_member(uniformName);
		check_type(GL_FLOAT_MAT4);
		GLsizei matrix_count = count >= 0 ? (GLsizei) count : (GLsizei) values.size();
		glProgramUniformMatrix4fv(programId, GetUniformLocation(uniformName), matrix_count, transpose, (float *)values.data());
	}

	void ProgramObjectImpl::GetUniform(const std::string& uniformName, GLint *value) {
		check_is_member(uniformName);
		GetOpenGLError(); // we do not want to make a direct type check here, as we would have to check INT, BOOL and all SAMPLER types ..
		glGetUniformiv(programId, GetUniformLocation(uniformName), value);
		GetOpenGLError();
	}

	void ProgramObjectImpl::GetUniform(const std::string& uniformName, GLuint *value) {
		check_is_member(uniformName);
		check_type(GL_UNSIGNED_INT);
		glGetUniformuiv(programId, GetUniformLocation(uniformName), value);
	}

	void ProgramObjectImpl::GetUniform(const std::string& uniformName, GLfloat *value) {
		check_is_member(uniformName);
		check_type(GL_FLOAT);
		glGetUniformfv(programId, GetUniformLocation(uniformName), value);
	}

	void ProgramObjectImpl::GetUniform(const std::string& uniformName, glm::vec2 *value) {
		check_is_member(uniformName);
		check_type(GL_FLOAT_VEC2);
		glGetUniformfv(programId, GetUniformLocation(uniformName), glm::value_ptr(*value));
	}

	void ProgramObjectImpl::GetUniform(const std::string& uniformName, glm::vec3 *value) {
		check_is_member(uniformName);
		check_type(GL_FLOAT_VEC3);
		glGetUniformfv(programId, GetUniformLocation(uniformName), glm::value_ptr(*value));
	}

	void ProgramObjectImpl::GetUniform(const std::string& uniformName, glm::vec4 *value) {
		check_is_member(uniformName);
		check_type(GL_FLOAT_VEC4);
		glGetUniformfv(programId, GetUniformLocation(uniformName), glm::value_ptr(*value));
	}

	void ProgramObjectImpl::GetUniform(const std::string& uniformName, glm::mat3x3 *value) {
		check_is_member(uniformName);
		check_type(GL_FLOAT_MAT3);
		glGetUniformfv(programId, GetUniformLocation(uniformName), glm::value_ptr(*value));
	}

	void ProgramObjectImpl::GetUniform(const std::string& uniformName, glm::mat4x4 *value) {
		check_is_member(uniformName);
		check_type(GL_FLOAT_MAT4);
		glGetUniformfv(programId, GetUniformLocation(uniformName), glm::value_ptr(*value));
	}

	
	template <class T>
	static void SetTextureT(ProgramObjectImpl* program, const T& texture, const std::string& uniformSamplerName, GLuint textureUnit, GLint sampler)
	{
		check_is_member(uniformSamplerName);
		texture->Bind(textureUnit);
		program->SetUniform(uniformSamplerName, GLint(textureUnit) );
		if (sampler >= 0) glBindSampler(textureUnit, sampler);
	}

	void ProgramObjectImpl::SetTexture(const std::string& uniformSamplerName, const TextureObject1D& texture, GLuint textureUnit, SamplerObject sampler) {
		SetTextureT<TextureObject1D>(this, texture, uniformSamplerName, textureUnit, sampler->id);
	}

	void ProgramObjectImpl::SetTexture(const std::string& uniformSamplerName, const TextureObject2D& texture, GLuint textureUnit, SamplerObject sampler) {
		SetTextureT<TextureObject2D>(this, texture, uniformSamplerName, textureUnit, sampler->id);
	}

	void ProgramObjectImpl::SetTexture(const std::string& uniformSamplerName, const TextureObject3D& texture, GLuint textureUnit, SamplerObject sampler) {
		SetTextureT<TextureObject3D>(this, texture, uniformSamplerName, textureUnit, sampler->id);
	}

	void ProgramObjectImpl::SetTexture(const std::string& uniformSamplerName, const TextureObjectCube& texture, GLuint textureUnit, SamplerObject sampler) {
		SetTextureT<TextureObjectCube>(this, texture, uniformSamplerName, textureUnit, sampler->id);
	}

	void ProgramObjectImpl::SetTexture(const std::string& uniformSamplerName, const TextureObjectBuffer& texture, GLuint textureUnit, SamplerObject sampler) {
		SetTextureT<TextureObjectBuffer>(this, texture, uniformSamplerName, textureUnit, sampler->id);
	}

	void ProgramObjectImpl::SetTexture(const std::string& uniformSamplerName, const TextureObject2DArray& texture, GLuint textureUnit, SamplerObject sampler) {
		SetTextureT<TextureObject2DArray>(this, texture, uniformSamplerName, textureUnit, sampler->id);
	}

	void ProgramObjectImpl::SetTexture(const std::string& uniformSamplerName, const TextureObjectCubeArray& texture, GLuint textureUnit, SamplerObject sampler) {
		SetTextureT<TextureObjectCubeArray>(this, texture, uniformSamplerName, textureUnit, sampler->id);
	}

	void ProgramObjectImpl::SetImageTexture(const TextureObject& texture, GLuint textureUnit, GLenum access, GLint level, GLboolean layered, GLint layer)
	{
		check_is_member(uniformImageName);
		// TODO : Check texture.internalFormat is correct for glBindImageTexture (cannot be rgb texture, only 1, 2, or 4 channels)
		glBindImageTexture(textureUnit, texture->id, level, layered, layer, access, texture->internalFormat);
		check_opengl();
	}

	//void ProgramObjectImpl::setTexture(std::string uniformSamplerName, const TextureObject& texture, GLenum textureTarget) {
	//	check_is_member(uniformSamplerName);
	//	int textureUnit = getTextureUnit(uniformSamplerName);
	//	setUniform(uniformSamplerName, textureUnit);
	//	glActiveTexture(GL_TEXTURE0 + textureUnit);
	//	glBindTexture(textureTarget, textureId);
	//}

	//void ProgramObjectImpl::setImageTexture(std::string uniformImageName, Texture *texture, GLint level, GLboolean layered, GLint layer, GLenum access) {
	//	check_is_member(uniformImageName);
	//	int textureUnit = getImageTextureUnit(uniformImageName);
	//	setUniform(uniformImageName, textureUnit);
	//	glBindImageTexture(textureUnit, texture->textureId, level, layered, layer, access, texture->internalFormat);
	//}

	void ProgramObjectImpl::CompileProgram(const ShaderPaths& paths, LogToFile* log, const Detail::ShaderCommandLine& cmd)
	{
		{ /// Initialize render program

			GLuint vertexShader = 0;
			GLuint tessControlShader = 0;
			GLuint tessEvalShader = 0;
			GLuint geometryShader = 0;
			GLuint fragmentShader = 0;

			ShaderCompiler compiler(log);

			GLuint newProgram = glCreateProgram();

			if (paths.vertexShaderFilename.size())
			{ //// Load vertex shader source

				String vertShaderFullFilename = paths.commonPath + paths.vertexShaderFilename;
				vertexShader = compiler.Create(GL_VERTEX_SHADER, vertShaderFullFilename, cmd);
				check_opengl();
			}

			if (paths.tesselationControlShaderFilename.size())
			{ //// Load tesselation control shader source
				String tessControlShaderFullFilename = paths.commonPath + paths.tesselationControlShaderFilename;
				tessControlShader = compiler.Create(GL_TESS_CONTROL_SHADER, tessControlShaderFullFilename, cmd);
				check_opengl();
			}

			if (paths.tesselationEvaluationShaderFilename.size())
			{ //// Load tesselation evaluation shader source
				String tessEvalShaderFullFilename = paths.commonPath + paths.tesselationEvaluationShaderFilename;
				tessEvalShader = compiler.Create(GL_TESS_EVALUATION_SHADER, tessEvalShaderFullFilename, cmd);
				check_opengl();
			}

			if (paths.geometryShaderFilename.size())
			{ //// Load geometry shader source

				String geomShaderFullFilename = paths.commonPath + paths.geometryShaderFilename;
				geometryShader = compiler.Create(GL_GEOMETRY_SHADER, geomShaderFullFilename, cmd);
				check_opengl();
			}
		

			if (paths.fragmentShaderFilename.size())
			{ //// Load fragment shader source

				String fragShaderFullFilename = paths.commonPath + paths.fragmentShaderFilename;
				fragmentShader = compiler.Create(GL_FRAGMENT_SHADER, fragShaderFullFilename, cmd);
				check_opengl();
			}

			{ //// Compile shaders and create programs
				bool compileCheck = compiler.Check();
				Assert(compileCheck);

				if (vertexShader)      glAttachShader(newProgram, vertexShader);
				if (tessControlShader) glAttachShader(newProgram, tessControlShader);
				if (tessEvalShader)    glAttachShader(newProgram, tessEvalShader);
				if (geometryShader)    glAttachShader(newProgram, geometryShader);
				if (fragmentShader)    glAttachShader(newProgram, fragmentShader);


				glDeleteShader(vertexShader);
				glDeleteShader(tessControlShader);
				glDeleteShader(tessEvalShader);
				glDeleteShader(fragmentShader);
				glDeleteShader(geometryShader);

				glLinkProgram(newProgram);
				check_opengl();
			}

			bool valid = CheckProgram(newProgram, log);
			Verify(valid);

			if (valid) {
				//// Release previous program
				glDeleteProgram(programId);
				check_opengl();
				programId = newProgram;
			} else {
				glDeleteProgram(newProgram);
				check_opengl();
			}
		}
	}

	void ProgramObjectImpl::CompileProgram(const ShaderPaths& paths, LogToFile* log)
	{
		CompileProgram(paths, log, Detail::ShaderCommandLine());
	}


	void ProgramObjectImpl::CompileComputeProgram(LogToFile* log, String computeShaderFilename, const Detail::ShaderCommandLine& cmd)
	{
		{ /// Initialize render program

			GLuint computeShader = 0;

			ShaderCompiler compiler(log);

			GLuint newProgram = glCreateProgram();

			{ //// Load vertex shader source
				String compShaderFullFilename = computeShaderFilename;
				computeShader = compiler.Create(GL_COMPUTE_SHADER, compShaderFullFilename, cmd);
				check_opengl();
			}

			{ //// Compile shader and create program
				bool compileCheck = compiler.Check();
				Assert(compileCheck);

				glAttachShader(newProgram, computeShader);

				glDeleteShader(computeShader);

				glLinkProgram(newProgram);
				check_opengl();
			}

			bool valid = CheckProgram(newProgram, log);
			Verify(valid);

			if (valid) {
				//// Release previous program
				glDeleteProgram(programId);
				check_opengl();
				programId = newProgram;
				return;
			} else {
				glDeleteProgram(newProgram);
				check_opengl();
				return;
			}
		}
	}

	void ProgramObjectImpl::CompileComputeProgram(LogToFile* log, String computeShaderFilename)
	{
		CompileComputeProgram(log, computeShaderFilename, Detail::ShaderCommandLine());
	}

	void ProgramObjectImpl::DispatchCompute(const int& gridSize, const int& groupSize, bool useVariableGroupSizeIfAvailable)
	{
		DispatchCompute(glm::ivec3(gridSize, 1, 1), glm::ivec3(groupSize, 1, 1), useVariableGroupSizeIfAvailable);
	}

	void ProgramObjectImpl::DispatchCompute(const glm::ivec2& gridSize, const glm::ivec2& groupSize, bool useVariableGroupSizeIfAvailable)
	{
		DispatchCompute(glm::ivec3(gridSize.x, gridSize.y, 1), glm::ivec3(groupSize.x, groupSize.y, 1), useVariableGroupSizeIfAvailable);
	}

	void ProgramObjectImpl::DispatchCompute(const glm::ivec3& gridSize, const glm::ivec3& groupSize, bool useVariableGroupSizeIfAvailable)
	{
		Use();

		GLuint groupsX = groupSize.x == 0 || gridSize.x == 0 ? 0 : (gridSize.x - 1) / groupSize.x + 1;
		GLuint groupsY = groupSize.y == 0 || gridSize.y == 0 ? 0 : (gridSize.y - 1) / groupSize.y + 1;
		GLuint groupsZ = groupSize.z == 0 || gridSize.z == 0 ? 0 : (gridSize.z - 1) / groupSize.z + 1;

		if (useVariableGroupSizeIfAvailable && glDispatchComputeGroupSizeARB) {
			glDispatchComputeGroupSizeARB(groupsX, groupsY, groupsZ, groupSize.x, groupSize.y, groupSize.z);
		}
		else {
			glDispatchCompute(groupsX, groupsY, groupsZ);
		}

		check_opengl(); 
	}



#pragma pop_macro("check_type")
#pragma pop_macro("check_is_member")

}

