#pragma once

#include "common.h"

namespace GLHelpers {


	/// @brief Encapsulates an OpenGL Shader Object (Vertex, TessCtrl, TessEval, Geometry, Fragment)
	class ShaderObjectImpl
	{
	public:
		ShaderObjectImpl();
		~ShaderObjectImpl();

		bool							Compile() const;

		std::string						GetError() const;
		std::string						GetInfoLog() const;

		/// @brief For adding source code to this shader object
		/// @param szCode Source code of the shader, must be zero-terminated
		void							UploadTextCode( const char* szCode, size_t sNumCharacters = 0 );

		/// @brief For adding source code to this shader object
		/// @param listCode Array of source code of the shader, must all be zero-terminated
		void							UploadTextCode( std::vector< const GLchar* >& listCode );

		GLuint							GetOpenGLID() const
		{ 
			return m_uiShaderHandle;
		}

		/// @brief Wrapper around the glReleaseShaderCompiler
		static void 					ReleaseShaderCompiler();
		
	private:
		GLuint							m_uiShaderHandle;		///< OpenGL ID


	}; // end of class ShaderObject
	using ShaderObject = std::shared_ptr<ShaderObjectImpl>;
	ShaderObject create_shader_object();

}
