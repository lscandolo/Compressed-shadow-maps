/**
 * @author Matthias Holländer and Leonardo Scandolo
 * The character codes in this file are Copyright(c)
 * 2009 - 2013 Matthias Holländer.
 * 2015 - 2019 Leonardo Scandolo.
 * All rights reserved.
 *
 * No claims are made as to fitness for any particular purpose.
 * No warranties of any kind are expressed or implied.
 * The recipient agrees to determine applicability of information provided.
 *
 * The origin of this software, code, or partial code must not be misrepresented;
 * you must not claim that you wrote the original software.
 * Further, the author's name must be stated in any code or partial code.
 * If you use this software, code or partial code (including in the context
 * of a software product), an acknowledgment in the product documentation
 * is required.
 *
 * Altered source versions must be plainly marked as such, and must not be
 * misrepresented as being the original software.
 *
 * In no event shall the author or copyright holders be liable for any claim,
 * damages or other liability, whether in an action of contract, tort or otherwise,
 * arising from, out of or in connection with the software or the use or other
 * dealings in the software. Further, in no event shall the author be liable
 * for any direct, indirect, incidental, special, exemplary, or consequential
 * damages (including, but not limited to, procurement of substitute goods or
 * service; loss of use, data, or profits; or business interruption).
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies, portions or substantial portions of the Software.
 */

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
