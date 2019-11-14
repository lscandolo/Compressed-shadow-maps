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
