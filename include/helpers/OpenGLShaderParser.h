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

#include <map>
#include <set>
#include <tuple>

namespace GLHelpers 
{
	
	typedef std::pair<std::string, std::string> NameValue;

	namespace Detail
	{
		enum class GLSLProfile
		{
			CORE,
			COMPATIBILITY,
			NONE,
		};

		class ShaderCommandLine
		{
		public:
			typedef std::map<std::string, std::string>	DefineList;
			typedef std::vector< String >		        IncludeList;

			ShaderCommandLine( );
			ShaderCommandLine( const String& strCommandLine );

			int					m_iVersion;			///< GLSL version, might be -1
			GLSLProfile			m_Profile;			///< GLSL profile
			DefineList			m_Defines;			///< defines for GLSL source code
			IncludeList			m_Includes;			///< list of paths to include

			std::string& operator[](const std::string& lhs);
		};

		class PragmaData
		{
		public:
			std::string			m_strPragma;		///< contains the (l&r-)trimmed string without the '#pragma '.
		};
	

		// usually parser return an Abstract Syntax Tree (AST).
		// This parser returns additional info of the source code,
		// as well as the 'unfolded' source code (preprocessed)
		class ShaderParserOutput
		{
		public:
			ShaderParserOutput();
			ShaderParserOutput( ShaderParserOutput&& moveFrom );
			ShaderParserOutput& operator=( ShaderParserOutput&& moveFrom );

			ShaderParserOutput( const ShaderParserOutput& copyFrom )
			{
				Assert( 0 );
			}


			std::string							m_strPreprocessedSource;

			// for tracking from which file/define/etc. line x in the source did code come from
			// large shader source codes might need a lot of strings (for each line), thus this
			// is only a pointer to the internal cached string
			typedef std::pair< const String*, size_t >	SourceLineData;
			std::vector< SourceLineData >		m_LineTrack;

			// additional options via #pragma directive
			std::vector< PragmaData >			m_Pragmas;

			int									m_iVersion;

			/// @brief Will retrieve a cache entry. If the entry does not exist it will be created
			const String*						RetrieveEntry( const String& strEntry );
		private:
			std::set< String >					m_LineTrackCache;	///< std::set is safe for keeping a pointer from outside
		};

	}
	
	class ShaderParser
	{
	public:
		Detail::ShaderParserOutput				Parse( const Detail::ShaderCommandLine& cmd, const String& strFilename );
		Detail::ShaderParserOutput				ParseFromMemory( const Detail::ShaderCommandLine& cmd, const std::string& strSource  );

		Detail::ShaderParserOutput				ParseGeneric( const Detail::ShaderCommandLine& cmd, const std::string& strSource, const String& strOrigin  );
	private:
		Detail::ShaderParserOutput				ParseGeneric_Internal( const Detail::ShaderCommandLine& cmd, const std::string& strSource, const String& strOrigin  );

		Detail::ShaderParserOutput          	IncludeFile( const Detail::ShaderCommandLine& cmd, const std::string& strIncludeFileName, const String& strOrigin );
		void									ReplaceLine( const Detail::ShaderCommandLine& cmd, std::string& strLine, const String& strOrigin);
		NameValue                               ParseDefinition(const std::string& strLine);
	};

}


