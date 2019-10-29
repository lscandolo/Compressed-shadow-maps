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


