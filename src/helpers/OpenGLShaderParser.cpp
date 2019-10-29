#include "common.h"
#include "helpers/OpenGLShaderParser.h"

#include <sstream>
#include <cctype>
#include <iostream>

/*
	Include order:
	Specifications of HLSL state: http://msdn.microsoft.com/en-us/library/windows/desktop/dd607349%28v=vs.85%29.aspx
	#include "filename" 	Searches for the include file:

		in the same directory as the file that contains the #include directive.
		in the directories of any files that contain a #include directive for the file that contains the #include directive.
		in paths specified by the /I compiler option, in the order in which they are listed.
		in paths specified by the INCLUDE environment variable, in the order in which they are listed.

		Note   The INCLUDE environment variable is ignored in an development environment.
			   Refer to your development environment's documentation for information about how to set the include paths for your project.

	#include <filename>	Searches for the include file:

		in paths specified by the /I compiler option, in the order in which they are listed.
		in paths specified by the INCLUDE environment variable, in the order in which they are listed.

		Note   The INCLUDE environment variable is ignored in an development environment.
			   Refer to your development environment's documentation for information about how to set the include paths for your project.


	// nice trick, but kind of despised
	//
	// http://stackoverflow.com/questions/15826884/glsl-binding-attributes-to-semantics
	// adding defines
	//char *sources[2] = { "#define FOO\n", sourceFromFile };
	//glShaderSourceARB(shader, 2, sources, NULL);

*/

namespace
{
	const std::string c_strTokenGLSLVersion = "#version ";
	const std::string c_strTokenGLSLInclude = "#include ";
	const std::string c_strTokenGLSLPragma  = "#pragma ";
	const std::string c_strTokenGLSLDefine  = "#define ";
	const std::string c_strTokenGLSLIfDef   = "#ifdef ";
	const std::string c_strTokenGLSLIfNDef  = "#ifndef ";

	const String c_strFromMemory = String( "<From memory>" );
}


namespace GLHelpers {

	std::string IncludeFile( const std::string& strNameFromGLSL )
	{
		return std::string();
	}

	Detail::ShaderParserOutput ShaderParser::ParseGeneric( const Detail::ShaderCommandLine& cmd, const std::string& strSource, const String& strOrigin )
	{
		Detail::ShaderParserOutput ret = ParseGeneric_Internal( cmd, strSource, strOrigin );

		// make sure the version is the first line
		if ( ret.m_iVersion != -1 ) {
			std::stringstream ss;
			ss << "#version " << ret.m_iVersion << std::endl;

			ret.m_strPreprocessedSource = ss.str() + ret.m_strPreprocessedSource;
			const String* pLineTrack = ret.RetrieveEntry( String( "<added by Mx::OpenGL::ShaderParser>" ) );
			ret.m_LineTrack.insert( ret.m_LineTrack.begin(), std::make_pair( pLineTrack, 0 ) );			// add to front
		}

		// make sure to move it out so that the cache-pointers can be maintained
		return std::move( ret );
	}


	Detail::ShaderParserOutput ShaderParser::ParseFromMemory( const Detail::ShaderCommandLine& cmd, const std::string& strSource )
	{
		return ParseGeneric( cmd, strSource, c_strFromMemory );
	}

	void Append( Detail::ShaderParserOutput& d, const Detail::ShaderParserOutput& d2 )
	{
		d.m_strPreprocessedSource.append(d2.m_strPreprocessedSource);

		// Merge two shader parser outputs. used for includes
		//Assert( 0 );
	}

    NameValue ShaderParser::ParseDefinition(const std::string& strLine)
    {
		NameValue nv;

		std::size_t definePos = strLine.find(c_strTokenGLSLDefine); 
		if (definePos == std::string::npos) return nv;

		std::string::const_iterator it;
		size_t nameStartPos, nameEndPos, valueStartPos, valueEndPos;

		{ // Get name string
			it = strLine.begin() + definePos + c_strTokenGLSLDefine.size();

			while (it != strLine.end() && std::isspace(*it)) it++; // Skip spaces, tabs, etc.

			nameStartPos = std::distance(strLine.begin(), it);

			while (it != strLine.end() && !std::isspace(*it)) it++; // Advance until next whitespace

			nameEndPos = std::distance(strLine.begin(), it);

			nv.first = MakeString(strLine.substr(nameStartPos, nameEndPos - nameStartPos));
		}
		
		
		{ // Get value string

			while (it != strLine.end() && std::isspace(*it)) it++; // Skip spaces, tabs, etc.

			valueStartPos = std::distance(strLine.begin(), it);

			while (it != strLine.end() && !std::isspace(*it)) it++; // Advance until next whitespace

			valueEndPos = std::distance(strLine.begin(), it);

			nv.second = MakeString(strLine.substr(valueStartPos, valueEndPos - valueStartPos));
		}

		return nv;

    }


	void ShaderParser::ReplaceLine(const Detail::ShaderCommandLine& cmd, std::string& strLine, const String& strOrigin)
	{
			// (todo: check that we're not modifying a define of the same name, for now it works because it transforms 
			//        a line that looks like #define name (whatever) into #define value value, which is stupid but works )
				
			// String replace from https://stackoverflow.com/questions/3418231/replace-part-of-a-string-with-another-string
				
			std::size_t versionPos = strLine.find(c_strTokenGLSLVersion); // Skip #version lines
			if (versionPos != std::string::npos) return;

			std::size_t includePos = strLine.find(c_strTokenGLSLInclude); // Skip #include lines
			if (includePos != std::string::npos) return;

			std::size_t ifdefPos = strLine.find(c_strTokenGLSLIfDef); // Skip #ifdef lines
			if (includePos != std::string::npos) return;

			std::size_t ifndefPos = strLine.find(c_strTokenGLSLIfNDef); // Skip #ifndef lines
			if (includePos != std::string::npos) return;

			auto ReplaceDefinitionImpl = [](std::string& str, const std::string& name, const std::string& value)
			{
				if (name.empty()) return;

				size_t start_pos = 0;
				while ((start_pos = str.find(name, start_pos)) != std::string::npos) {
					str.replace(start_pos, name.length(), value);
					start_pos += value.length(); // Advance position to skip until after the replaced value
				}
			};

			// Remove defines of a value in the cmd define list
			// That means that **the cmd defines OVERRIDE the source code define **

			NameValue nv = ParseDefinition(strLine);

			for (const auto& replacePair : cmd.m_Defines)
			{
				std::string name = ToStdString(replacePair.first);
				std::string value = ToStdString(replacePair.second);

				// If this is a define line, only replace if it's not defining the same name
				if (!nv.first.empty() && ToStdString(nv.first) == name) {

                    #if defined( _DEBUG ) || defined( DEBUG )
					    std::cout << "Warning: Overriding definition of " << name << " during shader parsing. Source: " << ToStdString(strOrigin) << std::endl;
                    #endif

					continue;
				}


				ReplaceDefinitionImpl(strLine, name, value);
			}
	}

	Detail::ShaderParserOutput ShaderParser::IncludeFile( const Detail::ShaderCommandLine& cmd, const std::string& strIncludeFileName, const String& strOrigin )
	{

		if ( strOrigin == c_strFromMemory ) {
			// From memory cannot include
			Assert( !"Shader strings from memory cannot include other files, currently." );
			return Detail::ShaderParserOutput();
		}

		// glsl is currently parsed in non-wide characters
		//Mx::Path p = Mx::Path( String( strIncludeFileName ) );
		Path p = Path( MakeString( strIncludeFileName ) );

		// checks for absolute path
		if( !p.has_root_directory() ) {
			Path pathOrigin( strOrigin );
			p = pathOrigin.parent_path() / p;
		}

		auto includeData = Parse( cmd, MakeString(p.string()) );
		
		return includeData;
	}

	Detail::ShaderParserOutput ShaderParser::ParseGeneric_Internal( const Detail::ShaderCommandLine& cmd, const std::string& strSource, const String& strOrigin  )
	{
		// Remove all comments before processing, this will not harm the actual lines in the code
		std::string strWithoutComments = strSource;//StringNS::RemoveCppComments( strSource ); //!!

		// This is what we have to return
		Detail::ShaderParserOutput ret;
		ret.m_iVersion = cmd.m_iVersion;

		// Actual parsing of #include
		std::string Line, Text;

		// ShaderCommandLine with no version for includes
		Detail::ShaderCommandLine noVersionCmd = cmd;
		noVersionCmd.m_iVersion = -1;

		std::istringstream ssIn( strWithoutComments );
		std::stringstream ssTokenizer;
		std::string strLine;
		std::string strToken;
		std::size_t sPos = std::string::npos;

		std::stringstream ssOut;
		int iOrgLineCount = -1;

		while( std::getline( ssIn, strLine ) ) {
			++iOrgLineCount;

			if ( strLine.empty() ) continue;
			if (strLine == "\r") continue;

			{ // Process #version
				sPos = strLine.find(c_strTokenGLSLVersion);		// space included!
				if (sPos != std::string::npos) {
					ssTokenizer.str(""); ssTokenizer.clear();
					ssTokenizer << strLine.substr(sPos + c_strTokenGLSLVersion.length());
					int iVersionToken = -1;
					ssTokenizer >> iVersionToken;

					ret.m_iVersion = std::max(ret.m_iVersion, iVersionToken);
					continue;
				}
			}

			{ // Process #include
				sPos = strLine.find(c_strTokenGLSLInclude);
				if (sPos != std::string::npos) {
					// call Generic_Internal

					// find the "" or ( todo <>)
					auto sPosTemp0 = strLine.find('\"', sPos);
					if (sPosTemp0 == std::string::npos) {
						Assert(0);
						continue;
					}

					// to simplify computations eat the "
					++sPosTemp0;

					auto sPosTemp1 = strLine.find('\"', sPosTemp0);
					if (sPosTemp0 == std::string::npos) {
						Assert(0);
						continue;
					}

					std::string strIncludeFileName = strLine.substr(sPosTemp0, sPosTemp1 - sPosTemp0);

					try {
						Detail::ShaderParserOutput inc = IncludeFile(noVersionCmd, strIncludeFileName, strOrigin);
						ssOut << inc.m_strPreprocessedSource;
					}
					catch (...) {
						Assert(!"Failed to include file");
					}
					continue;
				}
			}


						{ // Attempt replacement of defines contained in cmd 
				ReplaceLine(cmd, strLine, strOrigin);
			}

			// #pragmas will NOT be altered

			ssOut << strLine << std::endl;
			const String* pCacheEntry = ret.RetrieveEntry( strOrigin );
			ret.m_LineTrack.push_back( std::make_pair( pCacheEntry, iOrgLineCount ) );
		}

		ret.m_strPreprocessedSource = ssOut.str();

		return std::move( ret );
	}

	Detail::ShaderParserOutput ShaderParser::Parse( const Detail::ShaderCommandLine& cmd, const String& strFilename )
	{
		const std::vector< byte > fileData = LoadFile( strFilename );
		const std::string strOrgSourceCode( fileData.begin(), fileData.end() );

		return ParseGeneric( cmd, strOrgSourceCode, strFilename );
	}

	const String* Detail::ShaderParserOutput::RetrieveEntry( const String& strEntry )
	{
		const auto it = m_LineTrackCache.find( strEntry );
		if ( it != m_LineTrackCache.end() ) {
			const String& strRet = ( *it );
			return &strRet;
		}
		auto itInsert = m_LineTrackCache.insert( strEntry );
		const String& strRet = ( *itInsert.first );
		return &strRet;
	}

	Detail::ShaderParserOutput::ShaderParserOutput( ShaderParserOutput&& moveFrom ) :
		m_iVersion( moveFrom.m_iVersion )
	{
		m_LineTrackCache = std::move( moveFrom.m_LineTrackCache );
		m_LineTrack = std::move( moveFrom.m_LineTrack );
		m_strPreprocessedSource = std::move( moveFrom.m_strPreprocessedSource );

		moveFrom.m_LineTrackCache.clear();
	}

	Detail::ShaderParserOutput::ShaderParserOutput() :
		m_iVersion( -1 )
	{}

	Detail::ShaderParserOutput& Detail::ShaderParserOutput::operator=( ShaderParserOutput&& moveFrom )
	{
		m_iVersion = moveFrom.m_iVersion;
		m_LineTrackCache = std::move( moveFrom.m_LineTrackCache );
		m_LineTrack = std::move( moveFrom.m_LineTrack );
		m_strPreprocessedSource = std::move( moveFrom.m_strPreprocessedSource );

		moveFrom.m_LineTrackCache.clear();

		return *this;
	}

	//////////////////////////////////////////////////////////////////////////
	// ShaderCommandLine
	//////////////////////////////////////////////////////////////////////////
	Detail::ShaderCommandLine::ShaderCommandLine( ) :
		m_iVersion( -1 ),
		m_Profile( GLSLProfile::NONE )
	{
	}

	Detail::ShaderCommandLine::ShaderCommandLine( const String& strCommandLine ) :
		m_iVersion( -1 ),
		m_Profile( GLSLProfile::NONE )
	{
		// @todo: extract some stuff...

	}

	std::string& Detail::ShaderCommandLine::operator[](const std::string& lhs) { 
		return m_Defines[lhs];
	}


}
