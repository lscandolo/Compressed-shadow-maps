/**
 * @date 08/09/2011
 * @author Matthias Holländer
 * The character codes in this file are Copyright (c) 2009-2013 Matthias Holländer.
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
#include "helpers/OpenGLShaderObject.h"
#include "helpers/OpenGLBufferObject.h"
#include "helpers/OpenGLTextureObject.h"
#include "helpers/OpenGLShaderParser.h"
#include "helpers/OpenGLSamplerObject.h"

#include <glm/glm.hpp>

#include <vector>

namespace GLHelpers {

	// Helper for checking the linking status of a program object
	bool CheckProgram( GLuint uiProgramName, LoggerInterface* pLogger );


	class ProgramObjectImpl
	{

	public:
		GLuint programId;
		static size_t GetNumberOfBinaryFormats();
		static std::vector< GLint > GetBinaryFormats();

		ProgramObjectImpl();
		ProgramObjectImpl(GLenum eBinaryFormat, const char* pData, size_t sSize);

		~ProgramObjectImpl();

		bool Link();
		bool IsValid() const;
		void Delete();
		std::string GetLinkError() const;
		std::string GetError() const;
		std::string GetInfoLog() const;
		
		GLint GetUniformLocation(const char* szUniformName) const;
		GLint GetUniformLocation(const std::string strUniformName) const;
		GLint GetUniformBufferSize(const char* szBufferName) const;
		void  Use(bool enable = true) const;

		void   AttachShader(ShaderObject shaderObject) const;
		size_t GetNumberOfAttachedShaders() const;
		void   SetAttributeLocation(const char* attributeName, GLuint index) const;
		GLint  GetAttribLocation(const char* szAttribLocationName) const;
		size_t GetNumberOfActiveAttributes() const;
		size_t GetLengthLongestAttributeName() const;
		void   GetAttribute(GLuint uiAttribIndex, std::string& strAttributeName, GLint& iSize, GLenum& eType) const;
		void   GetAttribute(GLuint uiAttribIndex, std::string& strAttributeName, GLint& iSize, GLenum& eType, GLint& iLocation) const;
		void   GetAttribute(GLuint uiAttribIndex, std::string& strAttributeName, GLenum& type) const;
		void   GetAttribute(GLuint uiAttribIndex, std::string& strAttributeName, GLenum& type, GLint& iLocation) const;
		size_t GetNumberOfActiveUniforms() const;
		GLsizei GetLengthLongestUniformName() const;
		void   GetUniform(GLuint uiUniformIndex, std::string& strUniformName, GLint& iSize, GLenum& eType) const;
		void   GetUniform(GLuint uiUniformIndex, std::string& strUniformName, GLint& iSize, GLenum& eType, GLint& iLocation) const;
		GLuint GetUniformBlockIndex(const char* szUniformBlockName) const;
		void   SetUniformBlockBinding(GLuint uniformBlockIndex, GLuint uniformBlockBinding) const;
		void   SetUniformBlockBinding(const char* szUniformBlockName, GLuint uniformBlockBinding) const;
		size_t GetNumberOfActiveUniformBlocks() const;
		size_t GetLengthLongestUniformBlock() const;
		size_t GetUniformBlockDataSize(GLuint uiUniformBlockIndex) const;
		size_t GetUniformBlockActiveUniforms(GLuint uiUniformBlockIndex) const;
		size_t GetUniformBlockNameLength(GLuint uiUniformBlockIndex) const;
		void   SetUniformBlock(const char* szUniformBlockName, const BufferObject& buffer, GLuint uniformBlockBinding, GLintptr offset = 0, GLsizeiptr size = 0) const; // Gets block index, sets it to binding parameter passed and binds buffer to that binding point. Similar to SetTexture or SetImageTexture

		GLuint GetSSBOResourceIndex(const char* szUniformBlockName) const;
		void   SetSSBO(const char* szUniformBlockName, const BufferObject& buffer, GLuint bindingPointIndex, GLintptr offset = 0, GLsizeiptr size = 0) const;


		std::string GetActiveUniformName(GLuint uiUniformIndex) const;
		std::string GetUniformBlockName(GLuint uiUniformBlockIndex) const;

		void SetGeometryShaderNumVerticesOut(int32_t iNumVerticesOut) const;
		void SetGeometryShaderInputType(GLenum eType) const;
		void SetGeometryShaderOutputType(GLenum eType) const;

		GLint GetTessellationControlOutputVertices() const;
		GLint GetTessellationGenerationMode() const;
		GLint GetTessellationGenerationSpacing() const;
		GLint GetTessellationGenerationVertexOrder() const;
		GLint GetTessellationGenerationPointMode() const;

		void SetBinaryHint(bool bHint) const;
		std::unique_ptr< char[] > GetBinary(GLenum& refBinaryFormat, size_t& refSize) const;
		void UploadBinary(GLenum eBinaryFormat, const char* szData, size_t sSize);
		
		void SetUniform(const std::string& uniformName, GLboolean value);
		void SetUniform(const std::string& uniformName, GLint value);
		void SetUniform(const std::string& uniformName, GLuint value);
		void SetUniform(const std::string& uniformName, GLfloat value);
		void SetUniform(const std::string& uniformName, const glm::ivec2& value);
		void SetUniform(const std::string& uniformName, const glm::vec2& value);
		void SetUniform(const std::string& uniformName, const glm::ivec3& value);
		void SetUniform(const std::string& uniformName, const glm::vec3& value);
		void SetUniform(const std::string& uniformName, const glm::ivec4& value);
		void SetUniform(const std::string& uniformName, const glm::vec4& value);
		void SetUniform(const std::string& uniformName, const glm::quat& value);
		void SetUniform(const std::string& uniformName, const glm::mat3x3& value, GLboolean transpose = GL_FALSE);
		void SetUniform(const std::string& uniformName, const glm::mat4x4& value, GLboolean transpose = GL_FALSE);

		void SetUniform(const std::string& uniformName, const std::vector<GLint>& values, GLint count = -1);
		void SetUniform(const std::string& uniformName, const std::vector<GLuint>& values, GLint count = -1);
		void SetUniform(const std::string& uniformName, const std::vector<GLfloat>& values, GLint count = -1);
		void SetUniform(const std::string& uniformName, const std::vector<glm::vec2>& values, GLint count = -1);
		void SetUniform(const std::string& uniformName, const std::vector<glm::vec3>& values, GLint count = -1);
		void SetUniform(const std::string& uniformName, const std::vector<glm::vec4>& values, GLint count = -1);
		void SetUniform(const std::string& uniformName, const std::vector<glm::quat>& values, GLint count = -1);
		void SetUniform(const std::string& uniformName, const std::vector<glm::mat3x3>& values, GLboolean transpose = GL_FALSE, GLint count = -1);
		void SetUniform(const std::string& uniformName, const std::vector<glm::mat4x4>& values, GLboolean transpose = GL_FALSE, GLint count = -1);
	
		void GetUniform(const std::string& uniformName, GLint *value);
		void GetUniform(const std::string& uniformName, GLuint *value);
		void GetUniform(const std::string& uniformName, GLfloat *value);
		void GetUniform(const std::string& uniformName, glm::vec2 *value);
		void GetUniform(const std::string& uniformName, glm::vec3 *value);
		void GetUniform(const std::string& uniformName, glm::vec4 *value);
		void GetUniform(const std::string& uniformName, glm::mat3x3 *value);
		void GetUniform(const std::string& uniformName, glm::mat4x4 *value);
	
		// bind the given texture to the given uniform sampler name within this shader program
		void SetTexture(const std::string& uniformSamplerName, const TextureObject1D& texture,        GLuint textureUnit, SamplerObject sampler = NullSamplerObject);
		void SetTexture(const std::string& uniformSamplerName, const TextureObject2D& texture,        GLuint textureUnit, SamplerObject sampler = NullSamplerObject);
		void SetTexture(const std::string& uniformSamplerName, const TextureObject3D& texture,        GLuint textureUnit, SamplerObject sampler = NullSamplerObject);
		void SetTexture(const std::string& uniformSamplerName, const TextureObjectCube& texture,      GLuint textureUnit, SamplerObject sampler = NullSamplerObject);
		void SetTexture(const std::string& uniformSamplerName, const TextureObjectBuffer& texture,    GLuint textureUnit, SamplerObject sampler = NullSamplerObject);
		void SetTexture(const std::string& uniformSamplerName, const TextureObject2DArray& texture,   GLuint textureUnit, SamplerObject sampler = NullSamplerObject);
		void SetTexture(const std::string& uniformSamplerName, const TextureObjectCubeArray& texture, GLuint textureUnit, SamplerObject sampler = NullSamplerObject);

		void SetImageTexture(const TextureObject& texture, GLuint textureUnit, GLenum access, GLint level = 0, GLboolean layered = GL_TRUE, GLint layer = 0);

		struct ShaderPaths
		{
			String commonPath;
			String vertexShaderFilename;
			String tesselationControlShaderFilename;
			String tesselationEvaluationShaderFilename;
			String geometryShaderFilename;
			String fragmentShaderFilename;
		};

		void CompileComputeProgram(LogToFile* log, String computeShaderFilename, const GLHelpers::Detail::ShaderCommandLine& cmd);
		void CompileComputeProgram(LogToFile* log, String computeShaderFilename);


		void CompileProgram(const ShaderPaths& paths, LogToFile* log, const Detail::ShaderCommandLine& cmd);
		void CompileProgram(const ShaderPaths& paths, LogToFile* log);

		void DispatchCompute(const int& gridSize, const int& groupSize, bool useVariableGroupSizeIfAvailable = false);
		void DispatchCompute(const glm::ivec2& gridSize, const glm::ivec2& groupSize, bool useVariableGroupSizeIfAvailable = false);
		void DispatchCompute(const glm::ivec3& gridSize, const glm::ivec3& groupSize, bool useVariableGroupSizeIfAvailable = false);

	};
	using ProgramShaderPaths = ProgramObjectImpl::ShaderPaths;
	using ProgramObject = std::shared_ptr<ProgramObjectImpl>;
}
