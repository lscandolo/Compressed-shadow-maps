#include "helpers/OpenGLBufferObject.h"

#include <set>

namespace GLHelpers {


	///////////////////////////////////////////////////////
	///////////////////////// BufferObjectImpl

	BufferObjectImpl::BufferObjectImpl()
		: id(0)
		, target(GL_INVALID_ENUM)
		, bytesize(0)
		, usage(GL_INVALID_ENUM)
	{}

	BufferObjectImpl::~BufferObjectImpl()
	{
		Release();
	}
		
	void BufferObjectImpl::Create(GLenum targetHint)
	{
		Release();
#if OPENGL_BINDLESS
		glCreateBuffers(1, &id);
#else
		glGenBuffers(1, &id);
#endif
		target = targetHint;

		check_opengl();
	}

	void BufferObjectImpl::Release()
	{
		if ( id ) {
			glDeleteBuffers( 1, &id );
			*this = BufferObjectImpl(); // Set initial values
		}
		check_opengl();
	}

	void BufferObjectImpl::Bind(GLenum targetHint) const
	{
		if (targetHint == GL_INVALID_ENUM) targetHint = target;
		glBindBuffer(targetHint, id);

		check_opengl();
	}

	void BufferObjectImpl::Unbind(GLenum targetHint) const
	{
		if (targetHint == GL_INVALID_ENUM) targetHint = target;
		glBindBuffer(targetHint, 0);

		check_opengl();
	}

	void BufferObjectImpl::UploadData(GLsizeiptr databytesize, GLenum usage, const void* data, GLenum targetHint)
	{
		if (targetHint == GL_INVALID_ENUM) targetHint = target;
#if OPENGL_BINDLESS
		glNamedBufferData( id, databytesize, data, usage);
#else
		Bind(targetHint);
		glBufferData(targetHint, databytesize, data, usage);
#endif
		
		usage = usage;
		bytesize = databytesize;

		check_opengl();
	}

	void BufferObjectImpl::DownloadData(void* data, GLsizeiptr databytesize, GLintptr byteOffset) const
	{
		if (databytesize == 0 || databytesize > bytesize) databytesize = bytesize;


#if OPENGL_BINDLESS
		glGetNamedBufferSubData(id, byteOffset, databytesize, data);
#else
		Bind(target);
		glGetBufferSubData(target, byteOffset, databytesize, data);
#endif

		check_opengl();
	}

	///////////////////////////////////////////////////////
	///////////////////////// VertexArrayObjectImpl

	VertexArrayObjectImpl::VertexArrayObjectImpl() :
		id( 0 )
	{}

	VertexArrayObjectImpl::~VertexArrayObjectImpl() 
	{
		Release();
	}

	void VertexArrayObjectImpl::Create()
	{
		Release();
		glGenVertexArrays(1, &id);

		check_opengl();
	}

	void VertexArrayObjectImpl::Release()
	{
		if ( id ) 
		{
			glDeleteVertexArrays( 1, &id );
			id = 0;
			m_enabledAttributes.clear();
		}
		check_opengl();
	}

	void VertexArrayObjectImpl::Bind() const
	{
		Assert(id);
		glBindVertexArray(id);

		check_opengl();
	}

	void VertexArrayObjectImpl::Unbind()
	{
		glBindVertexArray(0);

		check_opengl();
	}

	void VertexArrayObjectImpl::SetAttributeBufferSource(BufferObject bufferObject, GLuint index, GLint components, GLenum type, GLboolean normalized, GLsizei stride, const GLvoid * pointer)
	{
		Bind();
		bufferObject->Bind(GL_ARRAY_BUFFER);
		
		static const std::set<GLenum> int_types   { GL_BYTE, GL_UNSIGNED_BYTE, GL_SHORT, GL_UNSIGNED_SHORT, GL_INT, GL_UNSIGNED_INT };
		static const std::set<GLenum> float_types { GL_HALF_FLOAT, GL_FLOAT, GL_DOUBLE, GL_FIXED, GL_INT_2_10_10_10_REV, GL_UNSIGNED_INT_2_10_10_10_REV, GL_UNSIGNED_INT_10F_11F_11F_REV };
		static const std::set<GLenum> long_types  { GL_DOUBLE };

		if (float_types.count(type) > 0) {
			glVertexAttribPointer(index, components, type, normalized, stride, pointer);
		} else if (int_types.count(type) > 0) {
			glVertexAttribIPointer(index, components, type, stride, pointer);
		} else if (long_types.count(type) > 0) {
			glVertexAttribLPointer(index, components, type, stride, pointer);
		} else {
			throw(ExceptionOpenGL("Wrong type sent to VertexArrayObjectImpl::SetAttributeBufferSource"));
			return;
		}



		EnableAttribute(index, false);

		check_opengl();
	}
	
	void VertexArrayObjectImpl::EnableAttribute(GLuint index, bool preBind)
	{

#if OPENGL_BINDLESS
		if (!m_enabledAttributes[index]) {
			glEnableVertexArrayAttrib(id, index);
			m_enabledAttributes[index] = true;
		}
#else 
		if (preBind) Bind();

		if (!m_enabledAttributes[index]) {
			glEnableVertexAttribArray(index);
			m_enabledAttributes[index] = true;
		}
#endif

		check_opengl();
	}

	void VertexArrayObjectImpl::DisableAttribute(GLuint index, bool preBind)
	{
#if OPENGL_BINDLESS
		if (m_enabledAttributes[index]) {
			glDisableVertexArrayAttrib(id, index);
			m_enabledAttributes[index] = false;
		}
#else 
		if (preBind) Bind();

		if (m_enabledAttributes[index]) {
			glDisableVertexAttribArray(index);
			m_enabledAttributes[index] = false;
		}
#endif

		check_opengl();
	}

	///////////////////////////////////////////////////////


	//PackedMeshBuffers::PackedMeshBuffers()
	//	: positionBuffer(nullptr)
	//	, normalBuffer(nullptr)
	//	, texCoordBuffer(nullptr)
	//	, indexBuffer(nullptr)
	//	, materialIndicesBuffer(nullptr)
	//	, materials(nullptr)
	//	, materialGroups(nullptr)
	//{}

	//PackedSceneBuffers::PackedSceneBuffers()
	//	: meshBuffers(nullptr)
	//	, materials(nullptr)
	//	, materialSSBO(nullptr)
	//	, scene(nullptr)
	//{}
}
