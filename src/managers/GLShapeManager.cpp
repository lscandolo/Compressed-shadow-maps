#include "managers/GLShapeManager.h"

using namespace GLHelpers;

static void createGLShape(GLShape& glshape)
{
	glshape.posbuffer->Create(GL_ARRAY_BUFFER);
	glshape.normalbuffer->Create(GL_ARRAY_BUFFER);
	glshape.texcoordbuffer->Create(GL_ARRAY_BUFFER);
	glshape.tangentbuffer->Create(GL_ARRAY_BUFFER);
	glshape.indexbuffer->Create(GL_ELEMENT_ARRAY_BUFFER);

	glshape.posbuffer->UploadData     (glshape.data.positions.size() * sizeof(glm::vec3), GL_STATIC_DRAW, glshape.data.positions.data());
	glshape.normalbuffer->UploadData  (glshape.data.normals.size()   * sizeof(glm::vec3), GL_STATIC_DRAW, glshape.data.normals.data());
	glshape.texcoordbuffer->UploadData(glshape.data.texcoords.size() * sizeof(glm::vec2), GL_STATIC_DRAW, glshape.data.texcoords.data());
	glshape.tangentbuffer->UploadData (glshape.data.tangents.size()  * sizeof(glm::vec4), GL_STATIC_DRAW, glshape.data.tangents.data());
	glshape.indexbuffer->UploadData   (glshape.data.indices.size()   * sizeof(uint16_t),  GL_STATIC_DRAW, glshape.data.indices.data());

	glshape.vao->Create();
	glshape.vao->SetAttributeBufferSource(glshape.posbuffer,      0, 3);
	glshape.vao->SetAttributeBufferSource(glshape.normalbuffer,   1, 3);
	glshape.vao->SetAttributeBufferSource(glshape.texcoordbuffer, 2, 2);
	glshape.vao->SetAttributeBufferSource(glshape.tangentbuffer,  3, 4);
}

GLShape::GLShape() :
	posbuffer(create_object<BufferObject>()),
	normalbuffer(create_object<BufferObject>()),
	texcoordbuffer(create_object<BufferObject>()),
	tangentbuffer(create_object<BufferObject>()),
	indexbuffer(create_object<BufferObject>()),
	vao(create_object<VertexArrayObject>())
	{
	}

void
GLShape::destroy()
{
	posbuffer->Release();
	normalbuffer->Release();
	texcoordbuffer->Release();
	tangentbuffer->Release();
	indexbuffer->Release();
	vao->Release();
}

void
GLShapeManagerInstance::initialize()
{
	if (initialized) return;

	createSphere(24, 24, unit_sphere.data);
	createCube(unit_cube.data);
	createQuad(unit_quad.data);

	createGLShape(unit_sphere);
	createGLShape(unit_cube);
	createGLShape(unit_quad);

	initialized = true;
}

void 
GLShapeManagerInstance::finalize()
{
	if (!initialized) return;

	unit_sphere.destroy();
	unit_cube.destroy();
	unit_quad.destroy();

	initialized = false;
}

GLShapeManagerInstance::GLShapeManagerInstance()
{
	initialized = false;
}

GLShapeManagerInstance::~GLShapeManagerInstance()
{}

const GLShape& GLShapeManagerInstance::getUnitSphere()
{
	return unit_sphere;
}

const GLShape& GLShapeManagerInstance::getUnitCube()
{
	return unit_cube;
}

const GLShape& GLShapeManagerInstance::getUnitQuad()
{
	return unit_quad;
}


