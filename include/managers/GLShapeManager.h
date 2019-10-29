#pragma once

#include "common.h"
#include "helpers/Singleton.h"
#include "helpers/Shapes.h"
#include "helpers/OpenGLHelpers.h"

#include <glm/glm.hpp>
#include <type_traits>

struct GLShape
{
	GLShape();

	ShapeBuffers data;
	GLHelpers::BufferObject      posbuffer;
	GLHelpers::BufferObject      normalbuffer;
	GLHelpers::BufferObject      texcoordbuffer;
	GLHelpers::BufferObject      tangentbuffer;
	GLHelpers::BufferObject      indexbuffer;
	GLHelpers::VertexArrayObject vao;

	void destroy();
};


class GLShapeManagerInstance
{

public:
	void           initialize();
	void           finalize();

	const GLShape& getUnitSphere();
	const GLShape& getUnitCube();
	const GLShape& getUnitQuad();

protected:
	GLShapeManagerInstance();
	~GLShapeManagerInstance();

	bool initialized;
	GLShape unit_sphere;
	GLShape unit_cube;
	GLShape unit_quad;

	

	friend Singleton<GLShapeManagerInstance>;
};

typedef Singleton<GLShapeManagerInstance> GLShapeManager;
