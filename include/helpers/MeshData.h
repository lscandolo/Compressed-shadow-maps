#pragma once

#include "common.h"
#include "math_types.h"

#include "helpers/OpenGLHelpers.h"

#include <tiny_obj_loader.h>

#include <set>

struct MaterialStructGL
{
	glm::uvec2 alpha_texhandle;
	glm::uvec2 ambient_texhandle;
	glm::uvec2 bump_texhandle;
	glm::uvec2 diffuse_texhandle;
	glm::uvec2 displacement_texhandle;
	glm::uvec2 emissive_texhandle;
	glm::uvec2 metallic_texhandle;
	glm::uvec2 normal_texhandle;
	glm::uvec2 reflection_texhandle;
	glm::uvec2 roughness_texhandle;
	glm::uvec2 sheen_texhandle;
	glm::uvec2 specular_highlight_texhandle;

	union { glm::vec3 ambient;       GLfloat _pad0[4]; };
	union { glm::vec3 diffuse;       GLfloat _pad1[4]; };
	union { glm::vec3 specular;      GLfloat _pad2[4]; };
	union { glm::vec3 transmittance; GLfloat _pad3[4]; };
	union { glm::vec3 emission;      GLfloat _pad4[3]; };

	GLfloat     shininess;

	// pbr
	GLfloat roughness;            // [0, 1] default 0
	GLfloat metallic;             // [0, 1] default 0
	GLfloat sheen;                // [0, 1] default 0
	GLfloat _pad5;
};

struct Material
{
	std::string name;

	MaterialStructGL gl;

	std::string alpha_texname;
	std::string ambient_texname;
	std::string bump_texname;
	std::string diffuse_texname;
	std::string displacement_texname;
	std::string emissive_texname;
	std::string metallic_texname;
	std::string normal_texname;
	std::string reflection_texname;
	std::string roughness_texname;
	std::string sheen_texname;
	std::string specular_highlight_texname;
};

struct MaterialData
{
	MaterialData();
	GLHelpers::BufferObject  materialListBuffer;
	std::vector<Material>    materials;

	int  updateGLResources();
	void clearGLResources();
	void setDirty();

private:
	bool gldirty;
};

struct Submesh
{
	std::string name;
	int start_index;
	int end_index;
	int largest_index_value;
	int lowest_index_value;
	int default_material_index;
};

struct MeshDataGL
{
	MeshDataGL();

	GLHelpers::VertexArrayObject vao;
	GLHelpers::BufferObject      posbuffer;
	GLHelpers::BufferObject      normalbuffer;
	GLHelpers::BufferObject      texcoordbuffer;
	GLHelpers::BufferObject      tangentbuffer;
	GLHelpers::BufferObject      colorbuffer;
	GLHelpers::BufferObject      matidbuffer;

	GLHelpers::BufferObject      indexbuffer;
	GLenum                       indexbuffertype; // GL_UNSIGNED_SHORT or GL_UNSIGNED_INT
};

struct MeshData
{
	std::vector<glm::vec3> positions;
	std::vector<glm::vec3> normals;
	std::vector<glm::vec2> texcoords;
	std::vector<glm::vec4> tangents;
	std::vector<glm::vec3> colors;
	std::vector<GLshort>   matIds;
	std::vector<GLuint>    indices;
	std::vector<Submesh>   submeshes;

	MeshDataGL gl;

	MaterialData           materialdata;
		
	// scale : final size of largest side of bbox of mesh file (if <= 0, keep original)
	int  load_obj(std::string filename, float scale = 0.f, bool optimize = true);

	void clear();
	void clearCPU();
	void clearGPU();
	void reloadGPU();


	float     original_scale;
	bbox3     bbox;
	bsphere3  bsphere;
	std::string source_filename;

	void makeGLTexturesResident();
	void makeGLTexturesNotResident();

private:

};


