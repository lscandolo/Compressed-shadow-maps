#pragma once

#include <vector>
#include <cinttypes>

#include <glm/glm.hpp>

struct ShapeBuffers
{
	std::vector<glm::vec3>    positions;
	std::vector<glm::vec3>    normals;
	std::vector<glm::vec2>    texcoords;
	std::vector<glm::vec4>    tangents;
	std::vector<uint16_t>     indices;
};

void createSphere(unsigned int rings, unsigned int sectors, ShapeBuffers& sb);

void createCube(ShapeBuffers& sb);

void createQuad(ShapeBuffers& sb);
