#include "helpers/Shapes.h"

using namespace glm;

static void computeTangents(const std::vector<vec3>& vertex, const std::vector<vec3>& normal,
	const std::vector<vec2>& texcoord, const std::vector<uint16_t>& indices, std::vector<vec4>& tangent)
{

	std::vector<vec3> tan1(vertex.size(), vec3(0, 0, 0));
	std::vector<vec3> tan2(vertex.size(), vec3(0, 0, 0));

	tangent.resize(vertex.size());

	for (size_t a = 0; a < indices.size() / 3; a++)
	{
		const unsigned int i1 = indices[3 * a + 0];
		const unsigned int i2 = indices[3 * a + 1];
		const unsigned int i3 = indices[3 * a + 2];

		const vec3& v1 = vertex[i1];
		const vec3& v2 = vertex[i2];
		const vec3& v3 = vertex[i3];

		const vec2& w1 = texcoord[i1];
		const vec2& w2 = texcoord[i2];
		const vec2& w3 = texcoord[i3];

		const float x1 = v2.x - v1.x;
		const float x2 = v3.x - v1.x;
		const float y1 = v2.y - v1.y;
		const float y2 = v3.y - v1.y;
		const float z1 = v2.z - v1.z;
		const float z2 = v3.z - v1.z;

		const float s1 = w2.x - w1.x;
		const float s2 = w3.x - w1.x;
		const float t1 = w2.y - w1.y;
		const float t2 = w3.y - w1.y;

		const float r = 1.0F / (s1 * t2 - s2 * t1);

		vec3 sdir((t2 * x1 - t1 * x2) * r,
			(t2 * y1 - t1 * y2) * r,
			(t2 * z1 - t1 * z2) * r);

		vec3 tdir((s1 * x2 - s2 * x1) * r,
			(s1 * y2 - s2 * y1) * r,
			(s1 * z2 - s2 * z1) * r);

		tan1[i1] += sdir;
		tan1[i2] += sdir;
		tan1[i3] += sdir;

		tan2[i1] += tdir;
		tan2[i2] += tdir;
		tan2[i3] += tdir;

	}

	for (long a = 0; a < vertex.size(); a++)
	{
		const vec3& n = normal[a];
		const vec3& t = tan1[a];

		// Gram-Schmidt orthogonalize
		vec3 tanxyz = normalize(t - n * dot(n, t));

		// Calculate handedness
		float tanw = (dot(cross(n, t), tan2[a]) < 0.0f) ? -1.0f : 1.0f;

		tangent[a] = vec4(tanxyz, tanw);
	}

}

void createSphere(unsigned int rings, unsigned int sectors, ShapeBuffers& sb)
{
	const float radius = 1.f;
	const float R = 1.f / (float)(rings - 1);
	const float S = 1.f / (float)(sectors - 1);
	unsigned int r, s;

	constexpr float pi = 3.14159265358979323846f;
	constexpr float pi_2 = pi / 2.f;

	sb.positions.resize(rings * sectors);
	sb.normals.resize(rings * sectors);
	sb.texcoords.resize(rings * sectors);
	std::vector<vec3>::iterator v = sb.positions.begin();
	std::vector<vec3>::iterator n = sb.normals.begin();
	std::vector<vec2>::iterator t = sb.texcoords.begin();
	for (r = 0; r < rings; r++) {
		for (s = 0; s < sectors; s++) {
			const float y = sin(-pi_2 + pi * r * R);
			const float x = cos(2 * pi * s * S) * sin(pi * r * R);
			const float z = sin(2 * pi * s * S) * sin(pi * r * R);

			t->x = s * S;
			t->y = r * R;
			t++;

			v->x = x * radius;
			v->y = y * radius;
			v->z = z * radius;
			v++;

			n->x = x;
			n->y = y;
			n->z = z;
			n++;
		}
	}

	// For quad indices
	//indices.resize(rings * sectors * 4);
	//std::vector<uint16_t>::iterator i = indices.begin();
	//for (r = 0; r < rings; r++) for (s = 0; s < sectors; s++) {
	//	*i++ = r * sectors + s;
	//	*i++ = r * sectors + (s + 1);
	//	*i++ = (r + 1) * sectors + (s + 1);
	//	*i++ = (r + 1) * sectors + s;
	//}

	sb.indices.resize((rings - 1) * (sectors - 1) * 6);
	std::vector<uint16_t>::iterator i = sb.indices.begin();
	for (r = 0; r < rings-1; r++) {
		for (s = 0; s < sectors-1; s++) {
			*i++ = (r + 1) * sectors + (s + 1);  // 2
			*i++ = r * sectors + (s + 1); // 1
			*i++ = r * sectors + s; // 0

			*i++ = (r + 1) * sectors + s; // 3
			*i++ = (r + 1) * sectors + (s + 1);  // 2
			*i++ = r * sectors + s; // 0
		}
	}

	computeTangents(sb.positions, sb.normals, sb.texcoords, sb.indices, sb.tangents);

}

void createCube(ShapeBuffers& sb)
{


	sb.indices = {
		 0,  1,  2,  0,  2,  3,   //front
		 4,  6,  5,  4,  7,  6,   //right
		 8,  10, 9,  8,  11, 10,  //back
		 12, 13, 14, 12, 14, 15,  //left
		 16, 18, 17, 16, 19, 18,  //upper
		 20, 21, 22, 20, 22, 23 }; //bottom
		 //0,  1,  2,  0,  2,  3,   //front
		 //4,  5,  6,  4,  6,  7,   //right
		 //8,  9,  10, 8,  10, 11,  //back
		 //12, 13, 14, 12, 14, 15,  //left
		 //16, 17, 18, 16, 18, 19,  //upper
		 //20, 21, 22, 20, 22, 23 }; //bottom

	sb.positions = {
		//front
		vec3(-1.0, -1.0,  1.0),
		vec3( 1.0, -1.0,  1.0),
		vec3( 1.0,  1.0,  1.0),
		vec3(-1.0,  1.0,  1.0),

		//right
		 vec3( 1.0,  1.0,  1.0),
		 vec3( 1.0,  1.0, -1.0),
		 vec3( 1.0, -1.0, -1.0),
		 vec3( 1.0, -1.0,  1.0),

		 //back
		 vec3(-1.0, -1.0, -1.0),
		 vec3( 1.0, -1.0, -1.0),
		 vec3( 1.0,  1.0, -1.0),
		 vec3(-1.0,  1.0, -1.0),

		 //left
		 vec3(-1.0, -1.0, -1.0),
		 vec3(-1.0, -1.0,  1.0),
		 vec3(-1.0,  1.0,  1.0),
		 vec3(-1.0,  1.0, -1.0),

		 //upper
		 vec3( 1.0,  1.0,  1.0),
		 vec3(-1.0,  1.0,  1.0),
		 vec3(-1.0,  1.0, -1.0),
		 vec3( 1.0,  1.0, -1.0),

		  //bottom
		  vec3(-1.0, -1.0, -1.0),
		  vec3( 1.0, -1.0, -1.0),
		  vec3( 1.0, -1.0,  1.0),
		  vec3(-1.0, -1.0,  1.0)
	};

	sb.texcoords = {
		//front
		vec2(0.0, 0.0),
		vec2(1.0, 0.0),
		vec2(1.0, 1.0),
		vec2(0.0, 1.0),

		//right
		vec2(1.0, 1.0),
		vec2(1.0, 0.0),
		vec2(0.0, 0.0),
		vec2(0.0, 1.0),

		//back
		vec2(0.0, 0.0),
		vec2(1.0, 0.0),
		vec2(1.0, 1.0),
		vec2(0.0, 1.0),

		//left
		vec2(0.0, 0.0),
		vec2(0.0, 1.0),
		vec2(1.0, 1.0),
		vec2(1.0, 0.0),

		//upper
		vec2(1.0, 1.0),
		vec2(0.0, 1.0),
		vec2(0.0, 0.0),
		vec2(1.0, 0.0),

		//bottom
		vec2(0.0, 0.0),
		vec2(1.0, 0.0),
		vec2(1.0, 1.0),
		vec2(0.0, 1.0)
	};

	sb.normals = {
		//front
		vec3( 0.0, 0.0, 1.0),
		vec3( 0.0, 0.0, 1.0),
		vec3( 0.0, 0.0, 1.0),
		vec3( 0.0, 0.0, 1.0),

		//right
		vec3( 1.0, 0.0, 0.0),
		vec3( 1.0, 0.0, 0.0),
		vec3( 1.0, 0.0, 0.0),
		vec3( 1.0, 0.0, 0.0),

		//back
		vec3( 0.0, 0.0, -1.0),
		vec3( 0.0, 0.0, -1.0),
		vec3( 0.0, 0.0, -1.0),
		vec3( 0.0, 0.0, -1.0),

		//left
		vec3(-1.0, 0.0, 0.0),
		vec3(-1.0, 0.0, 0.0),
		vec3(-1.0, 0.0, 0.0),
		vec3(-1.0, 0.0, 0.0),

		//upper
		vec3( 0.0, 1.0, 0.0),
		vec3( 0.0, 1.0, 0.0),
		vec3( 0.0, 1.0, 0.0),
		vec3( 0.0, 1.0, 0.0),

		//bottom
		vec3( 0.0, -1.0, 0.0),
		vec3( 0.0, -1.0, 0.0),
		vec3( 0.0, -1.0, 0.0),
		vec3( 0.0, -1.0, 0.0),
	};

	computeTangents(sb.positions, sb.normals, sb.texcoords, sb.indices, sb.tangents);

}

void createQuad(ShapeBuffers& sb)
{

	sb.indices = { 0,  1,  2,  0,  2,  3 }; //front 

	sb.positions = {
		//front
		vec3(-1.0, -1.0,  0.0),
		vec3( 1.0, -1.0,  0.0),
		vec3( 1.0,  1.0,  0.0),
		vec3(-1.0,  1.0,  0.0)
	};

	sb.texcoords = {
		//front
		vec2(0.0, 0.0),
		vec2(1.0, 0.0),
		vec2(1.0, 1.0),
		vec2(0.0, 1.0) };

	sb.normals = {
		//front
		vec3(0.0, 0.0, 1.0),
		vec3(0.0, 0.0, 1.0),
		vec3(0.0, 0.0, 1.0),
		vec3(0.0, 0.0, 1.0)
	};

	computeTangents(sb.positions, sb.normals, sb.texcoords, sb.indices, sb.tangents);
}
