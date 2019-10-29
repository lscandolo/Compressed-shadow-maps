#include "common.h"
#include "csm/Tiler.h"

#include "glm/gtc/matrix_transform.hpp"

//////////////////////////////////////////////////////////////////////////////////////
//////////// Matrix functions
//////////////////////////////////////////////////////////////////////////////////////

static glm::mat4 createRMatrix(glm::vec3 F, glm::vec3 U)
{
	glm::mat4 rMatrix = glm::identity<glm::mat4>();
	F = glm::normalize(F);
	U = glm::normalize(U);
	glm::vec3 R = glm::cross(F,U);
	R = glm::normalize(R);

	U = glm::cross(R,F);
	U = glm::normalize(U);

	rMatrix[0][0] = R.x;
	rMatrix[1][0] = R.y;
	rMatrix[2][0] = R.z;

	rMatrix[0][1] = U.x;
	rMatrix[1][1] = U.y;
	rMatrix[2][1] = U.z;

	rMatrix[0][2] = -F.x;
	rMatrix[1][2] = -F.y;
	rMatrix[2][2] = -F.z;

	return rMatrix;
}

static
glm::mat4 createTMatrix(glm::vec4 T)
{
	glm::mat4 tMatrix = glm::identity<glm::mat4>();
	tMatrix[3][0] = T.x;
	tMatrix[3][1] = T.y;
	tMatrix[3][2] = T.z;

	return tMatrix;
}

static
glm::mat4 createTMatrix(glm::vec3 T)
{
	glm::mat4 tMatrix = glm::identity<glm::mat4>();
	tMatrix[3][0] = T.x;
	tMatrix[3][1] = T.y;
	tMatrix[3][2] = T.z;

	return tMatrix;
}

Tiler::TileParameters
Tiler::computeTileParameters(Input input, glm::ivec3 tile_coord)
{
	TileParameters sample;
	sample.tile_coord = tile_coord;
	sample.tile_qty   = input.tile_qty;

	bool useSphereBounds = false;
	glm::vec3 D = glm::normalize(input.light_dir);
	glm::mat4 rMatrix = createRMatrix(D, DEFAULT_UP_VECTOR);

	if (!input.is_dir_light)
	{
		const float& z = input.light_near_plane;
		const float& Z = input.light_far_plane;
		glm::vec3& pos = input.light_spotpos;
		float radangle = input.light_spotangle * float(M_PI) /180.f;

		glm::vec4 P = glm::vec4(pos, 1.0);


		glm::mat4 pMatrix = glm::perspective(radangle, 1.f, z, Z);
		glm::mat4 tMatrix = createTMatrix(-P);
		glm::mat4 vpMatrix = pMatrix * rMatrix * tMatrix;

		sample.untiled_vpmatrix = vpMatrix;
		sample.tile_vpmatrix = vpMatrix;
		sample.light_dir = D;
		sample.light_pos = glm::vec3(P.x, P.y, P.z) / P.w;

		sample.near_plane = z;
		sample.far_plane = Z;

		float r = 1.0, l = -1.0;
		float t = 1.0, b = -1.0;
		float xstep = (r - l) / input.tile_qty.x;
		l = l + xstep * tile_coord.x;
		r = l + xstep;

		float ystep = (t - b) / input.tile_qty.y;
		b = b + ystep * tile_coord.y;
		t = b + ystep;

		glm::mat4 oMatrix = glm::ortho(l, r, b, t, 1.f, -1.f);
		sample.tile_vpmatrix = oMatrix * vpMatrix;

	} else {// Directional light

		glm::vec3 maxV = input.bbox.max;
		glm::vec3 minV = input.bbox.min;

		bbox3 rBounds, slicedBounds;
		glm::vec3 sphereCenter = input.bsphere.center;
		sphereCenter = rMatrix * glm::vec4(sphereCenter, 1.f);
		float d = 0;
		float dy = 0;
		for (int xi = 0; xi < 2; ++xi) {
			for (int yi = 0; yi < 2; ++yi) {
				for (int zi = 0; zi < 2; ++zi) {
					glm::vec3 vertex;
					vertex.x = xi ? maxV.x : minV.x;
					vertex.y = yi ? maxV.y : minV.y;
					vertex.z = zi ? maxV.z : minV.z;
					rBounds.extend(rMatrix * glm::vec4(vertex, 1.f));

					d = std::max(d, float((input.bbox.center() - vertex).length()) );
				}
			}
		}

		float z = 0.f, Z = 0.f , l = 0.f , b = 0.f , r = 0.f , t = 0.f;
		if (useSphereBounds) {
			////////////////////////////////////////
			////////////// For Sphere Bounds
			////////////////////////////////////////
			z = -sphereCenter.z - input.bsphere.radius;
			Z = -sphereCenter.z + input.bsphere.radius;

			l = sphereCenter.x - input.bsphere.radius;
			r = sphereCenter.x + input.bsphere.radius;
			b = sphereCenter.y - input.bsphere.radius;
			t = sphereCenter.y + input.bsphere.radius;
			//////////////////////////////////////////////////////////////
		} else {
			////////////////////////////////////////
			////////////// For Cube Bounds
			////////////////////////////////////////
			z = -rBounds.max.z;
			Z = -rBounds.min.z;

			maxV = rBounds.max;
			minV = rBounds.min;

			glm::vec3 centroid(rBounds.center());

			l = minV.x;
			r = maxV.x;

			b = minV.y;
			t = maxV.y;
		}

		glm::mat4 untiledProjMatrix = glm::ortho(l, r, b, t, z - 0.001f, Z);

		//////////////////// Tiled matrix
		float xstep = (r - l) / input.tile_qty.x;
		l = l + xstep * tile_coord.x;
		r = l + xstep;

		float ystep = (t - b) / input.tile_qty.y;
		b = b + ystep * tile_coord.y;
		t = b + ystep;
		///////////////////////////

		glm::mat4 projMatrix = glm::ortho(l, r, b, t, z - 0.001f, Z);

		sample.untiled_vpmatrix = untiledProjMatrix * rMatrix;
		sample.tile_vpmatrix = projMatrix  * rMatrix;
		sample.light_dir = D;
		sample.near_plane = z;
		sample.far_plane = Z;
	}

	return sample;
}
