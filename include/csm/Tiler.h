#pragma once

#include "common.h"
#include "glm/glm.hpp"
#include "math_types.h"

#define DEFAULT_UP_VECTOR (glm::vec3(0.0, 1.0, 0.0))

class Tiler
{
public:

	struct Input
	{
		bool      is_dir_light;

		glm::vec3 light_dir;
		glm::vec3 light_spotpos;
		float     light_spotangle;
		float     light_near_plane;
		float     light_far_plane;
		
		bbox3 bbox;
		bsphere3 bsphere;

		glm::ivec3 tile_qty;
	};
	
	struct TileParameters
	{
		glm::ivec3     tile_qty;
		glm::ivec3     tile_coord;
		glm::mat4      tile_vpmatrix;
		glm::mat4      untiled_vpmatrix;
		glm::mat4      untiled_vmatrix;
		glm::vec3      light_pos;
		glm::vec3      light_dir;
		float          near_plane;
		float          far_plane;
	};

	TileParameters computeTileParameters(Input input, glm::ivec3 tile_coord);
};

