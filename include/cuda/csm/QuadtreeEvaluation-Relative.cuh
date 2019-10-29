#pragma once

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include "csm/QuadtreeTypes.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//typedef Quadtree::children_t gpu_children_t;


namespace Quadtree{



	void evaluateShadow_Relative(const size_t width,
		                         const size_t height,
								 const uint32_t pcf_size,
								 const cuda_quadtree_data& qdata,
								 const cuda_scene_data& sdata,
								 const cudaTextureObject_t worldPosTex,
								 const cudaTextureObject_t worldDPosTex,
								 const cudaTextureObject_t worldNorTex,
								 const cudaSurfaceObject_t resultSurf);


};
