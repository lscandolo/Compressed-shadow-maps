#pragma once

#include <cuda_runtime.h>


void
createMinMipmapLevel(cudaSurfaceObject_t prevLevel, cudaSurfaceObject_t nextLevel, dim3 texDim);

void
createMaxMipmapLevel(cudaSurfaceObject_t prevLevel, cudaSurfaceObject_t nextLevel, dim3 texDim);

