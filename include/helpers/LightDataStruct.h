#pragma once

#ifndef __CUDACC__
#include <glm/glm.hpp>

struct LightDataStructGL
{
	int   type;
	float intensity;
	float spot_angle;
	float radius;

	union { glm::vec3 color;     float _pad1[4]; };
	union { glm::vec3 position;  float _pad2[4]; };
	union { glm::vec3 direction; float _pad3[4]; };
	union { glm::vec3 v1;        float _pad4[4]; };
	union { glm::vec3 v2;        float _pad5[4]; };
	union { glm::vec3 v3;        float _pad6[4]; };
	union { glm::vec3 v4;        float _pad7[4]; };

	// v1, .. , v4 parallelogram corners
};
#endif // __CUDACC__

#include <vector_types.h>
struct LightDataStructCUDA
{
	int   type;
	float intensity;
	float spot_angle;
	float radius;

	union { float3 color;     float _pad1[4]; };
	union { float3 position;  float _pad2[4]; };
	union { float3 direction; float _pad3[4]; };
	union { float3 v1;        float _pad4[4]; };
	union { float3 v2;        float _pad5[4]; };
	union { float3 v3;        float _pad6[4]; };
	union { float3 v4;        float _pad7[4]; };

	// v1, .. , v4 parallelogram corners
};