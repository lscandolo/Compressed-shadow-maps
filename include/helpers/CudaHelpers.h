#pragma once

#include <vector_types.h>
#include <vector_functions.h>
#include <glm/glm.hpp>

////////// GLM TO CUDA
inline float2 make_float2(glm::vec2 v) { return make_float2(v.x, v.y); }
inline float3 make_float3(glm::vec3 v) { return make_float3(v.x, v.y, v.z); }
inline float4 make_float4(glm::vec4 v) { return make_float4(v.x, v.y, v.z, v.w); }

inline int2 make_int2(glm::ivec2 v) { return make_int2(v.x, v.y); }
inline int3 make_int3(glm::ivec3 v) { return make_int3(v.x, v.y, v.z); }
inline int4 make_int4(glm::ivec4 v) { return make_int4(v.x, v.y, v.z, v.w); }

inline uint2 make_uint2(glm::uvec2 v) { return make_uint2(v.x, v.y); }
inline uint3 make_uint3(glm::uvec3 v) { return make_uint3(v.x, v.y, v.z); }
inline uint4 make_uint4(glm::uvec4 v) { return make_uint4(v.x, v.y, v.z, v.w); }

////////// CUDA TO GLM
inline glm::vec2 make_vec2(float2 v) { return glm::vec2(v.x, v.y); }
inline glm::vec3 make_vec3(float3 v) { return glm::vec3(v.x, v.y, v.z); }
inline glm::vec4 make_vec4(float4 v) { return glm::vec4(v.x, v.y, v.z, v.w); }

inline glm::ivec2 make_ivec2(int2 v) { return glm::ivec2(v.x, v.y); }
inline glm::ivec3 make_ivec3(int3 v) { return glm::ivec3(v.x, v.y, v.z); }
inline glm::ivec4 make_ivec4(int4 v) { return glm::ivec4(v.x, v.y, v.z, v.w); }

inline glm::uvec2 make_uvec2(uint2 v) { return glm::uvec2(v.x, v.y); }
inline glm::uvec3 make_uvec3(uint3 v) { return glm::uvec3(v.x, v.y, v.z); }
inline glm::uvec4 make_uvec4(uint4 v) { return glm::uvec4(v.x, v.y, v.z, v.w); }


