#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>

#define __CUDA_INTERNAL_COMPILATION__
#include "math_functions.h"
#undef __CUDA_INTERNAL_COMPILATION__

#include <math.h>

__device__ static int2   make_int2(int3 v) { return make_int2(v.x,v.y); };
__device__ static int2   make_int2(int v) { return make_int2(v,v); };
__device__ static float3 make_float3(float v) { return make_float3(v,v,v); };
__device__ static int2   operator & (int2 a, int2 b) { return make_int2(a.x&b.x, a.y&b.y); };
__device__ static uint2  operator & (uint2 a, uint2 b) { return make_uint2(a.x&b.x, a.y&b.y); };
__device__ static int2   operator - (int2 a, int2 b) { return make_int2(a.x - b.x, a.y - b.y); };
__device__ static float3 operator - (float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); };
__device__ static int3   operator >> (int3 a, int b) { return make_int3(a.x >> b, a.y >> b, a.z >> b); };
__device__ static int2   min(int2 a, int2 b) { return make_int2(min(a.x, b.x), min(a.y, b.y)); }
__device__ static int2   max(int2 a, int2 b) { return make_int2(max(a.x, b.x), max(a.y, b.y)); }
__device__ static float3 min(float3 a, float3 b) { return make_float3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)); }
__device__ static float3 max(float3 a, float3 b) { return make_float3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)); }
__device__ static void   operator += (float3& a, float3 &b) { a.x += b.x; a.y += b.y; a.z += b.z; }
__device__ static void   operator *= (float3 &a, float b) { a.x *= b; a.y *= b; a.z *= b; }
__device__ static void   operator *= (float4& a, float b) { a.x *= b; a.y *= b; a.z *= b; a.w *= b; }
__device__ static void   operator /= (float4& a, float b) { a.x /= b; a.y /= b; a.z /= b; a.w /= b; };
__device__ static float3 make_float3(float4 a) { return make_float3(a.x, a.y, a.z); }
__device__ static int2   operator + (int2 a, int2 b) { return make_int2(a.x + b.x, a.y + b.y); }
__device__ static float2 operator - (const float2& a, const float2& b) { return make_float2(a.x - b.x, a.y - b.y); }
__device__ static float4 operator - (const float4& a, const float4& b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
__device__ static float2 operator + (const float2& a, const float2& b) { return make_float2(a.x + b.x, a.y + b.y); }
__device__ static float3 operator + (const float3& a, const float3& b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ static float4 operator + (const float4& a, const float4& b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.z + b.z); }
__device__ static float4 operator * (const float4& a, const float4& b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.z * b.z); }
__device__ static float2 operator * (const float2& a, const float& b) { return make_float2(a.x * b, a.y * b); }
__device__ static float3 operator * (const float3& a, const float& b) { return make_float3(a.x * b, a.y * b, a.z * b); }
__device__ static float3 operator * (const float3& a, const float3& b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
__device__ static float4 operator * (const float4& a, const float& b) { return make_float4(a.x * b, a.y * b, a.z * b, a.w * b); }
__device__ static int2   operator * (const int2& a, const float& b) { return make_int2(a.x * b, a.y * b); }
__device__ static float  dot(const float4 a, const float4 b) { return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }
__device__ static float  dot(const float3 a, const float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ static float  length(float3 a) { return sqrtf(dot(a, a)); }

__device__ static float  clamp(float a,  float m, float M) { return min(max(a,m),M); }
__device__ static float2 clamp(float2 a, float m, float M) { return make_float2(clamp(a.x, m, M), clamp(a.y, m, M)); }
__device__ static float3 clamp(float3 a, float m, float M) { return make_float3(clamp(a.x, m, M), clamp(a.y, m, M), clamp(a.z, m, M)); }
__device__ static float4 clamp(float4 a, float m, float M) { return make_float4(clamp(a.x, m, M), clamp(a.y, m, M), clamp(a.z, m, M), clamp(a.w, m, M)); }

__device__ static int  clamp(int a,  int m, int M) { return min(max(a,m),M); }
__device__ static int2 clamp(int2 a, int m, int M) { return make_int2(clamp(a.x, m, M), clamp(a.y, m, M)); }
__device__ static int3 clamp(int3 a, int m, int M) { return make_int3(clamp(a.x, m, M), clamp(a.y, m, M), clamp(a.z, m, M)); }
__device__ static int4 clamp(int4 a, int m, int M) { return make_int4(clamp(a.x, m, M), clamp(a.y, m, M), clamp(a.z, m, M), clamp(a.w, m, M)); }
