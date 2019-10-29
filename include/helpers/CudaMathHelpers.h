#pragma once

#include <vector_types.h>
#include <vector_functions.h>
#include <optixu/optixu_math.h>

#ifdef __CUDACC__
#define _FTYPE __device__
#else
#define _FTYPE  
#endif


//////////// ABS
_FTYPE inline float2 abs(float2 v) { return make_float2(fabsf(v.x), fabsf(v.y)); }
_FTYPE inline float3 abs(float3 v) { return make_float3(fabsf(v.x), fabsf(v.y), fabsf(v.z)); }
_FTYPE inline float4 abs(float4 v) { return make_float4(fabsf(v.x), fabsf(v.y), fabsf(v.z), fabsf(v.w)); }

_FTYPE inline int2 abs(int2 v) { return make_int2(abs(v.x), abs(v.y)); }
_FTYPE inline int3 abs(int3 v) { return make_int3(abs(v.x), abs(v.y), abs(v.z)); }
_FTYPE inline int4 abs(int4 v) { return make_int4(abs(v.x), abs(v.y), abs(v.z), abs(v.w)); }

////////// MOD
_FTYPE inline float2 fmod(const float2& v, float f) { return make_float2(fmod(v.x, f), fmod(v.y, f)); }
_FTYPE inline float3 fmod(const float3& v, float f) { return make_float3(fmod(v.x, f), fmod(v.y, f), fmod(v.z, f)); }
_FTYPE inline float4 fmod(const float4& v, float f) { return make_float4(fmod(v.x, f), fmod(v.y, f), fmod(v.z, f), fmod(v.w, f)); }

////////// ISNAN
_FTYPE inline bool anynan(const float2& v) { return isnan(v.x) || isnan(v.y); }
_FTYPE inline bool anynan(const float3& v) { return isnan(v.x) || isnan(v.y) || isnan(v.z); }
_FTYPE inline bool anynan(const float4& v) { return isnan(v.x) || isnan(v.y) || isnan(v.z) || isnan(v.w); }

////////// ISINF
_FTYPE inline bool anyinf(const float2& v) { return isinf(v.x) || isinf(v.y); }
_FTYPE inline bool anyinf(const float3& v) { return isinf(v.x) || isinf(v.y) || isinf(v.z); }
_FTYPE inline bool anyinf(const float4& v) { return isinf(v.x) || isinf(v.y) || isinf(v.z) || isinf(v.w); }

////////// ISFINITE
_FTYPE inline bool allfinite(const float2& v) { return isfinite(v.x) && isfinite(v.y); }
_FTYPE inline bool allfinite(const float3& v) { return isfinite(v.x) && isfinite(v.y) && isfinite(v.z); }
_FTYPE inline bool allfinite(const float4& v) { return isfinite(v.x) && isfinite(v.y) && isfinite(v.z) && isfinite(v.w); }

/* cuda optix namespace covers this
//////////// Operators
//// Plus
_FTYPE inline float2 operator+(const float2& lhs, const float2& rhs) { return make_float2(lhs.x + rhs.x, lhs.y + rhs.y); }
_FTYPE inline float3 operator+(const float3& lhs, const float3& rhs) { return make_float3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z); }
_FTYPE inline float4 operator+(const float4& lhs, const float4& rhs) { return make_float4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w); }

_FTYPE inline int2 operator+(const int2& lhs, const int2& rhs) { return make_int2(lhs.x + rhs.x, lhs.y + rhs.y); }
_FTYPE inline int3 operator+(const int3& lhs, const int3& rhs) { return make_int3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z); }
_FTYPE inline int4 operator+(const int4& lhs, const int4& rhs) { return make_int4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w); }

_FTYPE inline uint2 operator+(const uint2& lhs, const uint2& rhs) { return make_uint2(lhs.x + rhs.x, lhs.y + rhs.y); }
_FTYPE inline uint3 operator+(const uint3& lhs, const uint3& rhs) { return make_uint3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z); }
_FTYPE inline uint4 operator+(const uint4& lhs, const uint4& rhs) { return make_uint4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w); }

//// Minus
_FTYPE inline float2 operator-(const float2& lhs, const float2& rhs) { return make_float2(lhs.x - rhs.x, lhs.y - rhs.y); }
_FTYPE inline float3 operator-(const float3& lhs, const float3& rhs) { return make_float3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z); }
_FTYPE inline float4 operator-(const float4& lhs, const float4& rhs) { return make_float4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w); }

_FTYPE inline int2 operator-(const int2& lhs, const int2& rhs) { return make_int2(lhs.x - rhs.x, lhs.y - rhs.y); }
_FTYPE inline int3 operator-(const int3& lhs, const int3& rhs) { return make_int3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z); }
_FTYPE inline int4 operator-(const int4& lhs, const int4& rhs) { return make_int4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w); }

_FTYPE inline uint2 operator-(const uint2& lhs, const uint2& rhs) { return make_uint2(lhs.x - rhs.x, lhs.y - rhs.y); }
_FTYPE inline uint3 operator-(const uint3& lhs, const uint3& rhs) { return make_uint3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z); }
_FTYPE inline uint4 operator-(const uint4& lhs, const uint4& rhs) { return make_uint4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w); }

//// Times
_FTYPE inline float2 operator*(const float2& lhs, const float2& rhs) { return make_float2(lhs.x * rhs.x, lhs.y * rhs.y); }
_FTYPE inline float3 operator*(const float3& lhs, const float3& rhs) { return make_float3(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z); }
_FTYPE inline float4 operator*(const float4& lhs, const float4& rhs) { return make_float4(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w); }

_FTYPE inline int2 operator*(const int2& lhs, const int2& rhs) { return make_int2(lhs.x * rhs.x, lhs.y * rhs.y); }
_FTYPE inline int3 operator*(const int3& lhs, const int3& rhs) { return make_int3(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z); }
_FTYPE inline int4 operator*(const int4& lhs, const int4& rhs) { return make_int4(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w); }

_FTYPE inline uint2 operator*(const uint2& lhs, const uint2& rhs) { return make_uint2(lhs.x * rhs.x, lhs.y * rhs.y); }
_FTYPE inline uint3 operator*(const uint3& lhs, const uint3& rhs) { return make_uint3(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z); }
_FTYPE inline uint4 operator*(const uint4& lhs, const uint4& rhs) { return make_uint4(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w); }

//// Division
_FTYPE inline float2 operator/(const float2& lhs, const float2& rhs) { return make_float2(lhs.x / rhs.x, lhs.y / rhs.y); }
_FTYPE inline float3 operator/(const float3& lhs, const float3& rhs) { return make_float3(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z); }
_FTYPE inline float4 operator/(const float4& lhs, const float4& rhs) { return make_float4(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w); }

_FTYPE inline int2 operator/(const int2& lhs, const int2& rhs) { return make_int2(lhs.x / rhs.x, lhs.y / rhs.y); }
_FTYPE inline int3 operator/(const int3& lhs, const int3& rhs) { return make_int3(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z); }
_FTYPE inline int4 operator/(const int4& lhs, const int4& rhs) { return make_int4(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w); }

_FTYPE inline uint2 operator/(const uint2& lhs, const uint2& rhs) { return make_uint2(lhs.x / rhs.x, lhs.y / rhs.y); }
_FTYPE inline uint3 operator/(const uint3& lhs, const uint3& rhs) { return make_uint3(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z); }
_FTYPE inline uint4 operator/(const uint4& lhs, const uint4& rhs) { return make_uint4(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w); }

//// Length
_FTYPE inline float lengthSqr(float2 v) { return sqrtf(v.x*v.x + v.y*v.y); }
_FTYPE inline float lengthSqr(float3 v) { return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z); }
_FTYPE inline float lengthSqr(float4 v) { return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w); }

_FTYPE inline float length(float2 v) { return sqrtf(lengthSqr(v)); }
_FTYPE inline float length(float3 v) { return sqrtf(lengthSqr(v)); }
_FTYPE inline float length(float4 v) { return sqrtf(lengthSqr(v)); }

*/