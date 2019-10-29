#version 450

#include "reusable/math_constants.glsl"

in fData
{
	vec2 texcoord;
	vec3 dir;
} In;

layout(location = 0) out vec4 out_color;

uniform sampler2D map;
uniform vec3 wsCamPos;

uniform vec3 zenith_dir;
uniform vec3 zenith_color;
uniform vec3 nadir_color;
uniform vec3 horizon_color;

float stripes(float val, float d, float p = 10)
{	
	float v = mod(val, d)/d;
	v = 2*abs(v - 0.5);
	return 1.0 - pow(v, p);
}

void main()
{
	vec3 dir = normalize(In.dir);

	vec3 color;
	
	float coeff = dot(dir, zenith_dir);

	if (coeff >= 0 || true) color = mix(horizon_color, zenith_color, coeff);
	else                    color = mix(horizon_color, nadir_color,  -coeff);

	out_color = vec4(color, 1.0);
}