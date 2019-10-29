#ifndef MATERIAL_GLSL
#define MATERIAL_GLSL

struct Material {
	uvec2 alpha_texhandle;
	uvec2 ambient_texhandle;
	uvec2 bump_texhandle;
	uvec2 diffuse_texhandle;
	uvec2 displacement_texhandle;
	uvec2 emissive_texhandle;
	uvec2 metallic_texhandle;
	uvec2 normal_texhandle;
	uvec2 reflection_texhandle;
	uvec2 roughness_texhandle;
	uvec2 sheen_texhandle;
	uvec2 specular_highlight_texhandle;

	vec3  ambient;
	vec3  diffuse;
	vec3  specular;
	vec3  transmittance;
	vec3  emission;
	float shininess;

	// pbr
	float roughness;            // [0, 1] default 0
	float metallic;             // [0, 1] default 0
	float sheen;                // [0, 1] default 0
};

#endif // MATERIAL_GLSL

