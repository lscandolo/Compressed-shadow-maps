#version 450
#extension GL_ARB_bindless_texture : require

#define LINEARIZE_DEPTH   0
#define ENABLE_ALPHA_TEST 0

#include "../material.glsl"
#include "../reusable/phong.glsl"
#include "../reusable/linear_depth.glsl"


struct fragmentProperties
{
	vec3  ws_pos;
	vec3  ws_dpos;
	vec3  ws_nor;
	float ws_tan;
	vec2  texcoord;
	int   matid;
	vec3  diffuse_color;
	vec3  specular_color;
	vec3  ambient_color;
	float shininess;
} frag;

layout(std430, binding = 0) buffer material_ssbo
{
	Material materials[];
};

uniform bool  light_is_dir = false;
uniform vec3  light_dir;
uniform vec3  light_spotpos;
uniform float light_spotangle;
uniform vec3  light_color;
uniform float light_intensity;

uniform vec3  ws_campos;
uniform float alpha_test_threshold   = 0.5;
uniform vec3  ambient_light = vec3(0.2);
uniform vec2  cam_nearfar;

uniform sampler2D shadowtex;
uniform sampler2D wpostex;
uniform sampler2D wdpostex;
uniform sampler2D wnortex;
uniform sampler2D tcmattex;

layout(location = 0) out vec4 out_color;


void main()
{
	vec4 ws_pos = texelFetch(wpostex, ivec2(gl_FragCoord.xy), 0);
	
	if (ws_pos.w == 0) discard;
	
	frag.ws_pos = ws_pos.xyz;

	frag.ws_dpos = texelFetch(wdpostex, ivec2(gl_FragCoord.xy), 0).xyz;
	vec4 ws_nor  = texelFetch(wnortex, ivec2(gl_FragCoord.xy), 0).xyzw;
	frag.ws_nor = ws_nor.xyz;
	frag.ws_tan = ws_nor.w;

	vec3 tcmattex = texelFetch(tcmattex, ivec2(gl_FragCoord.xy), 0).xyz;
	frag.texcoord = tcmattex.xy;
	frag.matid    = int(tcmattex.z);

	//out_color = vec4(frag.ws_dpos,1); return;

	const Material m = materials[frag.matid];

	// Set fragment value
	const vec3  wsViewDir = normalize(ws_campos - frag.ws_pos);

	frag.shininess = m.shininess;

	if (m.diffuse_texhandle[0] != 0 || m.diffuse_texhandle[1] != 0) {
		vec4 tex_value = texture(sampler2D(m.diffuse_texhandle), frag.texcoord);
	#if ENABLE_ALPHA_TEST
		if (tex_value.a < alpha_test_threshold) discard;
	#endif
		frag.diffuse_color = tex_value.rgb ;
	} else {
		frag.diffuse_color = m.diffuse;
	}	

	if (m.specular_highlight_texhandle[0] != 0 || m.specular_highlight_texhandle[1] != 0) {
		frag.specular_color = texture(sampler2D(m.specular_highlight_texhandle), frag.texcoord).rgb;
	} else {
		frag.specular_color = m.specular;
	}

	if (m.ambient_texhandle[0] != 0 || m.ambient_texhandle[1] != 0) {
		frag.ambient_color = texture(sampler2D(m.ambient_texhandle), frag.texcoord).rgb;
	}
	else {
		frag.ambient_color = m.ambient;
	}

	if (m.bump_texhandle[0] != 0 || m.bump_texhandle[1] != 0) {
		vec3 normalCoeffs = texture(sampler2D(m.bump_texhandle), frag.texcoord).xyz - 0.5;
		// tangent decoding
		vec2 phi_theta = unpackHalf2x16(floatBitsToUint(frag.ws_tan));
		vec3 tangent = vec3(sin(phi_theta.y) * cos(phi_theta.x), cos(phi_theta.y), sin(phi_theta.x)*sin(phi_theta.y));
		
		vec3 bitangent    = normalize(cross(tangent, frag.ws_nor));
		frag.ws_nor       = normalize(-normalCoeffs.x * tangent - normalCoeffs.y * bitangent + 0.5 * normalCoeffs.z * frag.ws_nor );
	} 

	vec3 zen_col = vec3(0.59, 0.79, 1.0);	
	vec3 nad_col = vec3(0.8, 0.59, 0.49);
	vec3 sky_col = mix(nad_col, zen_col, clamp(frag.ws_nor.y*0.5+0.5, 0, 1) );
	vec3 color = frag.diffuse_color * ambient_light * sky_col;

	const vec3  wsLightDir = light_is_dir ? -light_dir : normalize(light_spotpos - frag.ws_pos);

	float shadow_val = texelFetch(shadowtex, ivec2(gl_FragCoord.xy), 0).x;
	if (dot(wsLightDir, frag.ws_nor) > 0)
	{

		vec3 reflected_color = phong_color(-wsLightDir, frag.ws_nor, wsViewDir, frag.diffuse_color, frag.specular_color, m.shininess);

		float shadow_coeff = 0;

		if (!light_is_dir) { 
			const float cosLightAngle = 2.0 * acos(dot(-wsLightDir, normalize(light_dir) ));
			const float angleDiff     = max(0, (light_spotangle - cosLightAngle) / light_spotangle);

			if (angleDiff > 0) {
				reflected_color *=   pow(angleDiff, 0.5);
			} else {
				reflected_color = vec3(0);
			}

		} 
		
		color += 4*reflected_color * light_intensity * light_color * shadow_val;
	} 

#if LINEARIZE_DEPTH
	gl_FragDepth = linearizeDepth(gl_FragCoord.z, cam_nearfar.x, cam_nearfar.y);
#endif

	// Gamma curve
	color = pow(color, vec3(0.9));

	out_color = vec4(color, 1.0);

}