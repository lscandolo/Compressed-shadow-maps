#version 430 core

// in bool gl_FrontFacing;

uniform float near;
uniform float far;
uniform int isDirectional;

uniform   sampler2D minDepthTex;
uniform  isampler2D statusTex;

layout(location = 0) out uint frontFacing;
//layout(location = 1) out float newDepth;

void main()
{
		int status = texelFetch(statusTex, ivec2(gl_FragCoord.xy), 0).r;
		
		if (status == 0) {
			discard;
			return; // needed ?
		}
			
		float depth = 1.0;
		if (isDirectional != 0) {
			depth = gl_FragCoord.z / gl_FragCoord.w;
		} else { 
			depth = (2.0 * near) / (far + near - gl_FragCoord.z * (far - near));
		}

		float minDepth = texelFetch(minDepthTex, ivec2(gl_FragCoord.xy), 0).r;

		if (depth <= minDepth) discard;
			
		frontFacing = gl_FrontFacing ? 1 : 0;
		gl_FragDepth = depth;
}
