#version 430 core

uniform float near;
uniform float far;
uniform int isDirectional;

void main()
{


	if (isDirectional != 0) {
		gl_FragDepth = gl_FragCoord.z / gl_FragCoord.w;
	} else { 
		gl_FragDepth = (2.0 * near) / (far + near - gl_FragCoord.z  * (far - near));
	}



}
