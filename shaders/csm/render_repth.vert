#version 430 core

uniform mat4 mvp;

layout( location = 0 ) in vec3 Position;

out gl_PerVertex
{
    vec4 gl_Position;
};

void main()
{	
	gl_Position = mvp * vec4( Position, 1.0 );
}
