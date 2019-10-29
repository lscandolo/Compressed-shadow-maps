#version 450

layout(points) in;
layout (triangle_strip, max_vertices=4) out;


uniform uint width, height;

void main()
{

  gl_Position  = vec4(-1.0, -1.0,  0.0, 1.0);
  EmitVertex();

  gl_Position  = vec4( 1.0, -1.0,  0.0, 1.0);
  EmitVertex();

  gl_Position  = vec4(-1.0,  1.0,  0.0, 1.0);
  EmitVertex();

  gl_Position  = vec4(1.0,  1.0,  0.0, 1.0);
  EmitVertex();

  EndPrimitive();
}

