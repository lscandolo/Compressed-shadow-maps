#version 450

out fData
{
	vec2 texcoord;
	vec3 dir;
} Out;

uniform mat4 vpMatrixInv;
uniform vec3 wsCamPos;

in int gl_VertexID;

void main()
{

	vec4 ndcVertPos = vec4(1.f);
	switch(gl_VertexID)
	{
		case 0: ndcVertPos.xy = vec2(-1.f,  1.f); break;
		case 1: ndcVertPos.xy = vec2(-1.f, -1.f); break;
		case 2: ndcVertPos.xy = vec2( 1.f, -1.f); break;
		case 3: ndcVertPos.xy = vec2( 1.f,  1.f); break;
	}

	gl_Position = ndcVertPos;
	
	Out.texcoord = ndcVertPos.xy * 0.5 + 0.5;
	
	vec4 wsVertPos = vpMatrixInv * ndcVertPos;
	wsVertPos.xyz /= wsVertPos.w;

	Out.dir = normalize(wsVertPos.xyz - wsCamPos.xyz);
}

