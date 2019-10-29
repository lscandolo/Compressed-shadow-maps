#version 450

layout(location = 0) in vec3 Position;
layout(location = 1) in vec3 Normal;
layout(location = 2) in vec2 TexCoord;
layout(location = 3) in vec4 Tangent;
layout(location = 4) in vec3 Color;
layout(location = 5) in int  MaterialId;

out vData
{
	vec3 fragWsPos;
	vec3 fragWsNormal;
	vec2 fragTexCoord;
	flat int fragMaterialId;
} Out;

uniform mat4 mMatrix;
uniform mat4 vMatrix;
uniform mat4 mvpMatrix;
uniform mat3 normalMatrix;

uniform bool view_space = false;

void main()
{
	vec4 ndcPos = mvpMatrix * vec4(Position, 1.0);
	gl_Position = ndcPos;

	vec4 wsPos = mMatrix * vec4(Position, 1.0);

	vec3 normal  = normalize(normalMatrix * Normal.xyz);

	Out.fragWsPos      = wsPos.xyz / wsPos.w;
	Out.fragWsNormal   = normal;
	Out.fragTexCoord   = TexCoord;
	Out.fragMaterialId = MaterialId;

	if (view_space)
	{
		Out.fragWsPos    = (vMatrix * mMatrix * vec4(Position, 1.0)).xyz;
		Out.fragWsNormal = normalize((vMatrix * vec4(Normal, 0.0)).xyz);
	}

}