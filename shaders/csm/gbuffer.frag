#version 450

in fData
{
	vec3 fragWsPos;
	vec3 fragWsNormal;
	vec3 fragWsTangent;
	vec2 fragTexCoord;
	flat int fragMaterialId;
} In;


layout( location = 0 ) out vec4 wpos;
layout( location = 1 ) out vec3 wdpos;
layout( location = 2 ) out vec4 wnor;
layout( location = 3 ) out vec3 tcmat;

void main()
{
	vec3 deriv = fwidth(In.fragWsPos);
	float maxDeriv = length(deriv);

	wpos = vec4(In.fragWsPos, 0.001*maxDeriv);
	wdpos = deriv; //dFdx(In.fragWsPos));//dFdy(In.fragWsPos));
	wnor = vec4(normalize(In.fragWsNormal), 1.0);
	tcmat = vec3(In.fragTexCoord, In.fragMaterialId);

	// tangent encoding as 16 bit angles
	float phi   = atan(In.fragWsTangent.z, In.fragWsTangent.x);
	float theta = acos(In.fragWsTangent.y);
	wnor.w = uintBitsToFloat(packHalf2x16(vec2(phi,theta)));
}