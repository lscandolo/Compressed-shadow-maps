#ifndef PHONG_GLSL
#define PHONG_GLSL

#include "math_constants.glsl"

float phongDiffuseNormalizationFactor()
{
	return 0.3183098861; //  == 1.0 / 3.141592;
}

vec3 diffuseComponentPhong(vec3 lightDir, vec3 normal, vec3 albedo)
{
	float fLambert = ( dot( -lightDir, normal )) ;
	fLambert = clamp(fLambert, 0 , 1);
	vec3 diffuse = vec3(fLambert) * albedo;
	return diffuse * phongDiffuseNormalizationFactor();
}

float specularNormalizationFactorPhong(float shininess)
{
	return (shininess + 1.0) / (2.0 * M_PI);
}

float specularNormalizationFactorBlinnPhong(float shininess)
{
	// Correct formula
	return ((shininess + 2.0) * (shininess + 4.0)) / (8.0 * M_PI * (shininess + pow(2.0, -shininess/2) ));
	
	// Approximation
	//return  (shininess + 3.0) * (8.0 * M_PI);
}

vec3 specularComponentPhong(vec3 lightDir, vec3 normal, vec3 vEye, vec3 specularColor, float shininess)
{
	//vec3 vHalf = normalize(lightDir + vEye);
	//float fTemp = max( 0, dot(normal, vHalf));

	vec3 r = normalize(reflect(lightDir, normal));
	float fTemp = max( 0, dot(r, vEye));

	shininess = max(0.0001, shininess);

	float fSpecularIntensity = pow( fTemp, shininess ) *  max(0, dot( -lightDir, normal ));

	fSpecularIntensity *= specularNormalizationFactorPhong(shininess);
	
	return fSpecularIntensity * specularColor;
}

vec3 specularComponentBlinnPhong(vec3 lightDir, vec3 normal, vec3 vEye, vec3 specularColor, float shininess)
{
	vec3 vHalf = normalize(lightDir + vEye);
	float fTemp = max( 0, dot(normal, vHalf));

	shininess = max(0.0001, shininess);

	float fSpecularIntensity = pow( fTemp, shininess )  *  max(0, dot( -lightDir, normal ));

	fSpecularIntensity *= specularNormalizationFactorBlinnPhong(shininess);
	
	return fSpecularIntensity * specularColor;
}


vec3 phong_color(vec3 lightDir, vec3 normal, vec3 vEye, vec3 albedo, vec3 specularColor, float shininess)
{
	return diffuseComponentPhong(lightDir, normal, albedo) + specularComponentPhong(lightDir, normal, vEye, specularColor, shininess);
}

vec3 blinn_phong_color(vec3 lightDir, vec3 normal, vec3 vEye, vec3 albedo, vec3 specularColor, float shininess)
{
	return diffuseComponentPhong(lightDir, normal, albedo) + specularComponentBlinnPhong(lightDir, normal, vEye, specularColor, shininess);
}

#endif // PHONG_GLSL