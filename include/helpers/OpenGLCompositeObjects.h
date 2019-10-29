#pragma once 

#include "common.h"
#include "helpers/OpenGLTextureObject.h"

// Notice:
// Please use this header to define structures used to transport structures with OpenGL resources
// To also add them as ports, please add them at OpenGLFlowgraphPorts.cpp/h
//

namespace GLHelpers	{

	//class LightResources
	//{
	//public:

	//	Mx::GFX::LightSource           lightSource;
	//	glm::mat4x4                    lightTransform; // vp transform (minus direction for point lights)
	//	float                          biasMultiplier; // bias multiplier for shadow map test
	//	TextureObject2D                shadowMap;     // Used for spot and directional lights, for point lights it's a dual paraboloid map with left/right for front/back
	//};

	//class SH3IrradianceVolume
	//{
	//public:
	//
	//	Mx::BoundingBox3f              volumeArea;
	//	glm::ivec3                      size; // Should coincide with texture sizes
	//	
	//	std::vector<Mx::SH3Color>      sh;   // Volume of SH should be size size.x * size.y * size.z 
	//	                                     // organized as index = x + y * size.x + z * size.x * size.y

	//	// 9 RGB textures (need 9 coefficients for each channel) of size size
	//	std::vector<Mx::OpenGL::TextureObject3D> shtex;  
	//};

	class PBROptions
	{
		public:

			enum normalDistributionFunctionClass
			{
				ND_PONG,
				ND_BECKMANN,
				ND_GGX,
			};
			
			enum visibilityFunctionClass
			{
				VF_IMPLICIT,
				VF_NEUMANN,
				VF_COOKTORRANCE,
				VF_KELEMEN,
				VF_ANISOTROPIC_BLINNPHONG,
				VF_ANISOTROPIC_BECKMANN,
				VF_ANISOTROPIC_GGX,
				VF_ANISOTROPIC_SCHLICK_BECKMANN,
				VF_ANISOTROPIC_SCHLICK_GGX,
			};

			enum fresnelFunctionClass
			{
				FR_SIMPLE, 
				FR_SCHLICK,
				FR_COOKTORRANCE,
			};

			bool                                    useAlbedoTextures;
			bool                                    useAlphaTesting;
			bool                                    useNormalTextures;
			bool                                    useSpecTextures;
			bool                                    useDetailTextures;
			bool                                    invertNormalMapX, invertNormalMapY;

			normalDistributionFunctionClass         normalDistributionFunction;
			visibilityFunctionClass                 visibilityFunction;
			fresnelFunctionClass                    fresnelFunction;

			bool                                    cullBackFace;
			bool                                    cullFrontFace;

		PBROptions() 
			: useAlbedoTextures(true)
			, useAlphaTesting(true)
			, useNormalTextures(true)
			, useSpecTextures(true)
			, useDetailTextures(true)
			, invertNormalMapX(true)
			, invertNormalMapY(true)
			, normalDistributionFunction(ND_GGX)
			, visibilityFunction(VF_ANISOTROPIC_GGX)
			, fresnelFunction(FR_SCHLICK)
			, cullBackFace(false)
			, cullFrontFace(false)
		{}
	};

}