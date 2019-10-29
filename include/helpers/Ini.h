#pragma once

#include "common.h"
#include "helpers/Camera.h"
#include "helpers/MeshData.h"
#include "helpers/LightData.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

struct IniInfo
{
	MeshData* mesh;
	Camera* camera;
	LightData* lights;
	std::string* technique_name;
	IniInfo();
	IniInfo(MeshData* meshptr, Camera* cameraptr, LightData* lightsptr, std::string* tecname_ptr);
};

class Ini
{
public:

	Ini();
	
	int save(std::string fullpath, IniInfo info);
	int load(std::string fullpath, IniInfo info);

};