#include "helpers/Ini.h"

#include <glm/gtc/type_ptr.hpp>

#include <helpers/Json.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>

using namespace glm;

Ini::Ini()
{}

void write_floats(std::ofstream& s, const std::string& ind, const std::string& name, const float* data, int n)
{
	s << ind << "\"" << name << "\": [";
	for (int i = 0; i < n-1; ++i)
	{
		s << data[i] << ", ";
	}
	s << data[n - 1] << "]\n";
}

int Ini::save(std::string fullpath, IniInfo info)
{

	Json j;

	JsonNodeRef root = j.root();

	if (info.technique_name) {
		JsonNodeRef tecnode = root["technique"];
		tecnode["name"] = *info.technique_name;
	}

	if (info.camera)
	{
		const Camera& camera = *info.camera;
		JsonNodeRef camnode = root["camdata"];
		camnode["nearplane"]   = camera.nearplane();
		camnode["farplane"]    = camera.farplane();
		camnode["position"]    = camera.position();
		camnode["orientation"] = camera.orientation();
		camnode["forward"]     = camera.forward();
		camnode["right"]       = camera.right();
		camnode["up"]          = camera.up();
		camnode["vmatrix"]     = camera.view_matrix();
		camnode["pmatrix"]     = camera.proj_matrix();
	}

	if (info.mesh)
	{
		const MeshData& mesh = *info.mesh;
		JsonNodeRef meshnode = root["meshdata"];
		meshnode["path"] = mesh.source_filename;
	}

	if (info.lights)
	{
		const LightData& lights = *info.lights;
		JsonNodeRef lightsnode = root["lightdata"];
		lights.saveToJSON(lightsnode);
	}

	j.SaveToFile(fullpath);

	return SUCCESS;
}

int Ini::load(std::string fullpath, IniInfo info)
{
	Json j;
	int r = j.LoadFromFile(fullpath);

	if (r != SUCCESS) return r;

	JsonNodeRef root = j.root();

	if (info.technique_name) {
		JsonNodeRef tecnode = root["technique"];
		*info.technique_name = tecnode.get_string("name");
	}

	if (info.camera)
	{
		Camera& camera = *info.camera;
		JsonNodeRef camnode = root["camdata"];
		camera.set_nearplane(camnode.get_float("nearplane"));
		camera.set_farplane(camnode.get_float("farplane"));
		camera.set_position(camnode.get_vec3("position"));
		camera.set_orientation(camnode.get_quat("orientation"));
		camera.set_proj_matrix(camnode.get_mat4("pmatrix"));
	}

	if (info.mesh)
	{
		MeshData& mesh = *info.mesh;
		JsonNodeRef meshnode = root["meshdata"];
		std::string meshfilepath = meshnode.get_string("path");

		if (mesh.load_obj(meshfilepath, 1.f, true) != SUCCESS) {
			std::cerr << "Error loading model file " << meshfilepath << std::endl;
		}
	}

	if (info.lights)
	{
		LightData& lights = *info.lights;
		JsonNodeRef lightsnode = root["lightdata"];
		lights.loadFromJSON(lightsnode);
	}

	return SUCCESS;
}

IniInfo::IniInfo() : mesh(nullptr), camera(nullptr), lights(nullptr) {}
IniInfo::IniInfo(MeshData* meshptr, Camera* cameraptr, LightData* lightsptr, std::string* tecname_ptr) : mesh(meshptr), camera(cameraptr), lights(lightsptr), technique_name(tecname_ptr) {}
