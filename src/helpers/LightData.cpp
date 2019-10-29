#include "common.h"
#include "helpers/LightData.h"

#include <json.inl>

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <functional>

using namespace GLHelpers;

LightDataGL::LightDataGL() :
	lightListBuffer(create_object<BufferObject>())
{
}

int LightData::loadFromJSON(std::string filename)
{

	Json json;
	int r = json.LoadFromFile(filename);
	if (r != SUCCESS) return r;

	return loadFromJSON(json.root());
}

int LightData::loadFromJSON(JsonNodeRef j)
{

	if (j.is_null()) {
		std::cerr << "Error: Empty json node" << std::endl;
		return ERROR_LOADING_RESOURCE;
	}

	clear();

	gl.ambient_light = j.get_vec3("ambient");

	JsonNodeRef light_array = j["lights"];

	if (!light_array.is_null()) {

		if (!light_array.is_array()) {
			std::cerr << "Error: Lights element not present correctly in light json node" << std::endl;
			return ERROR_LOADING_RESOURCE;
		}

		for (size_t i = 0; i < light_array.size(); ++i)
		{
			JsonNodeRef l = light_array[i];
			LightDataStructGL lgl;
			resetLightDataStruct(lgl);

			std::string type = l.get_string("type");

			if (type == "directional")    lgl.type = LightType::LT_DIRECTIONAL;
			else if (type == "point")          lgl.type = LightType::LT_POINT;
			else if (type == "spot")           lgl.type = LightType::LT_SPOT;
			else if (type == "sphere")         lgl.type = LightType::LT_SPHERE;
			else if (type == "parallelogram")  lgl.type = LightType::LT_PARALLELOGRAM;
			else if (type == "polygon")        lgl.type = LightType::LT_POLYGON;
			else { std::cout << "Warning: ignoring unrecognized light type '" << type << "'\n"; continue; }

			lgl.direction = l.get_vec3("direction");
			lgl.position = l.get_vec3("position");
			lgl.color = l.get_vec3("color");
			lgl.radius = l.get_float("radius");
			lgl.intensity = l.get_float("intensity");
			lgl.spot_angle = l.get_float("spotangle");

			gl.lightList.push_back(lgl);
		}
	}

	dirty = false;
	dirtygl = true;

	int resultGL = updateGLResources();
	if (resultGL != SUCCESS) return resultGL;

	return SUCCESS;
}

int LightData::saveToJSON(std::string filename) const
{
	Json json;
	saveToJSON(json.root());
	return (json.SaveToFile(filename));
}

int LightData::saveToJSON(JsonNodeRef j) const
{
	j["ambient"] = gl.ambient_light;
	
	JsonNodeRef light_array = j["lights"];

	for (size_t i = 0; i < gl.lightList.size(); ++i)
	{
		JsonNodeRef l = light_array[i];
		LightDataStructGL lgl = gl.lightList[i];

		switch (lgl.type)
		{
		default:
		case LightType::LT_DIRECTIONAL: 
			l["type"] = "directional"; break;
		case LightType::LT_POINT:
			l["type"] = "point"; break;
		case LightType::LT_SPOT:
			l["type"] = "spot"; break;
		case LightType::LT_SPHERE:
			l["type"] = "sphere"; break;
		case LightType::LT_PARALLELOGRAM:
			l["type"] = "parallelogram"; break;
		case LightType::LT_POLYGON:
			l["type"] = "polygon"; break;
		}

		l["direction"] = lgl.direction;
		l["position"]  = lgl.position;
		l["color"]     = lgl.color;
		l["radius"]    = lgl.radius;
		l["intensity"] = lgl.intensity;
		l["spotangle"] = lgl.spot_angle;

	}

	return SUCCESS;
}


int LightData::clear()
{
	releaseGLResources();
	
	gl.lightList.clear();
	gl.lightList.shrink_to_fit();

	releaseGLResources();

	return SUCCESS;
}

void LightData::setDirty(bool e)
{
	if (e) dirtygl = true;
	dirty = e;
}

bool LightData::isDirty() const
{
	return dirty;
}

std::hash<LightDataGL>::result_type std::hash<LightDataGL>::operator()(std::hash<LightDataGL>::argument_type const& l) const noexcept
{
	result_type h = 0;
	byte* hbyte = (byte*)&h;
	byte * data = (byte *)l.lightList.data();
	size_t data_size = l.lightList.size() * sizeof(LightDataStructGL);
	for (size_t i = 0; i < data_size; ++i) hbyte[i % sizeof(result_type)] ^= data[i];
	return h;
}


int LightData::updateGLResources()
{
	if (!gl.lightListBuffer->id) {
		gl.lightListBuffer->Release();
		gl.lightListBuffer->Create(GL_SHADER_STORAGE_BUFFER);
		hash = 0;
		dirtygl = true;
	}

	if (!dirtygl) return SUCCESS;

	if (gl.lightList.empty())
	{
		std::cerr << "Warning: Creating light resources but no lights specified." << std::endl;
	}

	size_t bufferSize = gl.lightList.size() * sizeof(LightDataStructGL);
	std::hash<LightDataGL>::result_type newhash = std::hash<LightDataGL>{}(gl);

	if (bufferSize != gl.lightListBuffer->bytesize || newhash != hash) {
		gl.lightListBuffer->UploadData(gl.lightList.size() * sizeof(LightDataStructGL), GL_STATIC_READ, gl.lightList.data());
		hash = newhash;
	}

	dirtygl = false;

	return SUCCESS;
}

int LightData::releaseGLResources()
{
	gl.lightListBuffer->Release();

	return SUCCESS;
}

void resetLightDataStruct(LightDataStructGL& l)
{
	l.type = LightType::LT_DIRECTIONAL;
	l.intensity = 1.f;
	l.color = glm::vec3(1.f, 1.f, 1.f);
	l.position = glm::vec3(0.f, 1, 0.f);
	l.direction = glm::vec3(0.f, -1.f, 0.f);
	l.spot_angle = 0.5f;
	l.radius = 0.1f;
	l.v1 = glm::vec3(-1.f, -1.f, 0.f);
	l.v2 = glm::vec3( 1.f, -1.f, 0.f);
	l.v3 = glm::vec3( 1.f,  1.f, 0.f);
	l.v4 = glm::vec3(-1.f,  1.f, 0.f);
}
