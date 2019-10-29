#include "common.h"
#include "helpers/Json.h"
#include <fstream>
#include <sstream>

Json::Json()
{
	json = json.object();
	initialized = true;
}

int Json::LoadFromFile(std::string filename)
{
	std::ifstream filestream(filename, std::ifstream::in);
	if (!filestream.is_open())
	{
		return ERROR_FILE_NOT_FOUND;
	}

	std::stringstream jsonstream;
	jsonstream << filestream.rdbuf();
	filestream.close();

	try {
		json = jsonreader::json::parse(jsonstream.str());
	} catch (jsonreader::detail::parse_error e)
	{
		std::cout << "Error trying to read json file " << filename << ": " << e.what() << std::endl;
		return ERROR_EXTERNAL_LIB;
	}

	initialized = true;

	return SUCCESS;
}

int Json::SaveToFile(std::string filename)
{
	std::ofstream filestream(filename, std::ifstream::out);
	if (!filestream.is_open())
	{
		return ERROR_FILE_NOT_FOUND;
	}

	filestream << json.dump(4);

	filestream.close();

	return SUCCESS;
}

JsonNodeRef Json::root()
{
	return JsonNodeRef(json);
}


JsonNodeRef::JsonNodeRef(jsonreader::json& json) : node(json)
{
}

bool JsonNodeRef::is_null() const
{
	return node.is_null();
}

bool JsonNodeRef::is_array() const
{
	return node.is_array();
}

bool JsonNodeRef::is_string() const
{
	return node.is_string();
}

size_t JsonNodeRef::size() const
{
	return node.size();
}

JsonNodeRef JsonNodeRef::operator[](const std::string& nodename)
{
	return JsonNodeRef(node[nodename]);
}

JsonNodeRef JsonNodeRef::operator[](int i)
{
	return JsonNodeRef(node[i]);
}

JsonNodeRef& JsonNodeRef::operator=(const float& v)
{
	node = v; return *this;
}

JsonNodeRef& JsonNodeRef::operator=(const glm::vec2& v)
{
	node = node.array({ v.x, v.y }); return *this;
}

JsonNodeRef& JsonNodeRef::operator=(const glm::vec3& v)
{
	node = node.array({ v.x, v.y, v.z }); return *this;
}

JsonNodeRef& JsonNodeRef::operator=(const glm::vec4& v)
{
	node = node.array({ v.x, v.y, v.z, v.w }); return *this;
}

JsonNodeRef& JsonNodeRef::operator=(const int& v)
{
	node = v; return *this;
}

JsonNodeRef& JsonNodeRef::operator=(const glm::ivec2& v)
{
	node = node.array({ v.x, v.y }); return *this;
}

JsonNodeRef& JsonNodeRef::operator=(const glm::ivec3& v)
{
	node = node.array({ v.x, v.y, v.z }); return *this;
}

JsonNodeRef& JsonNodeRef::operator=(const glm::ivec4& v)
{
	node = node.array({ v.x, v.y, v.z, v.w }); return *this;
}

JsonNodeRef& JsonNodeRef::operator=(const glm::quat& v)
{
	node = node.array({ v.x, v.y, v.z, v.w }); return *this;
}

JsonNodeRef& JsonNodeRef::operator=(const glm::mat2& v)
{
	node = node.array({ v[0][0], v[0][1],
						v[1][0], v[1][1] }); return *this;
}

JsonNodeRef& JsonNodeRef::operator=(const glm::mat3& v)
{
	node = node.array({ v[0][0], v[0][1], v[0][2],
						v[1][0], v[1][1], v[1][2],
						v[2][0], v[2][1], v[2][2] }); return *this;
}

JsonNodeRef& JsonNodeRef::operator=(const glm::mat4& v)
{
	node = node.array({ v[0][0], v[0][1], v[0][2], v[0][3],
						v[1][0], v[1][1], v[1][2], v[1][3],
						v[2][0], v[2][1], v[2][2], v[2][3],
						v[3][0], v[3][1], v[3][2], v[3][3] }); return *this;
}

JsonNodeRef& JsonNodeRef::set_float_array(const float* data, int num)
{
	node.array(); for (int i = 0; i < num; ++i) node[i] = data[i]; return *this;
}

JsonNodeRef& JsonNodeRef::set_int_array(const int* data, int num)
{
	node.array(); for (int i = 0; i < num; ++i) node[i] = data[i]; return *this;
}

JsonNodeRef& JsonNodeRef::operator=(const std::string& v)
{
	node = v; return *this;
}


float JsonNodeRef::get_float(const std::string& name, const float& fallback) const
{
	if (!node[name].is_number()) return fallback; return node[name];
}

glm::vec2 JsonNodeRef::get_vec2(const std::string& name, const glm::vec2& fallback) const
{
	if (!node[name].is_array() || !node[name][0].is_number()) return fallback; 
	return glm::vec2(node[name][0], node[name][1]);
}

glm::vec3 JsonNodeRef::get_vec3(const std::string& name, const glm::vec3& fallback) const
{
	if (!node[name].is_array() || !node[name][0].is_number()) return fallback;
	return glm::vec3(node[name][0], node[name][1], node[name][2]);
}

glm::vec4 JsonNodeRef::get_vec4(const std::string& name, const glm::vec4& fallback) const
{
	if (!node[name].is_array() || !node[name][0].is_number()) return fallback;
	return glm::vec4(node[name][0], node[name][1], node[name][2], node[name][3]);
}

int JsonNodeRef::get_int(const std::string& name, const int& fallback) const
{
	if (!node[name].is_number()) return fallback; return node[name];
}

glm::ivec2 JsonNodeRef::get_ivec2(const std::string& name, const glm::ivec2& fallback) const
{
	if (!node[name].is_array() || !node[name][0].is_number()) return fallback;
	return glm::ivec2(node[name][0], node[name][1]);
}

glm::ivec3 JsonNodeRef::get_ivec3(const std::string& name, const glm::ivec3& fallback) const
{
	if (!node[name].is_array() || !node[name][0].is_number()) return fallback;
	return glm::ivec3(node[name][0], node[name][1], node[name][2]);
}

glm::ivec4 JsonNodeRef::get_ivec4(const std::string& name, const glm::ivec4& fallback) const
{
	if (!node[name].is_array() || !node[name][0].is_number()) return fallback;
	return glm::ivec4(node[name][0], node[name][1], node[name][2], node[name][3]);
}

std::string JsonNodeRef::get_string(const std::string& name, const std::string& fallback) const
{
	if (!node[name].is_string()) return fallback; return node[name];
}

glm::quat JsonNodeRef::get_quat(const std::string& name, const glm::quat& fallback) const
{
	if (!node[name].is_array() || !node[name][0].is_number()) return fallback;
	return glm::quat(node[name][3], node[name][0], node[name][1], node[name][2]);
}

glm::mat2 JsonNodeRef::get_mat2(const std::string& name, const glm::mat2& fallback) const
{
	if (!node[name].is_array() || !node[name][0].is_number()) return fallback;
	return glm::mat2 (node[name][0], node[name][1],
					  node[name][2], node[name][3]);
}

glm::mat3 JsonNodeRef::get_mat3(const std::string& name, const glm::mat3& fallback) const
{
	if (!node[name].is_array() || !node[name][0].is_number()) return fallback;
	return glm::mat3 (node[name][0], node[name][1], node[name][2],
					  node[name][3], node[name][4], node[name][5], 
			          node[name][6], node[name][7], node[name][8]);
}

glm::mat4 JsonNodeRef::get_mat4(const std::string& name, const glm::mat4& fallback) const
{
	if (!node[name].is_array() || !node[name][0].is_number()) return fallback;
	glm::mat4 r;
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			r[i][j] = node[name][i*4+j];
		}
	}
	return r;
	//return glm::mat4 (node[name][0],  node[name][1],  node[name][2] , node[name][3],
	//				  node[name][4],  node[name][5],  node[name][6] , node[name][7],
	//		          node[name][8],  node[name][9],  node[name][10], node[name][11],
	//				  node[name][12], node[name][13], node[name][14], node[name][15]);
}

