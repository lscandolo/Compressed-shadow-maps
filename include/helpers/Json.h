#pragma once

#include <json.inl>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

class JsonNodeRef
{
public:

	bool is_null()   const;
	bool is_array()  const;
	bool is_string() const;
	size_t size()    const;

	JsonNodeRef operator[](const std::string& nodename);
	JsonNodeRef operator[](int i);

	////////////// Writing
	JsonNodeRef& operator=(const float&   v);
	JsonNodeRef& operator=(const glm::vec2&   v);
	JsonNodeRef& operator=(const glm::vec3&   v);
	JsonNodeRef& operator=(const glm::vec4&   v);

	JsonNodeRef& operator=(const int&   v);
	JsonNodeRef& operator=(const glm::ivec2&   v);
	JsonNodeRef& operator=(const glm::ivec3&   v);
	JsonNodeRef& operator=(const glm::ivec4&   v);

	JsonNodeRef& operator=(const glm::quat&   v);

	JsonNodeRef& operator=(const glm::mat2&   v);
	JsonNodeRef& operator=(const glm::mat3&   v);
	JsonNodeRef& operator=(const glm::mat4&   v);

	JsonNodeRef& operator=(const std::string& v);

	JsonNodeRef& set_float_array(const float* data, int num);
	JsonNodeRef& set_int_array(const int* data, int num);

	std::string get_string(const std::string& name, const std::string& fallback = "") const;

	////////////// Reading
	float     get_float(const std::string& name, const float& fallback = 0.f) const;
	glm::vec2 get_vec2 (const std::string& name, const glm::vec2& fallback = glm::vec2(0.f)) const;
	glm::vec3 get_vec3 (const std::string& name, const glm::vec3& fallback = glm::vec3(0.f)) const;
	glm::vec4 get_vec4 (const std::string& name, const glm::vec4& fallback = glm::vec4(0.f)) const;

	int        get_int  (const std::string& name, const int& fallback = 0.f) const;
	glm::ivec2 get_ivec2(const std::string& name, const glm::ivec2& fallback = glm::ivec2(0)) const;
	glm::ivec3 get_ivec3(const std::string& name, const glm::ivec3& fallback = glm::ivec3(0)) const;
	glm::ivec4 get_ivec4(const std::string& name, const glm::ivec4& fallback = glm::ivec4(0)) const;

	glm::quat get_quat(const std::string& name, const glm::quat& fallback = glm::quat(1.f, 0.f, 0.f, 0.f)) const;

	glm::mat2 get_mat2(const std::string& name, const glm::mat2& fallback = glm::mat2(0.f)) const;
	glm::mat3 get_mat3(const std::string& name, const glm::mat3& fallback = glm::mat3(0.f)) const;
	glm::mat4 get_mat4(const std::string& name, const glm::mat4& fallback = glm::mat4(0.f)) const;

protected:

	friend class Json;
	JsonNodeRef(jsonreader::json& json);

private:

	jsonreader::json& node;
}; 


class Json
{
public:
	Json();
	int LoadFromFile(std::string filename);
	int SaveToFile(std::string filename);

	JsonNodeRef root();

private:

	bool initialized;
	jsonreader::json json;
};

