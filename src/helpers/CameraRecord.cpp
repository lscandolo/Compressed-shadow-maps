#include "helpers/CameraRecord.h"

#include <json.inl>
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>

using namespace glm;

CameraRecord::timestamp_t CameraRecord::timestamp_now()
{
	std::chrono::high_resolution_clock c;
	uint64_t t = c.now().time_since_epoch().count();
	return t;
}

int64_t CameraRecord::diff_ms(timestamp_t& from, timestamp_t& to)
{
	std::chrono::system_clock::time_point timepoint_from{ std::chrono::duration_cast<std::chrono::system_clock::time_point::duration>(std::chrono::nanoseconds(from)) };
	std::chrono::system_clock::time_point timepoint_to{ std::chrono::duration_cast<std::chrono::system_clock::time_point::duration>(std::chrono::nanoseconds(to)) };

	return std::chrono::duration_cast<std::chrono::milliseconds>(timepoint_to - timepoint_from).count();
}

int64_t CameraRecord::diff_ns(timestamp_t& from, timestamp_t& to)
{
	std::chrono::system_clock::time_point timepoint_from{ std::chrono::duration_cast<std::chrono::system_clock::time_point::duration>(std::chrono::nanoseconds(from)) };
	std::chrono::system_clock::time_point timepoint_to{ std::chrono::duration_cast<std::chrono::system_clock::time_point::duration>(std::chrono::nanoseconds(to)) };

	return std::chrono::duration_cast<std::chrono::nanoseconds>(timepoint_to - timepoint_from).count();
}

CameraRecord::CameraRecord()
{
	current_index = -1;
	frames = 0;
	dirty = true;
}

void CameraRecord::reset()
{
	*this = CameraRecord();
}

template <typename T>
static T search_fallback(const std::vector<T>& list, int index, const T& fallback)
{
	if (index < 0 || index > list.size()) return fallback;
	return list[index]; 
}

vec3 CameraRecord::forward() const
{
	return search_fallback(forward_list, current_index, vec3(0,0,-1));
}

vec3 CameraRecord::right() const
{
	return search_fallback(right_list, current_index, vec3(1,0,0));
}

vec3 CameraRecord::up() const
{
	return search_fallback(up_list, current_index, vec3(0,1,0));
}


mat4 CameraRecord::view_matrix() const
{
	return search_fallback(vmatrix_list, current_index, identity<mat4>());
}

mat4 CameraRecord::proj_matrix() const
{
	return search_fallback(pmatrix_list, current_index, identity<mat4>());
}


vec3 CameraRecord::position() const
{
	return search_fallback(position_list, current_index, vec3(0.f));
}

quat CameraRecord::orientation() const
{
	return search_fallback(orientation_list, current_index, identity<quat>());
}


float CameraRecord::nearplane() const
{
	return search_fallback(nearplane_list, current_index, 0.f);
}

float CameraRecord::farplane() const
{
	return search_fallback(farplane_list, current_index, 1.f);
}


void CameraRecord::set_position(vec3 new_position)
{
}

void CameraRecord::set_orientation(quat new_orientation)
{
}


void CameraRecord::set_nearplane(float new_nearplane)
{
}

void CameraRecord::set_farplane(float new_farplane)
{
}

void CameraRecord::set_proj_matrix(mat4 new_farplane, bool ignore_aspect)
{
}

void CameraRecord::set_aspect(float new_aspect)
{
}

bool CameraRecord::is_dirty() const
{
	return dirty;
	//return current_index >= 0;
}

void CameraRecord::set_dirty(bool dirty)
{
	this->dirty = dirty;
}

void CameraRecord::add(const Camera& camera, timestamp_t timestamp)
{
	forward_list.push_back(camera.forward());
	up_list.push_back(camera.up());
	right_list.push_back(camera.right());
	vmatrix_list.push_back(camera.view_matrix());
	pmatrix_list.push_back(camera.proj_matrix());
	nearplane_list.push_back(camera.nearplane());
	farplane_list.push_back(camera.farplane());
	position_list.push_back(camera.position());
	orientation_list.push_back(camera.orientation());
	timestamp_list.push_back(timestamp);

	frames++;
}

static float read_float(jsonreader::json& jval)
{
	return jval;
}

static uint64_t read_uint64(jsonreader::json& jval)
{
	return jval;
}

static vec3 read_vec3(jsonreader::json& jval)
{
	return vec3(jval[0], jval[1], jval[2]);
}

static quat read_quat(jsonreader::json& jval)
{
	return quat(jval[3], jval[0], jval[1], jval[2]);
}

static mat4 read_mat4(jsonreader::json& jval)
{
	mat4 m;
	for (int i = 0; i < 16; ++i) {
		m[i/4][i%4] = jval[i];
	}
	return m;
}

static void write_float(std::ofstream& of, const char* name, float val, bool last = false)
{
	of << "   ";
	of << "\"" << name << "\": " << val << (last ? "\n" : ",\n");
}

static void write_uint64(std::ofstream& of, const char* name, uint64_t val, bool last = false)
{
	of << "   ";
	of << "\"" << name << "\": " << val << (last ? "\n" : ",\n");
}

static void write_vec3(std::ofstream& of, const char* name, const vec3& val, bool last = false)
{
	of << "   ";
	of << "\"" << name << "\": [" << val.x << "," << val.y << "," << val.z << "]" << (last ? "\n" : ",\n");
}

static void write_quat(std::ofstream& of, const char* name, const quat& val, bool last = false)
{
	of << "   ";
	of << "\"" << name << "\": [" << val.x << "," << val.y << "," << val.z << "," << val.w << "]" << (last ? "\n" : ",\n");
}

static void write_mat4(std::ofstream& of, const char* name, const mat4& val, bool last = false)
{
	of << "   ";
	of << "\"" << name << "\": [";
	
	for (int i = 0; i < 16; ++i) {
		of << val[i/4][i%4] << (i < 15 ? "," : "");
	}
	of << "]" << (last ? "\n" : ",\n");
}

int CameraRecord::load(std::string filename)
{
	std::ifstream filestream(filename, std::ifstream::in);
	if (!filestream.is_open())
	{
		return ERROR_FILE_NOT_FOUND;
	}

	std::stringstream jsonstream;
	jsonstream << filestream.rdbuf();
	filestream.close();

	jsonreader::json j = jsonreader::json::parse(jsonstream.str());

	if (j.is_null()) {
		std::cerr << "Error: Empty camera record file" << filename << std::endl;
		return ERROR_LOADING_RESOURCE;
	}

	jsonreader::json& j_frames = j["frames"];
	jsonreader::json& j_framedata = j["framedata"];

	frames = j_framedata.size();

	nearplane_list.resize(frames);
	farplane_list.resize(frames);
	forward_list.resize(frames);
	right_list.resize(frames);
	up_list.resize(frames);
	position_list.resize(frames);
	orientation_list.resize(frames);
	vmatrix_list.resize(frames);
	pmatrix_list.resize(frames);
	timestamp_list.resize(frames);

	for (int i = 0; i < frames; ++i) {
		jsonreader::json& j_frame = j_framedata[i];
		nearplane_list[i]   = read_float(j_frame["nearplane"]);
		farplane_list[i]    = read_float(j_frame["farplane"]);
		forward_list[i]     = read_vec3(j_frame["forward"]);
		right_list[i]       = read_vec3(j_frame["right"]);
		up_list[i]          = read_vec3(j_frame["up"]);
		position_list[i]    = read_vec3(j_frame["position"]);
		orientation_list[i] = read_quat(j_frame["orientation"]);
		vmatrix_list[i]     = read_mat4(j_frame["vmatrix"]);
		pmatrix_list[i]     = read_mat4(j_frame["pmatrix"]);
		timestamp_list[i]   = read_uint64(j_frame["timestamp"]);
	}

	current_index = -1;

	return SUCCESS;
}

int CameraRecord::save(std::string filename)
{
	std::ofstream filestream(filename, std::ofstream::out);
	if (!filestream.is_open())
	{
		return ERROR_FILE_NOT_FOUND;
	}

	filestream << "{\n";
	filestream << "\"frames\": " << frames << ",\n";

	
	float fov = 0.f;
	float aspect = 0.f;
	if (frames > 0) {
		mat4 p = pmatrix_list[0];
		fov = 2.f * atan(1.f/ p[1][1]);
		aspect = p[1][1] / p[0][0];
	}
	filestream << "\"fov_deg\": " << fov * 180.f / 3.141592f << ",\n";
	filestream << "\"fov_rad\": " << fov << ",\n";
	filestream << "\"aspect\": "  << aspect << ",\n";

	{
		filestream << "\"framedata\": \n";
		filestream << " [" << "\n";
		for (int i = 0; i < frames; i++)
		{
			filestream << "  {" << "\n";
			write_uint64(filestream, "timestamp",   timestamp_list[i]);
			write_float(filestream,  "nearplane",   nearplane_list[i]);
			write_float(filestream,  "farplane",    farplane_list [i]);
			write_vec3(filestream,   "forward",     forward_list[i]);
			write_vec3(filestream,   "right",       right_list[i]);
			write_vec3(filestream,   "up",          up_list[i]);
			write_vec3(filestream,   "position",    position_list[i]);
			write_quat(filestream,   "orientation", orientation_list[i]);
			write_mat4(filestream,   "vmatrix",     vmatrix_list[i]);
			write_mat4(filestream,   "pmatrix",     pmatrix_list[i], true);
			filestream << "  }" << (i < frames -1 ? ",\n" : "\n");
		}
		filestream << " ]\n";
	}

	filestream << "}\n";
	filestream.close();

	return SUCCESS;
}


int CameraRecord::set_index(int index)
{
	if (index >= 0 && index < frames && index != current_index) {
		current_index = index;
		set_dirty();
	}

	return SUCCESS;
}

int CameraRecord::advance(unsigned int steps)
{
	steps = std::max(steps, unsigned int (1));
	if (current_index >= 0 && current_index < frames-1) {
		current_index += steps;
		set_dirty();
	}
	if (current_index >= frames) current_index = -1;

	return SUCCESS;
}

int CameraRecord::start()
{
	if (frames > 0 && current_index != 0) {
		current_index = 0;
		set_dirty();
	}

	return SUCCESS;
}

int CameraRecord::stop()
{
	if (current_index != -1) {
		current_index = -1;
		set_dirty();
	}

	return SUCCESS;
}

