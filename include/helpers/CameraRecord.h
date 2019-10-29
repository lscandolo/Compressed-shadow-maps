#pragma once

#include "common.h"
#include "helpers/Camera.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

class CameraRecord : public Camera
{
public:

	typedef uint64_t   timestamp_t;
	static timestamp_t timestamp_now();
	static int64_t     diff_ms(timestamp_t& from, timestamp_t& to);
	static int64_t     diff_ns(timestamp_t& from, timestamp_t& to);


	CameraRecord();
	
	virtual void reset();

	virtual glm::vec3 forward() const;
	virtual glm::vec3 right() const;
	virtual glm::vec3 up() const;

	virtual glm::mat4 view_matrix() const;
	virtual glm::mat4 proj_matrix() const;

	virtual glm::vec3 position() const;
	virtual glm::quat orientation() const;

	virtual float nearplane() const;
	virtual float farplane() const;

	virtual void set_position(glm::vec3 new_position);
	virtual void set_orientation(glm::quat new_orientation);

	virtual void set_nearplane(float new_nearplane);
	virtual void set_farplane(float new_farplane); 

	virtual void set_proj_matrix(glm::mat4 new_proj_matrix, bool ignore_aspect = false);
	virtual void set_aspect(float new_aspect);

	virtual bool is_dirty() const;

	void set_dirty(bool dirty=true);
	void add(const Camera& camera, timestamp_t timestamp);

	std::vector<glm::mat4>   vmatrix_list;
	std::vector<glm::mat4>   pmatrix_list;
	std::vector<glm::vec3>   forward_list;
	std::vector<glm::vec3>   right_list;
	std::vector<glm::vec3>   up_list;
	std::vector<float>       nearplane_list;
	std::vector<float>       farplane_list;
	std::vector<glm::vec3>   position_list;
	std::vector<glm::quat>   orientation_list;
	std::vector<timestamp_t> timestamp_list;

	bool dirty;

	int load(std::string filename);
	int save(std::string filename);
	int set_index(int index);
	int advance(unsigned int steps=1);
	int start();
	int stop();
	size_t frames;
	int current_index;
};