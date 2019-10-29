#pragma once

#include "common.h"
#include "helpers/Camera.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

class FPCamera : public Camera
{
public:
	FPCamera();
	
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

	virtual void update(glm::vec2 rot, glm::vec3 motion);
	virtual void update();
	virtual void updateOrientation(glm::vec2 rot);
	virtual void updatePosition(glm::vec3 motion);
	virtual void updateJitter();

	float verticalFov_deg;
	float aspect;
	float near, far;
	float rot_speed;
	float mov_speed;
	float jitter_scale;
	glm::ivec2 jitter_plane;
	bool  copied;
	bool  moved;

	glm::vec3 pos;
	glm::quat ori;

	glm::vec2 jitter;
};