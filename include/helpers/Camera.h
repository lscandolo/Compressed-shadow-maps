#pragma once

#include "common.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

class Camera
{
public:
	
	virtual glm::vec3 position() const = 0;
	virtual glm::quat orientation() const = 0;

	virtual float nearplane() const = 0;
	virtual float farplane() const = 0;

	virtual void set_position(glm::vec3) = 0;
	virtual void set_orientation(glm::quat) = 0;

	virtual void set_nearplane(float) = 0;
	virtual void set_farplane(float) = 0;


	virtual void set_proj_matrix(glm::mat4, bool ignore_aspect=false) = 0;
	virtual void set_aspect(float) = 0;

	virtual void reset() = 0;

	virtual glm::vec3 forward() const = 0;
	virtual glm::vec3 right() const = 0;
	virtual glm::vec3 up() const = 0;

	virtual glm::mat4 view_matrix() const = 0;
	virtual glm::mat4 proj_matrix() const = 0;

	virtual bool is_dirty() const = 0;

	virtual void copy(const Camera& c, bool ignore_aspect = false)
	{
		set_position(c.position());
		set_orientation(c.orientation());
		set_nearplane(c.nearplane());
		set_farplane(c.farplane());
		set_proj_matrix(c.proj_matrix(), ignore_aspect);
	}
};