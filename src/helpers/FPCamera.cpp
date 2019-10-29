#include "helpers/FPCamera.h"
#include <random>

#include <glm/ext/quaternion_geometric.hpp>

using namespace glm;

struct JitterGenerator 
{
	std::mt19937 rng;
	std::uniform_real_distribution<float> d1, d2;
	JitterGenerator::JitterGenerator() : d1(-0.5f,0.5f), d2(-0.5f,0.5f) { rng.seed(0);}

	vec2 operator()() { return vec2(d1(rng), d2(rng)); }
};

static JitterGenerator jg;

FPCamera::FPCamera()
{
	reset();
	near = 0.1f;
	far = 100.f;
	rot_speed = 1.f;
	mov_speed = 1.f;
	moved = false;
	copied = false;
	jitter_plane = ivec2(0, 0);
	jitter_scale = 0.f;
	jitter       = vec2(0.f, 0.f);
}

void FPCamera::reset()
{
	pos = vec3(0.f, 0.f, 0.f);
	ori = normalize(quat(1.f, 0.0f, 0.0f, 0.0f));
}

vec3 FPCamera::forward() const
{
	return normalize (ori * vec3(0.f, 0.f, -1.f));
}

vec3 FPCamera::right() const
{
	return normalize (ori * vec3(1.f, 0.f, 0.f));
}

vec3 FPCamera::up() const
{
	return normalize (ori * vec3(0.f, 1.f, 0.f));
}

vec3 FPCamera::position() const
{
	return pos;
}

quat FPCamera::orientation() const
{
	return ori;
}

float FPCamera::nearplane() const
{
	return near;
}

float FPCamera::farplane() const
{
	return far;
}

void FPCamera::set_position(vec3 new_position)
{
	pos = new_position;
	copied = true;
}

void FPCamera::set_orientation(quat new_orientation)
{
	ori = normalize(new_orientation);
	copied = true;
}

void FPCamera::set_nearplane(float new_nearplane)
{
	near = new_nearplane;
	copied = true;
}

void FPCamera::set_farplane(float new_farplane)
{
	far = new_farplane;
	copied = true;
}

void FPCamera::set_proj_matrix(mat4 new_proj_matrix, bool ignore_aspect)
{
	verticalFov_deg = degrees(2.f * atan(1.f / new_proj_matrix[1][1]));

	if (!ignore_aspect) aspect = new_proj_matrix[1][1] / new_proj_matrix[0][0];

	copied = true;
}

void FPCamera::set_aspect(float new_aspect)
{
	aspect = new_aspect;
	copied = true;
}


bool FPCamera::is_dirty() const
{
	return moved || copied;
}

mat4 FPCamera::proj_matrix() const
{
	mat4 p = perspective(radians(verticalFov_deg), aspect, near, far);

	if (jitter_scale <= 0.f) return p;

	vec2 j = jitter_scale * jitter / vec2(jitter_plane);

	mat4 o = ortho(-1.f + j.x, 1.f + j.x, -1.f + j.y, 1.f + j.y, 1.f, -1.f);

	return o * p;



	//return ortho(0.f, 1.f, 0.f, 1.f, 0.f, 1.f) * p;
}

mat4 FPCamera::view_matrix()  const
{
	return lookAt(pos, pos + forward(), up());
}

void FPCamera::update(vec2 rot, vec3 motion)
{
	updateOrientation(rot);
	updatePosition(motion);
	updateJitter();
}

void FPCamera::update()
{
	updateJitter();
}

void FPCamera::updateJitter()
{
	jitter = jg();
}

void FPCamera::updateOrientation(vec2 rot)
{
	float pitch = -rot.y * rot_speed;
	float yaw   = -rot.x * rot_speed;

	quat q_yaw   = quat(vec3(0.f  , yaw, 0.f));
	quat q_pitch = quat(vec3(pitch, 0.f, 0.f));
	ori = q_yaw * ori * q_pitch;
}

void FPCamera::updatePosition(vec3 motion)
{
	motion *= mov_speed;
	pos += right() * motion.x + up() * motion.y + forward() * motion.z;
}

