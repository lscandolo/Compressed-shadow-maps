#pragma once

#include "common.h"
#include "helpers/Singleton.h"

#include <glm/glm.hpp>
#include <type_traits>

// POD struct (do not add any methods)
struct GLState
{
	GLboolean enable_blend;
	GLenum    blend_function_src;
	GLenum    blend_function_dst;

	GLboolean enable_depth_test;
	GLenum    depth_test_func;
	GLfloat   depth_range_far;
	GLfloat   depth_range_near;

	GLboolean enable_depth_write;
	
	GLboolean enable_cull_face;
	GLenum    cull_face;

	GLboolean enable_multisample;

	GLenum    polygon_draw_mode_front;
	GLenum    polygon_draw_mode_back;

	GLboolean enable_polygon_offset_fill;
	GLboolean enable_polygon_offset_line;
	GLboolean enable_polygon_offset_point;
	GLfloat   polygon_offset_factor;
	GLfloat   polygon_offset_units;

	GLboolean  enable_scissor_test;

	GLint   scissor_x, scissor_y;
	GLsizei scissor_width, scissor_height;

	GLboolean enable_stencil_test;
	GLenum    stencil_test_front_func;
	GLint     stencil_test_front_ref;
	GLuint    stencil_test_front_mask;
	GLenum    stencil_test_front_sfail_op;
	GLenum    stencil_test_front_dfail_op;
	GLenum    stencil_test_front_pass_op;

	GLenum    stencil_test_back_func;
	GLint     stencil_test_back_ref;
	GLuint    stencil_test_back_mask;
	GLenum    stencil_test_back_sfail_op;
	GLenum    stencil_test_back_dfail_op;
	GLenum    stencil_test_back_pass_op;

	GLint   viewport_x, viewport_y;
	GLsizei viewport_width, viewport_height;

	GLState();
	void reset();

	// Helper methods
	void setViewportBox(GLint x, GLint y, GLsizei width, GLsizei height);
	void setScissorBox(GLint x, GLint y, GLsizei width, GLsizei height);
};

static_assert(std::is_trivially_copyable<GLState>::value, "Compile error: GLState should be trivially copyable");

namespace std {
	template<> struct hash<GLState> {
		typedef GLState argument_type;
		typedef std::size_t result_type;
		result_type operator()(argument_type const& s) const noexcept;
	};
}

class GLStateManagerInstance
{

public:
	const GLState& getState() const;
	void           setState(const GLState& newState, bool force = false);
	void           resetState(bool force = false);

protected:
	GLStateManagerInstance();
	~GLStateManagerInstance();

private:
	GLState state;
	std::hash<GLState>::result_type hash;
	
	//void setDefaultState(GLState& state);

	friend Singleton<GLStateManagerInstance>;
};

typedef Singleton<GLStateManagerInstance> GLStateManager;
