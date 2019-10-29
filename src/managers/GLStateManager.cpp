#include "managers/GLStateManager.h"



GLState::GLState()
{
	reset();
}


void GLState::reset()
{

	std::memset(this, 0, sizeof(GLState));

	GLState& state = *this;
	
	state.enable_blend       = GL_FALSE;
	state.blend_function_src = GL_ONE;
	state.blend_function_dst = GL_ZERO;

	state.enable_depth_test  = GL_TRUE;
	state.enable_depth_write = GL_TRUE;
	state.depth_test_func    = GL_LESS;
	state.depth_range_far    = 1.f;
	state.depth_range_near   = 0.f;

	state.enable_cull_face = GL_FALSE;
	state.cull_face        = GL_BACK;

	state.enable_multisample = GL_TRUE;

	state.polygon_draw_mode_front = GL_FILL;
	state.polygon_draw_mode_back  = GL_FILL;

	state.enable_polygon_offset_fill  = GL_FALSE;
	state.enable_polygon_offset_line  = GL_FALSE;
	state.enable_polygon_offset_point = GL_FALSE;
	state.polygon_offset_factor = 0.f;
	state.polygon_offset_units  = 0.f;

	state.enable_scissor_test = GL_FALSE;
	state.scissor_x = state.scissor_y = 0;
	state.scissor_width = state.scissor_height = INT_MAX, INT_MAX;

	state.enable_stencil_test         = GL_FALSE;
	state.stencil_test_front_func     = GL_ALWAYS;
	state.stencil_test_front_ref      = 0;
	state.stencil_test_front_mask     = 0xFFFFFFFF;
	state.stencil_test_front_sfail_op = GL_KEEP;
	state.stencil_test_front_dfail_op = GL_KEEP;
	state.stencil_test_front_pass_op  = GL_KEEP;

	state.stencil_test_back_func     = GL_ALWAYS;
	state.stencil_test_back_ref      = 0;
	state.stencil_test_back_mask     = 0xFFFFFFFF;
	state.stencil_test_back_sfail_op = GL_KEEP;
	state.stencil_test_back_dfail_op = GL_KEEP;
	state.stencil_test_back_pass_op  = GL_KEEP;

	state.viewport_x = state.viewport_y = 0;
	state.viewport_width = state.viewport_height = INT_MAX, INT_MAX;

}

void GLState::setViewportBox(GLint x, GLint y, GLsizei width, GLsizei height)
{
	viewport_x = x; viewport_y = y; viewport_width = width; viewport_height = height;
}

void GLState::setScissorBox(GLint x, GLint y, GLsizei width, GLsizei height)
{
	scissor_x = x; scissor_y = y; scissor_width = width; scissor_height = height;
}


GLStateManagerInstance::GLStateManagerInstance()
{}

GLStateManagerInstance::~GLStateManagerInstance()
{}

const GLState& GLStateManagerInstance::getState() const
{
	return state;
}

void GLStateManagerInstance::resetState(bool force)
{
	GLState resetState;
	resetState.reset();

	setState(resetState, force);
}

std::hash<GLState>::result_type std::hash<GLState>::operator()(std::hash<GLState>::argument_type const& s) const noexcept
{
	result_type h = 0;
	byte* hbyte = (byte*)&h;
	byte * data = (byte *)&s;
	size_t data_size = sizeof(GLState);
	for (size_t i = 0; i < data_size; ++i) hbyte[i % sizeof(result_type)] ^= data[i];
	return h;
}


void GLStateManagerInstance::setState(const GLState& newstate, bool force)
{
	std::hash<GLState>::result_type newhash = std::hash<GLState>{}(newstate);

	if (!force && newhash == hash ) return;

	if (force || newstate.enable_blend != state.enable_blend)
	{
		if (newstate.enable_blend == GL_TRUE) glEnable (GL_BLEND);
		else                                  glDisable(GL_BLEND);
		check_opengl();
	}

	if (force || newstate.blend_function_src != state.blend_function_src || newstate.blend_function_dst != state.blend_function_dst)
	{
		glBlendFunc(newstate.blend_function_src, newstate.blend_function_dst);
		check_opengl();
	}

	if (force || newstate.enable_depth_test != state.enable_depth_test)
	{
		if (newstate.enable_depth_test == GL_TRUE) glEnable (GL_DEPTH_TEST);
		else                                       glDisable(GL_DEPTH_TEST);
		check_opengl();
	}

	if (force || newstate.enable_depth_write != state.enable_depth_write)
	{
		glDepthMask(newstate.enable_depth_write);
		check_opengl();
	}

	if (force || newstate.depth_test_func != state.depth_test_func)
	{
		glDepthFunc(newstate.depth_test_func);
		check_opengl();
	}

	if (force || newstate.depth_range_far != state.depth_range_far || newstate.depth_range_near != state.depth_range_near)
	{
		glDepthRangef(newstate.depth_range_near, newstate.depth_range_far);
		check_opengl();
		check_opengl();
	}

	if (force || newstate.enable_cull_face != state.enable_cull_face)
	{
		if (newstate.enable_cull_face == GL_TRUE) glEnable (GL_CULL_FACE);
		else                                      glDisable(GL_CULL_FACE);
		check_opengl();
	}

	if (force || newstate.cull_face != state.cull_face)
	{
		glCullFace(newstate.cull_face);
		check_opengl();
	}

	if (force || newstate.enable_multisample != state.enable_multisample)
	{
		if (newstate.enable_multisample == GL_TRUE) glEnable (GL_MULTISAMPLE);
		else                                        glDisable(GL_MULTISAMPLE);
		check_opengl();
	}

	if (force || newstate.polygon_draw_mode_front != state.polygon_draw_mode_front)
	{
		glPolygonMode(GL_FRONT, newstate.polygon_draw_mode_front);
		check_opengl();
	}

	if (force || newstate.polygon_draw_mode_back != state.polygon_draw_mode_back)
	{
		glPolygonMode(GL_BACK, newstate.polygon_draw_mode_back);
		check_opengl();
	}

	if (force || newstate.enable_polygon_offset_fill != state.enable_polygon_offset_fill)
	{
		if (newstate.enable_polygon_offset_fill == GL_TRUE) glEnable (GL_POLYGON_OFFSET_FILL);
		else                                                glDisable(GL_POLYGON_OFFSET_FILL);
		check_opengl();
	}

	if (force || newstate.enable_polygon_offset_line != state.enable_polygon_offset_line)
	{
		if (newstate.enable_polygon_offset_line == GL_TRUE) glEnable (GL_POLYGON_OFFSET_LINE);
		else                                                glDisable(GL_POLYGON_OFFSET_LINE);
		check_opengl();
	}

	if (force || newstate.enable_polygon_offset_point != state.enable_polygon_offset_point)
	{
		if (newstate.enable_polygon_offset_point == GL_TRUE) glEnable (GL_POLYGON_OFFSET_POINT);
		else                                                 glDisable(GL_POLYGON_OFFSET_POINT);
		check_opengl();
	}

	if (force || newstate.polygon_offset_factor != state.polygon_offset_factor || newstate.polygon_offset_units != state.polygon_offset_units)
	{
		glPolygonOffset(newstate.polygon_offset_factor, newstate.polygon_offset_units);
		check_opengl();
	}

	if (force || newstate.enable_scissor_test != state.enable_scissor_test)
	{
		if (newstate.enable_scissor_test == GL_TRUE) glEnable(GL_SCISSOR_TEST);
		else                                         glDisable(GL_SCISSOR_TEST);
		check_opengl();
	}

	if (force || newstate.scissor_x != state.scissor_x || newstate.scissor_y != state.scissor_y || newstate.scissor_width != state.scissor_width || newstate.scissor_height != state.scissor_height)
	{
		glScissor(newstate.scissor_x, newstate.scissor_y, newstate.scissor_width, newstate.scissor_height);
		check_opengl();
	}

	if (force || newstate.enable_stencil_test != state.enable_stencil_test)
	{
		if (newstate.enable_stencil_test == GL_TRUE) glEnable(GL_STENCIL_TEST);
		else                                         glDisable(GL_STENCIL_TEST);
		check_opengl();
	}

	if (force || newstate.stencil_test_front_func != state.stencil_test_front_func || newstate.stencil_test_front_ref != state.stencil_test_front_ref || newstate.stencil_test_front_mask != state.stencil_test_front_mask)
	{
		glStencilFuncSeparate(GL_FRONT, newstate.stencil_test_front_func, newstate.stencil_test_front_ref, newstate.stencil_test_front_mask);
		check_opengl();
	}

	if (force || newstate.stencil_test_back_func != state.stencil_test_back_func || newstate.stencil_test_back_ref != state.stencil_test_back_ref || newstate.stencil_test_back_mask != state.stencil_test_back_mask)
	{
		glStencilFuncSeparate(GL_BACK, newstate.stencil_test_back_func, newstate.stencil_test_back_ref, newstate.stencil_test_back_mask);
		check_opengl();
	}

	if (force || newstate.stencil_test_front_sfail_op != state.stencil_test_front_sfail_op || newstate.stencil_test_front_dfail_op != state.stencil_test_front_dfail_op || newstate.stencil_test_front_pass_op != state.stencil_test_front_pass_op)
	{
		glStencilOpSeparate(GL_FRONT, newstate.stencil_test_front_sfail_op, newstate.stencil_test_front_dfail_op, newstate.stencil_test_front_pass_op);
		check_opengl();
	}

	if (force || newstate.stencil_test_back_sfail_op != state.stencil_test_back_sfail_op || newstate.stencil_test_back_dfail_op != state.stencil_test_back_dfail_op || newstate.stencil_test_back_pass_op != state.stencil_test_back_pass_op)
	{
		glStencilOpSeparate(GL_BACK, newstate.stencil_test_back_sfail_op, newstate.stencil_test_back_dfail_op, newstate.stencil_test_back_pass_op);
		check_opengl();
	}

	if (force || newstate.viewport_x != state.viewport_x || newstate.viewport_y != state.viewport_y || newstate.viewport_width != state.viewport_width || newstate.viewport_height != state.viewport_height)
	{
		glViewport(newstate.viewport_x, newstate.viewport_y, newstate.viewport_width, newstate.viewport_height);
		check_opengl();
	}

	state = newstate;
	hash = newhash;
}
