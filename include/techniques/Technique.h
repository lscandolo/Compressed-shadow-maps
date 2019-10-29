#pragma once

#include "common.h"
#include "helpers/MeshData.h"
#include "helpers/LightData.h"
#include "helpers/Camera.h"
#include "gui/BaseGUI.h"

class Technique
{
public:

	struct SetupData
	{
		size2D     output_size;
		MeshData * mesh;
		LightData* lightdata;
		Camera   * camera;
	};


	virtual int  initial_setup(const SetupData& ts) = 0;
	virtual void output_resize_callback(size2D window_size) = 0;
	virtual void destroy() = 0;

	virtual void frame_prologue() = 0;
	virtual void frame_render() = 0;
	virtual void frame_epilogue() = 0;
	virtual void draw_gui(int order) {};

	virtual void reload_programs() = 0;
	virtual std::string name() = 0;
};