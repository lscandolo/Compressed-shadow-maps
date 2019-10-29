#pragma once

#include "common.h"
#include "gui/BaseGUI.h"
#include "helpers/RenderOptions.h"
#include "techniques/Technique.h"

class RenderGUI : public BaseGUI
{
public:
	RenderGUI(RenderOptions* options, RenderStats* stats);
	void setOptions(RenderOptions* options);
	void setStats(RenderStats* stats);
	virtual ~RenderGUI();
	virtual void draw(GLFWwindow* window, int order = 0);

	bool fullscreen;
	RenderOptions* options;
	RenderStats*   stats;

private:
	
};

