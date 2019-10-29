#pragma once

#include "common.h"
#include <gui/BaseGUI.h>
#include <helpers/LightData.h>

class LightGUI : public BaseGUI
{
public:
	LightGUI(LightData* lights = nullptr);
	void setLights(LightData* lights);
	virtual ~LightGUI();
	virtual void draw(GLFWwindow* window, int order = 0);

	bool dirty;
	
private:
	bool createLightFrame(LightDataStructGL& l);
	LightData* lightsptr;
};

