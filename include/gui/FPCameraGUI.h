#pragma once

#include "common.h"
#include "gui/BaseGUI.h"
#include "helpers/FPCamera.h"

class FPCameraGUI : public BaseGUI
{
public:
	FPCameraGUI(FPCamera* cam = nullptr);
	void setCamera(FPCamera* cam);
	virtual ~FPCameraGUI();
	virtual void draw(GLFWwindow* window, int order = 0);
private:
	FPCamera* camptr;
};

