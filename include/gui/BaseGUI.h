#pragma once

#include "common.h"
#include <GLFW/glfw3.h>

class BaseGUI
{
public:
	virtual void draw(GLFWwindow* window, int order = 0) = 0;
	bool        visible;
	std::string displayname;

	virtual ~BaseGUI();
};

typedef std::shared_ptr<BaseGUI> GUIptr;
