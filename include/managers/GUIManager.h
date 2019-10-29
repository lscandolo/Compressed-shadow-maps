#pragma once

#include "common.h"
#include "helpers/Singleton.h"
#include "gui/BaseGUI.h"
#include "techniques/Technique.h"

#include <GLFW/glfw3.h>
#include <map>

class GUIManagerInstance
{

public:

	void      initialize(GLFWwindow* window);
	void      finalize(GLFWwindow* window);
	int       addGUI(std::string name, GUIptr gui);
	int       removeGUI(std::string name);
	void      setTechnique(Technique* t);
	void      draw();
	
	bool      key_callback(int key, int, int action, int mods); // Return true when input was used
	bool      char_callback(unsigned int c); // Return true when input was used

protected:
	GUIManagerInstance()  = default;
	~GUIManagerInstance() = default;

private:
	GLFWwindow* window;
	std::map<std::string, GUIptr> guis;
	friend Singleton<GUIManagerInstance>;
	bool focused;
	Technique* current_technique;
};

typedef Singleton<GUIManagerInstance> GUIManager;
