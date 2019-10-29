#include "managers/GUIManager.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

void GUIManagerInstance::initialize(GLFWwindow* window)
{
	// Create a window called "My First Tool", with a menu bar.
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls
	io.IniFilename = nullptr;
	//io.WantCaptureKeyboard = true;

	ImGui_ImplGlfw_InitForOpenGL(window, false);
	const char* glsl_version = "#version 450";
	ImGui_ImplOpenGL3_Init(glsl_version);
	ImGui::StyleColorsDark();

	this->window = window;
	focused = false;
	current_technique = nullptr;
}

void GUIManagerInstance::finalize(GLFWwindow* window)
{
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	current_technique = nullptr;
}

int GUIManagerInstance::addGUI(std::string name, GUIptr gui)
{
	if (name.empty() || guis.count(name) != 0) {
		return ERROR_INCORRECT_NAME;
	}

	if (gui == nullptr) {
		return ERROR_INVALID_POINTER;
	}

	guis[name] = gui;

	return SUCCESS;
}

int GUIManagerInstance::removeGUI(std::string name)
{
	if (name.empty() || guis.count(name) != 0) {
		return ERROR_INCORRECT_NAME;
	}

	guis.erase(name);

	return SUCCESS;
}

void GUIManagerInstance::setTechnique(Technique* t)
{
	current_technique = t;
}


void GUIManagerInstance::draw()
{
	bool anyvisible = false;
	for (auto& it : guis) anyvisible = anyvisible || it.second->visible;
	if (!anyvisible) return;

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	
#if 0
		ImGui::ShowTestWindow();
#endif
		
	ImGui::SetNextWindowPos(ImVec2(10.f, 10.f), ImGuiCond_Once);
	ImGui::SetNextWindowSize(ImVec2(500.f, 500.f), ImGuiCond_Once);
	ImGui::SetNextWindowCollapsed(true, ImGuiCond_Once);
	ImGui::Begin("Options", nullptr, ImGuiWindowFlags_NoSavedSettings);

	int order = 0;
	for (auto& it : guis) 
	{
		GUIptr& gui = it.second;
		if (gui->visible) {
			if (ImGui::TreeNode(gui->displayname.c_str())) {
				gui->draw(window, order);
				ImGui::TreePop();
			}
		}
		order++;
	}
	if (current_technique) {
		if (ImGui::TreeNode(current_technique->name().c_str())) {
			current_technique->draw_gui(order);
			ImGui::TreePop();
		}
	}

	focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_AnyWindow);

	ImGui::Text("\n\nUse AWSD for movement.\n\nPress F to toggle mouse camera motion.\n\n", "");

	ImGui::End();
	ImGui::EndFrame();
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}


bool GUIManagerInstance::key_callback(int key, int, int action, int mods)
{
	if (focused) {
		ImGuiIO& io = ImGui::GetIO();
		if (action == GLFW_PRESS)
			io.KeysDown[key] = true;
		if (action == GLFW_RELEASE)
			io.KeysDown[key] = false;

		(void)mods; // Modifiers are not reliable across systems
		io.KeyCtrl = io.KeysDown[GLFW_KEY_LEFT_CONTROL] || io.KeysDown[GLFW_KEY_RIGHT_CONTROL];
		io.KeyShift = io.KeysDown[GLFW_KEY_LEFT_SHIFT] || io.KeysDown[GLFW_KEY_RIGHT_SHIFT];
		io.KeyAlt = io.KeysDown[GLFW_KEY_LEFT_ALT] || io.KeysDown[GLFW_KEY_RIGHT_ALT];
		io.KeySuper = io.KeysDown[GLFW_KEY_LEFT_SUPER] || io.KeysDown[GLFW_KEY_RIGHT_SUPER];
		return true;
	}

	return false;
}

bool GUIManagerInstance::char_callback(unsigned int c)
{
	if (focused) {
		ImGuiIO& io = ImGui::GetIO();
		if (c > 0 && c < 0x10000)
			io.AddInputCharacter((unsigned short)c);

		return true;
	}

	return false;
}


