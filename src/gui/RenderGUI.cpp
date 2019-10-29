#include "common.h"
#include "gui/RenderGUI.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <glm/gtc/type_ptr.hpp>

#include <map>

RenderGUI::RenderGUI(RenderOptions* options, RenderStats* stats)
{
	visible     = true;
	fullscreen  = false;
	displayname = "General Options";
	setOptions(options);
	setStats(stats);
}

void RenderGUI::setOptions(RenderOptions* options)
{
	this->options = options;
}

void RenderGUI::setStats(RenderStats* stats)
{
	this->stats = stats;
}

RenderGUI::~RenderGUI()
{}

void RenderGUI::draw(GLFWwindow* window, int order)
{

	ImVec2 outerSize = ImGui::GetItemRectSize();

	bool dirty = false;

    //////////////////////////////////////////////////
	////////////////////////// Timing display
	////////////////////////////////////////////////////
	if (ImGui::TreeNode("Timing")) {
		ImGui::Text("Render time cpu: %.3f", stats->time_cpu_ms);
		ImGui::Text("Render time opengl: %.3f", stats->time_gl_ms);
		ImGui::Text("Render time cuda: %.3f", stats->time_cuda_ms);
		ImGui::TreePop();
	}

	////////////////////////////////////////////////////
	////////////////////////// Display options
	////////////////////////////////////////////////////
	if (ImGui::TreeNode("Display")) {

		int width  = options->output_size.width;
		int height = options->output_size.height;
		ImGui::InputInt("Output width", &width);
		ImGui::InputInt("Output height", &height);
		options->output_size.width  = glm::clamp(width, 1, 8192);
		options->output_size.height = glm::clamp(height, 1, 8192);

		ImGui::Checkbox("Enable vsync", &options->vsync);

		int monitor_count; GLFWmonitor** monitors = glfwGetMonitors(&monitor_count);
		static int monitor_choice = 0;

		if (ImGui::Checkbox("Full screen", &fullscreen))
		{
			GLFWmonitor* monitor = monitor_choice < monitor_count ? glfwGetMonitors(&monitor_count)[monitor_choice] : glfwGetPrimaryMonitor();
			const GLFWvidmode* mode = glfwGetVideoMode(monitor);

			static int window_width, window_height, window_xpos, window_ypos;

			if (fullscreen) {
				glfwGetWindowSize(window, &window_width, &window_height);
				glfwGetWindowPos(window, &window_xpos, &window_ypos);
				glfwSetWindowMonitor(window, monitor, 0, 0, mode->width, mode->height, GLFW_DONT_CARE);
			}
			else {
				glfwSetWindowMonitor(window, nullptr, window_xpos, window_ypos, window_width, window_height, GLFW_DONT_CARE);
			}
		}

		static bool monitor_choice_dirty = false;
		if (ImGui::BeginCombo("Full screen monitor", std::to_string(monitor_choice).c_str())) {
			bool _b = true;
			for (int i = 0; i < monitor_count; ++i) {
				if (ImGui::Selectable(std::to_string(i).c_str(), _b)) {
					monitor_choice = i;
					monitor_choice_dirty = true;
				}
			}
			ImGui::EndCombo();
		}
		ImGui::TreePop();
	}

}