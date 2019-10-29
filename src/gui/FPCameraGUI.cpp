#include "common.h"
#include "gui/FPCameraGUI.h"
#include "managers/TextureManager.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <glm/gtc/type_ptr.hpp>

FPCameraGUI::FPCameraGUI(FPCamera* cam)
{
	setCamera(cam);
	visible     = true;
	displayname = "Camera Options";
}

FPCameraGUI::~FPCameraGUI()
{}

void FPCameraGUI::setCamera(FPCamera* cam)
{
	this->camptr = cam;
}

void FPCameraGUI::draw(GLFWwindow* window, int order)
{
	if (!camptr) return;

	ImVec2 outerSize = ImGui::GetItemRectSize();

	bool dirty = false;



	if (ImGui::DragFloat3("Position", glm::value_ptr(camptr->pos), 0.01f + abs(glm::length(camptr->position()) / 100.f ) )) dirty = true;
	if (ImGui::InputFloat4("Orientation", glm::value_ptr(camptr->ori), 3, ImGuiInputTextFlags_ReadOnly)) dirty = true;
	if (ImGui::DragFloat("Vertical FOV", &camptr->verticalFov_deg, 0.05f, 0.0f, 180.f)) dirty = true;
	if (ImGui::DragFloat("Near", &camptr->near, 0.0001f + camptr->nearplane() / 100.f, 0.00001f, camptr->farplane() - 0.0001f, "%.6f")) dirty = true;
	if (ImGui::DragFloat("Far", &camptr->far, 0.0001f + camptr->farplane() / 100.f, camptr->nearplane() + 0.0001f, 1e12f, "%.6f")) dirty = true;
	if (ImGui::DragFloat("Rotation speed", &camptr->rot_speed, 0.001f, 0.0f, 100.f, "%.6f")) dirty = true;
	if (ImGui::DragFloat("Motion speed", &camptr->mov_speed, 0.001f, 0.0f, 100.f, "%.6f")) dirty = true;
	if (ImGui::DragFloat("Jitter scale", &camptr->jitter_scale, 0.001f, 0.0f, 1.f, "%.6f")) dirty = true;

}