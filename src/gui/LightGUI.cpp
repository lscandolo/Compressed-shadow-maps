#include "common.h"
#include "gui/LightGUI.h"

#include <iostream> //!!

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <glm/gtc/type_ptr.hpp>

LightGUI::LightGUI(LightData* lights)
{
	setLights(lights);
	visible     = true;
	dirty       = true;
	displayname = "Light Options";
}

LightGUI::~LightGUI()
{}

void LightGUI::setLights(LightData* lights)
{
	this->lightsptr = lights;
}

bool LightGUI::createLightFrame(LightDataStructGL& l)
{
	dirty = false;

	{
		bool typePoint = false;
		bool typeDir = false;
		bool typeSpot = false;
		bool typeSphere = false;
		std::string currTypeStr;
		if (l.type == LightType::LT_POINT)       currTypeStr = "Point light";
		if (l.type == LightType::LT_DIRECTIONAL) currTypeStr = "Directional light";
		if (l.type == LightType::LT_SPOT)        currTypeStr = "Spot light";
		if (l.type == LightType::LT_SPHERE)        currTypeStr = "Sphere light";

		if (ImGui::BeginCombo("type", currTypeStr.c_str())) {
			if (ImGui::Selectable("Point light", &typePoint))     dirty = true;
			if (ImGui::Selectable("Directional light", &typeDir)) dirty = true;
			if (ImGui::Selectable("Spot light", &typeSpot))       dirty = true;
			if (ImGui::Selectable("Sphere light", &typeSphere))       dirty = true;
			ImGui::EndCombo();
		}

		if (typePoint)   l.type = LightType::LT_POINT;
		if (typeDir)     l.type = LightType::LT_DIRECTIONAL;
		if (typeSpot)    l.type = LightType::LT_SPOT;
		if (typeSphere)  l.type = LightType::LT_SPHERE;
	}

	if (ImGui::ColorEdit3("color", glm::value_ptr(l.color))) dirty = true;
	
	
	if (l.type != LightType::LT_DIRECTIONAL) {
		if (ImGui::DragFloat3("position", glm::value_ptr(l.position), 0.01f)) dirty = true;
	}

	if (l.type != LightType::LT_POINT && l.type != LightType::LT_SPHERE) {
		glm::vec3 ldir = l.direction;
		ImGui::DragFloat3("direction", glm::value_ptr(ldir), 0.01f, -1.f, 1.f);
		ldir = glm::normalize(ldir);
		if (glm::distance(ldir, l.direction) > 0.0001f) { l.direction = ldir; dirty = true; };
	}

	if (ImGui::DragFloat("intensity", &l.intensity, 0.001f)) dirty = true;
	
	if (l.type == LightType::LT_SPHERE) {
		if (ImGui::DragFloat("radius", &l.radius, 0.0001f + l.radius / 100.f, 0.f, 1e36f)) dirty = true;
	}

	if (l.type == LightType::LT_SPOT) {
		if (ImGui::DragFloat("spot angle", &l.spot_angle, 0.001f, 0.f, 3.14152f)) dirty = true;
	}

	return dirty;
};

static void pushButtonStyleColorByHue(float hue)
{
	ImGui::PushStyleColor(ImGuiCol_Button,        (ImVec4)ImColor::HSV(hue, 0.6f, 0.6f));
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(hue, 0.7f, 0.7f));
	ImGui::PushStyleColor(ImGuiCol_ButtonActive,  (ImVec4)ImColor::HSV(hue, 0.8f, 0.8f));
}

static void popButtonStyleColorByHue()
{
	ImGui::PopStyleColor(3);
}

void LightGUI::draw(GLFWwindow* window, int order)
{
	if (!lightsptr) return;

	lightsptr->setDirty(false);

	std::vector< LightDataStructGL>& lightList = lightsptr->gl.lightList;
		
	ImVec2 outerSize = ImGui::GetItemRectSize();

	pushButtonStyleColorByHue(0.28f);
	ImFont bigFont = *ImGui::GetFont(); bigFont.Scale = 1.5f; ImGui::PushFont(&bigFont);
	if (ImGui::Button(" + ")) {
		lightList.resize(lightList.size()+1);
		resetLightDataStruct(lightList.back());
		lightsptr->setDirty();
	}
	ImGui::Text("");
	popButtonStyleColorByHue();
	ImGui::PopFont();

	ImGui::DragFloat3("Ambient", glm::value_ptr(lightsptr->gl.ambient_light), 0.01f, 0.0f, 1e36f);


	using lightIt = std::vector<LightDataStructGL>::iterator;

	lightIt eraseIt = lightList.end();
	int lightnum = 0;
	for (lightIt it = lightList.begin(); it < lightList.end(); it++) {
		std::string lightName = std::string("Light ") + std::to_string(lightnum++);
			
		pushButtonStyleColorByHue(0.0f);
		std::string buttonName = std::string(" X ##") + lightName;
		if (ImGui::Button(buttonName.c_str() )) eraseIt = it;
		popButtonStyleColorByHue();
		ImGui::SameLine();

		if (ImGui::TreeNode(lightName.c_str()))
		{
			if (createLightFrame(*it)) { lightsptr->setDirty(); }
			ImGui::TreePop();
		}
	}

	if (eraseIt != lightList.end()) { lightList.erase(eraseIt); lightsptr->setDirty(); }

}