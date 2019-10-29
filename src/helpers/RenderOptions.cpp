#pragma once

#include "common.h"
#include "helpers/RenderOptions.h"


RenderOptions& DefaultRenderOptions() { return Singleton<RenderOptions>::instance(); }

RenderOptions::RenderOptions()
{
	char default_name[] = "default";
	output_id.fill(0);
	memcpy(output_id.data(), default_name, std::min(sizeof(default_name), output_id.size()) );
	is_recording = false;
	recording_frame = 0;
	current_technique_idx = -1;
	requested_technique_idx = -1;
}

void RenderOptions::add_technique(Technique* technique)
{
	techniques.push_back(technique);
}

Technique* RenderOptions::current_technique()
{
	if (current_technique_idx < 0 || current_technique_idx >= techniques.size()) return nullptr;

	return techniques[current_technique_idx];
}

std::vector<std::string> RenderOptions::get_technique_names()
{
	std::vector<std::string> names;
	for (auto& t : techniques) names.push_back(t->name());
	return names;
}

int RenderOptions::request_technique(std::string technique_name)
{
	requested_technique_idx = -1;
	for (int i = 0; i < techniques.size(); ++i) 
	{
		if (techniques[i]->name() == technique_name) {
			requested_technique_idx = i;
			break;
		}
	}

	if (requested_technique_idx < 0) return ERROR_INVALID_PARAMETER;

	set_flag(FLAG_DIRTY_TECHNIQUE);

	return SUCCESS;
}

int RenderOptions::switch_to_requested_technique()
{
	if (requested_technique_idx < 0 || requested_technique_idx >= techniques.size()) {
		return ERROR_INVALID_PARAMETER;
	}

	Technique* current_t = current_technique();
	if (current_t) current_t->destroy();

	Technique* requested_t = techniques[requested_technique_idx];
	current_technique_idx = requested_technique_idx;
	return SUCCESS;
}

int RenderOptions::initialize_technique(const Technique::SetupData& ts)
{
	Technique* current_t = current_technique();
	if (current_t) {
		current_t->destroy();
		return current_t->initial_setup(ts);
	}
	return ERROR_UNINITIALIZED_OBJECT;
}

void RenderOptions::start_recording()
{
	if (is_recording) return; 

	is_recording    = true;
	recording_frame = 0;
}

void RenderOptions::stop_recording()
{
	is_recording = false;
	recording_frame = 0;
}

void RenderOptions::new_frame()
{
	if (is_recording) recording_frame++;
}

std::string RenderOptions::get_output_directory()
{
	std::string dir = std::string(output_id.data());
	if (dir.find(':') == std::string::npos) {
		return std::string("output/").append(std::string(output_id.data()));
	} else {
		return dir;
	}
	
}

std::string RenderOptions::get_output_id()
{
	return std::string(output_id.data());
}

void RenderOptions::set_flag(FlagType f)
{
	flags[f] = true;
}

bool RenderOptions::get_flag(FlagType f)
{
	return flags[f];

}
void RenderOptions::reset_flag(FlagType f)
{
	flags[f] = false;
}
