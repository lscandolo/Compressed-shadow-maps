#pragma once

#include "common.h"
#include "helpers/Singleton.h"
#include "techniques/Technique.h"

#include <array>

// Quick and dirty structure to pass around data between parts of the program
//    by using a singleton object (DefaultRenderOptions)

struct RenderOptions
{
	bool vsync     = false;

	enum FlagType
	{
		FLAG_DIRTY_TECHNIQUE,
		FLAG_COUNT,
	};
		
	size2D window_size;
	size2D output_size;
	std::array<char, 256> output_id;
	bool    is_recording;
	int32_t recording_frame;
	void start_recording();
	void stop_recording();
	void new_frame();
	std::string get_output_directory();
	std::string get_output_id();

	std::array<float, 5> debug_var;

	void set_flag(FlagType f);
	bool get_flag(FlagType f);
	void reset_flag(FlagType f);

	Technique* current_technique();
	std::vector<std::string> get_technique_names();
	void add_technique(Technique* technique);
	int request_technique(std::string technique_name);
	int switch_to_requested_technique();
	int initialize_technique(const Technique::SetupData& ts);

private:

	std::array<bool, FlagType::FLAG_COUNT> flags;

	std::vector<Technique*> techniques;
	int current_technique_idx;
	int requested_technique_idx;

	friend Singleton<RenderOptions>;
	RenderOptions();
};

struct RenderStats
{
	float time_cpu_ms;
	float time_gl_ms;
	float time_cuda_ms;
};

RenderOptions& DefaultRenderOptions();
