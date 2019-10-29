#pragma once

#include "helpers/Singleton.h"

#include <chrono>
#include <map>
#include <vector>
#include <list>

//#ifdef _WINDOWS 
#if 0

struct HighResClock
{
	typedef long long                               rep;
	typedef std::nano                               period;
	typedef std::chrono::duration<rep, period>      duration;
	typedef std::chrono::time_point<HighResClock>   time_point;
	static const bool is_steady = true;

	static time_point now();
};

#else
typedef std::chrono::high_resolution_clock HighResClock;
#endif

#define RECORD_CUDA 1
#define RECORD_GL   1


#if RECORD_CUDA
#include <cuda_runtime_api.h>
#endif

#include <string>
#include <stdint.h>

struct Timer
{
public:

#if RECORD_GL
	using glquery_t = unsigned int;
	glquery_t m_gl_start_query;
	glquery_t m_gl_end_query;
#endif

#if RECORD_CUDA
	cudaEvent_t m_cuda_start_event;
	cudaEvent_t m_cuda_end_event;
#endif

	static uint64_t m_cpu_frequency;
	HighResClock::time_point m_clock_start, m_clock_end;

	std::string name;

#if RECORD_GL
	float m_gl_time_elapsed;
#endif

#if RECORD_CUDA
	float m_cuda_time_elapsed;
#endif

	float m_cpu_time_elapsed;

	void start();
	void stop();
	int resolveTiming();
};

class ScopeTimer
{
public:
	Timer t;

	bool  m_enabled;
	ScopeTimer(const std::string &name, bool enable = true);
	~ScopeTimer();
};

#define CAT_ID2(_id1_, _id2_) _id1_##_id2_
#define CAT_ID(_id1_, _id2_) CAT_ID2(_id1_, _id2_)

#define PROFILE_SCOPE(_string_id_) \
	ScopeTimer CAT_ID(_scope_timer_, __LINE__)(_string_id_);

#define PROFILE_SCOPE_ENABLE(_string_id_,_enable_) \
	ScopeTimer CAT_ID(_scope_timer_, __LINE__)(_string_id_, _enable_);



struct TimingStats
{
	float gl_time;
	float cpu_time;
	float cuda_time;

	TimingStats() : gl_time(0), cpu_time(0), cuda_time(0) {}
};

class TimingManagerInstance
{

	typedef uint64_t id_t;

private:
	TimingManagerInstance();
	~TimingManagerInstance();

	std::map<std::string, TimingStats> scopeStats;
	std::list<Timer> outstandingTimings;
	std::list<std::string> scopeList;

public:

	void endFrame();
	void registerScope(const Timer &scope);
	void reset();
	bool timingEnabled();
	TimingStats getTimingStats(std::string name);
	friend Singleton<TimingManagerInstance>;
};

typedef Singleton<TimingManagerInstance> TimingManager;