#include <common.h>

#include "helpers/ScopeTimer.h"
#include <GL/glew.h>
#include <sstream>
#include <algorithm>

#ifdef _WINDOWS

#include <Windows.h>

namespace
{
	const long long g_Frequency = []() -> long long
	{
		LARGE_INTEGER frequency;
		QueryPerformanceFrequency(&frequency);
		return frequency.QuadPart;
	}();
}

HighResClock::time_point HighResClock::now()
{
	LARGE_INTEGER count;
	QueryPerformanceCounter(&count);
	return time_point(duration(count.QuadPart * static_cast<rep>(period::den) / g_Frequency));
}
#endif


void Timer::start()
{
#if RECORD_GL
	m_gl_time_elapsed = 0;
#endif

#if RECORD_CUDA
	m_cuda_time_elapsed = 0;
#endif

	m_cpu_time_elapsed = 0;
	///////////////////////////////////////////////////////////////////////////
	// Register into PerformanceProfiler
	///////////////////////////////////////////////////////////////////////////
	// TimingManagerInstance::instance().enterScope(m_name);

#if RECORD_GL
	///////////////////////////////////////////////////////////////////////////
	// Record OpenGL start time
	///////////////////////////////////////////////////////////////////////////
	glGenQueries(1, &m_gl_start_query);
	glGenQueries(1, &m_gl_end_query);
	glQueryCounter(m_gl_start_query, GL_TIMESTAMP);
#endif

#if RECORD_CUDA
	///////////////////////////////////////////////////////////////////////////
	// Record CUDA start time
	///////////////////////////////////////////////////////////////////////////
	cudaEventCreate(&m_cuda_start_event);
	cudaEventCreate(&m_cuda_end_event);
	cudaEventRecord(m_cuda_start_event, 0);
#endif
	///////////////////////////////////////////////////////////////////////////
	// Record CPU start time
	///////////////////////////////////////////////////////////////////////////

	m_clock_start = HighResClock::now();

}

void Timer::stop()
{
	///////////////////////////////////////////////////////////////////////////
	// Record CPU end time
	///////////////////////////////////////////////////////////////////////////
	m_clock_end = HighResClock::now();

#if RECORD_CUDA
	///////////////////////////////////////////////////////////////////////////
	// Record CUDA end time
	///////////////////////////////////////////////////////////////////////////
	cudaEventRecord(m_cuda_end_event, 0);
#endif 

#if RECORD_GL
	///////////////////////////////////////////////////////////////////////////
	// Record OpenGL end time
	///////////////////////////////////////////////////////////////////////////
	glQueryCounter(m_gl_end_query, GL_TIMESTAMP);
#endif 

}

int Timer::resolveTiming()
{

#if RECORD_GL
	///////////////////////////////////////////////////////////////////////////
	// Query availability of GL results
	///////////////////////////////////////////////////////////////////////////
	uint64_t glstart_avail, glend_avail;
	glGetQueryObjectui64v(m_gl_start_query, GL_QUERY_RESULT_AVAILABLE, &glstart_avail);
	glGetQueryObjectui64v(m_gl_end_query, GL_QUERY_RESULT_AVAILABLE, &glend_avail);
	if (glstart_avail == GL_FALSE || glend_avail == GL_FALSE) return 1;
	check_opengl();

#endif


#if RECORD_CUDA
	///////////////////////////////////////////////////////////////////////////
	// Query availability of CUDA results
	///////////////////////////////////////////////////////////////////////////
	cudaError_t custart_res, cuend_res;
	custart_res = cudaEventQuery(m_cuda_start_event);
	cuend_res = cudaEventQuery(m_cuda_end_event);
	if (custart_res == cudaErrorNotReady || cuend_res == cudaErrorNotReady) return -1;
#endif 

#if RECORD_GL
	///////////////////////////////////////////////////////////////////////////
	// Synchronize GL and calculate elapsed time
	///////////////////////////////////////////////////////////////////////////
	uint64_t gl_start_time, gl_stop_time;
	glGetQueryObjectui64v(m_gl_start_query, GL_QUERY_RESULT_NO_WAIT, &gl_start_time);
	glGetQueryObjectui64v(m_gl_end_query, GL_QUERY_RESULT_NO_WAIT, &gl_stop_time);
	m_gl_time_elapsed = float(double(gl_stop_time - gl_start_time) / 1e6);
	glDeleteQueries(1, &m_gl_start_query);
	glDeleteQueries(1, &m_gl_end_query);
	check_opengl();

#endif 

#if RECORD_CUDA
	///////////////////////////////////////////////////////////////////////////
	// Synchronize CUDA and calculate elapsed time
	///////////////////////////////////////////////////////////////////////////
	cudaEventElapsedTime(&m_cuda_time_elapsed, m_cuda_start_event, m_cuda_end_event);
	cudaEventDestroy(m_cuda_start_event);
	cudaEventDestroy(m_cuda_end_event);
#endif 

	std::chrono::duration<double, std::milli> duration = m_clock_end - m_clock_start;
	m_cpu_time_elapsed = float(duration.count());

	return 0;
}



///////////////////////////////////////////////////////////////////////////////
// Query CPU's frequency ONCE
///////////////////////////////////////////////////////////////////////////////
ScopeTimer::ScopeTimer(const std::string &name, bool enable)
{
	m_enabled = enable;

	if (!m_enabled) return;

	t.name = name;

	if (TimingManager::instance().timingEnabled()) t.start();

};

ScopeTimer::~ScopeTimer()
{
	if (!m_enabled) return;

	if (TimingManager::instance().timingEnabled()) t.stop();

	///////////////////////////////////////////////////////////////////////////
	// Register into PerformanceProfiler
	///////////////////////////////////////////////////////////////////////////
	TimingManager::instance().registerScope(t);

};



////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////


TimingManagerInstance::TimingManagerInstance()
{
}

TimingManagerInstance::~TimingManagerInstance()
{
}

bool
TimingManagerInstance::timingEnabled()
{
	return true;
}

void
TimingManagerInstance::reset()
{
	scopeStats.clear();
	scopeList.clear();
}

void
TimingManagerInstance::registerScope(const Timer &scope)
{
	outstandingTimings.push_back(scope);
}

void TimingManagerInstance::endFrame()
{

	std::vector<std::list<Timer>::iterator> deleteList;

	for (auto& it = outstandingTimings.begin(); it != outstandingTimings.end(); ++it)
	{
		if (it->resolveTiming()) continue;

		scopeStats[it->name].cpu_time = it->m_cpu_time_elapsed;

#if RECORD_GL
		scopeStats[it->name].gl_time = it->m_gl_time_elapsed;
#endif

#if RECORD_CUDA
		scopeStats[it->name].cuda_time = it->m_cuda_time_elapsed;
#endif

		deleteList.push_back(it);
	}

	for (auto& it : deleteList) outstandingTimings.erase(it);

}


TimingStats 
TimingManagerInstance::getTimingStats(std::string name)
{
	auto& stats = scopeStats.find(name);
	if (stats == scopeStats.end())
		return TimingStats();

	return stats->second;
}

