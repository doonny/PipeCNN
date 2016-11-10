// timer.cpp
#include "timer.h"

#ifdef _WIN32

Timer::Timer() {
  QueryPerformanceFrequency( &m_ticks_per_second );
}

void Timer::start() {
  QueryPerformanceCounter( &m_start_time );
}

void Timer::stop() {
  QueryPerformanceCounter( &m_stop_time );
}

float Timer::get_time_s() {
  LONGLONG delta = (m_stop_time.QuadPart - m_start_time.QuadPart);
  return (float)delta / (float)(m_ticks_per_second.QuadPart);
}

#else // LINUX
#include <stdio.h>
Timer::Timer() {
}

void Timer::start() {
  m_start_time = get_cur_time_s();
}

void Timer::stop() {
  m_stop_time = get_cur_time_s();
}

float Timer::get_time_s() {
  return (float)(m_stop_time - m_start_time);
}


double Timer::get_cur_time_s(void) {
  struct timespec a;
  const double NS_PER_S = 1000000000.0;
  clock_gettime (CLOCK_REALTIME, &a);
  return ((double)a.tv_nsec / NS_PER_S) + (double)(a.tv_sec);
}
#endif

// Timer using the OpenCL event profiling API
cl_ulong getKernelStartEndTime(cl_event event) {
	cl_int status;
	
	cl_ulong start, end;
	status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
	if(status != CL_SUCCESS)
		printf("Error: cannot get kernel start time !");
	
	status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
	if(status != CL_SUCCESS)
		printf("Error: cannot get kernel end time !");

	return end - start;
}



