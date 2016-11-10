// timer.h
#ifndef _TIMER_H
#define _TIMER_H

#include <stdio.h>

#include "CL/opencl.h"


#ifdef _WIN32
#include <windows.h>

class Timer {
  public:
    Timer();
    void start();
    void stop();
    float get_time_s();
  private:
    LARGE_INTEGER m_start_time;
    LARGE_INTEGER m_stop_time;
    LARGE_INTEGER m_ticks_per_second;
};

#else // LINUX

#include <time.h>

class Timer {
  public:
    Timer();
    void start();
    void stop(); 
    float get_time_s();
    
  private:
    double m_start_time;
    double m_stop_time;
    double get_cur_time_s(void);
};
#endif

// Timer using the OpenCL event profiling API
cl_ulong getKernelStartEndTime(cl_event event);

#endif // _TIMER_H

