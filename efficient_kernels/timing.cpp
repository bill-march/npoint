#include "timing.hpp"
#include <cstdio>
#include <cassert>
#include <stdlib.h>
#include <stdarg.h>
#include <cstring>

struct timespec Timing::m_process_init_time = { 0, 0 };

void Timing::Init()
{
  /*  
  int ret = clock_gettime(CLOCK_REALTIME, &m_process_init_time);
    if (ret != 0)
    {
        perror("clock_gettime");
        // FIXME: This does not have to be fatal, but it should disable the timer
        exit(-1);
    }
   */
  
}

double Timing::ElapsedTimeMs()
{
  //struct timespec t;
  //assert(clock_gettime(CLOCK_REALTIME, &t) == 0);
  //double diff_ms = (1000.0 * t.tv_sec + 1.0e-6 * t.tv_nsec) -
  //    (1000.0 * m_process_init_time.tv_sec + 1.0e-6 * m_process_init_time.tv_nsec);
  //return diff_ms;
  return 0.0;
}
