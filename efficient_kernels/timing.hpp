#ifndef _TIMING_HPP_
#define _TIMING_HPP_

#include <time.h>
#include <string>
#include <vector>

class Timing
{
public:

    // Sets the initial time to which ElapsedTimeMs() will be relative and initializers
    // the string names of the timers
    static void Init();

    // Milliseconds elapsed since calling Init()
    static double ElapsedTimeMs();

private:

    Timing() {}
    ~Timing() {}
    
    // The approximate time at which the process started.  This is used to report
    // relative timings to generate a timeline.
    static struct timespec m_process_init_time;
};

#endif // _TIMING_HPP_
