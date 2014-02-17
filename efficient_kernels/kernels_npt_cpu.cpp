//#include <vector>
//#include <cstdio>
//#include <cstring>
//#include <stdint.h>
//#include "timing.hpp"

#include "kernels_npt_cpu.hpp"
//#include <vector_functions.h>

/*
#ifdef _MSC_VER
#include <intrin.h>
#else

#ifdef __arm__
	#include <arm_neon.h>
#else
	#include <emmintrin.h>
#endif
#ifdef HAVE_SSE3
	#include <pmmintrin.h>
#endif
#ifdef HAVE_SSSE3
	#include <tmmintrin.h>
#endif
#ifdef HAVE_XOP
	#include <x86intrin.h>
#endif
#endif



#ifdef _MSC_VER
#define NPT_INLINE __forceinline
#else
#define NPT_INLINE __inline__
#endif
*/
// Kernel versions are as follows:
//   Kernel0 : LUT-based
//   Kernel1 : Converted to non-LUT popcnt approach using NPT_UNROLL_FACTOR
//   Kernel2 : Non-LUT with special treatment for each of the 8 possible
//             bit arrangements for the first sat variable.
//#define CPU_KERNEL_VERSION_0
//#define CPU_KERNEL_VERSION_1
//#define CPU_KERNEL_VERSION_2
//#define CPU_KERNEL_VERSION_4
#define CPU_KERNEL_VERSION_5

/*
static __inline__ double3 operator-(double3 a, double3 b)
{
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static __inline__ uint64_t npt_popcount64(uint64_t x) {
#ifdef _MSC_VER
#ifdef HAVE_POPCNT
#ifdef _WIN64
	return __popcnt64(x);
#else
	return __popcnt(uint32_t(x)) + __popcnt(uint32_t(x >> 32));
#endif
#else
	// Code snippet from http://en.wikipedia.org/wiki/Hamming_weight
    x -= (x >> 1) & 0x5555555555555555ull;
    x = (x & 0x3333333333333333ull) + ((x >> 2) & 0x3333333333333333ull);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0Full;
    return (x * 0x0101010101010101ull) >> 56;
#endif
#else
	return __builtin_popcountll(x);
#endif
}

static __inline__ uint32_t npt_popcount32(uint32_t x) {
#ifdef _MSC_VER
#ifdef HAVE_POPCNT
	return __popcnt(x);
#else
	// Code snippet from http://stackoverflow.com/questions/109023/best-algorithm-to-count-the-number-of-set-bits-in-a-32-bit-integer/
	x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
#endif
#else
	return __builtin_popcount(x);
#endif
}

static __inline__ uint32_t npt_popcount16(uint16_t x) {
#ifdef _MSC_VER
#ifdef HAVE_POPCNT
	return __popcnt(x);
#else
	// Code snippet from http://stackoverflow.com/questions/109023/best-algorithm-to-count-the-number-of-set-bits-in-a-32-bit-integer/
	// Todo: write 16-bit version
	x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
#endif
#else
	return __builtin_popcount(x);
#endif
}

static __inline__ uint32_t npt_popcount8(uint8_t x) {
#ifdef _MSC_VER
#ifdef HAVE_POPCNT
	return __popcnt(x);
#else
	// Code snippet from http://stackoverflow.com/questions/109023/best-algorithm-to-count-the-number-of-set-bits-in-a-32-bit-integer/
	// Todo: write 8-bit version
	x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
#endif
#else
	return __builtin_popcount(x);
#endif
}
*/
void FillThreePointCorrelationLUT(std::vector<unsigned char> & satisfiabilityLUT)
{
    satisfiabilityLUT.resize(512);

    const unsigned short VALID_MASKS[] = {   
        0x111,   // 100 010 001
        0x10A,   // 100 001 010
        0x0A1,   // 010 100 001
        0x08C,   // 010 001 100
        0x062,   // 001 100 010
        0x054 }; // 001 010 100

    for (unsigned short i = 0; i < 512; ++i)
    {
        if ( ((i & VALID_MASKS[0]) == VALID_MASKS[0]) ||
           ((i & VALID_MASKS[1]) == VALID_MASKS[1]) ||
           ((i & VALID_MASKS[2]) == VALID_MASKS[2]) ||
           ((i & VALID_MASKS[3]) == VALID_MASKS[3]) ||
           ((i & VALID_MASKS[4]) == VALID_MASKS[4]) ||
           ((i & VALID_MASKS[5]) == VALID_MASKS[5]) )
        {
            satisfiabilityLUT[i] = 1;
        }
        else
        {
            satisfiabilityLUT[i] = 0;
        }
      
      //printf("satisfiability[%hu]: %hu\n", i, satisfiabilityLUT[i]);
    }
}

#if defined(CPU_KERNEL_VERSION_0)
    #include "kernels_npt_cpu.cpp.kern0.hpp"
#elif defined(CPU_KERNEL_VERSION_1)
    #include "kernels_npt_cpu.cpp.kern1.hpp"
#elif defined(CPU_KERNEL_VERSION_2)
    #include "kernels_npt_cpu.cpp.kern2.hpp"
#elif defined(CPU_KERNEL_VERSION_3)
    #include "kernels_npt_cpu.cpp.kern3.hpp"
#elif defined(CPU_KERNEL_VERSION_4)
    #include "kernels_npt_cpu.cpp.kern4.hpp"
#elif defined(CPU_KERNEL_VERSION_5)
    #include "kernels_npt_cpu.cpp.kern5.hpp"
    #include "efficient_3pt_multi_kernel.hpp"
#else
    #error "Must choose a valid CPU kernel implementation"
#endif

