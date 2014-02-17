#ifndef _COMMON_H_
#define _COMMON_H_

struct NptRuntimes
{
    double twoWay;
    double threeWay;
};

#include <vector>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <stdint.h>
#include "timing.hpp"

#define HAVE_SSE2
#define HAVE_SSE3

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

typedef struct double3_t {
  double x;
  double y;
  double z;
} double3;

static __inline__ double3 make_double3(double x, double y, double z)
{
  double3 t; t.x = x; t.y = y; t.z = z; return t;
}



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

#if defined(HAVE_SSE3)
static NPT_INLINE __m128d npt_mm_loaddup_pd(const double* ptr) {
  return _mm_loaddup_pd(ptr);
}
#elif defined(HAVE_SSE2)
static NPT_INLINE __m128d npt_mm_loaddup_pd(const double* ptr) {
  __m128d temp = _mm_load_sd(ptr);
  return _mm_unpacklo_pd(temp, temp);
}
#endif



#endif // _COMMON_H_
