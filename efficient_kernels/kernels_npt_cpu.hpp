#ifndef _KERNELS_NPT_CPU_H_
#define _KERNELS_NPT_CPU_H_

//#include <vector>
//#include <iostream>
//#include <cstdio>
//#include <cstring>
#include "common.hpp"

//#define HAVE_SSE2
//#define HAVE_SSE3

/*
typedef struct double3_t {
  double x;
  double y; 
  double z;
} double3;

static __inline__ double3 make_double3(double x, double y, double z)
{
  double3 t; t.x = x; t.y = y; t.z = z; return t;
}
*/
void FillThreePointCorrelationLUT(std::vector<unsigned char> & satisfiabilityLUT);

void ComputeThreePointCorrelationCountsCPU(
    uint64_t *counts,
    NptRuntimes & nptRuntimes,
    const double3 *pointsA,
    int numPointsA,
    const double3 *pointsB,
    int numPointsB,
    const double3 *pointsC,
    int numPointsC,
    const double *rMinSq,
    const double *rMaxSq,
    const unsigned char *satisfiabilityLUT);

void ComputeThreePointCorrelationCountsMultiCPU(
                                           uint64_t **counts,
                                           NptRuntimes & nptRuntimes,
                                           const double3 *pointsA,
                                           int numPointsA,
                                           const double3 *pointsB,
                                           int numPointsB,
                                           const double3 *pointsC,
                                           int numPointsC,
                                           double **rMinSq,
                                           double **rMaxSq,
                                           int numMatchers,
                                           const unsigned char *satisfiabilityLUT);


#endif // _KERNELS_NPT_CPU_H_
