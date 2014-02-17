//
//  kernel_2pt_cpu.hpp
//  contrib_march
//
//  Created by William March on 11/1/12.
//
//

#ifndef _NPOINT_MLPACK_EFFICIENT_KERNELS_KERNEL_4PT_CPU_HPP_
#define _NPOINT_MLPACK_EFFICIENT_KERNELS_KERNEL_4PT_CPU_HPP_

#include "common.hpp"

void ComputeFourPointCorrelationCountsCPU(uint64_t *counts,
                                          NptRuntimes & nptRuntimes,
                                          const double3 *pointsA,
                                          int numPointsA,
                                          const double3 *pointsB,
                                          int numPointsB,
                                          const double3 *pointsC,
                                          int numPointsC,
                                          const double3 *pointsD,
                                          int numPointsD,
                                          const double *rMinSq,
                                          const double *rMaxSq,
                                          const unsigned char *satisfiabilityLUT);

void ComputeFourPointCorrelationCountsMultiCPU(uint64_t **counts,
                                              NptRuntimes & nptRuntimes,
                                              const double3 *pointsA,
                                              int numPointsA,
                                              const double3 *pointsB,
                                              int numPointsB,
                                               const double3 *pointsC,
                                               int numPointsC,
                                               const double3 *pointsD,
                                               int numPointsD,
                                              double **rMinSq,
                                              double **rMaxSq,
                                              int numMatchers,
                                              const unsigned char *satisfiabilityLUT);




#endif

