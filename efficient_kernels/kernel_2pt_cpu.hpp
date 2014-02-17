//
//  kernel_2pt_cpu.hpp
//  contrib_march
//
//  Created by William March on 11/1/12.
//
//

#ifndef _NPOINT_MLPACK_EFFICIENT_KERNELS_KERNEL_2PT_CPU_HPP_
#define _NPOINT_MLPACK_EFFICIENT_KERNELS_KERNEL_2PT_CPU_HPP_

#include "common.hpp"

void ComputeTwoPointCorrelationCountsCPU(uint64_t *counts,
                                         NptRuntimes & nptRuntimes,
                                         const double3 *pointsA,
                                         int numPointsA,
                                         const double3 *pointsB,
                                         int numPointsB,
                                         const double *rMinSq,
                                         const double *rMaxSq,
                                         const unsigned char *satisfiabilityLUT);

void ComputeTwoPointCorrelationCountsMultiCPU(uint64_t **counts,
                                         NptRuntimes & nptRuntimes,
                                         const double3 *pointsA,
                                         int numPointsA,
                                         const double3 *pointsB,
                                         int numPointsB,
                                         double **rMinSq,
                                         double **rMaxSq,
                                         int numMatchers,
                                         const unsigned char *satisfiabilityLUT);




#endif

