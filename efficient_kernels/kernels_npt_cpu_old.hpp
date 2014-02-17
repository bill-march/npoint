//#include <vector>
//#include <cstdio>
//#include <cstring>
//#include <stdint.h>
//#include "timing.hpp"

//#include "kernels_npt_cpu.hpp"

// When using CUDA, the following double3 functions (and many others)
// are in <vector_types.h> and <vector_functions.h>

/*
typedef struct double3_t {
  double x;
  double y; 
  double z;
} double3;
*/

/*
static __inline__ double3 operator-(double3 a, double3 b)
{
  return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}
 */
/*
void FillThreePointCorrelationLUT(std::vector<unsigned char> & satisfiabilityLUT)
{
  // 9 bits needed
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
    // if i has ones in the same places as at least one of the masks
    // if so, then the three octets correspond to a set of distances that 
    // satisfy the matcher in some permutation
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
  }
}
*/
static void PrecomputePairwiseInteractionsCPU(
                                              unsigned char *pairwiseInteractions,
                                              uint64_t *counts,
                                              const double3 *pointsA,
                                              int numPointsA,
                                              const double3 *pointsB,
                                              int numPointsB,
                                              const double *rMinSq,
                                              const double *rMaxSq)
{
  const double rMinSq0 = rMinSq[0], rMinSq1 = rMinSq[1], rMinSq2 = rMinSq[2];
  const double rMaxSq0 = rMaxSq[0], rMaxSq1 = rMaxSq[1], rMaxSq2 = rMaxSq[2];
  
#pragma omp parallel for schedule(dynamic,1)
  for (int i = 0; i < numPointsA; ++i)
  {
    const double3 a = pointsA[i];
    for (int j = 0; j < numPointsB; ++j)
    {
      const double3 b = pointsB[j];
      const double3 ba_diff = b - a;
      const double dist_ba_sq =
      ba_diff.x*ba_diff.x + ba_diff.y*ba_diff.y + ba_diff.z*ba_diff.z;
      
      unsigned char sat = 0x0;
      // We no longer track pairwise satisfied matcher counts
      //if (dist_ba_sq > rMinSq0 && dist_ba_sq < rMaxSq0) { sat |= 0x1; counts[0]++; }
      //if (dist_ba_sq > rMinSq1 && dist_ba_sq < rMaxSq1) { sat |= 0x2; counts[1]++; }
      //if (dist_ba_sq > rMinSq2 && dist_ba_sq < rMaxSq2) { sat |= 0x4; counts[2]++; }
      
      // set the 0th bit if the 0th matcher distance is satisfied
      if (dist_ba_sq > rMinSq0 && dist_ba_sq < rMaxSq0) { sat |= 0x1; }
      if (dist_ba_sq > rMinSq1 && dist_ba_sq < rMaxSq1) { sat |= 0x2; }
      if (dist_ba_sq > rMinSq2 && dist_ba_sq < rMaxSq2) { sat |= 0x4; }
      pairwiseInteractions[i*numPointsB+j] = sat;
    }
  }
}

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
                                           const unsigned char *satisfiabilityLUT)
{
  unsigned char *pairwise_AB = new unsigned char[numPointsA*numPointsB];
  unsigned char *pairwise_AC = new unsigned char[numPointsA*numPointsC];
  unsigned char *pairwise_BC = new unsigned char[numPointsB*numPointsC];
  
  memset(counts, 0, sizeof(uint64_t)*4);
  
  //    double start = Timing::ElapsedTimeMs();
  PrecomputePairwiseInteractionsCPU(
                                    pairwise_AB, counts, pointsA, numPointsA, pointsB, numPointsB, rMinSq, rMaxSq);
  PrecomputePairwiseInteractionsCPU(
                                    pairwise_AC, counts, pointsA, numPointsA, pointsC, numPointsC, rMinSq, rMaxSq);
  PrecomputePairwiseInteractionsCPU(
                                    pairwise_BC, counts, pointsB, numPointsB, pointsC, numPointsC, rMinSq, rMaxSq);
  //    double elapsedPairwise = Timing::ElapsedTimeMs() - start;
  
#pragma omp parallel for schedule(dynamic,1)
  for (int i = 0; i < numPointsA; ++i)
  {
    int count3_addition = 0;
    for (int j = 0; j < numPointsB; ++j)
    {
      const unsigned char sat0 = pairwise_AB[i*numPointsB+j];
      
      // If this (A,B) pairing does not satisfy any of the matcher distances,
      // then we can short-circuit out of testing (A,B,*) triplets.
      if (! sat0) { continue; }
      
      //printf("found a pair that works.\n");
      // were only looking at the part of the table that corresponds to the 
      // AB octet
      const unsigned short sat0_shift6 = sat0 << 6;
      
      const uint64_t *pairwise_AC_s = (const uint64_t *) (pairwise_AC + i*numPointsC);
      const uint64_t *pairwise_BC_s = (const uint64_t *) (pairwise_BC + j*numPointsC);
      
      const unsigned char * localSatisfiabilityLUT = &satisfiabilityLUT[sat0_shift6];
      
      // consider 8 C points at a time
      for (int k = 0; k < numPointsC/8; ++k)
      {
        const uint64_t sat1 = pairwise_AC_s[k];
        const uint64_t sat2 = pairwise_BC_s[k];
        const uint64_t sat12 = (sat1 << 3) | sat2;
        
        // right shift and take 6 lower bits
        const size_t ind0 = ((sat12 >> 0) & 0x3F);
        const size_t ind1 = ((sat12 >> 8) & 0x3F);
        const size_t ind2 = ((sat12 >> 16) & 0x3F);
        const size_t ind3 = ((sat12 >> 24) & 0x3F);
        const size_t ind4 = ((sat12 >> 32) & 0x3F);
        const size_t ind5 = ((sat12 >> 40) & 0x3F);
        const size_t ind6 = ((sat12 >> 48) & 0x3F);
        const size_t ind7 = ((sat12 >> 56) & 0x3F);
        
        // what if many of these are satisfied
        // this is considering many tuples at once?
        count3_addition += localSatisfiabilityLUT[ ind0 ];
        count3_addition += localSatisfiabilityLUT[ ind1 ];
        count3_addition += localSatisfiabilityLUT[ ind2 ];
        count3_addition += localSatisfiabilityLUT[ ind3 ];
        count3_addition += localSatisfiabilityLUT[ ind4 ];
        count3_addition += localSatisfiabilityLUT[ ind5 ];
        count3_addition += localSatisfiabilityLUT[ ind6 ];
        count3_addition += localSatisfiabilityLUT[ ind7 ];
      }
      
      // take care of the numPointsC not divisible by 8 case
      for (int k = 8*(numPointsC/8); k < numPointsC; ++k)
      {
        const unsigned char sat1 = pairwise_AC[i*numPointsC+k];
        const unsigned char sat2 = pairwise_BC[j*numPointsC+k];
        const unsigned short sat = sat0_shift6 | (sat1 << 3) | sat2;
        count3_addition += satisfiabilityLUT[sat];
      }
    }
    
    //printf("found %d working pairs\n", count3_addition);
    
#pragma omp critical
    counts[3] += count3_addition;
  }
  
  //double elapsed = Timing::ElapsedTimeMs() - start;
  
  //    printf("ComputeThreePointCorrelationCountsCPU: Timing = %.2f [ms] (%.2f [ms] for pairwise computations)\n",
  //        elapsed, elapsedPairwise);
  //nptRuntimes.twoWay = elapsedPairwise;
  //  nptRuntimes.threeWay = elapsed-elapsedPairwise;
  //printf("CPU: 2-way: %.04f\n", elapsedPairwise);
  //printf("CPU: 3-way: %.04f\n", elapsed-elapsedPairwise);
  //printf("CPU: Total: %.04f\n", elapsed);
  
  delete [] pairwise_AB;
  delete [] pairwise_AC;
  delete [] pairwise_BC;
}

