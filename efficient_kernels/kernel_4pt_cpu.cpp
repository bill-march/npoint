//
//  kernel_4pt_cpu.cpp
//  contrib_march
//
//  Created by William March on 11/15/12.
//
//

#include "kernel_4pt_cpu.hpp"

static void PrecomputePairwiseInteractions4ptCPU(uint64_t *__restrict pairwiseInteractions,
                                                 uint64_t * /* counts */,
                                                 const double3 *__restrict pointsA,
                                                 int numPointsA,
                                                 const double3 *__restrict pointsB,
                                                 int numPointsB,
                                                 const double *__restrict rMinSq,
                                                 const double *__restrict rMaxSq)
{
#ifdef HAVE_SSE2
  // makes both values of the packed double the same
  const __m128d rMinSq0 = npt_mm_loaddup_pd(&rMinSq[0]);
  const __m128d rMaxSq0 = npt_mm_loaddup_pd(&rMaxSq[0]);
  
  const __m128d rMinSq1 = npt_mm_loaddup_pd(&rMinSq[1]);
  const __m128d rMaxSq1 = npt_mm_loaddup_pd(&rMaxSq[1]);
  
  const __m128d rMinSq2 = npt_mm_loaddup_pd(&rMinSq[2]);
  const __m128d rMaxSq2 = npt_mm_loaddup_pd(&rMaxSq[2]);
  
  const __m128d rMinSq3 = npt_mm_loaddup_pd(&rMinSq[3]);
  const __m128d rMaxSq3 = npt_mm_loaddup_pd(&rMaxSq[3]);

  const __m128d rMinSq4 = npt_mm_loaddup_pd(&rMinSq[4]);
  const __m128d rMaxSq4 = npt_mm_loaddup_pd(&rMaxSq[4]);

  const __m128d rMinSq5 = npt_mm_loaddup_pd(&rMinSq[5]);
  const __m128d rMaxSq5 = npt_mm_loaddup_pd(&rMaxSq[5]);

#else
  const double rMinSq0 = rMinSq[0];
  const double rMaxSq0 = rMaxSq[0];

  const double rMinSq1 = rMinSq[1];
  const double rMaxSq1 = rMaxSq[1];
  
  const double rMinSq2 = rMinSq[2];
  const double rMaxSq2 = rMaxSq[2];
  
  const double rMinSq3 = rMinSq[3];
  const double rMaxSq3 = rMaxSq[3];

  const double rMinSq4 = rMinSq[4];
  const double rMaxSq4 = rMaxSq[4];
  
  const double rMinSq5 = rMinSq[5];
  const double rMaxSq5 = rMaxSq[5];

#endif
  
	for (int i = 0; i < numPointsA; i += 1) {
#ifdef HAVE_SSE2
    const __m128d ax_ay = _mm_loadu_pd(static_cast<const double*>(&pointsA[i].x));
    const __m128d ay_az = _mm_loadu_pd(static_cast<const double*>(&pointsA[i].y));
    const __m128d az_ax = _mm_shuffle_pd(ay_az, ax_ay, 1);
#else
    const double3 a = pointsA[i];
#endif
		const double3* localPointsB = pointsB;
		uint64_t sat0 = 0;
		uint64_t sat1 = 0;
		uint64_t sat2 = 0;
		uint64_t sat3 = 0;
		uint64_t sat4 = 0;
		uint64_t sat5 = 0;
#ifdef HAVE_SSE2
    for (int k = 1; k < numPointsB; k += 2) {
      const __m128d b0x_b0y = _mm_loadu_pd(&localPointsB[0].x);
      const __m128d b0z_b1x = _mm_loadu_pd(&localPointsB[0].z);
      const __m128d b1y_b1z = _mm_loadu_pd(&localPointsB[1].y);
      localPointsB += 2;
			
      // compute b - a
      const __m128d b0x_b0y_diff_a = _mm_sub_pd(b0x_b0y, ax_ay);
      const __m128d b0z_b1x_diff_a = _mm_sub_pd(b0z_b1x, az_ax);
      const __m128d b1y_b1z_diff_a = _mm_sub_pd(b1y_b1z, ay_az);
      
      // compute (b - a)^2
      const __m128d b0x_b0y_diff_a_sq = _mm_mul_pd(b0x_b0y_diff_a, b0x_b0y_diff_a);
      const __m128d b0z_b1x_diff_a_sq = _mm_mul_pd(b0z_b1x_diff_a, b0z_b1x_diff_a);
      const __m128d b1y_b1z_diff_a_sq = _mm_mul_pd(b1y_b1z_diff_a, b1y_b1z_diff_a);
      
#ifdef HAVE_SSE3
      // now we have the squared norm of the vectors b0 - a and b1 - a in a single packed double
      const __m128d b0_b1_diff_a_norm = _mm_add_pd(_mm_hadd_pd(b0x_b0y_diff_a_sq, b1y_b1z_diff_a_sq), b0z_b1x_diff_a_sq);
#else
      const __m128d b0_b1_diff_a_norm = _mm_add_pd(_mm_add_pd(_mm_unpacklo_pd(b0x_b0y_diff_a_sq, b1y_b1z_diff_a_sq), _mm_unpackhi_pd(b0x_b0y_diff_a_sq, b1y_b1z_diff_a_sq)), b0z_b1x_diff_a_sq);
#endif
      
      // does the distance satisfy the matcher
      const uint32_t localSat0 = uint32_t(_mm_movemask_pd(_mm_and_pd(_mm_cmplt_pd(b0_b1_diff_a_norm, rMaxSq0), _mm_cmpgt_pd(b0_b1_diff_a_norm, rMinSq0))));
      const uint32_t localSat1 = uint32_t(_mm_movemask_pd(_mm_and_pd(_mm_cmplt_pd(b0_b1_diff_a_norm, rMaxSq1), _mm_cmpgt_pd(b0_b1_diff_a_norm, rMinSq1))));
      const uint32_t localSat2 = uint32_t(_mm_movemask_pd(_mm_and_pd(_mm_cmplt_pd(b0_b1_diff_a_norm, rMaxSq2), _mm_cmpgt_pd(b0_b1_diff_a_norm, rMinSq2))));
      const uint32_t localSat3 = uint32_t(_mm_movemask_pd(_mm_and_pd(_mm_cmplt_pd(b0_b1_diff_a_norm, rMaxSq3), _mm_cmpgt_pd(b0_b1_diff_a_norm, rMinSq3))));
      const uint32_t localSat4 = uint32_t(_mm_movemask_pd(_mm_and_pd(_mm_cmplt_pd(b0_b1_diff_a_norm, rMaxSq4), _mm_cmpgt_pd(b0_b1_diff_a_norm, rMinSq4))));
      const uint32_t localSat5 = uint32_t(_mm_movemask_pd(_mm_and_pd(_mm_cmplt_pd(b0_b1_diff_a_norm, rMaxSq5), _mm_cmpgt_pd(b0_b1_diff_a_norm, rMinSq5))));
      
      // pack the satisfiability results into the overall result
      // move over 2 locations because we're processing pairs of points at a time
      sat0 = (sat0 >> 2) | (uint64_t(localSat0) << 62);
      sat1 = (sat1 >> 2) | (uint64_t(localSat1) << 62);
      sat2 = (sat2 >> 2) | (uint64_t(localSat2) << 62);
      sat3 = (sat3 >> 2) | (uint64_t(localSat3) << 62);
      sat4 = (sat4 >> 2) | (uint64_t(localSat4) << 62);
      sat5 = (sat5 >> 2) | (uint64_t(localSat5) << 62);
    }

    // edge case for an odd number of points in B
    if ((numPointsB & 1) != 0) {
      const __m128d b0x_b0y = _mm_loadu_pd(&localPointsB[0].x);
      const __m128d b0z = _mm_load_sd(&localPointsB[0].z);
			
      const __m128d b0x_b0y_diff_a = _mm_sub_pd(b0x_b0y, ax_ay);
      const __m128d b0z_diff_a = _mm_sub_sd(b0z, az_ax);
      
      const __m128d b0x_b0y_diff_a_sq = _mm_mul_pd(b0x_b0y_diff_a, b0x_b0y_diff_a);
      const __m128d b0z_diff_a_sq = _mm_mul_sd(b0z_diff_a, b0z_diff_a);
      
#ifdef HAVE_SSE3
      const __m128d b0_diff_a_norm = _mm_add_sd(_mm_hadd_pd(b0x_b0y_diff_a_sq, _mm_setzero_pd()), b0z_diff_a_sq);
#else
      const __m128d b0_diff_a_norm = _mm_add_sd(_mm_add_sd(_mm_unpacklo_pd(b0x_b0y_diff_a_sq, _mm_setzero_pd()), _mm_unpackhi_pd(b0x_b0y_diff_a_sq, _mm_setzero_pd())), b0z_diff_a_sq);
#endif
      
      const uint32_t localSat0 = uint32_t(_mm_movemask_pd(_mm_and_pd(_mm_cmplt_sd(b0_diff_a_norm, rMaxSq0), _mm_cmpgt_sd(b0_diff_a_norm, rMinSq0))));
      const uint32_t localSat1 = uint32_t(_mm_movemask_pd(_mm_and_pd(_mm_cmplt_sd(b0_diff_a_norm, rMaxSq1), _mm_cmpgt_sd(b0_diff_a_norm, rMinSq1))));
      const uint32_t localSat2 = uint32_t(_mm_movemask_pd(_mm_and_pd(_mm_cmplt_sd(b0_diff_a_norm, rMaxSq2), _mm_cmpgt_sd(b0_diff_a_norm, rMinSq2))));
      const uint32_t localSat3 = uint32_t(_mm_movemask_pd(_mm_and_pd(_mm_cmplt_sd(b0_diff_a_norm, rMaxSq3), _mm_cmpgt_sd(b0_diff_a_norm, rMinSq3))));
      const uint32_t localSat4 = uint32_t(_mm_movemask_pd(_mm_and_pd(_mm_cmplt_sd(b0_diff_a_norm, rMaxSq4), _mm_cmpgt_sd(b0_diff_a_norm, rMinSq4))));
      const uint32_t localSat5 = uint32_t(_mm_movemask_pd(_mm_and_pd(_mm_cmplt_sd(b0_diff_a_norm, rMaxSq5), _mm_cmpgt_sd(b0_diff_a_norm, rMinSq5))));
      
      sat0 = (sat0 >> 1) | (uint64_t(localSat0) << 63);
      sat1 = (sat1 >> 1) | (uint64_t(localSat1) << 63);
      sat2 = (sat2 >> 1) | (uint64_t(localSat2) << 63);
      sat3 = (sat3 >> 1) | (uint64_t(localSat3) << 63);
      sat4 = (sat4 >> 1) | (uint64_t(localSat4) << 63);
      sat5 = (sat5 >> 1) | (uint64_t(localSat5) << 63);
    }
    // without sse, we just compute the distance and directly compare it to the matcher
#else
    for (int k = 0; k < numPointsB; k += 1) {
      const double3 b0 = *localPointsB++;
      const double3 b0a_diff = b0 - a;
      const double dist_b0a_sq = b0a_diff.x*b0a_diff.x + b0a_diff.y*b0a_diff.y + b0a_diff.z*b0a_diff.z;
      
      size_t localSat0 = ((dist_b0a_sq > rMinSq0) & (dist_b0a_sq < rMaxSq0));
      size_t localSat1 = ((dist_b0a_sq > rMinSq1) & (dist_b0a_sq < rMaxSq1));
      size_t localSat2 = ((dist_b0a_sq > rMinSq2) & (dist_b0a_sq < rMaxSq2));
      size_t localSat3 = ((dist_b0a_sq > rMinSq3) & (dist_b0a_sq < rMaxSq3));
      size_t localSat4 = ((dist_b0a_sq > rMinSq4) & (dist_b0a_sq < rMaxSq4));
      size_t localSat5 = ((dist_b0a_sq > rMinSq5) & (dist_b0a_sq < rMaxSq5));

      sat0 = (sat0 >> 1) | (uint64_t(localSat0) << 63);
      sat1 = (sat1 >> 1) | (uint64_t(localSat1) << 63);
      sat2 = (sat2 >> 1) | (uint64_t(localSat2) << 63);
      sat3 = (sat3 >> 1) | (uint64_t(localSat3) << 63);
      sat4 = (sat4 >> 1) | (uint64_t(localSat4) << 63);
      sat5 = (sat5 >> 1) | (uint64_t(localSat5) << 63);

    }
#endif
    // now store the results of comparing one point in A to all the points in B
		pairwiseInteractions[i * 6 + 0] = (sat0 >> (64 - numPointsB));
		pairwiseInteractions[i * 6 + 1] = (sat1 >> (64 - numPointsB));
		pairwiseInteractions[i * 6 + 2] = (sat2 >> (64 - numPointsB));
		pairwiseInteractions[i * 6 + 3] = (sat3 >> (64 - numPointsB));
		pairwiseInteractions[i * 6 + 4] = (sat4 >> (64 - numPointsB));
		pairwiseInteractions[i * 6 + 5] = (sat5 >> (64 - numPointsB));

	} // loop over points in A

}


void ComputeFourPointCorrelationCountsCPU(uint64_t *counts,
                                          NptRuntimes & /*nptRuntimes*/,
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
                                          const unsigned char */*satisfiabilityLUT*/)
{
  
  // D must be the array with the largest number of points
	if (numPointsA > numPointsD) {
		const double3 *__restrict pointsTemp = pointsA;
		int numPointsTemp = numPointsA;
		pointsA = pointsD;
		numPointsA = numPointsD;
		pointsD = pointsTemp;
		numPointsD = numPointsTemp;
	}
	if (numPointsB > numPointsD) {
		const double3 *__restrict pointsTemp = pointsB;
		int numPointsTemp = numPointsB;
		pointsB = pointsD;
		numPointsB = numPointsD;
		pointsD = pointsTemp;
		numPointsD = numPointsTemp;
	}
  if (numPointsC > numPointsD) {
		const double3 *__restrict pointsTemp = pointsC;
		int numPointsTemp = numPointsC;
		pointsC = pointsD;
		numPointsC = numPointsD;
		pointsD = pointsTemp;
		numPointsD = numPointsTemp;
	}

  
#ifdef HAVE_NEON
	int32x2_t pairwise_AB[64 * 6];
	int32x2_t pairwise_AC[64 * 6];
	int32x2_t pairwise_AD[64 * 6];
	int32x2_t pairwise_BC[64 * 6];
	int32x2_t pairwise_BD[64 * 6];
	int32x2_t pairwise_CD[64 * 6];
#else
	uint64_t pairwise_AB[64 * 6];
	uint64_t pairwise_AC[64 * 6];
	uint64_t pairwise_AD[64 * 6];
	uint64_t pairwise_BC[64 * 6];
	uint64_t pairwise_BD[64 * 6];
	uint64_t pairwise_CD[64 * 6];
#endif
  
	memset(counts, 0, sizeof(uint64_t));
	
	//double start = Timing::ElapsedTimeMs();
	memset(pairwise_AB, 0, sizeof(uint64_t)*64 * 6);
  memset(pairwise_AC, 0, sizeof(uint64_t)*64 * 6);
  memset(pairwise_AD, 0, sizeof(uint64_t)*64 * 6);
  memset(pairwise_BC, 0, sizeof(uint64_t)*64 * 6);
  memset(pairwise_BD, 0, sizeof(uint64_t)*64 * 6);
  memset(pairwise_CD, 0, sizeof(uint64_t)*64 * 6);
  
	PrecomputePairwiseInteractions4ptCPU((uint64_t*)pairwise_AB, counts,
                                       pointsA, numPointsA, pointsB, numPointsB,
                                       rMinSq, rMaxSq);
	PrecomputePairwiseInteractions4ptCPU((uint64_t*)pairwise_AC, counts,
                                       pointsA, numPointsA, pointsC, numPointsC,
                                       rMinSq, rMaxSq);
	PrecomputePairwiseInteractions4ptCPU((uint64_t*)pairwise_AD, counts,
                                       pointsA, numPointsA, pointsD, numPointsD,
                                       rMinSq, rMaxSq);
	PrecomputePairwiseInteractions4ptCPU((uint64_t*)pairwise_BC, counts,
                                       pointsB, numPointsB, pointsC, numPointsC,
                                       rMinSq, rMaxSq);
	PrecomputePairwiseInteractions4ptCPU((uint64_t*)pairwise_BD, counts,
                                       pointsB, numPointsB, pointsD, numPointsD,
                                       rMinSq, rMaxSq);
	PrecomputePairwiseInteractions4ptCPU((uint64_t*)pairwise_CD, counts,
                                       pointsC, numPointsC, pointsD, numPointsD,
                                       rMinSq, rMaxSq);
  // now, pairwiseAB should be an array of size 64
  // each element is a 64 bit value.  bit j of entry i represents whether or
  // not point i from set A and point j from set B satisfy the matcher
  
#ifndef HAVE_NEON
	size_t count3_addition = 0;
	for (int i = 0; i < numPointsA; i += 1) {
    
		uint64_t sat0AB_array = pairwise_AB[i*6 + 0];
		uint64_t sat1AB_array = pairwise_AB[i*6 + 1];
		uint64_t sat2AB_array = pairwise_AB[i*6 + 2];
		uint64_t sat3AB_array = pairwise_AB[i*6 + 3];
		uint64_t sat4AB_array = pairwise_AB[i*6 + 4];
		uint64_t sat5AB_array = pairwise_AB[i*6 + 5];
		
    for (int j = 0; j < numPointsB; j += 1) {

      // This grabs the 1 B we care about in this iteration
			const uint64_t sat0AB = -(sat0AB_array & 1);
			const uint64_t sat1AB = -(sat1AB_array & 1);
			const uint64_t sat2AB = -(sat2AB_array & 1);
    	const uint64_t sat3AB = -(sat3AB_array & 1);
    	const uint64_t sat4AB = -(sat4AB_array & 1);
    	const uint64_t sat5AB = -(sat5AB_array & 1);

      uint64_t sat0AC_array = pairwise_AC[i*6 + 0];
      uint64_t sat1AC_array = pairwise_AC[i*6 + 1];
      uint64_t sat2AC_array = pairwise_AC[i*6 + 2];
      uint64_t sat3AC_array = pairwise_AC[i*6 + 3];
      uint64_t sat4AC_array = pairwise_AC[i*6 + 4];
      uint64_t sat5AC_array = pairwise_AC[i*6 + 5];
      
      uint64_t sat0BC_array = pairwise_BC[j*6 + 0];
			uint64_t sat1BC_array = pairwise_BC[j*6 + 1];
			uint64_t sat2BC_array = pairwise_BC[j*6 + 2];
			uint64_t sat3BC_array = pairwise_BC[j*6 + 3];
			uint64_t sat4BC_array = pairwise_BC[j*6 + 4];
			uint64_t sat5BC_array = pairwise_BC[j*6 + 5];

      for (int k = 0; k < numPointsC; k++) {
      
        const uint64_t sat0AC = -(sat0AC_array & 1);
        const uint64_t sat1AC = -(sat1AC_array & 1);
        const uint64_t sat2AC = -(sat2AC_array & 1);
        const uint64_t sat3AC = -(sat3AC_array & 1);
        const uint64_t sat4AC = -(sat4AC_array & 1);
        const uint64_t sat5AC = -(sat5AC_array & 1);
        
        const uint64_t sat0BC = -(sat0BC_array & 1);
        const uint64_t sat1BC = -(sat1BC_array & 1);
        const uint64_t sat2BC = -(sat2BC_array & 1);
        const uint64_t sat3BC = -(sat3BC_array & 1);
        const uint64_t sat4BC = -(sat4BC_array & 1);
        const uint64_t sat5BC = -(sat5BC_array & 1);
        
        
        const uint64_t sat0AD = pairwise_AD[i*6 + 0];
        const uint64_t sat1AD = pairwise_AD[i*6 + 1];
        const uint64_t sat2AD = pairwise_AD[i*6 + 2];
        const uint64_t sat3AD = pairwise_AD[i*6 + 3];
        const uint64_t sat4AD = pairwise_AD[i*6 + 4];
        const uint64_t sat5AD = pairwise_AD[i*6 + 5];
        
        const uint64_t sat0BD = pairwise_BD[j*6 + 0];
        const uint64_t sat1BD = pairwise_BD[j*6 + 1];
        const uint64_t sat2BD = pairwise_BD[j*6 + 2];
        const uint64_t sat3BD = pairwise_BD[j*6 + 3];
        const uint64_t sat4BD = pairwise_BD[j*6 + 4];
        const uint64_t sat5BD = pairwise_BD[j*6 + 5];
        
        const uint64_t sat0CD = pairwise_CD[k*6 + 0];
        const uint64_t sat1CD = pairwise_CD[k*6 + 1];
        const uint64_t sat2CD = pairwise_CD[k*6 + 2];
        const uint64_t sat3CD = pairwise_CD[k*6 + 3];
        const uint64_t sat4CD = pairwise_CD[k*6 + 4];
        const uint64_t sat5CD = pairwise_CD[k*6 + 5];
        
			  
        //(sat001 & sat102 & sat203 & sat312 & sat413 & sat523)
        count3_addition += npt_popcount64(  (sat0AB & sat1AC & sat2AD & sat3BC & sat4BD & sat5CD) // ABCD
                                          | (sat0AB & sat1AD & sat2AC & sat3BD & sat4BC & sat5CD) // ABDC
                                          | (sat0AC & sat1AB & sat2AD & sat3BC & sat4CD & sat5BD) // ACBD
                                          | (sat0AC & sat1AD & sat2AB & sat3CD & sat4BC & sat5BD) // ACDB
                                          | (sat0AD & sat1AB & sat2AC & sat3BD & sat4CD & sat5BC) // ADBC
                                          | (sat0AD & sat1AC & sat2AB & sat3CD & sat4BD & sat5BC) // ADCB
                                          
                                          | (sat0AB & sat1BC & sat2BD & sat3AC & sat4AD & sat5CD) // BACD
                                          | (sat0AB & sat1BD & sat2BC & sat3AD & sat4AC & sat5CD) // BADC
                                          | (sat0BC & sat1AB & sat2BD & sat3AC & sat4CD & sat5AD) // BCAD
                                          | (sat0BC & sat1BD & sat2AB & sat3CD & sat4AC & sat5AD) // BCDA
                                          | (sat0BD & sat1AB & sat2BC & sat3AD & sat4CD & sat5AC) // BDAC
                                          | (sat0BD & sat1BC & sat2AB & sat3CD & sat4AD & sat5AC) // BDCA
                                          
                                          | (sat0BC & sat1AC & sat2CD & sat3AB & sat4BD & sat5AD) // CBAD
                                          | (sat0BC & sat1CD & sat2AC & sat3BD & sat4AB & sat5AD) // CBDA
                                          | (sat0AC & sat1BC & sat2CD & sat3AB & sat4AD & sat5BD) // CABD
                                          | (sat0AC & sat1CD & sat2BC & sat3AD & sat4AB & sat5BD) // CADB
                                          | (sat0CD & sat1BC & sat2AC & sat3BD & sat4AD & sat5AB) // CDBA
                                          | (sat0CD & sat1AC & sat2BC & sat3AD & sat4BD & sat5AB) // CDAB
                                          
                                          | (sat0BD & sat1CD & sat2AD & sat3BC & sat4AB & sat5AC) // DBCA
                                          | (sat0BD & sat1AD & sat2CD & sat3AB & sat4BC & sat5AC) // DBAC
                                          | (sat0CD & sat1BD & sat2AD & sat3BC & sat4AC & sat5AB) // DCBA
                                          | (sat0CD & sat1AD & sat2BD & sat3AC & sat4BC & sat5AB) // DCAB
                                          | (sat0AD & sat1BD & sat2CD & sat3AB & sat4AC & sat5BC) // DABC
                                          | (sat0AD & sat1CD & sat2BD & sat3AC & sat4AB & sat5BC) // DACB
                                          );

/*
        ((sat4BD & sat5CD) | (sat5BD & sat4CD))
        
        ((sat3BD & sat5CD) | (sat5BD & sat3CD))
        
        ((sat2BD & sat5CD) | (sat5BD & sat2CD))
        
        ((sat1BD & sat5CD) | (sat5BD & sat1CD))
        
        ((sat3BD & sat4CD) | (sat4BD & sat3CD))
        
        ((sat2BD & sat4CD) | (sat4BD & sat2CD))

        ((sat1BD & sat4CD) | (sat4BD & sat1CD))

        ((sat2BD & sat3CD) | (sat3BD & sat2CD))
        
        ((sat1BD & sat3CD) | (sat3BD & sat1CD))
        
        ((sat1BD & sat2CD) | (sat2BD & sat1CD))
        
        
        ((sat3BC & ((sat4BD & sat5CD) | (sat5BD & sat4CD))) |
         (sat2BC & ((sat4BD & sat5CD) | (sat5BD & sat4CD))) |
         (sat1BC & ((sat4BD & sat5CD) | (sat5BD & sat4CD))))

     */
        //count3_addition += npt_popcount64((sat0AB & ((sat1AC & sat2BC) | (sat2AC & sat1BC))) |
        //                                  (sat1AB & ((sat0AC & sat2BC) | (sat2AC & sat0BC))) |
        //                                  (sat2AB & ((sat0AC & sat1BC) | (sat1AC & sat0BC))));
        
        sat0BC_array >>= 1;
        sat1BC_array >>= 1;
        sat2BC_array >>= 1;
        sat3BC_array >>= 1;
        sat4BC_array >>= 1;
        sat5BC_array >>= 1;

        sat0AC_array >>= 1;
        sat1AC_array >>= 1;
        sat2AC_array >>= 1;
        sat3AC_array >>= 1;
        sat4AC_array >>= 1;
        sat5AC_array >>= 1;
        
      } // for k
      
      sat0AB_array >>= 1;
      sat1AB_array >>= 1;
      sat2AB_array >>= 1;
      sat3AB_array >>= 1;
      sat4AB_array >>= 1;
      sat5AB_array >>= 1;

		} // for j

	} // for i

	counts[0] = count3_addition;

#else
/*
 const int32x2_t one = vcreate_s32(1);
	uint32x2_t count3_addition = vdup_n_u32(0);
	for (int i = 0; i < numPointsA; i += 1) {
		uint16x4_t count3_subaddition = vdup_n_u16(0);
		int32x2_t sat0AB_array = pairwise_AB[i*3 + 0];
		int32x2_t sat1AB_array = pairwise_AB[i*3 + 1];
		int32x2_t sat2AB_array = pairwise_AB[i*3 + 2];
		for (int j = 0; j < numPointsB; j += 1) {
			const int32x2_t sat0AB = vdup_lane_s32(vneg_s32(vand_s32(sat0AB_array, one)), 0);
			const int32x2_t sat1AB = vdup_lane_s32(vneg_s32(vand_s32(sat1AB_array, one)), 0);
			const int32x2_t sat2AB = vdup_lane_s32(vneg_s32(vand_s32(sat2AB_array, one)), 0);
      
			const int32x2_t sat0AC = pairwise_AC[i*3 + 0];
			const int32x2_t sat1AC = pairwise_AC[i*3 + 1];
			const int32x2_t sat2AC = pairwise_AC[i*3 + 2];
      
			const int32x2_t sat0BC = pairwise_BC[j*3 + 0];
			const int32x2_t sat1BC = pairwise_BC[j*3 + 1];
			const int32x2_t sat2BC = pairwise_BC[j*3 + 2];
      
			count3_subaddition = vadd_u16(
                                    count3_subaddition,
                                    vpaddl_u8(vcnt_u8(vreinterpret_u8_s32(
                                                                          vorr_s32(
                                                                                   vorr_s32(
                                                                                            vand_s32(sat0AB, vorr_s32(vand_s32(sat1AC, sat2BC), vand_s32(sat2AC, sat1BC))),
                                                                                            vand_s32(sat1AB, vorr_s32(vand_s32(sat0AC, sat2BC), vand_s32(sat2AC, sat0BC)))
                                                                                            ),
                                                                                   vand_s32(sat2AB, vorr_s32(vand_s32(sat0AC, sat1BC), vand_s32(sat1AC, sat0BC)))
                                                                                   )
                                                                          )))
                                    );
			sat0AB_array = vreinterpret_s32_u64(vshr_n_u64(vreinterpret_u64_s32(sat0AB_array), 1));
			sat1AB_array = vreinterpret_s32_u64(vshr_n_u64(vreinterpret_u64_s32(sat1AB_array), 1));
			sat2AB_array = vreinterpret_s32_u64(vshr_n_u64(vreinterpret_u64_s32(sat2AB_array), 1));
		}
		count3_addition = vadd_u32(count3_addition, vpaddl_u16(count3_subaddition));
	}
	counts[3] = vget_lane_u32(vreinterpret_u32_u64(vpaddl_u32(count3_addition)), 0);
 */
#endif

}

// For now, we assume that the matchers have been repackaged here
static void PrecomputePairwiseInteractionsMulti4ptCPU(uint64_t **__restrict pairwiseInteractions,
                                                      uint64_t **/*counts*/,
                                                      const double3 *__restrict pointsA,
                                                      int numPointsA,
                                                      const double3 *__restrict pointsB,
                                                      int numPointsB,
                                                      double **__restrict rMinSq,
                                                      double **__restrict rMaxSq,
                                                      int numMatchers)
{
  /*
  // put this out here so we don't have to keep reallocating it
  uint64_t* sat0 = new uint64_t[numMatchers];
  
  for (int i = 0; i < numPointsA; i += 1) {
#ifdef HAVE_SSE2
    const __m128d ax_ay = _mm_loadu_pd(static_cast<const double*>(&pointsA[i].x));
    const __m128d ay_az = _mm_loadu_pd(static_cast<const double*>(&pointsA[i].y));
    const __m128d az_ax = _mm_shuffle_pd(ay_az, ax_ay, 1);
#else
    const double3 a = pointsA[i];
#endif
		const double3* localPointsB = pointsB;
		
    memset(sat0, 0, sizeof(uint64_t)*numMatchers);
    
#ifdef HAVE_SSE2
    for (int k = 1; k < numPointsB; k += 2) {
      const __m128d b0x_b0y = _mm_loadu_pd(&localPointsB[0].x);
      const __m128d b0z_b1x = _mm_loadu_pd(&localPointsB[0].z);
      const __m128d b1y_b1z = _mm_loadu_pd(&localPointsB[1].y);
      localPointsB += 2;
			
      // compute b - a
      const __m128d b0x_b0y_diff_a = _mm_sub_pd(b0x_b0y, ax_ay);
      const __m128d b0z_b1x_diff_a = _mm_sub_pd(b0z_b1x, az_ax);
      const __m128d b1y_b1z_diff_a = _mm_sub_pd(b1y_b1z, ay_az);
      
      // compute (b - a)^2
      const __m128d b0x_b0y_diff_a_sq = _mm_mul_pd(b0x_b0y_diff_a, b0x_b0y_diff_a);
      const __m128d b0z_b1x_diff_a_sq = _mm_mul_pd(b0z_b1x_diff_a, b0z_b1x_diff_a);
      const __m128d b1y_b1z_diff_a_sq = _mm_mul_pd(b1y_b1z_diff_a, b1y_b1z_diff_a);
      
#ifdef HAVE_SSE3
      // now we have the squared norm of the vectors b0 - a and b1 - a in a single packed double
      const __m128d b0_b1_diff_a_norm = _mm_add_pd(_mm_hadd_pd(b0x_b0y_diff_a_sq, b1y_b1z_diff_a_sq), b0z_b1x_diff_a_sq);
#else
      const __m128d b0_b1_diff_a_norm = _mm_add_pd(_mm_add_pd(_mm_unpacklo_pd(b0x_b0y_diff_a_sq, b1y_b1z_diff_a_sq), _mm_unpackhi_pd(b0x_b0y_diff_a_sq, b1y_b1z_diff_a_sq)), b0z_b1x_diff_a_sq);
#endif
      
      for (int matcher_ind = 0; matcher_ind < numMatchers; matcher_ind++)
      {
        const __m128d rMinSq0 = npt_mm_loaddup_pd(&rMinSq[matcher_ind][0]);
        const __m128d rMaxSq0 = npt_mm_loaddup_pd(&rMaxSq[matcher_ind][0]);
        
        // does the distance satisfy the matcher in position 0,1,2
        const uint32_t localSat0 = uint32_t(_mm_movemask_pd(_mm_and_pd(_mm_cmplt_pd(b0_b1_diff_a_norm, rMaxSq0), _mm_cmpgt_pd(b0_b1_diff_a_norm, rMinSq0))));
        // pack the satisfiability results into the overall result
        sat0[matcher_ind] = (sat0[matcher_ind] >> 2) | (uint64_t(localSat0) << 62);
      } // loop over matchers
      
    } // loop over points in B
      // edge case for an odd number of points in B
    if ((numPointsB & 1) != 0) {
      const __m128d b0x_b0y = _mm_loadu_pd(&localPointsB[0].x);
      const __m128d b0z = _mm_load_sd(&localPointsB[0].z);
			
      const __m128d b0x_b0y_diff_a = _mm_sub_pd(b0x_b0y, ax_ay);
      const __m128d b0z_diff_a = _mm_sub_sd(b0z, az_ax);
      
      const __m128d b0x_b0y_diff_a_sq = _mm_mul_pd(b0x_b0y_diff_a, b0x_b0y_diff_a);
      const __m128d b0z_diff_a_sq = _mm_mul_sd(b0z_diff_a, b0z_diff_a);
      
#ifdef HAVE_SSE3
      const __m128d b0_diff_a_norm = _mm_add_sd(_mm_hadd_pd(b0x_b0y_diff_a_sq, _mm_setzero_pd()), b0z_diff_a_sq);
#else
      const __m128d b0_diff_a_norm = _mm_add_sd(_mm_add_sd(_mm_unpacklo_pd(b0x_b0y_diff_a_sq, _mm_setzero_pd()), _mm_unpackhi_pd(b0x_b0y_diff_a_sq, _mm_setzero_pd())), b0z_diff_a_sq);
#endif
      
      for (int matcher_ind = 0; matcher_ind < numMatchers; matcher_ind++)
      {
        const __m128d rMinSq0 = npt_mm_loaddup_pd(&rMinSq[matcher_ind][0]);
        const __m128d rMaxSq0 = npt_mm_loaddup_pd(&rMaxSq[matcher_ind][0]);
        
        const uint32_t localSat0 = uint32_t(_mm_movemask_pd(_mm_and_pd(_mm_cmplt_sd(b0_diff_a_norm, rMaxSq0), _mm_cmpgt_sd(b0_diff_a_norm, rMinSq0))));
        
        sat0[matcher_ind] = (sat0[matcher_ind] >> 1) | (uint64_t(localSat0) << 63);
      } // loop over matchers
      
    }
    // without sse, we just compute the distance and directly compare it to the matcher
#else
    for (int k = 0; k < numPointsB; k += 1) {
      const double3 b0 = *localPointsB++;
      const double3 b0a_diff = b0 - a;
      const double dist_b0a_sq = b0a_diff.x*b0a_diff.x + b0a_diff.y*b0a_diff.y + b0a_diff.z*b0a_diff.z;
      
      for (int matcher_ind = 0; matcher_ind < numMatchers; matcher_ind++)
      {
        
        const double rMinSq0 = rMinSq[matcher_ind][0];
        const double rMaxSq0 = rMaxSq[matcher_ind][0];
        
        size_t localSat0 = ((dist_b0a_sq > rMinSq0) & (dist_b0a_sq < rMaxSq0));
        sat0[matcher_ind] = (sat0[matcher_ind] >> 1) | (uint64_t(localSat0) << 63);
      } // loop over matchers
      
    } // loop over points in B
#endif
    // now store the results of comparing one point in A to all the points in B
		
    for (int matcher_ind = 0; matcher_ind < numMatchers; matcher_ind++)
    {
      pairwiseInteractions[matcher_ind][i] = (sat0[matcher_ind] >> (64 - numPointsB));
    } // loop over matchers
    
  } // loop over points in A
  
  delete sat0;
  */
}


void ComputeFourPointCorrelationCountsMultiCPU(
                                              uint64_t **__restrict counts,
                                              NptRuntimes & /*nptRuntimes*/,
                                              const double3 *__restrict pointsA,
                                              int numPointsA,
                                              const double3 *__restrict pointsB,
                                              int numPointsB,
                                              double **__restrict rMinSq,
                                              double **__restrict rMaxSq,
                                              int numMatchers,
                                              const unsigned char *__restrict /*satisfiabilityLUT*/)
{
	/*
  // C must be the array with the largest number of points
	if (numPointsA > numPointsB) {
		const double3 *__restrict pointsTemp = pointsA;
		int numPointsTemp = numPointsA;
		pointsA = pointsB;
		numPointsA = numPointsB;
		pointsB = pointsTemp;
		numPointsB = numPointsTemp;
	}
	
#ifdef HAVE_NEON
  int32x2_t** pairwise_AB = new int32x2_t*[numMatchers];
  for (int i = 0; i < numMatchers; i++)
  {
    pairwise_AB[i] = new int32x2_t[64];
  }
#else
  //std::cout << "allocating pairwise stuff.\n";
  uint64_t** pairwise_AB = new uint64_t*[numMatchers];
  
  for (int i = 0; i < numMatchers; i++) {
    pairwise_AB[i] = new uint64_t[64];
  }
#endif
  
  // zero out the values
  for (int i = 0; i < numMatchers; i++) {
    memset(counts[i], 0, sizeof(uint64_t));
    memset(pairwise_AB[i], 0, sizeof(uint64_t)*64);
  }
	
	//double start = Timing::ElapsedTimeMs();
	PrecomputePairwiseInteractionsMulti4ptCPU(pairwise_AB, counts,
                                            pointsA, numPointsA,
                                            pointsB, numPointsB,
                                            rMinSq, rMaxSq, numMatchers);
	//double elapsedPairwise = Timing::ElapsedTimeMs() - start;
  
  //std::cout << "Done with pairwise stuff\n";
  
#ifndef HAVE_NEON
  
  for (int matcher_ind = 0; matcher_ind < numMatchers; matcher_ind++)
  {
    size_t count3_addition = 0;
    for (int i = 0; i < numPointsA; i += 1) {
      uint64_t sat0AB_array = pairwise_AB[matcher_ind][i];
      
      size_t this_add = npt_popcount64(sat0AB_array);
      //std::cout << "this popcount " << this_add << "\n";
      //count3_addition += npt_popcount64(sat0AB_array);
      count3_addition += this_add;
      
    }
    counts[matcher_ind][0] = count3_addition;
  } // loop over matchers and do population counts
#else
	const int32x2_t one = vcreate_s32(1);
  
  for (int matcher_ind = 0; matcher_ind < numMatchers; matcher_ind++)
  {
    uint32x2_t count3_addition = vdup_n_u32(0);
    for (int i = 0; i < numPointsA; i += 1) {
      uint16x4_t count3_subaddition = vdup_n_u16(0);
      int32x2_t sat0AB_array = pairwise_AB[matcher_ind][i];
      const int32x2_t sat0AB = vdup_lane_s32(vneg_s32(vand_s32(sat0AB_array, one)), 0);
      
      count3_subaddition = vadd_u16(
                                    count3_subaddition,
                                    vpaddl_u8(vcnt_u8(vreinterpret_u8_s32(sat0AB
                                                                          )))
                                    );
      sat0AB_array = vreinterpret_s32_u64(vshr_n_u64(vreinterpret_u64_s32(sat0AB_array), 1));
      count3_addition = vadd_u32(count3_addition, vpaddl_u16(count3_subaddition));
    }
    counts[matcher_ind][0] = vget_lane_u32(vreinterpret_u32_u64(vpaddl_u32(count3_addition)), 0);
  } // loop over matchers
#endif
	//double elapsed = Timing::ElapsedTimeMs() - start;
  
	//nptRuntimes.twoWay = elapsedPairwise;
	//nptRuntimes.threeWay = elapsed-elapsedPairwise;
  
  // free memory
  for (int i = 0; i < numMatchers; i++) {
    delete pairwise_AB[i];
  }
  delete pairwise_AB;
  */
} 





