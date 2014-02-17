//
//  kernel_2pt_cpu.cpp
//  contrib_march
//
//  Created by William March on 11/1/12.
//
//

#include "kernel_2pt_cpu.hpp"

static void PrecomputePairwiseInteractions2ptCPU(uint64_t *__restrict pairwiseInteractions,
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
#else
  const double rMinSq0 = rMinSq[0];
  const double rMaxSq0 = rMaxSq[0];
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
      // pack the satisfiability results into the overall result
      // move over 2 locations because we're processing pairs of points at a time
      sat0 = (sat0 >> 2) | (uint64_t(localSat0) << 62);
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
      sat0 = (sat0 >> 1) | (uint64_t(localSat0) << 63);
    }
    // without sse, we just compute the distance and directly compare it to the matcher
#else
    for (int k = 0; k < numPointsB; k += 1) {
      const double3 b0 = *localPointsB++;
      const double3 b0a_diff = b0 - a;
      const double dist_b0a_sq = b0a_diff.x*b0a_diff.x + b0a_diff.y*b0a_diff.y + b0a_diff.z*b0a_diff.z;
      
      size_t localSat0 = ((dist_b0a_sq > rMinSq0) & (dist_b0a_sq < rMaxSq0));
      sat0 = (sat0 >> 1) | (uint64_t(localSat0) << 63);
    }
#endif
    // now store the results of comparing one point in A to all the points in B
		pairwiseInteractions[i] = (sat0 >> (64 - numPointsB));
	} // loop over points in A
}


void ComputeTwoPointCorrelationCountsCPU(uint64_t *counts,
                                         NptRuntimes & /*nptRuntimes*/,
                                         const double3 *pointsA,
                                         int numPointsA,
                                         const double3 *pointsB,
                                         int numPointsB,
                                         const double *rMinSq,
                                         const double *rMaxSq,
                                         const unsigned char */*satisfiabilityLUT*/)
{
  
  // B must be the array with the largest number of points
	if (numPointsA > numPointsB) {
		const double3 *__restrict pointsTemp = pointsA;
		int numPointsTemp = numPointsA;
		pointsA = pointsB;
		numPointsA = numPointsB;
		pointsB = pointsTemp;
		numPointsB = numPointsTemp;
	}
  
#ifdef HAVE_NEON
	int32x2_t pairwise_AB[64];
#else
	uint64_t pairwise_AB[64];
#endif
  
	memset(counts, 0, sizeof(uint64_t));
	
	//double start = Timing::ElapsedTimeMs();
	memset(pairwise_AB, 0, sizeof(uint64_t)*64);

	PrecomputePairwiseInteractions2ptCPU((uint64_t*)pairwise_AB, counts,
                                    pointsA, numPointsA, pointsB, numPointsB,
                                    rMinSq, rMaxSq);
  // now, pairwiseAB should be an array of size 64
  // each element is a 64 bit value.  bit j of entry i represents whether or
  // not point i from set A and point j from set B satisfy the matcher
  
#ifndef HAVE_NEON
	size_t count3_addition = 0;
	for (int i = 0; i < numPointsA; i += 1) {
    uint64_t sat0AB_array = pairwise_AB[i];
    //const uint64_t sat0AB = -(sat0AB_array & 1);
    count3_addition += npt_popcount64(sat0AB_array);
  }
	*counts = count3_addition;
#else
	const int32x2_t one = vcreate_s32(1);
	uint32x2_t count3_addition = vdup_n_u32(0);
	for (int i = 0; i < numPointsA; i += 1) {
    uint16x4_t count3_subaddition = vdup_n_u16(0);
    int32x2_t sat0AB_array = pairwise_AB[i];
      
    const int32x2_t sat0AB = vdup_lane_s32(vneg_s32(vand_s32(sat0AB_array, one)), 0);
      
    count3_subaddition = vadd_u16(
                                  count3_subaddition,
                                  vpaddl_u8(vcnt_u8(vreinterpret_u8_s32(sat0AB
                                                                        )))
                                  );
    sat0AB_array = vreinterpret_s32_u64(vshr_n_u64(vreinterpret_u64_s32(sat0AB_array), 1));
    count3_addition = vadd_u32(count3_addition, vpaddl_u16(count3_subaddition));
	} // loop over points in A
	*counts = vget_lane_u32(vreinterpret_u32_u64(vpaddl_u32(count3_addition)), 0);
#endif
	//double elapsed = Timing::ElapsedTimeMs() - start;

}

// For now, we assume that the matchers have been repackaged here
static void PrecomputePairwiseInteractionsMulti2ptCPU(uint64_t **__restrict pairwiseInteractions,
                                                   uint64_t **/*counts*/,
                                                   const double3 *__restrict pointsA,
                                                   int numPointsA,
                                                   const double3 *__restrict pointsB,
                                                   int numPointsB,
                                                   double **__restrict rMinSq,
                                                   double **__restrict rMaxSq,
                                                   int numMatchers)
{
  
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
  
}


void ComputeTwoPointCorrelationCountsMultiCPU(
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
	PrecomputePairwiseInteractionsMulti2ptCPU(pairwise_AB, counts,
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
  
} 





