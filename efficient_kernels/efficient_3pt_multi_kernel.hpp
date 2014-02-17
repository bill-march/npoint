/*
 * efficient_3pt_multi_kernel.hpp
 * @author Bill March (march@gatech.edu)
 *
 * Implementations of the efficient CPU kernel code for multiple 3-point 
 * matchers.
 */

#ifndef _NPOINT_MLPACK_EFFICIENT_KERNELS_EFFICIENT_3PT_MULTI_KERNEL_
#define _NPOINT_MLPACK_EFFICIENT_KERNELS_EFFICIENT_3PT_MULTI_KERNEL_

// For now, we assume that the matchers have been repackaged here
static void PrecomputePairwiseInteractionsMultiCPU(
                                              uint64_t **__restrict pairwiseInteractions,
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
  uint64_t* sat1 = new uint64_t[numMatchers];
  uint64_t* sat2 = new uint64_t[numMatchers];
  
  
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
    memset(sat1, 0, sizeof(uint64_t)*numMatchers);
    memset(sat2, 0, sizeof(uint64_t)*numMatchers);
    
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
        const __m128d rMinSq1 = npt_mm_loaddup_pd(&rMinSq[matcher_ind][1]);
        const __m128d rMinSq2 = npt_mm_loaddup_pd(&rMinSq[matcher_ind][2]);
        const __m128d rMaxSq0 = npt_mm_loaddup_pd(&rMaxSq[matcher_ind][0]);
        const __m128d rMaxSq1 = npt_mm_loaddup_pd(&rMaxSq[matcher_ind][1]);
        const __m128d rMaxSq2 = npt_mm_loaddup_pd(&rMaxSq[matcher_ind][2]);
        
        // does the distance satisfy the matcher in position 0,1,2
        const uint32_t localSat0 = uint32_t(_mm_movemask_pd(_mm_and_pd(_mm_cmplt_pd(b0_b1_diff_a_norm, rMaxSq0), _mm_cmpgt_pd(b0_b1_diff_a_norm, rMinSq0))));
        const uint32_t localSat1 = uint32_t(_mm_movemask_pd(_mm_and_pd(_mm_cmplt_pd(b0_b1_diff_a_norm, rMaxSq1), _mm_cmpgt_pd(b0_b1_diff_a_norm, rMinSq1))));
        const uint32_t localSat2 = uint32_t(_mm_movemask_pd(_mm_and_pd(_mm_cmplt_pd(b0_b1_diff_a_norm, rMaxSq2), _mm_cmpgt_pd(b0_b1_diff_a_norm, rMinSq2))));
        // pack the satisfiability results into the overall result
        sat0[matcher_ind] = (sat0[matcher_ind] >> 2) | (uint64_t(localSat0) << 62);
        sat1[matcher_ind] = (sat1[matcher_ind] >> 2) | (uint64_t(localSat1) << 62);
        sat2[matcher_ind] = (sat2[matcher_ind] >> 2) | (uint64_t(localSat2) << 62);
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
        const __m128d rMinSq1 = npt_mm_loaddup_pd(&rMinSq[matcher_ind][1]);
        const __m128d rMinSq2 = npt_mm_loaddup_pd(&rMinSq[matcher_ind][2]);
        const __m128d rMaxSq0 = npt_mm_loaddup_pd(&rMaxSq[matcher_ind][0]);
        const __m128d rMaxSq1 = npt_mm_loaddup_pd(&rMaxSq[matcher_ind][1]);
        const __m128d rMaxSq2 = npt_mm_loaddup_pd(&rMaxSq[matcher_ind][2]);

        const uint32_t localSat0 = uint32_t(_mm_movemask_pd(_mm_and_pd(_mm_cmplt_sd(b0_diff_a_norm, rMaxSq0), _mm_cmpgt_sd(b0_diff_a_norm, rMinSq0))));
        const uint32_t localSat1 = uint32_t(_mm_movemask_pd(_mm_and_pd(_mm_cmplt_sd(b0_diff_a_norm, rMaxSq1), _mm_cmpgt_sd(b0_diff_a_norm, rMinSq1))));
        const uint32_t localSat2 = uint32_t(_mm_movemask_pd(_mm_and_pd(_mm_cmplt_sd(b0_diff_a_norm, rMaxSq2), _mm_cmpgt_sd(b0_diff_a_norm, rMinSq2))));
        
        sat0[matcher_ind] = (sat0[matcher_ind] >> 1) | (uint64_t(localSat0) << 63);
        sat1[matcher_ind] = (sat1[matcher_ind] >> 1) | (uint64_t(localSat1) << 63);
        sat2[matcher_ind] = (sat2[matcher_ind] >> 1) | (uint64_t(localSat2) << 63);
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
        
        const double rMinSq0 = rMinSq[0], rMinSq1 = rMinSq[1], rMinSq2 = rMinSq[2];
        const double rMaxSq0 = rMaxSq[0], rMaxSq1 = rMaxSq[1], rMaxSq2 = rMaxSq[2];

        size_t localSat0 = ((dist_b0a_sq > rMinSq0) & (dist_b0a_sq < rMaxSq0));
        size_t localSat1 = ((dist_b0a_sq > rMinSq1) & (dist_b0a_sq < rMaxSq1));
        size_t localSat2 = ((dist_b0a_sq > rMinSq2) & (dist_b0a_sq < rMaxSq2));
        sat0[matcher_ind] = (sat0[matcher_ind] >> 1) | (uint64_t(localSat0) << 63);
        sat1[matcher_ind] = (sat1[matcher_ind] >> 1) | (uint64_t(localSat1) << 63);
        sat2[matcher_ind] = (sat2[matcher_ind] >> 1) | (uint64_t(localSat2) << 63);
      } // loop over matchers
      
    }
#endif
    // now store the results of comparing one point in A to all the points in B
		
    for (int matcher_ind = 0; matcher_ind < numMatchers; matcher_ind++)
    {
      pairwiseInteractions[matcher_ind][i * 3 + 0] = (sat0[matcher_ind] >> (64 - numPointsB));
      pairwiseInteractions[matcher_ind][i * 3 + 1] = (sat1[matcher_ind] >> (64 - numPointsB));
      pairwiseInteractions[matcher_ind][i * 3 + 2] = (sat2[matcher_ind] >> (64 - numPointsB));
    } // loop over matchers
    
  } // loop over points in A

  delete sat0;
  delete sat1;
  delete sat2;
  
}

void ComputeThreePointCorrelationCountsMultiCPU(
                                           uint64_t **__restrict counts,
                                           NptRuntimes & /*nptRuntimes*/,
                                           const double3 *__restrict pointsA,
                                           int numPointsA,
                                           const double3 *__restrict pointsB,
                                           int numPointsB,
                                           const double3 *__restrict pointsC,
                                           int numPointsC,
                                           double **__restrict rMinSq,
                                           double **__restrict rMaxSq,
                                                int numMatchers,
                                           const unsigned char *__restrict /*satisfiabilityLUT*/)
{
	// C must be the array with the largest number of points
	if (numPointsA > numPointsC) {
		const double3 *__restrict pointsTemp = pointsA;
		int numPointsTemp = numPointsA;
		pointsA = pointsC;
		numPointsA = numPointsC;
		pointsC = pointsTemp;
		numPointsC = numPointsTemp;
	}
	if (numPointsB > numPointsC) {
		const double3 *__restrict pointsTemp = pointsB;
		int numPointsTemp = numPointsB;
		pointsB = pointsC;
		numPointsB = numPointsC;
		pointsC = pointsTemp;
		numPointsC = numPointsTemp;
	}
  
#ifdef HAVE_NEON
  std::cout << "allocating pairwise stuff neon.\n";
  int32x2_t** pairwise_AB = new int32x2_t*[numMatchers];
  int32x2_t** pairwise_AC = new int32x2_t*[numMatchers];
  int32x2_t** pairwise_BC = new int32x2_t*[numMatchers];
  for (int i = 0; i < numMatchers; i++)
  {
    pairwise_AB[i] = new int32x2_t[64 * 3];
    pairwise_AC[i] = new int32x2_t[64 * 3];
    pairwise_BC[i] = new int32x2_t[64 * 3];
  }
#else
  //std::cout << "allocating pairwise stuff.\n";
  uint64_t** pairwise_AB = new uint64_t*[numMatchers];
  uint64_t** pairwise_AC = new uint64_t*[numMatchers];
  uint64_t** pairwise_BC = new uint64_t*[numMatchers];
  
  for (int i = 0; i < numMatchers; i++) {
    pairwise_AB[i] = new uint64_t[64*3];
    pairwise_AC[i] = new uint64_t[64*3];
    pairwise_BC[i] = new uint64_t[64*3];
  }
#endif
  
  // zero out the values
  for (int i = 0; i < numMatchers; i++) {
    memset(counts[i], 0, sizeof(uint64_t)*4);
    memset(pairwise_AB[i], 0, sizeof(uint64_t)*64*3);
    memset(pairwise_AC[i], 0, sizeof(uint64_t)*64*3);
    memset(pairwise_BC[i], 0, sizeof(uint64_t)*64*3);
  }
	
	//double start = Timing::ElapsedTimeMs();
	PrecomputePairwiseInteractionsMultiCPU(pairwise_AB, counts, pointsA, numPointsA, pointsB, numPointsB, rMinSq, rMaxSq, numMatchers);
	PrecomputePairwiseInteractionsMultiCPU(pairwise_AC, counts, pointsA, numPointsA, pointsC, numPointsC, rMinSq, rMaxSq, numMatchers);
	PrecomputePairwiseInteractionsMultiCPU(pairwise_BC, counts, pointsB, numPointsB, pointsC, numPointsC, rMinSq, rMaxSq, numMatchers);
	//double elapsedPairwise = Timing::ElapsedTimeMs() - start;
  
  //std::cout << "Done with pairwise stuff\n";
  
#ifndef HAVE_NEON
  
  for (int matcher_ind = 0; matcher_ind < numMatchers; matcher_ind++)
  {
    size_t count3_addition = 0;
    for (int i = 0; i < numPointsA; i += 1) {
      uint64_t sat0AB_array = pairwise_AB[matcher_ind][i*3 + 0];
      uint64_t sat1AB_array = pairwise_AB[matcher_ind][i*3 + 1];
      uint64_t sat2AB_array = pairwise_AB[matcher_ind][i*3 + 2];
      for (int j = 0; j < numPointsB; j += 1) {
        const uint64_t sat0AB = -(sat0AB_array & 1);
        const uint64_t sat1AB = -(sat1AB_array & 1);
        const uint64_t sat2AB = -(sat2AB_array & 1);
        
        const uint64_t sat0AC = pairwise_AC[matcher_ind][i*3 + 0];
        const uint64_t sat1AC = pairwise_AC[matcher_ind][i*3 + 1];
        const uint64_t sat2AC = pairwise_AC[matcher_ind][i*3 + 2];
        
        const uint64_t sat0BC = pairwise_BC[matcher_ind][j*3 + 0];
        const uint64_t sat1BC = pairwise_BC[matcher_ind][j*3 + 1];
        const uint64_t sat2BC = pairwise_BC[matcher_ind][j*3 + 2];
        
        count3_addition += npt_popcount64((sat0AB & ((sat1AC & sat2BC) | (sat2AC & sat1BC))) |
                                          (sat1AB & ((sat0AC & sat2BC) | (sat2AC & sat0BC))) |
                                          (sat2AB & ((sat0AC & sat1BC) | (sat1AC & sat0BC))));
        
        sat0AB_array  >>= 1;
        sat1AB_array  >>= 1;
        sat2AB_array  >>= 1;
      }
    }
    counts[matcher_ind][3] = count3_addition;
  } // loop over matchers and do population counts
#else
	const int32x2_t one = vcreate_s32(1);
  
  for (int matcher_ind = 0; matcher_ind < numMatchers; matcher_ind++)
  {
    uint32x2_t count3_addition = vdup_n_u32(0);
    for (int i = 0; i < numPointsA; i += 1) {
      uint16x4_t count3_subaddition = vdup_n_u16(0);
      int32x2_t sat0AB_array = pairwise_AB[matcher_ind][i*3 + 0];
      int32x2_t sat1AB_array = pairwise_AB[matcher_ind][i*3 + 1];
      int32x2_t sat2AB_array = pairwise_AB[matcher_ind][i*3 + 2];
      for (int j = 0; j < numPointsB; j += 1) {
        const int32x2_t sat0AB = vdup_lane_s32(vneg_s32(vand_s32(sat0AB_array, one)), 0);
        const int32x2_t sat1AB = vdup_lane_s32(vneg_s32(vand_s32(sat1AB_array, one)), 0);
        const int32x2_t sat2AB = vdup_lane_s32(vneg_s32(vand_s32(sat2AB_array, one)), 0);
        
        const int32x2_t sat0AC = pairwise_AC[matcher_ind][i*3 + 0];
        const int32x2_t sat1AC = pairwise_AC[matcher_ind][i*3 + 1];
        const int32x2_t sat2AC = pairwise_AC[matcher_ind][i*3 + 2];
        
        const int32x2_t sat0BC = pairwise_BC[matcher_ind][j*3 + 0];
        const int32x2_t sat1BC = pairwise_BC[matcher_ind][j*3 + 1];
        const int32x2_t sat2BC = pairwise_BC[matcher_ind][j*3 + 2];
        
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
    counts[matcher_ind][3] = vget_lane_u32(vreinterpret_u32_u64(vpaddl_u32(count3_addition)), 0);
  } // loop over matchers
#endif
	//double elapsed = Timing::ElapsedTimeMs() - start;
  
	//nptRuntimes.twoWay = elapsedPairwise;
	//nptRuntimes.threeWay = elapsed-elapsedPairwise;
  
  // free memory
  for (int i = 0; i < numMatchers; i++) {
    delete pairwise_AB[i];
    delete pairwise_AC[i];
    delete pairwise_BC[i];
  }
  delete pairwise_AB;
  delete pairwise_AC;
  delete pairwise_BC;
  
} // three way counts

#endif 
