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

static void PrecomputePairwiseInteractionsCPU(
	uint64_t *__restrict pairwiseInteractions,
	uint64_t *counts,
	const double3 *__restrict pointsA,
	int numPointsA,
	const double3 *__restrict pointsB,
	int numPointsB,
	const double *__restrict rMinSq,
	const double *__restrict rMaxSq)
{
	#ifdef HAVE_SSE2
		const __m128d rMinSq0 = npt_mm_loaddup_pd(&rMinSq[0]);
		const __m128d rMinSq1 = npt_mm_loaddup_pd(&rMinSq[1]);
		const __m128d rMinSq2 = npt_mm_loaddup_pd(&rMinSq[2]);
		const __m128d rMaxSq0 = npt_mm_loaddup_pd(&rMaxSq[0]);
		const __m128d rMaxSq1 = npt_mm_loaddup_pd(&rMaxSq[1]);
		const __m128d rMaxSq2 = npt_mm_loaddup_pd(&rMaxSq[2]);
	#else
		const double rMinSq0 = rMinSq[0], rMinSq1 = rMinSq[1], rMinSq2 = rMinSq[2];
		const double rMaxSq0 = rMaxSq[0], rMaxSq1 = rMaxSq[1], rMaxSq2 = rMaxSq[2];
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
		#ifdef HAVE_SSE2
			for (int k = 1; k < numPointsB; k += 2) {
				const __m128d b0x_b0y = _mm_load_pd(&localPointsB[0].x);
				const __m128d b0z_b1x = _mm_load_pd(&localPointsB[0].z);
				const __m128d b1y_b1z = _mm_load_pd(&localPointsB[1].y);
				localPointsB += 2;
			
				const __m128d b0x_b0y_diff_a = _mm_sub_pd(b0x_b0y, ax_ay);
				const __m128d b0z_b1x_diff_a = _mm_sub_pd(b0z_b1x, az_ax);
				const __m128d b1y_b1z_diff_a = _mm_sub_pd(b1y_b1z, ay_az);
				
				const __m128d b0x_b0y_diff_a_sq = _mm_mul_pd(b0x_b0y_diff_a, b0x_b0y_diff_a);
				const __m128d b0z_b1x_diff_a_sq = _mm_mul_pd(b0z_b1x_diff_a, b0z_b1x_diff_a);
				const __m128d b1y_b1z_diff_a_sq = _mm_mul_pd(b1y_b1z_diff_a, b1y_b1z_diff_a);
				
				#ifdef HAVE_SSE3
					const __m128d b0_b1_diff_a_norm = _mm_add_pd(_mm_hadd_pd(b0x_b0y_diff_a_sq, b1y_b1z_diff_a_sq), b0z_b1x_diff_a_sq);
				#else
					const __m128d b0_b1_diff_a_norm = _mm_add_pd(_mm_add_pd(_mm_unpacklo_pd(b0x_b0y_diff_a_sq, b1y_b1z_diff_a_sq), _mm_unpackhi_pd(b0x_b0y_diff_a_sq, b1y_b1z_diff_a_sq)), b0z_b1x_diff_a_sq);
				#endif
	
				const uint32_t localSat0 = uint32_t(_mm_movemask_pd(_mm_and_pd(_mm_cmplt_pd(b0_b1_diff_a_norm, rMaxSq0), _mm_cmpgt_pd(b0_b1_diff_a_norm, rMinSq0))));
				const uint32_t localSat1 = uint32_t(_mm_movemask_pd(_mm_and_pd(_mm_cmplt_pd(b0_b1_diff_a_norm, rMaxSq1), _mm_cmpgt_pd(b0_b1_diff_a_norm, rMinSq1))));
				const uint32_t localSat2 = uint32_t(_mm_movemask_pd(_mm_and_pd(_mm_cmplt_pd(b0_b1_diff_a_norm, rMaxSq2), _mm_cmpgt_pd(b0_b1_diff_a_norm, rMinSq2))));
				sat0 = (sat0 >> 2) | (uint64_t(localSat0) << 62);
				sat1 = (sat1 >> 2) | (uint64_t(localSat1) << 62);
				sat2 = (sat2 >> 2) | (uint64_t(localSat2) << 62);
			}
			if ((numPointsB & 1) != 0) {
				const __m128d b0x_b0y = _mm_load_pd(&localPointsB[0].x);
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
				sat0 = (sat0 >> 1) | (uint64_t(localSat0) << 63);
				sat1 = (sat1 >> 1) | (uint64_t(localSat1) << 63);
				sat2 = (sat2 >> 1) | (uint64_t(localSat2) << 63);
			}
		#else
			for (int k = 0; k < numPointsB; k += 1) {
				const double3 b0 = *localPointsB++;
				const double3 b0a_diff = b0 - a;
				const double dist_b0a_sq = b0a_diff.x*b0a_diff.x + b0a_diff.y*b0a_diff.y + b0a_diff.z*b0a_diff.z;
									
				size_t localSat0 = ((dist_b0a_sq > rMinSq0) & (dist_b0a_sq < rMaxSq0));
				size_t localSat1 = ((dist_b0a_sq > rMinSq1) & (dist_b0a_sq < rMaxSq1));
				size_t localSat2 = ((dist_b0a_sq > rMinSq2) & (dist_b0a_sq < rMaxSq2));
				sat0 = (sat0 >> 1) | (uint64_t(localSat0) << 63);
				sat1 = (sat1 >> 1) | (uint64_t(localSat1) << 63);
				sat2 = (sat2 >> 1) | (uint64_t(localSat2) << 63);
			}
		#endif
		pairwiseInteractions[i * 3 + 0] = (sat0 >> (64 - numPointsB));
		pairwiseInteractions[i * 3 + 1] = (sat1 >> (64 - numPointsB));
		pairwiseInteractions[i * 3 + 2] = (sat2 >> (64 - numPointsB));
	}
}

void ComputeThreePointCorrelationCountsCPU(
	uint64_t *__restrict counts,
	NptRuntimes & nptRuntimes,
	const double3 *__restrict pointsA,
	int numPointsA,
	const double3 *__restrict pointsB,
	int numPointsB,
	const double3 *__restrict pointsC,
	int numPointsC,
	const double *__restrict rMinSq,
	const double *__restrict rMaxSq,
	const unsigned char *__restrict satisfiabilityLUT)
{
	uint64_t pairwise_AB[64 * 3];
	uint64_t pairwise_AC[64 * 3];
	uint64_t pairwise_BC[64 * 3];
	
	memset(pairwise_AB, 0, sizeof(uint64_t)*64*3);
	memset(pairwise_AC, 0, sizeof(uint64_t)*64*3);
	memset(pairwise_BC, 0, sizeof(uint64_t)*64*3);
	memset(counts, 0, sizeof(uint64_t)*4);
	
	//double start = Timing::ElapsedTimeMs();
	PrecomputePairwiseInteractionsCPU(
		pairwise_AB, counts, pointsA, numPointsA, pointsB, numPointsB, rMinSq, rMaxSq);
	PrecomputePairwiseInteractionsCPU(
		pairwise_AC, counts, pointsA, numPointsA, pointsC, numPointsC, rMinSq, rMaxSq);
	PrecomputePairwiseInteractionsCPU(
		pairwise_BC, counts, pointsB, numPointsB, pointsC, numPointsC, rMinSq, rMaxSq);
	//double elapsedPairwise = Timing::ElapsedTimeMs() - start;

	for (int i = 0; i < numPointsA; i += 1) {
		size_t count3_addition = 0;
		
		uint64_t sat0AB_array = pairwise_AB[i*3 + 0];
		uint64_t sat1AB_array = pairwise_AB[i*3 + 1];
		uint64_t sat2AB_array = pairwise_AB[i*3 + 2];
		for (int j = 0; j < numPointsB; j += 1) {
			const uint64_t sat0AB = -(sat0AB_array & 1);
			const uint64_t sat1AB = -(sat1AB_array & 1);
			const uint64_t sat2AB = -(sat2AB_array & 1);

			const uint64_t sat0AC = pairwise_AC[i*3 + 0];
			const uint64_t sat1AC = pairwise_AC[i*3 + 1];
			const uint64_t sat2AC = pairwise_AC[i*3 + 2];

			const uint64_t sat0BC = pairwise_BC[j*3 + 0];
			const uint64_t sat1BC = pairwise_BC[j*3 + 1];
			const uint64_t sat2BC = pairwise_BC[j*3 + 2];

			count3_addition += npt_popcount64((sat0AB & sat1AC & sat2BC) | (sat0AB & sat2AC & sat1BC) |
				(sat1AB & sat0AC & sat2BC) | (sat1AB & sat2AC & sat0BC) |
				(sat2AB & sat0AC & sat1BC) | (sat2AB & sat1AC & sat0BC));

			sat0AB_array  >>= 1;
			sat1AB_array  >>= 1;
			sat2AB_array  >>= 1;
		}

		counts[3] += count3_addition;
	}

	//double elapsed = Timing::ElapsedTimeMs() - start;

	//nptRuntimes.twoWay = elapsedPairwise;
	//nptRuntimes.threeWay = elapsed-elapsedPairwise;
}
