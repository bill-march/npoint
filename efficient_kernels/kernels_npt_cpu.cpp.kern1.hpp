#ifndef NPT_UNROLL_FACTOR
	#define NPT_UNROLL_FACTOR 64
#endif

#if NPT_UNROLL_FACTOR == 64
	typedef uint64_t npt_bitmask_t;
#elif NPT_UNROLL_FACTOR == 32
	typedef uint32_t npt_bitmask_t;
#elif NPT_UNROLL_FACTOR == 16
	typedef uint16_t npt_bitmask_t;
#elif NPT_UNROLL_FACTOR == 8
	typedef uint8_t npt_bitmask_t;
#else
	#error "Unsupported unroll factor"
#endif

static __inline__ uint64_t npt_popcount(uint64_t x) {
	return __builtin_popcountll(x);
}

static __inline__ uint32_t npt_popcount(uint32_t x) {
	return __builtin_popcount(x);
}

static __inline__ uint16_t npt_popcount(uint16_t x) {
	return __builtin_popcount(x);
}

static __inline__ uint8_t npt_popcount(uint8_t x) {
	return __builtin_popcount(x);
}

static void PrecomputePairwiseInteractionsCPU(
	uint8_t *pairwiseInteractions,
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
	for (int i = 0; i < numPointsA; i += 1) {
		const double3 a = pointsA[i];
		const double3* localPointsB = pointsB;
		uint8_t* localPairwiseInteractions = &pairwiseInteractions[i * (numPointsB / 8) * 3];
		for (int j = 0; j < numPointsB / NPT_UNROLL_FACTOR; j += 1) {
			for (int k = 0; k < (NPT_UNROLL_FACTOR / 8); k += 1) {
				const double3 b0 = *localPointsB++;
				const double3 b1 = *localPointsB++;
				const double3 b2 = *localPointsB++;
				const double3 b3 = *localPointsB++;
				const double3 b4 = *localPointsB++;
				const double3 b5 = *localPointsB++;
				const double3 b6 = *localPointsB++;
				const double3 b7 = *localPointsB++;
				
				const double3 b0a_diff = b0 - a;
				const double3 b1a_diff = b1 - a;
				const double3 b2a_diff = b2 - a;
				const double3 b3a_diff = b3 - a;
				const double3 b4a_diff = b4 - a;
				const double3 b5a_diff = b5 - a;
				const double3 b6a_diff = b6 - a;
				const double3 b7a_diff = b7 - a;
				
				const double dist_b0a_sq = b0a_diff.x*b0a_diff.x + b0a_diff.y*b0a_diff.y + b0a_diff.z*b0a_diff.z;
				const double dist_b1a_sq = b1a_diff.x*b1a_diff.x + b1a_diff.y*b1a_diff.y + b1a_diff.z*b1a_diff.z;
				const double dist_b2a_sq = b2a_diff.x*b2a_diff.x + b2a_diff.y*b2a_diff.y + b2a_diff.z*b2a_diff.z;
				const double dist_b3a_sq = b3a_diff.x*b3a_diff.x + b3a_diff.y*b3a_diff.y + b3a_diff.z*b3a_diff.z;
				const double dist_b4a_sq = b4a_diff.x*b4a_diff.x + b4a_diff.y*b4a_diff.y + b4a_diff.z*b4a_diff.z;
				const double dist_b5a_sq = b5a_diff.x*b5a_diff.x + b5a_diff.y*b5a_diff.y + b5a_diff.z*b5a_diff.z;
				const double dist_b6a_sq = b6a_diff.x*b6a_diff.x + b6a_diff.y*b6a_diff.y + b6a_diff.z*b6a_diff.z;
				const double dist_b7a_sq = b7a_diff.x*b7a_diff.x + b7a_diff.y*b7a_diff.y + b7a_diff.z*b7a_diff.z;
				
				size_t sat0 = ((dist_b0a_sq > rMinSq0) & (dist_b0a_sq < rMaxSq0)) |
					(((dist_b1a_sq > rMinSq0) & (dist_b1a_sq < rMaxSq0)) << 1) |
					(((dist_b2a_sq > rMinSq0) & (dist_b2a_sq < rMaxSq0)) << 2) |
					(((dist_b3a_sq > rMinSq0) & (dist_b3a_sq < rMaxSq0)) << 3) |
					(((dist_b4a_sq > rMinSq0) & (dist_b4a_sq < rMaxSq0)) << 4) |
					(((dist_b5a_sq > rMinSq0) & (dist_b5a_sq < rMaxSq0)) << 5) |
					(((dist_b6a_sq > rMinSq0) & (dist_b6a_sq < rMaxSq0)) << 6) |
					(((dist_b7a_sq > rMinSq0) & (dist_b7a_sq < rMaxSq0)) << 7);
				size_t sat1 = ((dist_b0a_sq > rMinSq1) & (dist_b0a_sq < rMaxSq1)) |
					(((dist_b1a_sq > rMinSq1) & (dist_b1a_sq < rMaxSq1)) << 1) |
					(((dist_b2a_sq > rMinSq1) & (dist_b2a_sq < rMaxSq1)) << 2) |
					(((dist_b3a_sq > rMinSq1) & (dist_b3a_sq < rMaxSq1)) << 3) |
					(((dist_b4a_sq > rMinSq1) & (dist_b4a_sq < rMaxSq1)) << 4) |
					(((dist_b5a_sq > rMinSq1) & (dist_b5a_sq < rMaxSq1)) << 5) |
					(((dist_b6a_sq > rMinSq1) & (dist_b6a_sq < rMaxSq1)) << 6) |
					(((dist_b7a_sq > rMinSq1) & (dist_b7a_sq < rMaxSq1)) << 7);
				size_t sat2 = ((dist_b0a_sq > rMinSq2) & (dist_b0a_sq < rMaxSq2)) |
					(((dist_b1a_sq > rMinSq2) & (dist_b1a_sq < rMaxSq2)) << 1) |
					(((dist_b2a_sq > rMinSq2) & (dist_b2a_sq < rMaxSq2)) << 2) |
					(((dist_b3a_sq > rMinSq2) & (dist_b3a_sq < rMaxSq2)) << 3) |
					(((dist_b4a_sq > rMinSq2) & (dist_b4a_sq < rMaxSq2)) << 4) |
					(((dist_b5a_sq > rMinSq2) & (dist_b5a_sq < rMaxSq2)) << 5) |
					(((dist_b6a_sq > rMinSq2) & (dist_b6a_sq < rMaxSq2)) << 6) |
					(((dist_b7a_sq > rMinSq2) & (dist_b7a_sq < rMaxSq2)) << 7);
				localPairwiseInteractions[(NPT_UNROLL_FACTOR / 8) * 0 + k] = sat0;
				localPairwiseInteractions[(NPT_UNROLL_FACTOR / 8) * 1 + k] = sat1;
				localPairwiseInteractions[(NPT_UNROLL_FACTOR / 8) * 2 + k] = sat2;
			}
			localPairwiseInteractions += 3 * (NPT_UNROLL_FACTOR / 8);
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
	npt_bitmask_t *pairwise_AB = new npt_bitmask_t[numPointsA*(numPointsB / 64)*3];
	npt_bitmask_t *pairwise_AC = new npt_bitmask_t[numPointsA*(numPointsC / 64)*3];
	npt_bitmask_t *pairwise_BC = new npt_bitmask_t[numPointsB*(numPointsC / 64)*3];
	
	memset(counts, 0, sizeof(uint64_t)*4);
	
	//double start = Timing::ElapsedTimeMs();
	PrecomputePairwiseInteractionsCPU(
		(uint8_t*)pairwise_AB, counts, pointsA, numPointsA, pointsB, numPointsB, rMinSq, rMaxSq);
	PrecomputePairwiseInteractionsCPU(
		(uint8_t*)pairwise_AC, counts, pointsA, numPointsA, pointsC, numPointsC, rMinSq, rMaxSq);
	PrecomputePairwiseInteractionsCPU(
		(uint8_t*)pairwise_BC, counts, pointsB, numPointsB, pointsC, numPointsC, rMinSq, rMaxSq);
	//double elapsedPairwise = Timing::ElapsedTimeMs() - start;

	#pragma omp parallel for schedule(dynamic,1)
	for (int i = 0; i < numPointsA; i += 1) {
		int count3_addition = 0;
		for (int j = 0; j < numPointsB; j += 1) {
			const npt_bitmask_t sat0AB = -((pairwise_AB[(i*(numPointsB / NPT_UNROLL_FACTOR) + (j / NPT_UNROLL_FACTOR)) * 3 + 0] >> (j % NPT_UNROLL_FACTOR)) & 1);
			const npt_bitmask_t sat1AB = -((pairwise_AB[(i*(numPointsB / NPT_UNROLL_FACTOR) + (j / NPT_UNROLL_FACTOR)) * 3 + 1] >> (j % NPT_UNROLL_FACTOR)) & 1);
			const npt_bitmask_t sat2AB = -((pairwise_AB[(i*(numPointsB / NPT_UNROLL_FACTOR) + (j / NPT_UNROLL_FACTOR)) * 3 + 2] >> (j % NPT_UNROLL_FACTOR)) & 1);

			// If this (A,B) pairing does not satisfy any of the matcher distances,
			// then we can short-circuit out of testing (A,B,*) triplets.
			//~ if (! sat0) { continue; }

			for (int k = 0; k < numPointsC / NPT_UNROLL_FACTOR; k += 1) {
				const npt_bitmask_t sat0AC = (pairwise_AC[(i*(numPointsC / NPT_UNROLL_FACTOR) + k) * 3 + 0]);
				const npt_bitmask_t sat1AC = (pairwise_AC[(i*(numPointsC / NPT_UNROLL_FACTOR) + k) * 3 + 1]);
				const npt_bitmask_t sat2AC = (pairwise_AC[(i*(numPointsC / NPT_UNROLL_FACTOR) + k) * 3 + 2]);

				const npt_bitmask_t sat0BC = (pairwise_BC[(j*(numPointsC / NPT_UNROLL_FACTOR) + k) * 3 + 0]);
				const npt_bitmask_t sat1BC = (pairwise_BC[(j*(numPointsC / NPT_UNROLL_FACTOR) + k) * 3 + 1]);
				const npt_bitmask_t sat2BC = (pairwise_BC[(j*(numPointsC / NPT_UNROLL_FACTOR) + k) * 3 + 2]);

				count3_addition += npt_popcount((sat0AB & sat1AC & sat2BC) | (sat0AB & sat2AC & sat1BC) |
					(sat1AB & sat0AC & sat2BC) | (sat1AB & sat2AC & sat0BC) |
					(sat2AB & sat0AC & sat1BC) | (sat2AB & sat1AC & sat0BC));
			}
		}

		#pragma omp critical
		counts[3] += count3_addition;
	}

	//double elapsed = Timing::ElapsedTimeMs() - start;

//    printf("ComputeThreePointCorrelationCountsCPU: Timing = %.2f [ms] (%.2f [ms] for pairwise computations)\n",
//        elapsed, elapsedPairwise);
	//nptRuntimes.twoWay = elapsedPairwise;
	//nptRuntimes.threeWay = elapsed-elapsedPairwise;
//printf("CPU: 2-way: %.04f\n", elapsedPairwise);
//printf("CPU: 3-way: %.04f\n", elapsed-elapsedPairwise);
//printf("CPU: Total: %.04f\n", elapsed);

	delete [] pairwise_AB;
	delete [] pairwise_AC;
	delete [] pairwise_BC;
}
