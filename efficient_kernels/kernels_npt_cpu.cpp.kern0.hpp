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
    
    //double start = Timing::ElapsedTimeMs();
    PrecomputePairwiseInteractionsCPU(
        pairwise_AB, counts, pointsA, numPointsA, pointsB, numPointsB, rMinSq, rMaxSq);
    PrecomputePairwiseInteractionsCPU(
        pairwise_AC, counts, pointsA, numPointsA, pointsC, numPointsC, rMinSq, rMaxSq);
    PrecomputePairwiseInteractionsCPU(
        pairwise_BC, counts, pointsB, numPointsB, pointsC, numPointsC, rMinSq, rMaxSq);
  //double elapsedPairwise = Timing::ElapsedTimeMs() - start;

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

            const unsigned short sat0_shift6 = sat0 << 6;

            const uint64_t *pairwise_AC_s = (const uint64_t *) (pairwise_AC + i*numPointsC);
            const uint64_t *pairwise_BC_s = (const uint64_t *) (pairwise_BC + j*numPointsC);

            const unsigned char * localSatisfiabilityLUT = &satisfiabilityLUT[sat0_shift6];
            for (int k = 0; k < numPointsC/8; ++k)
            {
                const uint64_t sat1 = pairwise_AC_s[k];
                const uint64_t sat2 = pairwise_BC_s[k];
                const uint64_t sat12 = (sat1 << 3) | sat2;

                const size_t ind0 = ((sat12 >> 0) & 0x3F);
                const size_t ind1 = ((sat12 >> 8) & 0x3F);
                const size_t ind2 = ((sat12 >> 16) & 0x3F);
                const size_t ind3 = ((sat12 >> 24) & 0x3F);
                const size_t ind4 = ((sat12 >> 32) & 0x3F);
                const size_t ind5 = ((sat12 >> 40) & 0x3F);
                const size_t ind6 = ((sat12 >> 48) & 0x3F);
                const size_t ind7 = ((sat12 >> 56) & 0x3F);

                count3_addition += localSatisfiabilityLUT[ ind0 ];
                count3_addition += localSatisfiabilityLUT[ ind1 ];
                count3_addition += localSatisfiabilityLUT[ ind2 ];
                count3_addition += localSatisfiabilityLUT[ ind3 ];
                count3_addition += localSatisfiabilityLUT[ ind4 ];
                count3_addition += localSatisfiabilityLUT[ ind5 ];
                count3_addition += localSatisfiabilityLUT[ ind6 ];
                count3_addition += localSatisfiabilityLUT[ ind7 ];
            }
            for (int k = 8*(numPointsC/8); k < numPointsC; ++k)
            {
                const unsigned char sat1 = pairwise_AC[i*numPointsC+k];
                const unsigned char sat2 = pairwise_BC[j*numPointsC+k];
                const unsigned short sat = sat0_shift6 | (sat1 << 3) | sat2;
                count3_addition += satisfiabilityLUT[sat];
            }
        }

        #pragma omp critical
        counts[3] += count3_addition;
    }

  //double elapsed = Timing::ElapsedTimeMs() - start;

  //nptRuntimes.twoWay = elapsedPairwise;
  //nptRuntimes.threeWay = elapsed-elapsedPairwise;

    delete [] pairwise_AB;
    delete [] pairwise_AC;
    delete [] pairwise_BC;
}

