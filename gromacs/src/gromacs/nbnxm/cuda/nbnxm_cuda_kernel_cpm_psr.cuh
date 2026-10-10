/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * bisheng 2026 - CPM (constant potential method): potential-only CUDA kernel.
 *
 * Accumulates, at every electrode atom, the short-range electrostatic potential
 * from its electrolyte partners,
 *
 *     psr_i = sum_j epsfac * q_j * [ erf(eta_ij r) / r - erf(beta r) / r ],   r < rcoulomb,
 *
 * which is exactly the psr term of the CPM block in the force kernel
 * (nbnxm_cuda_kernel.cuh), evaluated with the same pair list, cut-off, exclusion
 * and eta handling and the same single-precision expressions. It computes no
 * LJ, no forces and no energies, and it never reads an electrode charge (only
 * electrolyte charges enter psr), so it can run before the electrode-charge
 * solve. The force kernel then runs without the psr work.
 *
 * Cluster pairs with no electrode atom on either side are skipped by a warp
 * vote on the per-atom electrode mask (+1 electrode, -1 electrolyte, 0 filler),
 * so the kernel costs a fraction of the force kernel: the pair-list traversal
 * plus the erf work of the mixed pairs only.
 *
 * One kernel serves every electrostatics / VdW flavour of the force kernel: it
 * only needs the Ewald splitting parameter and the Coulomb cut-off. It must be
 * compiled in the same translation unit as its launch (GROMACS 2020 builds
 * without relocatable device code), so nbnxm_cuda.cu includes it directly.
 */

__launch_bounds__(c_clSize* c_clSize) __global__
        void nbnxn_kernel_cpm_psr_cuda(const cu_atomdata_t atdat, const cu_nbparam_t nbparam, const cu_plist_t plist)
{
    const nbnxn_sci_t*  pl_sci      = plist.sci;
    const nbnxn_cj4_t*  pl_cj4      = plist.cj4;
    const nbnxn_excl_t* excl        = plist.excl;
    const float4*       xq          = atdat.xq;
    const float3*       shift_vec   = atdat.shift_vec;
    const float         rcoulomb_sq = nbparam.rcoulomb_sq;
    const float         beta        = nbparam.ewald_beta;
    const float         epsfac      = nbparam.epsfac;

    const unsigned int tidxi = threadIdx.x;
    const unsigned int tidxj = threadIdx.y;
    const unsigned int tidx  = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int bidx  = blockIdx.x;
    const unsigned int widx  = tidx / warp_size; /* warp index */

    /* i super-cluster data: x + q (epsfac folded in), electrode mask, eta */
    __shared__ float4 xqib[c_numClPerSupercl * c_clSize];
    __shared__ float  maskib[c_numClPerSupercl * c_clSize];
    __shared__ float  etaib[c_numClPerSupercl * c_clSize];
    /* cj indices, for each warp separately */
    __shared__ int cjs[c_nbnxnGpuClusterpairSplit * c_nbnxnGpuJgroupSize];

    const unsigned superClInteractionMask = ((1U << c_numClPerSupercl) - 1U);

    const nbnxn_sci_t nb_sci     = pl_sci[bidx];
    const int         sci        = nb_sci.sci;
    const int         cij4_start = nb_sci.cj4_ind_start;
    const int         cij4_end   = nb_sci.cj4_ind_end;

    {
        const int    ci       = sci * c_numClPerSupercl + tidxj;
        const int    ai       = ci * c_clSize + tidxi;
        const float* shiftptr = (const float*)&shift_vec[nb_sci.shift];
        float4       xqbuf = xq[ai] + make_float4(LDG(shiftptr), LDG(shiftptr + 1), LDG(shiftptr + 2), 0.0f);
        xqbuf.w *= epsfac;
        xqib[tidxj * c_clSize + tidxi]   = xqbuf;
        maskib[tidxj * c_clSize + tidxi] = LDG(&atdat.q_ele_mask[ai]);
        etaib[tidxj * c_clSize + tidxi]  = LDG(&atdat.eta_per_atom[ai]);
    }
    __syncthreads();
    /* block-uniform: does the i super-cluster hold any electrode atom? */
    const bool iHasEle = (__syncthreads_or(maskib[tidxj * c_clSize + tidxi] > 0.0f) != 0);

    float psr_ci[c_numClPerSupercl];
    for (int i = 0; i < c_numClPerSupercl; i++)
    {
        psr_ci[i] = 0.0f;
    }

    const int nonSelfInteraction = !(nb_sci.shift == CENTRAL & tidxj <= tidxi);

    for (int j4 = cij4_start; j4 < cij4_end; j4++)
    {
        const unsigned int wexcl_idx = pl_cj4[j4].imei[widx].excl_ind;
        const unsigned int imask     = pl_cj4[j4].imei[widx].imask;
        const unsigned int wexcl     = excl[wexcl_idx].pair[(tidx) & (warp_size - 1)];

        if (imask)
        {
            /* Pre-load cj into shared memory on both warps separately */
            if ((tidxj == 0 | tidxj == 4) & (tidxi < c_nbnxnGpuJgroupSize))
            {
                cjs[tidxi + tidxj * c_nbnxnGpuJgroupSize / c_splitClSize] = pl_cj4[j4].cj[tidxi];
            }
            __syncwarp(c_fullWarpMask);

            for (int jm = 0; jm < c_nbnxnGpuJgroupSize; jm++)
            {
                if (imask & (superClInteractionMask << (jm * c_numClPerSupercl)))
                {
                    unsigned int mask_ji = (1U << (jm * c_numClPerSupercl));

                    const int   cj    = cjs[jm + (tidxj & 4) * c_nbnxnGpuJgroupSize / c_splitClSize];
                    const int   aj    = cj * c_clSize + tidxj;
                    const float maskJ = LDG(&atdat.q_ele_mask[aj]);

                    /* Warp-uniform skip: no electrode atom on either side of
                       this cluster pair means no psr contribution at all. */
                    if (!iHasEle && !__any_sync(c_fullWarpMask, maskJ > 0.0f))
                    {
                        continue;
                    }

                    const float4 xqbuf = xq[aj];
                    const float3 xj    = make_float3(xqbuf.x, xqbuf.y, xqbuf.z);
                    const float  qj_f  = xqbuf.w;
                    const float  etaJ  = LDG(&atdat.eta_per_atom[aj]);
                    float        psr_cj = 0.0f;

                    for (int i = 0; i < c_numClPerSupercl; i++)
                    {
                        if (imask & mask_ji)
                        {
                            const int   ci    = sci * c_numClPerSupercl + i;
                            const float maskI = maskib[i * c_clSize + tidxi];
                            /* mask product < 0 selects exactly the electrode-electrolyte pairs */
                            if (maskI * maskJ < 0.0f)
                            {
                                const float4 xqi = xqib[i * c_clSize + tidxi];
                                const float3 rv  = make_float3(xqi.x, xqi.y, xqi.z) - xj;
                                float        r2  = norm2(rv);
                                const float  int_bit = (wexcl & mask_ji) ? 1.0f : 0.0f;

                                if ((r2 < rcoulomb_sq) * (nonSelfInteraction | (ci != cj)))
                                {
                                    r2                 = max(r2, NBNXN_MIN_RSQ);
                                    const float inv_r  = rsqrt(r2);
                                    const float r1     = r2 * inv_r;
                                    const float etaIJ  = etaib[i * c_clSize + tidxi] + etaJ;
                                    const float ersq   = etaIJ * etaIJ * r2;
                                    const bool  inEta  = (ersq < 16.0f);
                                    const float psrSub = (inEta ? erff(etaIJ * r1) * inv_r : int_bit * inv_r)
                                                         - erff(beta * r1) * inv_r;
                                    if (maskI > 0.0f)
                                    {
                                        /* electrode i, electrolyte j */
                                        psr_ci[i] += epsfac * qj_f * psrSub;
                                    }
                                    else
                                    {
                                        /* electrolyte i, electrode j (qi carries epsfac) */
                                        psr_cj += xqi.w * psrSub;
                                    }
                                }
                            }
                        }
                        /* shift the mask bit by 1 */
                        mask_ji += mask_ji;
                    }

                    if (psr_cj != 0.0f)
                    {
                        atomicAdd(&atdat.psr_coul[aj], psr_cj);
                    }
                }
            }
        }
        // avoid shared memory WAR hazards between loop iterations
        __syncwarp(c_fullWarpMask);
    }

    for (int i = 0; i < c_numClPerSupercl; i++)
    {
        if (psr_ci[i] != 0.0f)
        {
            const int ai = (sci * c_numClPerSupercl + i) * c_clSize + tidxi;
            atomicAdd(&atdat.psr_coul[ai], psr_ci[i]);
        }
    }
}
