/*
 * The Becke weights' second derivative contracted with one vector over the
 * grid, without storing it (catalogue B9; electrolites/fastgridresp.py).
 *
 * GPU4PySCF's get_d2weight_dAdB returns d2w[G, H, 3, 3, g] with the blocks
 * that involve the grid point's own atom A(g) zero, then fills them by
 * translation invariance:  d2w[G, A] = -sum_H d2w[G, H],
 * d2w[A, H] = -sum_G d2w[G, H]  (after the first fill).  Contracted with
 * eps[g], that is, with K the kernels' blocks (zero where G or H is A(g)),
 *
 *     out[G, H] =   sum_g eps K[G, H]                         (P)
 *                 - sum_g eps K[G, H'] summed into out[G, A(g)]  (U[G, A])
 *                 - sum_g eps K[G', H] summed into out[A(g), H]  (V[A, H])
 *                 + sum_A U[., A] on out[A, A]
 *
 * so each point's block goes, scaled by eps, to P[G,H], U[G,A] and V[A,H].
 * The W lanes of a warp that share (G, H) (W = 16 off the diagonal, 32 on
 * it, from GPU4PySCF's launch shapes) sum their blocks with shuffles first;
 * when the W points also share A(g) -- the usual case, the grid being in
 * atom order -- one lane writes all three, otherwise each lane writes its
 * own U and V.
 */

#define FULL_MASK 0xffffffffu

template <int W>
__device__ __forceinline__ void d2w_accumulate(
        double* __restrict__ P, double* __restrict__ U, double* __restrict__ V,
        const double9& d, const double e, const int G, const int H,
        const int A, const bool in_range, const int natm)
{
    double v[9] = {e*d.x.x, e*d.x.y, e*d.x.z, e*d.y.x, e*d.y.y, e*d.y.z,
                   e*d.z.x, e*d.z.y, e*d.z.z};
    const int lane = threadIdx.x % W;
    const int A0 = __shfl_sync(FULL_MASK, A, 0, W);
    const bool uniform = __all_sync(FULL_MASK, A == A0);
    double r[9];
#pragma unroll
    for (int k = 0; k < 9; k++) {
        double s = v[k];
#pragma unroll
        for (int off = W / 2; off > 0; off >>= 1)
            s += __shfl_xor_sync(FULL_MASK, s, off, W);
        r[k] = s;
    }
    if (!in_range)
        return;
    if (lane == 0) {
        double* p = P + ((size_t)G * natm + H) * 9;
#pragma unroll
        for (int k = 0; k < 9; k++) atomicAdd(p + k, r[k]);
        if (uniform && A0 >= 0) {
            double* u = U + ((size_t)G * natm + A0) * 9;
            double* w = V + ((size_t)A0 * natm + H) * 9;
#pragma unroll
            for (int k = 0; k < 9; k++) { atomicAdd(u + k, r[k]); atomicAdd(w + k, r[k]); }
        }
    }
    if (!uniform && A >= 0 && e != 0.) {
        double* u = U + ((size_t)G * natm + A) * 9;
        double* w = V + ((size_t)A * natm + H) * 9;
#pragma unroll
        for (int k = 0; k < 9; k++) { atomicAdd(u + k, v[k]); atomicAdd(w + k, v[k]); }
    }
}

#define D2W_ARGS                                                                \
    const double* __restrict__ grid_coords,                                     \
    const double* __restrict__ grid_quadrature_weights,                         \
    const double* __restrict__ atm_coords, const double* __restrict__ a_factor, \
    const double* __restrict__ inv_atom_distance,                               \
    const int* __restrict__ atm_idx, const double* __restrict__ Ar_distance,    \
    const double* __restrict__ PB, const double* __restrict__ invsumPB,         \
    const int ngrids, const int natm
#define D2W_FWD grid_coords, grid_quadrature_weights, atm_coords, a_factor,     \
    inv_atom_distance, atm_idx, Ar_distance, PB, invsumPB, ngrids, natm

template <bool RA, GridPartitionScheme S>
__device__ __forceinline__ void d2w_offdiag_body(
        double* P, double* U, double* V, const double* eps, D2W_ARGS)
{
    const int i_grid = blockIdx.x * blockDim.x + threadIdx.x;
    const int G = blockIdx.y * blockDim.y + threadIdx.y;
    const int H = blockIdx.z * blockDim.z + threadIdx.z;
    double9 d;
    const bool ok = d2w_offdiag_point<RA, S>(&d, D2W_FWD);
    const int A = i_grid < ngrids ? atm_idx[i_grid] : -1;
    const double e = ok ? eps[i_grid] : 0.;
    if (!ok) d = {0,0,0, 0,0,0, 0,0,0};
    d2w_accumulate<16>(P, U, V, d, e, G, H, A, G < natm && H < natm, natm);
}

template <bool RA, GridPartitionScheme S>
__device__ __forceinline__ void d2w_diag_body(
        double* P, double* U, double* V, const double* eps, D2W_ARGS)
{
    const int i_grid = blockIdx.x * blockDim.x + threadIdx.x;
    const int G = blockIdx.y * blockDim.y + threadIdx.y;
    double9 d;
    const bool ok = d2w_diag_point<RA, S>(&d, D2W_FWD);
    const int A = i_grid < ngrids ? atm_idx[i_grid] : -1;
    const double e = ok ? eps[i_grid] : 0.;
    if (!ok) d = {0,0,0, 0,0,0, 0,0,0};
    d2w_accumulate<32>(P, U, V, d, e, G, G, A, G < natm, natm);
}

#define ENTRY(NAME, BODY, RA, S)                                                \
extern "C" __global__ void NAME(double* P, double* U, double* V,               \
                                const double* eps, D2W_ARGS)                    \
{ BODY<RA, S>(P, U, V, eps, D2W_FWD); }

ENTRY(d2w_offdiag_becke,      d2w_offdiag_body, false, GridPartitionScheme::original_becke)
ENTRY(d2w_offdiag_becke_ra,   d2w_offdiag_body, true,  GridPartitionScheme::original_becke)
ENTRY(d2w_offdiag_stratmann,  d2w_offdiag_body, false, GridPartitionScheme::stratmann)
ENTRY(d2w_offdiag_stratmann_ra, d2w_offdiag_body, true, GridPartitionScheme::stratmann)
ENTRY(d2w_diag_becke,         d2w_diag_body, false, GridPartitionScheme::original_becke)
ENTRY(d2w_diag_becke_ra,      d2w_diag_body, true,  GridPartitionScheme::original_becke)
ENTRY(d2w_diag_stratmann,     d2w_diag_body, false, GridPartitionScheme::stratmann)
ENTRY(d2w_diag_stratmann_ra,  d2w_diag_body, true,  GridPartitionScheme::stratmann)
