/*
 * extern "C" entry points for the int3c1e kernels.
 *
 * GPU4PySCF launches these from lib/gint/nr_fill_ao_int3c1e.cu, which is host
 * code NVRTC cannot compile, and its kernels are `static` templates that a
 * module lookup by name cannot reach.  So each angular-momentum class the
 * dispatch tables in that file can select gets one `extern "C" __global__`
 * entry point here, and fast1e.py reproduces the dispatch in Python.
 *
 * The two by-value structs are flattened to ints: CuPy marshals scalars and
 * pointers, not structs.  bas_kl and primitive_kl are -1 in every int3c1e
 * launch upstream makes, and no int3c1e kernel reads them.
 */

/* Only the angular-momentum classes a basis can reach are compiled; fast1e.py
 * passes -DE1E_LMAX and falls back to the full module if a driver ever asks for
 * a name the subset left out. */
#ifndef E1E_LMAX
#define E1E_LMAX 4
#endif

#define OFFS(nij, nkl, bij, pij)                    \
    BasisProdOffsets offsets;                       \
    offsets.ntasks_ij = (nij);                      \
    offsets.ntasks_kl = (nkl);                      \
    offsets.bas_ij = (bij);                         \
    offsets.bas_kl = -1;                            \
    offsets.primitive_ij = (pij);                   \
    offsets.primitive_kl = -1;

#define HDO(doff, poff, npair)                      \
    HermiteDensityOffsets hdo;                      \
    hdo.density_offset_of_angular_pair = (doff);    \
    hdo.pair_offset_of_angular_pair = (poff);       \
    hdo.n_pair_of_angular_pair = (npair);

/* ---- the uncontracted integral, (mu nu | 1/|r-C|) for every point -------- */

#define PLAIN_ARGS                                                            \
    double* output, const int ntasks_ij, const int ngrids,                    \
    const int bas_ij, const int primitive_ij, const int nprim_ij,             \
    const int stride_j, const int stride_ij,                                  \
    const int ao_offsets_i, const int ao_offsets_j, const double omega,       \
    const double* grid_points, const double* charge_exponents

#define PLAIN_FIXED(NAME, BODY)                                               \
extern "C" __global__ void NAME(PLAIN_ARGS) {                                 \
    OFFS(ntasks_ij, ngrids, bas_ij, primitive_ij)                             \
    BODY(output, offsets, nprim_ij, stride_j, stride_ij,                      \
         ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents);   \
}

#define PLAIN_GENERAL(NAME, NROOTS, GSIZE)                                    \
extern "C" __global__ void NAME(PLAIN_ARGS, const int i_l, const int j_l) {   \
    OFFS(ntasks_ij, ngrids, bas_ij, primitive_ij)                             \
    GINTfill_int3c1e_kernel_general<NROOTS, GSIZE>(                           \
        output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij,             \
        ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents);    \
}

PLAIN_FIXED(e1e_p00, GINTfill_int3c1e_kernel00)
#if E1E_LMAX >= 1
PLAIN_FIXED(e1e_p10, GINTfill_int3c1e_kernel10)
#endif
#if E1E_LMAX >= 1
PLAIN_GENERAL(e1e_pg2, 2, GSIZE2_INT3C_1E)
#endif
#if E1E_LMAX >= 2
PLAIN_GENERAL(e1e_pg3, 3, GSIZE3_INT3C_1E)
#endif
#if E1E_LMAX >= 3
PLAIN_GENERAL(e1e_pg4, 4, GSIZE4_INT3C_1E)
#endif
#if E1E_LMAX >= 4
PLAIN_GENERAL(e1e_pg5, 5, GSIZE5_INT3C_1E)
#endif

/* ---- contracted with the point charges: the Fock-matrix contribution ----- */

#define CHARGE_ARGS                                                           \
    double* output, const int ntasks_ij, const int ngrids,                    \
    const int bas_ij, const int primitive_ij, const int nprim_ij,             \
    const int stride_j, const int stride_ij,                                  \
    const int ao_offsets_i, const int ao_offsets_j, const double omega,       \
    const double* grid_points, const double* charge_exponents

#define CHARGE_FIXED(NAME, BODY)                                              \
extern "C" __global__ void NAME(CHARGE_ARGS) {                                \
    OFFS(ntasks_ij, ngrids, bas_ij, primitive_ij)                             \
    BODY(output, offsets, nprim_ij, stride_j, stride_ij,                      \
         ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents);   \
}

#define CHARGE_EXPANDED(NAME, LI, LJ)                                         \
extern "C" __global__ void NAME(CHARGE_ARGS) {                                \
    OFFS(ntasks_ij, ngrids, bas_ij, primitive_ij)                             \
    GINTfill_int3c1e_charge_contracted_kernel_expanded<LI, LJ>(               \
        output, offsets, nprim_ij, stride_j, stride_ij,                       \
        ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents);    \
}

#define CHARGE_GENERAL(NAME, NROOTS, GSIZE)                                   \
extern "C" __global__ void NAME(CHARGE_ARGS, const int i_l, const int j_l) {  \
    OFFS(ntasks_ij, ngrids, bas_ij, primitive_ij)                             \
    GINTfill_int3c1e_charge_contracted_kernel_general<NROOTS, GSIZE>(         \
        output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij,             \
        ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents);    \
}

CHARGE_FIXED(e1e_c00, GINTfill_int3c1e_charge_contracted_kernel00)
#if E1E_LMAX >= 1
CHARGE_FIXED(e1e_c10, GINTfill_int3c1e_charge_contracted_kernel10)
#endif
#if E1E_LMAX >= 1
CHARGE_EXPANDED(e1e_c11, 1, 1)
#endif
#if E1E_LMAX >= 2
CHARGE_EXPANDED(e1e_c20, 2, 0)
#endif
#if E1E_LMAX >= 2
CHARGE_EXPANDED(e1e_c21, 2, 1)
#endif
#if E1E_LMAX >= 2
CHARGE_EXPANDED(e1e_c22, 2, 2)
#endif
#if E1E_LMAX >= 3
CHARGE_EXPANDED(e1e_c30, 3, 0)
#endif
#if E1E_LMAX >= 3
CHARGE_EXPANDED(e1e_c31, 3, 1)
#endif
#if E1E_LMAX >= 3
CHARGE_EXPANDED(e1e_c32, 3, 2)
#endif
#if E1E_LMAX >= 4
CHARGE_EXPANDED(e1e_c40, 4, 0)
#endif
#if E1E_LMAX >= 4
CHARGE_EXPANDED(e1e_c41, 4, 1)
#endif
#if E1E_LMAX >= 3
CHARGE_GENERAL(e1e_cg4, 4, GSIZE4_INT3C_1E)
#endif
#if E1E_LMAX >= 4
CHARGE_GENERAL(e1e_cg5, 5, GSIZE5_INT3C_1E)
#endif

/* ---- contracted with the density: the potential on the cavity points ----- */

#define DENS_ARGS                                                             \
    double* output, const double* density,                                    \
    const int density_offset, const int pair_offset, const int n_pair,        \
    const int ntasks_ij, const int ngrids,                                    \
    const int bas_ij, const int primitive_ij, const int nprim_ij,             \
    const double omega, const double* grid_points,                            \
    const double* charge_exponents

#define DENS_FIXED(NAME, BODY)                                                \
extern "C" __global__ void NAME(DENS_ARGS) {                                  \
    OFFS(ntasks_ij, ngrids, bas_ij, primitive_ij)                             \
    HDO(density_offset, pair_offset, n_pair)                                  \
    BODY(output, density, hdo, offsets, nprim_ij, omega,                      \
         grid_points, charge_exponents);                                      \
}

#define DENS_GENERAL(NAME, LSUM)                                              \
extern "C" __global__ void NAME(DENS_ARGS) {                                  \
    OFFS(ntasks_ij, ngrids, bas_ij, primitive_ij)                             \
    HDO(density_offset, pair_offset, n_pair)                                  \
    GINTfill_int3c1e_density_contracted_kernel_general<LSUM>(                 \
        output, density, hdo, offsets, nprim_ij, omega,                       \
        grid_points, charge_exponents);                                       \
}

DENS_FIXED(e1e_d0, GINTfill_int3c1e_density_contracted_kernel00)
#if E1E_LMAX >= 1
DENS_FIXED(e1e_d1, GINTfill_int3c1e_density_contracted_kernel10)
#endif
#if E1E_LMAX >= 1
DENS_GENERAL(e1e_d2, 2)
#endif
#if E1E_LMAX >= 2
DENS_GENERAL(e1e_d3, 3)
#endif
#if E1E_LMAX >= 2
DENS_GENERAL(e1e_d4, 4)
#endif
#if E1E_LMAX >= 3
DENS_GENERAL(e1e_d5, 5)
#endif
#if E1E_LMAX >= 3
DENS_GENERAL(e1e_d6, 6)
#endif
#if E1E_LMAX >= 4
DENS_GENERAL(e1e_d7, 7)
#endif
#if E1E_LMAX >= 4
DENS_GENERAL(e1e_d8, 8)
#endif
