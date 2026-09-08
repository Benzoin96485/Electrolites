/*
 * NVRTC prologue for the three-centre one-electron integrals.
 *
 * NVRTC compiles a string and resolves #include only against the paths it is
 * given, and it has no <math.h>, <stdint.h> or <stdlib.h>.  So the declarations
 * the int3c1e path needs out of GPU4PySCF's lib/gint/{gint.h, config.h,
 * gint1e.h} are reproduced here instead of those headers being included.
 *
 * BasisProdCache in particular is copied VERBATIM and its layout is
 * load-bearing: fast1e.py memcpy's GPU4PySCF's own host-side struct into
 * c_bpcache below, so the two definitions must agree field for field.  fast1e
 * checks that at run time (the nbas field, and a12/x12 against values recomputed
 * from the molecule) and falls back to GPU4PySCF if they do not.
 */

#pragma once

typedef short int16_t;
typedef unsigned long long uint64_t;

#define M_PI            3.1415926535897932384626433832795028
#define SQRTPIE4        .8862269254527580136
#define PIE4            .7853981633974483096

#define GPU_LMAX        6
#define TOT_NF          (1+3+6+10+15+21+28)

/* gint1e.h */
#define MAX_NROOTS_INT3C_1E 5
#define GSIZE1_INT3C_1E 6
#define GSIZE2_INT3C_1E 36
#define GSIZE3_INT3C_1E 108
#define GSIZE4_INT3C_1E 240
#define GSIZE5_INT3C_1E 450

/* gint.h, verbatim */
typedef struct {
        int ntasks_ij;
        int ntasks_kl;
        int bas_ij;
        int bas_kl;
        int primitive_ij;
        int primitive_kl;
} BasisProdOffsets;

typedef struct {
    int density_offset_of_angular_pair;
    int pair_offset_of_angular_pair;
    int n_pair_of_angular_pair;
} HermiteDensityOffsets;

typedef struct {
    int l_bra;
    int l_ket;
    int nprim_12;
    int npairs;
} ContractionProdType;

typedef struct {
    int nbas;
    int ncptype;
    ContractionProdType *cptype;
    int *bas_pairs_locs;
    int *primitive_pairs_locs;
    int *bas_pair2shls;
    double *aexyz;

    // Data below held on GPU global memory
    double *bas_coords;
    int *bas_atm;
    int *bas_pair2bra;
    int *bas_pair2ket;
    int *ao_loc;
    double *a12;
    double *e12;
    double *x12;
    double *y12;
    double *z12;
    double *a1;
    double *a2;
} BasisProdCache;

/* constant.cu's three symbols.  c_bpcache is written from the host by
 * fast1e.py; c_idx and c_l_locs are compile-time tables and live in
 * my1e/constant.cu. */
extern __constant__ BasisProdCache c_bpcache;
extern __constant__ int c_idx[TOT_NF*3];
extern __constant__ int c_l_locs[GPU_LMAX+2];
