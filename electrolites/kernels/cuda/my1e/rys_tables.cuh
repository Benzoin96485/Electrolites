/*
 * The Rys root/weight tables as device pointers rather than device arrays.
 *
 * GPU4PySCF's lib/gvhf-rys/rys_roots.cuh declares these as `extern __device__
 * double X[]` and defines them in rys_roots_dat.cu, ~1 MB of initialisers that
 * NVRTC would have to parse on every cold cache.  This repository already ships
 * the same tables repacked as kernels/data/rys_roots_dat.npz (see NOTICE), so
 * here they are plain pointers that fast1e.py fills in once per module, and the
 * indexing in rys_roots.cu is unchanged.
 *
 * The five macros are GPU4PySCF's, verbatim.
 */

#pragma once

#define MAX_ORDER       9
#define DEGREE          13
#define DEGREE1         (DEGREE+1)
#define INTERVAL        2.5
#define INTERVALS       40

__device__ double *ROOT_SMALLX_R0;
__device__ double *ROOT_SMALLX_R1;
__device__ double *ROOT_SMALLX_W0;
__device__ double *ROOT_SMALLX_W1;
__device__ double *ROOT_LARGEX_R_DATA;
__device__ double *ROOT_LARGEX_W_DATA;
__device__ double *ROOT_RW_DATA;
