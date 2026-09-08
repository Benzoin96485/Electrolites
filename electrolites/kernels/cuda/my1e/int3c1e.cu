/* Translation unit for the int3c1e kernels.  fast1e.py hands this to NVRTC with
 * -I on kernels/cuda, and _ksplit trims it to the classes a basis can reach. */
#include "my1e/int3c1e_prologue.cuh"
#include "my1e/constant.cu"
#include "my1e/rys_roots.cu"
#include "my1e/g1e.cu"
#include "my1e/g1e_root_1.cu"
#include "my1e/g3c1e.cu"
#include "my1e/int3c1e_wrappers.cu"
