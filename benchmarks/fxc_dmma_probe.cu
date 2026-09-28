// W36: a plain fp64 tensor-core GEMM (mma.m8n8k4), row-major
// C[M][N] = A[M][K] B[K][N], the shape of W33's Y GEMM -- the kernel a
// fused density epilogue or W-generating prologue would have to be built on.
// It is not used by `electrolites/`: `benchmarks/fxc_dmma_probe.py` times it
// against cuBLAS, which is the gate `docs/README_fastxcfxclocal.md` 3 prices
// the fusion with.  16-byte cp.async (lda, ldb even; K and N tails
// zero-filled per pair), an NST-stage pipeline, BM x BN block tile,
// WM x WN warp tile.
#ifndef BM
#define BM 128
#endif
#ifndef BN
#define BN 128
#endif
#ifndef WM
#define WM 64
#endif
#ifndef WN
#define WN 32
#endif
#ifndef BK
#define BK 16
#endif
#ifndef NST
#define NST 3
#endif
#define NWM (BM / WM)
#define NWN (BN / WN)
#define NT (NWM * NWN * 32)
#define TM (WM / 8)
#define TN (WN / 8)
#define SA (BK + 4)
#define SB (BN + 4)

__device__ __forceinline__ void cpa16(double *s, const double *g, int bytes)
{
    unsigned sa = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(sa), "l"(g), "r"(bytes));
}

extern "C" __global__ void __launch_bounds__(NT)
dmma_nn(double *C, const double *A, const double *B, int M, int N, int K,
        int lda, int ldb, int ldc, double beta)
{
    extern __shared__ double sm[];
    double *As = sm;
    double *Bs = sm + NST * BM * SA;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / NWN, wn = warp % NWN;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    double acc[TM][TN][2];
#pragma unroll
    for (int a = 0; a < TM; ++a)
#pragma unroll
        for (int b = 0; b < TN; ++b) acc[a][b][0] = acc[a][b][1] = 0.;

    auto load = [&](int st, int k0) {
        double *as = As + st * BM * SA, *bs = Bs + st * BK * SB;
#pragma unroll
        for (int e = tid; e < BM * BK / 2; e += NT) {
            int m = e / (BK / 2), k = 2 * (e % (BK / 2));
            int rem = K - (k0 + k);
            int by = (m0 + m < M) ? (rem >= 2 ? 16 : (rem == 1 ? 8 : 0)) : 0;
            const double *g = by ? A + (size_t)(m0 + m) * lda + k0 + k : A;
            cpa16(as + m * SA + k, g, by);
        }
#pragma unroll
        for (int e = tid; e < BK * BN / 2; e += NT) {
            int k = e / (BN / 2), n = 2 * (e % (BN / 2));
            int rem = N - (n0 + n);
            int by = (k0 + k < K) ? (rem >= 2 ? 16 : (rem == 1 ? 8 : 0)) : 0;
            const double *g = by ? B + (size_t)(k0 + k) * ldb + n0 + n : B;
            cpa16(bs + k * SB + n, g, by);
        }
    };
    const int nk = (K + BK - 1) / BK;
#pragma unroll
    for (int s = 0; s < NST - 1; ++s) {
        if (s < nk) load(s, s * BK);
        asm volatile("cp.async.commit_group;\n");
    }
    for (int t = 0; t < nk; ++t) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(NST - 2));
        __syncthreads();
        int tn = t + NST - 1;
        if (tn < nk) load(tn % NST, tn * BK);
        asm volatile("cp.async.commit_group;\n");
        const double *as = As + (t % NST) * BM * SA + (wm * WM) * SA;
        const double *bs = Bs + (t % NST) * BK * SB + wn * WN;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 4) {
            double af[TM], bf[TN];
#pragma unroll
            for (int a = 0; a < TM; ++a)
                af[a] = as[(a * 8 + (lane >> 2)) * SA + kk + (lane & 3)];
#pragma unroll
            for (int b = 0; b < TN; ++b)
                bf[b] = bs[(kk + (lane & 3)) * SB + b * 8 + (lane >> 2)];
#pragma unroll
            for (int a = 0; a < TM; ++a)
#pragma unroll
                for (int b = 0; b < TN; ++b)
                    asm volatile(
                        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                        "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
                        : "+d"(acc[a][b][0]), "+d"(acc[a][b][1])
                        : "d"(af[a]), "d"(bf[b]));
        }
    }
#pragma unroll
    for (int a = 0; a < TM; ++a)
#pragma unroll
        for (int b = 0; b < TN; ++b) {
            int m = m0 + wm * WM + a * 8 + (lane >> 2);
            int n = n0 + wn * WN + b * 8 + 2 * (lane & 3);
            if (m < M) {
                double *c = C + (size_t)m * ldc + n;
                if (n < N) c[0] = acc[a][b][0] + (beta != 0. ? beta * c[0] : 0.);
                if (n + 1 < N) c[1] = acc[a][b][1] + (beta != 0. ? beta * c[1] : 0.);
            }
        }
}
