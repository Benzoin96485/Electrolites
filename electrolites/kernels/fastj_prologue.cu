// ---------------------------------------------------------------------------
// Fast Coulomb-matrix (J) build kernels for GPU4PySCF's McMurchie-Davidson
// J engine.
//
// The integral arithmetic -- the Hermite (Rt) recurrences and the density
// contraction -- is lifted verbatim from GPU4PySCF's generated file
// gvhf-md/unrolled_md_j.cu, so both codes evaluate the same expressions in
// the same order.  What changes is the scaffolding:
//
//   * the incomplete-gamma (Boys) values live in registers.  GPU4PySCF stages
//     them through the dynamic shared-memory buffer with a stride of 256, so
//     every one of the ~O(1) to O(50) reads per shell-pair quartet is an LDS.
//   * no double-precision division or sqrt in the innermost loop.  GPU4PySCF
//     evaluates fac/(aij*akl*sqrt(aij+akl)) and aij*akl/(aij+akl) once per
//     quartet -- two DP divisions and a DP sqrt, each ~20 instructions on
//     sm_80.  1/aij is uniform over a bra tile and 1/akl over a ket tile, so
//     both are cached alongside the product centres, one rsqrt(aij+akl)
//     supplies 1/(aij+akl) and 1/sqrt(aij+akl), and every division becomes a
//     multiply.
//   * omega == 0 (plain Coulomb) is a compile-time fact, so the range-
//     separation branches of boys_fn disappear along with their runtime test.
//   * the downward recursion of the incomplete gamma function multiplies by
//     tabulated reciprocals of half-integers instead of dividing, and the
//     large-argument branch takes one rsqrt in place of a sqrt and two
//     divisions.
//   * occupancy is set per angular-momentum class.  GPU4PySCF compiles the
//     whole family with __maxnreg__(128), which pins every class at 2 blocks
//     per SM; how many registers a class actually needs is a property of the
//     class, so each gets its own __launch_bounds__.
//
// All arithmetic is double precision.
// ---------------------------------------------------------------------------

#define BAS_SLOTS       8
#define ATOM_OF         0
#define ANG_OF          1
#define NPRIM_OF        2
#define PTR_EXP         5
#define PTR_COEFF       6
#define PTR_BAS_COORD   7

#define PI_FAC          34.98683665524972497
#define SQRTPIE4        .886226925452758013
#define EPS_FLOAT64     2.220446049250313e-16

// 1/(i+1/2) for i = 0..31.  The downward recursion of the incomplete gamma
// function divides by m+1/2, m-1/2, ... , 1/2 and the ascending series divides
// by m+1/2, m+3/2, ... ; all of these are half-integers, so the reciprocals
// are a table lookup rather than a DP division.  Each entry is the correctly
// rounded double, i.e. exactly what 1./(i+.5) evaluates to.
__device__ __constant__ static double c_inv_half[36] = {
    2.0000000000000000e+00, 6.6666666666666663e-01, 4.0000000000000002e-01,
    2.8571428571428570e-01, 2.2222222222222221e-01, 1.8181818181818182e-01,
    1.5384615384615385e-01, 1.3333333333333333e-01, 1.1764705882352941e-01,
    1.0526315789473684e-01, 9.5238095238095233e-02, 8.6956521739130432e-02,
    8.0000000000000002e-02, 7.4074074074074070e-02, 6.8965517241379309e-02,
    6.4516129032258065e-02, 6.0606060606060608e-02, 5.7142857142857141e-02,
    5.4054054054054057e-02, 5.1282051282051280e-02, 4.8780487804878049e-02,
    4.6511627906976744e-02, 4.4444444444444446e-02, 4.2553191489361701e-02,
    4.0816326530612242e-02, 3.9215686274509803e-02, 3.7735849056603772e-02,
    3.6363636363636362e-02, 3.5087719298245612e-02, 3.3898305084745762e-02,
    3.2786885245901641e-02, 3.1746031746031744e-02, 3.0769230769230771e-02,
    2.9850746268656716e-02, 2.8985507246376812e-02, 2.8169014084507043e-02,
};

// exp(-t) is below 2^-53 relative to the leading term for t beyond this, so
// the ascending recursion of the large-argument branch can drop the exp() and
// the erf() together: erf(sqrt(t)) = 1 to within 1e-16 there.
#define GAMMA_INC_ASYMPTOTIC  36.

// ---------------------------------------------------------------------------
// F_n(t), n = 0..ORDER, in registers.  Same three branches, same recurrences
// and the same branch boundaries as gvhf-rys/gamma_inc.cu with block_size = 1.
// ---------------------------------------------------------------------------
template <int ORDER> __device__ __forceinline__
static void eval_gamma_inc_reg(double *f, double t)
{
    if (t < EPS_FLOAT64) {
        f[0] = 1.;
#pragma unroll
        for (int i = 1; i <= ORDER; i++) {
            f[i] = 1./(2*i+1);
        }
    } else if (ORDER > 0 && t < ORDER*.5+.5) {
        // ascending series for F_ORDER, then downward recursion
        double bi = ORDER + .5;
        double e = .5 * exp(-t);
        double x = e;
        double s = e;
        double tol = EPS_FLOAT64 * e;
        int k = ORDER;
        while (x > tol) {
            bi += 1.;
            k += 1;
            x *= t * (k < 36 ? c_inv_half[k] : 1./bi);
            s += x;
        }
        double fval = s * c_inv_half[ORDER];
        f[ORDER] = fval;
#pragma unroll
        for (int i = ORDER-1; i >= 0; i--) {
            fval = (e + t * fval) * c_inv_half[i];
            f[i] = fval;
        }
    } else {
        // one rsqrt supplies 1/sqrt(t) and 1/t
        double rt = rsqrt(t);
        double fval;
        double e;
        if (t > GAMMA_INC_ASYMPTOTIC) {
            fval = SQRTPIE4 * rt;
            e = 0.;
        } else {
            fval = SQRTPIE4 * rt * erf(t * rt);
            e = .5 * exp(-t);
        }
        f[0] = fval;
        if (ORDER > 0) {
            double b = rt * rt;
            double b1 = .5;
#pragma unroll
            for (int i = 1; i <= ORDER; i++) {
                fval = b * (b1 * fval - e);
                f[i] = fval;
                b1 += 1.;
            }
        }
    }
}

// F_n(theta*rr) scaled the way md_j wants it, for omega == 0 only.
template <int ORDER> __device__ __forceinline__
static void boys0_fn_reg(double *out, double theta, double rr, double fac)
{
    eval_gamma_inc_reg<ORDER>(out, theta * rr);
    out[0] *= fac;
    double a2 = -2. * theta;
#pragma unroll
    for (int n = 1; n <= ORDER; n++) {
        fac *= a2;
        out[n] *= fac;
    }
}
