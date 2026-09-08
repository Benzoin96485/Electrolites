/*
 * Far-field fast path for the Rys roots/weights of the three-centre one-electron
 * integrals with two or more roots (int3c1e for (pp|, (dp|, (dd| ... ).
 *
 * GINTrys_root<NROOTS> already has a large-x branch, taken for
 * x > 35 + 5*NROOTS, in which
 *
 *     r_k = R_k / x,   w_k = W_k * sqrt(pi/(4x)),   u_k = r_k / (1 - r_k)
 *
 * with R_k, W_k the tabulated Hermite data.  Substituting those exact forms into
 * what GINT_g1e does with them makes the whole charge-smearing/range-separation
 * algebra cancel.  With f = q/(p+q), p = aij, a0 = p*f*theta and x = a0*r2:
 *
 *     u2 = a0*u_k = a0*R_k/(x - R_k)
 *     qt2_over_p_plus_q = u2/(u2 + p*f)*f = R_k/(p*r2)               (theta = 1)
 *     prefactor * w_k   = 2*pi*SQRTPIE4*eij*W_k*rsqrt(p^3*r2)
 *
 * Neither expression refers to q or theta any more -- far from a smeared charge
 * only its total charge is visible.  Because the branch is taken at the
 * baseline's own threshold, these are the same numbers the baseline computes,
 * reached with 2 fp64 divisions instead of 4 + 3*NROOTS divisions/square roots
 * and with no table lookup.
 *
 * prefactor is folded into gz (returned as pw = prefactor*w) instead of gy.  The
 * three axis recursions are separately linear, so moving a scalar from one axis
 * to another leaves every gx*gy*gz product unchanged.
 */

#pragma once

#define GINT_G1E_LARGEX_X(NROOTS) (35.0 + (NROOTS) * 5.0)

template <int NROOTS>
__device__
static inline void GINT_g1e_roots(double* __restrict__ qt2, double* __restrict__ pw,
                                  const double aij, const double eij,
                                  const double charge_exponent, const double omega,
                                  const double r2)
{
    constexpr int off = NROOTS * (NROOTS - 1) / 2;
    constexpr double xlarge = GINT_G1E_LARGEX_X(NROOTS);

    if (omega <= 0.0 &&
        (charge_exponent > 0.0
            ? aij * charge_exponent * r2 > xlarge * (aij + charge_exponent)
            : aij * r2 > xlarge)) {
        const double inv_aij_r2 = 1.0 / (aij * r2);
        const double t = (2.0 * M_PI * SQRTPIE4) * eij * rsqrt(aij * aij * aij * r2);
#pragma unroll
        for (int k = 0; k < NROOTS; k++) {
            qt2[k] = ROOT_LARGEX_R_DATA[off + k] * inv_aij_r2;
            pw[k] = ROOT_LARGEX_W_DATA[off + k] * t;
        }
        return;
    }

    double a0 = aij;
    const double q_over_p_plus_q = charge_exponent > 0.0 ? charge_exponent / (aij + charge_exponent) : 1.0;
    const double sqrt_q_over_p_plus_q = charge_exponent > 0.0 ? sqrt(q_over_p_plus_q) : 1.0;
    a0 *= q_over_p_plus_q;
    const double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0;
    const double sqrt_theta = omega > 0.0 ? sqrt(theta) : 1.0;
    a0 *= theta;

    const double prefactor = 2.0 * M_PI / aij * eij * sqrt_theta * sqrt_q_over_p_plus_q;
    const double boys_input = a0 * r2;
    double uw[NROOTS * 2];
    GINTrys_root<NROOTS>(boys_input, uw);
    GINTscale_u<NROOTS>(uw, theta);

#pragma unroll
    for (int k = 0; k < NROOTS; k++) {
        const double u2 = a0 * uw[k];
        qt2[k] = u2 / (u2 + aij * q_over_p_plus_q) * q_over_p_plus_q;
        pw[k] = prefactor * uw[NROOTS + k];
    }
}
