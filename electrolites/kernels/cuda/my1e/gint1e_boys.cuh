/*
 * Far-field fast path for the order-0 and order-1 Boys functions that drive the
 * three-centre one-electron integrals (int3c1e: PCM, QM/MM, ESP, CHELPG).
 *
 * ---------------------------------------------------------------------------
 * Why a far field exists here at all
 *
 * For one primitive pair with Gaussian product exponent p = aij at centre P, a
 * point C carrying a Gaussian charge of exponent q, and the Boys argument
 *
 *     x = a0 * r2,   a0 = p*q/(p+q) * theta,   r2 = |P - C|^2,
 *
 * the integral needs erf(sqrt(x)) and exp(-x).  Above x = 40 both are constant
 * to double precision:
 *
 *     erfc(sqrt(40)) = 3.7e-19  -> relative error of F_0 is 3.7e-19  (0.002 ulp)
 *     exp(-40)       = 4.2e-18  -> enters as exp(-x)/F_0(x) = 3.0e-17 (0.13 ulp)
 *
 * so this is not an approximation with an accuracy knob: both terms are below
 * one unit in the last place of the fp64 result they modify.
 *
 * ---------------------------------------------------------------------------
 * What that buys
 *
 * With erf -> 1 the whole charge-smearing factor cancels analytically.  Writing
 * f = q/(p+q) and theta for the range-separation factor,
 *
 *     prefactor * F_0 = (2*pi*eij/p) * sqrt(theta) * sqrt(f) * sqrt(pi/4) / sqrt(p*f*theta*r2)
 *                     = 2*pi*sqrt(pi/4) * eij / (p * sqrt(p*r2))
 *                     = 2*pi*SQRTPIE4 * eij * rsqrt(p^3 * r2),
 *
 * independent of q and of theta -- which is just the statement that far from a
 * smeared charge only the total charge is visible.  One rsqrt and three
 * multiplies replace erf, exp, two sqrt and three fp64 divisions.  The order-1
 * term follows from R_1 = -a0*R_0/x = -R_0/r2, again with no reference to q.
 *
 * The far-field test itself is written division-free:
 *     a0*r2 > X  <=>  p*q*r2 > X*(p+q)     (theta == 1, i.e. omega == 0)
 *
 * On a PCM cavity at the GPU4PySCF default (lebedev_order = 29) 92 % of all
 * (primitive pair, cavity point) tasks are in this far field, because the cavity
 * surrounds the whole molecule while each shell pair is local to one bond.
 */

#pragma once

#ifndef SQRTPIE4
#define SQRTPIE4        .8862269254527580136
#endif

#define GINT_ERF_ASYMPTOTIC_X 40.0

__device__
static inline bool GINT_int3c1e_far_field(const double aij, const double charge_exponent,
                                          const double omega, const double r2)
{
    if (omega > 0.0) {
        return false;
    }
    if (charge_exponent > 0.0) {
        return aij * charge_exponent * r2 > GINT_ERF_ASYMPTOTIC_X * (aij + charge_exponent);
    }
    return aij * r2 > GINT_ERF_ASYMPTOTIC_X;
}

/* P0 = prefactor * F_0(x) */
__device__
static inline double GINT_int3c1e_boys0(const double aij, const double eij,
                                        const double charge_exponent, const double omega,
                                        const double r2)
{
    if (GINT_int3c1e_far_field(aij, charge_exponent, omega, r2)) {
        return (2.0 * M_PI * SQRTPIE4) * eij * rsqrt(aij * aij * aij * r2);
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
    if (boys_input > 1e-14) {
        const double sqrt_boys_input = sqrt(boys_input);
        return prefactor * (SQRTPIE4 / sqrt_boys_input * erf(sqrt_boys_input));
    }
    return prefactor;
}

/* P0 = prefactor * F_0(x), P1 = prefactor * R_1 with R_1 = -a0*(F_0 - exp(-x))/x */
__device__
static inline void GINT_int3c1e_boys01(double& P0, double& P1,
                                       const double aij, const double eij,
                                       const double charge_exponent, const double omega,
                                       const double r2)
{
    if (GINT_int3c1e_far_field(aij, charge_exponent, omega, r2)) {
        P0 = (2.0 * M_PI * SQRTPIE4) * eij * rsqrt(aij * aij * aij * r2);
        P1 = -P0 / r2;
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
    if (boys_input > 1e-14) {
        const double sqrt_boys_input = sqrt(boys_input);
        const double R000_0 = SQRTPIE4 / sqrt_boys_input * erf(sqrt_boys_input);
        P0 = prefactor * R000_0;
        P1 = prefactor * (-a0 * (R000_0 - exp(-boys_input)) / boys_input);
    } else {
        P0 = prefactor;
        P1 = prefactor * (-a0 / 3);
    }
}
