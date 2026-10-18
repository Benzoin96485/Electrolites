"""DLPNO thresholds.

The closed-shell presets follow Psi4 1.11 (``psi4/src/psi4/dlpno/dlpno.cc``,
``read_options.cc``), which in turn follow ORCA's LoosePNO / NormalPNO /
TightPNO for the quantities both programs have:

=============  =========  ==========  =========
               Loose      Normal      Tight
=============  =========  ==========  =========
T_CUT_PNO      1e-6       3.33e-7     1e-7
T_CUT_PAIRS    1e-3       1e-4        1e-5
T_CUT_DO       2e-2       1e-2        5e-3
T_CUT_MKN      1e-3       1e-3        1e-3
=============  =========  ==========  =========

plus the trace / energy completeness criteria of Jiang et al. (JCP 161,
082502, 2024) for the PNOs of the CC and of the preceding LMP2 step.
"""
import copy

DEFAULTS = dict(
    # PNO truncation (CC pairs)
    T_CUT_PNO=3.33e-7, T_CUT_TRACE=0.99, T_CUT_ENERGY=0.99,
    # PNO truncation for the preceding LMP2
    T_CUT_PNO_MP2=3.33e-9, T_CUT_TRACE_MP2=0.999, T_CUT_ENERGY_MP2=0.997,
    T_CUT_PNO_DIAG_SCALE=3e-2, T_CUT_PNO_CORE_SCALE=1e-2,
    # domains
    T_CUT_DO=1e-2, T_CUT_MKN=1e-3, T_CUT_DO_IJ=1e-5, T_CUT_DO_UV=1e-5,
    T_CUT_DO_PRE=3e-2, T_CUT_CLMO=1e-4, T_CUT_CPAO=1e-4, S_CUT=1e-8,
    # pairs
    T_CUT_PAIRS=1e-4, T_CUT_PAIRS_MP2=None, T_CUT_PRE=1e-6,
    F_CUT=1e-5, MIN_PNOS=5,
    # triples
    T_CUT_TNO=1e-9, TRIPLES_MAX_WEAK_PAIRS=1, T_CUT_TNO_STRONG_SCALE=10.0,
    T_CUT_TNO_WEAK_SCALE=100.0, T_CUT_TNO_PRE=1e-7, T_CUT_MKN_TRIPLES_PRE=0.1,
    T_CUT_DO_TRIPLES_PRE=2e-2, T_CUT_TRIPLES_WEAK=1e-8,
    T_CUT_MKN_TRIPLES=1e-2, T_CUT_DO_TRIPLES=1e-2, F_CUT_T=1e-3,
    T_CUT_ITER=1e-5, MIN_TNOS=9,
    # convergence
    E_CONVERGENCE=1e-6, R_CONVERGENCE=1e-6, MAXITER=50, DIIS_MAX_VECS=6,
    # residual measure: largest per-pair RMS (Psi4) or largest element (R_NORM="max", ORCA)
    R_NORM="rms",
    LOCAL_CONVERGENCE=1e-12,
    # DOI grid (radial, angular) -- Psi4 uses 25 radial x 50 Lebedev
    DOI_GRID=(25, 50),
)

PRESETS = {
    'loose': dict(T_CUT_PNO=1e-6, T_CUT_TRACE=0.9, T_CUT_ENERGY=0.9,
                  T_CUT_PAIRS=1e-3, T_CUT_PNO_MP2=1e-8, T_CUT_TRACE_MP2=0.99,
                  T_CUT_ENERGY_MP2=0.99, T_CUT_DO=2e-2, T_CUT_MKN=1e-3),
    'normal': dict(T_CUT_PNO=3.33e-7, T_CUT_TRACE=0.99, T_CUT_ENERGY=0.99,
                   T_CUT_PAIRS=1e-4, T_CUT_PNO_MP2=3.33e-9,
                   T_CUT_TRACE_MP2=0.999, T_CUT_ENERGY_MP2=0.997,
                   T_CUT_DO=1e-2, T_CUT_MKN=1e-3),
    'tight': dict(T_CUT_PNO=1e-7, T_CUT_TRACE=0.999, T_CUT_ENERGY=0.997,
                  T_CUT_PAIRS=1e-5, T_CUT_PNO_MP2=1e-9,
                  T_CUT_TRACE_MP2=0.9999, T_CUT_ENERGY_MP2=0.999,
                  T_CUT_DO=5e-3, T_CUT_MKN=1e-3),
    'very_tight': dict(T_CUT_PNO=1e-8, T_CUT_TRACE=0.999, T_CUT_ENERGY=0.997,
                       T_CUT_PAIRS=1e-6, T_CUT_PNO_MP2=1e-10,
                       T_CUT_TRACE_MP2=0.9999, T_CUT_ENERGY_MP2=0.999,
                       T_CUT_DO=5e-3, T_CUT_MKN=1e-4),
}


def make_options(pno='normal', **overrides):
    """Thresholds for one PNO level, with user overrides applied last."""
    pno = pno.lower().replace('pno', '').strip('_') or 'normal'
    if pno not in PRESETS:
        raise ValueError(f'unknown PNO level {pno!r}; one of {list(PRESETS)}')
    opt = copy.deepcopy(DEFAULTS)
    opt.update(PRESETS[pno])
    unknown = set(overrides) - set(opt)
    if unknown:
        raise KeyError(f'unknown DLPNO option(s) {sorted(unknown)}')
    user_pre = 'T_CUT_PRE' in overrides
    opt.update(overrides)
    if not user_pre:
        opt['T_CUT_PRE'] = min(DEFAULTS['T_CUT_PRE'], 0.01 * opt['T_CUT_PAIRS'])
    if opt['T_CUT_PAIRS_MP2'] is None:
        opt['T_CUT_PAIRS_MP2'] = min(1e-6, 0.1 * opt['T_CUT_PAIRS'])
    opt['PNO'] = pno
    return opt
