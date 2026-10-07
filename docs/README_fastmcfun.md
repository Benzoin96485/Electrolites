# fastmcfun — the multi-collinear XC functional on half the spin samples (A21)

GKS with `collinear='mcol'` (`dft/mcfun_gpu.py`) projects each grid point's
magnetisation onto every direction `n` of a 770-point Lebedev set and
evaluates the collinear functional on `(rho, s = m·n)`, one derivative order
above the one asked for, for every sample: 90–94 % of a patched GKS SCF
(`docs/KERNEL_TARGETS_1.8.1.md` §11.8).

## The symmetry

A Lebedev set is inversion-symmetric: `-n` is a sample whenever `n` is, with
the same weight.  A collinear functional is symmetric under exchanging the
spins, `f(rho, -s) = f(rho, s)`.  So the per-sample term the multi-collinear
functional integrates, `g_n(rho, m) = f(rho, m·n)`, satisfies
`g_{-n} = g_n` *as a function of `(rho, m)`*, and so does every derivative
with respect to `rho` and `m` — which is exactly what `_eval_xc_lebedev`
accumulates per sample into `exc_eff`, `vxc_eff`, `fxc_eff`, `kxc_eff`.
Each antipodal pair contributes twice one member's share, at every order.

`electrolites/fastmcfun.py` replaces `mcfun_gpu._make_sph_samples` with one
member of each pair at twice the weight (checking the pairing and the
weights, and returning the full set otherwise): half the functional
evaluations and half the per-sample contractions.  Nothing else changes; the
collinear-threshold path is untouched.  `FASTMCFUN_OFF=1` keeps all samples.

## Result

GKS B3LYP/def2-SVP, `collinear='mcol'`, `--patch all`, against
`FASTMCFUN_OFF=1`:

| | `mcfun.eval_xc_eff` | SCF |
|---|---|---|
| CuCl₂ (doublet, 15 cycles) | 71.32 → 33.56 s | **79.11 → 41.68 s (1.90x)** |
| FeCl₄⁻ (sextet, 50 cycles, unconverged either way) | 365.94 → 176.80 s | **390.18 → 201.50 s (1.94x)** |

Energies agree to 1e-12 Eh (CuCl₂) and, after 50 identical unconverged
cycles, 4e-10 (FeCl₄⁻).

## Correctness

`tests/test_mcfun.py`: the halved sets are one member of every pair (110,
302, 770, 1202 samples); `exc_eff`, `vxc_eff`, `fxc_eff` from the halved and
the full set on the same densities for LDA, PBE, B3LYP and TPSS, ≤ 4e-15
relative; O₂ triplet GKS (PBE, B3LYP) against `FASTMCFUN_OFF=1`, ≤ 6e-14 Eh.

## Found on the way

GKS with a `-d3bj` functional fails in `scf/hf.py:energy_tot` on stock
GPU4PySCF 1.8.1 (`TypeError: Unsupported type <class 'numpy.ndarray'>`,
the dispersion energy added to a CuPy scalar) — the measurements above use
plain B3LYP for that reason.
