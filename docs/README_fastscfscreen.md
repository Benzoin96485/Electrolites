# fastscfscreen — a loose early screen in direct SCF, then a tight rebuild (A20)

GPU4PySCF's direct SCF already builds J and K incrementally from
`dm − dm_last` (`dft/rks.py:get_veff` with `vhf_last.vj`), but at the
converged precision, `direct_scf_tol` = 1e-13, from the first cycle — when the
density is still wrong in the third digit.  Per-cycle `get_k` falls only ~3x
over a taxol SCF (0.84 → 0.27 s) and ~3x over PfPMT's (3.3 → 0.96 s).

`electrolites/fastscfscreen.py`: while `max|dm − dm_last|` is above 1e-3,
the J/K builds screen at 1e-10; at the first cycle below it the Fock matrix
is rebuilt from scratch at `direct_scf_tol` and the SCF continues
incrementally and tight.  The last full build and every increment after it
are at the original threshold, so what the converged answer is made of is
unchanged; the loose cycles only shape the path.  The wrapper sits on the
SCF classes' `get_veff` (RKS, UKS, RHF, UHF) and raises `direct_scf_tol` on
the SCF object and the J/K option objects it holds (and `fastk`'s
full-range sibling) for the duration of a loose call.  Density-fitted SCF is
untouched.  `FASTSCFSCREEN_OFF=1`; `FASTSCFSCREEN_LOOSE`,
`FASTSCFSCREEN_SWITCH` set the two thresholds.

## Result

B3LYP/def2-SVP, `conv_tol` 1e-9, `--patch all`, against `FASTSCFSCREEN_OFF=1`:

| | SCF | cycles | ΔE |
|---|---|---|---|
| penicillin V | 5.35 → 5.21 s | 14 / 14 | 1e-10 |
| Ni(II) dithiolene⁻ (UKS) | 21.28 → 20.28 s | 35 / 35 | 1e-10 |
| taxol | 16.45 → **13.81 s (1.19x)** | 15 / 15 | 1.6e-10 |
| PfPMT⁻ (284 atoms) | 64.67 → **51.99 s (1.24x)** | 21 / 18 | 7e-9 |

PfPMT's 7e-9 is within its SCF noise at this `conv_tol`: a tight-throughout
run at `conv_tol` 1e-11 had not converged after 50 cycles and sat 2.6e-7
below both.  A loose threshold of 1e-9 moved taxol's energy by 1e-8 and was
rejected; switching at 1e-4 instead of 1e-3 was 0.7 s faster on taxol with a
3e-10 shift, and 1e-3 is kept.

## Correctness

`tests/test_scfscreen.py`: RKS B3LYP, UKS B3LYP, RHF and RKS ωB97X on alanine
dipeptide at `conv_tol` 1e-10 against `FASTSCFSCREEN_OFF=1`: ≤ 3.6e-11 Eh,
each run with loose builds and exactly one rebuild.
