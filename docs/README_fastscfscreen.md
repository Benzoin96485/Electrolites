# fastscfscreen — a loose early screen in direct SCF, then a tight rebuild (A20)

GPU4PySCF's direct SCF already builds J and K incrementally from
`dm − dm_last` (`dft/rks.py:get_veff` with `vhf_last.vj`), but at the
converged precision, `direct_scf_tol` = 1e-13, from the first cycle — when the
density is still wrong in the third digit.  Per-cycle `get_k` falls only ~3x
over a taxol SCF (0.84 → 0.27 s) and ~3x over PfPMT's (3.3 → 0.96 s).

`electrolites/fastscfscreen.py`: while the density step
`max|S^½ (dm − dm_last) S^½|` is above 1e-3, the J/K builds screen at 1e-10;
at the first cycle below it, or after 30 loose builds, the Fock matrix is
rebuilt from scratch at `direct_scf_tol` and the SCF continues incrementally
and tight.  The last full build and every increment after it are at the
original threshold, so what the converged answer is made of is unchanged;
the loose cycles only shape the path.  The wrapper sits on the SCF classes'
`get_veff` (RKS, UKS, RHF, UHF) and raises `direct_scf_tol` on the SCF
object and the J/K option objects it holds (and `fastk`'s full-range
sibling) for the duration of a loose call.  Density-fitted SCF is
untouched.  `FASTSCFSCREEN_OFF=1`; `FASTSCFSCREEN_LOOSE`,
`FASTSCFSCREEN_SWITCH`, `FASTSCFSCREEN_MAX_LOOSE` set the two thresholds
and the cap.

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

(These rows were measured with the switch on the AO step `max|dm − dm_last|`;
the next section is why it moved to the Loewdin basis.  Re-run with the
Loewdin step, penicillin V and taxol switch on the same cycle.)

## The switch criterion with diffuse functions

The first version switched on `max|dm − dm_last|` in the AO basis.  With
diffuse functions that number does not fall with the SCF error: the overlap
is near singular (GPU4PySCF warns of a low-rank overlap), and AO density
elements carry the near-linear-dependent combinations with large
coefficients of opposite sign.  On the PfPMT QM region (OPE + SAM, 64 atoms,
charge −1, ωB97M-V/def2-TZVPD, `conv_tol` 1e-10) the AO step was 70–400
times the Loewdin step, and still 1e-3 at `|g|` 3e-6.

The switch then never came, and waiting is not harmless: a loose build drops
quartets below 1e-10 in absolute terms, not relative to the increment, and
the dropped part accumulates in `vhf` until the rebuild.  The energy of the
incrementally built Fock matrix sank up to 1.6e-5 Eh below the converged
minimum, while a from-scratch build at the same densities stayed above it;
`|g|` sat at 1–7e-4 from cycle 30 to 80, and the density kept moving enough
to keep the screen loose.  The SCF only escaped when an AO step happened to
fall under 1e-3 (vacuum cycle 83 of 95, embedded 89), after which DIIS
spent cycles on its polluted history.  With every other Electrolites patch
on and this one off, cycle counts and energies matched stock GPU4PySCF
exactly.

The step is now the largest element of `S^½ (dm − dm_last) S^½`, the
density change in the orthonormal Loewdin basis (`S^½` from one `eigh` of
the overlap per SCF run).  On the same frame it crosses 1e-3 at cycle 25 of
37, where the SCF leaves its erratic early phase (`|g|` 1.4e-3; the AO step
is 9e-2 there), and at cycle 6 of the embedded SCF.  As a backstop, 30
loose builds end the loose phase whatever the step (`STATS['capped']`).

ωB97M-V/def2-TZVPD, VV10 on, the PfPMT QM region, a vacuum SCF from the
default guess and then the point-charge-embedded SCF from its density
(cycles vacuum / embedded):

| | tight throughout | AO switch | Loewdin switch |
|---|---|---|---|
| frame 0 | 37 / 14 | 84 / 72 (95 / >96 in a re-run) | **37 / 14** |
| frame 69 | 41 / 14 | 121 / 67 | **43 / 14** |

On frame 0, the Loewdin switch's cycles 2–37 take 433 s against 509 s tight
(1.18x), and the embedded SCF 150 s against 160 s; the energies differ by
1.1e-9 and 2.3e-10 Eh.  Across the 66 production frames run both ways with
the AO switch, the SCF took a median 113 cycles against stock's 56.  The
converged energies were still right (the final builds were tight); the cost
was time.

Where the basis is not near singular the two criteria agree to a cycle or
two, earlier as often as later; cycles and walls are unchanged at
B3LYP/def2-SVP (penicillin V 14, taxol 15), B3LYP/def2-TZVPD (penicillin V
16), B3LYP/aug-cc-pVTZ (glycine 12, alanine dipeptide 13) and
ωB97M-V/def2-TZVPD (glycine 12, alanine dipeptide 13; glycine def2-QZVPPD
12).

The cap is what ends the loose phase when the SCF stays erratic past 30
cycles.  The same QM region in B3LYP/def2-SVPD is such a case: the tight SCF
takes 77 cycles, and the run with the Loewdin switch follows it to 1e-4 Eh
through the 30 loose builds, then passes through the same +25 Eh excursion
at cycle 31 and drifts onto another path, converging at 103 (102 with the
cap at 10; 122 with the AO switch).  That spread is the SCF's, not the
screen's: with no loose phase at all and `direct_scf_tol` 1e-12 it takes
113 cycles, at 3e-13 77, and with the loose threshold at 1e-11 or 1e-12 67
and 78.  All end within 5e-10 Eh of the tight run's energy except the
1e-12 tight run (1.3e-7 above, its own threshold).

A note on `conv_tol` below 1e-10: alanine dipeptide in B3LYP/def2-TZVPD at
1e-11 converges or not by a hair whatever the screen.  Stock GPU4PySCF, the
patches with this one off, and this one all reach `|g|` 4e-8 by cycle 16 and
then rise by 2.2e-10 Eh per cycle, the incremental Fock build's own drift;
whether `|ΔE|` < 1e-11 happens before that (14 cycles) or not (no
convergence) is luck.  The tests stay at 1e-10.

## Correctness

`tests/test_scfscreen.py`: RKS B3LYP, UKS B3LYP, RHF and RKS ωB97X on alanine
dipeptide at `conv_tol` 1e-10 against `FASTSCFSCREEN_OFF=1`: ≤ 3.6e-11 Eh,
each run with loose builds, exactly one rebuild, and at most one cycle more
than the tight SCF.  Two more cases: the cap (`FASTSCFSCREEN_SWITCH=0`,
`FASTSCFSCREEN_MAX_LOOSE=3`: three loose builds, then the rebuild, 1.2e-11
Eh), and ωB97M-V/def2-TZVPD (13 cycles both ways, 5.7e-12 Eh).
