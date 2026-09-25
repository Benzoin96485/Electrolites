# A density screen for the three-centre Coulomb build — measured, built, and off

`docs/DF_WORK_LIST.md` **W20**, found by W9.  `J` is linear in `D`, so
`J(D) = J(D_last) + J(D − D_last)` exactly, and GPU4PySCF's four-centre MD J
engine screens on a density-weighted Schwarz bound; the three-centre build has
no density screen of any kind.  The entry asks for the pair count through an
SCF first, and warns that the falsification is the filtered list costing more
to rebuild each cycle than the pairs it removes save.

That is what happened, for a reason the entry could not have priced: **W19
made the thing being screened 3.4x–5.0x cheaper first.**  The screen is
implemented, correct and off by default (`FASTDFJ3C_INCREMENTAL=1`), and on
every whole run measured it is a loss or a wash against W19 alone.

---

## 0. Summary

* **How much a screen can drop** (`benchmarks/df_j3c_screen_probe.py`, every
  Coulomb build of a converging DF-PBE SCF, the Schwarz bounds the build
  already has).  At a threshold of 1e-11 on the bound of what a dropped pair
  contributes, the **density contraction** keeps 80 → 38 % of its pair work
  from the first cycle to the last at 113 atoms (88 → 55 % at 42), and the
  **auxiliary-vector contraction** keeps essentially all of its auxiliary
  shells and 73–100 % of its pairs: the fitted coefficients' difference is
  small everywhere and local nowhere, because the metric solve couples every
  auxiliary function to every other.  At 284 atoms the same screen keeps 57–78 %
  of the density work on a DF-PBE SCF that does not converge in 50 cycles.
* **What it is worth on a whole run**: against W19's kernels alone, 0.95x at
  42 atoms, 0.96x at 113, 0.89x on the Cu macrocycle, 0.97x over a patched
  baseline and 1.00x on a 284-atom run — never a gain, with the energy moved by 6e-10 to 7e-9 Eh and
  one extra SCF cycle at 113 atoms and three on the Cu complex (§2).
* **Why**: at 113 atoms W19's two kernels are 24.8 ms a cycle, so dropping
  46 % of one and 13 % of the other saves at most 7.4 ms a cycle, 0.10 s over
  the 14 builds made from the difference — and the extra cycle costs 0.33 s.
  At 42 atoms the saving is under 1 ms a cycle and the host work of the
  screen (the shell-block maxima, two device-to-host reductions, the gather,
  a block table) is about ten.  Against GPU4PySCF's kernels the same screen
  would have saved 5–6x more per cycle; against these it cannot pay for
  itself.
* **What would make it pay** is not a better list but a tighter bound on the
  auxiliary side, and that is a kernel-side screen (per `(pair block,
  auxiliary block)`) rather than a host-side one; recorded as W26.

---

## 1. The measurement

Fractions of each kernel's work — pairs weighted by `nf3(lij)`, auxiliary
shells by `nf3(lk)` — that a screen at `τ` keeps, first and last cycle:

| | τ | density, first → last | aux shells | aux pairs, first → last |
|---|---|---|---|---|
| penicillin V, 42 atoms, 15 cycles | 1e-11 | 0.877 → 0.546 | 1.000 | 1.000 → 0.797 |
| | 1e-13 | 0.963 → 0.694 | 1.000 | 1.000 → 0.909 |
| taxol, 113 atoms, 17 cycles | 1e-11 | 0.797 → 0.379 | 1.000 | 1.000 → 0.725 |
| | 1e-13 | 0.927 → 0.541 | 1.000 | 1.000 → 0.877 |
| PfPMT, 284 atoms, 50 cycles unconverged | 1e-11 | 0.755 → 0.57–0.68 | 1.000 | 1.000 → 0.94–1.00 |

(The 42- and 113-atom rows were taken on a login node; the probe times
nothing.)  The bounds: `q_ij = sqrt(max (ij|ij))` as `Int3c2eOpt.build` keeps
it, and `q_P = sqrt(max (P|P))` for each sorted auxiliary shell in closed form
— `electrolites/_j3cinc.py:self_coulomb`, which reproduces
`SortedMole.intor('int2c2e')`'s diagonal to the last digit and replaces a
one-shell libcint call that costs 0.2 s on a loaded login node.

## 2. The implementation and what it costs

`fastdfj3c._get_j_incremental`, for a single symmetric density: the previous
call's sorted-basis density, its fitted density before the metric, its
coefficients after it and its sorted-basis Coulomb matrix are kept on the `DF`
object; the density contraction is run on the difference over the pairs the
bound keeps and accumulated, the metric is solved on the whole accumulated
density (a few milliseconds, and it couples everything), and the
auxiliary-vector contraction is run on the coefficients' difference over the
pairs *its* bound keeps.  Screened pair lists are *gathered*
(`_j3ck.Plan.subset`), so W19's kernels run on them unchanged and
GPU4PySCF's run on the groups past W19's ceiling.  A full build every eight
(`FASTDFJ3C_INC_REBUILD`) and whenever the difference is more than a tenth of
the density (`FASTDFJ3C_INC_MAXFRAC`) — which is what makes a response
density handed to the same object after an SCF correct: it is built whole.

`benchmarks/df_j3c_bench.py --scf --scf-variants w9,fast,inc`:

| DF-PBE | W19 | W19 + screen | | cycles | ΔE | pairs kept (density / aux) |
|---|---|---|---|---|---|---|
| penicillin V | 1.894 s | 2.001 s | 0.947x | 15 / 15 | 6.5e-10 | 0.705 / 0.916 |
| taxol | 5.644 | 5.889 | 0.958x | 17 / 18 | 6.9e-9 | 0.541 / 0.870 |
| Cu macrocycle | 5.436 | 6.092 | 0.892x | 23 / 26 | 8.4e-10 | 0.637 / 0.885 |
| penicillin V, patched baseline | 1.332 | 1.372 | 0.971x | 15 / 15 | 6.5e-10 | 0.705 / 0.916 |
| PfPMT, 50 cycles, unconverged | 55.974 | 56.020 | 0.999x | 50 / 50 | 6.3e-10 | 0.588 / 0.978 |

At 284 atoms the density oscillates (this SCF does not converge), so 38 of the
51 builds exceed the one-tenth guard and are built whole anyway.

`tests/test_dfj3c.py` keeps it honest while it is off: three SCFs with and
without it to 1e-9 Eh and within one cycle, and the response-density guard to
round-off.

## 3. Tools

```bash
python benchmarks/df_j3c_screen_probe.py --xyz benchmarks/molecules/taxol.xyz \
    --basis def2-svp --xc pbe [--taus 1e-9,1e-11,1e-13]
python benchmarks/df_j3c_bench.py --cases taxol --variants w9 --reps 1 \
    --scf --scf-variants w9,fast,inc
```

Switches: `FASTDFJ3C_INCREMENTAL=1`, `FASTDFJ3C_INC_TOL`,
`FASTDFJ3C_INC_REBUILD`, `FASTDFJ3C_INC_MAXFRAC`.
