# A density screen inside the three-centre kernels — built, bounded, and worth nothing measurable

`docs/DF_WORK_LIST.md` **W26**, found by W20.  W20's screen of a Coulomb build
from the density difference is host-side: a shell pair is kept or dropped
for *every* auxiliary shell at once, so its density-side bound carries
`max_P q_P` and its auxiliary-side bound `sum_P q_P |dc_P|` over the whole
auxiliary set.  The auxiliary-vector kernel kept 87–92 % of its work.  The
entry proposed testing inside W19's kernels, per block of pairs against a
batch of auxiliary shells, so that both sides get the tight factor and no
list has to be rebuilt.  It named its falsification: the extra SCF cycles a
screened Fock matrix costs outweigh whatever a bound tight enough to matter
can drop.  It also said to measure the cycle count at 1e-13 first.

That is what was done.  The screen now lives inside the kernels, it is
correct, and it is bounded.  At every threshold tested it is better than
W20's screen: no extra cycles, and a loss turned into a wash.  The item
still fails, and not on the ground the entry named.  At 1e-13 the cycle
count does not move on any system.  What fails is the drop itself: at the
threshold that keeps the cycle count, the bound skips **6–17 %** of the
difference builds' kernel time.  Those kernels are a few per cent of a
pure-functional SCF, so the saving is **0.1–1.0 %** of a run, which is
below the run-to-run noise.  Incremental builds stay **off by default**.
`FASTDFJ3C_INCREMENTAL=1` now uses this screen, with W20's list kept as
`FASTDFJ3C_INC_SCREEN=list`.

---

## 0. Summary

*(A100-SXM4-80GB of a whole Perlmutter GPU node, GPU4PySCF 1.8.1, DF-PBE,
def2-SVP and its jkfit set unless another basis is named.)*

* **The falsification check, first** (W20's code, `FASTDFJ3C_INC_TOL=1e-13`):
  penicillin V 15/15 cycles, taxol 17/17, Cu macrocycle 23/23.  W20's 1e-11
  cost one extra cycle on taxol and three on the Cu complex; 1e-13 costs none.
  So the cycles are not what kills the idea at 1e-13.
* **What the bound keeps at 1e-13** (`benchmarks/df_j3c_blockscreen_probe.py`,
  measured kernel time over every difference of a converging SCF; the
  unscreened row is the same kernels with no test):

  | | differences | unscreened | τ = 1e-13 | 1e-12 | 1e-11 |
  |---|---|---|---|---|---|
  | penicillin V, 42 atoms | 15 | 50.0 ms | 0.931 | 0.892 | 0.847 |
  | penicillin V, def2-TZVPP | 17 | 211.4 | 0.919 | | |
  | Cu macrocycle, 60, quartet | 23 | 194.5 | 0.939 | 0.903 | 0.864 |
  | taxol, 113 | 17 | 445.0 | 0.827 | 0.759 | 0.687 |
  | PfPMT, 284, 50 cycles unconverged | 50 | 6320.3 | 0.885 | 0.812 | |

  On the last cycle at 113 atoms, where the difference is smallest, the
  screen keeps 62 % (1e-13) to 46 % (1e-11) of the kernels' time.  On the first
  few cycles it keeps essentially all of them.
* **Whole SCFs**, `inck` (this screen, 1e-13) against `fast` (W19 alone),
  each process two repetitions with the second quoted, up to four processes per
  case at 1e-13 over two jobs and one at 1e-12:

  | DF-PBE | inck / fast | cycles | ΔE |
  |---|---|---|---|
  | penicillin V | 1.019x, 1.053x, 1.016x, 1.019x (1e-12: 1.004x) | 15 / 15 | 1e-11 |
  | taxol | 1.045x, 1.044x, 0.967x, 1.003x (1e-12: 1.014x) | 17 / 17 | 9e-11–1.3e-10 |
  | Cu macrocycle | 0.964x, 0.909x, 1.011x, 0.998x (1e-12: 1.010x) | 23 / 23 | 5e-12–1e-11 |
  | penicillin V, def2-TZVPP | 1.096x, 1.003x | 17 / 17 | 6e-12–1.3e-11 |
  | PfPMT, 50 cycles | 1.000x | 50 / 50 | 7e-11 |

  The spread, 0.91x–1.10x, is the noise of a 2–6 s run with the node's other three cards busy.
  What the kernel table predicts is 0.1 % at 42 atoms, 1.0 % at 113, 0.15 %
  on the Cu complex, 0.4 % in def2-TZVPP and 0.35 % at 284 atoms.  W20's
  list, run in the same processes, read 0.973x, 0.991x and 0.901x, with its
  extra cycles.
* **The shipped build is unchanged.**  A first version passed the screen
  to the shipped kernels as a run-time flag, and cost every build 0.8–1.0 %
  (taxol 29.08 → 29.34 ms, PfPMT 139.4 → 140.8 ms, on the same card and
  both repetitions).  The screen is now a separate module that is compiled
  only when a build from a difference asks for it.  The unscreened
  generator's output is byte-identical to the previous commit's, and it
  times identically (§4).
* **Why so little.**  The bound, not the granularity, is what limits this
  screen (§3).  Even on the last cycle the largest `|dD|` element is ~4e-6.
  So a skipped unit needs `q_ij q_P` below ~1e-8 at 1e-13, and only the tail
  of the stored pairs, the ones just above the build's own 1e-12 pair
  cutoff, is that small.  The ceiling is lower still: a screen that dropped
  *everything* in the difference builds would save 13 × 24.8 ms = 0.32 s
  of a 5.6 s run at 113 atoms, **6 %**.

---

## 1. What was built

**The test** (`electrolites/codegen/gen_j3c.py`, `scr=True`):

* *Density kernel.*  A block is `[p0, p1)` pairs of one class against
  `kb` (1–4) auxiliary shells.  The block takes `qb = max_batch q_P`, and a
  warp skips an iteration of its pair loop when none of its 32 pairs has
  `q_ij max|dD_ij| >= τ / qb`.  The loop is rewritten to be warp-uniform, so
  that `__any_sync` sees every lane.
* *Auxiliary-vector kernel.*  A block is 128 pairs against up to 16
  auxiliary shells of one class, with each thread owning one pair.  A warp
  leaves when none of its pairs has `q_ij · Σ_batch q_P max|dc_P| >= τ`.
  Otherwise it skips each shell for which none has `q_ij q_P max|dc_P| >= τ`.
  The probe priced this per-shell test against the per-batch sum the entry
  proposed (column `aux1` against `aux`).  At 113 atoms the per-batch sum
  kept 67–100 % and the per-shell test 52–97 % at 1e-13, so the per-shell
  test ships.  Under the screen, a lane past the group's end stays in the
  loop so that the vote is well defined; it computes with the last pair
  and writes nothing.

**The inputs** (`electrolites/_j3cinc.py`): `pair_weights(ddm)` gives
`q_ij max|dD|` over each stored pair's shell block, and `aux_weights(dc)`
gives `q_P max|dc|` over each auxiliary shell.  Each is one launch.  There
is no gather, no block table and no host synchronisation.  Those were W20's
~10 ms per cycle of host work at 42 atoms.

**The driver** (`fastdfj3c._get_j_incremental`, `INC_SCREEN='kernel'`) is
W20's.  It keeps the previous sorted-basis density, the fitted density
before and after the metric, and the Coulomb matrix.  It builds everything
whole every eight calls and whenever the difference exceeds a tenth of the
density.  Only the two contractions change.  The shell-pair groups past
W19's ceiling are still GPU4PySCF's kernels, built whole from the
difference.

**The threshold is not W20's number.**  A pair's Coulomb element is a sum
of one term per auxiliary shell, and each skipped (warp, shell) can
contribute up to τ.  So the same τ admits more error here than W20's
one-decision-per-pair test.  At 1e-11 an ethanol DF-PBE SCF at
`conv_tol=1e-11` stopped converging (50 cycles against 11), and at 1e-12 it
took 15.  The kernel mode therefore defaults to **1e-13**.  At that value
every case in this document keeps its cycle count, with energies within
1.3e-10 Eh.

---

## 2. The block granularity, and the ordering it needs

`df_j3c_blockscreen_probe.py` prices each bound on a model of the kernel's
work.  A (warp, batch) is weighted by pairs × shells × (`nf3(L)` + `nf3(lij)`
× `nk`).  The probe reports W20's per-pair screen on the same model next to
it:

| taxol, last cycle | W20 dm / aux | warp dm / aux (batch sum) / aux (per shell) |
|---|---|---|
| τ = 1e-11 | 0.379 / 0.725 | 0.406 / 0.491 / 0.339 |
| τ = 1e-13 | 0.541 / 0.876 | 0.572 / 0.668 / 0.522 |

On the density side, a warp of 32 pairs keeps slightly *more* than W20's
per-pair test, and the per-batch `q_P` does not quite make up for it.  On
the auxiliary side the tight factor does what the entry said it would: 0.52
against 0.88 at 1e-13.

**A spatial ordering would be worse, not better.**  GPU4PySCF sorts each
angular-momentum group by Schwarz factor, descending, so 32 consecutive
pairs are anywhere in the molecule.  That looked like a defect for a
per-warp `max|dD|`, and the probe also prices the same bounds with each
group re-sorted along a Morton curve through the product centres.  At every
threshold on penicillin V, every warp is then kept (0.98–1.00 on both
kernels, against 0.48–0.98 in GPU4PySCF's order).  The bound is dominated
by `q_ij`, not by `|dD_ij|`, and the Schwarz-sorted order already puts the
small-`q` pairs together at each group's tail.  A Morton order mixes them
back in with the large ones.

---

## 3. Why the bound cannot be made to matter

`max|dD|` in the sorted, decontracted basis on taxol runs from 2 on the
first cycles to **4.1e-6 on the last** (`max|dc|` 1.4 → 4.4e-6).  A skipped
density-side unit needs `q_ij |dD_ij| q_P < τ`, with `q_P` up to 17, and
the per-element `|dD|` is not much smaller than its maximum.  So at 1e-13
the skipped units are the pairs with `q_ij ≲ 1e-9`, only slightly above
the build's own cut (`Int3c2eOpt.build(cutoff=1e-12)` stores a pair only
if its Schwarz product can reach 1e-12).  The auxiliary side is the same
with the roles swapped.

Four things follow:

* No regrouping of the same Schwarz bound changes this.  The per-pair test
  (W20), the per-warp one and the per-shell one all drop the same tail.
* Distance does not rescue it.  The three-centre Coulomb integral decays
  only as `1/R` between a pair and an auxiliary charge, and the fitted
  coefficients' difference "is small everywhere and local nowhere" (W20).
* A looser τ drops more, and costs cycles.  At 1e-11 the kernel time is
  0.69 of unscreened at 113 atoms, but the Fock matrix is then wrong by
  enough to move a DIIS trajectory.  The entry's own falsification applies
  there.
* The ceiling is small in any case.  At 113 atoms the kernels of the
  difference builds are 6 % of the SCF.  At 284 atoms the SCF oscillates, so
  38 of 51 builds exceed the one-tenth guard and are built whole anyway.

---

## 4. The shipped kernels, unchanged

`df_j3c_bench.py --variants fast --reps 20` (Cu and PfPMT `--reps 10`),
out of the previous commit's tree and out of this one, alternated on one
card, twice each:

| | previous commit | run-time flag (first version) | separate module (shipped) |
|---|---|---|---|
| penicillin V | 4.263–4.287 ms | 4.295–4.306 | 4.266–4.280 |
| taxol | 29.075–29.167 | 29.306–29.378 | 29.066–29.075 |
| Cu macrocycle | 19.307–19.394 | 19.464–19.519 | 19.296–19.365 |
| PfPMT | 139.395–139.403 | 140.728–140.852 | — |

`python -m electrolites.codegen.gen_j3c` writes the same bytes as before.
The screened twin is a second module, `_j3ck._module(..., scr=True)`, with
its own tier kernels (`j3c_{dm,aux}_tier<t>_scr`).  It is compiled on the
first build from a difference and never otherwise.

## 5. Correctness

`tests/test_dfj3c.py` adds `screened()`.  At τ = 0 the screened launches
equal the unscreened ones to 8e-17–4.7e-16, relative, on ethanol and on the
Ni complex, with and without GPU4PySCF's kernels in the same output.  At
τ = 1e-10 the error is nonzero (1.3e-10–7.5e-10) and under the sum of the
Schwarz bounds of what was skipped.  `incremental()` now runs its three
SCFs in both modes, each at its own default threshold, to 1e-9 Eh and
within one cycle.  The guard that builds a response density whole is
unchanged.

## 6. Tools

```bash
python benchmarks/df_j3c_blockscreen_probe.py --xyz benchmarks/molecules/taxol.xyz \
    --basis def2-svp --xc pbe [--taus 1e-11,1e-12,1e-13] [--orders stock,morton] \
    [--time 3]                   # timing: whole node only
python benchmarks/df_j3c_bench.py --cases taxol --variants fast --reps 1 \
    --scf --scf-variants fast,inck,inc [--inc-tol 1e-12]
python tests/test_dfj3c.py
```

Switches: `FASTDFJ3C_INCREMENTAL=1` (off by default),
`FASTDFJ3C_INC_SCREEN=kernel|list` (`kernel` by default),
`FASTDFJ3C_INC_TOL` (1e-13 for `kernel`, 1e-11 for `list`),
`FASTDFJ3C_INC_REBUILD`, `FASTDFJ3C_INC_MAXFRAC`.
