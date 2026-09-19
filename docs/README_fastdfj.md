# fastdfj — RIJONX, and the 86 GB that does not have to exist

`docs/DF_WORK_LIST.md` **W1**, the first item on that list and the cheapest on
it by a wide margin.  It closes.

The work list ranked W1 first on the strength of one observation: a
density-fitted **Coulomb-only** build never constructs a cderi.  `df/df.py:111`
returns before the tensor is touched and sends the call to
`df/j_engine_3c2e.py`, which contracts the three-centre integrals with the
density directly.  So the configuration ORCA calls **RIJONX** — fit J, take K
from the four-centre integrals — has none of the storage that makes a large DF
run expensive, and the exchange it needs instead is `fastk`, which this
repository has run at 284 and 545 atoms on the direct path.

It could not be tried, because it does not run.  `only_dfj=True` is **broken
in GPU4PySCF 1.8.1**: `_DFHF.get_jk` calls

```python
vj = self.with_df.get_jk(mol, dm, hermi, with_k=False)[0]   # df_jk.py:213
```

and `DF.get_jk`'s signature is `(dm, hermi, with_j, with_k, ...)` — there is no
`mol` argument.  The molecule arrives as the density, `hermi` as `with_j`, and
the run dies in `df_jk.get_j` with `'Mole' object has no attribute 'shape'`
before it has computed anything.  The GHF spin-free branch twenty lines above
(`df_jk.py:170`) makes the same call correctly, which is what the repaired line
is copied from.

---

## 0. Summary

* **The one-line fix works, and what it unlocks is worth having.**  At 284
  atoms (PfPMT, def2-SVP, nao 2697), one hybrid Fock build:

  | | full DF | **RIJONX** | direct |
  |---|---|---|---|
  | J + K, per call | 10.64 s | **4.01 s** | 4.51 s |
  | ↳ J | 3.19 s | **0.74 s** | 1.23 s |
  | ↳ K | 10.51 s | 3.28 s | 3.28 s |
  | one-time, before the first Fock matrix | **88.7 s** | **1.08 s** | — |
  | stored | **86.4 GB**, host | **none** | none |
  | **whole SCF**, 18 cycles | **300.5 s** | **64.1 s** | 65.4 s |

  W1's acceptance criterion was a two-electron half that beats full DF's
  **12.5 s** per call.  It is **4.01 s**, i.e. **2.65x** against the same
  measurement taken in the same process (10.64 s — `patch_all()` includes
  `fastdf`, which is worth 1.2x on that row).  End to end the margin is
  larger, **4.69x**, because full DF also pays 88.7 s for a tensor RIJONX
  never builds.
* **W1's falsification criterion is cleared by a factor of four.**  It named
  the three-centre J engine as the thing that might not pay, with a bar of
  **3.28 s** — full DF's J-only call, which is all PCIe.  The J engine does it
  in **0.73 s**: **4.34x** against the cderi route, and **1.67x** against the
  *direct* four-centre Coulomb build as well.
* **The exchange is the direct path's own build**, as it must be — the same
  `scf/jk.py` call, timed at 3.277 s against 3.278 s and checked element-wise
  against a direct SCF object's `get_k` at **2.3e-16** relative (§4).  That is
  the check that the repaired branch is doing what RIJONX means, not a
  speedup.
* **The larger half of the module is a driver change, not a kernel one.**
  `_DFHF.get_veff` rebuilds from the *whole* density every cycle — correct for
  a cderi contraction, which does not screen, and badly wrong for a four-centre
  exchange build, which does.  An `only_dfj` object is handed to its own
  class's `get_veff` instead, so the exchange is built from `D - D_last`.  §3.3
  measures what that is worth; without it RIJONX is *cheaper per call* than the
  direct path and *slower over a whole SCF*, which is how the change was found.
* **It changes the answer**, and not in the direction a reader might guess.
  RIJONX has a *larger* fitting error than full DF, not a smaller one: the
  fitted J and the fitted K of a full-DF hybrid cancel in part, and RIJONX,
  having no fitted K, cannot do that.  §5 has both columns on seven systems.
* **The gradient does not follow.**  Nothing in `df/grad/` reads `only_dfj`,
  so the gradient GPU4PySCF returns for such an object is the derivative of
  the full-DF energy.  The module warns once rather than pretending otherwise
  (§6).

---

## 1. What the module changes

Three entry points on `df/df_jk.py:_DFHF`, all of them reached only when
`only_dfj=True`.  Everything else — every `only_dfj=False` call, which is the
whole of `fastdf`'s territory, and every GHF/DHF object, whose branch is
already correct — is handed back to GPU4PySCF unchanged, and `stats()` counts
which.

### 1.1 `get_jk` — the argument order

The repaired branch is the definition of RIJONX and nothing more:

```python
vj = self.with_df.get_jk(dm, hermi, with_j=True, with_k=False, omega=omega)[0]
vk = hf.SCF.get_k(self, mol, dm, hermi, omega=omega)
```

`tests/test_dfj.py` checks both halves against independently assembled
references — a plain density-fitting object's J-only call, and a *direct* SCF
object's `get_k` — element-wise at fixed density, at `hermi` 0 and 1, one and
two densities, restricted and unrestricted, def2-SVP and def2-TZVPP.  Worst
over all of it: **2.2e-14 (J) / 3.4e-16 (K)** relative.

One thing was added rather than copied.  The branch reaches `df_jk.get_j` for
*every* Coulomb matrix it will ever build — there is no cderi to fall back on
— and that function asserts `1 < ndim <= 3` (`gto/mole.py:907`).  An
unrestricted response solve asks for `(2, nset, nao, nao)`, which a full-DF run
survives only because its SCF has already built a cderi and taken the other
branch.  The leading axes are flattened for the call and restored after it.

### 1.2 `get_veff` — the exchange built from the difference

This is the half that decides whether RIJONX is worth using.

`_DFHF.get_veff` ignores `dm_last` and `vhf_last` and rebuilds from the whole
density every cycle, and `_DFHF.__init__` sets `direct_scf = False` to say so.
That is right for a cderi contraction: it is dense, it does not screen on the
density, and building it from `D - D_last` costs exactly what building it from
`D` costs while accumulating round-off for nothing.

Under `only_dfj` the exchange is the four-centre build, whose cost *is* set by
how much of the shell-pair space survives screening — and a converged SCF's
`D - D_last` screens away most of it.  So an `only_dfj` object is handed to
the `get_veff` of the class it was built from (`dft/rks.py`, `dft/uks.py`,
`scf/hf.py` or `scf/uhf.py`), which is the incremental one.  Its Coulomb half
still comes
from `_DFHF.get_j`, i.e. the three-centre fit — nothing about the
*approximation* changes, only how many times the exchange is built from
scratch.

It needed one line elsewhere.  `fastxcnlc`'s restricted wrapper declines any
object carrying a `with_df`, because on a full-DF run `_DFHF.get_veff` is the
real entry point and fusing the XC and VV10 grid passes twice would be wrong.
Routing an `only_dfj` object to `RKS.get_veff` walked straight into that
guard, and a RIJONX wB97M-V run lost the fused sweep — the same silent
inertness `docs/README_fastdf.md` §5 found the other way round.  `only_dfj` is
now named as the exception, `tests/test_dfj.py` asserts `fused > 0` on such a
run rather than trusting it, and the fused and unfused energies agree to
**1.7e-13 Eh**.

ROHF and GHF keep `_DFHF.get_veff`: the first rewrites the density into a
two-row array before it calls `nr_uks` and the second is complex, and neither
has been measured here.

`FASTDFJ_NO_INCREMENTAL=1` restores upstream's, which is the ablation column of
§3.3.  The two must converge to the same energy in the same number of cycles,
and `tests/test_dfj.py` checks exactly that on five cases — RKS/UKS, hybrid and
range-separated, RHF and UHF — at **1e-12 Eh** and identical cycle counts.

### 1.3 `get_k` — the range-separated request kept in one call

`_DFHF.get_k` serves `lr·erf(ωr)/r + sr·erfc(ωr)/r` as *two* `get_jk` calls
that it then scales and adds.  That is the right shape when the exchange comes
out of a cderi, where the two operators need two different tensors.  Under
`only_dfj` the exchange is the direct path's, and `scf/jk.py` has served the
fused request in a single pass over the Rys integrals since 1.8.0 — `fastk`
does too.  Handing the factors straight through is the whole of this function,
and it halves the exchange work of a range-separated RIJONX run.

---

## 2. Where the crossover is

A density-fitted exchange build is a **dense** contraction of a tensor whose
size grows as about `N^2.5` (`docs/DF_WORK_LIST.md` §3.2).  A four-centre
exchange build is a **screened** sum over shell-pair quartets, and its cost
grows far more slowly.  So the two curves cross, and where they cross is the
whole question.

Two things push the crossing down, and at 284 atoms they arrive together:

1. **When the cderi stops fitting on the device.**  At 113 atoms the tensor is
   14.3 GB and lives on the A100; every block is already there when the
   contraction asks for it.  At 284 atoms it is 86.4 GB, lives in host DRAM,
   and every cycle streams all of it across PCIe.  `docs/DF_WORK_LIST.md` §3.3
   measured that stream at 26.4 GB/s — a saturated gen4 x16 link, i.e. nothing
   left to win — and found it *fully hidden* behind the exchange contractions.
   Hidden is not free: it is hidden because the contractions take longer than
   the transfer.
2. **When the one-time build stops being a rounding error.**  `_cholesky_eri`
   is 0.36 s at 42 atoms, 3.3 s at 113 and **88.7 s** at 284 — which is 1.4x
   the *entire* RIJONX SCF of the same molecule.

RIJONX pays neither, because a Coulomb-only DF build never constructs the
tensor at all.  What it pays instead is an exact exchange build every cycle,
and this repository has spent four rounds making that cheap.

The numbers below put the crossing, end to end, between **42 and 113 atoms**
in def2-SVP on a 40 GB A100: full DF wins the 42-atom run by 2.07x and loses
the 113-atom one by 1.52x.  Neither endpoint was bisected, and nothing here
measures where it sits in a larger basis — but both of the two reasons above
get worse with `naux`, which is linear in the basis, so it should move down.
That is a prediction, not a measurement.

---

## 3. The measurement

`benchmarks/dfj_probe.py`.  One A100-SXM4-40GB of a whole Perlmutter GPU node,
251 GB of host DRAM, GPU4PySCF 1.8.1, `patch_all()`, B3LYP/def2-SVP, level-3
grids, `conv_tol = 1e-9`.  Variants are timed **round robin** so that a
competing job's drift lands on every row equally rather than on whichever one
ran while it was there.

### 3.1 One Fock build, three ways, at fixed density

Milliseconds, minimum of two (one at 284 atoms):

| | | full DF | **RIJONX** | direct |
|---|---|---|---|---|
| **penicillin V** 42 atoms, nao 430 | J+K | **17.3** | 172.9 | 211.8 |
| cderi 1.07 GB, on the device | J | **2.8** | 26.3 | 63.8 |
| | K | **17.9** | 146.7 | 146.7 |
| **taxol** 113 atoms, nao 1123 | J+K | **403.6** | 1064.9 | 1205.9 |
| cderi 14.31 GB, on the device | J | **28.7** | 157.6 | 296.3 |
| | K | **381.6** | 907.5 | 907.9 |
| **PfPMT** 284 atoms, nao 2697 | J+K | 10636 | **4012** | 4510 |
| cderi 86.41 GB, **on the host** | J | 3193 | **736** | 1228 |
| | K | 10508 | **3277** | 3278 |

Four things, two of which the smaller systems would have got backwards.

**Full DF wins the fixed-density row until the tensor leaves the device, and
then loses by 2.65x.**  At 42 atoms it is *ten times* faster than either
four-centre build; at 113 it is 2.7x; at 284 it is 2.65x **slower** than
RIJONX.  That is the first crossing of §2 and it is sharp, because the cost of
a host-resident cderi is not a gentle penalty — it is a different regime.

**The three-centre Coulomb engine beats the four-centre one at every size.**
W1 named it as the thing most likely to falsify the item, with a bar of
3.28 s — full DF's J-only call at 284 atoms, which `docs/DF_WORK_LIST.md` §3.3
measured as pure PCIe.  It does it in **0.736 s**, **4.34x**; and it beats the
*direct* Coulomb build (which is `fastj`) by 2.43x at 42 atoms, 1.88x at 113
and 1.67x at 284.  `df/j_engine_3c2e.py` is a McMurchie–Davidson kernel nobody
in this repository had profiled; it is very good.

Read the `J` column the other way and it is a caution, not a saving.  Full
DF's J-only row is fast at 42 and 113 atoms (2.8 and 28.7 ms) because the
whole tensor is already on the device, and catastrophic at 284 (3193 ms)
because the call streams 86 GB to build one matrix.  It is **not** what a
full-DF hybrid pays for J per cycle: fused with K the Coulomb half costs
`10636 − 10508 = 128 ms`, because both come out of one pass over the blocks.
The 3.19 s figure is W1's bar and a J-*only* call on an object that already
has a cderi, which is a narrow case.

**The exchange is the direct path's, to the third digit.**  3.277 s against
3.278 s at 284 atoms, 907.5 against 907.9 ms at 113, 146.7 against 146.7 at
42.  It is the same `scf/jk.py` call — this row is the check that the repaired
branch means what RIJONX means, not a speedup.

**And a *pure* functional is already on the fast column.**  It has no
exchange, so `df/df.py:111` returns before the tensor is touched and the whole
run goes to `j_engine_3c2e` — verified rather than assumed: a DF-PBE SCF
leaves `with_df._cderi` at `None` and `fastdf.stats()` at **zero calls**
(`docs/README_fastdf.md` §5 found the same).  So the RIJONX `J` column *is*
the pure-functional path, and these rows say it is in good shape; there is no
`_cholesky_eri` to avoid there, and nothing to take.

### 3.2 What each one costs before the first Fock matrix

| | full DF (`_cholesky_eri`) | RIJONX (option objects + metric) |
|---|---|---|
| penicillin V, 42 atoms | 0.358 s, 1.07 GB | 0.176 s, no tensor |
| taxol, 113 atoms | 3.33 s, 14.31 GB | 0.330 s, no tensor |
| PfPMT, 284 atoms | **88.7 s, 86.41 GB** | **1.08 s, no tensor** |

The 284-atom row is the one to read twice.  88.7 s is 1.4x the whole RIJONX
SCF of that molecule (§3.3), and 86.4 GB is a third of the node's DRAM — a
limit `_cholesky_eri` never checks, because its host branch calls
`empty_mapped` without consulting `max_memory` (`docs/DF_WORK_LIST.md` §3.3).

### 3.3 Whole SCF runs, and the driver

Seconds, the second of two runs of each in one process, 18 cycles at 284 atoms
and 14–15 at the other two — **every variant converged in the same number of
cycles as every other**, so these are per-cycle costs and not convergence
luck.

| | full DF | **RIJONX** | RIJONX, upstream driver | direct |
|---|---|---|---|---|
| penicillin V, 42 atoms | **1.67** | 3.46 | 3.69 | 3.79 |
| taxol, 113 atoms | 22.73 | **14.98** | 19.65 | 15.48 |
| PfPMT, 284 atoms | 300.54 | **64.12** | 85.54 | 65.38 |

| | RIJONX against full DF | the driver (§1.2) |
|---|---|---|
| penicillin V | 0.48x | 1.07x |
| taxol | **1.52x** | **1.31x** |
| PfPMT | **4.69x** | **1.33x** |

**The end-to-end crossing is earlier than the per-call one.**  At 113 atoms
full DF still wins the fixed-density row by 2.7x and loses the whole run by
1.52x — because `_cholesky_eri` is 3.3 s of a 22.7 s run and because the
fixed-density row is not the build an SCF does: the direct and RIJONX columns
build from `D - D_last` and screen accordingly, and the DF column cannot.

**The driver column is how that change was found.**  Before it, RIJONX was
*cheaper per call* than the direct path at 113 atoms (1.065 s against 1.206 s)
and *slower over the whole run* (19.65 s against 15.48 s).  The only thing
that can produce that combination is a Fock matrix rebuilt from scratch, and
`_DFHF.get_veff` was doing exactly that.  With `D - D_last` restored, RIJONX
is level with the direct path at both sizes — 14.98 against 15.48 s, and 64.12
against 65.38 s — which is what it should be: the same exchange build, and a
Coulomb build that is 1.67x cheaper but a small share of the total.

The two agree to **8.7e-11 Eh** at 284 atoms and **7.3e-12** at 113, in the
same number of cycles.  At def2-TZVPP the cycle count can differ by one
(§4) — an incremental build screens on the difference, so the Fock matrix
differs by of order `direct_scf_tol` and a DIIS trajectory can cross the
threshold one iteration early or late.

**What this does not say.**  Full DF is still the right answer below about a
hundred atoms in this basis, by a factor of two — every row above is one
basis on one card, and the crossing moves with both.  `--scf-ablate` reruns
the table.

### 3.4 Which code paths a RIJONX run actually reaches

`benchmarks/coverage_probe.py --only-dfj`, penicillin V, 14 cycles.  The rule
this repository keeps relearning is that *a patch target is not a code path*,
so the claim "the exchange left the density-fitting path" is a counter, not a
sentence:

| entry point | `--only-dfj` | `--density-fit` |
|---|---|---|
| `fastdfj` `_DFHF.get_jk` (repaired branch) | **15** | 0 of 15 calls |
| `fastdf` `df_jk.get_jk` (the cderi build) | **(none)** | 15 |
| `fastk` `_VHFOpt.get_k` (exact exchange) | **15** | (none) |
| ↳ its class blocks | **6930** | (none) |
| `fastxcnlc` `RKS.get_veff` | **15** | (none) |
| `fastj` `_VHFOpt.get_j` | (none) | (none) |

`fastdf` reads **zero** on a RIJONX run: no cderi is ever built, so there is
nothing for it to contract.  `fastk` carries the run instead — 6930 class
blocks of it.  `RKS.get_veff` is reached only because of §1.2; with upstream's
driver a RIJONX run stops at `_DFHF.get_veff`.  And `fastj` reads zero in both
columns, because the Coulomb build is the *three-centre* engine either way.

B3LYP has no non-local correlation, so `fastxcnlc` passes all 15 of those
calls through (`stats()` reads `fused: 0, declined: 15`).  The same run on
**wB97M-V** is where the §1.2 guard matters: `fused: 14 of 14` with the
exception in place, `0 of 14` without it.

---

## 4. Correctness

There is no upstream answer to compare against — the branch does not run on
stock 1.8.1 — so "it no longer crashes" is not a bar.  What is checked instead
is that the branch computes **its definition**, from references assembled by
objects that never saw `only_dfj`:

| | reference | worst, def2-SVP | worst, def2-TZVPP |
|---|---|---|---|
| J | a plain DF object's J-only call (`j_engine_3c2e`, no cderi) | 3.8e-15 | 1.9e-14 |
| K | a *direct* SCF object's `get_k` | 2.3e-16 | 6.7e-16 |

relative to the largest element of the reference matrix, over ethanol and CH3,
`hermi` 0 and 1, one and two densities, restricted and unrestricted.  Then:

* the four `(with_j, with_k)` combinations return what was asked and nothing
  else;
* a **GHF** object under `only_dfj` is handed to GPU4PySCF, counted by
  `stats()['passthrough']` rather than asserted in prose;
* a **pure** functional's `only_dfj` energy equals its full-DF energy to
  **1.1e-13 Eh** (def2-SVP) and **5.7e-14** (def2-TZVPP) — there is no
  exchange, so the flag can change nothing, and this is the cheapest test that
  the Coulomb half was not disturbed;
* the **incremental** driver of §1.2 converges to the same energy as the
  whole-density one on RKS, UKS, RHF, UHF, a hybrid, a range-separated hybrid
  and wB97M-V: **1e-12 Eh** or better at def2-SVP, all in the same number of
  cycles.

The one thing that is *not* identical is the cycle count at def2-TZVPP, and it
is worth stating rather than tuning away.  wB97X converged in 10 cycles
incrementally and 11 from the whole density, wB97M-V in 12 against 11, with
the two energies **1.7e-11** and **3.1e-11 Eh** apart on a `conv_tol` of
1e-11.  An incremental build screens on `D - D_last`, so its Fock matrix
differs from the whole-density one by of order `direct_scf_tol`, and near
convergence that is enough to move a DIIS trajectory across the threshold one
iteration either way.  The test allows one cycle of difference and holds the
energy bar at 1e-9 Eh; a larger drift, or a count that moves by more than one,
is not this effect.  None of the three sizes in §3.3 showed it — all four
variants took the same number of cycles there.

```bash
python tests/test_dfj.py                      # ethanol + CH3, def2-SVP
python tests/test_dfj.py --basis def2-tzvpp   # up to f functions
```

`tests/test_df.py` and `tests/test_full.py` are unchanged by this module and
were re-run to confirm it: worst relative difference on the density-fitted J
and K build **3.4e-15**, i.e. `fastdf` is where it was.

---

## 5. It is a different approximation, not a different route

RIJONX fits J and leaves K exact.  Full DF fits both.  Neither is the exact
answer, and the one with *fewer* approximations is the one with the **larger**
error:

| | E(RIJONX) − E(direct) | E(full DF) − E(direct) |
|---|---|---|
| ethanol / PBE (def2-SVP) | −7.85e-05 | −7.85e-05 |
| ethanol / B3LYP | −8.03e-05 | −4.71e-05 |
| ethanol / wB97X | −8.14e-05 | −5.51e-05 |
| CH3 / B3LYP (UKS) | −2.49e-05 | −1.90e-05 |
| CH3 / wB97X (UKS) | −2.57e-05 | −2.09e-05 |
| penicillin V / B3LYP (42 atoms) | −4.19e-04 | −1.65e-04 |
| taxol / B3LYP (113 atoms) | −1.03e-03 | −4.24e-04 |
| PfPMT / B3LYP (284 atoms, −1) | −2.43e-03 | −9.59e-04 |

Two things to read off it.  The **pure** functional row is the control: with no
exchange the two columns are identical, to every digit.  And on every hybrid
RIJONX is worse by roughly a factor of two — the fitted J and the fitted K of a
full-DF hybrid carry errors of opposite sign that partly cancel, and RIJONX has
only the first of them.  Both errors grow with system size at about the same
rate — the ratio is 2.5 at 42 atoms, 2.4 at 113 and 2.5 at 284 — and neither
is a *relative* energy; nothing here measures what either does to one.  At
def2-TZVPP the gap narrows (ethanol −1.06e-04 against −7.85e-05) because the
larger orbital basis is better spanned by its own fitting set.

So this is a choice a user makes, not an optimisation applied behind their
back.  `only_dfj` is off by default in GPU4PySCF and this module does not
change that; it only makes the flag do what it says.

---

## 6. What it does not cover

**The gradient.**  `_DFHF.Gradients()` returns the density-fitted gradient
whatever `only_dfj` is — nothing in `df/grad/` reads the flag — so a gradient
taken from a converged RIJONX object is the derivative of a *different* energy.
That is upstream's behaviour and this module does not change it; it issues one
`RuntimeWarning` the first time such a gradient is asked for.  Making it
consistent needs the exchange term from the direct path's
`grad/rhf.py:_jk_energy_per_atom` (which is `fastejk`) and the Coulomb term
from the three-centre engine's own derivative, which GPU4PySCF has only for the
cderi route.  It is a piece of work, not an oversight, and it is why the
benchmark takes `--no-grad`.

**ROHF and GHF** keep `_DFHF.get_veff` (§1.2).  A GHF object keeps
`_DFHF.get_jk` as well, because that branch is correct already.

**The Hessian and the response solve** were not measured.  Both reach
`_DFHF.get_jk` and will therefore take the repaired branch, and §1.1's
flattening exists for exactly that case; but no timing here says whether
RIJONX is the right choice for them, and the Hessian in particular moves the
cderi to the host on purpose (`docs/DF_WORK_LIST.md` W8), which is a different
trade.

**Multiple GPUs**, as everywhere else in this repository.  The J engine and
`fastk` both assert a single device.

---

## 7. Reproducing

```bash
# the correctness bar
python tests/test_dfj.py [--basis def2-tzvpp]

# the measurement: three (or four) builds of the same Fock matrix
python benchmarks/dfj_probe.py --xyz benchmarks/molecules/taxol.xyz \
    --basis def2-svp --xc b3lyp --patch all --cycles 1 --no-grad \
    --scf --scf-ablate

# 284 atoms: the end-to-end half only, which is all that fits in 29 minutes
python benchmarks/dfj_probe.py --xyz benchmarks/molecules/PfPMT.xyz \
    --charge -1 --basis def2-svp --xc b3lyp --patch all --cycles 1 \
    --no-grad --no-percall --scf --scf-ablate

# where a RIJONX run spends its time, stage by stage
python benchmarks/df_anatomy.py --xyz ... --patch all --only-dfj

# ... and the failure this module repairs, on demand
FASTDFJ_OFF=1 python benchmarks/df_anatomy.py --xyz ... --patch all --only-dfj

# which entry points such a run reaches
python benchmarks/coverage_probe.py --xyz ... --patch all --only-dfj
```

Switches:

```
FASTDFJ_OFF=1              hand every call back to GPU4PySCF, i.e. restore
                           the broken `only_dfj` and the whole-density driver
FASTDFJ_NO_INCREMENTAL=1   keep the argument fix, restore GPU4PySCF's
                           `_DFHF.get_veff` (the ablation of §3.3)
```

Every number in this document is a whole Perlmutter GPU node
(`--qos=debug`, one A100-SXM4-40GB of the four used, 251 GB of host DRAM,
`OMP_NUM_THREADS=16`), GPU4PySCF 1.8.1, with `patch_all()` unless a row says
otherwise.  `docs/DF_WORK_LIST.md` §5 says why the login node is not usable for
any of it.
