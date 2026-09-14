# The XC Hessian on an open-shell reference

`docs/OPENSHELL_TMC.md` §4 ranks what an open-shell transition-metal complex
still runs on GPU4PySCF's own code with every module of this repository
loaded.  Its **item 2** is the analytic Hessian's two XC terms:

> `hessian.uks._get_vxc_deriv1` and `_get_vxc_deriv2` are `hessian.rks`'s
> twins with a spin index, and `fastxchess` patches only the `hessian.rks`
> pair.

This round takes it.  The shape of the gap is the one the two previous
open-shell rounds found four times between them -- `NumInt.nr_uks`,
`grad.uks.get_exc`, `dft.uks.UKS.get_veff` and `hessian.uks.nr_rks_fnlc_mo` --
and this is the fifth: **`hessian/uks.py` defines its own copies rather than
importing the restricted ones**, so the module was loaded, correct and never
called on an open-shell Hessian, and the only symptom was the wall clock.
`benchmarks/coverage_probe.py` grew a row for each of the four functions and a
`--hessian` flag, and §3b is those rows before and after.

What the port is worth: **6.7x-8.8x** on the two terms together across four
systems, two chemical classes and a 3x range of size, against **7.25x** for
the restricted pair on the same machine.  It is *above* the restricted number
rather than equal to it, and §4a says why in one line: upstream pays 2.38x for
the second spin channel where this module pays 1.96x.

---

## 1. Three corrections to §4 item 2, before the measurements

**"`hessian.rks`'s twins with a spin index" understates the difference, in
the one place it matters.**  Most of the unrestricted pair really is the
restricted body with an index — but `fxc` is a **2x2 block in the spin
index**, so the perturbed potential of spin `b` takes *both* spins' perturbed
densities:

    wv2[b] = sum_s  fxc[s, :, b] . dR_rho1[s]

That is not a relabelling, and it has a structural consequence for how the
work can be organised: the two channels cannot be run as two independent
passes over the grid, because each one needs the other's `d rho / d R` on the
same block.  They have to share a block — which, as it happens, is exactly
what makes the unrestricted port *faster* than the restricted one relative to
upstream: sharing the block means the per-block work that does not depend on
the spin is done once, and only the arithmetic that genuinely has two channels
is doubled (§4a).

A second, smaller difference in the same sentence: on 1.8.1 the **restricted**
pair is multi-device (`_get_vxc_deriv1_task` dispatched through
`multi_gpu.run`) and the unrestricted pair is a single plain function.  This
module asserts one device either way, so nothing turns on it here, but "twins"
would lead a reader to expect otherwise.

**"`fastxchess` is 7.3x on those terms at 42 atoms, so this is the largest
single number in the list" is the right conclusion from the wrong number.**
7.3x is the *restricted* ratio, and a ratio measured on one implementation is
not evidence about a different implementation of a different function.  It
happens to be conservative — the unrestricted pair measures 8.82x on the same
molecule at the same size (§4a) — but that could not be known before
measuring, and on the 60-atom case it is 6.69x, i.e. below the quoted figure.
The honest form of item 2's claim is that the *term* is 8.8 % of a stock
unrestricted Hessian (§4c), which is a smaller and more useful number than
either ratio.

**Item 2 does not mention the fourth caller.**  `df/hessian/uks.py:101`
reaches `_get_vxc_deriv1` as `uks_hess._get_vxc_deriv1` — a module-global
lookup at call time — so the density-fitting unrestricted Hessian goes through
the same patch, without a second patch point.  That is the opposite of the
`hessian/uks.py: from ... import nr_rks_fnlc_mo` case that
`docs/VV10_OPENSHELL.md` §1 had to work around, and the difference is only
which side of the import the name was bound on.  `tests/test_patch_aliases.py`
now imports `gpu4pyscf.df.hessian.{rks,uks}` before patching so that this stays
checked rather than argued.

---

## 2. What was added

`electrolites/fastxchess.py` grew `_deriv1_u` and `_deriv2_u` and patches
`hessian.uks._get_vxc_deriv1` and `_get_vxc_deriv2` alongside the
`hessian.rks` pair.  All four mechanisms `docs/README_fastxchess.md` records
carry over unchanged — the masked AO space, the atom screen, the batched atom
dimension, and the fixed-size scratch that turned out to be most of the
restricted function — and they are **shared between the spin channels** rather
than run twice:

| per grid block | GPU4PySCF's unrestricted pair | here |
|---|---|---|
| AO evaluation and mask | once | once |
| expansion to the full `nao` | once | not done at all |
| `D_s . ao` | twice (once per spin) | twice |
| `d rho / d R_A`, all atoms | `_make_dR_rho1` per atom **per spin** — 26 slice contractions an atom | one contraction with a 0/1 atom matrix per spin |
| the atom list | every atom, every block | the atoms with an AO in the mask, shared |
| `fxc` contraction | per atom, both spins | one call, both spins, atom as a free index |
| the two GEMMs (`deriv1`) | per (atom, direction, spin) | one batched GEMM per (atom block, spin) |
| the atom loop (`deriv2`) | ~10 kernels an atom **per spin** | six contractions per block per spin |

The only genuinely new piece of arithmetic is `_wv2_u`, the channel coupling
of §1.  Everything else is the restricted code with a spin index, which is why
this round's diff is small and its correctness suite is not.

Two implementation notes worth keeping:

* `_dR_rho1` grew an explicit `out=`.  Both spins' `d rho / d R` have to be
  resident at once (the coupling needs them together), so they are two rows of
  one `(2, natm_local, 3, 4, ngrids)` buffer.  The *AO-resolved product* it is
  built from — the `(3, 4, nao_sub, ngrids)` intermediate, which is the large
  one, 441 MB at 1123 AOs — is still built one spin and one direction at a
  time and never doubled.
* `_setup` grew an `unrestricted` argument, and the test is now that the
  **caller and the occupancy agree** (`mo_occ.ndim == 2` for the `hessian.uks`
  entry points, `== 1` for the `hessian.rks` ones) rather than that the
  occupancy has some particular shape.  The restricted path previously used
  `ndim != 1` as its "this is the other module's problem" guard; with both
  patched, an asymmetric guard would have let one entry point answer for the
  other, and nothing numerical would have said so.

---

## 3. Correctness

`tests/test_uxchess.py` calls both implementations on the same converged UKS
density in one process: `_get_vxc_deriv1` and `_get_vxc_deriv2` term by term,
and then the whole analytic Hessian patched against unpatched.  Eleven cases
over four functionals including a range-separated hybrid, five basis sets to
def2-TZVPP (which reaches g), multiplicities 2 to 6, an ECP, and the two paths
the module declines.

Three things the restricted suite cannot check and this one does: **both spin
blocks separately** (they have different shapes wherever `nocca != noccb`, so
a dropped index would show in one column only, and the number reported is the
worse of the two); **the channel coupling**, which a doublet separates from a
transposed one; and **that the patch was reached at all**, since an
unrestricted run falling through to upstream agrees with upstream perfectly.

Absolute maxima in atomic units, the worse of the two spin blocks.  `self` is
GPU4PySCF's `_get_vxc_deriv1` called twice on the same input, and `hself` is
the whole Hessian run twice on GPU4PySCF's own code -- both are upstream
against itself, and they are what the two right-hand columns have to be read
against:

| case | xc | atoms | AO | `deriv1` | `deriv2` | whole Hessian | `hself` | `self` |
|---|---|---|---|---|---|---|---|---|
| O2 / def2-SVP | B3LYP | 2 | 28 | 6.7e-16 | 5.7e-14 | 1.9e-12 | 1.8e-12 | 0 |
| O2 / def2-TZVPP (**g**) | PBE | 2 | 62 | 2.7e-15 | 2.8e-14 | 3.7e-12 | 3.7e-12 | 0 |
| CH3 / 6-31G\* (`nocca != noccb`) | PBE0 | 4 | 20 | 1.6e-15 | 1.4e-14 | 6.3e-13 | 8.5e-13 | 0 |
| NO / def2-SVP | ωB97X-D3BJ | 2 | 28 | 1.0e-15 | 1.4e-14 | 3.5e-12 | 2.9e-12 | 0 |
| CuCl2 / def2-SVP | B3LYP | 3 | 67 | 1.4e-14 | 1.8e-12 | 4.6e-10 | 5.9e-11 | 0 |
| CuCl2 / def2-TZVP (**8-prim s**) | BLYP | 3 | 119 | 2.3e-14 | 1.8e-12 | 4.7e-10 | **1.4e-09** | 0 |
| FeCl4(-) / def2-SVP (sextet) | B3LYP | 5 | 103 | 1.8e-14 | 4.6e-13 | 1.1e-09 | 9.1e-10 | 0 |
| Pt amine / def2-SVP (**ECP**) | B3LYP | 23 | 249 | 5.2e-14 | 1.3e-12 | -- | -- | 0 |
| Ni dithiolene / def2-SVP | B3LYP | 21 | 327 | 2.8e-14 | 2.7e-12 | -- | -- | 0 |
| O2 / def2-SVP (**declined**) | SVWN | 2 | 28 | 0 | 0 | 3.6e-12 | 1.8e-12 | 0 |
| O2 / def2-SVP (**declined**) | M06-2X | 2 | 28 | 0 | 0 | 1.9e-12 | 1.8e-12 | 0 |

**The two term columns are the ones with teeth**: 5.2e-14 at worst on
`deriv1` (the 249-AO case with an ECP) and 2.7e-12 on `deriv2`, against
quantities whose own scale is 0.6-1.1 and 90-4500 respectively, on both spin
blocks, with `self` exactly zero everywhere -- so upstream is deterministic
here and these numbers are the reassociation of the grid sums and nothing
else.

**The whole-Hessian column has a floor that this suite cannot lower, and
`hself` is what says so.**  On the organics it is 1e-12; on the d-block cases
it is 5e-10 to 1e-09 -- and the run-to-run spread of the *same* calculation
against itself is 6e-11 to 1.4e-09 there, which on CuCl2/def2-TZVP is
**larger** than the patched-unpatched difference it is supposed to bound.  The
J and K builds accumulate with `atomicAdd` and the CPHF iterates on top of
that at `conv_tol_cpscf = 1e-6`; on an open-shell transition-metal complex
that is a 1e-09 effect.  The whole-Hessian column is therefore reported as
bounded by the solver rather than asserted as an agreement, and the pass
criterion is the larger of 1e-8 of the Hessian's own scale and ten times
`hself`.  The 21- and 23-atom rows run the two terms only: a UKS Hessian of
those is minutes of CPHF for a column the two columns to its left already
cover.

The two declined rows run GPU4PySCF's code on **both** sides, so what is
asserted there is the fall-through itself (`_setup` returning None), not a
number.

`tests/test_xchess.py`, the restricted suite, is **unchanged at 11/11** — the
regression check that matters, because `_dR_rho1` and `_setup` are shared and
both changed.

`tests/test_xchess_fd.py` is the independent check: the analytic Hessian
against central differences of the analytic gradient, `eps = 1e-3` bohr,
`grids.level = 6`, tolerance 1e-3.  It does not compare the two
implementations, so an error they shared would still show -- and it is worth
more on the unrestricted rows than on the restricted ones, because a spin
coupling written the wrong way round still produces a symmetric,
plausible-looking Hessian that finite differences of the gradient do not
care about.  Three unrestricted rows were added:

| molecule / basis / functional | reference | max &#124;H_analytic - H_fd&#124; | max &#124;H&#124; | xctype |
|---|---|---|---|---|
| H2O / def2-SVP / B3LYP | RKS | 1.4e-05 | 0.71 | GGA (patched) |
| H2O / 6-31G\* / PBE | RKS | 1.8e-05 | 0.70 | GGA (patched) |
| NH3 / def2-SVP / B3LYP | RKS | 1.9e-05 | 0.67 | GGA (patched) |
| H2O / def2-SVP / SVWN | RKS | 1.0e-05 | 0.72 | LDA (declined) |
| **O2 / def2-SVP / B3LYP** | **UKS** | **3.2e-06** | 0.82 | GGA (patched) |
| **CH3 / 6-31G\* / PBE** | **UKS** | **8.0e-05** | 0.64 | GGA (patched) |
| **O2 / def2-SVP / SVWN** | **UKS** | **8.5e-06** | 0.81 | LDA (declined) |

Seven of seven within 1e-3, and all seven within 8e-5, which is the size of
the finite-difference truncation error at `eps = 1e-3` rather than of anything
in the Hessian.  The unrestricted LDA row is the same control the restricted
LDA row is: `fastxchess` declines LDA on both spin cases, so what that row
reports is upstream's own number, and it is here so that "the declined branch
still agrees" stays a measurement.  The restricted rows are unchanged from
`docs/README_fastxchess.md` §5, which is the regression check on the shared
helpers.

---

### 3b. And that the pair is on the code path

Correctness cannot see a fall-through.  `benchmarks/coverage_probe.py` grew a
row for each of the four functions and a `--hessian` flag, since an SCF and a
gradient reach none of them.  A UKS B3LYP/def2-SVP SCF, gradient **and
analytic Hessian** on the CuCl2 doublet, everything patched, against the same
run with `FASTXCHESS_OFF=1` — which is the work distribution before this round:

| entry point | `FASTXCHESS_OFF=1` | patched |
|---|---|---|
| `fastxchess vxc_deriv1` (restricted) | (-) | (-) |
| `fastxchess vxc_deriv2` (restricted) | (-) | (-) |
| **`fastxchess uks_deriv1`** | 1 call / **0 ours** / 1 stock | 1 / **1** / 0 |
| **`fastxchess uks_deriv2`** | 1 / **0** / 1 | 1 / **1** / 0 |
| `fastxcfxc nr_rks_fxc` | (-) | (-) |
| `fastxc nr_uks` | 16 / 16 | 16 / 16 |
| `fastgradh get_hcore` | 2 / 2 | 2 / 2 |

and the same run on closed-shell glycine reads `vxc_deriv1` and `vxc_deriv2`
**1 / 1 / 0** with both `uks_` rows `(-)`, i.e. neither path answers for the
other.

Two rows in that table are not about this round and are worth recording
because this is the first coverage run in the repository that includes a
Hessian:

* **`fastgradh get_hcore` is called, twice.**  `docs/OPENSHELL_TMC.md` §2
  corrected the README to say the module "is inert on a gradient on this
  release" because `GradientsBase.get_hcore` is reached only from
  `get_grad_hcore`, i.e. a Hessian's CPHF right-hand side.  Here is that
  claim's other half measured: on a Hessian it *is* reached, once per spin.
* **`fastxcfxc nr_rks_fxc` reads `(-)` on both spin cases.**  On the
  unrestricted side the Hessian's CPHF grid half is `NumInt.nr_uks_fxc`
  (§4c), which nothing here patches; on the restricted side it is
  `nr_rks_fxc_mo`, which `fastxcfxc` declines by design.  So a Hessian's
  response solve gets no XC acceleration on either spin case — which the
  README says in prose and no table had shown.

One observation that is not this round's: the `fastj` class-block counter
reads **1047 / 552 / 495** on the UKS Hessian — 47 % of the multi-density
Coulomb class blocks handed back — against 840 / 714 / 126 (15 %) on the
closed-shell glycine Hessian.  Recorded, not diagnosed.

---

## 4. What it is worth

One **A100-SXM4-40GB**, one GPU visible to the process, on a batch node with
nothing else on it (`docs/OPENSHELL_TMC.md` §5 is the round that learned why
that sentence has to be here).  B3LYP/def2-SVP, level-3 grids.

### 4a. The two terms, at fixed density

`benchmarks/xchess_bench.py`, both implementations in one process on one
converged density, minimum of three.  `no screen` is the ablation that keeps
the masked AO space and the batching and drops the atom screen.  The last row
is the **closed-shell** 42-atom case, which is the regression check on the
restricted path and reproduces `docs/README_fastxchess.md` §6's 5.07x / 12.94x
/ 7.34x to within run-to-run scatter:

| system | | `_get_vxc_deriv1` | | `_get_vxc_deriv2` | | **both** | | no atom screen |
|---|---|---|---|---|---|---|---|---|
| | | stock -> ours | | stock -> ours | | | | |
| Ni dithiolene, 21 at, 327 AO, doublet | UKS | 5.727 -> 1.103 | 5.19x | 6.315 -> 0.682 | 9.26x | 12.04 -> 1.785 | **6.75x** | 6.26x |
| alanine dipeptide cation, 22 at, 200 AO | UKS | 4.658 -> 0.820 | 5.68x | 5.100 -> 0.559 | 9.13x | 9.76 -> 1.379 | **7.08x** | 7.01x |
| penicillin V cation, 42 at, 430 AO | UKS | 17.415 -> 2.969 | 5.87x | 19.026 -> 1.162 | 16.38x | 36.44 -> 4.130 | **8.82x** | 7.63x |
| Cu macrocycle, 60 at, 515 AO, quartet | UKS | 33.444 -> 8.570 | 3.90x | 36.265 -> 1.857 | 19.53x | 69.71 -> 10.427 | **6.69x** | 6.52x |
| penicillin V, 42 at, 430 AO | RKS | 7.600 -> 1.503 | 5.06x | 7.700 -> 0.606 | 12.70x | 15.30 -> 2.109 | 7.25x | 6.33x |

Four things to take from this table.

**The last two rows are the same molecule, and they are the point of the
round.**  Going from one spin channel to two, on penicillin V at 42 atoms:

| | `deriv1` | `deriv2` | both |
|---|---|---|---|
| GPU4PySCF pays | 2.29x | 2.47x | **2.38x** |
| this module pays | 1.98x | 1.92x | **1.96x** |

Upstream pays *more* than double for the second channel: it calls
`_make_dR_rho1` and the `fxc` contraction once per spin **inside** its
`for ia in range(natm)` loop, so the second copy is not cheaper than the first
and the per-atom Python and launch overhead is paid on twice as many kernels.
This module pays slightly *less* than double, because what is spin-independent
is done once a block rather than once a spin -- the atom list and the 0/1 atom
matrix, the mask indexing, and the flat index arrays the scatter uses -- and
only the arithmetic with two real channels is doubled.  That gap is the whole
reason the unrestricted ratio (8.82x) sits above the restricted one (7.25x)
rather than beside it, and it is measured on one molecule at one size rather
than inferred.  Note what is *not* on that list: `D_s . ao`, the two spin
densities and the AO-resolved `d rho / d R` products are genuinely per spin
here too, and are built one spin at a time so that the largest intermediate --
441 MB at 1123 AOs -- is never doubled.

**The two terms disagree about size, in the same direction as the restricted
pair.**  `deriv1` falls with size (5.19 -> 5.68 -> 5.87 -> 3.90x) and `deriv2`
rises (9.26 -> 9.13 -> 16.38 -> 19.53x).  `docs/README_fastxchess.md` §6
records the same disagreement at 42 and 113 atoms and the reason: what the
batching removes is a *per-atom* cost, which hurts most when `nao` and `nocc`
are small, while what the screens remove grows as a grid block reaches a
smaller fraction of the molecule.  `deriv2` has no real GEMM, so only the
second effect acts on it and its ratio climbs; `deriv1` has both and they
cancel to something between 4x and 6x over this range.  **Neither ratio is a
ceiling that holds at 500 atoms**, and four points is not a scaling law.

**The atom screen is a property of shape, not of atom count.**  It is worth
1.21x on `deriv1` for penicillin V at 42 atoms, 1.13x for Ni dithiolene at 21,
and **1.01x for the alanine dipeptide cation at 22 and 1.03x for the Cu
macrocycle at 60**.  A 60-atom macrocycle is a compact globular thing and a
4096-point grid block reaches most of it; penicillin V is extended.  So the
"the screen improves with system size" of `docs/README_fastxchess.md` §2 is
true along a series of *similar* molecules and is not a rule about atom
counts — which is exactly the kind of claim that only shows up when a second
chemical class is measured.  §4b has the screening fractions that say so
directly.

**Two chemical classes, and the ratio does not care.**  The organic radical
cations (7.08x at 22 atoms, 8.82x at 42) bracket the d-block complexes
(6.75x at 21, 6.69x at 60) rather than separating from them.  Nothing in this
module knows what a metal is, and the measurement says so.

### 4b. The screening ceiling, which both spin cases share

`benchmarks/xchess_locality.py` weights both screening fractions by grid
points.  They are properties of the molecule, the basis and the grid -- the AO
mask does not know what the density is, let alone how many spin channels it
has -- so this is one ceiling that the restricted and the unrestricted pair
share.  The last column is the atom screen measured on `_get_vxc_deriv1` in
§4a, against the `1 / natm_local` it can at best be worth:

| system | atoms | `natm_local/natm` | `nao_sub/nao` | product | ceiling | atom screen: ceiling | measured |
|---|---|---|---|---|---|---|---|
| alanine dipeptide cation | 22 | **0.978** | 0.723 | 0.712 | 1.4x | 1.02x | **1.01x** |
| Ni dithiolene | 21 | 0.774 | 0.516 | 0.409 | 2.4x | 1.29x | **1.13x** |
| penicillin V cation | 42 | 0.745 | 0.486 | 0.379 | 2.6x | 1.34x | **1.21x** |
| Cu macrocycle | 60 | **0.948** | 0.563 | 0.541 | 1.8x | 1.05x | **1.03x** |

sorted by `natm_local/natm` rather than by size, because that is the variable:
the measured atom screen tracks its own ceiling to within 13 % on all four,
and the ceiling has no relation at all to the atom count.  A 4096-point grid
block reaches **98 %** of a 22-atom alanine dipeptide and **95 %** of a
60-atom macrocycle, because both are compact; it reaches 75 % of a 42-atom
penicillin V because that molecule is extended.  Weight the same numbers by
`nao_sub/nao` and the *combined* ceiling is 1.4x for alanine against 2.6x for
penicillin V -- so the same two screens are worth nearly twice as much on one
22-atom organic radical as on another molecule of similar size.

The measured ratios in §4a are far above these ceilings (5.19x against 2.4x on
Ni dithiolene, 3.90x against 1.8x on the macrocycle) for the reason
`docs/README_fastxchess.md` §3b gives: the ceiling counts only the work the
screens remove, and most of what this module removes is work GPU4PySCF was not
doing on purpose -- a per-block allocation of `nao_sub`-sized scratch, and one
GEMM per (atom, direction) too small to fill the card.

### 4c. A whole unrestricted Hessian

`benchmarks/hessian_anatomy.py --spin`, Ni dithiolene (21 atoms, 327 AO,
doublet), stock GPU4PySCF 1.8.1 against `patch_all()`.  The two Hessians agree
to **2e-10 relative** on the trace (22.03455454509 against 22.03455454044) and
1.3e-10 on the Frobenius norm.

| stage | stock | share | patched | share | |
|---|---|---|---|---|---|
| **whole Hessian** | **139.70 s** | 100 % | **87.52 s** | 100 % | **1.60x** |
| `partial_hess_elec` | 43.60 | 31.2 % | 27.22 | 31.1 % | |
| ↳ `ejk_ip2` | 36.30 | 26.0 % | 25.51 | 29.1 % | |
| ↳ **`vxc_deriv2`** | **6.43** | **4.6 %** | **0.83** | 1.0 % | **7.7x** |
| ↳ `vxc_diag`, untouched | 0.41 | 0.3 % | 0.41 | 0.5 % | |
| `make_h1` | 30.20 | 21.6 % | 25.25 | 28.8 % | |
| ↳ **`vxc_deriv1`** | **5.84** | **4.2 %** | **1.15** | 1.3 % | **5.1x** |
| ↳ `jk_ip1`, untouched | 23.35 | 16.7 % | 23.42 | 26.8 % | |
| `solve_mo1` (CPHF) | 65.81 | 47.1 % | 34.96 | 39.9 % | 1.88x |
| ↳ **`nr_uks_fxc`, untouched** | **4.56** | **3.3 %** | **4.56** | **5.2 %** | **1.00x** |
| ↳ `get_k` in CPHF | 36.06 | 25.8 % | 21.19 | 24.2 % | |
| ↳ `get_j` in CPHF | 23.98 | 17.2 % | 8.12 | 9.3 % | |

**This round's two terms are 8.8 % of a stock unrestricted Hessian and become
2.3 % of the patched one**: 12.27 s -> 1.98 s, **6.2x** in situ against 6.75x
at fixed density -- the in-situ figure is one cold call each where the
benchmark takes the minimum of three, so the two agreeing this closely is the
check that the fixed-density table is measuring the thing a Hessian pays.
On its own the module is **1.08x** on the whole run; the rest of the 1.60x is
`fastk`, `fastj` and `fastejk2` inside the CPHF solve and the `ip2` build.

The rows to read next are the three that did not move.

* **`jk_ip1` is 16.7 % stock and 26.8 % patched, at 1.00x.**
  `hessian.rhf._get_jk_ip1` is the largest untouched item in `make_h1`, and an
  unrestricted Hessian calls it **twice**, once per spin density, where a
  restricted one calls it once.
* **`nr_uks_fxc` is 3.3 % stock and 5.2 % patched, at exactly 1.00x.**  This is
  `docs/OPENSHELL_TMC.md` §4 **item 1** — the unrestricted response path —
  showing up inside a Hessian rather than in a TDDFT run.  `fastxcfxc`
  replaces `nr_rks_fxc`; there is no `nr_uks_fxc` replacement, so the grid half
  of an open-shell CPHF solve is stock.  It is the natural next item, and its
  value is larger than 5.2 % suggests because the same function is the grid
  half of every open-shell polarizability, TDA/TDDFT, NMR and C6 run.
* **`vxc_diag` is 0.3 %**, as on the restricted side, and is still not worth
  writing.

---

## 5. Scope, and what is left

Covered, now on both spin cases: **GGA** (so B3LYP, PBE0, ωB97X-D3BJ and the
rest of the hybrids), one GPU.  The density-fitting Hessian reaches the same
patch, by the module-global lookup of §1, and nothing in `_setup` declines a
fitted mean field -- correct, since the XC Hessian term does not depend on how
the Coulomb build is done, but no suite here runs one, so it is recorded as
what the code does rather than as a tested claim.

Declined, falling through to GPU4PySCF unchanged and asserted to do so by
`tests/test_uxchess.py`: **meta-GGA**, **LDA**, the **grid-response** path
(`hessobj.grid_response`, which `hessian/uks.py` routes to
`_get_vxc_deriv1_grid_response`), the **VV10 Hessian**
(`_get_enlc_deriv2`), and multi-GPU.  `_get_vxc_diag`, the third XC Hessian
term, is 0.3 % and is left alone.

One negative result worth keeping.  `docs/README_fastxchess.md` §5 records
that GPU4PySCF's **restricted** meta-GGA `_get_vxc_deriv1` is not reproducible
run to run — 3.0e-5 against itself on H2O/M06-2X — and gives that as a reason
meta-GGA is not covered: there is no reference to converge against.  The
declined M06-2X row of `tests/test_uxchess.py` is the same test on the
unrestricted function, and it reads **exactly 0.0** on O2/def2-SVP, on both
runs of this suite.  That is one molecule and one functional, so it does not
overturn the restricted observation; it does mean the nondeterminism is not a
property of the meta-GGA branch as such, and that whoever takes meta-GGA next
should find out what it *is* a property of before concluding there is nothing
to converge against.

What is left in this neighbourhood, in the order the profile of §4c puts them:

1. **`nr_uks_fxc`** — `docs/OPENSHELL_TMC.md` §4 item 1.  5.2 % of a patched
   unrestricted Hessian, and the grid half of *every* open-shell response
   property besides.  The arithmetic is `nr_rks_fxc`'s per spin with `spin=1`
   in the libxc call, i.e. the same relation `nr_uks` bears to `nr_rks`, and
   `fastxcfxc` is already written against that structure.
2. **`_get_jk_ip1`** — 26.8 % of a patched unrestricted Hessian and untouched
   by anything here, on either spin case.  Twice the calls on an open-shell
   reference.
3. **meta-GGA**, on both spin cases, subject to the paragraph above.
4. **LDA**, still 0.3-0.5 % and still not the next hour.

## Files

| file | what |
|---|---|
| `electrolites/fastxchess.py` | the module; `_deriv1_u`, `_deriv2_u`, `_wv2_u` and the `hessian.uks` patch are this round |
| `tests/test_uxchess.py` | §3, eleven unrestricted cases against GPU4PySCF in one process |
| `tests/test_xchess_fd.py` | §3, the finite-difference check, three unrestricted rows added |
| `tests/test_xchess.py` | the restricted suite, unchanged, run as the regression check |
| `tests/test_patch_aliases.py` | the two new saved originals, and the `df/hessian` imports |
| `benchmarks/xchess_bench.py` | §4a, `--spin` picks the unrestricted pair |
| `benchmarks/xchess_locality.py` | §4b, `--spin` only so that it will build an open-shell molecule |
| `benchmarks/hessian_anatomy.py` | §4c, `--spin` selects a separately wired unrestricted stage list |
| `benchmarks/coverage_probe.py` | §3b, four new rows and `--hessian` |
