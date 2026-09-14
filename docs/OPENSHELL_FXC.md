# The XC response kernel on an open-shell reference

`docs/OPENSHELL_TMC.md` §4 item 1, the last entry on that list that is not a
basis-set limit:

> **The unrestricted response path.**  `fastxcfxc` replaces `nr_rks_fxc`;
> `nr_uks_fxc` has no replacement, so the grid half of *every* open-shell
> polarizability, TDA/TDDFT, NMR and C6 run is stock. […] The arithmetic is
> `nr_rks_fxc`'s per spin with `spin=1` in the libxc call, i.e. the same
> relation `nr_uks` bears to `nr_rks`.

This round takes it, and it closes the loop on the previous one: the same
function turns out to be the grid half of an **unrestricted Hessian's CPHF
solve** as well, which is the row `docs/OPENSHELL_HESSIAN.md` §4c measured at
exactly 1.00x and ranked first on its own to-do list without noticing it was
this entry.

* **2.2x-11.9x** on the term, over a sweep of the one number the molecule does
  not fix -- `nset`, the right-hand sides a solver asks for -- at two sizes
  and in two chemical classes (§4a).
* **5.5x** in situ inside an open-shell TDA run, which is **1.16x** on the
  whole run because the XC half is only 10.5 % of it on this system (§4b).
* **1.9x** in situ inside an unrestricted Hessian's CPHF (§4c).
* One new CUDA kernel.  Everything else is a **layout** choice (§2).

---

## 1. Two corrections to §4 item 1, before the measurements

**"The arithmetic is `nr_rks_fxc`'s per spin" is the one thing that is not
true, and it is the whole design.**  `fxc` on an unrestricted reference is a
**2x2 block in the spin index** -- `(2, ncomp, 2, ncomp, ngrids)` -- so the
perturbed potential of spin `t` takes *both* spins' perturbed densities:

    wv[t][y] = sum_{s,x} fxc[t][y][s][x] . rho1[s][x]

Running `nr_rks_fxc` once per spin would drop the off-diagonal blocks and give
the wrong answer, not a slower one.  What *is* per spin -- the AO evaluation,
the masked density product, the density contraction, the weighted AOs, the
potential GEMM -- is everything else, and §2 is what follows from that split.

The rest of the sentence is right in a way worth keeping: `fastxcfxc` really
was already written against the structure, and the round is small because of
it.  **One** new CUDA kernel, `fxc_wv_u`, for the contraction above; the other
three serve both spin cases unchanged.

**Item 1's list of consumers is missing the one the previous round cared
about.**  It names "polarizability, TDA/TDDFT, NMR and C6" -- everything that
reaches `nr_uks_fxc` through `scf/_response_functions.py` with a plain
perturbed density matrix.  It does not name the **analytic Hessian**, and the
reason it is easy to miss is that the restricted and unrestricted Hessians do
not agree about which function to call:

| | a restricted Hessian's CPHF | an unrestricted Hessian's CPHF |
|---|---|---|
| calls | `hessian/rks.py:nr_rks_fxc_mo` | `hessian/uks.py` → `ni.nr_uks_fxc` |
| a separate MO-basis function? | yes | **no -- there is no `nr_uks_fxc_mo`** |
| the density it is handed | MO-basis, in that function's own form | a `tag_array` with `.mo1`, so `nr_uks_fxc`'s `mo1` branch |
| covered here? | no, and for a measured reason | **yes** |

So `docs/OPENSHELL_HESSIAN.md` §4c's "the natural next item" and
`docs/OPENSHELL_TMC.md` §4 item 1 are the same function, and neither document
said so.  The practical consequence is that this round had to cover two
branches of one function rather than one, and §2 says what the second cost.

One more thing that fell out of looking: **nothing on 1.8.1 appears to reach
the `mo1` branch of `nr_rks_fxc`**.  `hessian/rhf.py:gen_vind` tags the
density, but `hessian/rks.py` routes it to `nr_rks_fxc_mo`; every other caller
passes a plain matrix.  That is why the restricted `mo1` decline stays -- it
is a decline of code with no caller found, and enabling it would be shipping
an untested path -- and why the unrestricted one does not.

---

## 2. What was added

### The layout, which is most of it

The two spins' density matrices are **interleaved** -- a, b, a, b, … rather
than all of a then all of b.  With that one choice, `rho1` for `nset`
unrestricted right-hand sides occupies exactly the memory of `rho1` for
`2*nset` restricted ones:

    rho1[(2k+s)][x][g]        the restricted kernels' view, nset -> 2*nset
    rho1[k][(s,x)][g]         the contraction's view, the pair index

so `fxc_rho1_*` (the density), `fxc_scale_*` (the weighted AOs) and both GEMMs
run **unchanged** with `nset` doubled, and only the `fxc` contraction has to
know that the component index is a pair.  `_prep` does the interleaving,
`_fxc_blocked` carries an `nspin` argument, and the restricted and
unrestricted paths are the same function rather than two copies -- which
matters more here than it usually does, because the two differ in six places
and every one of them is an index.

The new kernel, `fxc_wv_u<NC>`, is `fxc_wv<NCOMP>` with `NCOMP = 2*NC` and the
scale factors applied to the component *within* a spin (`y` or `y - NC`),
which is the whole diff.

### The `mo1` branch

An unrestricted Hessian's CPHF hands the perturbation in the MO basis, and
GPU4PySCF evaluates the density from it with `eval_rho4` instead of from a
density matrix.  **That evaluation is left to GPU4PySCF**: the module calls
`eval_rho4` once per spin into a scratch buffer, transposes the result into
the interleaved layout, and batches everything downstream of it -- the `fxc`
contraction, the weighted AOs and the potential GEMM, which is where the
per-set Python loop was.

That is a deliberate half-measure and §4c measures what it leaves: batching
`eval_rho4` itself would be a third density kernel, and the profile says how
much is left on the table before writing one.

---

## 3. Correctness

`tests/test_ufxc.py` calls both implementations on the same converged UKS SCF
and the same `fxc` cache.  Fourteen cases over seven functionals including a
range-separated hybrid and three meta-GGAs, five basis sets to def2-TZVPP,
multiplicities 2 to 6, an ECP, and 20 to 327 AOs.

Every case is checked at **both `hermi` values** -- 1 is CPKS/NMR/the
polarizability, 0 is TDA/TDDFT and the one place where the two
implementations do genuinely different arithmetic -- and in **both modes**:
`blocked` is the shipped path with the custom kernels, `nokernel` keeps the
same layout and the same batched GEMMs and puts GPU4PySCF's own per-set
helpers back.  Running both separates a layout error (which fails both
columns) from a kernel error (which fails only `blocked`).  The two spin
blocks are compared separately, and the right-hand sides are built so the two
channels differ, because a coupling written the wrong way round would still be
symmetric and still look like a potential.

| worst of the 13 that ran, relative | `hermi = 1` | `hermi = 0` |
|---|---|---|
| `blocked` | **3.1e-15** | **1.0e-13** |
| `nokernel` | 3.3e-15 | 1.0e-13 |

and a whole **UKS TDA** spectrum, patched against unpatched on the same SCF,
agrees to **6.3e-14 eV** at worst over the eleven cases small enough to run
one.  The `hermi = 0` worst case is the 249-AO Pt complex (with an ECP); it
reads 1.0e-13 in *both* modes, so it is the once-per-call symmetrisation
reassociating a larger sum, not the kernel.  One case is skipped rather than
passed: CuCl2/def2-SVP/**M06-2X** does not converge in 200 UKS cycles, which
`docs/OPENSHELL_TMC.md` §3 records for the same molecule and functional.

`tests/test_fxc.py`, the restricted suite, is **unchanged at 18/18** -- the
regression check that matters, since `_prep` changed signature and the shipped
path was generalised over the spin count.

### 3b. The `mo1` branch

Nothing in the table above reaches it: an unrestricted Hessian's CPHF is the
only caller that tags the perturbed density with `mo1`, so the check is a
whole UKS Hessian with `nr_uks_fxc` patched and unpatched -- and `hself`, the
same Hessian run twice on GPU4PySCF's own code, beside it, because the CPHF's
own run-to-run spread is the floor and on a d-block system it is not small.
`fastxcfxc.stats()` is the assertion that the branch was taken at all: a
fall-through would agree perfectly and say nothing.

| case | xc | atoms | AO | whole Hessian | `hself` | `mo1` calls |
|---|---|---|---|---|---|---|
| O2 / def2-SVP | B3LYP | 2 | 28 | 1.9e-12 | 1.9e-12 | 8 |
| CH3 / 6-31G\* | PBE0 | 4 | 20 | 5.8e-13 | 8.8e-13 | 7 |
| CuCl2 / def2-SVP | B3LYP | 3 | 67 | 4.6e-10 | **4.7e-10** | 9 |
| O2 / def2-SVP | M06-2X | 2 | 28 | 1.9e-12 | 1.8e-12 | 8 |

`fastxcfxc.stats()` reads `{'ours_u': 0, 'ours_u_mo': 32, 'stock_u': 0}` --
32 calls through the branch and not one decline.  Every row's difference is
at or below the same calculation's spread against itself, which on CuCl2 is
4.7e-10 against the 4.6e-10 being bounded: the whole-Hessian column there is
the CPHF solve at `conv_tol_cpscf = 1e-6` on an open-shell transition metal,
not this module.  `docs/OPENSHELL_HESSIAN.md` §3 records the same floor on
the same molecule for a different term.

### 3c. And that it is on the code path

`benchmarks/coverage_probe.py` grew a `nr_uks_fxc` row and prints
`fastxcfxc.stats()`.  A UKS B3LYP/def2-SVP SCF, gradient **and analytic
Hessian** on the CuCl2 doublet, everything patched, against the same run on
closed-shell glycine:

| entry point | CuCl2, UKS | glycine, RKS |
|---|---|---|
| `fastxcfxc nr_rks_fxc` | (-) | **(-)** |
| **`fastxcfxc nr_uks_fxc`** | **9 calls / 9 ours / 0 stock** | (-) |
| `fastxchess uks_deriv1` / `uks_deriv2` | 1 / 1 / 0 each | (-) |
| `fastxchess vxc_deriv1` / `vxc_deriv2` | (-) | 1 / 1 / 0 each |
| `fastxcfxc.stats()` | `ours_u: 0, ours_u_mo: 9, stock_u: 0` | all zero |

Two rows to read rather than count.  **All nine unrestricted calls are the
`mo1` branch** (`ours_u: 0`), because an SCF and a gradient never reach this
function and a Hessian reaches it only through CPHF -- the plain-density
branch belongs to TDDFT and the polarizability, which §4b exercises instead.
And **`fastxcfxc nr_rks_fxc` reads `(-)` on the closed-shell Hessian**, which
is §1's table measured: a restricted Hessian's CPHF is served by
`hessian/rks.py:nr_rks_fxc_mo` and does not come through here at all, so the
restricted side of this module gets no share of a Hessian while the
unrestricted side gets all of it.

---

## 4. What it is worth

B3LYP/def2-SVP, level-3 grids, one GPU visible to the process, on a batch node
with nothing else on it.  §4a and §4c are an **A100-SXM4-40GB**; §4b is an
**A100-SXM4-80GB**, so its absolute seconds are not comparable with the other
two sections' -- its three arms are one job on one card, so its ratios are.

### 4a. The term, swept over `nset`

`benchmarks/fxc_bench.py --spin`, both implementations in one process on one
converged density, minimum of three.  `nset` is the number of right-hand sides
the solver hands the function -- three dipole directions for a polarizability,
a Davidson subspace for TDDFT, `3*natm`-ish for a CPHF -- and it is swept
because it is the one parameter the molecule does not fix.  `nokernel` is the
ablation that keeps the layout and the batched GEMMs and drops the custom
kernels.

**Ni dithiolene, 21 atoms, 327 AO, doublet:**

| `nset` | `hermi=1` stock | ours | | nokernel | | `hermi=0` stock | ours | |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0701 s | 0.0274 | 2.56x | 0.0512 | 1.37x | 0.0914 s | 0.0276 | 3.31x |
| 3 | 0.1794 | 0.0416 | 4.31x | 0.0710 | 2.53x | 0.2434 | 0.0416 | 5.86x |
| 5 | 0.2895 | 0.0577 | 5.02x | 0.0941 | 3.08x | 0.3799 | 0.0577 | 6.59x |
| 10 | 0.5665 | 0.0909 | 6.23x | 0.1595 | 3.55x | 0.7403 | 0.0909 | 8.14x |
| 20 | 1.0780 | 0.1654 | 6.52x | 0.3007 | 3.59x | 1.4645 | 0.1653 | 8.86x |
| 40 | 2.1270 | 0.3164 | **6.72x** | 0.5794 | 3.67x | 2.8886 | 0.3155 | **9.16x** |

and the other three systems at the ends of the same sweep:

| system | | `hermi=1`, `nset` 1 → 40 | `hermi=0`, `nset` 1 → 40 |
|---|---|---|---|
| Ni dithiolene, 21 at, 327 AO | UKS | 2.56x → **6.72x** | 3.31x → **9.16x** |
| alanine dipeptide cation, 22 at, 200 AO | UKS | 2.78x → **9.11x** | 3.51x → **11.87x** |
| penicillin V cation, 42 at, 430 AO | UKS | 2.19x → **4.96x** | 2.78x → **6.74x** |
| penicillin V, 42 at, 430 AO | RKS | 1.69x → 4.14x | 2.03x → 5.79x |

Four things to take from this.

**The unrestricted ratio is above the restricted one on the same molecule, and
for the same reason the previous round found.**  Penicillin V at 42 atoms,
`nset = 40`:

| | `hermi = 1` | `hermi = 0` |
|---|---|---|
| GPU4PySCF pays for the second spin channel | 2.37x | 2.28x |
| this module pays | **1.98x** | **1.95x** |

`docs/OPENSHELL_HESSIAN.md` §4a measured 2.38x against 1.96x for the two XC
Hessian terms on this same molecule.  Two different functions, measured
independently, agree on the number -- which is what makes it a property of
the treatment (share the spin-independent per-block work, double only the
arithmetic that has two channels) rather than a property of either function.

**`hermi = 0` gains more than `hermi = 1`, by about 1.35x, everywhere.**  That
is the docstring's item 3 measured on the unrestricted path: for a
non-Hermitian right-hand side GPU4PySCF forms a second `dm^T . ao` product and
three more contractions *per grid block*, where this module symmetrises the
density once per call.  TDA and TDDFT are the `hermi = 0` callers, so they are
the ones that gain.

**The custom kernel is worth about 1.8x on top of the batched GEMMs**, roughly
independent of `nset` past 5 (0.3164 s against 0.5794 s at `nset = 40`).  The
`nokernel` column is the honest form of "what the CUDA in this module buys",
and it is smaller than the whole ratio by design: the layout and the batching
are most of it.

**The ratio falls with system size and rises with `nset`.**  4.96x at 430 AO
against 6.72x at 327 and 9.11x at 200, at the same `nset = 40`; and 2.2-2.8x
at `nset = 1` against 5-12x at `nset = 40` everywhere.  What this module
removes is per-set launch and per-set kernel overhead, so it is worth most
where there are many small sets and least where each set's GEMM is already
large enough to fill the card.  **None of these is a ceiling at 1000 AO**, and
four systems is not a scaling law.

### 4b. Inside an open-shell TDA run

`benchmarks/response_anatomy.py --prop tda --spin`, five roots, Ni dithiolene,
three arms.  Every arm was run **three times**, for a reason §4b' records, and
all nine runs give the same five excitation energies to nine digits
(1.041293633, 1.092923309, 1.525416022, 1.666957927, 1.758700831 eV).

| stage | stock | `--patch fastxcfxc` | `--patch all` |
|---|---|---|---|
| **whole TDA** | 66.43 / 67.24 / **64.80 s** | 57.48 / 59.04 / **59.36 s** | (68.16) / 30.47 / **30.45 s** |
| ↳ **XC response (`nr_uks_fxc`)** | 7.00 / 7.11 / **7.15** | 1.27 / 1.27 / **1.26** | 1.26 / 1.27 / **1.28** |
| ↳ Coulomb + exchange, untouched by this round | 55.08 / 56.76 / **55.13** | 53.66 / 55.01 / **55.53** | 64.17 / 26.54 / **26.56** |
| share of the run the XC response is | 10.5-11.0 % | 2.1-2.2 % | 4.2 % |

**The term is 5.5x-5.7x in situ** (7.00-7.15 s -> 1.26-1.28 s), against
**6.6x** for the same molecule at the `nset = 5` and `hermi = 0` this run asks
for in §4a's fixed-density sweep.  The in-situ figure covers all 18 Davidson
calls including the first, cold one, so the two agreeing to 20 % is the check
that §4a is measuring what a TDA actually pays.  On the whole run
`fastxcfxc` alone is **1.09x-1.16x**, and with every module loaded the TDA is
**2.1x-2.2x**.

**Why 5.5x on the term is only 1.1x on the run: the XC half is 10.5-11.0 % of
an open-shell TDA on this system.**  `docs/README_fastxcfxc.md` measures the
same term at **25.2 %** of a five-root *closed-shell* TDA on penicillin V, so
the share has more than halved -- for the reason `docs/OPENSHELL_TMC.md` §5
gives for the SCF and the gradient: on a transition-metal complex the exchange
build is the whole game.  Anyone carrying the 25 % figure over to an
open-shell d-block system would be wrong by a factor of two and a half.  The
in-situ *ratio* is what transfers between systems; the *share* is not.

### 4b'. The number that needed a second run

The first job to run these three arms measured `--patch all` at **68.16 s**,
with its Coulomb-and-exchange row at **64.17 s** against stock's 55.08 -- a
1.16x *slow*down of a row this round does not touch, which would have said
that `patch_all()` is the wrong thing to do to an open-shell TDA on a d-block
complex.  Repeating the three arms twice more puts that row at **26.54 and
26.56 s**, i.e. 2.1x *faster* than stock, and the whole run at 30.47 and
30.45 s.

The 68 s was **first-run NVRTC**: `--patch all` was the first arm in that job
to use `fastjk`'s kernels, and the compile landed inside the timed region.
CuPy caches on disk on the source hash, and Perlmutter's `$HOME` is shared, so
by the next job the cache was warm and the cost was gone.  This is the same
trap `docs/OPENSHELL_TMC.md` §5 records for a cold-node glycine SCF (21.5 s
against 6.4 s stock, `get_j` alone 0.45 -> 8.56 s), and the same rule caught
it: **one run is not a baseline**.  It is recorded here rather than quietly
dropped because the plausible-looking wrong number and the right one differ by
a factor of 2.2 and nothing but a repetition separates them.

### 4c. Inside an unrestricted Hessian's CPHF

`benchmarks/hessian_anatomy.py --spin`, Ni dithiolene, everything patched,
against the same run before this round (`docs/OPENSHELL_HESSIAN.md` §4c, same
card):

| stage | before this round | after | |
|---|---|---|---|
| whole Hessian | 87.52 s | **86.39 s** | 1.01x |
| `solve_mo1` (CPHF) | 34.96 | 32.97 | 1.06x |
| ↳ **`nr_uks_fxc`** | **4.56** | **2.43** | **1.88x** |
| ↳ `jk_mo`, untouched | 29.48 | 29.47 | |

The Hessians agree: trace 22.03455454327 against the stock 22.03455454509.

**1.88x, not the 5-6x of §4a, and the difference is the `mo1` branch.**  That
branch keeps GPU4PySCF's `eval_rho4` for the density and batches only what
comes after it (§2), so the part of the function this round did not touch sets
the floor.  Taking the rest would mean a third density kernel; at 2.4 s of an
86 s Hessian that is a 1.03x prize on the whole run, and it goes on the list
rather than into this round.

---

## 5. Scope, and what is left

Covered, now on both spin cases: **LDA, GGA and meta-GGA**, `hermi` 0 and 1,
any number of right-hand sides, one GPU -- and, on the unrestricted side only,
the **`mo1` branch** that an open-shell Hessian's CPHF takes.

Declined, falling through to GPU4PySCF unchanged: the **restricted** `mo1`
branch (§1: no caller found on 1.8.1, and the function a restricted Hessian
does use is a different one that is not launch-bound), **more than one GPU**,
an `fxc` cache that is not the contiguous shape the kernels index, and
`FASTXCFXC_MODE=grouped`, which is an ablation of the catalogue's prescription
and has no unrestricted arm because it lost on the restricted one.

What is left, in the order the profiles above put it:

1. **The two-electron half.**  84-85 % of a stock open-shell TDA and 34 % of
   an unrestricted Hessian.  `fastjk` is already on the TDA half and is worth
   2.1x there (§4b); the Hessian's `jk_mo` is the one that is not moving.
2. **`eval_rho4`, batched** -- the half of the `mo1` branch this round left to
   GPU4PySCF.  1.03x on a whole unrestricted Hessian, by §4c.
3. **`_get_jk_ip1`**, 27 % of a patched unrestricted Hessian and untouched on
   either spin case, which `docs/OPENSHELL_HESSIAN.md` §5 already ranks.

## Files

| file | what |
|---|---|
| `electrolites/fastxcfxc.py` | the module; `_fxc_blocked`'s `nspin`, `_prep`'s interleaving, `_usable_u`, the `mo1` branch and the `nr_uks_fxc` patch are this round |
| `electrolites/kernels/fastxcfxc.cu` | `fxc_wv_u`, the one kernel the unrestricted case needs of its own |
| `tests/test_ufxc.py` | §3, fourteen cases in two modes at both `hermi` values, a UKS TDA spectrum, and the `mo1` branch through a whole UKS Hessian |
| `tests/test_fxc.py` | the restricted suite, unchanged, run as the regression check |
| `benchmarks/fxc_bench.py` | §4a, `--spin` sweeps the unrestricted pair over `nset` |
| `benchmarks/response_anatomy.py` | §4b, `--spin` wires a separate unrestricted stage list |
| `benchmarks/hessian_anatomy.py` | §4c |
| `benchmarks/coverage_probe.py` | §3b, the `nr_uks_fxc` row and `fastxcfxc.stats()` |
