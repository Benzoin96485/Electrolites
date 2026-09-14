# VV10 on an open-shell reference

`docs/OPENSHELL_TMC.md` §4 ranks what an open-shell transition-metal complex
still runs on GPU4PySCF's own code with every module of this repository
loaded.  Three of its eight entries are the **VV10 non-local correlation**
term, in the three places a calculation meets it:

| §4 | term | what was on it |
|---|---|---|
| 3 | the SCF: `uks.get_veff` walks the grid four times, twice for XC and twice for VV10 | `fastxcnlc` patched `dft.rks.RKS.get_veff`; `dft/uks.py` binds its own |
| 4 | the gradient: `grad.rks.get_nlc_exc` at two density matrices | `fastxcgrad` declined `n_dm != 1` |
| 5 | the response: `hessian.uks.nr_uks_fnlc_mo` | `fastfnlc` declined UKS, *and* the name it forwards to was bound at import |

This round takes all three.  It is a small amount of arithmetic and two
patch points that were not where they looked like they were, and the second
half of that sentence is the part worth reading: **two of the three gaps were
entry points bound under another name, not missing kernels.**  The previous
round found the same shape twice (`grad.uks.get_exc`, and `dft/uks.py` calling
`nr_uks` where every module was patching `nr_rks`), which is why
`tests/test_patch_aliases.py` exists; here it is `dft.uks.UKS.get_veff` and
`hessian.uks.nr_rks_fnlc_mo`, and that test's `EXPECTED` list of deliberate
stale aliases is two entries shorter than it was.

---

## 1. Four corrections to §4, before the measurements

§4's ranked list is a to-do list, and three of its entries describe the work
wrongly.  These are corrections to `docs/OPENSHELL_TMC.md`, recorded here
rather than edited into it, so that what that round measured stays what it
says it measured.

**§4 item 4 says "the fix is `get_exc_u`'s, a spin loop around the same
body".  There is no spin loop.**  VV10 is a functional of the total density
alone: `E_nlc[rho_a, rho_b] = E_nlc[rho_a + rho_b]`, so its potential is the
same in both spin channels and the per-atom derivative it contributes
contracts the *total* density matrix.  GPU4PySCF's own `get_nlc_exc` says as
much in longer form — at `nset = 2` it forms `rhoa + rhob` for the double sum
and `dms[0] + dms[1]` for the contraction, and from there the two branches are
one function.  So the fix is `D_a + D_b` and nothing else, and it is *cheaper*
than the restricted case relative to GPU4PySCF rather than equal to it:
summing the density matrices before the grid loop is one GEMM per block where
GPU4PySCF's unrestricted branch calls `eval_rho2` once per spin.

**§4 item 4 also says the term "still falls through".  GPU4PySCF's own
function does not.**  `grad/rks.py:get_nlc_exc` on 1.8.1 carries `assert nset
== 1 or nset == 2` and a full two-density branch; what fell through was
*ours*, through `_prep`'s `if dm.shape[0] != 1: return opt, None`.  The
difference matters for what the fix is worth: this is not a term that had no
implementation, it is a term where the reference implementation was the only
one running, so the ratio in §4 below is the module against a real baseline
rather than against nothing.

**§4 item 5 says a UKS-capable `fastfnlc` "would not be reached without
patching that binding too".  That is right, and it understates how the
binding is reached.**  `hessian/uks.py:28` is
`from gpu4pyscf.hessian.rks import ..., nr_rks_fnlc_mo`, and
`hessian/uks.py:1935`'s `nr_uks_fnlc_mo` calls that module-level name.  So
the patch has to go on `hessian.uks.nr_rks_fnlc_mo`, not on
`hessian.uks.nr_uks_fnlc_mo` — the forwarding function itself is fine and is
left alone, because summing the two perturbed densities is exactly what the
restricted function needs.

**And the entry that is missing from §4 entirely: the unrestricted response
path is not the same *set of callers* as the restricted one.**  On a
restricted reference `nr_rks_fnlc_mo` is reached from CPKS, NMR, C6 and a
Hessian's CPHF, and *not* from TDDFT: `tdscf/rhf.py` sets
`exclude_nlc = True` and passes `with_nlc=not self.exclude_nlc` into
`gen_response`.  On an unrestricted one it *is* reached from TDDFT.
`tdscf/uhf.py:752`, which `tdscf/uks.py`'s `TDA` and `TDDFT` inherit, is

```python
def gen_response(self, mo_coeff=None, mo_occ=None, hermi=0):
    return self._scf.gen_response(mo_coeff=None, mo_occ=None, hermi=hermi)
```

with no `with_nlc` argument and no `exclude_nlc` attribute anywhere in the
class, and `_gen_uhf_response`'s default is `with_nlc=True`.  An open-shell
TDA or TDDFT run with wB97M-V therefore pays the VV10 response term on every
Davidson iteration where the closed-shell one pays none of it.  That makes
UKS TDDFT the largest single consumer of this function in GPU4PySCF, and it
was the one caller nothing in this repository was on.

---

## 2. What was added

### `fastxcgrad`: `get_nlc_exc` at two density matrices, and `grad.uks.get_nlc_exc`

`_prep_nlc` replaces the `_prep` call in `get_nlc_exc`: one or two matrices,
summed to the total, and the body unchanged.  `apply_patch` also assigns
`grad.uks.get_nlc_exc`, which `grad/uks.py:413` binds at import.  Nothing on
1.8.1 calls that alias — an unrestricted gradient reaches the term through
`rks_grad._get_denlc`, whose lookup is a module-global in `grad/rks.py` and
does see the patch — but a stale alias to an unpatched function is the exact
shape of the two misses this round is about, so it is patched and
`tests/test_patch_aliases.py`'s `EXPECTED` list is two entries shorter.

### `fastxcnlc`: `nr_uks_nlc`, and `dft.uks.UKS.get_veff`

The fused pass, with a spin index on the density, the potential and the
contraction, and one thing deliberately without one:

    rho[s]              one grid walk, both spins
    exc, vxc            libxc once, spin=1
    rho4 = rho_a + rho_b
    e, v = vv10nlc(rho4, ...)          the double sum, once, on the total
    wv[0,:4] += wvn ;  wv[1,:4] += wvn
    vmat[s]             one grid walk, both spins

Two of the four grid walks and one of the two potential contractions go away.
Be precise about *which* contraction, because §4b's measurement turns on it:
the one that goes away is the **NLC** one, which is single-density and
GGA-shaped whatever the XC functional is, and the XC one — two spin blocks,
and four operand pairs per block per spin for a meta-GGA — stays.  So what
the fusion removes is two `eval_ao` passes at `deriv = 1` plus one fixed-shape
contraction, and none of it depends on how many components `rho` has.  §4b
measures exactly that: the same absolute saving for a GGA and a meta-GGA on
the same grid.

Adding the same `wvn` to both spin channels is what `dft/uks.py` gets by
adding an `(nao, nao)` `vnlc` to a `(2, nao, nao)` `vxc` and letting it
broadcast, so this changes how often the grid is walked and nothing else.

The patch point is `dft.uks.UKS.get_veff` **and** the module-level
`dft.uks.get_veff` (which `dft/ukspu.py` calls, and which `dft/roks.py`
reaches through the class).  As on the restricted side, no copy of `get_veff`
is carried: the fused build runs first and then GPU4PySCF's own `get_veff`
runs unchanged with `nr_uks` temporarily answering from the result and
`do_nlc` answering False, so the Coulomb build, the exchange build, the
incremental `ecoul` bookkeeping and the returned `tag_array` stay upstream's.

One thing the fusion looks like it should cost and does not: both passes hold
`rho` and then `wv` for the **whole** grid, where `fastxc`'s grouped `nr_uks`
holds one group at a time -- `docs/OPENSHELL_TMC.md` §3 sized that at 1.2 GB
against 60 MB on the 113-atom complex.  But the arm being replaced holds the
same two: GPU4PySCF's `_nr_uks_task` allocates `rho_tot` and `vxc` at
`(2, nset, nvar, ngrids)` each and `nr_nlc_vxc` stacks a whole-grid `rho` of
its own on top of them.  So the peak is stock's peak, and there is
deliberately **no memory guard** on this path.  A first draft of it had one --
25 % of free memory, on the restricted path too -- and taking it out is the
more defensible choice: an allocation that fails raises, `get_veff` catches it
and hands the case back, and a threshold test would only have added a way to
decline *silently* on a case that would have worked.  That is the failure mode
`docs/ROUND2_1.8.1.md` §5 records, and it cost that round its whole
measurement.

What the module did grow is `FASTXCNLC_OFF=1`, which hands every `get_veff`
back.  §4b is why: the fusion competes with `fastxc`'s grouped `nr_rks` /
`nr_uks` rather than adding to it, so the comparison a user needs is
fusion-on against fusion-off *with everything else patched*, in one process,
on their own system.

### `fastfnlc`: the unrestricted reference

`V_munu` in this function is spin-free, so the arithmetic changes in exactly
two places: `_ground_state` sums `eval_rho2` over the two spin channels for
the density the double sum needs, and an MO-basis answer is projected onto
two MO bases instead of one and returned as GPU4PySCF's two-element tuple.
`return_in_mo=False` — which is what the response function and therefore
TDDFT asks for — needs no spin index at all, so the batched contraction of
`docs/README_fastfnlc.md` §3 covers the unrestricted case unchanged.

The patch is `hessian.uks.nr_rks_fnlc_mo` alongside
`hessian.rks.nr_rks_fnlc_mo`; §1 says why that is the name.  The module also
grew a `stats()` counter, for the reason `fastxcnlc` has one: a module that
is allowed to decline returns the right answer either way, and this is the
second time in two rounds that a silent decline hid behind a correct number.

---

## 3. Correctness

Two new suites, both calling GPU4PySCF's implementation and ours on the same
converged UKS density in one process.  Neither patches `fastnlc`: the double
sum has to be the same code on both sides of the comparison, or the suite
would be checking that kernel rather than the grid passes around it.

`tests/test_unlc.py` -- the SCF and gradient halves -- is 8 cases over
`wb97x-v` (GGA + VV10), `wb97m-v` and `b97m-v` (meta-GGA + VV10), at def2-SVP,
def2-TZVP, def2-TZVPP (which reaches g) and 6-31G\*, on O2, CH3, NO, CuCl2,
FeCl4(-) and the 23-atom Pt complex (i.e. with an ECP), 20 to 249 AO:

| column | what it checks | worst of the 8 |
|---|---|---|
| `fused exc` | the fused pass's XC + VV10 energy against `nr_uks` + `nr_nlc_vxc` | **2.9e-16** relative |
| `fused vmat` | both spin blocks of its potential matrix | **5.9e-15** relative |
| `nlc1(grp)` | the VV10 gradient term at two density matrices, grouped | **4.0e-14** relative |
| `nlc1(blk)` | the same, in the per-block ablation | **4.0e-14** relative |
| `whole grad` | the whole UKS gradient, ours against GPU4PySCF's | **1.2e-13** Eh/Bohr |

and, separately, a whole UKS wB97M-V/def2-SVP SCF fused against the two
separate passes, in the same process: **-150.16217160680** both ways,
**1.4e-13 Eh** apart.  `fastxcnlc.stats()` at the end of the suite reads
`fused_u: 157, declined_u: 0` -- 157 fused `get_veff` calls and not one
decline, which is the assertion a correctness test cannot make for itself.

`tests/test_ufnlc.py` -- the response half -- 7 cases, every one entering
through `hessian.uks.nr_uks_fnlc_mo` so that the forwarding *and* the binding
are on the tested path, and each called twice so the second reads the
ground-state cache:

| column | what it checks | worst of the 7 |
|---|---|---|
| `mo` | `return_in_mo=True`, which keeps GPU4PySCF's contraction, returned as its two-element tuple | **bit-identical**, all 7, both calls |
| `ao` | `return_in_mo=False`, whose contraction is batched over the perturbed densities | **4.7e-15** relative |

and a whole **UKS TDA** run on O2/def2-SVP/wB97M-V -- the caller §1 says the
restricted side does not have -- with the three roots agreeing to
**1.4e-13 Eh**: 0.2306440273, 0.2306444539, 0.2383091200 both ways.

The **restricted** paths of the two modules this round touched are unchanged:
`tests/test_fnlc.py` still returns bit-identical results on its
`return_in_mo` path and 2.9e-15 on the other, with the polarizability tensor
at 1.2e-16, and `tests/test_uxcgrad.py` (which exercises `_prep`'s
neighbourhood in `fastxcgrad`) is unchanged over LDA, GGA and meta-GGA.
`_prep_nlc` replaced a `_prep` call on a path both spins take, so this is the
check that the one-density case still goes through it the way it did.

`tests/test_patch_aliases.py` passes with **two fewer** deliberate stale
aliases than before: `grad.uks.get_nlc_exc` and
`hessian.uks.nr_rks_fnlc_mo` are patched, and 18 rather than 17 saved
originals are accounted for (`fastxcnlc` now saves two).

### 3b. And that the three entry points are on the code path

Correctness cannot see a fall-through, so `benchmarks/coverage_probe.py` grew
a row for each new patch point and prints both modules' `stats()`.  A **UKS
wB97M-V/def2-SVP** SCF and gradient on CuCl2, everything patched, against the
same run with `FASTXCNLC_OFF=1 FASTXCGRAD_OFF=1` — which is not literally the
state before this round (before it, `dft.uks.UKS.get_veff` was not patched at
all and the row would read `(-)`) but is the same *work distribution*:

| entry point | both switches off | patched |
|---|---|---|
| **`fastxcnlc uks_veff`** | 22 calls, `fused_u: 0, declined_u: 22` | 22 calls, **`fused_u: 22, declined_u: 0`** |
| `fastxc nr_uks` | 22 / 22 | **(-)** |
| **`fastxcgrad get_nlc`** | 1 / **0** ours / 1 stock | 1 / **1** / 0 |
| `fastxcgrad uks_exc` | 1 / 0 / 1 | 1 / 1 / 0 |
| `fastnlc _vv10nlc` | 23 / 23 | 23 / 23 |
| `fastk get_k` | 22 / 22 | 22 / 22 |
| `fastxcnlc get_veff` (the restricted one) | (-) | (-) |
| `fastfnlc uks_fnlc` | (-) | (-) |

and the same functional on closed-shell glycine reads
`fastxcnlc get_veff 12 / 12` with `fused: 12, fused_u: 0` and
`fastxcnlc uks_veff (-)`, i.e. neither path is answering for the other.

Three rows need reading rather than counting.

**`fastxcnlc uks_veff` has no ours/stock split**, because a decline there is
not a call to a saved original -- the wrapper hands the whole `get_veff` back.
That is why the module has `stats()` and why the table quotes it: the row
counts 22 either way and only the counter says which.

**`fastxc nr_uks` going to `(-)` is the fusion, not a regression.**
`get_veff_u` computes the XC and NLC potentials first and then lets
GPU4PySCF's `get_veff` run with `ni.nr_uks` temporarily answering from that
result, so `NumInt.nr_uks` -- which is what the probe wraps -- is genuinely
not called; the work moved to the row above it.  Anyone reading this table
without that in mind would conclude the previous round's work had gone inert,
which is the same misreading the probe exists to prevent.

**`fastfnlc uks_fnlc` reads `(-)` because an SCF and a gradient do not reach
the response path.**  `tests/test_ufnlc.py` is where that row is exercised.

One thing that turned up and is not this round's: the `fastejk` class-block
counter reads **3 298 / 1 649 / 1 649** on this run -- half the blocks to
GPU4PySCF -- where `docs/OPENSHELL_TMC.md` §1 recorded 1 649 / 1 649 for the
same molecule.  The difference is the functional: wB97M-V is
range-separated, so `grad/uks.py:energy_ee` makes **two**
`jk_energy_per_atom` calls and the second is the **short-range** (`erfc`,
`omega < 0`) operator, which `fastejk._usable` declines by design -- "short
range needs twice the roots and the `s_estimator` screening", as its own
comment says.  That is a documented limit rather than a bug, and no coverage
table here had used a range-separated functional before.  But the
closed-shell glycine run of the *same* functional does not show it
(532 / 532 / 0), which the short-range explanation alone does not account
for, so this is recorded as an observation with a likely cause and not as a
finding.

---

## 4. What it is worth

All of it on **one A100-SXM4-80GB** with one GPU visible to the process, on a
batch node with nothing else on it.  That is not the A100-SXM4-40GB or
A100-PCIE-40GB the tables in `README.md` and `docs/README_fastfnlc.md` were
taken on, so the absolute seconds here are not comparable with theirs; the
ratios are, and the one row that appears in both is called out below.

### 4a. The gradient's VV10 term, at two density matrices

`benchmarks/unlc_bench.py`, converged UKS density, both implementations in one
process, min of 3.  The **organic radical cation** rows are there because
every other open-shell case in this repository is a d-block complex and the
question "does this depend on there being a metal in the molecule" has to be
answered rather than assumed:

| system | functional | GPU4PySCF | `fastxcgrad` | | of which the double sum | **the grid passes** |
|---|---|---|---|---|---|---|
| CuCl2, 3 atoms, 67 AO, doublet | wB97M-V | 0.0566 s | **0.0334 s** | 1.70x | 0.0229 s | **3.21x** |
| Ni dithiolene, 21 atoms, 327 AO, doublet | wB97M-V | 0.4572 s | **0.3335 s** | 1.37x | 0.2847 s | **3.53x** |
| Ni dithiolene | wB97X-V | 0.4526 s | **0.3347 s** | 1.35x | 0.2829 s | **3.28x** |
| Cu macrocycle, 60 atoms, 515 AO, quartet | wB97M-V | 1.5910 s | **1.3624 s** | 1.17x | 1.2282 s | **2.70x** |
| alanine dipeptide cation, 22 atoms, 200 AO, doublet | wB97M-V | 0.3662 s | **0.2681 s** | 1.37x | 0.2300 s | **3.57x** |
| alanine dipeptide cation | wB97X-V | 0.3637 s | **0.2755 s** | 1.32x | 0.2266 s | **2.80x** |
| penicillin V cation, 42 atoms, 430 AO, doublet | wB97M-V | 0.9072 s | **0.7208 s** | 1.26x | 0.6314 s | **3.09x** |
| penicillin V cation, 6-31G\*, 376 AO | wB97M-V | 0.8852 s | **0.7064 s** | 1.25x | 0.6208 s | **3.09x** |

The last column is the one to read.  `get_nlc_exc` is a double sum with two
grid passes around it, this module replaces only the passes, and the double
sum -- timed separately in the same process, on the same grid and the same
density, and identical in both implementations -- is 40-77 % of the term.
Subtract it and what the change is actually worth is **2.7x-3.6x**, and that
number is stable over a 20x range of system size, two chemical classes, three
basis sets and both functional types.  The whole-term column is the same
result divided by however much double sum happens to sit on top of it, which
is why it looks like it falls with size: it does not, the double sum's share
rises.

Two things not to over-read.  The wB97M-V and wB97X-V rows of one molecule are
two different converged densities and therefore two different sets of points
above the 1e-10 cut, so they are two systems rather than one system twice.
And an independent earlier job measured the Ni dithiolene / wB97M-V baseline
at **0.5966 s** rather than 0.4572 s -- a 30 % outlier, against agreement to
1 % on the other three cases the two jobs shared, which would have put that
row at 1.19x instead of 1.37x.  So the case was re-run four more times in a
separate job, twice at three repetitions and twice at seven:

| | stock | ours | |
|---|---|---|---|
| reps 3 | 0.4525 s | 0.3358 s | 1.35x |
| reps 3 | 0.4460 s | 0.3391 s | 1.32x |
| reps 7 | 0.4508 s | 0.3333 s | 1.35x |
| reps 7 | 0.4513 s | 0.3343 s | 1.35x |

-- so the baseline is 0.446-0.453 s and the ratio 1.32x-1.35x, and the
outlier was the baseline rather than the ratio.  A ratio is only as good as
its baseline, and one run is not a baseline; `docs/OPENSHELL_TMC.md` §5 is
the round that learned that the expensive way.

### 4b. The SCF's fused XC-and-VV10 grid pass

Three arms on the same converged density in one process, and **all three with
`fastnlc`'s double sum underneath**, so that what varies between them is the
grid passes and not the kernel none of the three touches.  The double sum is
again timed separately and subtracted; the table below is the grid passes
alone, and the whole-term ratios are at the end of the section.

| system | functional | grid points | stock | `fastxc` grouped + separate NLC | **fused** |
|---|---|---|---|---|---|
| CuCl2, 3 atoms, 67 AO | wB97M-V | 58 624 | 0.0504 s | 0.0314 s (1.61x) | 0.0356 s (**1.42x**) |
| alanine dipeptide cation, 22 atoms, 200 AO | wB97M-V | 258 560 | 0.1921 s | 0.1238 s (1.55x) | 0.1394 s (**1.38x**) |
| alanine dipeptide cation | wB97X-V | 258 560 | 0.0983 s | 0.0776 s (1.27x) | 0.0541 s (**1.82x**) |
| Ni dithiolene, 21 atoms, 327 AO | wB97M-V | 322 560 | 0.2409 s | 0.1624 s (1.48x) | 0.1778 s (**1.35x**) |
| Ni dithiolene | wB97X-V | 322 560 | 0.1295 s | 0.0972 s (1.33x) | 0.0676 s (**1.92x**) |
| penicillin V cation, 42 atoms, 430 AO | wB97M-V | 519 680 | 0.3769 s | 0.2771 s (1.36x) | 0.2890 s (**1.30x**) |
| penicillin V cation, 6-31G\*, 376 AO | wB97M-V | 519 680 | 0.3663 s | 0.2623 s (1.40x) | 0.2776 s (**1.32x**) |
| Cu macrocycle, 60 atoms, 515 AO | wB97M-V | 689 152 | 0.4871 s | 0.4074 s (1.20x) | 0.3782 s (**1.29x**) |

**The middle column is why this is reported as a comparison and not as a
speedup.**  The fusion and `fastxc`'s grouped `nr_uks` remove cost from the
same two grid passes from opposite directions -- the fusion by walking the
grid twice instead of four times, `fastxc` by batching the GEMMs of every
block and both spins into one launch -- and the fused pass has to *give up*
the grouping to get the fusion, because VV10 needs the whole grid's density
before any potential can be formed and a grouped pass holds one group at a
time.  So they do not compose, and on six of these eight cases the grouping is
ahead.

What makes this readable rather than a coin toss is the **absolute** saving
each one makes, which is the same table with the subtraction done the other
way:

| grid points | 58 624 | 258 560 | 258 560 | 322 560 | 322 560 | 519 680 | 519 680 | 689 152 |
|---|---|---|---|---|---|---|---|---|
| functional | mGGA | mGGA | GGA | mGGA | GGA | mGGA | mGGA | mGGA |
| `fastxc`'s grouping saves | 0.019 s | 0.068 s | **0.021 s** | 0.079 s | **0.032 s** | 0.100 s | 0.104 s | 0.080 s |
| the fusion saves | 0.015 s | 0.053 s | **0.044 s** | 0.063 s | **0.062 s** | 0.088 s | 0.089 s | 0.109 s |
| the fusion, per grid point | 2.5e-7 | 2.0e-7 | 1.7e-7 | 2.0e-7 | 1.9e-7 | 1.7e-7 | 1.7e-7 | 1.6e-7 |

* **The fusion's saving is linear in the grid and independent of the
  functional.**  1.6e-7 to 2.5e-7 s per grid point over a 12x range of grid
  size, two chemical classes and both functional types -- and 0.063 s against
  0.062 s for a meta-GGA and a GGA on the *same* grid.  That is exactly what it
  removes: two `eval_ao` passes at `deriv = 1` and one NLC potential
  contraction, which is single-density and GGA-shaped whatever the XC
  functional is.  None of it knows how many components `rho` has, and the
  measurement says so.
* **The grouping's saving depends on the functional and does not scale.**
  0.079 s for the meta-GGA against 0.032 s for the GGA on the same grid (and
  0.068 against 0.021 on the other molecule's), and 0.100 → 0.080 s going from
  519 680 to 689 152 points.  That is also exactly what it removes -- launch
  latency, and `_vmat_pairs` gives meta-GGA four operand pairs per block per
  spin where GGA gets one, so meta-GGA is where there is four times as much to
  batch; and as `nao_sub` grows the per-block GEMMs become flop-bound and
  batching them buys less per block.

So the two are not a wash that happens to land near each other; they are a
**work** saving and a **latency** saving, and which is bigger is predictable:
the fusion wins on a GGA+VV10 functional at any size measured here (1.82x and
1.92x against 1.27x and 1.33x), and on a meta-GGA it wins once the grid is
large enough -- at 689 152 points it is ahead by 8 %, and that is the same
direction `docs/ROUND2_1.8.1.md` §5's restricted measurement points at 284
atoms.  Nothing about the metal enters: the organic-cation rows and the
complex rows at comparable grid sizes agree to a few per cent on both lines
of the table.

The fusion is therefore **on** by default -- the cases where it loses are the
cases where the term is small, and `FASTXCNLC_OFF=1` re-runs this comparison
in one process for a system that is not one of these eight.  Nothing here is
tuned on a size or a functional threshold; eight cases are enough to say what
the two changes *are*, and not enough to pick a switch-over point.

What a *grouped and fused* pass would be worth is the obvious question and the
table answers it: the two savings are on different things, so it would get
both -- 0.079 + 0.062 = 0.141 s of the Ni dithiolene's 0.241 s of grid passes,
about 1.7x where the better of the two columns is 1.48x.  It is not in this
round because it needs `fastxc`'s grouped machinery split into a density pass
and a potential pass with a whole-grid buffer between them, and that is a
refactor of the module every earlier measurement in this repository rests on.

Finally the number a caller sees, which is against a fully unpatched
GPU4PySCF -- its double sum plus its grid passes, against `fastnlc` plus the
fusion:

| system | functional | GPU4PySCF | + `fastnlc` + the fusion | |
|---|---|---|---|---|
| CuCl2 | wB97M-V | 0.0775 s | **0.0585 s** | **1.32x** |
| alanine dipeptide cation | wB97M-V | 0.4884 s | **0.3694 s** | **1.32x** |
| alanine dipeptide cation | wB97X-V | 0.3940 s | **0.2807 s** | **1.40x** |
| Ni dithiolene | wB97M-V | 0.6942 s | **0.4625 s** | **1.50x** |
| Ni dithiolene | wB97X-V | 0.5819 s | **0.3505 s** | **1.66x** |
| penicillin V cation | wB97M-V | 1.4551 s | **0.9203 s** | **1.58x** |
| Cu macrocycle | wB97M-V | 2.2985 s | **1.6064 s** | **1.43x** |

Most of that is `fastnlc`, which is not this round -- and `fastnlc` already
covered the unrestricted case before it, because `nr_nlc_vxc` sums the two
densities before the double sum.  The table is what the term is worth
end to end, not a claim about either change alone.

### 4c. The response term

`benchmarks/fnlc_bench.py --spin`, entering through
`hessian.uks.nr_uks_fnlc_mo`, min of 3, `cached` being the second and later
calls of a solve (the ones a CPKS iteration or a Davidson step makes):

| system | reference | `nset` | GPU4PySCF | `fastfnlc` cached | |
|---|---|---|---|---|---|
| CuCl2, 3 atoms, 67 AO | UKS doublet | 1 | 0.1539 s | **0.0955 s** | **1.61x** |
| | | 3 | 0.1771 s | **0.1088 s** | **1.63x** |
| Ni dithiolene, 21 atoms, 327 AO | UKS doublet | 1 | 2.4274 s | **1.6475 s** | **1.47x** |
| | | 3 | 2.6434 s | **1.7765 s** | **1.49x** |
| penicillin V, 42 atoms, 430 AO | RKS | 1 | 5.6587 s | **3.9564 s** | **1.43x** |
| | | 3 | 6.0844 s | **4.2361 s** | **1.44x** |

The last two rows are the regression check, and they are the one place this
table meets an earlier one: `docs/README_fastfnlc.md` §6 has 1.44x-1.49x for
the same molecule, basis, functional and grid on 40 GB hardware.  Nothing this
round did to that path changed it.

The unrestricted rows are **1.47x-1.63x on a path that was 1.00x by
construction** before this round, and the first call of each solve gets
1.01x-1.15x because it fills the ground-state cache instead of reading it —
the same split §2 of that document describes.  On a UKS TDA run with wB97M-V,
where §1 says this term is on every Davidson iteration, essentially all of the
iterations are cached calls.

---

### 4d. `fastnlc`'s grid trim on the response term, and why it is off

Both `O(n_grid^2)` sums in the response term carry a factor `rho_j w_j`, so
`fastnlc`'s error-budgeted trim -- drop the smallest-`|rho w|` points until
their total reaches a budget, and state the approximation as the error it
makes rather than as a density -- applies to it verbatim, and the saving is
quadratic in the point count because the sums are.  `FASTFNLC_BUDGET` does
it; one SCF serves every budget, `nset = 3`, cached call:

| | budget | points kept | dropped | cached | vs GPU4PySCF | rel. err |
|---|---|---|---|---|---|---|
| **Ni dithiolene**, 322 560 grid points | 0 (default) | 285 802 | 11.4 % | 1.7763 s | 1.49x | 2.4e-15 |
| | 1e-14 | 283 268 | 12.2 % | 1.7456 s | 1.51x | 2.3e-15 |
| | 1e-12 | 280 715 | 13.0 % | 1.7558 s | 1.50x | 1.3e-13 |
| | 1e-10 | 276 090 | 14.4 % | **1.6442 s** | **1.61x** | 1.3e-11 |
| **penicillin V**, 519 680 | 0 (default) | 446 839 | 14.0 % | 4.2254 s | 1.44x | 2.3e-15 |
| | 1e-14 | 442 708 | 14.8 % | 4.1497 s | 1.46x | 3.9e-15 |
| | 1e-12 | 438 312 | 15.7 % | **3.9796 s** | **1.53x** | 4.0e-13 |
| | 1e-10 | 429 386 | 17.4 % | 3.9465 s | 1.54x | 2.7e-11 |

**8 % and 6 %**, which is real and is a tenth of what the same criterion is
worth in `fastnlc`.  The `budget = 0` row says why, and it is the useful part
of this measurement: **11-14 % of the grid is already gone before any budget
is applied**, because this function's own filter is
`rho >= 1e-10 and |w| > 1e-14` and it is applied to every point.
`docs/README_fastnlc.md` §3 measured 14 % of a level-3 grid carrying 1e-14 of
the density-weight *against an SCF path where GPU4PySCF's own cut removes
0.8 % of the points* -- so there the budget had 14 % to take and here it has
2-3 %.  The quadratic model then predicts what is measured: keeping 276 090
of 285 802 points should cost `(285802/276090)^2 = 1.071` less, and 1.080 is
what came out.

So it is **off by default**, for three reasons in order of weight: what it
perturbs is a response operator rather than an energy; this module's
correctness guarantee on the `return_in_mo` path is currently *bit-identity*
against GPU4PySCF, which a trim ends; and 6-8 % is not what that guarantee is
worth trading for on two systems' evidence.  The switch and this table are
here so that someone whose wB97M-V TDDFT run is dominated by this term can
make that trade knowingly.  For the record the errors are the ones the budget
promises -- 1.3e-13 relative at 1e-12, 1.3e-11 at 1e-10 -- and both are far
below the 1e-9 a Davidson or CPKS solve converges to.

## 5. What is still not covered, and what this round learned about it

`docs/OPENSHELL_TMC.md` §4 items 1, 2, 6, 7 and 8 are untouched: the
unrestricted XC response kernel (`nr_uks_fxc`), the unrestricted Hessian's XC
terms, `MAX_PRIM_PAIR` on a def2 metal basis, g shells in `fastj` and
`fastejk`, and first-run JIT.  Item 1 is the one that now pairs with this
round -- an open-shell wB97M-V TDDFT run has its VV10 response term covered
here and its XC response term still stock -- and it is the next thing to take.

Three VV10 things this round found and did not do, in the order they are
worth:

1. **`VXC_vv10nlc_hess_eval_f_t`**, still 65 % of a response call.
   `fastfnlc`'s module docstring says writing it "needs GPU4PySCF's CUDA
   source for that kernel, which a wheel does not ship" -- and
   `docs/README_fastfnlc.md` §4 puts it more carefully, that what is needed is
   either the source or a re-derivation of equations (20)-(24) of the paper the
   function's docstring cites, with the `.so` as the oracle either way.  §4 is
   right and the docstring is the loose version: **the shipped kernel is the
   oracle, so the missing thing is algebra, not CUDA.**  `fastnlc`'s three
   measured techniques -- a MUFU-seeded reciprocal, single precision beyond
   10 Bohr, an error-budgeted grid trim -- apply to it verbatim, and §4d here
   is the cheap partial version of the third one (worth 6-8 %, against the
   3.92x the full one is worth on the SCF's double sum, and §4d says why the
   gap is that large).
2. **A grouped *and* fused grid pass**.  §4b's two savings are on different
   things -- work and latency -- so it would get both: about 1.7x on the Ni
   dithiolene's grid passes where the better of the two columns is 1.48x.  It
   needs `fastxc`'s grouped machinery split into a density pass and a
   potential pass with a whole-grid buffer between them, which is a refactor
   of the module every earlier measurement in this repository rests on.
3. **The VV10 Hessian** (`hessian.rks._get_enlc_deriv2`, `_get_vnlc_deriv1`),
   which no module here is on for either spin.

And one correction this round made to a *different* module's document, which
came out of §4a and §4b needing the double sum timed on its own.
`docs/README_fastnlc.md` recorded 2.68x-3.92x on that sum and closed with
"nothing here is tuned to a problem size", on the evidence of two clusters
whose grids are millions of points.  Timed at five much smaller grids it is
**1.18x at 59 000 points, 1.29x at 259 000, 1.59x at 323 000 and 1.71x at
520 000** -- because the pair sum is quadratic in the point count and that
module's per-call overhead (a Morton sort, a bounding box per tile, the trim's
own sort) is not.  The claim about tuning is still right; the claim a reader
would take from the ratio is not, and that document now carries the small-grid
table.  It also explains something in §4b: switching `fastnlc` off changed the
XC-and-VV10 term by nothing measurable on the two smallest systems, which
looked like the module being inert and is 1.18x on 40 % of a term.

## Files

| file | what |
|---|---|
| `electrolites/fastxcnlc.py` | `nr_uks_nlc`, `get_veff_u`, the `fused_u`/`declined_u` counters |
| `electrolites/fastxcgrad.py` | `_prep_nlc`, and `grad.uks.get_nlc_exc` in `apply_patch` |
| `electrolites/fastfnlc.py` | the unrestricted ground-state density and MO projection, the `hessian.uks` binding, `stats()`, and `FASTFNLC_BUDGET` (off) |
| `tests/test_unlc.py` | the unrestricted fused SCF pass and VV10 gradient term |
| `tests/test_ufnlc.py` | the unrestricted VV10 response term, and a UKS TDA run |
| `benchmarks/unlc_bench.py` | the three-arm A/B of §4, all in one process on one density |
| `benchmarks/fnlc_bench.py` | `--spin` selects the unrestricted reference |
| `benchmarks/scf_anatomy.py` | `..nr_nlc_vxc` and `..fused xc+nlc` rows, so the fusion is attributable |
| `benchmarks/coverage_probe.py` | `fastxcnlc uks_veff` and `fastfnlc uks_fnlc` rows, and the modules' `stats()` |
