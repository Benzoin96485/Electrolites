# fastxcfxc — the XC response kernel, and the entry the catalogue gets backwards

Catalogue entry **B4d**, rank 12 in `instructions/06`'s §5 list, paired there
with **D8**.  It names `dft/numint.py`'s `nr_rks_fxc`, which is the single
entry point for the exchange–correlation part of every response property
GPU4PySCF computes:

| caller | reaches it through | `nset` |
|---|---|---|
| polarizability, CPKS (`properties/polarizability.py`) | `gen_response` → `nr_rks_fxc` | 3, falling to 1 as directions converge |
| TDA / TDDFT (`tdscf/rhf.py:317 vind`) | `gen_response(singlet=…)` → `nr_rks_fxc_st` → `nr_rks_fxc` | `nstates × space_inc` |
| NMR shielding, IR, Raman, C6 (`properties/*`) | the same two | 3 to 9 |

This module replaces it for LDA, GGA and meta-GGA.

---

## 0. Summary

* Three changes, none of which touches the grid, the AO screening, `fxc`
  itself or any symmetry factor: **one GEMM per grid block instead of one per
  right-hand side**, twice over; **three custom kernels** (`kernels/fastxcfxc.cu`)
  in place of ten GPU4PySCF launches per right-hand side; and **one
  symmetrisation of a non-Hermitian right-hand side per call** instead of an
  extra matrix product per grid block.
* At the `nset` a response solve actually uses, at fixed density: a
  polarizability's `nset = 3, hermi = 1` is **2.93x** at 42 atoms and
  **1.43x** at 113; a five-root TDA's `nset = 5, hermi = 0` is **4.65x** and
  **2.21x**.  End to end that is **1.14x** on the whole polarizability and
  **1.28x** on the whole TDA run at 42 atoms, taking `nr_rks_fxc` from 25.2 %
  of the first to 13.4 % and from 28.4 % of the second to 9.4 %; at 113 atoms
  the whole-run figures fall to 1.06x and 1.07x, for the reason in §6.
* **The catalogue's prescription for this entry is measured to be the slowest
  of the three arrangements tried.**  B4d and G1 both ask for P8 — GPU4PySCF's
  `_grouped_block_loop` plus `grouped_gemm`, which is what carried
  `fastxcgrad` to 3.4–4.5x on the XC gradient.  Here it is *worse than leaving
  the issue order alone*: 0.72 s against 0.22 s at `nset = 20`.  §2 says why,
  and it is a general point about when P8 applies.
* **D8, the other half of rank 12, is a 1.7 %–2.2 % item on 1.8.1, not the
  16.3 % the catalogue measured on 1.7.0.**  The call count it rests on is
  exactly right and reproduces; the cost per call is not.  §5.
* The two sizes disagree by a factor of two on how much this is worth, in a
  direction the stage split explains: what is left after this change is the
  two GEMMs, 63 % of the function at 42 atoms and 70 % at 113.  §6.

---

## 1. What GPU4PySCF does per grid block

`_nr_rks_fxc_task` (`dft/numint.py:1237`) walks `ni.block_loop` and, for each
block, runs two Python loops over the `nset` right-hand sides.  For GGA, with
`nao_sub` the block's screened AO count and `ng` its grid points:

| step | per set | launches |
|---|---|---|
| `take_last2d(dms[i], mask)` | gather the masked density | 1 |
| `eval_rho` | `c0 = D . ao_0`, one GEMM `nao_sub² × ng` | 1 GEMM |
| ↳ | four `GDFTcontract_rho`, then `rho[1:4] *= 2` | 5 |
| `_contract_rho1_fxc` | batched over sets already | (2 total) |
| `wv[0] *= .5` | | 1 |
| `_scale_ao` | `aow = Σ_n wv_n ao_n` | 1 |
| `ao_0 . aow^T` | one GEMM `nao_sub² × ng` | 1 GEMM |
| `add_sparse` | scatter back to `nao` | 1 |

That is about `11·nset` launches and `2·nset` GEMMs per block, and every one of
those kernels re-reads all of `ao`.  `_contract_rho1_fxc` deserves its own
line: it evaluates `(rho1[:,None] * fxc).sum(axis=2)`, which materialises an
`(nset, 4, 4, ng)` temporary — four times the traffic of the result it returns.

## 2. Three arrangements, and why the catalogue's loses

All three are in the module and selected by `FASTXCFXC_MODE`, because the
comparison is the result.  B3LYP/def2-SVP, level-3 grids, penicillin V (42
atoms, `nao = 430`), one A100-PCIE-40GB, `min` of three, both implementations
in one process (`benchmarks/fxc_bench.py`).

`hermi = 1` is the CPKS case; `hermi = 0` is TDDFT's.

| `nset` | stock | `blocked` | `nokernel` | `grouped` | blocked | nokernel | grouped |
|---|---|---|---|---|---|---|---|
| 1 | 0.0813 | **0.0449** | 0.0868 | 0.0755 | **1.81x** | 0.94x | 1.08x |
| 2 | 0.1288 | **0.0512** | 0.0933 | 0.0959 | **2.52x** | 1.38x | 1.34x |
| 3 | 0.1742 | **0.0595** | 0.1024 | 0.1235 | **2.93x** | 1.70x | 1.41x |
| 5 | 0.2693 | **0.0805** | 0.1206 | 0.1713 | **3.35x** | 2.23x | 1.57x |
| 10 | 0.4897 | **0.1284** | 0.1884 | 0.3016 | **3.81x** | 2.60x | 1.62x |
| 20 | 0.9546 | **0.2226** | 0.3512 | 0.7183 | **4.29x** | 2.72x | 1.33x |
| 40 | 1.8738 | **0.4185** | 0.6864 | 1.2803 | **4.48x** | 2.73x | 1.46x |

and at `hermi = 0`:

| `nset` | stock | `blocked` | `nokernel` | `grouped` | blocked | nokernel | grouped |
|---|---|---|---|---|---|---|---|
| 1 | 0.1071 | **0.0454** | 0.0884 | 0.0770 | **2.36x** | 1.21x | 1.39x |
| 2 | 0.1680 | **0.0514** | 0.0908 | 0.0908 | **3.27x** | 1.85x | 1.85x |
| 3 | 0.2386 | **0.0601** | 0.0989 | 0.1193 | **3.97x** | 2.41x | 2.00x |
| 5 | 0.3776 | **0.0812** | 0.1249 | 0.1698 | **4.65x** | 3.02x | 2.22x |
| 10 | 0.6782 | **0.1295** | 0.1886 | 0.3034 | **5.24x** | 3.60x | 2.24x |
| 20 | 1.3091 | **0.2238** | 0.3531 | 0.7476 | **5.85x** | 3.71x | 1.75x |
| 40 | 2.6459 | **0.4213** | 0.6946 | 1.2156 | **6.28x** | 3.81x | 2.18x |

* **`grouped`** is catalogue B4d/G1 as written: `_grouped_block_loop` plus
  `grouped_gemm` over every `(block, set)` pair, with GPU4PySCF's own
  contraction kernels inside.  It is 1.1x–2.2x, and at `nset = 20` it is
  **three times slower than `blocked`**.
* **`nokernel`** keeps `blocked`'s two batched GEMMs and puts GPU4PySCF's
  per-set contraction kernels back.  It is the ablation for
  `kernels/fastxcfxc.cu`.
* **`blocked`** is what ships.

**Meta-GGA**, same molecule and sweep, `blocked` against stock with the
`nokernel` ablation (`grouped` is LDA/GGA only).  M06-2X/def2-SVP:

| `nset` | hermi | stock | `blocked` | `nokernel` | blocked | nokernel |
|---|---|---|---|---|---|---|
| 1 | 1 | 0.1900 | **0.1063** | 0.1883 | **1.79x** | 1.01x |
| 3 | 1 | 0.5040 | **0.1855** | 0.2598 | **2.72x** | 1.94x |
| 5 | 1 | 0.8159 | **0.2740** | 0.3423 | **2.98x** | 2.38x |
| 10 | 1 | 1.5365 | **0.4733** | 0.6011 | **3.25x** | 2.56x |
| 20 | 1 | 3.0652 | **0.8571** | 1.1539 | **3.58x** | 2.66x |
| 3 | 0 | 0.5317 | **0.1868** | 0.2520 | **2.85x** | 2.11x |
| 5 | 0 | 0.8636 | **0.2769** | 0.3300 | **3.12x** | 2.62x |
| 20 | 0 | 3.2618 | **0.8597** | 1.1649 | **3.79x** | 2.80x |

The meta-GGA rows are the ones where the kernels earn the most: 1.3x-1.8x on
top of the batched GEMMs, against 1.2x-1.6x for GGA, because meta-GGA's
`eval_rho` and `_tau_dot` between them issue about 28 launches per (block,
density) where GGA issues 11.

**Why P8 does not transfer here.** In the XC *gradient* — B2, where
`fastxcgrad` measured P8 as its largest single effect — one grid block issues
a handful of GEMMs and then waits, so there is a launch bubble per block to
fill and grouping fills it.  On the response path there are already `nset`
independent GEMMs back to back per block with nothing synchronising between
them, so much less of the latency is exposed.  What `grouped_gemm` adds is a
host-side pointer table built in a Python loop for every `(block, set)` pair —
`As_ptr.append(a.data.ptr)`, then `np.array(...)`, three times, per chunk.  At
`nset = 20` and ~30 blocks in a chunk that is 1800 Python-level attribute
reads per call site, and it costs more than the batching saves.

The batching that *does* pay is along `nset`, and the reason is that `nset` is
a **stride in memory, not a list of pointers**: the `nset` masked density
matrices `take_last2d` returns are one contiguous `(nset, nao_sub, nao_sub)`
array, so `D . ao_0` for all of them is a single
`(nset·nao_sub, nao_sub) × (nao_sub, ng)` GEMM, and the `nset` weighted AO
blocks are a single `(nset·nao_sub, ng) × (ng, nao_sub)` on the way back.  No
pointer table, one launch.

*(One latent bug avoided rather than fixed: GPU4PySCF's own loop tracks the
grid offset it slices `fxc` by as a running sum over the blocks
`ni.block_loop` yields, and that loop **drops** a block on which no basis
function survives the screen (`dft/numint.py:2059`) without the caller
noticing — so the offset and the block would silently disagree from there on.
It does not happen on any system in `tests/` and cannot be provoked at these
grid levels, but this module passes `strict_grid_order=True` and skips the
empty blocks itself, which makes the offset the block's real one by
construction.)*

*(One transposition to be aware of: `aow_i . ao_0^T` is the transpose of
GPU4PySCF's `ao_0 . aow_i^T`.  For GGA the `transpose_sum` that
`nr_rks_fxc` already applies at the end absorbs it; for LDA the block matrix is
symmetric, because `aow = ao_0 · wv` there.  `tests/test_fxc.py` covers both.)*

## 3. The three kernels

`kernels/fastxcfxc.cu`, ~150 lines, built by NVRTC through CuPy like the rest
of the repository.  All three put the grid index on `threadIdx.x`, so every
access is coalesced, and the right-hand-side index on `threadIdx.y`, so the
`ao` values a block reads are shared by its `threadIdx.y` rows and come from
L1 rather than from memory.  Nothing is reduced across threads, so there is no
shared memory and no `__syncthreads`.

| kernel | what it computes | what it replaces |
|---|---|---|
| `fxc_rho1` | `rho1[i][n][g] = Σ_u c0[i][u][g] · ao[n][u][g]`, the `n > 0` components doubled | `4·nset` `GDFTcontract_rho` plus one `rho1[:,1:4] *= 2` |
| `fxc_rho1_mgga` | the same, plus `rho1[i][4] = ½ Σ_{n=1..3} Σ_u c0[n][i][u][g] ao[n][u][g]` | `7·nset` launches of `eval_rho`'s meta-GGA branch |
| `fxc_wv` | `wv[i][y][g] = w[g] · Σ_x rho1[i][x][g] · fxc[y][x][g]`, with `wv[0] *= .5` folded in as `scale0` | `_contract_rho1_fxc` and its `(nset,4,4,ng)` temporary, the `fxc·w` product, and the `wv[0]` scaling |
| `fxc_scale` | `aow[i][v][g] = Σ_{n<NW} wv[i][w0+n][g] · ao[a0+n][v][g]` | `nset` `GDFTscale_ao`, and for meta-GGA the three `_scale_ao` calls inside `_tau_dot` as well |

`fxc_scale` writes one output per AO, so it has no reduction and can tile the
AO index over `blockIdx.z` (`FASTXCFXC_UPB`, default 32).  That matters: at
`nset = 3` and one 4096-point block, the `(g, set)` grid alone is 128 blocks
of 128 threads on a 108-SM card, and the AO tiling is what brings it to a
thousand.  `fxc_rho1` sums over the AO index and so keeps it in the thread.

Per block the launch count is now `take_last2d`, GEMM, `fxc_rho1`, `fxc_wv`,
`fxc_scale`, GEMM, `add_sparse` — **seven, independent of `nset`**, against
GPU4PySCF's `11·nset + 3`.

## 4. Correctness

`tests/test_fxc.py`, both implementations in one process on the same converged
SCF and the same `fxc` cache, **eighteen cases**: four basis sets to cc-pVQZ,
ten functionals — two LDA (`svwn`, `lda,vwn`), five GGAs and hybrids including
two range-separated ones (`wb97x-d3bj`, `wb97m-v`), three meta-GGAs (`m06-2x`
twice, `tpss` twice) — a third-row element, a cation, `nset` from 1 to 8, and
both `hermi` values.  Worst relative deviation on the potential matrix over
the suite is **4.0e-15**, and most cases are at 2e-16 to 1e-15.

The end of each case is the quantity a user asks for, computed **on the same
SCF** with the module patched and unpatched: the polarizability tensor
(worst 1.1e-15 relative) and a three-root TDA spectrum (worst 6.0e-14 eV).

Sharing the SCF is the point, and it cost a wrong conclusion to learn.  An
earlier version of this suite converged a second SCF for the reference half.
Seventeen cases were unaffected; one — alanine dipeptide with pure LDA, the
slowest SCF here — came out at **2.1e-7** on the tensor and 2.7e-6 eV on the
spectrum, with its `hermi` columns at 2e-15 like everything else.  The
explanation is not that a 1e-15 kernel difference tips the CPKS convergence
test (which is what it looked like): it is that two independent SCFs land on
slightly different densities and the response solve amplifies that to its own
tolerance.  With one SCF the same case is 5.0e-16.  **A property-level check
of a response kernel has to hold the SCF fixed, or it measures the SCF.**

`hermi = 0` is checked directly rather than left to the algebra, because it is
the one place where the two implementations do genuinely different arithmetic.
GPU4PySCF's `eval_rho` at `hermi = 0` forms a second product `D^T · ao_0` and
three more contractions per grid block.  This module instead symmetrises the
density once per call, which gives the same `rho1`:

```
rho1[0] = Σ_u (D·ao_0)_u ao_{0u}          # already only sees (D + D^T)/2
rho1[n] = Σ_u ((D + D^T)·ao_0)_u ao_{nu}  # n = 1,2,3
```

so both are the `hermi = 1` formulae with `D → (D + D^T)/2`.  The cost is one
`nset·nao²` transpose-add per call instead of one `nao_sub²·ng` GEMM and three
contractions per block, which is where most of the extra margin at `hermi = 0`
in §2 comes from.

## 5. D8, the other half of rank 12

The catalogue measures AO re-evaluation between response iterations at
**16.3 %** of a polarizability and **9.7 %** of a TDA run, from 2921 and 1651
`eval_ao` calls, and lists three possible routes.  On 1.8.1
(`benchmarks/response_anatomy.py`, same molecule and basis):

| | catalogue (1.7.0) | measured here (1.8.1) |
|---|---|---|
| polarizability, `eval_ao` calls | 2921 | **2921** |
| polarizability, share | 16.3 % | **2.2 %** (0.35 s of 16.09 s) |
| TDA (5 roots), `eval_ao` calls | 1651 | 2159 |
| TDA, share | 9.7 % | **1.2 %** (0.27 s of 22.16 s) |

The call count is exactly right and reproduces to the number, so the
phenomenon the entry describes is real: the grid and the AOs *are* invariant
across the whole response solve and *are* recomputed every iteration.  What is
not right is the price.  Caching them would cost `Σ_blocks 4·nao_sub·ng`
doubles — 4.2 GB on this molecule — to save 0.35 s of a 16 s run, and route
(b), doing all of one iteration's trial vectors in one grid pass, is what the
`nset` dimension already does.  **Rank 12 is worth taking for B4d and not for
D8**, and this is the second entry in the catalogue (after B5 and F2) whose
stated mechanism is not the one that pays.

## 6. What it is worth, and why the two sizes disagree

At fixed density, taxol (113 atoms, `nao = 1123`), same functional and grid:

| `nset` | hermi | stock | `blocked` | | penicillin V, same row |
|---|---|---|---|---|---|
| 1 | 1 | 0.2188 | 0.1915 | 1.14x | 1.81x |
| 3 | 1 | 0.4765 | 0.3332 | 1.43x | 2.93x |
| 5 | 1 | 0.7459 | 0.4731 | 1.58x | 3.35x |
| 10 | 1 | 1.4478 | 0.8146 | 1.78x | 3.81x |
| 20 | 1 | 2.9159 | 1.5223 | 1.92x | 4.29x |
| 3 | 0 | 0.6453 | 0.3370 | 1.91x | 3.97x |
| 5 | 0 | 1.0574 | 0.4793 | 2.21x | 4.65x |
| 20 | 0 | 4.1959 | 1.5331 | 2.74x | 5.85x |

and the meta-GGA path, M06-2X, the same two molecules:

| `nset` | hermi | taxol stock | taxol `blocked` | | penicillin V |
|---|---|---|---|---|---|
| 1 | 1 | 0.5483 | 0.5355 | 1.02x | 1.79x |
| 3 | 1 | 1.5360 | 1.1461 | 1.34x | 2.72x |
| 5 | 1 | 2.5260 | 1.7237 | 1.47x | 2.98x |
| 3 | 0 | 1.6464 | 1.1477 | 1.43x | 2.85x |
| 5 | 0 | 2.7109 | 1.7299 | 1.57x | 3.12x |

Every row of both tables is a gain and the trend in `nset` is the same, so by
catalogue §3.4's rule the target is real at both sizes and for both functional
types — but the **magnitude halves** between 42 and 113 atoms, and that has to
be stated rather than averaged away.  `FASTXCFXC_TIME=1`
says why (`nset = 5`, `hermi = 1`; the per-stage synchronisations inflate the
totals by about 10 %, so read the shares, not the seconds):

| stage | penicillin V | taxol |
|---|---|---|
| `D . ao_0` GEMM | 0.048 s (40 %) | 0.227 s (41 %) |
| `fxc_rho1` | 0.021 s (18 %) | 0.085 s (15 %) |
| `fxc_wv` | 0.004 s (3 %) | 0.012 s (2 %) |
| `fxc_scale` | 0.009 s (8 %) | 0.033 s (6 %) |
| `aow . ao_0^T` GEMM | 0.027 s (23 %) | 0.162 s (29 %) |
| `add_sparse` | 0.008 s (7 %) | 0.031 s (6 %) |
| **the two GEMMs together** | **63 %** | **70 %** |

The two GEMMs are `2·nset·nao_sub²·ng` flops each and are irreducible in this
formulation; measured against the A100's fp64 rate the first runs at about
two thirds of peak.  What this module removes is everything else, and
everything else is a smaller fraction of a larger molecule, because `nao_sub`
grows with the molecule while the launch count per block does not.  **The
speedup on this function will keep falling towards 1.4x–1.5x as systems get
bigger; the arithmetic floor is already 70 % of it at 113 atoms.**

That is also the argument against the orbital route.  `rho1` could be built
from the perturbed orbitals instead of the perturbed density matrix, as
`eval_rho4` does for the Hessian's MO path, which costs
`4·nocc·nao_sub·ng` per set against `nao_sub²·ng`.  On taxol that is
`4 × 226 × 500 = 452000` against `500² = 250000`: the density-matrix route is
already the cheaper one, as `fastxc`'s own `_rho_route` test would say.  And
it would only touch the first of the two GEMMs.

End to end, `benchmarks/response_anatomy.py`, both molecules:

| | stock | + fastxcfxc | |
|---|---|---|---|
| **penicillin V (42 atoms)** | | | |
| polarizability (CPKS, 3 RHS), whole run | 16.09 s | **14.07 s** | **1.14x** |
| ↳ `nr_rks_fxc` inside it | 4.05 s (25.2 %) | **1.88 s** (13.4 %) | **2.15x** |
| TDA, 5 roots, whole run | 22.16 s | **17.36 s** | **1.28x** |
| ↳ `nr_rks_fxc*` inside it | 6.30 s (28.4 %) | **1.64 s** (9.4 %) | **3.84x** |
| **taxol (113 atoms)** | | | |
| polarizability, whole run | 100.50 s | **95.22 s** | **1.06x** |
| ↳ `nr_rks_fxc` inside it | 17.41 s (17.3 %) | **11.46 s** (12.0 %) | **1.52x** |
| TDA, 5 roots, whole run | 189.54 s | **177.62 s** | **1.07x** |
| ↳ `nr_rks_fxc*` inside it | 20.83 s (11.0 %) | **8.87 s** (5.0 %) | **2.35x** |

Two things to read off that, both of which have to be said rather than
averaged:

* The end-to-end ratios are smaller than §2's for the CPKS rows because a real
  solve drops to `nset = 2` and then 1 as the three directions converge, so
  most of its 22 iterations are the `nset = 1` and 2 rows of §2, not the
  `nset = 3` one.
* **The whole-run ratio falls to 1.06x–1.07x at 113 atoms because the J and K
  build takes over**: 61 % of the TDA run and 71 % of the polarizability at
  42 atoms, **87 %** and **82 %** at 113.  This module does not touch it and
  neither does any other module here — see `docs/RESPONSE_1.8.1.md` §3.2 for
  why (the response path's fused `get_jk` is a third code path) and §4 for
  what closing it is worth (2.30x, measured on these kernels).

## 7. Scope

Covered: LDA, GGA and **meta-GGA**.  Meta-GGA is here even though `fastxc`
declines it in the SCF, and the reason is a measurement rather than symmetry:
on the response path it is not a tail.  An M06-2X polarizability spends
**48.4 %** of its wall clock in this function and a five-root M06-2X TDA run
**47.0 %**, against 25-28 % for B3LYP (`docs/RESPONSE_1.8.1.md` §2b) --
because `tau` needs one density product per gradient component and four
potential GEMMs where GGA needs one of each.  So the meta-GGA path here does
four `D . ao[n]` GEMMs into one buffer, one `fxc_rho1_mgga` that folds the
`tau` sum in, and four accumulating potential GEMMs, three of which are the
`tau` term written as `fxc_scale_lda` with `w0 = 4` and `a0 = 1, 2, 3`.

Declined, and handed back to GPU4PySCF rather than failed:

| | why |
|---|---|
| the `mo1` / `occ_coeff` path | that is the Hessian's CPHF solve.  On 1.8.1 it is a different function (`hessian/rks.py:3486 _nr_rks_fxc_mo_task`), and `docs/HESSIAN_1.8.1.md` measures it at a third of the card's fp64 peak — not launch-bound, so this treatment is not the one it wants |
| UKS (`nr_uks_fxc`) | open-shell response.  Same shape, twice the bookkeeping |
| more than one GPU | as every module here |
| an `fxc` cache that is not a contiguous `(ncomp, ncomp, ngrids)` array | the kernels index it with a stride of the whole grid; checked, not assumed |

## Files

| file | what |
|---|---|
| `electrolites/fastxcfxc.py` | the module: three modes, the dispatch and the fall-through conditions |
| `electrolites/kernels/fastxcfxc.cu` | `fxc_rho1`, `fxc_wv`, `fxc_scale` |
| `tests/test_fxc.py` | 15 cases against GPU4PySCF, plus the polarizability tensor and a TDA spectrum end to end |
| `benchmarks/fxc_bench.py` | the `nset` sweep of §2 and §6, all three modes, with the correctness check inline |
| `benchmarks/response_anatomy.py` | where a polarizability and a TDA run spend their time on 1.8.1 |
| `docs/RESPONSE_1.8.1.md` | that profile, and the J/K half this module does not reach |

## Environment switches

| variable | effect |
|---|---|
| `FASTXCFXC_MODE` | `blocked` (default), `nokernel` (the kernel ablation), `grouped` (catalogue B4d as written) |
| `FASTXCFXC_TIME` | print the per-stage split of every call |
| `FASTXCFXC_WX`, `FASTXCFXC_TSY`, `FASTXCFXC_UPB` | grid points, right-hand sides and AO indices per thread block |
| `FASTXCFXC_CHUNK_BYTES` | the memory budget of one `grouped` chunk |
