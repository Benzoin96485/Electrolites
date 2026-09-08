# fastxchess — the XC part of an analytic Hessian

Catalogue entry **B5**, rank 7 in `instructions/06`'s §5 list and, until this
round, the highest-ranked entry with nothing done to it.  It names
`hessian/rks.py`'s two XC Hessian terms:

| function | called from | what it returns |
|---|---|---|
| `_get_vxc_deriv1` | `make_h1` | `d V_xc / d R_A` in the MO basis — the XC part of the CPHF right-hand side |
| `_get_vxc_deriv2` | `_get_exc_deriv2` | the second derivative of the XC energy, partially contracted with the density |

Both run a Python `for ia in range(natm)` loop *inside* `ni.block_loop`, which
is what the catalogue entry describes.  This module replaces both, for GGA.

---

## 0. Summary

* Two screens, one batching and one buffer, none of which changes the
  arithmetic: work in the **masked AO space** the grid block already produced
  instead of expanding back to the full `nao`; **skip the atoms that own no AO
  on the block**, whose contribution is identically zero; **contract the atom
  dimension in one call** rather than issuing 5–13 kernels per atom; and give
  the result **fixed-size scratch**, because the array the batching creates is
  2.6 GB a block at 113 atoms and allocating it per block turned out to be
  48 % of `_get_vxc_deriv1` (§3b — this one was found by measurement, not by
  reading the code).
* **7.34x** on the two terms together at 42 atoms and **5.30x** at 113, at
  fixed density; the 42-atom Hessian as a whole goes 84.26 s → 53.93 s with
  the rest of Electrolites, of which this module is 1.19x on its own.
* The catalogue's own prescription for B5 — "the same change as G1,
  generalized", i.e. batch the grid blocks so libxc is called once per group —
  is measured here and **is not where the time is**: `eval_xc_eff` is 1.5 % of
  `_get_vxc_deriv1` on penicillin V, against 42 % of the XC *gradient* where
  `fastxcgrad` found it.  The atom loop is 90 %.
* One thing the catalogue does not say, and the reason the entry is worth more
  than its 14.8 % suggests: `d rho / d R_A` is supported only where atom A's
  own basis functions are, so most of the atom loop is arithmetic on exact
  zeros.  That fraction is a property of how much space one grid block covers,
  so it improves with system size.
* Measured *not* to pay, before writing it: sub-dividing the 4096-point block
  to sharpen the atom screen.  §4.

---

## 1. Where the time is

`benchmarks/vxc_hess_probe.py` re-runs the GGA branch of
`_get_vxc_deriv1_task` stage by stage with a synchronisation on each.
B3LYP/def2-SVP, level-3 grids, one A100-SXM4-40GB.  It is a re-run, not an
instrumented call: it allocates its per-atom scratch inside the loop where
GPU4PySCF reuses a buffer, so it takes 8.2 s where the function itself takes
7.6 s, and the shares below are of its own total.

| stage | penicillin V (42 atoms) | taxol (113 atoms) |
|---|---|---|
| expand the masked AO block to `nao` | 2.6 % | 2.8 % |
| `eval_rho2` | 1.1 % | 0.4 % |
| **`eval_xc_eff` (libxc, deriv=2)** | **1.5 %** | **0.3 %** |
| weights and the atom-independent `v_ip` | 1.4 % | 1.6 % |
| `mo`, `ao_dm0` | 1.1 % | 1.2 % |
| **the atom loop** | **90.4 %** | **93.0 %** |

The same script times libxc both ways: all 519 680 points of the penicillin V
grid in one `eval_xc_eff(deriv=2)` call is 26.7 ms, and the same points in 127
calls is 119.3 ms.  So grouping the blocks — the catalogue's G1 treatment,
which `fastxcgrad` measured as the single largest item in the XC gradient —
can save at most 93 ms out of 8.2 s here.  **The reason it matters in the
gradient and not here is the block count**: the gradient's `get_exc` and this
function walk the same grid, but a deriv=2 libxc call does enough work per
point that its per-call overhead is amortised, and at 4096 points per block
there are only 127 calls.  This is a case where an entry's stated mechanism
survives only as a rounding error and a different mechanism in the same
function carries all of it.

## 2. What the atom loop actually costs

Per grid block, per atom, per Cartesian direction, `_get_vxc_deriv1` forms

    aow[j,g] = sum_y ao[y,j,g] wv2[y,g]        (nao x ngrids)
    mow[q,g] = sum_y mo[y,q,g] wv2[y,g]        (nocc x ngrids)

and contracts each against the other AO factor — two
`(nao x ngrids) . (ngrids x nocc)` GEMMs.  Over the whole grid that is

    2 terms x 3 directions x natm x nao x nocc x ngrids

multiply-adds: **10.4 TFLOP** on penicillin V.  GPU4PySCF takes 7.5 s over it,
i.e. **1.4 TFLOP/s**, on a card whose DGEMM peak is an order of magnitude
higher — the GEMMs are issued one atom at a time and are too small to fill it.

Two of the three factors in that product are larger than they need to be.

**`nao` should be `nao_sub`.**  Both functions open with

    ao1 = contract('nip,ij->njp', ao_mask, coeff[mask])

which scatters the block's masked AO values back into a full-`nao` array so
that `mol.aoslice_by_atom()`'s `p0:p1` slices apply.  Everything downstream
then runs at `nao`.  The catalogue flags this line; the fix is to stay in
GPU4PySCF's *sorted* AO order, where a density matrix between two AO factors
becomes `coeff . D . coeff.T` (which is exactly `opt.sort_orbitals`) and an
atom-restricted sum over AOs is still an atom-restricted sum, because `coeff`
is block diagonal in the shell.

**`natm` should be `natm_local`.**  `d rho / d R_A` is a sum over atom A's own
basis functions.  If none of them is in the block's mask, every term the
iteration computes is zero — *exactly* zero, since an AO outside the mask is
zero as far as GPU4PySCF is concerned, which is what makes this a screen with
no error term of its own rather than an approximation.

`benchmarks/xchess_locality.py` weights both fractions by grid points:

| | penicillin V (42 atoms) | taxol (113 atoms) |
|---|---|---|
| `nao_sub / nao` | 0.498 | 0.328 |
| `natm_local / natm` | 0.745 | 0.626 |
| product | 0.379 | 0.214 |

2.6x and 4.7x respectively, before anything is batched, and **the ratio
improves with system size** — `natm_local` counts the atoms whose basis
reaches one grid block, which stops growing once the block is smaller than the
molecule.

## 3. What is batched, and what cannot be

**`_get_vxc_deriv2` loses its atom loop entirely.**  Its two terms are

    t1[A,s,x,j] = sum_y sum_g W[x,y,j,g] wv2[A,s,y,g]
    t2[A,s,y,j] = sum_z sum_g U[y,z,j,g] wv2[A,s,z,g]

where `W[x,y,j,g]` is the AO component `_make_dR_dao_w` would combine times
the density product `(D . ao)[0,j,g]`, and `U[y,z,j,g] = (d_y ao)[j,g]
(D . ao)[z,j,g]`.  Neither depends on the atom.  GPU4PySCF rebuilds the
equivalent of `W` once per atom (three `_make_dR_dao_w` calls, each three
`nao x ngrids` arrays) and then contracts; building it once per block and
letting the atom index ride along as a free index of one cuTENSOR call
replaces about ten kernels per atom per block with six per block, and
materialises nothing of size `natm x nao_sub x ngrids`.

**`_get_vxc_deriv1` cannot do that.**  Its output carries a free AO index *and*
a free occupied index, so `aow[A,s]` has to exist and the two GEMMs per (atom,
direction) are inherent.  They are stacked over a block of atoms instead and
issued as one GEMM each — `(natm_blk*3*nao_sub x ngrids) . (ngrids x nocc)` —
which is what turns 1.4 TFLOP/s into a DGEMM.  The block of atoms is sized to
a fraction of free memory (`FASTXCHESS_MEM`, default 0.2), since `aow` is the
one array here whose size grows with the atom count.

`d rho / d R_A` itself, for all present atoms at once, is a **segmented sum
over AOs**: `d rho / d R_A = sum_{p in A} (...)`.  The AO-resolved products
are formed for the whole block and the atom dimension comes from one
contraction with a 0/1 matrix `P[a,p]`, which cuTENSOR runs as a GEMM — in
place of the thirteen slice contractions per atom that
`hessian.rks._make_dR_rho1` issues.

## 3b. The scratch, which turned out to be most of it

Batching the atom dimension makes `aow` an array of `natm_blk x 3 x nao_sub x
ngrids` — 2.6 GB a block on taxol — and `nao_sub` changes from block to block,
so allocating it per block means the memory pool cannot reuse the last one.
The first version of this module did exactly that, and a stage split says what
it cost (`_get_vxc_deriv1`, taxol, one atom block; a one-off instrumentation
of that version's inner loop with a `cudaDeviceSynchronize()` on each stage,
not a committed script — `benchmarks/vxc_hess_probe.py` is the same treatment
for GPU4PySCF's side):

| stage | time | |
|---|---|---|
| forming `aow` | **19.00 s** | 48 % |
| forming `mow` | 5.72 s | 15 % |
| the two GEMMs `aow . mo` and `mow . ao` | 3.57 + 4.03 s | 19 % |
| `d rho / d R` for every atom | 1.22 s | 3 % |
| `v_ip`, libxc, `rho`, `mo`, the scatter | 1.73 s | 4 % |
| unattributed — the stage timers' own synchronisation | 4.0 s | 10 % |
| total | 39.3 s | 100 % |

Two things in that table.  The **GEMMs are at 12.8 TFLOP/s** — 97.6 TFLOP of
fp64 in 7.6 s, which is where a DGEMM of that shape should be, so the
arithmetic this module set out to reorganise is not the problem.  And forming
`aow` costs five times what the GEMM that consumes it does, for an operation
that is a weighted sum of four arrays.  Swapping cuTENSOR for GPU4PySCF's own
`_scale_ao` kernel changes it by 14 % (21.7 s), which is the tell: two
unrelated kernels do not agree to 14 % on the arithmetic, they agree because
neither of them is the cost.

The cost is the allocation, and the fix is the one `fastxcgrad` records for
`_grouped_block_loop` — one buffer per name, sized once for the largest
`nao_sub` any block has, with `cupy.ndarray(shape, memptr=...)` views on top,
which is also GPU4PySCF's own idiom inside these two functions.  Measured on
penicillin V, that change alone takes `_get_vxc_deriv1` from **3.24 s to
1.55 s**, and it is why §6's numbers exceed the screening ceiling §2 sets:
the ceiling only counts the work the screens remove, and this removes work
neither GPU4PySCF nor the first version of this module was doing on purpose.

## 4. The screen that does not pay

The atom screen above is at block granularity: 4096 grid points.  A 4096-point
block covers far more space than one atom's basis, which is why
`natm_local / natm` is as high as 0.745 — so the obvious next step is to
sub-divide the block and keep a per-chunk atom list.  Measured first
(`benchmarks/xchess_locality.py --chunks 4096,1024,256,64`):

| chunk | penicillin V `natm_local/natm` | taxol |
|---|---|---|
| 4096 | 0.745 | 0.626 |
| 1024 | 0.724 | 0.591 |
| 256 | 0.705 | 0.565 |
| 64 | 0.683 | 0.537 |

Dividing the block by 64 buys 9 % on penicillin V and 17 % on taxol, for a
per-chunk atom list, a gather, and GEMMs with a 64-deep contraction.  The
reason it is so flat is the cutoff: GPU4PySCF's AO mask keeps a shell whose
value anywhere in the block exceeds **1e-10**, and a Gaussian tail stays above
1e-10 long past the point where it changes the answer.  A screen with a real
error budget — the P11 treatment `fast1e` used for `int3c1e` — would cut
deeper, but it would also be an approximation where this module's two screens
are not, so it belongs in a different entry.  Not implemented.

## 5. Correctness

`tests/test_xchess.py` calls both implementations on the same converged SCF,
in one process: `_get_vxc_deriv1` and `_get_vxc_deriv2` term by term, and then
the whole analytic Hessian with the module patched and unpatched.  Eleven
cases: five basis sets to def2-TZVPP, seven functionals including a
range-separated hybrid, and the two paths the module declines.

| basis / functional / grid | atoms | `_get_vxc_deriv1` | `_get_vxc_deriv2` | whole Hessian | `self` |
|---|---|---|---|---|---|
| def2-SVP / B3LYP / L3 | 3 | 3.4e-15 | 2.8e-14 | 1.5e-12 | 0 |
| def2-TZVPP / PBE / L3 | 3 | 2.2e-15 | 1.4e-14 | 2.6e-12 | 0 |
| 6-31G\* / PBE0 / L1 | 3 | 2.5e-15 | 2.8e-14 | 1.9e-12 | 0 |
| def2-SVP / B3LYP / L3 | 6 | 1.0e-15 | 5.7e-14 | 1.9e-12 | 0 |
| cc-pVDZ / BLYP / L3 | 6 | 2.6e-15 | 1.4e-14 | 3.6e-12 | 0 |
| 6-31G\* / B3LYP / L3 | 10 | 1.8e-15 | 5.7e-14 | 4.1e-12 | 0 |
| def2-SVP / ωB97X-D3BJ / L3 | 10 | 2.2e-15 | 2.8e-14 | 6.4e-12 | 0 |
| def2-SVP / B3LYP / L3 | 22 | 3.3e-15 | 5.4e-13 | 1.2e-11 | 0 |
| 6-31G\*\* / PBE / L2 | 22 | 7.6e-15 | 4.8e-13 | 7.6e-12 | 0 |
| def2-SVP / **SVWN** / L3 (declined) | 3 | 0 | 0 | 1.2e-12 | 0 |
| def2-SVP / **M06-2X** / L3 (declined) | 3 | 8.2e-05 | 0 | 2.5e-05 | 8.2e-05 |

Absolute maxima, in atomic units; the Hessian's own scale is `max |H|` ~ 0.7-0.9.
The two declined rows run GPU4PySCF's code on **both** sides, so what the test
asserts there is the fall-through itself (`_setup` returning None), not a
number — the LDA row's exact zeros are what a fall-through looks like, and the
M06-2X row's 8.2e-05 is upstream's meta-GGA path disagreeing with itself
between two calls.  Those two numbers are why a declined row's columns cannot
be a pass criterion: across three runs of this suite they read 8.2e-05 / 3.0e-05,
4.9e-05 / 0, and 8.2e-05 / 8.2e-05, the second of them with the two ORIG calls
agreeing exactly by luck.  The 2.1e-12 on the LDA row, with two
zeros beside it, is the rest of the Hessian's own run-to-run spread — the J
and K builds accumulate with `atomicAdd`, and the CPHF solve iterates on top
of that at `conv_tol_cpscf = 1e-6` — so it is the floor for every row's
whole-Hessian column, not something this module contributes.

`tests/test_xchess_fd.py` is the independent check the proposal's T10 asks
for: the analytic Hessian against central differences of the analytic
gradient, `eps = 1e-3` bohr, `grids.level = 6`, tolerance 1e-3.  It does not
compare the two implementations, so an error they shared would still show.

| molecule / basis / functional | max &#124;H_analytic - H_fd&#124; | max &#124;H&#124; | xctype |
|---|---|---|---|
| H2O / def2-SVP / B3LYP | 1.4e-05 | 0.71 | GGA (patched) |
| H2O / 6-31G\* / PBE | 1.8e-05 | 0.70 | GGA (patched) |
| NH3 / def2-SVP / B3LYP | 1.9e-05 | 0.67 | GGA (patched) |
| H2O / def2-SVP / SVWN | 1.0e-05 | 0.72 | LDA (declined, so upstream's) |

Four of four within 1e-3, and all four within 2e-5 — which is the size of the
finite-difference truncation error at `eps = 1e-3`, not of anything in the
Hessian.  The test does have teeth: the same four cases at `--level 1` come
out at **1.3e-2 to 6.5e-2**, three orders of magnitude worse, because a
level-1 grid is not good enough to differentiate twice.  That is the grid, not
the implementation, and it is why the protocol fixes `grids.level = 6`.  The LDA row is the check on §5's algebra: if the factor 2 really
were missing from GPU4PySCF's LDA branch, that row would be off by the size of
the `fxc` term, and it is not.

### The LDA branch's missing factor 2, which is not missing

Worth writing down because it looks like a bug and is not, and because it is
the reason LDA is a separate piece of work rather than a special case of this
module.  `_get_vxc_deriv1` builds `d rho / d R` in its GGA branch through
`_make_dR_rho1`, which ends with

    rho1 *= 2      # *2 for |mu> DM <d_X nu|

and in its LDA branch inline, with

    # First order density = rho1 * 2.  *2 is not applied because + c.c. in the end

Both branches then perform the same two contractions -- one with the
derivative on the bra, one on the ket -- so on the face of it the LDA
quantity is half the GGA one.  It is not, because the GGA branch also applies

    wv2[:, 0] *= .5

to the *density* component only.  For the density component the two
contractions produce the same term twice, so the `+ c.c.` supplies the factor
2 and the explicit one has to come back off; for the gradient components they
produce two different terms (`d ao . ao` and `ao . d ao`) and the 2 stays.
Net: `2 x 0.5 = 1` on the density component, `2` on the gradient components,
and the LDA branch's `1` for its only component agrees with the GGA branch's
`1` for the same component.

So there is nothing to fix, and nothing to pick between -- but the factor
conventions differ term by term between the branches, so LDA needs its own
`_dR_rho1`, its own weights and its own `W`/`U` rather than falling out of the
GGA code with `ncomp = 1`.  At 0.4-0.1 % of a Hessian (`_get_vxc_diag`'s
share, which is where an LDA Hessian's XC time mostly goes) that is not this
round's hour.  `tests/test_xchess_fd.py` keeps an LDA row against finite
differences so that "they agree" stays a measurement.

### One more thing the test found

The `self` column of `tests/test_xchess.py` is GPU4PySCF against itself: the
same function, called twice, on the same input.  It is exactly zero for every
LDA and GGA case.  For **meta-GGA (M06-2X) it is 3.0e-5** — GPU4PySCF's
meta-GGA `_get_vxc_deriv1` is not reproducible run to run at that level, and
the resulting Hessian moves by 1.7e-5.  That is a different statement from
`hessian/rks.py`'s own warning that "MGGA Hessian is sensitive to dft grids";
this is at a fixed grid.  It is one reason meta-GGA is not covered here: there
is no reference to converge against until that is understood.

## 6. What it is worth

`benchmarks/xchess_bench.py` times both terms on one converged density, in
one process, min of three; `no screen` is the ablation that keeps the masked
AO space and the batching and drops the atom screen, so the two mechanisms can
be separated.  One A100-SXM4-40GB, B3LYP/def2-SVP, level-3 grids.

| | GPU4PySCF | fastxchess | | no atom screen | | the screens alone |
|---|---|---|---|---|---|---|
| **penicillin V**, 42 atoms, 430 AOs, 92 occ | | | | | | |
| `_get_vxc_deriv1` | 7.619 s | **1.502 s** | **5.07x** | 1.816 s | 4.19x | 2.6x |
| `_get_vxc_deriv2` | 7.850 s | **0.607 s** | **12.94x** | 0.601 s | 13.06x | 2.6x |
| both | 15.469 s | **2.109 s** | **7.34x** | 2.418 s | 6.40x | 2.6x |
| **taxol**, 113 atoms, 1123 AOs, 226 occ | | | | | | |
| `_get_vxc_deriv1` | 62.512 s | **19.789 s** | **3.16x** | 27.707 s | 2.26x | 4.7x |
| `_get_vxc_deriv2` | 55.307 s | **2.428 s** | **22.78x** | 2.719 s | 20.34x | 4.7x |
| both | 117.819 s | **22.217 s** | **5.30x** | 30.425 s | 3.87x | 4.7x |

The last column is §2's product of the two screening fractions — what the two
screens are worth on their own, if the arithmetic that survives them ran at
the efficiency GPU4PySCF gets.  Both terms beat it, for different reasons, and
the ablation column separates them:

* **`_get_vxc_deriv2`: 12.9x, and the atom screen is worth nothing** (0.601 s
  without it against 0.607 s with — the difference is noise).  Its atom loop
  was never about the work; it was about issuing ~10 kernels per atom per
  block for arithmetic that costs `nao_sub x ngrids` each.  Batching is the
  whole of it.
* **`_get_vxc_deriv1`: 5.07x, of which the atom screen is 1.21x.**  Its atom
  loop *was* real work — two GEMMs per (atom, direction) — so the screens
  matter here; but 5.07x against their 2.6x is the other half of §3b, the
  scratch that GPU4PySCF allocates per atom per block and this module
  allocates once.

### The two sizes disagree, in a way worth stating

`_get_vxc_deriv2` gets **better** with size (12.9x → 22.8x) and
`_get_vxc_deriv1` gets **worse** (5.07x → 3.16x), while the screening ceiling
goes the other way (2.6x → 4.7x).  Both are the same effect seen from two
sides:

* the **screens** are worth more on a bigger molecule, because a grid block
  reaches a smaller fraction of it — that is the ceiling column, and the
  ablation column tracks it (the atom screen is 1.21x at 42 atoms and 1.40x
  at 113);
* the **batching** is worth more on a *smaller* one, because what it removes
  is a per-atom cost — a launch, an allocation, a GEMM too small to fill the
  card — and those hurt most when `nao` and `nocc` are small.  At 113 atoms
  GPU4PySCF's own per-atom GEMMs are 1123x226x4096 and already run well; at
  42 atoms they are 430x92x4096 and do not.

`_get_vxc_deriv2` has no real GEMM, so for it the first effect is all there is
and the ratio climbs.  `_get_vxc_deriv1` has both, and they cancel to
something between 3x and 5x over this size range.  Neither term is close to
being launch-bound any more: after §3b, `deriv1`'s two GEMMs are 38 % of its
remaining time at 113 atoms and run at 12.8 TFLOP/s.  (The AO evaluation is
*not* the floor here, which is worth saying because it would be the natural
guess: walking `ni.block_loop` at `deriv=2` and doing nothing else is
**0.015 s** on penicillin V and **0.056 s** on taxol —
`benchmarks/vxc_hess_probe.py` times it.  The 10 % gap in §3b's table is the
stage timers' own synchronisation, eighteen of them per block.)

`benchmarks/hessian_anatomy.py` puts that into a whole Hessian.  `stock` is
GPU4PySCF 1.8.1; `+ Electrolites` is `patch_all()` as it stood before this
round, where the only module that reaches a Hessian is `fastj`, through the
multi-density Coulomb build in CPHF.

| | GPU4PySCF 1.8.1 | this module only | | + the other eleven | |
|---|---|---|---|---|---|
| **penicillin V**, 42 atoms | | | | | |
| whole Hessian | 84.26 s | 70.87 s | 1.19x | **53.93 s** | **1.56x** |
| ↳ `_get_vxc_deriv1` | 7.53 | 1.53 | 4.92x | 1.55 | |
| ↳ `_get_vxc_deriv2` | 7.86 | 0.70 | 11.2x | 0.68 | |
| ↳ `get_k` in CPHF, untouched | 20.20 | 20.43 | | 20.12 (37 % of the run) | |
| **taxol**, 113 atoms | | | | | |
| whole Hessian | 1519.0 s | — | | **1313.8 s** | **1.16x** |
| ↳ `_get_vxc_deriv1` | 62.39 | — | | 19.92 | 3.13x |
| ↳ `_get_vxc_deriv2` | 54.09 | — | | 2.55 | 21.2x |
| ↳ `get_k` in CPHF, untouched | 718.75 | — | | 720.73 (55 % of the run) | |

The Hessians agree to 2e-11 on the trace and the Frobenius norm at both sizes.
`docs/HESSIAN_1.8.1.md` has the full stage tables, and the reason the
whole-run column is 1.16-1.56x rather than 5-7x: after this round every
remaining Hessian target is inside the CPHF solve, and the exchange build
there is 37 % of a 42-atom run and 55 % of a 113-atom one, with `fastk`
handing every multi-density call straight back to GPU4PySCF.

The two sizes disagree about the *share* and agree about the *speedup*, which
is the shape catalogue §3.4 asks about: the mechanism generalises (it gets
stronger with size, because `natm_local/natm` falls), while the term's weight
in a Hessian falls faster still, because the CPHF cost grows as
`n_dm x nao^2`-ish where a grid term grows as `natm x ngrids`.
`docs/HESSIAN_1.8.1.md` §4 has that comparison for every Hessian entry, and
the conclusion it comes to about what to do next.

## 7. Scope, and what is left

Covered: GGA (so B3LYP, PBE0, ωB97X-D3BJ and the rest of the hybrids), spin
restricted, one GPU.

Declined, falling through to GPU4PySCF unchanged: **meta-GGA** (§5),
**LDA** (§5), the **grid-response** path (`hessobj.grid_response`, which
GPU4PySCF routes to `_get_vxc_deriv1_grid_response` — and which, unlike the
default path, already carries the masked-and-sparse machinery this module
adds), the **VV10 Hessian** (`_get_enlc_deriv2`, catalogue A6), unrestricted
references (`hessian/uks.py` has its own copies of both functions), density
fitting, and multi-GPU.

`_get_vxc_diag`, the third XC Hessian term, has the same shape but is 0.4–0.1 %
of the Hessian and is left alone.

What is left in the same neighbourhood, sized by the profiles in
`docs/HESSIAN_1.8.1.md`:

1. **The same treatment for `hessian/uks.py`.**  Both functions are duplicated
   there for two spin channels.
2. **Meta-GGA**, once the nondeterminism above is understood.  The batching
   generalises directly; the tau terms add four more GEMMs per (atom,
   direction), so the atom screen is worth proportionally more there.
3. **`nr_rks_fxc_mo`** (catalogue B4d) is on this list, but well down it, and
   the reason is worth writing down because the catalogue puts it higher.  Its
   per-set loop looks like the same launch problem B5 has, and it is not: its
   arithmetic is `nset x nao_sub^2 x ngrids`, which for penicillin V's CPHF is
   about 6.0 TFLOP per call, and it does that in 0.91 s (5.46 s over six
   calls) — **6.6 TFLOP/s**, a third of the card's fp64 peak.  A function
   running at a third of peak is not launch-bound, so batching its sets is
   worth the difference between a 214x214x4096 GEMM and a large one, not the
   5-13x that B5's atom loop gave.  **But** §3b is a reason to look again
   before believing that: `nr_rks_fxc_mo` allocates a `nset x nao x nao`
   `vmat` and a per-block `aow`, and 6.6 TFLOP/s is exactly the sort of
   two-thirds-of-peak number that turned out, here, to be an allocation and
   not the arithmetic.  The cheap experiment is to preallocate its scratch and
   re-time it, before writing any batching.
4. **`_get_vxc_deriv1`'s remaining 1.5 s.**  With the scratch fixed, the
   screens explain 2.6x of its 5.07x and the rest is §3b, so what is left is
   the arithmetic itself: two GEMMs at 12.8 TFLOP/s over
   `natm_local x 3 x nao_sub x nocc x ngrids`.  Cutting that means cutting
   `natm_local` or `nao_sub` further, which is §4 — measured at 9-17 % for a
   per-chunk atom list — or finding a formulation whose intermediate is
   smaller than `aow`.  `aow` is already the smaller of the two candidates
   (`nao_sub` against `4 x nocc`), so this looks close to done.

## Files

| file | what |
|---|---|
| `electrolites/fastxchess.py` | the module |
| `benchmarks/vxc_hess_probe.py` | the stage split of §1, and libxc one call against 127 |
| `benchmarks/xchess_locality.py` | the two screening fractions of §2 and §4, at four grid granularities |
| `benchmarks/xchess_bench.py` | §6, the two terms A/B'd on one converged density, with the atom-screen ablation |
| `benchmarks/hessian_anatomy.py` | the end-to-end Hessian profile, stage by stage |
| `tests/test_xchess.py` | §5, eleven cases against GPU4PySCF in one process |
| `tests/test_xchess_fd.py` | §5, against finite differences of the gradient |
| `benchmarks/molecules/generate.py` | the two geometries, from SMILES, reproducibly |
