# Where a response property's wall clock goes, on GPU4PySCF 1.8.1

`instructions/06` §2.4 and §2.5 profile a polarizability and a five-root TDA
run on **1.7.0**, with a sampling profiler, and leave 18.4 s of the 44.31 s
polarizability as "not instrumented; mostly `nr_rks_fxc` `[?]`".  Its §5 then
ranks **B4d + D8** twelfth from those two profiles.  1.8.1 is a different
code: it runs the response J and K from a *fused* kernel, and its `eval_ao` is
several times faster.  So the two profiles were re-taken before anything was
written, with a `cudaDeviceSynchronize()` on both sides of every stage rather
than with a sampler.

`benchmarks/response_anatomy.py`.  B3LYP/def2-SVP, level-3 grids, one
A100-PCIE-40GB, `OMP_NUM_THREADS=32`, `conv_tol = 1e-9`, CPKS tolerance
1e-7.  Nested rows are sub-intervals of their parent.  Penicillin V (42
atoms, `nao = 430`) and taxol (113 atoms, `nao = 1123`) are the same two
molecules the catalogue and `docs/HESSIAN_1.8.1.md` use.

---

## 1. Polarizability (CPKS, three right-hand sides)

| stage | penicillin V | share | taxol | share |
|---|---|---|---|---|
| **total** | **16.09 s** | 100 % | **100.50 s** | 100 % |
| `cphf.solve` | 15.65 s | 97.2 % | 99.67 s | 99.2 % |
| ↳ the matrix–vector product | 15.50 s | 96.3 % | 99.42 s | 98.9 % |
| ↳↳ **`nr_rks_fxc`** | **4.05 s** | **25.2 %** | **17.41 s** | **17.3 %** |
| ↳↳↳ `eval_ao` | 0.35 s | 2.2 % | 1.50 s | 1.5 % |
| ↳↳ **`mf.get_jk`** | **11.42 s** | **71.0 %** | **81.97 s** | **81.6 %** |
| everything else | 0.45 s | 2.8 % | 0.83 s | 0.8 % |

22 CPKS iterations on penicillin V, 30 on taxol; `nset` runs 3, then 2, then
1 as the three dipole directions converge.

## 2. TDA, five roots

| stage | penicillin V | share | taxol | share |
|---|---|---|---|---|
| **total** | **22.16 s** | 100 % | **189.54 s** | 100 % |
| the Davidson matrix–vector product | 20.19 s | 91.1 % | 186.81 s | 98.6 % |
| ↳ **`nr_rks_fxc_st`** | **6.30 s** | **28.4 %** | **20.83 s** | **11.0 %** |
| ↳↳ `eval_ao` | 0.27 s | 1.2 % | 0.85 s | 0.4 % |
| ↳ **`mf.get_jk`** | **13.56 s** | **61.2 %** | **165.66 s** | **87.4 %** |
| the Davidson subspace algebra and the trial-space bookkeeping | 1.97 s | 8.9 % | 2.37 s | 1.3 % |

16 Davidson iterations on penicillin V, 17 on taxol; `nset = 5` throughout.

**The two sizes say different things about where to work.**  At 42 atoms the
XC response kernel is 25–28 % of a response property and the two-electron
build 61–71 %; at 113 atoms it is 11–17 % against **82–87 %**.  Both shares
move monotonically, and the reason is the same one as in
`docs/README_fastxcfxc.md` §6: the grid work grows as `nao_sub² · ngrids`
while the two-electron build grows faster than that until screening takes
over.  So a response-property target chosen from the 42-atom profile —
which is what catalogue §5 does — over-weights the grid half by a factor of
two.

## 2b. The same two properties across four functional types

The catalogue profiles one functional, B3LYP, and its C- and D-section entries
inherit that split.  The split is not a property of the response solver; it is
a property of the functional.  Penicillin V/def2-SVP, level-3 grids, stock
GPU4PySCF 1.8.1, five roots for TDA:

| functional | property | total | XC response | J + K | VV10 response |
|---|---|---|---|---|---|
| PBE (GGA, no exchange) | polarizability | 9.49 s | **45.6 %** | 47.9 % | — |
| | TDA | 9.16 s | **47.7 %** | 40.5 % | — |
| B3LYP (GGA hybrid) | polarizability | 16.33 s | 24.8 % | **71.4 %** | — |
| | TDA | 22.06 s | 28.1 % | **61.5 %** | — |
| M06-2X (meta-GGA hybrid) | polarizability | 22.53 s | **48.4 %** | 50.0 % | — |
| | TDA | 32.30 s | **47.0 %** | 45.3 % | — |
| wB97M-V (range-separated meta-GGA + VV10) | polarizability | 154.67 s | 6.4 % | 12.7 % | **80.6 %** |
| | TDA | 49.02 s | 34.3 % | **58.8 %** | excluded |

Three things follow, and each of them changes what is worth writing.

* **On a pure functional the grid half is the larger one**, and the two-electron
  half is `mf.get_j` -- which goes to the McMurchie-Davidson `j_engine` and so
  *is* on `fastj`'s patch target, unlike the hybrid case in §3.2.  A PBE or
  BLYP polarizability is the one response property `patch_all()` already
  covered end to end before this round.
* **A meta-GGA doubles the grid half**, because `rho1` needs a density product
  per gradient component for the `tau` term and the potential needs four GEMMs
  instead of one.  That is why `fastxcfxc` covers meta-GGA even though
  `fastxc` declines it in the SCF: here it is 47-48 % of the run, not a tail.
* **On an NLC functional neither of the catalogue's two stages is the target**
  -- see §3.3.

## 3. Four things this changes about `instructions/06`

### 3.1 D8 is a 1–2 % item on 1.8.1, not a 16 % one

The catalogue's D8 — "the grid and the AOs are invariant across the whole
response solve, yet recomputed every iteration" — rests on a call count and a
share.  The **count is exactly right and reproduces**: 2921 `eval_ao` calls in
a polarizability, ≈ 22 × 133 as stated.  The **share does not**: 16.3 % on
1.7.0 against **2.2 %** here, and 9.7 % against **1.2 %** for TDA.  0.35 s of a
16 s run.

Two of the entry's three routes are therefore not worth building.  Caching the
AOs (route a) would need `Σ_blocks 4·nao_sub·ng` doubles — 4.2 GB on
penicillin V — to save 0.35 s.  Doing a whole iteration's trial vectors in one
grid pass (route b) is what the `nset` dimension already does.  Route (c),
making `eval_ao` itself faster, is catalogue **A5** and is about the SCF, where
it is 18–21 %.

### 3.2 The response J and K come from a *third* code path, which neither `fastj` nor `fastk` reaches

`scf/_response_functions.py` calls `mf.get_jk(...)`, which lands in
`scf/jk.py:401 _VHFOpt.get_jk` — the fused `RYS_build_jk`, which builds J and
K from one pass over the Rys integrals.  The SCF does not go there:
`scf/hf.py:get_veff` calls `mf.get_j` (the McMurchie–Davidson `j_engine`) and
`mf.get_k` (Rys) as two separate builds.

`fastj` patches `j_engine._VHFOpt.get_j`, `fastk` patches
`jk._VHFOpt.get_k`.  **Neither is on the response path of a hybrid**, so a
B3LYP polarizability or TDDFT run is stock GPU4PySCF in the 61–82 % of it that
is the two-electron build: patching `fastj,fastk` and nothing else leaves the
`mf.get_jk` row of §1 where it is.  That is not a limitation any module's
"Limits" section stated, because the fused entry point was not in view.

A **pure** functional is the exception in reach if not yet in effect: with no
exchange, `_gen_rhf_response` calls `mf.get_j`, which goes to the
McMurchie–Davidson `j_engine` — so a PBE or BLYP response property *does*
land on `fastj`'s patch target.  It gains nothing there today, though: §4's
probe puts `fastj` at 0.221 s against GPU4PySCF's 0.221 s at `n_dm = 3` on
this molecule, against 2.6x at `n_dm = 1`, which is `docs/ROUND3_B4A.md`'s
result about the multi-density Coulomb kernels seen at a size and a density
count they were not tuned for (they were measured at `n_dm` = 4 to 32 on
284- and 545-atom clusters).  A CPKS solve asks for 3, then 2, then 1.  §2b
has the shares.

At `hermi = 0` — the TDDFT case — that kernel is additionally asked for twice
the densities it was given: `jk.py:443` stacks the transposed densities
(`dms = cp.vstack([dms, dms.transpose(0,2,1)])`) so that a tril-only quartet
enumeration can reach both triangles.

### 3.3 On an NLC functional there is a third stage, and it is the biggest one

For wB97M-V, wB97X-V, B97M-V or VV10, a CPKS solve adds
`hessian.rks.nr_rks_fnlc_mo` — the response of the VV10 non-local correlation
term — to every matrix-vector product.  It is **80.6 %** of a wB97M-V
polarizability on penicillin V, against 12.7 % for the J and K build and
6.4 % for `nr_rks_fxc`:

| stage | wB97M-V polarizability | share |
|---|---|---|
| **total** | **154.67 s** | 100 % |
| **`nr_rks_fnlc_mo`** | **124.74 s** | **80.6 %** |
| `mf.get_jk` + the long-range `get_k` | 19.64 s | 12.7 % |
| `nr_rks_fxc_st` | 9.95 s | 6.4 % |

The catalogue's C1 entry lists `nr_rks_fxc` and `get_jk` as what a
polarizability does; on an NLC functional that is 20 % of it.  `fastfnlc`
takes **1.43x** of this stage in this round, which is 1.39x on the whole
polarizability, and `docs/README_fastfnlc.md` §4 sizes what is left.

TDDFT does **not** pay it: `tdscf/rhf.py:408 gen_response` passes
`with_nlc=not self.exclude_nlc` and `exclude_nlc` is on by default, so the
same molecule's TDA run never calls the term.  A Hessian does, through
`hessian/rks.py:get_veff_resp_mo`, with `nset = 3·natm`, and there it is just
as dominant: on alanine dipeptide with wB97M-V the CPHF solve is **83.3 %**
`nr_rks_fnlc_mo` (112.88 s of 135.56 s) against 11.9 % for the multi-density J
and K build that `docs/HESSIAN_1.8.1.md`'s B3LYP profile ranks first.  What
`fastfnlc` is worth there is much less, though, and for a structural reason —
`docs/README_fastfnlc.md` §4b.

### 3.4 The catalogue's shares are 1.7.0 shares, and both totals fell by half

| | 1.7.0 (catalogue) | 1.8.1 (here) |
|---|---|---|
| polarizability, penicillin V | 44.31 s | 16.09 s |
| ↳ `get_jk` share | 58.5 % | 71.0 % |
| ↳ `nr_rks_fxc` share | ~41.5 % `[?]` | 25.2 % |
| TDA (5 roots), penicillin V | 42.07 s | 22.16 s |
| ↳ `get_jk` share | 62.5 % | 61.2 % |
| ↳ `nr_rks_fxc` share | 21.8 % | 28.4 % |

The catalogue's inference that the polarizability's unattributed 18.4 s was
"mostly `nr_rks_fxc`" is confirmed in kind — the function is the only other
thing on the path — but it was 41.5 % of a 1.7.0 run and is 25.2 % of a 1.8.1
one.

## 4. What the two-electron half would be worth, measured on *these* kernels

`docs/ROUND3_B4A.md` §4's rule: a ceiling measured on GPU4PySCF is not the
ceiling for a kernel this repository has already rewritten.  So the question
"should the response path split the fused build into `fastj` + `fastk`?" is
asked with a probe rather than by reasoning.

`benchmarks/response_jk_probe.py`, penicillin V/def2-SVP, on one converged
density with realistic perturbed densities (the first CPKS guess for the three
dipole directions, extended by the lowest single excitations).  `min` of three.
"fused" is `_VHFOpt.get_jk`; "split" is `j_engine.get_j` + `jk.get_k` — which
is the arrangement the SCF uses.  J and K agree between the two to 6e-12 and
6e-13 absolute.

**Stock GPU4PySCF:**

| hermi | `nset` | fused | split | ↳ J | ↳ K | fused/split |
|---|---|---|---|---|---|---|
| 1 | 1 | 0.522 | 0.679 | 0.171 | 0.509 | 0.77x |
| 1 | 3 | 0.562 | 0.749 | 0.221 | 0.527 | 0.75x |
| 1 | 5 | 0.645 | 0.918 | 0.351 | 0.566 | 0.70x |
| 1 | 10 | 0.898 | 1.253 | 0.572 | 0.681 | 0.72x |
| 0 | 5 | 0.886 | 1.041 | 0.349 | 0.692 | 0.85x |

The fused kernel wins on stock code at every `nset`: J costs essentially
nothing next to K when the two share the integrals (0.52 s fused against
0.51 s for K alone at `nset = 1`).  Upstream's choice is right.

**With Electrolites patched:**

| hermi | `nset` | fused | split | ↳ `fastj` | ↳ `fastk` | fused/split |
|---|---|---|---|---|---|---|
| 1 | **1** | 0.505 | **0.220** | 0.066 | 0.153 | **2.30x** |
| 1 | 3 | 0.562 | 0.771 | 0.221 | 0.550 | 0.73x |
| 1 | 5 | 0.631 | 0.820 | 0.254 | 0.566 | 0.77x |
| 1 | 10 | 0.894 | 1.040 | 0.344 | 0.696 | 0.86x |

The single-density row is the one to read.  `fastk` at `n_dm = 1` is **3.3x**
GPU4PySCF's K, `fastj` is **2.6x** its J, and the two together are **2.30x the
fused kernel**.  Every row below it is `fastk.py:175 if n_dm != 1: return
False` — the multi-density call falls straight back, and the split loses to the
fusion again.

**And the arrangement flips sign between the two molecules, so it cannot be
chosen once.**  The same probe on taxol (113 atoms, `nao = 1123`):

| hermi | `nset` | fused | split (stock) | fused/split | split (patched) | fused/split |
|---|---|---|---|---|---|---|
| 1 | 1 | 2.343 | 2.558 | 0.92x | **1.460** | **1.60x** |
| 1 | 3 | 4.252 | 4.547 | 0.93x | 4.566 | 0.93x |
| 1 | 5 | 6.246 | 6.369 | 0.98x | 5.722 | 1.09x |
| 0 | 1 | 3.263 | 3.036 | **1.07x** | 2.746 | **1.20x** |
| 0 | 3 | 7.336 | 6.220 | **1.18x** | 6.250 | **1.18x** |
| 0 | 5 | 11.538 | 9.286 | **1.24x** | 8.661 | **1.33x** |

At 42 atoms splitting is a 0.70x–0.93x loss on stock code at every `nset`; at
113 atoms it is a **1.07x–1.24x win at `hermi = 0`**, before any Electrolites
kernel is involved.  The reason is structural rather than accidental: at
`hermi = 0` the fused kernel is handed `2·nset` densities
(`jk.py:443` stacks the transposed ones so a tril-only quartet enumeration
reaches both triangles) while a split's J needs only `nset`, because
`j_engine.get_j` symmetrises the density instead.  As `nao` grows, K's share
of the fused kernel grows and the J it gives away for free is worth less, so
the wasted half of the J work stops being free.

That is exactly the size dependence catalogue §3.4 forbids relying on: a
`fastjk` that decided fused-versus-split from a threshold would be tuned to
one molecule.  It is also the argument for the third option — **one kernel that
builds both operators from one Rys pass, with the P1–P5 treatment and
`n_dm > 1`** — which has no arrangement to choose.

One more gap that has to be closed in that round, and is not in the catalogue:
`fastk` declines `hermi = 0` outright.  The `hermi = 0` rows above show
`..k` at GPU4PySCF's own timing even at `n_dm = 1` (2.409 s against stock's
2.395 s), because `_get_k_dispatch` hands the non-Hermitian request back.
Every TDDFT call is `hermi = 0`, so a multi-density `fastk` alone would still
not reach an excited-state calculation.

So the prize on the response path is between 1.5x and 2.3x of 61–82 % of the
wall clock, and it is **catalogue B4b**, which
`docs/CATALOGUE_06_STATUS.md` §7 already nominates as "first" from the Hessian
profile.  Two things this probe adds to that nomination:

1. **How much of the 2.30x survives at `n_dm > 1` depends on `fastj`, not on
   `fastk`.**  GPU4PySCF's K costs 0.509 s at one density and 0.527 s at three
   — its amortisation over densities is already excellent, so a multi-density
   `fastk` should stay near 0.16 s.  `fastj`, on the other hand, goes 0.066 s →
   0.221 s over the same step, which is `docs/ROUND3_B4A.md`'s finding about
   the Coulomb build's per-density cost showing up again.  At `n_dm = 3` a
   split with a multi-density `fastk` would be ≈ 0.16 + 0.22 = 0.38 s against
   0.56 s fused — **1.5x, not 2.3x**, and the J half would be the larger one.
2. **Which argues for a fused `fastjk` rather than a split.**  The fused Rys
   kernel gets J for free from integrals it computes for K; the split pays for
   the Coulomb integrals a second time through the MD engine.  Accumulating
   both operators from one Rys pass is what `fastejk` already does for the
   *gradient* of `j·J − k·K` (`codegen/gen_ejk.py`), so the machinery for it
   exists in this repository.  That is the shape the next round should take,
   and its ceiling is the 2.30x above rather than the 1.5x of a split.

## 4b. What this round is worth, by functional

Whole runs, penicillin V/def2-SVP, level-3 grids, one A100-PCIE-40GB.  "+
Electrolites" is `patch_all()`; the two modules this round added are
`fastxcfxc` (the XC response kernel) and `fastfnlc` (the VV10 response term).
Every pair is stock-then-patched **back-to-back in one script**, for the
reason below the table.

| functional | property | GPU4PySCF 1.8.1 | + Electrolites | | its XC response kernel | the untouched J+K row |
|---|---|---|---|---|---|---|
| PBE | polarizability | 9.492 s | **7.117 s** | **1.33x** | 4.324 → 1.962 s (2.20x) | 4.550 → 4.548 s |
| | TDA, 5 roots | 9.158 s | **5.127 s** | **1.79x** | 4.370 → 1.178 s (3.71x) | 3.705 → 2.859 s (`fastj`) |
| B3LYP | polarizability | 16.329 s | **14.258 s** | **1.15x** | 4.044 → 1.922 s (2.10x) | 11.666 → 11.701 s |
| | TDA, 5 roots | 22.063 s | **17.739 s** | **1.24x** | 6.192 → 1.713 s (3.61x) | 13.563 → 13.639 s |
| M06-2X | polarizability | 22.530 s | **15.587 s** | **1.45x** | 10.896 → 4.082 s (2.67x) | 11.256 → 11.142 s |
| | TDA, 5 roots | 32.304 s | **21.752 s** | **1.49x** | 15.172 → 4.697 s (3.23x) | 14.627 → 14.547 s |
| wB97M-V | polarizability | 154.671 s | **111.259 s** | **1.39x** | 9.954 → 3.666 s (2.72x) | 19.637 → 20.065 s |
| | TDA, 5 roots | 49.017 s | **38.102 s** | **1.29x** | 16.792 → 5.251 s (3.20x) | 28.804 → 29.547 s |

For the wB97M-V polarizability the row that matters is the third one this
profile added: its VV10 response term goes **124.741 s → 87.208 s (1.43x)**,
which is nearly all of that 1.39x.  Its TDA row has no VV10 term at all
(TDDFT excludes it by default) and is `fastxcfxc` alone.

The last column is the point of the round as much as the ratios are: on a
**hybrid** the two-electron build does not move at all — 11.666 → 11.701 s,
13.563 → 13.639 s, 11.256 → 11.142 s, 19.637 → 20.065 s, i.e. ±0.6 % — which
is the direct evidence for §3.2, measured back-to-back with everything
patched.  On a **pure** functional it does: PBE's TDA `get_j` goes
3.705 → 2.859 s through `fastj`'s multi-density kernels at `nset = 5`, while
at the `nset = 2`–3 a polarizability asks for they are neutral
(4.550 → 4.548 s), which is `docs/ROUND3_B4A.md`'s result seen at a density
count those kernels were not tuned for.

**How these were measured, and why it matters here.**  This is a shared node
whose CPU load swings between about 5 and 70, and a response run is
launch-heavy: the *same* stock B3LYP TDA measured 22.16 s, 22.45 s and, on a
loaded node, **43.31 s**, with its SCF going 5.7 s → 26.1 s and the fused J
and K build 13.5 s → 26.7 s.  Absolute wall clocks from different times are
therefore not comparable, and every stock/patched pair in the table above is
run **back-to-back in one script** so that only the ratio inside a pair is
quoted.

Within a pair the untouched stage is stable: five back-to-back measurements of
the fused `mf.get_jk` build on the same B3LYP TDA input give 13.51–13.56 s
(±0.4 %), and patching everything rather than only `fastxcfxc` costs it
nothing — 13.526 s against 13.556 s, whole runs 17.50 s against 17.56 s.  One
run in six came out at 15.06 s; that is an outlier, not a cost of patching,
and the modules register **zero** calls into `fastj` or `fastk` on a hybrid's
response path anyway (§3.2).

The per-stage ratios in the last column, and the two fixed-density benchmarks
(`fxc_bench.py`, `fnlc_bench.py`), run both implementations inside one process
and carry none of this.

## 5. D2, and the rest of the TDA remainder

Catalogue **D2** (rank 13, "not measured") is `tdscf/_lr_eig.py:133-134`
`xs = cp.vstack([xs, xt])` — the whole trial space reallocated and copied every
Davidson iteration.  It is still there on 1.8.1.  Sized: penicillin
V/def2-SVP has `nocc·nvir = 92 × 338 = 31096`, the subspace grows by 5 a
cycle over 16 cycles, so the copying totals `2 × 8 × 31096 × Σ(5…60) ≈ 0.19 GB`
— **under a millisecond of a 22 s run**.  On taxol, with
`226 × 897 = 202722`, it is 1.3 GB, still a few milliseconds.  The entry's own
premise ("at `nov ~ 10⁶` and space ~200 that is GB-scale copying per
iteration") needs `nov ~ 10⁶`, which is a 500-atom molecule at a triple-zeta
basis, and even there it is against Davidson iterations that cost seconds
each.  **Not worth taking**, and the fix already exists upstream as the
`V_holder`/`W_holder` pattern in the same file, so it costs nothing to leave.

What the 8.9 % actually is: `_qr` over the trial space, the `heff` assembly and
the `cupy.linalg.eigh` of the subspace matrix, plus the initial guess and the
final vector assembly, none individually above a few per cent.

## 6. Reproducing

```bash
python benchmarks/response_anatomy.py --prop pol \
    --xyz benchmarks/molecules/penicillinV.xyz --basis def2-svp --xc b3lyp
python benchmarks/response_anatomy.py --prop tda --nstates 5 \
    --xyz benchmarks/molecules/penicillinV.xyz --basis def2-svp --xc b3lyp
python benchmarks/response_jk_probe.py \
    --xyz benchmarks/molecules/penicillinV.xyz --basis def2-svp --xc b3lyp \
    [--patch all]
python benchmarks/fxc_bench.py \
    --xyz benchmarks/molecules/penicillinV.xyz --basis def2-svp --xc b3lyp
python benchmarks/fnlc_probe.py \
    --xyz benchmarks/molecules/penicillinV.xyz --basis def2-svp --xc wb97m-v
python benchmarks/fnlc_bench.py \
    --xyz benchmarks/molecules/penicillinV.xyz --basis def2-svp --xc wb97m-v
```

`--xc wb97m-v` on `response_anatomy.py` is what shows §3.3; `--xc pbe` shows
the non-hybrid case, where `mf.get_j` -- which `fastj` *does* patch -- replaces
the fused build and the XC response kernel is 45 % of the run; `--xc m06-2x`
shows the meta-GGA case, where it is 46 %.

Add `--patch fastxcfxc` (or `--patch all`) to the first two.
