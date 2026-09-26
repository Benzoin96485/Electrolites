# The ip2 term's metric as a Cholesky factor, and the round-off that decides where it may be used

`docs/DF_WORK_LIST.md` **W24**, found by W17.  After W17, at 113 atoms, 13.4 s
of the 33.3 s `df/hessian/rhf.py:_jk_energy_per_atom` was the same
`2·3·naux²·nao·nocc`-flop contraction three times — `(10|0)` into the fitting
metric's eigenbasis (D3), back out of it (D5), and once more against
`dm_oo` (D5) — and the `ejk_int3c2e_ip2` kernel was 44–50 % of the call
wherever a transition metal's f shells were in the auxiliary set.  The entry
asked for D3 + D5 at two contractions' cost, or the kernel moved.

It is the first: D3 + D5 are now **two** contractions' cost, both of them
with the auxiliary index spherical where one of every three was Cartesian,
and the function is **2.3x–2.5x** GPU4PySCF's at 113 atoms (W17 left it at
1.89x).  The kernel is profiled class by class and not moved.

---

## 0. Summary

* **The metric as `J = L Lᵀ` on the spherical auxiliary index** (item 8,
  §2).  Upstream's `metric_v` is `C v` — `C` the Cartesian-to-spherical
  coefficients, block diagonal by shell — so the eigenbasis contraction of
  D3 is really a spherical one wearing a Cartesian index.  With the
  spherical `(10|0)` held whole (**exactly the memory the one-chunk eigen
  pass already holds**, `3·naux_sph·nao·nocc` words), D3 becomes a pass over
  memory, the eigenbasis in and out of D5 becomes two triangular *multiplies*
  against `L⁻¹` (which is one contraction's cost between them), and the
  third contraction is spherical in both auxiliary indices.  At 113 atoms
  D3 + D4 + D5 go **17.7 s → 11.7 s** of the call, D8 included.
* **The factor has to be used everywhere in the call or nowhere** (§3).
  The first version used it in D3–D5 and upstream's eigenbasis in the other
  five places this function applies the metric.  On a Ni(II) complex, whose
  metric has condition number 1.6e9, that differed from upstream by
  **1.7e-9** relative, and a one-ulp perturbation of the metric moved *it*
  by 2e-9 while it moved upstream's by 1e-11: the terms of this function
  cancel, and two different round-offs of `J⁻¹` do not.  Since upstream only
  ever uses `V W⁻¹ Vᵀ`, `V = C L⁻ᵀ` with `W = 1` feeds every other line
  unchanged, and the difference drops to **3.2e-11** with a perturbation
  sensitivity of 5e-11.  A triangular *solve* instead of the multiply is
  bit-for-bit as far off (§3.2): it was never the explicit inverse.
* **Three smaller items** (§4): the two symmetric products of the `h_aux`
  block and D4's `R` as `dsyrk` (items 9 and 10; D2 4.7 → 3.4 s, D4 4.2 →
  2.1 s at 113 atoms), `Q` from one GEMM and a shared-memory transpose
  instead of two GEMMs (item 10), and the `h_aux` block's `naux²`
  temporaries carved from the integral buffers the fused pass has finished
  with, so that item 3's fusion is no longer declined at 113 atoms for want
  of 8.4 GB (item 11; 1.0 s there).
* **What it is worth** (§5): the function is **1.24x** W17's at 113 atoms
  without item 11 (33.3 → 26.9 s, one process) and another 1.04x with it
  (25.8 → 24.8 s), 1.11x at def2-TZVPP; at 21–42 atoms in def2-SVP it is
  1.05x–1.06x, because there the contractions were already small and
  **the integral kernels are 70 % of what is left**.  In a whole Hessian
  that is 1.00x–1.02x: the ip2 term was 15–30 % of one after W17.
* **The kernel** (§6): `ejk_int3c2e_ip2` is 46 % of the call at 42 atoms,
  51–52 % on the two transition-metal complexes and 16 % at 113.  By
  auxiliary angular momentum its cost per function rises twentyfold from
  `l = 0` to `l = 6`: on the Ni complex **one `l = 6` shell (28 functions,
  1 % of the set) is 11 % of the kernel**.  That is W19's situation — a
  Rys kernel whose cost is in the high classes — and it is **W29**, not a
  part of this round.
* **The falsification criterion fires in its letter and not in its
  intent** (§2.3).  The resident block fits exactly where the chunked form
  is one chunk — they are the same size — but every system this repository
  runs a density-fitted Hessian on, up to 113 atoms on an 80 GB card, is one
  chunk.  Past it the factor still runs, through upstream's chunked
  arithmetic, and is as fast as the eigenbasis there.

Everything here is one A100-SXM4-80GB of a whole `--qos=debug` node,
`CUDA_VISIBLE_DEVICES=0`, GPU4PySCF 1.8.1, def2-SVP with def2-SVP-jkfit
unless a row says otherwise, B3LYP (`j = 1, k = 0.2`), `patch_all()`.
Function-level rows are `benchmarks/df_hess_terms_probe.py`, variants round
robin in one process against GPU4PySCF's own function object
(`DF_WORK_LIST.md` §5.2); "W17" is this module with items 8–11 switched off
(`FASTDFHESSIP_METRIC=eigen FASTDFHESSIP_SYRK=0 FASTDFHESSIP_D4=gemm
FASTDFHESSIP_CARVE=0`; the 113-atom W17 column was taken before item 11
existed, so that switch was implicit there).

---

## 1. Where the time was

`FASTDFHESSIP_TIME=1`, taxol (113 atoms, `nao_cart` 1185, `nocc` 226,
`naux` 6598 Cartesian / 5598 spherical), one call:

| stage | what | W17 | this round |
|---|---|---|---|
| A | `(pq|P)` → `j3c_oo` | 1166 ms | 1012 |
| B | the metric → `dm_oo` | 878 | 677 |
| C | pseudo-density | 486 | 479 |
| C | the `ejk_int3c2e_ip2` kernel | 4107 | 4104 |
| D1 | the first `ipaux` pass | 2395 (not fused) | **1315** (fused, item 11) |
| D2 | `h_aux`, the auxiliary-auxiliary block | 4686 | **3442** |
| D3 | `ip1` + `ipaux` integrals | 2242 | 2250 |
| D3 | `(10|0)` → `j3c_100`; here the spherical `K` | 4903 | **1364** |
| D8 | `L⁻¹ K` and `L⁻ᵀ Y`, two `dtrmm` | — | **3443** |
| D4 | the per-atom occupied block (W17's Gram form) | 4168 | **2082** |
| D5 | the per-atom cross terms | 8584 | **4767** |
| | **whole call** | **33.7 s** | **25.0 s** |

(W17's column is `docs/README_fastdfhessip.md` §3, taken with other GPUs of
the node busy; this round's is the second of two calls in one process.  The
untimed calls, round robin in one process, are in §5.)  D3 + D4 + D5 —
the three stages the entry was about, with D8 now between them — go
**17.7 s → 11.7 s**; the first pass is fused again (item 11) and D2 and D4
are `dsyrk` (items 9 and 10).

---

## 2. Item 8: the Cholesky form

### 2.1 What upstream computes, and what it needs

With `K[x,r,u,i] = sum_q (10|0)[u,q,r] C_occ[q,i]` (Cartesian auxiliary `r`),
upstream forms

* D3: `j3c_100 = Vᵀ K` — into the eigenbasis, chunked over it;
* D4: `sum_r' S_A[r'] S_B[r'] / w_r'` from `j3c_100` (W17's Gram form);
* D5, per atom `A`: `tmp0 = (V/w) j3c_100` — back out, `3 naux_cart ×
  metric_size` — contracted element-wise with `(1|00)`-derived `T1`; and
  `sum_uj P[s,u,j] j3c_100[x,r',u,j] / w_r'` against `j2c_10 V` — the third.

Three `2·3·naux_cart·naux_sph·nao·nocc` contractions.  But `V = C v` with
`v` the spherical eigenvectors (`_factorize_j2c` ends in
`auxmol.apply_C_dot(v)`), and every quantity above is a function of
`J⁻¹ = C v W⁻¹ vᵀ Cᵀ` alone.  With `J_sph = L Lᵀ` and `K_sph = Cᵀ K`:

* `Y = L⁻¹ K_sph` is a whitened `j3c_100` — a rotation of `W^{-1/2} vᵀ
  K_sph` — so every `sum_r' X_r' Y_r' / w_r'` of D4 is the same sum over
  `Y`, and D4 runs unchanged on it;
* `tmp0 = C Z` with `Z = L⁻ᵀ Y`.  `C` does not mix atoms, and the output
  index `s` of that term is only ever summed over an auxiliary atom, so
  `sum_{s in B} (C Z)[s] T1[s] = sum_{s' in B} Z[s'] (Cᵀ T1)[s']` — `Cᵀ` is
  applied to `T1` (one pass over memory), not `C` to `Z` (a contraction);
* the third term is `sum_t j2c_10C[y,s,t] sum_uj P[s,uj] Z[x,t,uj]`, and by
  the same argument `P` and `j2c_10C` can both be taken to spherical `s`
  (`FASTDFHESSIP_T2SPH`).

So D3 is the block-diagonal `Cᵀ` applied to each integral batch as it
arrives (`_SphAux.cols`: each batch of whole shells maps to a contiguous run
of spherical functions once the spherical index is ordered by first
appearance), the two triangular multiplies are one `naux_sph² · 3 nao nocc`
contraction between them, and the third term is `2·3·naux_sph²·nao·nocc`.
Upstream's three Cartesian-by-spherical contractions become the equivalent of
two spherical-by-spherical ones: `2 · 5598 / (3 · 6598) = 0.57` of the
arithmetic at 113 atoms.

`dtrmm` rather than `dtrsm`: `docs/README_fastdfcderi.md` measured the solve
at 39 % of `dgemm` at these shapes and the multiply at GEMM rate, and so it
is here (D8: 1.67 s for `5598² × 803 k`, 15 TF/s).  `L⁻¹` is formed once, by
a triangular solve against the identity.

### 2.2 In the code

`electrolites/fastdfhessip.py`: `_SphAux` (the permuted `C`, its
block-diagonal apply, the factor), the D3 accumulation `contract('rs,pir->spi',
ms[aux0:aux1, c0:c1], ...)` in place of `metric_v[aux0:aux1, v0:v1]`,
`_chol_d4_d5`, and `_dfaux.trmm_left_lower`.  Stage B takes `L` instead of
`eigh` when the factor is used (§3), which is why B is 0.2 s faster.

The factor is used when the metric is the full-range one (`omega == 0`),
spherical, positive definite, one occupied batch, W17's Gram form, and
`FASTDFHESSIP_METRIC=chol` (the default).  The long-range operator of a
range-separated functional, whose metric upstream may truncate, keeps the
eigenbasis.

### 2.3 Memory, and the falsification criterion

The entry: *falsified if the block only fits where the chunked form is
already one chunk*.  The resident block is `3·naux_sph·nao·nocc` words and
the one-chunk eigen pass holds `3·metric_size·nao·nocc` — the same number
when nothing is truncated.  So the criterion fires, in the sense that the
two conditions are one condition.  Its premise does not hold: the entry
priced the resident block at 34 GB "instead of one metric chunk of it", and
at 113 atoms on an 80 GB card there already is only one chunk
(`vchunk=5598`).  Past that size — none measured — the factor still serves
every line of upstream's chunked second pass as its `V` (`chol_chunked` in
`stats()`), at the eigenbasis's cost, and `tests/test_dfhessip.py` drives
that path (`FASTDFHESSIP_MEM` down to 10.9 MB, 48 chunks).

---

## 3. The factor has to be the only factor

### 3.1 What the first version got wrong

The first version used `L` in D3–D5 and left stage B (`dm_oo`), the `h_aux`
block (`j2c_inv`, `j2c_10v`, `V/w`) and the ip2 kernel's pseudo-density on
upstream's eigenbasis.  It passed every test at 7e-12 on ethanol and
glycine, and on the Ni(II) dithiolene dianion it differed from upstream by
**1.7e-9** relative (8.3e-8 Eh/bohr² in the largest element) where the
eigenbasis path, reassociated, differs by 7e-12.

A difference between two implementations does not say which is off.
`benchmarks/df_hess_metric_sens.py` does: it multiplies the metric
element-wise by `1 + 2.2e-16 r` before either path factorises it and
compares each path with itself.

| Ni(II) dithiolene dianion, metric condition 1.6e9 | vs upstream | moved by a 1-ulp metric perturbation (3 seeds) |
|---|---|---|
| eigenbasis (W17) | 4.3e-12 | 1.8e-11, 1.3e-11, 1.8e-12 |
| `FASTDFHESSIP_METRIC=mixed` — the factor in D3–D5 only | **1.7e-9** | **2.4e-9, 1.9e-9, 2.1e-9** |
| the factor everywhere (shipped) | **3.2e-11** | 3.6e-11, 5.9e-11, 5.6e-11 |
| the factor everywhere, `FASTDFHESSIP_TRI=solve` | 3.2e-11 | 3.6e-11, 5.9e-11, 5.6e-11 |

The mixed version carries 2e-9 of the metric's round-off in its answer and
the eigenbasis version does not, so the error was the mixed version's.  The
terms of this function cancel: the D5 cross terms, the `h_aux` block and
the stage-B density each contain `J⁻¹`, and what they sum to is much smaller
than each.  Any single factorisation of `J` rounds `J⁻¹` by `cond(J)·eps`
in its worst directions, and within one factorisation those errors cancel
with the terms; between two they do not.

### 3.2 The fix, and what it rules out

Upstream's arithmetic never uses the eigenbasis as an eigenbasis: every
line is `V W⁻¹ Vᵀ` in some grouping.  So `V = C L⁻ᵀ` (`_SphAux.factor`) and
`W = 1` make every other line of the call — stage B, the `h_aux` block, the
j-part of D4 — use the same `J⁻¹` as D3–D5 without a line of theirs
changing.  The last row of the table is the other suspect: the explicit
`L⁻¹` multiply, replaced by a triangular solve against `L`, gives the same
numbers to the digit, so the explicit inverse costs nothing here.

The shipped version is 3.2e-11 from upstream on the Ni complex and 2–5e-12
on the organic cases (§5), against `tests/test_dfhessip.py`'s 1e-10 bar;
`FASTDFHESSIP_METRIC=mixed` is kept so that the table can be re-measured.

---

## 4. Items 9–11

* **Item 9, the `h_aux` block's two Gram matrices as `dsyrk`.**
  `h_aux = contract('xrij,ysij->xrys', j3c_oo1p, j3c_oo1p)` — `3 naux` rows
  and columns, `nocc²` deep — and `dm_aux` are symmetric; cuBLAS `dsyrk`
  computes one triangle and a small kernel mirrors it (`_gram`).
  `docs/README_fastdf.md` §3.2 found `dsyrk` slower than the GEMM below 800
  rows; these are 7770–19794.  D2 4686 → 3442 ms at 113 atoms, where
  `FASTDFHESSIP_SYRK=0` costs 1.36 s on the call; nothing measurable at 21–42
  atoms (§5).
* **Item 10, D4's `R` as `dsyrk`, and `Q` without its second GEMM.**
  `R[(x,u),(y,v)] = sum_{r,i} Y[x,r,u,i] Y[y,r,v,i]` is `A Aᵀ` for
  `A = Y` laid out `(x,u,r,i)`; each chunk is copied into that layout (a
  pass over memory) and `dsyrk`'d.  In the same layout `M'[x,u,r,v]` is one
  GEMM, and `Q = sum_r M'[x,u,r,v] M'[y,v,r,u]` reads the transpose through
  shared memory (`_HADAMARD_T`) where W17 formed it with a second GEMM.
  D4 4168 → 2082 ms at 113 atoms; `FASTDFHESSIP_D4=gemm`, which is W17's,
  costs 1.75 s on the call there.
* **Item 11, the `h_aux` block in the integral buffers.**  Item 3 fuses the
  first `ipaux` pass into the second pass's first chunk, and holds back
  `24 naux²` words for the `h_aux` block that then runs inside it; at 113
  atoms that is 8.4 GB, the fused plan came out at two chunks
  (`plan_fused=2800` of 5598), and the fusion was declined, costing the
  2.4 s first pass.  But by the time the block runs, the chunk's integral
  buffers — at least `6·batch_size·nao_pair` words, 6.9 GB at 113 atoms,
  whatever the plan — are finished with, and the block's `naux²`
  temporaries (15–16 of the 24) fit in them (`_h_aux_block(work=...)`).  The
  margin is then `10 naux²`, taxol's fused plan is one chunk
  (`plan_fused=5598`), and the first pass costs 1.3 s inside the second
  instead of 2.4 s on its own: **1.0 s** on the call (`FASTDFHESSIP_CARVE=0`,
  §5).

---

## 5. What it is worth

Function level, round robin, min of 3 (1 at 113 atoms); "W17" as defined
above:

| system | basis | GPU4PySCF | W17 | this round | vs W17 | vs GPU4PySCF | rel. to upstream |
|---|---|---|---|---|---|---|---|
| penicillin V, 42 atoms | def2-SVP | 4427 ms | 3023 | **2891** | 1.05x | **1.53x** | 3.8e-12 |
| Pt(II) amine complex, 23 atoms, ECP | def2-SVP | 1422 | 1195 | **1141** | 1.05x | 1.25x | 4.4e-12 |
| Ni(II) dithiolene dianion, 21 atoms | def2-SVP | 2967 | 2580 | **2438** | 1.06x | 1.22x | 3.2e-11 |
| penicillin V | def2-TZVPP | 12413 | 7292 | **6582** | 1.11x | **1.89x** | 5.0e-12 |
| taxol, 113 atoms, items 8–10 | def2-SVP | 63080 | 33339 | **26935** | **1.24x** | **2.34x** | 9.7e-12 |
| taxol, all four items (another process: 25844 with item 11 off) | | | | **24845** | | ≈ 2.5x | |

and the ablations (each item off on its own, same process):

| | penicillin V | Pt | Ni | taxol |
|---|---|---|---|---|
| all on | 2891 | 1141 | 2438 | **24845** |
| `METRIC=eigen` (item 8 off) | 2969 | 1187 | 2518 | 28913 (+4.07 s) |
| `SYRK=0` (item 9 off) | 2873 | 1136 | 2468 | 26204 (+1.36 s) |
| `D4=gemm` (item 10 off) | 2896 | 1133 | 2470 | 26600 (+1.75 s) |
| `T2SPH=0` | 2841 | 1128 | 2418 | 25237 (+0.39 s) |
| `CARVE=0` (item 11 off) | fuses anyway | fuses anyway | fuses anyway | 25844 (+1.00 s) |

(the three smaller systems in one job, taxol's column in another; item 11
changes nothing where the fusion was never declined.)

At 21–42 atoms every item is within 2 % of the others and of noise: the
contractions this round rearranges were 0.6 s of 3.0 at 42 atoms, and the
integral kernels — `ejk_int3c2e_ip2`, the `ip1`/`ipaux` fills and the plain
`int3c2e` — are 2.0 s of the 2.9 left.

Whole Hessians, `benchmarks/df_hess_anatomy.py --rounds 2 --patch all`, W17
and this round in alternating processes, the pair run twice, round 1 of each
(`DF_WORK_LIST.md` §5.3); the two pairs agree to 0.1 s and the table is the
second:

| system | basis | W17 | this round | | ip2 term, W17 → here | traces, W17 / here |
|---|---|---|---|---|---|---|
| penicillin V, 42 atoms | def2-SVP | 15.53 s | **15.26 s** | 1.02x | 3.02 → 2.84 s | 41.7858224565 / …571 |
| Pt(II) amine complex | def2-SVP + ECP | 6.60 s | 6.61 s | 1.00x | 1.22 → 1.16 s | 20.8027830586 / …596 |
| Ni(II) dithiolene dianion | def2-SVP | 8.60 s | **8.48 s** | 1.01x | 2.58 → 2.45 s | 21.9590103867 / …991 |
| penicillin V | def2-TZVPP | 46.18 s | **45.68 s** | 1.01x | 7.05 → 6.49 s | 39.7773170376 / …384 |

This is what the entry's own numbers predicted: the ip2 term is 15–30 % of
a density-fitted Hessian after W17, and at these sizes most of it is the
kernels.  At 113 atoms the term goes 33.3 → 24.8 s of a 701 s Hessian whose
remainder is W8's `_get_jk` (72 %), so the whole run was not re-measured.
The Ni traces differ by 1.2e-8 over 63 diagonal elements — the Cholesky
and eigen roundings of a metric of condition 1.6e9, §3 — against the
`tests/test_dfhessip.py` bar of 1e-8 per element.

---

## 6. The kernel, class by class

`FASTDFHESSIP_TIME=2` times `ejk_int3c2e_ip2` per auxiliary batch with CUDA
events; a batch is one angular momentum and contraction pattern.

| aux `l` | penicillin V | | Pt(II) complex | | Ni(II) complex | | taxol | |
|---|---|---|---|---|---|---|---|---|
| | ms (%) | functions | ms (%) | functions | ms (%) | functions | ms (%) | functions |
| 0 | 154 (12) | 279 | 44 (7) | 149 | 68 (5) | 228 | 473 (11) | 722 |
| 1 | 304 (23) | 693 | 93 (15) | 375 | 182 (14) | 555 | 827 (20) | 1794 |
| 2 | 466 (35) | 918 | 135 (22) | 558 | 255 (20) | 714 | 1478 (36) | 2382 |
| 3 | 224 (17) | 340 | 121 (20) | 250 | 214 (17) | 490 | 700 (17) | 770 |
| 4 | 166 (13) | 360 | 105 (17) | 225 | 314 (25) | 375 | 638 (16) | 930 |
| 5 | | | 49 (8) | 63 | 103 (8) | 63 | | |
| 6 | | | 75 (12) | 56 | **137 (11)** | **28** | | |
| total | 1313 | | 622 | | 1274 | | 4116 | |

Per auxiliary function the cost is 0.30 ms at `l = 0` and **4.9 ms at
`l = 6`** on the Ni complex, and the metal's `h` and `i` functions — 3.7 %
of its auxiliary set — are 19 % of the kernel.  On the organic molecules
the weight sits at `l = 2`, the most numerous class.  Nothing outside the
kernel changes what it does per shell triple: W17 swept its launch
constant and its `gout_width` (±3 %).  What would move it is what moved
W19's kernels — pair data from a table, the Hermite indices as literals,
one thread per triple where it fits — on a kernel that computes second
derivatives, which is a generator of its own.  That is **W29**.

---

## 7. Correctness

`tests/test_dfhessip.py`, element-wise against GPU4PySCF's own function in
one process, relative to the largest element; every item's switch is a
column of its own:

| | worst |
|---|---|
| ip2, ethanol and water, B3LYP / HF / `k = 0` factors, each switch alone | **5.6e-13** |
| the same, `METRIC=eigen` (W17) | 1.9e-13 |
| the long-range operator, `omega = 0.3` (eigenbasis by design) | 4.7e-14 |
| memory-short paths: 2–48 chunks with the factor as `V`, fused and not, 2–3 occupied batches | 5.0e-13 |
| a whole ethanol DF-B3LYP Hessian, patched against unpatched | **4.0e-12 Eh/bohr²** |

The chunked path with the factor (`chol_chunked` in `stats()`) and the
one-chunk structured path (`chol`) are both reached.  §5's last column is
the same comparison on the benchmark molecules.

---

## 8. Tools

```bash
python tests/test_dfhessip.py [--no-hessian]
python benchmarks/df_hess_terms_probe.py --xyz ... --terms ip2 --variants stock,ours \
    --env 'w17:FASTDFHESSIP_METRIC=eigen;FASTDFHESSIP_SYRK=0;FASTDFHESSIP_D4=gemm'
FASTDFHESSIP_TIME=1 python benchmarks/df_hess_terms_probe.py ...   # §1's table
FASTDFHESSIP_TIME=2 python benchmarks/df_hess_terms_probe.py ...   # §6's, per aux l
python benchmarks/df_hess_metric_sens.py --xyz benchmarks/molecules/tmc_ni_dithiolene.xyz --charge -2
```

Switches added by this round (read at call time): `FASTDFHESSIP_METRIC`
(`chol`, `eigen`, `mixed`), `FASTDFHESSIP_TRI` (`mult`, `solve`),
`FASTDFHESSIP_SYRK`, `FASTDFHESSIP_D4` (`syrk`, `gemm`),
`FASTDFHESSIP_T2SPH`, `FASTDFHESSIP_CARVE`, and `FASTDFHESSIP_TIME=2`.
