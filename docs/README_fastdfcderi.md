# fastdfcderi — the cderi build: the metric contraction, the launch configuration, and the page-locked buffer nobody was timing

`docs/DF_WORK_LIST.md` **W4**, the fourth item on that list.  It closes, and
the item's headline lead is the one that loses.

W4 names `df/df.py:_cholesky_eri` — the one-time three-centre build every
density-fitted run makes before its first SCF cycle, **39.4 %** of a 15-cycle
taxol DF SCF and 12–23 % of the shorter ones — and gives two leads.  The
first is the metric contraction: it is a dense `npair × naux_sorted × naux`
GEMM against a matrix that is a triangular factor folded with contraction
coefficients that are *0.06 % dense*, so it should be a sparse gather and a
triangular solve.  The second, handed over by W3, is the `nsp_per_block*4`
shell-pair aggregation the integral kernel underneath is launched with.  The
item's instruction for both is the same: measure them **on the build**, not
on a kernel in isolation.

Measured on the build, lead one is rejected by its own falsification
criterion and then rescued by a different cuBLAS routine, lead two is worth
what W3 said it would be, and the largest single item in the whole function
turns out to be neither: it is `cudaHostAlloc`.

---

## 0. Summary

* **Where the time is.**  `benchmarks/df_cderi_probe.py` runs a traced clone
  of `_cholesky_eri` — checked against the real function on every run — and
  the split moves with system size.  The three-centre integrals are 92 % of
  the loop at 10 atoms and **24 %** at 284; the metric contraction is 8 %
  and **60 %**.  §2.
* **The metric contraction, as W4 proposes it, does not pay.**  W4 attaches
  its own falsification — *"`dtrsm` at these shapes runs at less than half of
  `dgemm`"* — and that is exactly what happens: cuBLAS's `dtrsm` reaches
  **1.1–10.9 TF/s** against `dgemm`'s 12.4–19.0 across the whole size range
  measured, never better than 0.66 of it.  Sparse gather and solve is
  **0.37x–1.76x** on the contraction, and below 2000 auxiliary functions it
  is a loss.  §4.3.
* **The same algebra with a triangular *multiply* is 1.9x–2.1x.**  The
  product `C L^{-T}` is formed once and explicitly — which is no more
  conservative than upstream, whose `_decompose_j2c` solves for the same
  matrix — and cuBLAS's `dtrmm` then runs the per-block half at **15.4–16.8
  TF/s**, GEMM throughput, for half the arithmetic.  With the contraction
  coefficients applied by a one-pass sparse kernel, the metric contraction is
  **2.08x** at 42 atoms, **2.05x** at 21, **1.94x** at 113 and **1.90x** at
  284.  §4.2.
* **The integral kernel is launched four shell pairs deep where one is
  faster**, the same constant and the same direction as W3's — now measured
  on a whole build rather than on the kernel: **1.00x–1.50x** on the loop,
  growing with system size, and `1` is the best of `{1,2,4,8}` on every
  system measured.  §3.
* **And 65 % of a host-resident 113-atom build is `cudaHostAlloc`.**
  `empty_mapped` page-locks the cderi's host buffer at **2.09 GB/s**, so
  taxol's 14.31 GB costs **6.8 s** and PfPMT's 86.41 GB costs **41.3 s** — of
  a 72.5 s build.  Page-locking *pre-allocated* memory with
  `cudaHostRegister` gives the same mapped, pinned buffer at **8.6 GB/s**:
  1.70 s and 10.1 s, a **4.1x** on the single largest stage of a large build.
  It is the allocation that is slow, not the pinning.  This is not a lead W4
  names; it is what timing the function stage by stage turned up.  §5.
* **What it is worth**, on `_cholesky_eri` itself, every variant round robin
  in one process (§6):

  | | cderi | GPU4PySCF | + fastdfcderi | |
  |---|---|---|---|---|
  | penicillin V, def2-SVP (42 atoms) | device | 166.4 ms | **128.1 ms** | **1.30x** |
  | penicillin V, def2-TZVPP | device | 589.8 ms | **339.6 ms** | **1.74x** |
  | taxol, def2-SVP (113 atoms) | device | 3208 ms | **1751 ms** | **1.83x** |
  | taxol, def2-SVP | host | 12764 ms | **3990 ms** | **3.20x** |
  | PfPMT, def2-SVP (284 atoms) | host | 87.1 s | **29.5 s** | **2.95x** |
  | Ni dithiolene, def2-SVP (21 atoms, open shell) | device | 116.8 ms | **102.1 ms** | **1.14x** |

  and on the whole SCF the build sits in, **1.05x** (penicillin V,
  def2-TZVPP), **1.11x** (taxol, cderi on the device) and **1.37x** (taxol,
  cderi on the host, which is the residency a larger molecule gets).
* **It is the same tensor.**  Patched against stock, element-wise in one
  process over six molecule/basis pairs, both residencies, both aggregations
  and all three contraction routes: **6.5e-12 relative**, and a density-fitted
  SCF energy unchanged to 2e-13 Eh.  §7.

---

## 1. The function

`df/df.py:_cholesky_eri` builds the three-index tensor every density-fitted
calculation contracts against:

    cderi[L,pq] = sum_v (pq|v) aux_coef[v,L],   aux_coef = C L^{-T}

where `v` runs over the **sorted, Cartesian, decontracted** auxiliary basis
the Rys kernel evaluates in, `C = auxmol.ctr_coeff` is the
Cartesian-to-spherical and recontraction matrix that takes it to the
`naux` functions the cderi is indexed by, and `L` is the Cholesky factor of
the fitting metric `(u|v)`.  `df/df.py:105` is its only caller, and it looks
it up on the module, so one patch target covers every density-fitted run.

The build has four stages, and this module is on three of them:

1. `_decompose_j2c` — the two-centre integrals, the Cholesky factorisation,
   and the solve that forms `C L^{-T}`;
2. the host buffer, when the cderi does not fit on the device;
3. per block of shell pairs, the three-centre integrals (`fill_int3c2e`
   through `int3c2e_evaluator`);
4. per block, the metric contraction, and either a write into the device
   tensor or a `transpose_write` to the host.

---

## 2. Where the time is

`benchmarks/df_cderi_probe.py`, GPU4PySCF's own configuration, one A100 of a
whole node.  The clone is checked against `_cholesky_eri` element-wise on
every run and agrees to **0.0** — it is the same arithmetic in the same
order, with a synchronising timer between stages.

| | atoms | nao | naux | naux_sorted | `eval_j3c` | metric | store |
|---|---|---|---|---|---|---|---|
| glycine, def2-SVP | 10 | 95 | 471 | 557 | 91.7 % | 8.2 % | — |
| Pt amine, def2-SVP | 23 | 249 | 1358 | 1676 | 75.1 % | 24.9 % | — |
| Ni dithiolene, def2-SVP | 21 | 327 | 1992 | 2453 | 77.8 % | 22.2 % | — |
| alanine dipeptide, def2-TZVPP | 22 | 478 | 974 | 1146 | 82.1 % | 17.9 % | — |
| penicillin V, def2-SVP | 42 | 430 | 2191 | 2590 | 63.9 % | 36.1 % | — |
| penicillin V, def2-TZVPP | 42 | 1007 | 2191 | 2590 | 57.1 % | 42.9 % | — |
| taxol, def2-SVP | 113 | 1123 | 5598 | 6598 | 52.0 % | 48.0 % | — |
| taxol, def2-SVP, **host** | 113 | 1123 | 5598 | 6598 | 39.4 % | 36.5 % | 24.0 % |
| PfPMT, def2-SVP, **host** | 284 | 2697 | 13365 | 15761 | 24.5 % | **59.5 %** | 16.0 % |

Percentages are of the block loop, which is not the whole build: the two
host-resident rows above spend **6.8 s** and **41.5 s** page-locking the
buffer they write into, against block loops of 3.4 s and 30.4 s.  That stage
is §5 and it is the largest one in the function at both sizes.

Three things follow.

**The metric contraction grows against the integrals.**  It is `2 npair
naux_sorted naux` flops where the integral kernel's cost grows more slowly
with the auxiliary basis, so the fraction rises monotonically with system
size — 8 % at 10 atoms, 36 % at 42, 48 % at 113, 60 % at 284.  W4's instinct
that this is where to look is right; what it proposes doing about it is not.

**The setup stages are not negligible at the small end.**  `_decompose_j2c`
is 20–62 ms across these systems — 40 ms at 23 atoms against a 27 ms block
loop — because it is two cuSOLVER calls and a cuBLAS solve whose cost is
mostly latency.  That is why this module's own setup has a threshold on it
(§4.4) rather than being switched on unconditionally.

**And the store is not free once the cderi is on the host.**
`transpose_write` is 24 % of taxol's host-resident block loop and 16 % of
PfPMT's, on one host thread, with no overlap with the next block's
integrals.  Nothing here touches it; §9.

---

## 3. The launch configuration

`int3c2e_evaluator` aggregates shell pairs at `nsp_per_block[0]*4`
(`df/int3c2e_bdiv.py:176`), so each thread block loops four times over shell
pairs instead of once and the device runs a quarter of the blocks it could.
It is the same constant W3 found costing the gradient's derivative kernel
1.26x–3.15x (`docs/README_fastdfgrad.md` §4.2), and W4 records the sweep of
it on the *plain* kernel.  Here it is swept on the whole build, which is what
W4 asks for — the build has a metric contraction and, on the host branch, a
PCIe write on top of the integrals, so a kernel ratio does not transfer.

Block loop, milliseconds, `sp_mult` as a multiple of the scheme's
`nsp_per_block`:

| | 1 | 2 | 4 (GPU4PySCF) | 8 | 4 → 1 |
|---|---|---|---|---|---|
| glycine, def2-SVP | 3.17 | 3.17 | 3.17 | 5.16 | 1.00x |
| Pt amine, def2-SVP | 26.4 | 27.0 | 27.1 | 28.4 | 1.02x |
| Ni dithiolene, def2-SVP | 55.8 | 59.4 | 59.0 | 59.0 | 1.06x |
| penicillin V, def2-SVP | 100.4 | 110.9 | 119.4 | 120.0 | **1.19x** |
| penicillin V, def2-TZVPP | 390.7 | 417.0 | 523.0 | 612.2 | **1.34x** |
| alanine dipeptide, def2-TZVPP | 49.8 | 66.9 | 68.6 | 75.1 | **1.38x** |
| taxol, def2-SVP, host | 2552 | 2930 | 3408 | 3593 | **1.34x** |
| taxol, def2-SVP, device | 1740 | — | 2604 | — | **1.50x** |
| PfPMT, def2-SVP, host | 25620 | — | 30328 | — | **1.18x** |

`1` is the best of the four on every system, the margin grows with size, and
the cderi is identical at every value — which is what says this is a launch
configuration and not an approximation.  The probe checks that on every row.

The constant lives inside upstream's evaluator, so the module reaches it by
intercepting what the evaluator hands `SortedMole.aggregate_shl_pairs`, for
the duration of one call, and asserting that what it sees is the scheme's own
array times four.  If upstream ever changes the constant the assertion fires
rather than the module quietly changing nothing.

---

## 4. The metric contraction

### 4.1 The algebra

Upstream forms `aux_coef = C L^{-T}` densely — `_decompose_j2c` solves
`L X = C^T` — and then spends one dense GEMM per block:

    upstream    j3c . (C L^{-T})     2 npair naux_sorted naux   flops
    here        (j3c . C) L^{-T}       npair naux^2 + sparse    flops

`C` is the Cartesian-to-spherical and recontraction matrix of the auxiliary
basis, and a spherical function is a short combination of the Cartesian ones
of its own shell: measured, **0.010–0.28 %** of its entries are non-zero
(20 694 of 15 761 × 13 365 at 284 atoms; 3 412 of 2 590 × 2 191 at 42), about
1.5 per column.  So the first factor is a gather, and the arithmetic ratio is
`2 naux_sorted / naux` — **2.36x** at def2-SVP — *provided the triangular
routine runs at the GEMM's throughput*.

The associativity is the only thing that changes.  Both forms are the same
tensor in exact arithmetic; §7 measures what the difference in rounding is
worth.

### 4.2 The components

`benchmarks/df_cderi_probe.py` prices each piece separately, on one block of
the build, in achieved TF/s:

penicillin V, def2-SVP, 60 845 pairs × 2590 → 2191:

| | | |
|---|---|---|
| `j3c . (C L^-T)` dense — GPU4PySCF | 43.7 ms | 15.8 TF/s |
| `j3c . C` — this module's sparse kernel | 2.7 ms | |
| `j3c . C` — GPU4PySCF's own `mat_dot_C` | 9.6 ms | |
| `. L^-T` — `dtrsm` | 39.2 ms | 7.5 TF/s |
| `. L^-T` — `dtrmm` against an explicit `L^{-1}` | 18.4 ms | 15.8 TF/s |
| `. L^-T` — `dgemm` against an explicit `L^{-1}` | 32.1 ms | 18.2 TF/s |

and the same table on taxol (45 683 × 6598 → 5598) gives 179.9 ms for the
dense product, 5.8 ms for the sparse gather, 123.5 ms for `dtrsm`, 87.0 ms
for `dtrmm` and 167.3 ms for the dense `dgemm` against `L^{-1}`.

Put together, against upstream's one GEMM:

| | glycine | Pt amine | ala/TZVPP | Ni dithiolene | penicillin V | taxol | PfPMT |
|---|---|---|---|---|---|---|---|
| naux | 471 | 1358 | 974 | 1992 | 2191 | 5598 | 13365 |
| sparse + **`dtrmm`** | 0.95x | **1.51x** | **1.53x** | **2.05x** | **2.08x** | **1.94x** | **1.90x** |
| sparse + `dtrsm` (W4's proposal) | 0.37x | 0.66x | 0.56x | 0.97x | 1.05x | 1.39x | 1.76x |
| sparse + `dgemm` on `L^{-1}` | 0.85x | 1.05x | 1.01x | 1.21x | 1.26x | 1.04x | 1.15x |
| `mat_dot_C` + `dtrsm` | 0.19x | 0.57x | 0.45x | 0.83x | 0.91x | 1.27x | 1.68x |

Three readings.

**The solve is the problem, not the algebra.**  `sparse + dtrsm` is W4's
proposal exactly, and it is at or below break-even everywhere it matters.
`sparse + dtrmm` does the *same arithmetic* against the same factor, differing
only in whether the triangular matrix is inverted once in advance, and is
1.9x–2.1x.

**Inverting `L` is not a liberty upstream does not take.**  `_decompose_j2c`
already solves `L X = C^T`, which for a matrix with one non-zero per column
*is* a column-by-column inversion; and it is a *larger* solve than this
module's, `naux_sorted` right-hand sides against `naux`.  §7 measures the
answer, and the difference is 6.5e-12 relative.

**GPU4PySCF's own sparse apply is not the one to use here.**  `mat_dot_C`
(the `ket_sorted2sph` kernel behind `apply_CT_dot`) is 2–3.5x slower than the
one-pass kernel this module uses, because it zeroes its output first, rebuilds
five index arrays on the device per call, and — the reason it cannot be used
at all without giving something back — it expects the auxiliary axis in its
natural order, where `int3c2e_evaluator` writes it in the interleaved layout
`reorder_aux` produces.  Asking the evaluator for the natural order instead
costs the integral kernel more than the contraction saves.  The module's
kernel folds `aux_sorting` into the column indices, so the integral kernel
keeps the layout it prefers and nothing is permuted per block.

### 4.3 Why `dtrsm` loses, as a function of size

The molecules above conflate the order of the triangular matrix with
everything else about the system.  The routines can be measured on their own:
`benchmarks/df_cderi_probe.py --trmm-ladder` times all three on synthetic
matrices of a fixed 400 MB footprint.

| naux | `dtrmm` | `dtrsm` | `dgemm` | `dtrmm` at equal work |
|---|---|---|---|---|
| 256 | 7.9 TF/s | 1.1 TF/s | 12.4 TF/s | 0.64 |
| 512 | 10.5 | 2.1 | 14.0 | 0.75 |
| 768 | 11.6 | 2.9 | 14.3 | 0.81 |
| 1024 | 15.7 | 4.6 | 18.6 | 0.84 |
| 1536 | 16.6 | 6.1 | 18.9 | 0.88 |
| 2048 | 17.3 | 7.2 | 18.9 | 0.91 |
| 3072 | 17.8 | 9.0 | 19.0 | 0.94 |
| 5598 | 17.3 | 10.9 | 16.3 | 1.06 |

The last column is `t(dgemm)/t(dtrmm)/2` — what the multiply achieves per
flop relative to the GEMM.  `dtrmm` is within 16 % of GEMM throughput from
1024 functions up and reaches it by 3072; `dtrsm` never exceeds **0.66** of
it, and at the small end it is a factor of ten behind.  This is the same trap
`dsyrk` fell into in `docs/README_fastdf.md` §1, with the same shape: the
routine that does half the arithmetic is not the routine cuBLAS tunes.

### 4.4 The threshold

What the module does to the contraction costs about a millisecond of setup —
one triangular inversion, one pass over `C` to extract its non-zeros — and
saves about half of the contraction.  On a build whose contraction is a
couple of milliseconds that is a net loss, and two systems measure it:

| | `npair × naux_sorted × naux` | whole `_cholesky_eri`, metric route only |
|---|---|---|
| glycine, def2-SVP | 1.2e9 | **0.97x** |
| alanine dipeptide, def2-SVP | 2.0e10 | **0.99x** |
| Pt amine, def2-SVP | 6.1e10 | 1.01x |
| Ni dithiolene, def2-SVP | 1.8e11 | **1.11x** |
| penicillin V, def2-SVP | 3.4e11 | **1.10x** |

so `_MIN_WORK` is set at **4e10**, between the largest loss and the smallest
gain, and below it the dense product is kept.  The aggregation of §3 has no
threshold: it is never a loss on any system measured.

---

## 5. The page-locked buffer

When the cderi does not fit on the device — which is any molecule past about
60 atoms in def2-SVP on a 40 GB card, and the only option at 284 —
`_cholesky_eri` allocates it with `empty_mapped`, a single `cudaHostAlloc`
with `hostAllocMapped`.  Timed, on the machine the rest of this page is
measured on:

| | penicillin V, 1.07 GB | taxol, 14.31 GB | PfPMT, 86.41 GB | |
|---|---|---|---|---|
| `cudaHostAlloc` (GPU4PySCF) | 515 ms | 6.84 s | 41.26 s | 2.07–2.09 GB/s |
| `cudaHostRegister`, 4 kB pages | **124 ms** | — | — | 8.62 GB/s |
| `cudaHostRegister` + `MADV_HUGEPAGE` | **123 ms** | **1.70 s** | **10.10 s** | **8.40–8.65 GB/s** |

At 42 atoms that allocation is 56 % of the host-resident build, at 113 atoms
**65 %** and at 284 atoms **57 %** — larger than the integrals and the metric
contraction put together, and it computes nothing.  Nothing in
`docs/DF_WORK_LIST.md` mentions it, because nothing had timed the function
stage by stage; §3.3 of that document measures `_cholesky_eri` at 88.7 s on
this molecule, and about 40 s of that is `cudaHostAlloc`.

The replacement is ordinary memory handed to `cudaHostRegister` with
`cudaHostRegisterMapped`.  It is the same page-locked, device-mapped buffer —
`DF.loop`'s asynchronous prefetch needs page-locked memory and gets it, which
is what the host-resident SCF in `tests/test_dfcderi.py` checks — and it is
**4.1x** faster.

**It is the allocation that is slow, not the page-locking**, and the third
row of the table is what says so: registering *pre-allocated* memory reaches
8.6 GB/s on 4 kB pages, and aligning the buffer to a 2 MB transparent huge
page and asking the kernel for huge pages changes that by **0.3 %**.  The
huge-page advice is kept — on a shared login GPU, where the absolute rates
are a third of these, it was worth 2.21 → 2.94 GB/s, so it is one `madvise`
against a case where it does help — but the factor comes from not going
through `cudaHostAlloc`, which page-locks and *zeroes* a fresh mapping where
`cudaHostRegister` only page-locks one that already exists.
`FASTDFCDERI_HUGEPAGE=0` runs the middle row.

Pinning does not parallelise either way: registering disjoint chunks of a
4 GB buffer from 1, 4, 8, 16 and 32 threads lands within 8 % of the
single-threaded rate, because the kernel serialises on the address space.

The registration has to be released when the buffer goes.  The array that
owns it holds the `_Pinned` object that owns the registration, and every view
of that array holds the array, so the `cudaHostUnregister` happens when the
last view is dropped — including the per-device slices `_cholesky_eri`
returns.

---

## 6. What it is worth

`benchmarks/df_cderi_bench.py`, every variant **round robin** in one process
(a competing job's load drifts over the minutes a row takes and a sequential
schedule puts all of that drift on one variant — `docs/DF_WORK_LIST.md`
§5.2), minimum of three, one A100 of a whole node.  Two consecutive repeats
of the taxol row in one job agree to 0.1 % (1.832x, 1.833x).

### 6.1 `_cholesky_eri`, cderi on the device

| | GPU4PySCF | `sp_mult` | `trmm` | `trsm` | all | |
|---|---|---|---|---|---|---|
| glycine, def2-SVP (10 atoms) | 32.7 ms | 1.007x | 0.972x | 0.959x | 33.6 ms | 0.97x |
| alanine dipeptide, def2-SVP (22) | 43.4 ms | 1.002x | 0.995x | 0.958x | 43.6 ms | 0.99x |
| Pt amine, def2-SVP (23) | 91.1 ms | 1.015x | 1.014x | 0.941x | **89.6 ms** | 1.02x |
| Ni dithiolene, def2-SVP (21) | 116.8 ms | 1.046x | 1.107x | 1.005x | **102.1 ms** | **1.14x** |
| alanine dipeptide, def2-TZVPP (22) | 106.6 ms | 1.057x | 1.022x | 0.912x | **84.8 ms** | **1.26x** |
| penicillin V, def2-SVP (42) | 166.4 ms | 1.132x | 1.102x | 1.001x | **128.1 ms** | **1.30x** |
| penicillin V, def2-TZVPP (42) | 589.8 ms | 1.290x | 1.245x | 0.996x | **339.6 ms** | **1.74x** |
| taxol, def2-SVP (113) | 3208 ms | 1.366x | 1.435x | 1.257x | **1751 ms** | **1.83x** |

The `trmm` and `trsm` columns force `_MIN_WORK` to zero so that the
contraction is priced on every system; the `all` column keeps the threshold,
so it is what a run actually gets.  On the first two rows that is the
aggregation alone — and reading 0.97x and 0.99x rather than the `sp_mult`
column's 1.007x and 1.002x is the size of the noise there, not a cost: a
32 ms build repeats to about ±3 % on this machine, which is the other half
of why the threshold exists.  From 21 atoms up the rows separate by more
than that.

The `trsm` column is W4's proposal on its own, and it is a loss on six of the
eight rows.

### 6.2 `_cholesky_eri`, cderi on the host

This is the residency any large molecule gets, and the one §5's allocation
dominates.

| | GPU4PySCF | `sp_mult` | `trmm` | `trsm` | `pin` | all | |
|---|---|---|---|---|---|---|---|
| penicillin V, def2-SVP (42 atoms) | 918.3 ms | 1.026x | 1.027x | 1.005x | **2.375x** | **337.1 ms** | **2.72x** |
| taxol, def2-SVP (113) | 12764 ms | 1.069x | 1.053x | 1.027x | **2.327x** | **3990 ms** | **3.20x** |
| PfPMT, def2-SVP (284) | 87116 ms | 1.060x | 1.103x | 1.091x | **2.035x** | **29502 ms** | **2.95x** |

The `pin` column is §5 on its own, and on this branch it is worth more than
everything else in this module put together.  The 284-atom row is also the
one measurement here that can be compared with something written before this
round: `docs/DF_WORK_LIST.md` §3.3 measures `_cholesky_eri` at **88.7 s** on
this molecule inside a full SCF, against the 87.1 s of this table's first
column.

### 6.3 The SCF the build sits in

Both rounds, second quoted — the first SCF of a process pays every one-time
cost there is, and a stock-then-patched ordering reads all of it as a
speedup (`docs/DF_WORK_LIST.md` §5.3; the first draft of this table read
1.82x for exactly that reason).

| | GPU4PySCF | + fastdfcderi | | build's share |
|---|---|---|---|---|
| penicillin V, def2-SVP, B3LYP, 14 cycles | 2.17 s | 2.20 s | 0.99x | 8 % |
| penicillin V, def2-TZVPP, 14 cycles | 4.09 s | **3.90 s** | 1.05x | 14 % |
| Ni dithiolene, def2-SVP, UKS, 35 cycles | 4.94 s | **4.88 s** | 1.01x | 2 % |
| taxol, def2-SVP, 15 cycles, cderi on the device | 14.53 s | **13.11 s** | **1.11x** | 22 % |
| taxol, def2-SVP, 15 cycles, cderi on the **host** | 24.27 s | **17.69 s** | **1.37x** | 53 % |

The honest reading of this table is that the cderi build is a one-time cost
and a module on it can only be worth the share it holds.  That share is what
moves with the calculation: 8 % of a small def2-SVP SCF and **53 %** of one
whose cderi has to live on the host, which is every large molecule.  The
energies agree to 5e-11–3e-10 Eh.

---

## 7. Correctness

`python tests/test_dfcderi.py`.  The bar is the tensor itself, element-wise
against GPU4PySCF's own build, both implementations in one process.

What is covered, and why each case is there:

* **every angular-momentum class the auxiliary basis reaches** — def2-SVP's
  jkfit set already carries f functions and def2-TZVPP's g — because the
  Cartesian-to-spherical part of `C` is what the sparse kernel replaces and
  it is the identity on s and p;
* a **Cartesian** orbital basis (6-31G\*) as well as spherical ones;
* an **ECP** case, where the auxiliary basis is the metal's own;
* **both residencies**, because the host branch contracts into a work buffer
  and page-locks its own memory where the device branch writes straight into
  the output;
* **both aggregations**, since a launch configuration may only change how the
  same integrals are distributed over threads;
* **all three contraction routes**, with `_MIN_WORK` forced to zero — every
  molecule small enough to check quickly is below the threshold, so that is
  the only way the suite covers the path at all — and the threshold itself
  checked separately;
* and, end to end, a density-fitted SCF energy against the unpatched one,
  including one with the cderi on the host, which is the only check that
  `DF.loop`'s asynchronous prefetch is still reading page-locked memory.

Worst over that suite: **6.5e-12** relative on the cderi and **2.3e-13 Eh**
on the SCF energy.  The `gemm` route reproduces upstream bit for bit (0.0);
the 6.5e-12 is the associativity of §4.1 — `(j3c C) L^{-T}` against
`j3c (C L^{-T})` — in a fitting metric that `docs/README_fastdfgrad.md` §5.2
measures as conditioned at 2e9 to 2e10.

---

## 8. What it does not cover

* **More than one device.**  A device's slice of the cderi is a column block
  of `L^{-T}`, which a triangular routine cannot take; the module hands
  multi-device builds back to the dense product, which is what
  `docs/DF_WORK_LIST.md` W2 is about in any case.  The aggregation and the
  page-locking still apply.
* **A linearly dependent auxiliary basis.**  `_decompose_j2c` falls back to a
  filtered eigendecomposition, which has no triangular factor; the dense
  product is kept.
* **A contraction smaller than `_MIN_WORK`** (§4.4), where the setup costs
  more than the contraction saves.
* The **`only_dfj`/RIJONX** configuration builds no cderi at all
  (`docs/README_fastdfj.md` §6), so this module never sees it.

`fastdfcderi.stats()` reports which of these happened, because nothing
numerical can.  `benchmarks/coverage_probe.py --density-fit` confirms the
entry point is on the path: one call, answered, per density-fitted SCF.

---

## 9. What is left

* **The store.**  At 284 atoms `transpose_write` is **16 %** of the block
  loop, 4.9 s, and it runs on one host thread with no overlap with the next
  block's integrals.  Nothing here touches it.
* **`_decompose_j2c` at the small end** — 20–62 ms, most of it cuSOLVER
  latency on a matrix of a few thousand rows, against block loops of 3–120 ms
  on the same systems.  It is the reason `_MIN_WORK` exists and it is
  untouched.
* **The remaining half of the metric contraction.**  `dtrmm` at GEMM
  throughput is the floor for this arrangement; below it would need the
  contraction not to be done at all — which is what W6's local fitting would
  buy, and what W5's block sparsity would buy on the exchange side.
* **Pinning at 8.6 GB/s is still 10 s of a 284-atom build.**  It does not
  parallelise across threads (§5) and the only way past it is not to allocate
  a fresh 86 GB buffer per build — a pool across `DF.reset()` and across the
  two builds a range-separated functional makes.

---

## 10. Reproducing

```bash
# the correctness bar
python tests/test_dfcderi.py [--basis def2-tzvpp]

# where the time in `_cholesky_eri` is, stage by stage, with the clone
# checked against the real function
python benchmarks/df_cderi_probe.py --xyz benchmarks/molecules/taxol.xyz \
    --basis def2-svp [--cpu-cderi] [--no-check]

# the two triangular routines and the GEMM, against the auxiliary dimension
python benchmarks/df_cderi_probe.py --xyz benchmarks/molecules/glycine.xyz \
    --basis def2-svp --skip-metric --sp-mults 4 \
    --trmm-ladder 256,512,768,1024,1536,2048,3072,5598

# what each item is worth, round robin in one process
python benchmarks/df_cderi_bench.py --xyz benchmarks/molecules/taxol.xyz \
    --basis def2-svp [--cpu-cderi] [--scf]

# that the entry point is on the path at all
python benchmarks/coverage_probe.py --xyz benchmarks/molecules/penicillinV.xyz \
    --basis def2-svp --density-fit
```

Switches, for the ablation columns above:

    FASTDFCDERI_OFF=1         hand every build back to GPU4PySCF
    FASTDFCDERI_METRIC=trsm   the triangular solve W4 asks for, in place of
                              the multiply -- measured, and rejected
    FASTDFCDERI_METRIC=gemm   upstream's dense product
    FASTDFCDERI_SPMULT=4      GPU4PySCF's shell-pair aggregation
    FASTDFCDERI_MIN_WORK=0    take the sparse contraction at any size
    FASTDFCDERI_PIN=alloc     page-lock the host buffer with `cudaHostAlloc`
