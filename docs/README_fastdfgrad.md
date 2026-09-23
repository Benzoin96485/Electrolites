# fastdfgrad — the DF gradient's two-electron derivative, and the integrals it does not have to evaluate

`docs/DF_WORK_LIST.md` **W3**, the third item on that list.  It closes, and
not in the direction the item expected.

W3 names `df/grad/rhf.py:_jk_energy_per_atom` as the one uncovered stage of
the density-fitting path that *grows* with basis size — **29 %** of a
def2-SVP DF gradient and **83 %** of a def2-TZVPP one — and gives two
instructions.  The first is to profile it stage by stage before assuming
anything, with §2's lesson attached: *do not assume the contraction is the
cost*.  The second is a hypothesis: that the `(L,i,j)` layout §2 rejected for
the SCF build would, here, turn both contraction pairs into clean batched
GEMMs, and should be re-measured rather than inherited.

The profile answers both at once, and the hypothesis is the one that loses.

---

## 0. Summary

* **The contraction is not the cost, and this time it is not even close.**
  `benchmarks/df_grad_probe.py` runs a traced clone of the function — the
  same arithmetic in the same order with a synchronising timer between
  stages — and puts **76–85 %** of it in two Rys kernels: the derivative
  three-centre integrals (52–59 %) and the plain ones (24–26 %).  Every
  contraction, every fancy-index scatter and the `cp.take` together are
  **4.4 %** at 42 atoms and **14.5 %** at 113, and they already run at
  9–16 TF/s of an A100's 19.5 and at 640–1820 GB/s.  So the `(L,i,j)`
  question W3 asks is settled by arithmetic rather than by an experiment: a
  *perfect* version of every contraction in the function would be worth under
  5 % at 42 atoms.  §2 has the three tables.
* **A quarter of the function is integrals the SCF already computed.**  The
  first loop evaluates `(pq|P)` from scratch — and the cderi the SCF built
  holds exactly those integrals, contracted with the fitting metric.
  `jk_energy_per_atom` deletes it, with `with_df.reset()`, on its first line.
  The identity that makes the reuse exact is one line (§3), and the module is
  built on it.
* **And the derivative kernel is launched four shell pairs deep where one is
  faster.**  `df/grad/rhf.py:141` aggregates shell pairs at
  `nsp_per_block*4`, so each thread block loops four times over shell pairs
  instead of once and the device runs a quarter of the blocks it could.
  Swept over `{1,2,4,8,16,32}` and four `gout_width`s, **1** is the fastest
  of the six at every size and basis measured, by a margin that *grows* with
  the system: **1.26x** on the kernel at 42 atoms/def2-SVP, **1.77x** at
  def2-TZVPP and **3.15x** at 113 atoms.  The kernel's answer is identical at
  every `sp_mult`, which is what says this is a launch configuration and not
  an approximation — where `gout_width` above 54 is *not*, and the same sweep
  finds that too.  §4.2.
* **What that is worth**, one `jk_energy_per_atom` call at fixed density,
  every variant round robin in one process (§4.1):

  | | `jk_energy_per_atom` | whole gradient |
  |---|---|---|
  | penicillin V, def2-SVP, B3LYP (42 atoms) | 905 → **518 ms**, 1.75x | 1.240 → **0.858 s**, 1.45x |
  | penicillin V, def2-TZVPP, B3LYP | 2827 → **1315 ms**, 2.15x | 3.317 → **1.651 s**, 2.01x |
  | taxol, def2-SVP, B3LYP (113 atoms) | 7369 → **2850 ms**, 2.59x | 8.172 → **3.604 s**, 2.27x |

* **It is the same number.**  Patched against stock, element-wise, in the
  same process, the two agree to **1.5e-9 relative** at 42 atoms and
  **7.0e-8** at 113, tracking the condition number of the fitting metric
  (2.3e9 and 1.9e10) rather than anything about the reuse.
  `benchmarks/df_grad_metric_probe.py` shows that merely *associating* the
  metric contraction the other way round on GPU4PySCF's own intermediate
  moves it by more, on both systems (§5.2).
* **And one thing the round found rather than fixed.**  At **284 atoms** the
  density-fitted gradient does not run on one 40 GB A100, patched or not:
  `j3c_oo` is a dense `(naux, nocc, nocc)` array and that is **32.0 GB**
  there.  The SCF at the same size runs; its gradient does not.  §6.2, and
  `docs/DF_WORK_LIST.md` **W10**.
  *(**Taken since**, and it does run: `docs/README_fastdfgradoo.md`.  That
  round also corrects the size — the allocation is on the Cartesian sorted
  auxiliary dimension, not the spherical one, so the array is 37.7 GB and
  upstream holds two of them.  The tables in §4 below moved with it, and §4.5
  says by how much.)*
  What separates them is round-off in an ill-conditioned fitting metric, and
  §5 shows that merely *associating* the metric contraction the other way
  round on upstream's own `j3c_oo` moves the answer by as much.

---

## 1. The function, and where it is in a gradient

`df/grad/rhf.py:_jk_energy_per_atom` computes the J and K contributions to
the nuclear gradient of a density-fitted energy, per atom.  `df/grad/rks.py`
and `df/grad/uks.py` both reach it through
`Gradients.jk_energy_per_atom`, which every DF **hybrid** gradient calls once
— and a **range-separated** one calls twice, the second time for the
long-range operator.  A *pure* functional's gradient never reaches it:
`df/grad/rhf.py:68` sends a `k_factor == 0` call to `_j_energy_per_atom`,
which contracts the three-centre integrals with the density directly and
builds no cderi at all.

Its share of the gradient it sits in, measured on one A100-SXM4-40GB
(`benchmarks/df_grad_probe.py --whole`, B3LYP, level-3 grids, minimum of
three):

| | whole gradient | `jk_energy_per_atom` | |
|---|---|---|---|
| penicillin V, def2-SVP (42 atoms, nao 430) | 1.252 s | 0.906 s | 72 % |
| penicillin V, def2-TZVPP (nao 1007) | 3.373 s | 3.133 s | **93 %** |
| taxol, def2-SVP (113 atoms, nao 1123) | 8.062 s | 7.798 s | **97 %** |

W3's 29 %/83 % came from `benchmarks/df_anatomy.py`, which wraps the
module-level function and so misses the `SortedMole` and `Int3c2eOpt` rebuild
the *method* does on every call.  Timed at the method — which is where the
work actually starts — the share is larger, and `df_anatomy.py` now wraps the
method for that reason.

---

## 2. The profile W3 asks for

`benchmarks/df_grad_probe.py`.  The clone is checked against the stock
function on every run and agrees to 1e-13 Eh/Bohr, so the split below is a
split of the real thing and not of a paraphrase.  Minimum of three runs, the
first discarded — the first pays a JIT compile for the ip1 kernel and a
cuTENSOR plan for each of the four contractions, which
`docs/DF_WORK_LIST.md` §5 item 3 is about.

### 2.1 Penicillin V, def2-SVP — nao 454 (decontracted, Cartesian), naux 2590, nocc 92

| stage | ms | % | TF/s | GB/s |
|---|---|---|---|---|
| **ip1 kernel** (3c2e derivatives) | 508.13 | **54.8** | | |
| **eval_j3c** (3c2e integrals) | 224.67 | **24.2** | | |
| setup: `SortedMole` + `Int3c2eOpt.build` | 66.68 | 7.2 | | |
| metric solve (`naux × naux`) | 37.77 | 4.1 | | |
| `int2c2e_ip1` (auxiliary response) | 31.05 | 3.3 | | |
| pseudo-dm 2 `iqr,pi->pqr` (dense) | 10.84 | 1.2 | 9.06 | 393.8 |
| half-transform 1 `pqr,pi->iqr` | 10.44 | 1.1 | 9.41 | 409.0 |
| `dm_oo = metric · j3c_oo` | 8.39 | 0.9 | 13.54 | |
| pair addresses | 6.98 | 0.8 | | |
| **unpack** → dense `(nao,nao,blk)` | 6.50 | 0.7 | | 643.8 |
| pseudo-dm: J term `dm[p,q]·auxvec[r]` | 3.72 | 0.4 | | 1148.4 |
| pseudo-dm 1 `rji,qj->iqr` | 3.71 | 0.4 | 5.36 | |
| half-transform 2 `iqr,qj->rij` | 3.20 | 0.3 | 6.23 | |
| **`cp.take`** → pair list | 3.11 | 0.3 | | 1821.3 |
| ip1 setup | 2.11 | 0.2 | | |
| factorize dm | 0.65 | 0.1 | | |
| | **927.96** | | | |

### 2.2 Penicillin V, def2-TZVPP — nao 1146, naux 2590, nocc 92

| stage | ms | % |
|---|---|---|
| **ip1 kernel** | 1576.46 | **58.6** |
| **eval_j3c** | 697.14 | **25.9** |
| setup | 109.20 | 4.1 |
| pseudo-dm 2 (dense) | 71.24 | 2.6 |
| half-transform 1 | 52.84 | 2.0 |
| metric solve | 38.41 | 1.4 |
| `int2c2e_ip1` | 32.59 | 1.2 |
| unpack | 30.83 | 1.1 |
| J term | 20.99 | 0.8 |
| `dm_oo` | 18.23 | 0.7 |
| `cp.take` | 14.94 | 0.6 |
| the rest | 28.37 | 1.1 |
| | **2692.04** | |

### 2.3 Taxol, def2-SVP — nao 1185, naux 6598, nocc 226

| stage | ms | % |
|---|---|---|
| **ip1 kernel** | 3431.94 | **52.0** |
| **eval_j3c** | 1606.07 | **24.3** |
| pseudo-dm 2 (dense) | 302.35 | 4.6 |
| `int2c2e_ip1` | 274.78 | 4.2 |
| half-transform 1 | 268.46 | 4.1 |
| `dm_oo` | 266.18 | 4.0 |
| metric solve | 122.67 | 1.9 |
| unpack | 65.43 | 1.0 |
| pseudo-dm 1 | 59.45 | 0.9 |
| J term | 56.47 | 0.9 |
| setup | 53.76 | 0.8 |
| half-transform 2 | 53.37 | 0.8 |
| `cp.take` | 32.26 | 0.5 |
| the rest | 8.30 | 0.1 |
| | **6601.49** | |

### 2.4 What the three tables say

**The two Rys kernels are 76–85 % of the function at every size and basis
measured**, and the share of the derivative kernel alone only moves between
52 % and 59 %.  Everything W3 proposed to attack is in the remainder:

* The **two fancy-index scatters** W3 singles out — the `j3c[j_addr,i_addr] =
  j3c[i_addr,j_addr]` unpack and the `cp.take` that compresses the
  pseudo-density back — are **1.0 %** of the function at 42 atoms and
  **1.5 %** at 113, and they run at 644–1821 GB/s against an A100's 1555.
  (`cp.take` above peak is not an error: it gathers `nao_pair × blk` doubles
  out of a buffer that is L2-resident by the time it runs.)  There is nothing
  there, and a kernel of our own would be competing with a `fill_symmetric`
  that `docs/README_fastdf.md` §3.4 already measured at 393 GB/s against our
  own attempt's 148.
* The **four contractions** are 2.7 % / 5.5 % / 14.4 % of the three tables
  and run at 5–17 TF/s.  The largest of them, taxol's `pseudo-dm 2`, is at
  **13.85 TF/s of 19.5** — 71 % of the machine.  W3's `(L,i,j)` hypothesis is
  right in kind: `half-transform 1` *is* already a plain GEMM in the
  `(p,q,r)` layout it runs in (`j3c` viewed as `(nao, nao·blk)` against the
  density factor), and `half-transform 2` is the permuting one.  But
  `half-transform 2` is **0.3 %** and **0.8 %** of the function.  Turning it
  into a batched GEMM would buy a fraction of that, and it would cost the
  unpack its coalescing: the compressed pair list is aux-minor, so writing
  `(L,i,j)` scatters 8 bytes at a time where `(i,j,L)` writes `blk` contiguous
  doubles per pair.  **Re-measured, not inherited — and the answer is the
  same one §2 of the work list reached, for a different reason.**
* The **metric** (`metric solve` + `dm_oo`) is 5.0 % / 2.1 % / 5.9 %, and the
  **auxiliary response** term 3.3 % / 1.2 % / 4.2 %.

So there are exactly two things here that do not require writing a new
integral kernel: **not evaluating `eval_j3c` at all** (§3), and **launching
the ip1 kernel better** (§4.2).  The module is those two.

---

## 3. The identity

`eval_j3c` evaluates `(pq|P)` over every screened AO pair and every auxiliary
function.  The SCF that just converged evaluated the same integrals, and kept
them: the **cderi** is

    cderi[L,pq] = sum_v (pq|v) aux_coef[v,L]

where `aux_coef` is what `df/df.py:_decompose_j2c` returns and
`df/df.py:_cholesky_eri` builds the tensor with.  In the Cholesky branch
`aux_coef = C · L^-T` with `J = L L^T`; in the eigen branch
`aux_coef = C · V · w^-1/2` over the eigenvalues above `LINEAR_DEP_THR`.  In
**both**,

    aux_coef @ aux_coef.T  ==  metric

and `metric` is precisely what `_jk_energy_per_atom` builds for itself — a
Cholesky solve against the contraction coefficients when `mol.omega <= 0 and
not auxmol.mol.cart`, and the same filtered pseudo-inverse otherwise.
Measured on water/def2-SVP, `|metric − aux_coef @ aux_coef.T| / |metric|` is
**3.3e-15** (`tests/test_dfgrad.py` §1).

So with the occupied-occupied half-transform of the cderi,

    rhok[L,i,j] = sum_pq cderi[L,pq] C[p,i] C[q,j]

the pseudo-density the ip1 kernel needs is

    dm_oo = aux_coef @ rhok

exactly — and `rhok` is a sum over the whole auxiliary index, so it does not
care which permutation the SCF sorted the auxiliary functions into.  Only
`aux_coef` does, and it is rebuilt in the gradient's own ordering.  That is
what lets a tensor the SCF built under one auxiliary split feed a gradient
running under another; getting it wrong is the one failure mode here that
does not announce itself, which is why `tests/test_dfgrad.py` §2 is
element-wise over eight cases rather than one.

What the reuse removes is `eval_j3c`, the unpack and the metric *solve*.
What it adds is the half-transform of the cderi — which is the same
arithmetic the function was already doing in `half-transform 1` and `2`, and
on a **smaller** basis: the cderi is contracted and spherical (nao 1007 at
def2-TZVPP) where the gradient's own pair list is decontracted and Cartesian
(nao 1146), so the dominant `2·nao²·nocc·naux` term is 1.29x smaller.  And
`dm_oo = aux_coef @ rhok` replaces `metric · j3c_oo`, a `(naux_cart ×
naux_cart)` product, with a `(naux_cart × naux)` one.

Two details that are not optional:

1. **The `reset()` is partial.**  Upstream's first line is
   `mf.with_df.reset()`, and `reset` empties `_rsh_df` with everything else.
   For a **range-separated** functional that would throw away the long-range
   cderi the SCF built, before the second call could reuse it — and a
   range-separated functional's gradient makes two calls here, one per
   operator.  `_release` frees what has just been consumed and leaves a
   long-range object that still carries a tensor standing until its own call
   consumes it, so both calls reuse and nothing is held for longer than the
   SCF held it.
2. **The `Int3c2eOpt` is the SCF's**, for the full-range call.  `DF.build`
   constructs the same object with the same cutoff, and `reset()` throws it
   away one line before the gradient rebuilds it.  Only the full-range call:
   the screening behind it is the full-range operator's, and a long-range
   call needs its own.

---

## 4. What it is worth

`benchmarks/df_grad_bench.py`.  Every variant is timed **round robin** in one
process on the same converged density — `docs/DF_WORK_LIST.md` §5 item 2 —
and the first pass of each is thrown away, because the first call in a process
pays a JIT compile for the ip1 kernel and a cuTENSOR plan for each
contraction (item 3 of the same list).  One A100-SXM4-40GB,
`OMP_NUM_THREADS=32`, whole node, level-3 grids.

Between rows the cderi is rebuilt **untimed**.  That is the honest
accounting: an SCF hands its gradient a live cderi, and upstream's first line
deletes it.  What the rebuild costs, for the reader who wants it, is in the
last column of §4.4.

### 4.1 One `jk_energy_per_atom` call, at fixed density

Times in ms, minimum of two or three repetitions; the ratio in each cell is
against that row's `stock`.

| | stock | cderi only | + spmult | **fastdfgrad** |
|---|---|---|---|---|
| penicillin V, def2-SVP, B3LYP | 905.1 | 683.6 · 1.324x | 580.6 · 1.559x | **518.4 · 1.746x** |
| penicillin V, def2-TZVPP, B3LYP | 2826.8 | 1933.8 · 1.462x | 1260.7 · 2.242x | **1314.7 · 2.150x** |
| penicillin V, def2-SVP, ωB97X-V | 910.9 | 685.9 · 1.328x | 584.0 · 1.560x | **519.5 · 1.753x** |
| taxol, def2-SVP, B3LYP (113 atoms) | 7369.0 | 5138.8 · 1.434x | 2848.3 · 2.587x | **2850.1 · 2.586x** |
| Ni dithiolene⁻ doublet, def2-SVP, UKS (21 atoms) | 793.4 | 675.6 · 1.174x | 657.2 · 1.207x | **556.8 · 1.425x** |
| Cu macrocycle²⁺ quartet, def2-SVP, UKS (60 atoms) | 1868.5 | 1536.1 · 1.216x | 1099.4 · 1.700x | **973.1 · 1.920x** |

and the **whole analytic gradient** those sit in, same process, same round
robin:

| | stock | **fastdfgrad** | |
|---|---|---|---|
| penicillin V, def2-SVP, B3LYP | 1240.0 ms | **858.1 ms** | **1.445x** |
| penicillin V, def2-TZVPP, B3LYP | 3317.3 ms | **1650.6 ms** | **2.010x** |
| penicillin V, def2-SVP, ωB97X-V | 3405.6 ms | **2743.1 ms** | 1.242x |
| taxol, def2-SVP, B3LYP | 8171.8 ms | **3604.0 ms** | **2.267x** |
| Ni dithiolene⁻ doublet, def2-SVP | 1254.0 ms | **982.4 ms** | 1.277x |
| Cu macrocycle²⁺ quartet, def2-SVP | 2664.0 ms | **1770.8 ms** | **1.504x** |

Three things about this table.

**The ωB97X-V row is the same term twice.**  A range-separated functional's
gradient calls `jk_energy_per_atom` once per operator, and both reuse — which
is what `_release` is for; with upstream's `reset()` the long-range call
would find nothing and fall through, and the whole-gradient column reads
1.124x instead of 1.242x when it does.  Its whole-gradient ratio is small for
a reason that has nothing to do with this module: **77-80 %** of a wB97-family
DF run is the VV10 grid pass (`docs/README_fastdf.md` §5), and the
two-electron derivative is 0.9 s of a 3.4 s gradient rather than 0.9 s of
1.2 s.

**The last two columns are within the noise of each other on the two
penicillin rows**, and their order flips between the per-call and the
whole-gradient column at def2-TZVPP (1260.7/1756.0 against 1314.7/1650.6).
Reusing the SCF's `Int3c2eOpt` cannot cost anything — it skips a rebuild —
but it changes what the memory pool holds when `_ejk` asks
`get_avail_mem` how much is free, and therefore the block size it picks.  The
honest reading of that pair is "1.14x at 42 atoms/def2-SVP, nothing to speak
of past it", which is what a fixed setup cost against a growing term should
look like.

**The two items compose the way the profile says they should.**  At
def2-SVP/42 atoms the reuse removes 24 % of the function and the launch
configuration takes 1.26x off the 55 % that is the ip1 kernel: 1.324x then
1.559x.  At 113 atoms the kernel is 3.15x faster and the same two steps are
1.434x then 2.587x.  Nothing here is a fitted constant.

### 4.2 The launch configuration, swept

`benchmarks/df_grad_ip1_sweep.py` times `sum_ejk_int3c2e_ip1` over
`gout_width × sp_mult`, on a fixed pseudo-density, and checks every cell's
answer against the first.  The `sp_mult = 4` column is GPU4PySCF's.  Times in
ms, minimum of two:

| `gout_width` \ `sp_mult` | 1 | 2 | **4** | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| **penicillin V, def2-SVP** | | | | | | |
| 18 | 464.6 | 556.6 | 587.7 | 587.6 | 587.5 | 587.4 |
| 27 | 417.9 | 494.3 | 524.5 | 524.5 | 524.5 | 524.5 |
| 36 | 414.6 | 491.9 | 523.8 | 523.7 | 523.6 | 523.7 |
| **54** | **403.0** | 468.4 | *506.5* | 506.3 | 506.7 | 506.7 |
| 72 | 392.2 ✗ | 455.2 ✗ | 493.7 ✗ | 493.4 ✗ | 493.7 ✗ | 493.8 ✗ |
| 108 | 392.5 ✗ | 449.0 ✗ | 494.5 ✗ | 493.9 ✗ | 494.4 ✗ | 494.4 ✗ |
| **penicillin V, def2-TZVPP** | | | | | | |
| 18 | 670.3 | 1253.1 | 1873.2 | 1874.1 | 2548.9 | 2548.9 |
| 27 | 602.0 | 1102.2 | 1664.3 | 1661.0 | 2155.4 | 2155.6 |
| 36 | 973.0 | 1147.6 | 1659.4 | 1895.2 | 2159.5 | 2159.5 |
| **54** | **891.1** | 1071.6 | *1576.1* | 1718.7 | 2044.5 | 2051.1 |
| **taxol, def2-SVP** | | | | | | |
| 18 | 1239.7 | 2460.5 | 3970.0 | 4856.2 | 6103.7 | 6099.6 |
| 27 | 1109.6 | 2201.4 | 3536.5 | 4317.0 | 5450.3 | 5446.1 |
| 36 | 1111.2 | 2200.0 | 3535.5 | 4297.0 | 5443.4 | 5436.7 |
| **54** | **1085.7** | 2136.8 | *3422.7* | 4108.3 | 5275.2 | 5278.3 |
| 72 | 1062.0 ✗ | 2090.3 ✗ | 3337.1 ✗ | 3991.2 ✗ | 5155.7 ✗ | 5161.3 ✗ |

✗ = **the kernel's answer changes** — by 89 at `gout_width = 72` and 1.8e3 at
108, against a result whose elements are of order 1.  `gout_width` sets
`gout_stride = nearest_power2(⌈nfi·nfj·nfk / gout_width⌉)`, i.e. how many
threads one shell pair's output is spread over, and past 54 the kernel's
`gout` buffer no longer holds what it is given.  So 54 is not an arbitrary
constant: it is a ceiling, and the sweep finds it.  The *plain* integral
kernel underneath `int3c2e_evaluator` has the same ceiling and reaches it
later — its grid agrees everywhere on penicillin V but breaks at
`gout_width > 54` on taxol, whose auxiliary basis reaches higher — which is
the reminder that a `gout_width` that happens to work on one molecule is not
a `gout_width` that works.

Two things follow.

**`sp_mult = 1` is the fastest of the six in every row of every table**, and
the margin grows with the system: **1.257x** at 42 atoms/def2-SVP, **1.769x**
at def2-TZVPP, **3.153x** at 113 atoms, all at `gout_width = 54`.  It is a
pure launch configuration — four times the thread blocks, each looping over a
quarter as many shell pairs — and the answer is identical across the whole
column, which is what makes it safe to take.  That is the module's second
item.

**`gout_width` is left at 54, deliberately.**  27 is a further 1.48x at
def2-TZVPP (602.0 against 891.1) and 3.7 % *worse* at def2-SVP (417.9 against
403.0): the better value depends on how much angular momentum the basis
reaches, because that is what `nfi·nfj·nfk` is.  Two systems is not enough to
write that rule down, and a constant that is right for an f-shell basis and
wrong for a d-shell one is exactly the kind of dependence this repository is
supposed to audit rather than ship.  `FASTDFGRAD_GOUT` is the knob, the grid
above is the evidence, and the default stays where GPU4PySCF put it.

### 4.3 The item that was measured and rejected

Building the Coulomb term of the pseudo-density on the compressed pair list
does **strictly less work** than upstream's dense arrangement — it writes
`nao_pair` entries where upstream writes `nao²` of which `cp.take` keeps
`nao_pair`, and it lets the exchange contraction run with `beta = 0` instead
of reading the dense buffer back.  It is **0.4–1.4 % slower** on every system
it was tried on.  The dense write lands in a buffer the very next kernel
reads, so it is L2-resident and close to free, while the pair-list version is
an extra kernel launch and an extra temporary per block.  It ships **off**;
`FASTDFGRAD_JPAIR=1` runs it, because a rejected alternative that can be
re-run is worth more than one described in prose (`docs/DF_WORK_LIST.md` §2
makes the same point about three of `fastdf`'s).

### 4.4 A different chemical class, and the cderi the reuse depends on

The six rows above span 21 to 113 atoms, two basis sets, a hybrid, a
range-separated hybrid, closed shell and open shell, and an organic ligand
set and two **transition-metal complexes** — because a speedup that is really
a property of one molecule's shell structure is the failure mode this
repository has hit before (`docs/OPENSHELL_TMC.md` §5).  The unrestricted
rows are not a different code path being *declined*: `df/grad/uhf.py` has its
own `_jk_energy_per_atom` with a spin index, `df/grad/uks.py:77` binds its
own copy of the method, and this module patches all four classes for that
reason.

What the reuse costs, and what it depends on:

| | |
|---|---|
| the cderi it reuses | 0.58 GB (21 atoms) to 14.31 GB (113 atoms), device-resident in every row above |
| what rebuilding it would cost | 0.23 s at penicillin V/def2-SVP, 0.68 s at def2-TZVPP, **3.05 s** at taxol |
| extra memory held | none beyond what the SCF was already holding: the tensor is released as soon as `rhok` is formed, which is before the gradient's own buffers are sized |

The middle row is the one to keep in mind when reading the ratios.  The
reuse is free only because the SCF has just paid for the tensor; a gradient
run against a *reconstructed* density — a scanner that reset the object, or
a cderi that was never built — falls through, and `STATS['rebuilt']` counts
it.

---

## 5. Correctness

### 5.1 What is checked

`tests/test_dfgrad.py`, in one process, against GPU4PySCF's own function on
the same converged density.  The stock reference is reached with
`FASTDFGRAD_CDERI=0`, which puts upstream's first loop back and leaves the
rest of the method byte-for-byte what upstream's is, so the two rows differ
in exactly the thing being tested.

| | |
|---|---|
| `\|metric − aux_coef @ aux_coefᵀ\| / \|metric\|`, water/def2-SVP and def2-TZVPP | **3.3e-15** |
| the same on penicillin V and taxol, def2-SVP (`df_grad_metric_probe.py`) | 2.5e-13 and 8.0e-13 |
| per-atom `ejk`, ethanol/def2-SVP, B3LYP | 1.9e-10 Eh/Bohr |
| ethanol/def2-TZVPP | 1.8e-10 |
| ethanol/def2-SVP, PBE0 / M06-2X / ωB97X-V | 1.7e-10 / 2.5e-10 / 1.4e-10 |
| CH₃ doublet, UKS B3LYP / ωB97X-V | 5.1e-11 / 4.9e-11 |
| ethanol cation doublet, UKS B3LYP | 1.9e-10 |
| ωB97X-V's **long-range** term, ethanol (restricted) | 1.2e-08 |
| the whole analytic gradient, water/def2-SVP | 7.7e-11 |
| `benchmarks/df_grad_bench.py`, per atom and component against stock, on the six systems of §4.1 | 5.6e-08 / 6.3e-08 / 4.6e-08 / 4.1e-06 / 1.0e-07 / 5.2e-07 Eh/Bohr, i.e. **1.5e-9 to 7.0e-8 relative** |

and the gradient against a 3-point **finite difference** of the density-fitted
energy, which is the only check here that does not take GPU4PySCF's answer as
the definition of the right one: the patched and the stock analytic gradients
sit the same distance from it, and that distance is the step error.

### 5.2 Where the last digits go, and why it is not the reuse

The fitting metric is badly conditioned by construction, and the check that
says so is sharper than an appeal to conditioning.  Take **upstream's own
`j3c_oo`** — no cderi anywhere in sight — and contract it with the metric two
ways:

    metric @ j3c_oo                  (upstream's association)
    aux_coef @ (aux_coefᵀ @ j3c_oo)  (this module's)

`benchmarks/df_grad_metric_probe.py` does exactly that:

| | cond(j2c) | \|metric\|max | \|metric − aux_coefaux_coefᵀ\|/\|metric\| | the two associations differ by | and the **whole term** differs by (§4) |
|---|---|---|---|---|---|
| penicillin V, def2-SVP, ω = 0 | 2.26e9 | 2.02e5 | 2.5e-13 | **7.8e-07** in `dm_oo` | 5.6e-08 Eh/Bohr |
| penicillin V, def2-SVP, ω = 0.3 | rank deficient | 4.33e7 | 2.8e-15 | **1.1e-07** | — |
| taxol, def2-SVP, ω = 0 | 1.92e10 | 1.81e6 | 8.0e-13 | **2.3e-05** | 4.1e-06 Eh/Bohr |

**The reuse contributes nothing detectable.**  In both rows where the whole
term was measured, the patched and unpatched gradients are *closer to each
other* than the two associations of the same `j3c_oo` are — 5.6e-08 against
7.8e-07, and 4.1e-06 against 2.3e-05.  Whatever the cderi adds is below a
disagreement that exists without it, and both numbers track the condition
number rather than the system size: it goes 2.3e9 → 1.9e10 from 42 atoms to
113, and the gradients' disagreement goes 5.6e-08 → 4.1e-06 with it.

The long-range row is looser for a reason worth recording.  At ω = 0.3 the
long-range 2c2e matrix is **rank deficient**: its smallest eigenvalue is
−3.5e-13 and **1199 of 2191** auxiliary directions are below
`LINEAR_DEP_THR` — the fit runs in 992 of them rather than all 2191.  Both routes
go through the same filtered pseudo-inverse: `_decompose_j2c`'s Cholesky
fails there and falls back to exactly the eigen-decomposition the gradient
uses, so the two metrics agree to **2.8e-15** relative.  But the surviving
inverse has elements of **4.3e7**, and 1e-16 of that is 4e-9 in `dm_oo`
units.

### 5.3 The one thing that genuinely changes

For the **full-range** call on a spherical auxiliary basis both routes take
the Cholesky branch and there is nothing to choose between them.  Where they
can differ is the case `_jk_energy_per_atom` singles out: it takes the
filtered eigen-decomposition whenever `mol.omega > 0` or the auxiliary basis
is Cartesian, while `_decompose_j2c` takes the Cholesky whenever the Cholesky
succeeds.  Where those two choices differ, **upstream's gradient is not the
derivative of the energy it was computed from** and this module's is, because
this module inherits the decomposition the cderi — and therefore the SCF
energy — was actually built with.  Measured, the difference never leaves the
last digits; the point is which way round the argument runs.

---

## 6. What it does not cover, and what is left

### 6.1 Declines

`stats()` reports them, and `tests/test_dfgrad.py` §4 checks two of them
rather than asserting them:

| | |
|---|---|
| a **pure** functional's gradient | never reaches this function at all — `df/grad/rhf.py:68` sends `k_factor == 0` to `_j_energy_per_atom`, which builds no cderi.  A DF-PBE gradient reports zero reuse, and is right |
| `only_dfj` (RIJONX) | has no cderi by construction (`docs/README_fastdfj.md` §6); the call is handed back |
| a TDDFT or NAC gradient | `factorize_dm` returns an asymmetric pair, or the density carries `symmetrize`; `df/grad/tdrhf.py` has its own `_jk_energy_per_atom` anyway |
| `hermi != 1` | the pair list carries one triangle and the `×2` at the end assumes it |
| more than one device | the cderi is split across devices there, as every other module in `electrolites/` requires |
| an SCF whose cderi was already released | the reuse needs a live tensor; the call falls through and `STATS['rebuilt']` counts it |

The fall-through is GPU4PySCF's own `_jk_energy_per_atom`, so a patched run
computes the same gradient as an unpatched one in every one of those cases.

### 6.2 The size ceiling neither route removes

At **284 atoms** (PfPMT, def2-SVP, nao 2697, naux 13365, nocc 547) the
density-fitted gradient does **not run at all** on one A100-SXM4-40GB, and
that is upstream's limit rather than this module's: `_jk_energy_per_atom`
allocates `j3c_oo` as a dense `(naux, nocc, nocc)` array, which is

    13365 × 547 × 547 × 8 B  =  32.0 GB

*(**Corrected since.**  13365 is `mf.with_df.naux`, the spherical auxiliary
dimension; the allocation is on `int3c2e_opt.auxmol.nao`, the Cartesian
sorted one, which is 15762.  The array is **37.7 GB** and there are two of
them — `docs/README_fastdfgradoo.md` §4.)*

on a 40 GB card, and dies a few lines later in `cp.asarray(auxmol.ctr_coeff)`
with `cudaErrorMemoryAllocation` (`df/grad/rhf.py:116`).  This module needs
the same array — `dm_oo` is exactly it — so it dies in the same place.  The
SCF at that size does not: `_cholesky_eri` puts the 86.4 GB cderi on the
**host** and streams it, and `docs/DF_WORK_LIST.md` §3.3 measures the whole
thing running.  The gradient has no such fallback, because the fitting
coefficients are contracted against *every* auxiliary function in the same
kernel call.

So the largest system measured here is 113 atoms, and the ceiling above it is
a capacity one with a name: **an `naux · nocc²` array that is never batched.**
Batching it over the occupied index — `j3c_oo` and `dm_oo` in slices of `i`,
with one ip1 launch per slice — is a contained change and is the obvious next
item.  It is recorded on the work list rather than done here, because it is
the whole of what stands between this path and the 284-atom case the rest of
`docs/DF_WORK_LIST.md` is written around.

> **Done, in `docs/README_fastdfgradoo.md`, and not quite that way.**  One ip1
> launch per slice would have cost one pass of the derivative kernel per
> slice, which is 52–59 % of the function each time.  It is not needed: the
> array is *built* one auxiliary block at a time already, and the only step
> that needs the whole auxiliary index is the metric multiply — which slices
> the occupied index and runs **in place**, so the second array is never
> allocated and the ip1 loop is untouched.  Past that the array goes to
> page-locked host memory.  At 284 atoms the gradient then runs, at **64.5 s**
> against 80.8 s for the same function without the cderi reuse, and GPU4PySCF
> still cannot run it at all.

### 6.3 What is left

After the two items the function is **its own derivative kernel and little
else**: at 42 atoms/def2-SVP the ip1 kernel goes from 54.8 % of 928 ms to
about 77 % of 522 ms.  Making it smaller means one of two things, and neither
is a rearrangement:

* **A generated three-centre ip1 kernel**, the way `fastejk` generates the
  four-centre one.  It is a codegen project on that scale and it is the only
  route to a further large factor on this term.
* **Screening the auxiliary index.**  The pair list is screened; the
  auxiliary dimension is not, and `(pq|P)` decays with the distance between
  the pair centre and P.  That is the same "make the arithmetic smaller"
  move `docs/DF_WORK_LIST.md` W5 asks for on the SCF build, and it would
  apply to `eval_j3c`, to `_cholesky_eri` and to this kernel at once.

And one small thing inside this module.  The half-transform of the cderi goes
through GPU4PySCF's own `DF.loop`, which double-buffers the PCIe copy when the
tensor lives on the host but always streams *all* of it.  `fastdf`'s fourth
item — keep as many auxiliary rows resident as free memory allows and stream
only the remainder, worth 1.64x–4.53x on the SCF build — is inside
`fastdf.get_jk` and is not on this path.  Sharing it would be a refactor of
`fastdf`, not a new measurement, and it pays only where the gradient's own
cderi is host-resident, which on the systems above it never was.

And one thing this measurement hands to a **different** item.  The
`nsp_per_block*4` aggregation is not only in the gradient: the same constant
is in `int3c2e_evaluator` (`df/int3c2e_bdiv.py:176`), which is what
`_cholesky_eri` builds the cderi through, and a `*16` is in
`_j_energy_per_atom` and in `Int3c2eOpt.contract_dm`.  Swept on the plain
integral kernel, `sp_mult = 1` is **1.24x** at 42 atoms/def2-SVP, **1.46x**
at def2-TZVPP and **3.19x** at 113 atoms against upstream's `*4`
(188.4 → 151.9 ms, 675.2 → 463.5 ms, 1567.8 → 491.4 ms) — on a stage that
`docs/DF_WORK_LIST.md` §3.1 measures at **39.4 %** of a 15-cycle taxol DF
SCF.  That is W4's, not W3's, and it is recorded there.

---

## 7. Reproducing

```bash
# the correctness bar, including a finite-difference check of the gradient
python tests/test_dfgrad.py
python tests/test_dfgrad.py --basis def2-tzvpp --no-fd

# the profile of section 2 -- the traced clone, stage by stage
python benchmarks/df_grad_probe.py --xyz benchmarks/molecules/penicillinV.xyz \
    --basis def2-tzvpp --xc b3lyp --reps 3 --whole

# the measurement of section 4: five variants, round robin, one process
python benchmarks/df_grad_bench.py --xyz benchmarks/molecules/taxol.xyz \
    --basis def2-svp --xc b3lyp --reps 3 --whole --with-rebuild

# the launch configuration of the two Rys kernels, swept
python benchmarks/df_grad_ip1_sweep.py \
    --xyz benchmarks/molecules/penicillinV.xyz --basis def2-tzvpp

# where a density-fitted gradient's time goes, alongside its SCF
python benchmarks/df_anatomy.py --xyz benchmarks/molecules/penicillinV.xyz \
    --basis def2-tzvpp --patch all
```
