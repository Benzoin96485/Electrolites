# fastdfhess — the DF Hessian's CPHF build: a function that cannot be made faster, and one that cannot be run

`docs/DF_WORK_LIST.md` **W8**, the eighth item on that list.  It closes, and
the headline is negative: the function the item names is already at **78 % of
an A100's fp64 peak**, it is **15 % of the Hessian it sits in**, and three
rearrangements of it were written, measured and rejected.

What the round found instead is that the same function **does not run at all**
past a certain shape — a 42-atom def2-TZVPP Hessian and a 113-atom def2-SVP
one both die inside it with tens of GB of free memory — and that the
density-fitted Hessian's time is somewhere else entirely, in three functions
no entry on that list names.

---

## 0. Summary

* **W8's own target is at the machine.**  `df/hessian/rhf.py:_get_jk` runs one
  15.53 Tflop call in **1015 ms** on one A100-SXM4-80GB — **15.2 TF/s of
  19.5, 78 % of fp64 peak**.  Its dominant contraction is at **81 %** and
  every one of its four is within **1.14x** of a plain `dgemm` at its own
  shape; two of the four are *faster* than the `dgemm`.  §2.
* **Three rearrangements, three losses.**  Grouping the algebra by the
  perturbed density so the several-GB intermediate is written once and read
  once (`fused`): **0.90x–0.94x**.  Giving the two contractions that write
  into `vk` one accumulator each in the layout their own GEMM produces
  (`blasvk`): **0.91x–1.00x**.  GPU4PySCF's own ordering through this
  module's loop (`upstream`, the default): **0.97x–1.00x**.  §3.
* **And the target is 15 % of the run.**  A 42-atom DF-B3LYP/def2-SVP Hessian
  is **34.6 s**, of which `_get_jk` is **5.19 s**; a 22-atom def2-TZVPP one is
  10.98 s of which it is **6.3 %**.  Even deleting the function is 1.18x.
  §5 is the first anatomy of a density-fitted Hessian in this repository, and
  it re-ranks the whole path.
* **What the module is for**: past about 5e8 output elements
  `cutensor.createPlan` refuses `Lpq,snqi->snpiL` with
  **`CUTENSOR_STATUS_NOT_SUPPORTED`**, and GPU4PySCF sizes that contraction
  from free memory alone.  **A 42-atom def2-TZVPP DF Hessian and a 113-atom
  def2-SVP one therefore cannot be run at all** — on an 80 GB card with
  73.7 GB free and on a 40 GB card with 34.0 GB free alike.  It is the
  descriptor that is rejected, not the workspace that is missing.  Capping
  the density batch makes both run: the calls are **4.05 s** and **110.3 s**
  at 16.2 and 15.7 TF/s, and **a whole 42-atom DF-B3LYP/def2-TZVPP Hessian
  that GPU4PySCF raises on completes in 70.4 s**, twice, to the same trace.
  The cap costs **0.7 %** where the call already ran on an 80 GB card and is
  a small gain on a 40 GB one.  §4.
* **The "Leave space for storing Krylov subspace vectors" comment is
  guarding 0.37 GB with 31 GB.**  Measured, a CPHF converges in **five**
  Krylov cycles and its subspace is 0.37 GB at 42 atoms, against a cderi of
  0.99 GB and upstream's 0.4-of-free-memory reserve of 31 GB.  And the
  residency does not matter for the exchange call anyway — on the host it is
  **1017 ms against 1024** on the device, because the prefetch hides the
  whole stream behind the contractions.  It matters **9.0x** for a
  Coulomb-only call (81.8 ms against 9.1), which is the branch a pure
  functional takes.  §6.
* **It is the same J and K.**  Element-wise against GPU4PySCF in one process,
  restricted and unrestricted, one to six right-hand sides, J+K and J alone,
  full-range and long-range: **≤ 1.0e-14** relative on every row, and a whole
  water Hessian unchanged to **2.8e-12 Eh/bohr²**.  §7.

---

## 1. The function

`df/hessian/rhf.py:_get_jk` is the **fifth** two-electron code path in
GPU4PySCF, after `scf/jk.py:get_k`, `scf/j_engine.py:get_j`,
`scf/jk.py:_VHFOpt.get_jk` and `df/df_jk.py:get_jk`.  A CPHF iteration asks it
for J and K of the perturbed densities in the MO basis, and it goes straight
at the cderi rather than through `df_jk.get_jk` — which is why `fastdf`, whose
one patch covers the DF SCF, the DF response solve and a DF Hessian's
`get_veff`, is not on it.

With `M_L` the `L`-th auxiliary slice of the (unpacked, symmetric) cderi,
`C_o` the occupied coefficients and `X_n` the `n`-th perturbed one, the
quantity is

    vk[n] = sum_L M_L (X_n C_o^T + C_o X_n^T) M_L C_o

and GPU4PySCF evaluates it as four contractions per auxiliary block and
density batch:

| | contraction | cost |
|---|---|---|
| `rhok` | `Lpq,sqj->spjL` | `nao^2 nocc` per `L` |
| `rhok_oo` | `spjL,spi->sijL` | `nao nocc^2` per `L` |
| `rhok1` | `Lpq,snqi->snpiL` | `nao^2 nocc` per `L` **and density** |
| `vk_a` | `snpiL,sjiL->nspj` | `nao nocc^2` per `L` and density |
| `rhok1_oo` | `snpiL,spj->snjiL` | `nao nocc^2` per `L` and density |
| `vk_b` | `spiL,snjiL->nspj` | `nao nocc^2` per `L` and density |

so `2 nao nocc (nao + 3 nocc)` flops per auxiliary function and right-hand
side.  That count is the same in every arrangement §3 tries — the three are
re-associations, not different algorithms, and the module asserts it by
computing the same matrices to round-off.

---

## 2. Where the time is, and what the roofline says

`benchmarks/df_hess_probe.py` runs a traced clone of the function — the same
arithmetic in the same order, checked element-wise against the real one on
every run (it agrees to **1.7e-16**) — with a synchronising timer between
stages.  Penicillin V, DF-B3LYP/def2-SVP, `nao` 430, `naux` 2191, `nocc` 92,
126 right-hand sides, one A100-SXM4-80GB of a whole node:

| stage | time | share | Gflop | TF/s | of peak |
|---|---|---|---|---|---|
| `loop` (unpack) | 9.1 ms | 0.9 % | — | — | — |
| `rhok` | 6.0 ms | 0.6 % | 74.5 | 12.42 | 63.7 % |
| `rhok_oo` | 2.1 ms | 0.2 % | 15.9 | 7.64 | 39.2 % |
| **`rhok1`** | **592.1 ms** | **57.9 %** | 9392.2 | **15.86** | **81.3 %** |
| `vk_a` | 149.8 ms | 14.6 % | 2009.5 | 13.42 | 68.8 % |
| `rhok1_oo` | 132.5 ms | 13.0 % | 2009.5 | 15.16 | 77.8 % |
| `vk_b` | 119.0 ms | 11.6 % | 2009.5 | 16.89 | 86.6 % |
| `vj` | 3.4 ms | 0.3 % | 21.8 | 6.35 | 32.6 % |
| **whole call** | **1014.6 ms** | | **15533** | **15.18** | **77.8 %** |

**The roofline is the point of the table.**  `--roofline` times a plain
`cp.dot` at each stage's own `(m, n, k)`, on the same card in the same
process:

| stage | m × n × k | plain `dgemm` | the stage | headroom |
|---|---|---|---|---|
| `rhok1` | 110080 × 11592 × 430 | 18.04 TF/s | 15.86 | **1.14x** |
| `vk_a` | 54180 × 92 × 23552 | 13.55 TF/s | 13.42 | **1.01x** |
| `rhok1_oo` | 92 × 23552 × 430 | 10.20 TF/s | 15.16 | **0.67x** |
| `vk_b` | 430 × 11592 × 23552 | 14.80 TF/s | 16.89 | **0.88x** |

Two of the four run *faster in the loop* than a plain `dgemm` of the same
shape timed on its own — `rhok1_oo` because the loop batches 126 of them
where the roofline times one, and `vk_b` by 14 % at a shape the roofline
reproduces exactly — and the one contraction with headroom has 14 % of it,
on a stage that is 58 % of the call.  Taking `rhok1` all the way to its
`dgemm` rate would be **1.076x** on the whole function, and that is the
ceiling on everything §3 tries.  This is the same finding as
`docs/DF_WORK_LIST.md` §2's headline for `df_jk.get_jk`, reached
independently on a different function.

The block size is upstream's and stays upstream's: 64 / 128 / 256 / 512
auxiliary functions per pass measure **1065 / 1025 / 1019 / 1100 ms**, so the
`MIN_BLK_SIZE = 256` cap is the best of the four.

At 22 atoms in def2-TZVPP (`nao` 478, `nocc` 39, 66 right-hand sides) the same
table reads **13.57 TF/s** over a 104.5 ms call, with `rhok1` again at
**15.84 TF/s** and 68 % of the time.  On a 40 GB card the whole call is
**14.36 TF/s**.

---

## 3. Three rearrangements, and why each loses

All three are in `electrolites/fastdfhess.py` behind `FASTDFHESS_ARRANGE`,
because a rejected alternative that can be re-run is worth more than one
described in prose.  `benchmarks/df_hess_bench.py` times them **round robin**
against GPU4PySCF's own function in one process (`docs/DF_WORK_LIST.md` §5.2).
The table below is the arrangements alone, measured with §4's batch cap off,
so that it prices the ordering and nothing else.

| | 42 atoms, def2-SVP, 80 GB | 22 atoms, def2-TZVPP | 42 atoms, def2-SVP, 40 GB |
|---|---|---|---|
| GPU4PySCF | 1023.2 ms | 114.0 ms | 1064.5 ms |
| `upstream` (this module's loop, same ordering) | 1024.7 ms — **1.00x** | 116.9 ms — 0.97x | 1076.4 ms — 0.99x |
| `blasvk` | 1050.3 ms — 0.97x | 113.9 ms — 1.00x | 1102.3 ms — 0.97x |
| `fused` | 1114.7 ms — 0.92x | 121.4 ms — 0.94x | 1184.6 ms — 0.90x |

**`fused` is the one that should have worked.**  Group the algebra by the
perturbed density rather than by the density factor,

    vk[n] = sum_L M_L (D1_n M_L C_o),   D1_n = X_n C_o^T + C_o X_n^T,

and the bracket `G[n,q,j,L] = X_n B_L + C_o U` is two `nao nocc^2`
contractions while the single `nao^2 nocc` one *consumes* it and writes
straight into `vk`.  The flop count is identical term for term; what changes
is that the 4.5 GB intermediate is written once and read once instead of
written once and read twice, and that the expensive contraction's output is
40 MB rather than a second several-GB array.  Its expensive contraction is
indeed the fastest single kernel measured anywhere in this function —
**17.0 TF/s, 87 % of peak**, against `rhok1`'s 15.86 — but the two `nao nocc^2`
contractions that build `G` accumulate into the same buffer, so `G` is
written, read and written again, and that stage runs at **9.8 TF/s**:

| `fused` | | | upstream | |
|---|---|---|---|---|
| `U` (`nao nocc^2`) | 129.7 ms | 15.50 TF/s | `rhok1` (`nao^2 nocc`) | 592.1 ms, 15.86 TF/s |
| `G` (two `nao nocc^2`, the second accumulating) | 412.2 ms | **9.75 TF/s** | `vk_a` + `rhok1_oo` | 282.3 ms |
| `vk` (`nao^2 nocc`) | 553.7 ms | **16.96 TF/s** | `vk_b` | 119.0 ms |
| the four contractions | **1095.6 ms** | | | **993.4 ms** |
| the whole call | **1125.3 ms** | | | **1014.6 ms** |

The 38 ms the big contraction gains, `G` gives back 130.

There is one more move in that direction and it does not pay either.  Half of
`G` never needs `M_L` applied: `M_L C_o U = R_L U`, because `R_L = M_L C_o` is
already in hand, so the second term can accumulate into `vk` directly and `G`
stops being written twice.  Adding up the stage times above, that is
`U` 129.7 + one `G` contraction ~206 + a `vk_b`-shaped accumulation 119.0 +
the big one 553.7 = **~1008 ms** against upstream's 1014.6 — a tie, arrived at
by removing the only thing the `fused` arrangement had going for it.  It is
not carried.

**`blasvk` is the one the roofline suggested.**  Upstream's `vk` accumulator
is `(n_dm, nspin, nao, nocc)` and *neither* of the two contractions that write
into it produces that layout — `vk_a` comes out as `[(n,p), j]` and `vk_b` as
`[p, (n,j)]` — so one of the two is a permuting write whichever way the buffer
is shaped.  Giving them one accumulator each and adding the two with a
transpose of a 40 MB array at the end does move `vk_b` (16.89 → 14.59 TF/s,
i.e. *worse*: cuTENSOR's permuting write beats an explicit `cp.dot` plus an
accumulate) and costs `rhok1_oo` 15.16 → 13.71 TF/s.  Net 0.97x.

**Four smaller things were measured and are either free or worthless:**

* *The auxiliary vector of the Coulomb pass.*  Upstream takes it as the
  diagonal of the `(nocc, nocc)` intermediate, `cp.einsum('sniiL->nL', ...)`,
  one element in `nocc` out of several hundred MB.  The same vector is
  `sum_{q,i} rhok[q,i,L] X[n,q,i]`, one contraction against an array `nocc`
  times smaller, and it makes the Coulomb half independent of the exchange
  intermediate.  Kept — it is free and it is tidier — but the `vj` stage is
  0.3 % of the call either way.
* *`rhok1_oo` from `rhok` instead of from `rhok1`.*  The cderi block is
  symmetric in its two AO indices, so `C_o^T M_L X = (M_L C_o)^T X` and the
  occupied projection needs no `rhok1` at all.  Same flops, and the stage
  does get faster — 132.5 → 127.4 ms at 42 atoms, 8.97 → 6.65 at 22 — but
  the **call** does not: 1.00x and 0.92x, because `rhok1` still has to exist
  for `vk_a` and the alternative reads `rhok` once more.  Not carried.
* *Balanced density batches.*  `_balance_batch_size` is upstream's own helper,
  defined forty lines below the function that needs it and never called
  there: `n_dm = 126` at `dm_batch_size = 55` runs passes of 55, 55 and then
  **16**.  On an A100 the whole right-hand side fits in one pass at every size
  measured, so this is worth **1.00x** — it is kept because the configuration
  where it is not (a smaller card, a bigger basis) is exactly the
  configuration §4 is about.
* *The batch size's dependence on the allocator.*  Upstream derives it from
  `get_avail_mem(exclude_memory_pool=True)`, i.e. from `cudaMemGetInfo`,
  which does not count the blocks CuPy's pool is holding but not using — so
  in principle every call after the first sees less free memory than the
  first.  Measured over six consecutive calls it does not move (126, 126,
  126, 126, 126, 126; free memory 77.77 then 77.59 GB and flat), because the
  allocation pattern repeats exactly.  **1.00x**, and a real effect only when
  something else is churning the pool between cycles.

---

## 4. What the round is actually for: the function does not run

`benchmarks/df_hess_cutensor_limit.py`.  GPU4PySCF sizes the density batch
from free memory alone (`df/hessian/rhf.py:1291`) and hands the whole batch to
one cuTENSOR contraction.  Past a certain shape `cutensor.createPlan` returns

    CuTensorError: CUTENSOR_STATUS_NOT_SUPPORTED

for `Lpq,snqi->snpiL`, raised out of `estimateWorkspaceSize` — so it is the
**operation descriptor** that is refused, not the workspace that is missing,
and free memory does not help:

| | `nao` | `nocc` | `dm_batch` | output | GPU4PySCF |
|---|---|---|---|---|---|
| penicillin V / def2-SVP, 42 atoms | 430 | 92 | 126 | 1.28e9 elements | runs |
| alanine dipeptide / def2-TZVPP, 22 atoms | 478 | 39 | 66 | 3.15e8 | runs |
| **penicillin V / def2-TZVPP, 42 atoms** | 1007 | 92 | 126 | 2.99e9 | **raises**, 73.7 GB free |
| **taxol / def2-SVP, 8 atoms of RHS** | 1123 | 226 | 24 | 1.56e9 | **raises**, 65.5 GB free |

The def2-TZVPP case raises on a 40 GB card too, with 34.0 GB free, so it is
not a memory limit read the wrong way round.  Bisected, the largest planable
`dm_batch` is

| `nao`, `nocc` (blk 256) | largest `dm_batch` | output elements |
|---|---|---|
| 430, 92 | 530 | 5.4e9 |
| 478, 39 | 224 | 1.1e9 |
| 1007, 92 | **22** | 5.2e8 |
| 1123, 226 | **8** | 5.2e8 |

which is **not** a clean function of the shape — 430/92 plans five billion
output elements and 1123/226 refuses one and a half.  So the module does two
things rather than one: it caps the batch at `FASTDFHESS_MAX_PLAN` output
elements, default **5e8** (under all four measured boundaries), and it
**halves the batch and starts the call again** whenever a plan is refused
anyway, remembering the smaller batch on the density-fitting object so the
next Krylov cycle does not rediscover it.  A rejected plan raises before any
operand is read, so restarting is safe.

**What the cap costs, and what it buys.**  Density batches of
16 / 32 / 48 / 63 / 126 at 42 atoms measure **1126 / 1084 / 1083 / 1100 /
1094 ms** on a 40 GB card — the flat bottom is around 48, which is what a 5e8
cap picks there, and the largest batch is not the fastest — and
24 / 49 / 63 / 126 measure **1057 / 1039 / 1030 / 1031 ms** on an 80 GB one,
where the cap's 49 is **0.7 %** behind the uncapped 126.  Against that:

| | GPU4PySCF | `fastdfhess` |
|---|---|---|
| 42 atoms / def2-SVP | 1023 ms (126 RHS a pass) | 1039 ms (49, the cap) |
| 22 atoms / def2-TZVPP | 114 ms (66) | 117 ms (66, under the cap) |
| **42 atoms / def2-TZVPP, 126 RHS** | **raises** | **4049 ms** (16.2 TF/s) |
| **113 atoms / def2-SVP, 24 RHS** | **raises** | **8051 ms** (15.3 TF/s) |
| **113 atoms / def2-SVP, 339 RHS** | **raises** | **110.3 s** (15.7 TF/s) |

and end to end, a **whole 42-atom DF-B3LYP/def2-TZVPP Hessian**:

| | round 0 | round 1 | trace |
|---|---|---|---|
| GPU4PySCF 1.8.1 | raises | raises | — |
| + `fastdfhess` | **70.46 s** | **70.44 s** | 0.1896000471 |

That is the whole of what this round is worth, and it is not a ratio.

---

## 5. The anatomy of a density-fitted Hessian

Nothing in this repository had this.  `docs/HESSIAN_1.8.1.md` is the *direct*
path, whose `partial_hess_elec` and `make_h1` are different functions.
`benchmarks/df_hess_probe.py --whole` wraps the density-fitted ones — the two
top-level terms as **class attributes**, because `df/hessian/rks.py` binds
them in its class body and a patch of the module global never reaches them
(the same trap `tests/test_patch_aliases.py` exists for).  Penicillin V,
DF-B3LYP/def2-SVP, level-3 grids, `conv_tol_cpscf = 1e-6`, one
A100-SXM4-80GB.  Nested rows are sub-intervals of their parent.

| stage | unpatched | | with `patch_all()` | |
|---|---|---|---|---|
| **whole Hessian** | **34.58 s** | | **21.24 s** | **1.63x** |
| `partial_hess_elec` | 12.90 s | 37.3 % | 5.86 s | 27.6 % |
| ↳ `_get_vxc_deriv2` | 7.80 s | 22.6 % | **0.76 s** | 3.6 % |
| ↳ `_jk_energy_per_atom` (ip2) | 4.50 s | 13.0 % | 4.50 s | 21.2 % |
| `make_h1` | 10.16 s | 29.4 % | 3.93 s | 18.5 % |
| ↳ `_get_vxc_deriv1` | 7.74 s | 22.4 % | **1.55 s** | 7.3 % |
| ↳ `_get_veff` | 2.13 s | 6.2 % | 2.13 s | 10.0 % |
| ↳ `get_grad_hcore` | 0.28 s | 0.8 % | 0.25 s | 1.2 % |
| `solve_mo1` (CPHF) | 11.47 s | 33.2 % | 11.40 s | 53.7 % |
| ↳ `nr_rks_fxc_mo` | 5.36 s | 15.5 % | 5.40 s | 25.4 % |
| ↳ **`_get_jk` (W8)** | **5.19 s** | **15.0 %** | 5.13 s | 24.1 % |
| `_cholesky_eri` (the one rebuild) | 0.16 s | 0.5 % | 0.12 s | 0.6 % |

Four things follow, and three of them are new work rather than results.

**W8's target is 15 % of the run, and 24 % after everything else is fast.**
That is the number the item should have been ranked against.  With the
function already at 78 % of peak, the whole entry is worth about 1.01x.

**`patch_all()` is 1.63x on a density-fitted Hessian**, and all of it is
`fastxchess`: `_get_vxc_deriv1` is **5.0x** and `_get_vxc_deriv2` **10.3x**.
That is a bigger number than this round produced and it was already there.

**Three functions carry the remaining time and nothing here is on any of
them.**  After `patch_all()`, the 21.24 s is `nr_rks_fxc_mo` 5.40 s (25 %),
`_get_jk` 5.13 s (24 %), `df/hessian/rhf.py:_jk_energy_per_atom` 4.50 s
(21 %) and `df/hessian/rhf.py:_get_veff` 2.13 s (10 %).  The last two are the
density-fitted Hessian's *other* two-electron functions — the `ip2`
two-electron term of `partial_hess_elec` and the `h1` term of `make_h1` — and
neither is `df/grad/rhf.py:_jk_energy_per_atom` (which `fastdfgrad` replaces)
nor `hessian/rhf.py:_partial_ejk_ip2` (which `fastejk2` replaces).  They are a
sixth and seventh code path.

**And `fastxcfxc` is not on the CPHF's grid half.**  It replaces
`dft/numint.py:nr_rks_fxc`; a Hessian's CPHF calls
`hessian/rks.py:nr_rks_fxc_mo`, a different function, and the row does not
move (5.36 → 5.40 s).  That is the single largest stage of a patched DF
Hessian.

All four are now entries on `docs/DF_WORK_LIST.md`.

---

## 6. The cderi's residency, and the comment that sets it

`df/hessian/rhf.py:1258-1273` moves a device-resident cderi to the host
whenever it is more than **0.4** of the free memory, with the comment *"Leave
space for storing Krylov subspace vectors"*.  W8 carries that comment forward
as standing advice — `FASTDF_RESIDENT_FRAC=0` is the setting for a DF Hessian —
and nothing had measured either half of it.

`benchmarks/df_hess_krylov_probe.py` counts the cycles and samples the
allocator.  At 42 atoms in def2-SVP:

| | |
|---|---|
| Krylov cycles the CPHF actually takes | **5** (and 5 at 10 atoms) |
| one cycle's pair of trial vectors | 0.074 GB |
| the whole subspace | **0.37 GB** |
| the cderi | 0.99 GB |
| free memory the whole Hessian ever consumed | ~2 GB of 78 |
| what upstream's 0.4 rule reserves | **31 GB** |

So the rule reserves about eighty times what it is protecting.  `fastdfhess`
computes the reserve instead — `3 · cycles · n_dm · nmo · nocc` words, with
`cycles` a knob defaulting to 8 against the 5 measured — and keeps the tensor
on the device whenever that fits beside it.

**And for the exchange call the residency does not matter at all.**  With the
same tensor forced onto the host, the J+K call is **1017 ms** against
**1024 ms** device-resident — the double-buffered prefetch hides the whole
0.99 GB behind contractions that take a second.  That is the same finding as
`docs/DF_WORK_LIST.md` §3.3 for `df_jk.get_jk`, and it is why none of this is
worth a factor.

**For a Coulomb-only call it matters 9.0x**: 81.8 ms on the host against
9.1 ms on the device, which is exactly the PCIe time for 0.99 GB.  That is
the branch a **pure** functional's CPHF takes, and the branch where
`fastdf`'s partial residency would have helped had it been switched on.  The
Coulomb pass itself as an explicit GEMV/GEMM pair (`fastdf`'s item 3) is
**1.37x–1.42x** at 9 and 10 atoms, **1.01x–1.12x** at 42, and **1.00x** with
the tensor on the host, where it is bandwidth-bound and there is nothing to
rearrange — most of what it buys at 42 atoms is the auxiliary buffer
allocated once instead of per block, not the BLAS call.

---

## 7. Correctness

`tests/test_dfhess.py`, one process, one cderi tensor, element-wise against
GPU4PySCF's own `_get_jk`.  Ethanol and water in def2-SVP and def2-TZVPP,
restricted and unrestricted, one/three/six right-hand sides, J+K and J alone,
the full-range and the `omega = 0.3` long-range operator, and all three
arrangements:

| | worst relative error |
|---|---|
| J, every row | **4.0e-15** |
| K, `upstream` | **1.4e-16** |
| K, `blasvk` | **1.0e-15** |
| K, `fused` | **1.0e-14** |
| a whole water DF-B3LYP/def2-SVP Hessian, patched against unpatched | **2.8e-12 Eh/bohr²** |
| a whole penicillin V Hessian's trace, patched against unpatched | **2.4e-11** |

`fused` is looser than the other two by two orders of magnitude and that is
expected rather than wrong: it sums the same terms in a different order, and
its `G` accumulates two contributions into one buffer where upstream
accumulates two into `vk`.  It is an ablation, not the default.

The suite also checks the two things with no numerical symptom: that the
evened-out density passes cover `n_dm` exactly once for every `(n_dm, raw)`
pair below 200, and that a density without the CPHF's `mo1`/`occ_coeff` tags
is handed back to GPU4PySCF rather than mis-computed.

---

## 8. What is left

* **`hessian/rks.py:nr_rks_fxc_mo`** — 25 % of a patched DF Hessian, and
  `fastxcfxc` is not on it.  `docs/DF_WORK_LIST.md` W16.
* **`df/hessian/rhf.py:_jk_energy_per_atom`** — 21 %.  W17.
* **`df/hessian/rhf.py:_get_veff`** — 10 %.  W18.
* **The plan boundary itself.**  The cap is a measured number with no model
  behind it; a shape sweep that found the actual predicate would let the cap
  be tight instead of conservative, and would be worth knowing for every
  other cuTENSOR call in this repository.
* **A partially resident cderi for the Coulomb-only branch.**  It is 9.0x
  there and `electrolites/fastdf.py` already has the machinery; what it needs
  is the reserve from §6 so that it does not take back the memory the Hessian
  just freed.
