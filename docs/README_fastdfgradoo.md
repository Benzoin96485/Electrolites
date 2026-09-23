# The DF gradient's `(naux, nocc, nocc)` array — the size ceiling, and what removing it costs

`docs/DF_WORK_LIST.md` **W10**, found by **W3** rather than proposed by the
catalogue.  `docs/README_fastdfgrad.md` §6.2 is the measurement that opened it:
at **284 atoms** (PfPMT, def2-SVP, nao 2697, naux 13365, nocc 547) the
density-fitted gradient does not run at all on one A100-SXM4-40GB, patched or
not, because `df/grad/rhf.py:101` allocates `j3c_oo` as a dense
`(naux, nocc, nocc)` array — **32.0 GB** — and `df/grad/rhf.py:124` then
allocates `dm_oo = metric @ j3c_oo` as a second one.  The SCF at that size
runs; its gradient dies before it reaches an integral.

The item is a capacity item, and it comes with a falsification attached: *"the
extra passes cost more than the slicing saves at 113 atoms, i.e. the batched
version is slower where the unbatched one fits.  Guard it with a size
threshold the way `fastdf`'s `SYRK_MIN_NAO` is guarded, and measure the
threshold rather than guessing it."*

This is that round.  It is in `electrolites/fastdfgrad.py` — the same module
W3 produced — because the array in question is the one that module already
builds; `tests/test_dfgrad.py` §5 is its correctness bar and
`benchmarks/df_grad_oo_sweep.py` is the measurement.

---

## 0. Summary

* **The extra passes the item priced in are not needed.**  W10 expects the
  slicing to cost "extra passes over the cderi (or over `eval_j3c`,
  unpatched)".  It does not: the array is *built* one auxiliary block at a
  time in both routes — that is what the block loop already is — and it was
  only ever *held* whole.  The one step that has to see the whole auxiliary
  index at once is the metric multiply, and that one is a slice of the
  **occupied** index at a time, in place.  One pass over the cderi, as
  before.
* **Slicing is therefore free, measured.**  On the card, one slice against
  two, four and eight is **1.000x / 0.998x / 1.000x** at 42 atoms and
  **0.996x / 0.996x / 0.991x** at 113 (§2).  So the falsification does not
  fire on the slicing: what the size threshold has to guard is not the
  slicing, it is *where the array lives*.
* **The second copy is gone unconditionally.**  `dm_oo` is never allocated;
  the metric pass overwrites the array it reads.  Upstream's peak of
  `2 naux nocc²` becomes `(1 + 2/S) naux nocc²`, and at `S = 1` — the default
  wherever it fits — the output simply replaces the input with no copy at
  all.  That alone moves the ceiling from 20 GB of array to 40.
* **Past that, the array goes to page-locked host memory**, which is what
  `_cholesky_eri` already does one level down with the cderi.  That costs
  **1.087x** at 42 atoms and **1.294x** at 113, and the cost is *exactly*
  four traversals of the array over PCIe plus one page-locking (§2.3):
  47.0 ms predicted against 45.3 measured, and 721 against 719.5, from two
  rates measured elsewhere in this repository rather than fitted here.
  Nothing in it is unexplained.
* **And there is no threshold to find between the two.**  Walked up a ladder
  that withholds device memory until the device store will no longer
  allocate, the device store is ahead at **every** rung — 1.34x with 22 GB
  withheld, 1.18x with 24, 1.04x with 24.5 — with the host store's time flat
  across the whole table and the device store's the one degrading.  So
  `FASTDFGRAD_OOFRAC` is set inside the measured region rather than at a
  crossover, and what decides the hard cases is that the allocation is
  attempted and spills to the host on `OutOfMemoryError` (§3).
* **Three things only the host store needs, and all three are worth having.**
  The double-buffered prefetch under the ip1 kernel is **1.029x / 1.038x**;
  page-locking at all is **1.077x / 1.206x**; and page-locking with
  `cudaHostRegister` rather than `cudaHostAlloc` is **1.200x / 1.437x** —
  which is `docs/DF_WORK_LIST.md` **W12**'s 8.6 GB/s against 2.09 GB/s
  turning up in a second place, on an array a thirtieth the size of the one
  it was measured on.
* **And the route with no cderi to reuse is batched too.**  A size ceiling
  that only the *reuse* clears is not a cleared ceiling: a scanner, a
  gradient on a reconstructed density and `FASTDFGRAD_CDERI=0` all reach
  GPU4PySCF's own function, which cannot allocate at 284 atoms.
  `_fill_from_j3c` is that function's own first loop writing into the same
  store — the same integrals in the same order, agreeing with it element-wise
  at **1e-14** — so that route runs at 284 atoms as well, and the reuse has
  something to be priced against there.
* **The 284-atom gradient runs**: the whole analytic DF-B3LYP gradient in
  **68.8 s**, of which this term is 66.1 s or **96 %**, against 80.8 s for
  the same term without the cderi reuse and *nothing at all* from GPU4PySCF
  (§4).  The array it was stopped by is **37.7 GB**, not the 32.0 the work
  list assumed — the allocation is on the Cartesian sorted auxiliary
  dimension (15762), not the spherical one (13365) — and upstream holds two
  of them, so it asks a 40 GB card for 75.5 GB.
* **And what the round found rather than closed.**  Freeing the second array
  makes `mem_avail` larger, the derivative loop's block bigger, and the
  gradient reachable by the same cuTENSOR plan wall
  `docs/README_fastdfhess.md` hit on the DF Hessian: a
  `CUTENSOR_STATUS_NOT_SUPPORTED` with tens of GB free.  A halve-and-retry
  around that loop is the fix — and pinning the block says it has to be,
  because the refusal **does not depend on free memory** and is **not
  monotonic in the block**: 1096 is refused where 1240, 1456, 2048 and 3072
  are accepted, the last with 4.3e9 output elements, twice 2³¹.  No element
  cap of the kind the Hessian module carries can describe that, which makes
  the shared boundary `docs/DF_WORK_LIST.md` **W21** (§4.1).

---

## 1. Why there are two of the array, and why only one of them is avoidable

The quantity the ip1 kernel needs is a pseudo-density on the compressed AO
pair list, and it is built from

    dm_oo[r,i,j] = sum_v coef[r,v] oo[v,i,j]

where `oo` is the occupied-occupied transform of the three-centre integrals
(`j3c_oo` upstream, `rhok` on the reuse route) and `coef` is the fitting
metric (upstream) or `aux_coef` (the reuse — `docs/README_fastdfgrad.md` §3 is
the identity that makes the two the same thing).  Three facts about that line
decide the whole item:

1. **`coef` mixes the auxiliary index and nothing else.**  So `dm_oo[:, i, :]`
   for one slice of `i` depends on `oo[:, i, :]` for that same slice and on
   nothing else.  Slices of the occupied index are independent problems, and
   the multiply can be done in place, one slice at a time, with a working set
   of `2 naux nocc |slice|` instead of `2 naux nocc²`.
2. **But `dm_oo[r, :, :]` for one auxiliary block depends on *all* of `oo`.**
   So slicing the *auxiliary* index — which is what the ip1 loop is already
   batched over — does not decompose this step at all.  That is the reason
   upstream holds the whole array, and the reason the fix has to be on the
   occupied index.
3. **The array is built one auxiliary block at a time.**  Both routes produce
   `oo[p0:p1]` from a block of integrals and write it; neither ever needs the
   whole thing to build it.  This is what W10 did not have: the "extra passes"
   it priced in are the ones you would need if the *build* had to be sliced,
   and it does not.

Two more quantities do need the whole array, and both are sums over the
occupied indices, so both are accumulated in the same slice loop:

    auxvec[r] = sum_i dm_oo[r,i,i]
    koo[r,s]  = sum_ij dm_oo[r,i,j] dm_oo[s,i,j]

`koo` is what the auxiliary-basis response term contracts with the two-centre
derivative.  Upstream forms both *after* its ip1 loop, from an array it is
still holding; this forms them *before* it, from an array it is about to stop
holding.  The arithmetic is the same and the order of the sum is the only
thing that moves — which is why §6's tolerances do not.

So the function becomes three passes over an `naux nocc²` array instead of one
allocation of two of them:

| pass | what it does | what it needs at once |
|---|---|---|
| **fill** | one auxiliary block of `oo` per block of integrals | one block |
| **metric** | `oo <- coef @ oo`, in place, plus `auxvec` and `koo` | one slice of the occupied index, whole auxiliary index |
| **ip1** | the pseudo-density and `sum_ejk_int3c2e_ip1` | one auxiliary row slab |

and `_OOStore` is the object those three agree on.  In that layout a row slab
is contiguous and an occupied slice is a plain two-dimensional stride, so
every host transfer is one `cudaMemcpy` or one `cudaMemcpy2D` per spin
channel.  On the card the same three calls are views and cost nothing, which
is what lets one loop body serve both stores.

---

## 2. What the two knobs cost, where neither is needed

`benchmarks/df_grad_oo_sweep.py`, one `jk_energy_per_atom` call at fixed
density, every cell round robin in one process with the first pass thrown away
(`docs/DF_WORK_LIST.md` §5 items 2 and 3), one A100-SXM4-40GB,
`OMP_NUM_THREADS=16`, B3LYP/def2-SVP, level-3 grids.  Both systems are ones
where the array fits on the card easily, which is the point: this is the table
the falsification asks for.

### 2.1 Penicillin V, 42 atoms — nocc 92, the array 0.175 GB of 40.6 GB free

| | ms | vs the default |
|---|---|---|
| **device, 1 slice** (the default) | **518.2** | 1.000x |
| device, 2 slices | 517.0 | 1.002x |
| device, 4 slices | 519.3 | 0.998x |
| device, 8 slices | 518.4 | 1.000x |
| host, 1 slice | 563.5 | 0.920x |
| host, 2 slices | 562.0 | 0.922x |
| host, 4 slices | 563.4 | 0.920x |
| host, 8 slices | 570.4 | 0.909x |
| host, no prefetch | 579.7 | 0.894x |
| host, unpinned | 607.1 | 0.854x |
| host, `cudaHostAlloc` | 676.2 | 0.766x |

### 2.2 Taxol, 113 atoms — nocc 226, the array 2.696 GB of 26.9 GB free

| | ms | vs the default |
|---|---|---|
| **device, 1 slice** (the default) | **2443.0** | 1.000x |
| device, 2 slices | 2453.0 | 0.996x |
| device, 4 slices | 2451.9 | 0.996x |
| device, 8 slices | 2464.3 | 0.991x |
| host, 1 slice | 3162.5 | 0.772x |
| host, 2 slices | 3171.1 | 0.770x |
| host, 4 slices | 3156.9 | 0.774x |
| host, 8 slices | 3147.5 | 0.776x |
| host, no prefetch | 3281.8 | 0.744x |
| host, unpinned | 3814.6 | 0.640x |
| host, `cudaHostAlloc` | 4545.6 | 0.537x |

Every cell agrees with the first to **2e-13 to 9e-13 Eh/Bohr** against an
`|ejk|max` of order 1, which is the reassociation of a sum and not a
difference in what is computed.

### 2.3 What the four rows say

**Slicing the occupied index is free.**  Eight slices instead of one is 1.000x
at 42 atoms and 0.991x at 113.  This is the answer to W10's falsification, and
it is a *negative* answer in the useful direction: the item expected slicing to
cost extra passes and to need a size threshold of its own, and it costs
nothing because §1.3 — the array was already built a block at a time.  The
threshold is therefore not about slicing.

**The host store costs 1.087x at 42 atoms and 1.294x at 113, and the cost is
arithmetic.**  The array is traversed four times over PCIe — written by the
fill, read and written by the metric pass, read by the ip1 loop — and
page-locked once:

| | 4 traversals at 26.4 GB/s | page-lock at 8.6 GB/s | predicted | **measured** |
|---|---|---|---|---|
| penicillin V, 0.175 GB | 26.6 ms | 20.4 ms | 47.0 ms | **45.3 ms** |
| taxol, 2.696 GB | 408 ms | 313 ms | 721 ms | **719.5 ms** |

The two rates are not fitted to this table: 26.4 GB/s is the cderi's own
measured PCIe rate at 284 atoms (`docs/DF_WORK_LIST.md` §3.3) and 8.6 GB/s is
W12's `cudaHostRegister`.  So there is nothing unexplained in the host store's
cost, and nothing further to win in it either — it is the link, and one
page-locking.

**The prefetch is worth 1.029x and 1.038x.**  Small, because the read it hides
is a third of the four traversals and the ip1 kernel it hides behind is most
of the call; kept because it is fifteen lines and the same structure
`DF.loop` already uses for the cderi.  `FASTDFGRAD_OOPREFETCH=0` is the row.

**Page-locking is worth 1.077x and 1.206x, and how it is done is worth
1.200x and 1.437x.**  `docs/DF_WORK_LIST.md` **W12** measured
`cudaHostRegister` at 8.6 GB/s against `cudaHostAlloc`'s 2.09 on the *cderi's*
86 GB host buffer, and said the finding was about the allocation rather than
the page-locking.  Here it is again on an array a thirtieth of that size and
in a different module: `cudaHostAlloc` — which is what
`cupy_helper.empty_mapped` calls, and what any obvious implementation of this
store would have used — costs 1383 ms of a 3162 ms call at 113 atoms, against
the 313 ms the same 2.696 GB takes to register.  That is why `_pinned_empty`
allocates with NumPy and registers, and why `FASTDFGRAD_OOPIN` has three
settings rather than being a boolean.

## 3. The threshold, measured by taking the memory away

§2 says the device store is 1.09x–1.29x faster whenever it fits and that
slicing costs nothing, so `FASTDFGRAD_OOFRAC` — the fraction of free device
memory above which `auto` sends the array to the host — is guarding exactly
one thing: the point at which holding the array on the card stops leaving the
ip1 loop room for its own buffers.  `df/grad/rhf.py`'s block sizes are
`0.5` and `0.4` of what it believes is free, so the two are in direct
competition, and a 113-atom system on a 40 GB card never reaches the point on
its own (its array is 10 % of free memory).

So the memory is taken away instead.  `benchmarks/df_grad_oo_sweep.py
--ballast` holds a raw `cudaMalloc` — outside CuPy's pool, so that freeing it
actually frees it — for the duration of each timed call and nothing else, and
walks it up until the device store is the worse of the two.  Taxol,
def2-SVP, B3LYP; the array is 2.696 GB; the ladder is quoted in memory
**withheld** rather than as a ratio, because a ratio would need the free
memory at the moment of the call and this only knows it at the start of the
process:

| withheld | device store | host store | |
|---|---|---|---|
| — | **2442.0 ms** | 3171.2 ms | 1.30x |
| 8.0 GB | **2485.4** | 3332.6 | 1.34x |
| 14.0 GB | **2517.0** | 3307.5 | 1.31x |
| 17.0 GB | **2382.8** | 3303.1 | 1.39x |
| 19.0 GB | **2401.2** | 3261.0 | 1.36x |
| 20.5 GB | **2564.5** | 3332.7 | 1.30x |
| 21.5 GB | **2538.1** | 3309.6 | 1.30x |
| 22.0 GB | **2467.8** | 3313.1 | 1.34x |
| 23.0 GB | **2791.6** | 3410.9 | 1.22x |
| 23.5 GB | **2597.6** | 3413.3 | 1.31x |
| 24.0 GB | **2880.1** | 3411.1 | 1.18x |
| 24.5 GB | **3288.6** | 3410.8 | 1.04x |

**There is no crossover to find.**  The device store is ahead at every rung,
including the last one that can be run at all — 24.5 GB withheld, where what
is left on a 40 GB card is under 2.5 GB more than the array itself.  What the
ladder does show is the *shape* of the thing: the host store's time is
**flat** across the whole table (3171 → 3411 ms, 7 %) because its working set
is two row slabs and does not depend on how much is free, while the device
store's degrades from 2442 to 3289 (35 %) as the ip1 loop's blocks are
squeezed.  The two converge, and they converge to a meeting point that is at
or past the point where the device store stops being allocatable.

So `FASTDFGRAD_OOFRAC` is **not** set at a measured crossover, because there
is not one: it is set at **0.8**, inside the region where the device store is
still clearly ahead, and the thing that actually decides the hard cases is
that the allocation is *attempted* and the store falls back to the host on
`OutOfMemoryError` (`stats()['oo_spilled']` counts it).  A threshold can be
wrong about a machine it was not measured on; an allocation cannot.

*(The metric pass's own slice budget is a separate 0.4 and deliberately not
this number.  `_OOFRAC` is a placement threshold sitting near the allocation
limit; the slice budget is a working set the ip1 loop has to be left room
around, and since §2.3 says slicing costs nothing measurable, being generous
with it buys nothing.)*

## 4. The 284-atom case, which is what the item is for

PfPMT, def2-SVP, B3LYP, charge −1: 284 atoms, nao 2697, nocc 547, and a cderi
of **86.41 GB** that `_cholesky_eri` puts on the host.  One A100-SXM4-40GB in
a node with 251 GB of DRAM, one SCF cycle for a density with a live tensor
behind it, `benchmarks/df_grad_bench.py --cycles 1`.

**First, a correction to the size this item was opened with.**
`docs/DF_WORK_LIST.md` W10 and `docs/README_fastdfgrad.md` §6.2 both put the
array at **32.0 GB**, from `13365 × 547² × 8`.  13365 is `mf.with_df.naux`,
the *spherical* auxiliary dimension — and the one upstream allocates on is
`int3c2e_opt.auxmol.nao`, the **Cartesian sorted** one, which is **15762**.
So `j3c_oo` is **37.7 GB**, `dm_oo` is a second 37.7, and what GPU4PySCF asks
a 40 GB card for is **75.5 GB**, not 64.  (The store here is 37.7 GB for the
same reason: it is allocated at `max(naux_fit, naux_cart)` rows so that the
metric pass can overwrite its input, and the cderi's fitting dimension is the
smaller 13365.)

**And it runs.**

| | | |
|---|---|---|
| GPU4PySCF 1.8.1, `jk_energy_per_atom` | **cannot be run** | `cudaErrorMemoryAllocation` at `df/grad/rhf.py:116` |
| `_fill_from_j3c` — the same first loop, one array instead of two | 80.84 s | |
| **`fastdfgrad`** — and the cderi the SCF already built | **64.50 s** | **1.253x** |
| **the whole analytic DF-B3LYP gradient** | **68.81 s** | of which the above is 66.14 s, **96 %** |

The two routes agree to **2.03e-06 Eh/Bohr** on an `|ejk|max` of 52.85, i.e.
**3.8e-8 relative** — which is where the 1.5e-9 at 42 atoms and 7.0e-8 at 113
of `docs/README_fastdfgrad.md` §5 were heading, and tracks the fitting
metric's condition number rather than anything about the batching.  The
element-wise bar for the batching itself is §6, where the clone is compared
with GPU4PySCF's own function at a size where that function can be run, and
agrees with it at **1e-14**.

The store was host-resident on every call, 37.73 GB, and the metric pass took
**3 to 6 slices** of the occupied index depending on what else was live.  The
untimed cderi rebuild between rows — which an SCF pays for and hands its
gradient, and which upstream's first line throws away — is **76.4 s**,
comparable to the whole call.

**That the two-electron derivative is 96 % of the gradient at this size** is
the same number `docs/README_fastdfgrad.md` §1 measures at 97 % on taxol, and
it is why this item was the one standing between the path and the size: there
is nothing else in a 284-atom density-fitted gradient.

### 4.1 And a wall that only opens once this one is out of the way

The first attempt at the whole gradient did not run either, and not for
W10's reason: `cutensor.createPlan` answered
**`CUTENSOR_STATUS_NOT_SUPPORTED`** for the ip1 loop's
`contract('iqr,pi->pqr')` with tens of GB free.  That is the same wall
`docs/README_fastdfhess.md` hit from the other direction on the DF Hessian's
CPHF build, where it stops a 42-atom def2-TZVPP and a 113-atom def2-SVP
Hessian from running at all.

The answer here is the one that module already found to be the real safety
net: a **halve-and-retry**.  A rejected *plan* raises before any kernel is
launched and before any operand is read, and nothing in the ip1 loop writes
to the store, so halving the block and restarting the loop is safe — it fired
**three times** on the 284-atom run above and the gradient completed.
`tests/test_dfgrad.py` §5 injects the failure and checks that the restart is
clean, because the way this goes wrong quietly is `ejk` keeping the
half-accumulated contributions of the attempt that failed.

**And it had better be the answer, because a size cap cannot be.**  W10 made
this wall reachable — with no second `(naux, nocc, nocc)` array to reserve
for, `mem_avail` is larger than GPU4PySCF's and the block is bigger — so the
obvious reading is "the block got too big", and it is wrong.  Measured on an
80 GB card at taxol/def2-SVP, where the free memory is enough to sweep the
block over a wide range:

| ip1 block | `nao² × blk` (the output) | | |
|---|---|---|---|
| 8 / 128 / 512 | 1.1e7 / 1.8e8 / 7.2e8 | accepted | |
| **1096** | **1.539e9** | **refused** | at 70.3, 62.2, 54.2, 50.2 and 46.2 GB free alike |
| 1240 | 1.741e9 | accepted | at all five of the same |
| 1456 / 2048 / 3072 | 2.0e9 / 2.9e9 / **4.3e9** | accepted | |

Two things follow, and the second is the one that matters.

**It is not the memory.**  1096 is refused at 70 GB free and at 46 GB free,
and 1240 is accepted at both, so the plan's verdict does not depend on how
much room it has — which rules out the reading the first ladder suggested,
where withholding memory appeared to cause the refusal.  It appeared to
because withholding memory is also what shrank the block.

**And it is not a size.**  Acceptance is **not monotonic in the block**: 1096
is refused and 1240, 1456, 2048 and 3072 are accepted, the last of them at
**4.3e9 output elements — twice 2³¹**, which was the obvious suspect and is
not it.  So no threshold of the form "keep the output under N elements" can
describe this boundary, and `fastdfhess`'s measured `MAX_PLAN_ELEMENTS` may
be guarding the wrong variable on its own pattern too.  What *does* work is
changing the extent, which is what halving does: 1096 → 548 is accepted, and
every refusal seen in this round was cleared by one halving.

That two modules have now reached the same boundary from opposite directions,
and that at least one of them is guarding it with a constant that this
measurement says cannot be the right shape, is `docs/DF_WORK_LIST.md` **W21**.
`FASTDFGRAD_BLK` pins the block, which is how the table above was taken.

## 5. Against W3's own code, head to head

This round is a capacity item and should not have moved the sizes that
already worked.  Read against the table in `docs/README_fastdfgrad.md` §4.1
it appears to have moved them by 1.13x–1.17x, which for a capacity item is a
reason to go and look rather than a reason to be pleased.  So W3's code was
run out of a **git worktree at its own commit**, in the same job, on the same
node, with the same variant list (`docs/DF_WORK_LIST.md` §5 item 3 is why the
job boundary matters):

| `jk_energy_per_atom`, B3LYP | W3's code (27fe612) | this round | |
|---|---|---|---|
| taxol, def2-SVP — `stock` | 7369.9 ms | 7375.6 ms | 1.001x |
| taxol, def2-SVP — patched | **2549.2 ms** | **2536.5 ms** | **1.005x** |
| penicillin V, def2-TZVPP — `stock` | 3049.4 ms | 2841.0 ms | |
| penicillin V, def2-TZVPP — patched | **1160.6 ms** | **1153.6 ms** | **1.006x** |

**The round is performance-neutral, and the apparent 1.13x–1.17x was a
cross-table artifact.**  W3's *published* patched numbers are 2850.1 and
1314.7 ms; W3's own code produces 2549.2 and 1160.6 today.  Its `stock`
column, by contrast, reproduces to 0.1 % (7369.9 against a published 7369.0).
The difference is the variant list: the published table ran five variants
round robin and these ran two, and `docs/README_fastdfgrad.md` §4.1 already
says what that does — "it changes what the memory pool holds when
`_ejk_from_rhok` asks `get_avail_mem` how much is free, and therefore the
block size it picks".  Across the four runs of this round the same row reads
2439.7, 2492.4, 2506.6 and 2536.5 ms depending on how many variants were in
the list, which is the width of that effect and is wider than anything this
round changed.

Inside one process, where the comparison is clean, the two ablations say what
the round did to this term:

| | taxol | penicillin V, def2-TZVPP |
|---|---|---|
| `fastdfgrad` | 2536.5 ms | 1153.6 ms |
| `old sizing` — GPU4PySCF's own block expression, which reserves for an array this module no longer allocates | 2566.7 · **1.012x** | 1171.2 · **1.015x** |
| `old fill buf` — the per-block cuTENSOR temporary in the cderi half-transform, allocated and freed inside the block loop | 2658.0 · **1.048x** | 1160.4 · 1.006x |

Both are real and both are small, and the second is `docs/DF_WORK_LIST.md`
§5 item 8 again: a temporary allocated inside a block loop costs the
`cudaFree` that ends it, and there are 44 of them in a taxol cderi pass.
Neither is what this round is for, and together they are inside the
process-to-process width above — which is the honest reason the head-to-head
column reads 1.005x rather than 1.06x.

## 6. Correctness

`tests/test_dfgrad.py` §5, in one process, and it is in three parts because
the three things that can go wrong here are of different kinds.

**The store's own transfers.**  Every host transfer in `_OOStore` is raw
pointer arithmetic — one `cudaMemcpy` per spin for an auxiliary row slab and
one `cudaMemcpy2D` per spin for a slice of the occupied index — and the shapes
that break it are the ones the molecules in the suite do not produce: two spin
channels, and a store whose allocated row count exceeds the number of rows
actually written, which is what a Cartesian auxiliary dimension over a
spherical fitting one is (§4).  So the round trip is checked directly, on
`(nset, rows/nrows, nocc)` of `(1, 7/7, 5)`, `(2, 7/7, 5)`, `(2, 6/9, 4)` and
`(1, 3/4, 11)`, written in slabs of two rows and read back both in slabs of
three and in column slices of two, on both stores.  All eight are **exact**.

*(The first version of this was not.  A NumPy `int32` out of `aux_offsets`
added to a device pointer raises `OverflowError` rather than producing a
wrong address, which is the good case.  Every offset in `_OOStore` is a
Python `int` on purpose.)*

**The gradient, through every combination of the knobs.**  Ethanol/def2-SVP
B3LYP and the CH₃ doublet, against GPU4PySCF's own `_jk_energy_per_atom` on
the same converged density, for each of seven store configurations and both
routes:

| | against GPU4PySCF |
|---|---|
| the reuse route, device store, 1 / 3-column slices | 3.3e-10 / 3.3e-10 Eh/Bohr |
| the reuse route, host store, 1 / 2-column slices | 3.3e-10 / 3.3e-10 |
| the reuse route, host, no prefetch / unpinned / `cudaHostAlloc` | 3.3e-10 each |
| **the batched `j3c_oo` clone**, all seven | **5e-15 to 1.3e-14** |
| the same two on the unrestricted case | 4.6e-11 and 2.6e-13 |

The reuse rows are W3's number unchanged — the store does not move it, which
is the point.  The clone's rows are three to four orders tighter, and they
are the check that matters for this round: it is GPU4PySCF's own loop with
the same integrals in the same order, so anything above round-off there is a
batching bug and not a conditioning one.

**The halve-and-retry**, with the failure injected, because the shape that
actually triggers it is a 284-atom one.  Two forced refusals, and the answer
comes back identical to the un-refused one at **8.9e-15** — which is the
check that `ejk` did not keep the half-accumulated contributions of the
attempts that failed.

**And a race, which is the failure mode this arrangement actually has.**  The
first working version of the host store was wrong by **1e-2 Eh/Bohr**, with
nothing else to show for it, because the prefetch runs on a non-blocking
stream and a non-blocking stream does not synchronise with the legacy default
one: the ip1 loop read the first row slab before the metric pass had finished
writing it.  Nothing but an element-wise comparison finds that, the error
moves when the slice count moves, and it looks exactly like a tolerance.  It
is why the table above is seven configurations rather than one.

The whole-gradient and finite-difference checks of
`docs/README_fastdfgrad.md` §5.1 are unchanged and still pass, as do
`tests/test_patch_aliases.py` and the def2-TZVPP run of the same suite.

## 7. What this round did not do

* **`koo` is a GEMM where a `dsyrk` would do.**  The auxiliary-response
  accumulation `koo[r,s] = sum_ij dm_oo[r,i,j] dm_oo[s,i,j]` is symmetric and
  is computed as a general product, which is what upstream does too.  At 284
  atoms it is `2 naux² nocc²` ≈ 1.5e14 flop, of order 8 s of the 64.5 — the
  largest single thing in this function that is neither an integral kernel
  nor the link.  `fastdf`'s `_syrk` is the routine and `SYRK_MIN_NAO` is the
  precedent; it belongs in `electrolites/_dfaux.py` rather than in either
  module, and it is a different item.
* **The host store is four traversals and could be three.**  The metric pass
  reads and writes.  Folding its read into the fill would need the fill
  transposed into occupied-major order first, because the metric pass needs
  the *whole* auxiliary index, and that is a bigger change than the part of
  1.09x–1.29x it would recover.
* **545 atoms is still out of reach, and not because of this array.**
  `HcgC`/def2-SVP has a 414 GB cderi against a 251 GB node
  (`docs/README_dflocal.md` §1), so the SCF does not run there either;
  `FASTDF_FP32=1` halves it to 207 GB (`docs/README_fastdffp32.md`), and what
  this store would then need is `naux_cart × nocc² × 8` at that size.  That
  is the next capacity question and it is a different one.

## 8. Reproducing

```bash
# the correctness bar: the round trip, the knob matrix, the injected retry
python tests/test_dfgrad.py
python tests/test_dfgrad.py --basis def2-tzvpp --no-fd

# section 2: the two knobs, where neither is needed
python benchmarks/df_grad_oo_sweep.py --xyz benchmarks/molecules/taxol.xyz \
    --basis def2-svp --xc b3lyp --reps 2 --extras

# section 3: the threshold, measured by taking the memory away
python benchmarks/df_grad_oo_sweep.py --xyz benchmarks/molecules/taxol.xyz \
    --basis def2-svp --xc b3lyp --reps 2 --stores device,host --slices auto \
    --ballast 0,8,14,17,19,20.5,21.5,22,23,23.5,24,24.5

# section 4: the 284-atom case, which GPU4PySCF cannot run at all
python benchmarks/df_grad_bench.py --xyz benchmarks/molecules/PfPMT.xyz \
    --charge -1 --basis def2-svp --xc b3lyp --cycles 1 --reps 1 \
    --variants stock,fastdfgrad --with-rebuild
python benchmarks/df_grad_bench.py --xyz benchmarks/molecules/PfPMT.xyz \
    --charge -1 --basis def2-svp --xc b3lyp --cycles 1 --reps 1 \
    --variants fastdfgrad --whole

# section 4.1: what cuTENSOR actually refuses, with the block pinned and
# the free memory moved independently -- an 80 GB card, which is what makes
# the block sweepable
FASTDFGRAD_BLK=1096 python benchmarks/df_grad_bench.py \
    --xyz benchmarks/molecules/taxol.xyz --basis def2-svp --xc b3lyp \
    --reps 1 --variants fastdfgrad      # stats()['plan_retries'] is the answer

# section 5: the two ablations, inside one process.  W3's own code is a
# `git worktree add <dir> 27fe612` away, and running it in the same job is
# the only way the two rounds are comparable at all
python benchmarks/df_grad_bench.py --xyz benchmarks/molecules/taxol.xyz \
    --basis def2-svp --xc b3lyp --reps 2 \
    --variants stock,fastdfgrad,'old sizing','old fill buf'
```

The switches, all of them measurements rather than preferences:

| | |
|---|---|
| `FASTDFGRAD_OOSTORE=auto\|device\|host` | where the array lives; `auto` is §3 |
| `FASTDFGRAD_OOFRAC=<f>` | the placement threshold `auto` uses, 0.8 |
| `FASTDFGRAD_OOSLICE=<n>` | occupied columns per metric slice; 0 takes as many as fit |
| `FASTDFGRAD_OOPREFETCH=0` | do not overlap the row-slab read with the ip1 kernel |
| `FASTDFGRAD_OOPIN=register\|alloc\|none` | how the host array is page-locked |
| `FASTDFGRAD_J3CBATCH=auto\|always\|never` | whether the route with no cderi is batched too; `auto` takes it only where GPU4PySCF cannot allocate at all |
| `FASTDFGRAD_OOSIZING=new\|upstream` | the ip1 loop's block sizing; the ablation of §5 |
| `FASTDFGRAD_FILLBUF=0` | the other ablation of §5: put back the per-block cuTENSOR temporary |
| `FASTDFGRAD_BLK=<n>` | pin the ip1 loop's auxiliary block instead of deriving it from free memory; §4.1's table is taken with it |

and `fastdfgrad.stats()` reports `oo_host`, `oo_device`, `oo_spilled`,
`oo_slices`, `oo_gb`, `plan_retries` and `j3c_batched`, because which of
these ran is not visible in the answer.
