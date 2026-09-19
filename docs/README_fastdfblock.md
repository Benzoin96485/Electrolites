# The block-sparse exchange contraction, and the sparsity that is not spatial

`docs/DF_WORK_LIST.md` **W5** asks for the density-fitted exchange build's
first contraction to run only on the AO pairs the screening kept.  Its prize,
as the item states it, is **4.5x the arithmetic** at 284 atoms — the reciprocal
of the 22.2 % pair fill `df_fill_probe.py` measures there — "growing with
system size", against a contraction `df_probe.py` measures at **55.9 ms of a
115 ms block**.  It is the largest lever the work list has left on DF-K itself.

This round built it and measured it, and the answer is in three parts.

**The prize is not 4.5x.**  A GEMM addresses tiles, not pairs, and the tile
fill at 284 atoms is **69.4 %**, not 22.2 % — a ceiling of 1.44x on that
contraction rather than 4.5x.  §1 says why, and the reason is a measurement
nothing in this repository had made: the cderi's sparsity is the *product* of
an atom-pair factor, which a tile can address, and a within-atom-block factor
at one-to-five-AO granularity, which no AO tile of any size can address.  At
284 atoms those are 52.0 % and 43.7 %.

**The item's own falsification criterion fires, exactly where it said it
would.**  At taxol's fill "the tile GEMMs are slower than one dense
contraction" — measured, **0.91x** on the contraction and **0.72x** on the
whole exchange block.  So W5's acceptance criterion, which requires a win at
113 atoms *and* at 284, cannot be met.

**And above a measured crossover it pays — 1.21x on a whole 284-atom J+K
call — but only after a profiler.**  The first contraction is **1.31x** at 284
atoms in isolation and the exchange block around it 1.09x, and for two jobs
the whole call read **0.85x to 1.02x** instead.  Nsight Systems found why, and
it was not the algorithm: `RowCompressed.unpack` allocated a 763 MB temporary
**per block**, CuPy's pool went to the driver for it, and `cudaFree`
synchronises the device — so the double-buffered prefetch was drained 114
times a call and **2.33 s of an 11.73 s call was GPU idle**.  Allocating that
buffer once instead of once per block turns the same measurement into
**10825 ms → 8916 ms, 1.21x**, with the GPU 97.7 % busy.  §4.1 is the whole
story, and its two transferable lessons are that a kernel benchmark cannot
see an allocator, and that `cudaFree` is a synchronisation point.

It ships **on**, gated on the tile fill rather than on the system size,
because §2.1 shows a size threshold would be actively wrong: taxol in
def2-TZVPP has 2636 basis functions against PfPMT's 2697 in def2-SVP and a
tile fill of 98.9 % against 72.3 %.

Everything here is one A100-SXM4-80GB of a whole `--qos=debug` node,
`CUDA_VISIBLE_DEVICES=0`, GPU4PySCF 1.8.1, each basis with its own jkfit
auxiliary set.

---

## 1. What the pattern looks like when it is quantised to tiles

`benchmarks/df_block_probe.py`.  No timing, no SCF and no integrals — the
screening is a property of the geometry and the two basis sets, so this table
is reproducible on a login node in a minute.

`pairfill` is the fraction of the **square** `nao x nao` that the stored pairs
and their mirrors cover (`df_fill_probe.py`'s number in the triangle's
normalisation).  `tilefill` is the fraction covered by AO tiles holding at
least one stored pair — which is what a tiled contraction has to do arithmetic
on, and therefore the only ceiling that matters.  def2-SVP:

| | atoms | nao | pairfill | tilefill T=16 | T=32 | T=64 |
|---|---|---|---|---|---|---|
| penicillin V | 42 | 430 | 65.6 % | 98.3 % | 99.5 % | 100.0 % |
| taxol | 113 | 1123 | 50.6 % | 94.7 % | 98.2 % | 100.0 % |
| Co(II) sextet, `tmc_co_sextet` | 113 | 1059 | 52.8 % | 96.1 % | 99.5 % | 100.0 % |
| PfPMT | 284 | 2697 | **22.2 %** | 61.9 % | **69.4 %** | 73.5 % |
| HcgC | 545 | 5037 | **16.5 %** | 55.5 % | **66.1 %** | 77.9 % |

(the lowest of the four AO orderings of §2 at each tile size; the native order
is within three points of it everywhere except PfPMT/T=64, where it is nine.)

**The pair fill falls by 4x from 42 to 545 atoms; the tile fill falls by 1.5x
and then flattens.**  W5 assumed the second column tracked the first.  It does
not, and §2 shows the gap is not an artefact of a bad AO ordering.

### 1.1 The two factors, and which one a tile can reach

`--breakdown` splits the pair fill into the two independent things it is a
product of: how many **atom pairs** survive at all, and how dense the AO block
of a surviving atom pair is.

| def2-SVP | atoms | pair fill | = atom-pair fill | x AO density inside a surviving atom block |
|---|---|---|---|---|
| penicillin V | 42 | 65.6 % | 93.4 % | 69.6 % |
| taxol | 113 | 50.6 % | 90.3 % | 56.0 % |
| Co(II) sextet | 113 | 52.8 % | 92.2 % | 56.4 % |
| PfPMT | 284 | 22.2 % | 52.0 % | 43.7 % |
| HcgC | 545 | 16.5 % | 45.0 % | 38.5 % |

The first factor is spatial: two atoms far enough apart contribute no pair at
all.  The second is not.  Inside a surviving atom pair — two atoms close
enough that *something* survives — 56 to 62 % of the AO pairs are still
dropped, and which ones depends on the two shells' exponents and angular
momenta, not on where the atoms are.  A def2-SVP second-row atom carries 14
AOs in five shells of one to five functions, so that structure lives at
**one-to-five-AO granularity**.

An AO tile is a contiguous run of AO indices.  At def2-SVP's 9.5 AOs per atom
a tile of 32 is about three and a half atoms and a tile of 64 about seven, so
a tile pair is a *block of atom pairs*, and it is present as soon as any one
of its `T^2` entries is.  It can capture the first factor and not even all of
that; the second factor — more than half the sparsity at 284 atoms and nearly
two thirds at 545 — is invisible to it.  **That is a statement about the basis
set, not about this implementation**, and it bounds every executor equally:
`cuSPARSE` BSRMM, a custom batched GEMM over present tile pairs and the
per-tile-row contraction §3 actually runs all do the same arithmetic on the
same tiles.  Measuring one good executor bounds them all.

### 1.2 The one padded batched call does not exist

The natural GPU shape here is a single batched contraction over tile rows,
padded to a common width.  `--hist` reports the width distribution and kills
that: the widest tile row is **93.7 %** of `nao` at PfPMT/T=32 and **96.6 %**
at HcgC/T=32 in the best ordering, so padding to it throws away everything the
tiling won.  The contraction has to be one call per tile row — at the shipped
tile of 16, 169 of them per block at PfPMT and 315 at HcgC.  §3 measures what that costs: the tiled
contraction reaches **0.82 to 0.93** of the dense one's flop rate.

---

## 2. Four AO orderings, and why none of them rescues it

Both AO indices of the exchange contraction may be permuted freely:
`rhok[i,k,L] = sum_j A[i,j,L] f[j,k]` and `vk = rhok rhok^T` are covariant
under a permutation `P` applied to `A`'s two indices and `f`'s first, and `vk`
comes back as `P vk P^T` — one gather of the density factor at `O(nao*nocc)`
and one scatter of the result at `O(nao^2)`, against a contraction at
`O(nao^2 nocc naux)`.  The reordering is free in everything but code;
`electrolites/_dfblock.py` implements it and `fastdf` un-permutes `vk` in one
line.

This matters because the cderi's pair addresses are in the **original** Mole's
AO order (`int3c2e_bdiv.py:506`, `original_ao_order=True`), which is
atom-by-atom in the order the geometry file happens to list the atoms.  The
tile locality of the pattern is inherited from an xyz file and is nobody's
design — which makes "reorder it properly" the obvious first thing to try.

Measured, tile fill at T=32:

| | native | `rcm` | `rcb` | `morton` |
|---|---|---|---|---|
| taxol, 113 atoms | 98.2 % | 99.6 % | 98.8 % | 98.5 % |
| PfPMT, 284 atoms | 72.3 % | **69.4 %** | 73.0 % | 74.2 % |
| HcgC, 545 atoms | 67.4 % | 69.6 % | **66.1 %**\* | 67.0 % |

\* `rcb` is lowest at HcgC/T=32 and highest at HcgC/T=64 (79.0 %); no ordering
is best at both sizes and both tiles.

`rcm` is reverse Cuthill-McKee on the atom adjacency the surviving pairs imply
— the textbook bandwidth reduction, and the only one of the four that reads
the pattern rather than the coordinates.  `rcb` is recursive coordinate
bisection, so every contiguous chunk is a compact spatial cluster.  `morton`
is a Z-order curve.

**The best of them is worth three points of fill and the worst is worth minus
two.**  A fifth was tried and is worse than all of them: ordering AOs by shell
class (angular momentum, then exponent decade) with space as the minor key, on
the theory that grouping tight functions together would expose §1.1's second
factor.  At PfPMT/T=32 that reads **76.9 % to 87.5 %** depending on how finely
the exponents are bucketed, against 72.3 % for the native order — it loses
more spatial locality than it gains shell purity, because a 32-AO tile of one
class spans ten to thirty atoms instead of three.  The two factors cannot both
be blocked by one ordering of one index.

So `native` is the shipped default: no graph to build, and the measured
difference is inside the noise of the block ratio either way.

### 2.1 The fill does not track the basis-set size, and this is the whole design

def2-TZVPP, same three molecules, same probe:

| def2-TZVPP | atoms | nao | pair fill | = atom-pair | x in-block | tilefill T=32 |
|---|---|---|---|---|---|---|
| penicillin V | 42 | 1007 | 61.8 % | 98.4 % | 62.5 % | 99.7 % |
| taxol | 113 | 2636 | 46.3 % | 96.4 % | 47.6 % | 98.9 % |
| PfPMT | 284 | 6395 | 20.1 % | 63.5 % | 31.3 % | 72.0 % |

Going from def2-SVP to def2-TZVPP at 284 atoms moves the pair fill from 22.2 %
to 20.1 % and the **tile fill not at all** (69.4 → 72.0 %).  The two factors
move in opposite directions and cancel: the atom-pair fill *rises* from 52.0 %
to 63.5 %, because the larger set's diffuse functions reach further, while the
within-block density falls from 43.7 % to 31.3 %.  Only the first is
available to a tile.

The consequence is the design decision of this round.  **Taxol in def2-TZVPP
has 2636 basis functions, within three per cent of PfPMT's 2697 in def2-SVP,
and a tile fill of 98.9 % against 72.3 %** — so a threshold on `nao`, which is
how `fastdf`'s `SYRK_MIN_NAO` is guarded, would switch the tiled contraction
*on* for a case where it loses.  Timed, at block 32 and T=32:

| def2-TZVPP | nao | fill | step1 dense | tiled | step1 | block |
|---|---|---|---|---|---|---|
| taxol, 113 atoms | 2636 | 98.9 % | 7.00 ms | 9.65 ms | **0.73x** | **0.72x** |
| PfPMT, 284 atoms | 6395 | 72.0 % | 81.40 ms | 67.01 ms | **1.21x** | **1.09x** |

Same two systems, same tile, 2.4x apart in `nao` in the *wrong* direction for
a size threshold — and the fill predicts both.  The gate is therefore the
measured fill itself (`_use_tiled`), which costs one `unique` over the pair
list, no device work, and is exact.  It also carries across the basis change
without retuning: PfPMT reads 1.09x on the block in def2-SVP and 1.09x in
def2-TZVPP.

---

## 3. What it measures

`benchmarks/df_block_bench.py` times four things per (molecule, ordering, tile
size), round robin so a competing job's drift does not land on one variant
(`DF_WORK_LIST.md` §5.2): GPU4PySCF's `fill_symmetric` unpack and its dense
`contract('Lij,njk->nikL')`, against a scatter into the row-compressed buffer
and one `contract('ijL,jk->ikL')` per tile row.  It needs no cderi and no SCF —
the contraction's cost does not depend on the values — which is what lets the
284- and 545-atom cases run in seconds.  `--check` verifies the two agree:
**bit-identical** (`max |tiled - dense| = 0`) on every case up to 284 atoms and
2697 AOs, and **9.0e-15** relative at 545 atoms and 5037, which is the
summation order and nothing else.

One caveat on the 545-atom row, since the rest of this repository quotes
end-to-end numbers.  HcgC/def2-SVP has 2 094 539 stored pairs and 24 722
auxiliary functions, so its cderi is **414 GB** — larger than the 251 GB of
host DRAM `DF_WORK_LIST.md` §3.3 measures PfPMT's 86 GB against.  That row is
the real pattern and the real contraction, on a node where the whole
calculation does not currently fit.  The 284-atom rows are a size that runs.

def2-SVP, block 128 (the size `df_probe.py` measures the 284-atom build
choosing), `n_dm = 1`.  `step1` and `block` are dense ÷ tiled, `block` over
unpack + step 1 + the `dsyrk` step 2 that is the same call in both:

| | nao | tile | fill | step1 dense | tiled | TF/s | step1 | block |
|---|---|---|---|---|---|---|---|---|
| penicillin V, 42 | 430 | 32 | 100.0 % | 0.47 ms | 1.24 ms | 9.2 → 3.5 | 0.38x | 0.48x |
| taxol, 113 | 1123 | 16 | 94.9 % | 5.57 | 7.26 | 13.1 → 9.5 | 0.77x | 0.69x |
| taxol, 113 | 1123 | 32 | 98.2 % | 4.53 | 5.04 | 16.1 → 14.2 | **0.90x** | **0.75x** |
| Co(II) sextet, 113 | 1059 | 32 | 100.0 % | 4.08 | 4.55 | 14.4 → 12.9 | 0.90x | 0.74x |
| PfPMT, 284 | 2697 | 16 | 61.9 % | 55.93 | 42.02 | 18.2 → 15.0 | **1.33x** | **1.10x** |
| PfPMT, 284 | 2697 | 32 | 72.3 % | 55.94 | 44.22 | 18.2 → 16.7 | 1.27x | 1.07x |
| PfPMT, 284, `rcm` | 2697 | 32 | 69.4 % | 55.94 | 42.47 | 18.2 → 16.7 | 1.32x | 1.10x |
| PfPMT, 284 | 2697 | 64 | 82.7 % | 55.93 | 52.42 | 18.2 → 16.1 | 1.07x | 0.98x |

and HcgC at block 32, which is what a dense `(nao, nao, blk)` buffer allows at
that size:

| | nao | tile | fill | step1 dense | tiled | TF/s | step1 | block |
|---|---|---|---|---|---|---|---|---|
| HcgC, 545 | 5037 | 16 | 55.5 % | 90.63 ms | 65.85 ms | 18.1 → 13.8 | 1.38x | 1.20x |
| HcgC, 545 | 5037 | 32 | 67.4 % | 90.58 | 64.71 | 18.1 → 17.1 | **1.40x** | **1.21x** |
| HcgC, 545 | 5037 | 64 | 78.2 % | 90.60 | 76.59 | 18.1 → 16.8 | 1.18x | 1.10x |

Three things to read out of that.

**The gain is the fill, divided by an efficiency the tiling costs.**  That
relation is an identity — the flop counts differ by exactly the fill — so the
only thing the clock adds is the value of the efficiency, and it is
**0.76 to 0.95**: 85 to 315 cuTENSOR contractions per block where there was
one cost between 5 % and 24 % of the dense contraction's flop rate, and
nothing worse.  That is why the result is a ceiling argument: the executor is
close to as good as a tiled executor gets, so §1's fill is what decides, for
this implementation and for the `cuSPARSE` BSRMM and custom batched-GEMM ones
the item also proposed.

**Bigger tiles lose more fill than they win back in efficiency.**  T=64 gains
1-3 TF/s over T=32 and pays ten points of fill, and it is behind both at both
large sizes.  Between T=16 and T=32 this table is nearly tied — 1.10x against
1.07x at 284 atoms, 1.20x against 1.21x at 545 — and §4 breaks the tie the
other way: **end to end** they are 1.21x and 1.07x, and 16 is what ships.

**This table is the second version of itself.**  The first was taken before
§4.1's allocator fix, and its tiled unpack column was inflated: the
microbenchmark's own per-block temporary was being served out of CuPy's pool
while the real call's went to the driver, so the benchmark saw a cost it could
not attribute and the call paid one the benchmark could not see.  The
conclusions here did not move; the reason to say it is that *no* re-run of
this script would have found what §4.1 found.

### 3.1 Block size

PfPMT, T=32, native, sweeping the auxiliary block:

| blk | dense step1 | TF/s | tiled step1 | TF/s | step1 | block |
|---|---|---|---|---|---|---|
| 32 | 16.41 ms | 15.5 | 14.41 ms | 12.8 | 1.14x | 1.04x |
| 64 | 31.28 | 16.3 | 23.57 | 15.6 | **1.33x** | **1.13x** |
| 128 | 56.29 | 18.1 | 44.63 | 16.5 | 1.26x | 1.06x |
| 256 | 112.41 | 18.1 | 92.23 | 16.0 | 1.22x | 1.04x |

The dense contraction needs `blk >= 128` to reach its plateau; the tiled one
is within 6 % of its own plateau at 64, because each of its calls is `ntile`
times smaller and the extra `L` depth matters correspondingly less.  The ratio
is therefore best at an *intermediate* block, which is not where either curve
peaks.  The default block is left at GPU4PySCF's own choice — `fastdf`'s
`_blksize` accounts for the smaller buffer (below) and chooses 241 at PfPMT on
the 80 GB card these rows were taken on, against the 128 `DF_WORK_LIST.md`
§3.3 records on a 40 GB one — which is why every row here fixes the block
explicitly rather than taking what the memory allows.

### 3.2 Memory

The unpacked buffer is exactly `fill` times the dense `(nao, nao, blk)` one,
and the staging buffer of §4.1 adds `npair * blk`.  Together, against the
dense path's single array: **5.44 GB against 7.45** at PfPMT/T=16 and
**4.14 against 6.50** at HcgC/T=16.  Against that, the density factor has to
be gathered onto the tile rows once per `get_jk` call, and that array is
`fill * nao / TILE` times as tall as the factor.  `_blksize` counts the two
per-block arrays in its divisor and subtracts the gathered factor from the
budget, because that one does not shrink when the block does.  Net, the
tiled path is about a quarter smaller per block than the dense one — W5's
"it also removes the dense work buffer" is true, but by 1.4x and not by
`1/fill`.

---

## 4. What ships

`FASTDF_STEP1=auto` is the default: §4.2's fill test, then either the
block-sparse first contraction or GPU4PySCF's dense one.  `TILE` is **16**,
`TILE_ORDER` is `native`, and `tensor`, `bgemm` and `tiled` force one branch
each for the ablation columns.

End to end, at a converged density, both implementations in one process on one
cderi tensor, each row its own round robin of two
(`benchmarks/df_jk_bench.py`), 284 atoms with the cderi on the host:

| | tile | dense reference | tiled | |
|---|---|---|---|---|
| PfPMT, 284 atoms | 16 | 10824.87 ms | **8916.35 ms** | **1.21x** |
| PfPMT, 284 atoms | 32 | 10799.20 | 10087.46 | 1.07x |
| *before the §4.1 fix* | 16 | 10738.19 | 11896.06 | *0.90x* |
| *before the §4.1 fix* | 32 | 10323.71 | 12214.31 | *0.85x* |

and the single profiled call of `benchmarks/df_block_profile.py`, which is a
different script and a different process and agrees: **10850 ms → 8808 ms**,
1.23x.

For scale, in the same script's eight-variant table stock GPU4PySCF reads
**13091 ms** on that call and `fastdf` with the dense contraction **10684** —
1.23x, which is the four items the module already had.  The two ratios are
from different round robins and §4.1 says why they must not be divided into
each other.

**Tile 16 beats tile 32 end to end by 0.14x**, where §3's kernel benchmark
puts them within 0.03x.  That is the same lesson as §4.1 from the other side:
once the per-block allocation is gone, what is left is the fill, and T=16's
61.9 % against T=32's 72.3 % is the whole difference.  T=16's extra calls —
169 contractions a block against 85 — cost nothing measurable.

### 4.1 The gap was an allocator, and Nsight Systems is what found it

The kernels §3 times summed to 1.09x tiled at 284 atoms and the call came out
at 0.85-1.02x over two jobs.  Three explanations were checked from wall clocks
and none accounted for it: the block size the real call runs at (worth part of
it, §3.1), a cached-buffer reallocation (the fix is in `_tiles` and made the
tiled path *slower*), and the resident cderi prefix (`no-resident` is within
half a per cent of `fastdf` on this call, exactly as `DF_WORK_LIST.md` §3.3
found).

`benchmarks/df_block_profile.py` annotates every stage of every block with
NVTX and settles it in one table.  One `get_jk` per variant, PfPMT, host
cderi, tile 16:

| | GPU kernel time | wall | GPU busy | GPU ops | `cudaFree` calls |
|---|---|---|---|---|---|
| dense | 10.64 s | 10.66 s | 99.9 % | 1 130 | **4** |
| tiled, before | **9.40 s** | **11.73 s** | **80.1 %** | 20 728 | **118** |
| tiled, after | 8.61 s | 8.81 s | 97.7 % | 21 970 | 4 |

The tiled path was already doing **1.13x less GPU work** and then leaving the
GPU idle for **2.33 s**.  And the cause is the last column: one
allocate-and-free pair **per block**.

`RowCompressed.unpack` began with `cp.ascontiguousarray(cderi_blk.T)`, and at
284 atoms that temporary is `blksize x npair x 8` = **763 MB**.  CuPy's pool
went to the driver for it every block, and **`cudaFree` synchronises the
device** — so the double-buffered prefetch `fastdf` inherits from GPU4PySCF
was drained 114 times a call, and the host-to-device copy that
`DF_WORK_LIST.md` §3.3 measures as *fully hidden* stopped being hidden.
763 MB at 26.4 GB/s is 29 ms a block; 114 blocks of it partly exposed is the
2.33 s, to the digit.

The fix is three lines — the staging buffer is allocated once per pattern
rather than once per block, and `_blksize` counts it — and the same
measurement then reads 1.21x.

Stage by stage after the fix, per variant (`cuda_gpu_kern_sum`):

| | dense | tiled |
|---|---|---|
| step 1 | 6.497 s, 158 cuTENSOR GETT | **4.308 s**, 21 800 GETT |
| step 2 (`dsyrk`) | 3.557 s | 3.403 s |
| unpack | 0.647 s, `decompress_and_transpose` | 0.457 s transpose + 0.295 s scatter |
| Coulomb pass | 0.134 s | 0.133 s |
| **total** | **10.84 s** | **8.61 s** |

The first contraction is **1.51x** inside the real call at this block size,
and the tiled unpack — 0.752 s against the dense kernel's 0.647 — is now
within 16 % of GPU4PySCF's hand-written one rather than four times it.
**W14**, a CUDA kernel for the tiled scatter, is worth that remaining 0.1 s
plus the staging buffer it would remove, and is no longer the headline it
looked like.

Three things to take from this beyond W5.  A **kernel benchmark cannot see an
allocator**: `df_block_bench.py` reuses one device-resident block, so its
timed region allocated once where the real call allocated 114 times — which
is exactly why it read 1.09x where the call read 0.90x.  **`cudaFree` is a
synchronisation point**, so a per-iteration temporary in a pipelined loop does
not cost its own allocation time, it costs the pipeline.  And **a cached
buffer must not be sized from free memory**, which is the `_tiles` change that
came out of the wrong diagnosis and is kept on its own merits.

### 4.2 The gate

`_use_tiled` takes the block-sparse contraction when the tile fill is at or
below **`TILE_MAX_FILL = 0.75`**.

**At the shipped tile of 16 it separates all five systems exactly**: it
rejects penicillin V (98.3 %), taxol (94.9 %) and the Co(II) sextet (98.9 %),
where the tiled contraction is 0.33x to 0.69x on the block, and accepts PfPMT
(61.9 %) and HcgC (55.5 %), where it is 1.10x and 1.20x.  End to end, both
sides of that are measured: at taxol `fastdf` reads **403.54 ms** and its
`s1=dense` ablation 404.48 — the gate declined, as it should, because forcing
the tiled contraction there costs 0.88x — and at PfPMT it accepts and reads
1.21x.

Over all 30 timed (system, tile, ordering) rows behind §3 the gate makes no
false *accept*: every row it lets through is 1.04x or better on the block.  It
does make two conservative rejects, both at 545 atoms and tile 64 (78.2 % and
77.9 % fill, 1.10x on the block), where it falls back to the dense
contraction and forgoes a gain.  Since 16 is the shipped tile that case does
not arise by default, and a gate that declines a win is the right way round.

The gate is on the fill and not on `nao` for the reason §2.1 measures, and
that is the part of this round with a life beyond W5: **the tile fill is not
a function of the basis-set size**, and the two systems that prove it are 2.4x
apart in `nao` in the direction that would fool a size threshold.

### 4.3 Correctness

`tests/test_df.py --ablate` runs the tiled contraction as two more
configurations — once as it ships and once at tile 8 with an `rcm` reordering,
so that the compression and the permutation are both non-trivial on molecules
this small.  The suite's worst relative difference over J and K is
**3.0e-16 / 1.5e-14** against a tolerance of 1e-11, next to a previous worst
of 5.4e-16 / 3.6e-15; `test_dfj`, `test_dfcderi` and `test_dfgrad` are
unchanged and pass.

Set against `DF_WORK_LIST.md` §3.3's framing — W5 as "the largest lever on
DF-K itself", worth 12.1 s of a 12.38 s build — 1.21x on the call is a fifth
of what the item priced.  §1.1 is most of the reason and it is a property of
the basis rather than of the code: two thirds of the sparsity the item costed
is not addressable by a tile of any size.

---

## 5. What would actually be needed

§1.1 says where the rest of the sparsity is, and it is not reachable by making
the tiles smaller: at tile 8 the fill is 53.1 % at PfPMT against 61.9 % at
tile 16, for twice the calls on `(8, w_I, nL)` operands.  The factor that is
left lives at shell granularity, and a *shell*-blocked contraction is not a
GEMM at all — blocks of one to five rows against a `(nao, nocc)` factor.

The two things that would reach it are both larger than W5, and both are now
on the work list:

* **Never form the block.**  Contract the three-centre integrals with the
  density directly on the shell-pair list, which is what
  `df/j_engine_3c2e.py` already does for the *Coulomb* build and what
  `DF_WORK_LIST.md` **W9** names.  The exchange version has an occupied index
  in the middle and is a generator project, not a driver rewrite.
* **Localise the occupied orbitals** — `DF_WORK_LIST.md` **W15**.
  `vk = rhok rhok^T` is *exactly* invariant under an orthogonal rotation of
  the density factor's second index, so the factor may be replaced by a
  localised one for free, and `rhok` is then sparse in `(i, k)` as well.  That
  is the route every linear-scaling exchange implementation takes, it is a
  different algorithm with its own screening and error budget, and the
  permutation machinery in `electrolites/_dfblock.py` is the half of it that
  already exists.

---

## 6. Tools

```bash
python benchmarks/df_block_probe.py --breakdown --hist        # §1, §2; no GPU time
python benchmarks/df_block_probe.py --basis def2-tzvpp        # §2.1
python benchmarks/df_block_bench.py --check --cases taxol:0:0,PfPMT:0:-1
python benchmarks/df_block_bench.py --cases PfPMT:0:-1 --nl 32,64,128,256
python benchmarks/df_jk_bench.py --xyz ... --tiled 16 --tile-orders native
python tests/test_df.py --ablate            # the correctness bar, tiled included

nsys profile -t cuda,nvtx -c cudaProfilerApi -o $SCRATCH/p \
    python benchmarks/df_block_profile.py --xyz ... --cpu-cderi --tile 16
python benchmarks/df_block_profile.py --stats $SCRATCH/p.nsys-rep   # §4.1
```

`nsys` cannot write its report onto the GPFS home (`errno 524`); point `-o` at
`$SCRATCH`, or recover a stranded `.qdstrm` with
`Nsight_Systems/host-linux-x64/QdstrmImporter -i` from a Lustre directory.

| switch | effect |
|---|---|
| `FASTDF_STEP1=auto` | default: the block-sparse first contraction where the tile fill passes `TILE_MAX_FILL`, GPU4PySCF's dense cuTENSOR one otherwise |
| `FASTDF_STEP1=tensor\|bgemm\|tiled` | force the dense contraction, the cuBLAS batch over the AO index, or the block-sparse one unconditionally |
| `FASTDF_TILE=<n>` | AO tile size (default 16; 16/32/64 measured) |
| `FASTDF_TILE_ORDER=native\|rcm\|rcb\|morton` | the AO ordering the tiles are cut in (default `native`) |
| `FASTDF_TILE_MAX_FILL=<f>` | the gate (default 0.75); 1.0 takes the tiled contraction whenever the pattern is built |
