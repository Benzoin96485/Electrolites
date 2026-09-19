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

**And the crossover that does exist is a kernel-level one that does not reach
the caller.**  Above a measured tile fill the first contraction is **1.31x**
at 284 atoms and **1.38x** at 545, and the exchange block around it **1.09x**
and **1.20x**.  A whole `get_jk` call at 284 atoms — the only end-to-end size
that runs — reads **0.85x, 0.86x, 0.90x, 1.02x and 1.02x** across five paired
measurements in two jobs.  §4 has the table and §4.1 the three explanations
that were checked and did not account for the roughly 1.5 s gap.  **So it
ships off**: `FASTDF_STEP1=auto` opts in, and the gate it then applies is the
tile fill rather than the system size, because §2.1 shows a size threshold
would be actively wrong (taxol in def2-TZVPP has 2636 basis functions against
PfPMT's 2697 in def2-SVP and a tile fill of 98.9 % against 72.3 %).

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
tiling won.  The contraction has to be one call per tile row — 85 of them per
block at PfPMT, 158 at HcgC.  §3 measures what that costs: the tiled
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
| penicillin V, 42 | 430 | 32 | 100.0 % | 0.42 ms | 1.16 ms | 10.3 → 3.8 | 0.37x | 0.48x |
| taxol, 113 | 1123 | 16 | 94.9 % | 5.02 | 6.59 | 14.5 → 10.5 | 0.76x | 0.66x |
| taxol, 113 | 1123 | 32 | 98.2 % | 5.02 | 5.51 | 14.5 → 13.0 | **0.91x** | **0.72x** |
| Co(II) sextet, 113 | 1059 | 32 | 100.0 % | 4.11 | 4.59 | 14.3 → 12.8 | 0.90x | 0.60x |
| PfPMT, 284 | 2697 | 16 | 61.9 % | 56.26 | 42.44 | 18.1 → 14.9 | **1.33x** | **1.09x** |
| PfPMT, 284 | 2697 | 32 | 72.3 % | 56.29 | 44.63 | 18.1 → 16.5 | 1.26x | 1.06x |
| PfPMT, 284, `rcm` | 2697 | 32 | 69.4 % | 56.27 | 42.80 | 18.1 → 16.5 | 1.31x | 1.09x |
| PfPMT, 284 | 2697 | 64 | 82.7 % | 56.27 | 52.76 | 18.1 → 16.0 | 1.07x | 0.97x |

and HcgC at block 32, which is what a dense `(nao, nao, blk)` buffer allows at
that size:

| | nao | tile | fill | step1 dense | tiled | TF/s | step1 | block |
|---|---|---|---|---|---|---|---|---|
| HcgC, 545 | 5037 | 32 | 67.4 % | 90.06 ms | 65.08 ms | 18.2 → 17.0 | **1.38x** | **1.20x** |
| HcgC, 545 | 5037 | 64 | 78.2 % | 90.03 | 77.07 | 18.2 → 16.7 | 1.17x | 1.08x |

Three things to read out of that.

**The gain is the fill, divided by an efficiency the tiling costs.**  That
relation is an identity — the flop counts differ by exactly the fill — so the
only thing the clock adds is the value of the efficiency, and it is
**0.82 to 0.93**: 85 to 158 cuTENSOR calls per block where there was one cost
7 to 18 % of the dense contraction's flop rate, and nothing worse.  That is
the good news in this table and it is why the result is a ceiling argument:
the executor is close to as good as a tiled executor gets, so §1's fill is
what decides, for this implementation and for the `cuSPARSE` BSRMM and custom
batched-GEMM ones the item also proposed.

**Bigger tiles lose more fill than they win back in efficiency.**  T=64 gains
only 0.5–1.6 TF/s over T=32 and pays ten points of fill, and it is behind at
both large sizes.  T=16 has the lowest fill and twice the calls, and it is
0.03x *ahead* of T=32 at 284 atoms (1.09x against 1.06x in the native order,
where T=32 needs the `rcm` reordering to reach 1.09x too).  **32** is the
shipped default because it is the best of the three at 545 atoms and within
0.03x of the best at 284; T=16 was not timed at 545.

**The unpack is the part that is not finished.**  The tiled scatter is a CuPy
fancy-index assignment where GPU4PySCF's is a CUDA kernel, and it costs
**8.4 ms against 2.0 ms** at PfPMT and 3.6 against 1.3 at HcgC.  That is the
whole of the gap between the contraction's ratio and the block's: give the
tiled unpack the dense one's cost and PfPMT/T=16 goes from 1.09x to **1.18x**
and HcgC/T=32 from 1.20x to 1.22x.  `DF_WORK_LIST.md` **W14** is that kernel.

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

The tiled buffer is exactly `fill` times the dense `(nao, nao, blk)` one:
**4.61 GB against 7.45** at PfPMT/T=16 and **4.38 against 6.50** at
HcgC/T=32.  Against that, the density factor has to be gathered onto the tile
rows once per `get_jk` call, and that array is `fill * nao / TILE` times as
tall as the factor — at 545 atoms and tile 32, as large as the buffer it
saved.  `_blksize` subtracts it from the budget rather than dividing it into
the block, because it does not shrink when the block does.  Net, the tiled
path's peak footprint is roughly the dense path's; W5's "it also removes the
dense work buffer" is true of one array and false of the pair.

---

## 4. What ships, and the end-to-end result that does not follow from §3

**It ships off.**  `FASTDF_STEP1` stays `tensor`; `auto` is the opt-in that
applies §4.2's gate, and `tiled` forces the branch unconditionally.  The
reason is not §1's ceiling — it is that **§3's block ratio does not survive
the call it is a part of**, and this round could not find out why.

Measured end to end, at a converged density, both implementations in one
process on one cderi tensor, each row its own round robin of two
(`benchmarks/df_jk_bench.py`):

| | tile | dense reference | tiled | |
|---|---|---|---|---|
| taxol, 113 atoms, cderi on the device | 32 | 382.02 ms | 469.93 ms | **0.81x** |
| PfPMT, 284 atoms, cderi on the host | 16 | 10466.57 ms | 10261.91 ms | **1.02x** |
| PfPMT, 284 atoms, `rcm` | 16 | 10468.78 | 10274.59 | **1.02x** |
| PfPMT, 284 atoms | 32 | 9961.05 | 11625.47 | **0.86x** |
| PfPMT, 284 atoms, second job | 16 | 10738.19 | 11896.06 | **0.90x** |
| PfPMT, 284 atoms, second job | 32 | 10323.71 | 12214.31 | **0.85x** |

Four of the five 284-atom readings are **below 1.0x**, the two that are above
it are 1.02x and did not reproduce in a second job, and §3's microbenchmark of
the same three kernels on the same pattern says 1.09x.  **So the block-sparse
contraction does not pay in a real `get_jk` call at the largest size that
runs**, and the honest summary of W5 is that it is a kernel-level win that
does not reach the caller.

For scale, from the second job's eight-variant table on the same call: stock
GPU4PySCF **12907.43 ms**, `fastdf` as shipped **10579.82** (1.22x, which is
the four items the module already had), its `s1=dense` ablation 10575.78, and
`no-resident` 10627.43.

### 4.1 The gap is about 1.5 s per call and is not explained

The kernels §3 times — unpack, first contraction, `dsyrk` — sum to 1.09x
tiled at 284 atoms and the call comes out at 0.85–1.02x.  Three explanations
were checked and none of them accounts for it:

* **Block size.**  The real call runs near `MAX_BLK = 256` where §3.1 already
  puts the block ratio at 1.04x rather than 1.09x.  That is part of the gap
  and nowhere near all of it.
* **Buffer churn.**  `_blksize` re-derives the auxiliary block from
  `get_avail_mem()` on every call, and the tiled buffer is 8.8 GB, so a block
  that moves by one between calls reallocates and re-zeroes all of it.  This
  looked like the answer — it would have been about the right size — so the
  pattern was made to own its block size (`_tiles` fixes `ldL` at the first
  call and `get_jk` takes it back).  **It made the tiled path slower**, 10262
  → 11896 ms at T=16, because the block is then frozen at whatever the memory
  state was when the pattern was first built.  The fix is kept because a
  cached multi-gigabyte buffer should not be resized from free memory, but it
  is not the explanation.
* **The resident cderi prefix.**  The tiled buffer competes with `fastdf`
  item 4's resident rows (2298/13365 in one job, 1078/13365 in the other), so
  more of the 86 GB streams.  But `no-resident` is 10627 against 10580 here —
  half a per cent — so residency is worth nothing on this call to begin with,
  exactly as `DF_WORK_LIST.md` §3.3 found.  Not the explanation either.

What is left, and is the leading candidate rather than a result: the
microbenchmark runs its kernels on a block already on the device with nothing
else in flight, and the real call runs them interleaved with a saturated PCIe
stream on a second CUDA stream and its events.  The tiled path issues 85 to
169 contractions and two scatters per block where the dense path issues one
contraction and one kernel — about 5400 launches per call against 130 — and
every one of them is scheduled against the prefetch.  **Measuring that needs a
profile of the block loop, not another wall clock**, and it is the first thing
to do for anyone who picks this up: if it is launch scheduling, `W14`'s unpack
kernel and a batched contraction over tile rows of equal width would both
attack it; if it is not, W5 is finished.

### 4.2 The gate

`_use_tiled` takes the block-sparse contraction when the tile fill is at or
below **`TILE_MAX_FILL = 0.75`**.  Over the 28 timed (system, tile, ordering)
rows behind §3 — five systems, three tile sizes, two orderings, the 545-atom
one at two tiles — that gate separates every win from every loss on the
exchange block without a misclassification: it rejects penicillin V, both 113-atom
systems at every tile, and PfPMT at T=64 (82.7 % fill, 0.97x); it accepts
PfPMT at T=16 and T=32 and HcgC at T=32.  `rcm`'s 73.5 % at PfPMT/T=64 is the
one marginal acceptance (1.02x on the block), and a gate of 0.70 would drop
it.

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
DF-K itself", worth 12.1 s of a 12.38 s build — what the item actually buys is
nothing on the call, and §1.1 is most of the reason: two thirds of the
sparsity it priced is not addressable by a tile at all.

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
python benchmarks/df_jk_bench.py --xyz ... --tiled 32 --tile-orders native
python tests/test_df.py --ablate            # the correctness bar, tiled included
```

| switch | effect |
|---|---|
| `FASTDF_STEP1=tensor` | default: GPU4PySCF's dense cuTENSOR contraction |
| `FASTDF_STEP1=auto` | opt in: the block-sparse first contraction where the tile fill passes `TILE_MAX_FILL`, the dense one otherwise |
| `FASTDF_STEP1=bgemm\|tiled` | force the cuBLAS batch over the AO index, or the block-sparse contraction unconditionally |
| `FASTDF_TILE=<n>` | AO tile size (default 32; 16/32/64 measured) |
| `FASTDF_TILE_ORDER=native\|rcm\|rcb\|morton` | the AO ordering the tiles are cut in (default `native`) |
| `FASTDF_TILE_MAX_FILL=<f>` | the gate (default 0.75); 1.0 takes the tiled contraction whenever the pattern is built |
