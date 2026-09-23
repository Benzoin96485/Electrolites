# The tiled unpack as one kernel, and what a localised occupied basis is actually worth

`docs/DF_WORK_LIST.md` **W14** and **W15** — the two items
`docs/README_fastdfblock.md` left behind when W5 closed.  They are the two
halves of the same sentence in that write-up: W5's block-sparse exchange
contraction reaches its row-compressed buffer through a CuPy scatter where
GPU4PySCF's dense path has a hand-written CUDA kernel (**W14**), and the
sparsity a tile *cannot* address is reachable only through a localised
occupied basis (**W15**).

One is small, concrete and pays.  The other is the larger of the two on paper
and this round's job was to price it before anything was written, which is
what the item itself asks for.

---

## 0. Summary

* **W14 is done and it is bigger than the item priced it.**  The tiled unpack
  is one CUDA kernel now, in two variants picked by which of the packed
  block's strides is unit.  In a profiled 284-atom `get_jk` it is **0.279 s
  against the dense `decompress_and_transpose`'s 0.618** — the item asked for
  *at or below* and the answer is 2.2x below — and against the CuPy scatter it
  replaces, **0.786 s**, it is 2.8x.  On a whole call that is **8215.77 ms →
  7638.62**, which moves W5's block-sparse contraction from **1.19x to
  1.28x** at 284 atoms.  The `npair * blk` staging buffer is gone (0.67 GB at
  284 atoms, and 5.44 → **4.61 GB** of per-block footprint), the two scatters
  are **bit-identical** on every stored pair up to 5037 AOs, and
  `FASTDF_TILE_UNPACK=cupy` keeps the old one runnable.
* **Two of W5's conclusions move with it.**  The scatter's cost is now exactly
  linear in the auxiliary block, so the block ratio rises monotonically to
  1.25x at block 256 instead of peaking at an intermediate block
  (`README_fastdfblock.md` §3.1); and the break-even tile fill moves from
  **0.80 to 0.90**.  The gate is deliberately **not** raised — §4.2.
* **W15 is done, and it is a rejection.**  Its premise is right: a pivoted
  Cholesky reproduces the 284-atom density to 1.5e-15 in 0.35 s and its
  columns touch 266 of 2697 AOs on 66 of 284 atoms.  Its **falsification
  criterion does not fire** — the supports are small — and the item fails
  anyway, on something the criterion did not name: the *row* support of even a
  single localised orbital is **2697 of 2697**, because an orbital's 266 AOs
  share a stored cderi pair with the whole basis.  Step 2, 3.45 s of an 8.93 s
  call, therefore gets **nothing** (1.000x), and step 1 gets less than the
  contraction that already ships.
* **And that is not a "needs more work".**  Screened with an **oracle** — the
  real `rhok`, a magnitude threshold, no bound and no cost — the blocked
  pattern at the threshold that just meets `tests/test_df.py`'s 1e-11 keeps
  **99.8 %** of the blocks.  Halving the blocked cost costs `relK ~ 1e-4`.
  Two controls close the two escape routes: the canonical factor keeps 4.3x
  more elements, so the localisation really is doing the work; and a single
  auxiliary function is no less blocked than 64 of them (0.995 against 0.998),
  so localising the auxiliary index is not the missing ingredient.

Everything here is one A100-SXM4 of a whole `--qos=debug` node,
`CUDA_VISIBLE_DEVICES=0`, GPU4PySCF 1.8.1, def2-SVP with its own jkfit
auxiliary set.  §3's rows and §4's are on an 80 GB card and §4.1's and §4.2's
on 40 GB ones; `DF_WORK_LIST.md` §5.6 says why that has to be stated.

---

## 1. W14: what the two scatters are

`fastdf`'s tiled first contraction runs on `RowCompressed`, whose buffer holds
the tile rows of the unpacked cderi block with the absent tile columns
squeezed out (`electrolites/_dfblock.py`).  Reaching it means taking the
packed block — `(nL, npair)`, one column per stored AO pair — and writing each
pair's `nL` values into **two** rows of that buffer, the pair's own and its
mirror's.  The destination rows are a property of the pattern, computed once
in `RowCompressed.__init__` as `_d1` and `_d2`.

The dense path does the same job with `fill_symmetric`, which is
`libcupy_helper`'s `decompress_and_transpose`: one CUDA kernel, a `divmod` per
pair to recover `(i, j)` from the packed address, and a write into
`(nao, nao, nL)`.

W5 shipped the tiled version as two CuPy operations instead:

```python
blk = self.tbuf[:npair * nL].reshape(npair, nL)
cp.copyto(blk, cderi_blk.T)         # a transpose into a staging buffer
flat[self.d1, :nL] = blk            # a fancy-index scatter
flat[self.d2, :nL] = blk            # and its mirror
```

That is three passes over the data where one would do, and it carries an
`npair * blk`-word staging buffer — **0.67 GB at 284 atoms** — whose only
purpose is to hold the transpose between the two.  The staging buffer is
itself the residue of `docs/README_fastdfblock.md` §4.1: the transpose used to
allocate its target per block, `cudaFree` drained the prefetch pipeline 114
times a call, and hoisting the allocation out of the loop was what turned
0.90x into 1.21x.  Hoisting it is not the same as not needing it.

## 2. The kernel

`electrolites/_dfblock.py` now writes the row-compressed buffer straight from
the packed block, which is W14 as the item states it: *"the same kernel
GPU4PySCF has, with a destination table instead of a `divmod`"*.

The access pattern is fixed by the two layouts and it is worth being explicit
about which of them is the awkward one.  The write is `nL` **contiguous**
doubles per pair per destination — a whole row of the buffer — so the store
wants the auxiliary index `L` to vary within a warp.  The read is one
*column* of the packed block.  Whether those two agree depends on which of
the block's strides is unit, and `fastdf._blocks` produces both:

* **pair major**, `in[L*ldin + p]` — a row slice of a `(naux, npair)`
  tensor, which is every block of a host-staged cderi and of a device-resident
  `float64` one.  The read coalesces along `p` and the write along `L`, so
  they disagree and the transpose goes through a 32x32 shared tile.  The tile
  is padded to `[32][33]`: unpadded, every thread of a store phase reads the
  same two shared-memory banks and the read serialises sixteen ways.
* **auxiliary major**, `in[L + p*ldp]` — the transpose of a
  `(npair, blksize)` buffer, which is what W7's widener
  (`electrolites/_dffp32.py`) hands back for a device-resident `float32`
  tensor.  Here the read is *already* coalesced along `L`, which is the
  direction the write wants, so there is no transpose, no shared memory and
  no `__syncthreads` — a second, simpler kernel.

A block whose strides are unit in neither index raises rather than falling
back.  That is deliberate: a fall-through here would be a silent 763 MB
temporary per block, which is exactly the failure §4.1 of the W5 write-up
spent a profiler finding.

Two things do **not** change.  The buffer is still zeroed once per `get_jk`
call rather than per block, because the AO pairs that sit inside a present
tile pair but were dropped by the screening are never written and have to be
zero; and the diagonal pairs still get two stores of the same value into the
same row, as the two fancy-index scatters also did.

`FASTDF_TILE_UNPACK=cupy` puts the old one back, and
`fastdf.stats()['tile_unpack']` says which ran — the two are bit-identical, so
nothing numerical can.

## 3. What W14 measures

`benchmarks/df_block_bench.py --unpacks kernel,cupy` times the two scatters
against each other and against GPU4PySCF's dense one, on the same pattern and
in the same round robin.  def2-SVP, `native` order, block 128, one
A100-SXM4-80GB of a whole `--qos=debug` node:

| | nao | tile | fill | dense `fill_symmetric` | **kernel** | CuPy scatter | vs dense | vs CuPy |
|---|---|---|---|---|---|---|---|---|
| penicillin V, 42 | 430 | 16 | 98.3 % | 0.16 ms | **0.16** | 0.51 | 1.00x | 3.2x |
| taxol, 113 | 1123 | 16 | 94.9 % | 0.80 | **0.70** | 3.23 | 1.14x | 4.6x |
| taxol, 113 | 1123 | 32 | 98.2 % | 0.80 | **0.70** | 2.97 | 1.14x | 4.2x |
| PfPMT, 284 | 2697 | 16 | 61.9 % | 1.99 | **1.76** | 7.43 | **1.13x** | **4.2x** |
| PfPMT, 284 | 2697 | 32 | 72.3 % | 1.99 | **1.77** | 7.42 | 1.12x | 4.2x |

and HcgC at block 32, which is what a dense `(nao, nao, blk)` buffer allows
at 545 atoms:

| | nao | tile | fill | dense | **kernel** | CuPy | vs dense | vs CuPy |
|---|---|---|---|---|---|---|---|---|
| HcgC, 545 | 5037 | 16 | 55.5 % | 1.46 ms | **1.37** | 3.08 | 1.07x | 2.2x |
| HcgC, 545 | 5037 | 32 | 67.4 % | 1.46 | **1.36** | 3.08 | 1.07x | 2.3x |

`--check` verifies each scatter against `fill_symmetric`'s own output **pair
by pair** rather than comparing the two contractions, which is a weaker
statement about a scatter: `max |tiled - fill_symmetric|` over every stored
pair is **0.0** on every case and every ordering, up to 5037 AOs and
2 094 539 pairs.

**The item asked for parity with GPU4PySCF's kernel and the answer is 1.13x
past it.**  W14 priced itself at "perhaps 0.1 s of an 8.9 s call" on the
theory that the tiled unpack was 16 % behind the dense one; it is 13 % ahead,
because the two kernels move *different amounts of data for the same job*.
Both write `2 * npair * nL` doubles — `fill_symmetric` fills only the stored
pairs and their mirrors, it does not touch the rest of its `(nao, nao, nL)`
buffer — but the tiled destination is `fill` times smaller, so the same
scattered writes land in a denser working set.  At 284 atoms, block 128, both
move 2.48 GB: the dense kernel at **1.25 TB/s**, W14's at **1.42**.

**And the CuPy scatter was not bandwidth-bound, which is what the item said.**
Its falsification criterion — "falsified if the scatter is bandwidth-bound" —
does not fire: three passes (a transpose into the staging buffer, then two
fancy-index scatters each of which re-reads it) at 7.43 ms against one pass at
1.76 is 4.2x, and 4.2 is about what three passes plus a gather of a 64-bit
index array should cost.

### 3.1 The block size, and a conclusion of W5's that this reverses

Same system, tile 16, sweeping the auxiliary block:

| blk | dense | **kernel** | CuPy | step 1 | whole block |
|---|---|---|---|---|---|
| 32 | 0.52 ms | **0.45** | 0.97 | 0.90x | 0.93x |
| 64 | 1.00 | **0.88** | 2.57 | 1.30x | 1.18x |
| 128 | 1.99 | **1.75** | 7.36 | 1.33x | 1.19x |
| 256 | 3.97 | **3.51** | 14.80 | 1.47x | **1.25x** |

**The kernel is exactly linear in the block and the CuPy scatter was not.**
0.45, 0.88, 1.75, 3.51 is a clean factor of two per doubling; 0.97, 2.57,
7.36, 14.80 is 2.6x, 2.9x and 2.0x.  `docs/README_fastdfblock.md` §3.1
concluded from the old numbers that "the ratio is best at an *intermediate*
block, which is not where either curve peaks"; with the scatter linear, the
block ratio rises monotonically with the block instead, 0.93x to 1.25x.  That
compounds with the staging buffer going away, because `_blksize` spends the
freed memory on a larger block.

### 3.2 Memory

The staging buffer is `npair * blk` words and it is gone.  Per block at 284
atoms and tile 16, against the dense path's single `(nao, nao, blk)` array:
**4.61 GB against 7.45**, where W5 shipped 5.44.  At tile 32 it is 5.39
against 7.45.  `fastdf._blksize` no longer counts a pair-wide buffer for the
tiled path, so that 0.83 GB at block 128 comes back as auxiliary functions.

## 4. What it is worth on a whole call

`benchmarks/df_jk_bench.py --tiled 16 --tile-unpacks kernel,cupy`, one
process, one cderi tensor, at a fixed density, 284 atoms with the cderi on the
host.  Every row is its own round robin against the *dense* first contraction
forced, because `fastdf`'s default would otherwise be compared against itself:

| PfPMT, 284 atoms | reference | tiled | |
|---|---|---|---|
| dense first contraction | 9791.19 ms | | |
| tiled, CuPy scatter (what W5 shipped) | | 8215.77 | 1.19x |
| **tiled, W14's kernel** | | **7638.62** | **1.28x** |

**W14 is worth 577 ms of an 8.2 s call — 1.076x — and it moves W5's headline
from 1.19x to 1.28x.**  In the same round robin stock GPU4PySCF reads
12398.95 ms and `fastdf` 7645.22, which is **1.62x**; the `s1=dense` ablation
reads 10322.69, so the tiled contraction is 1.35x of that total.  `relK` is
2.5e-13 with either scatter, as it must be — they are bit-identical.

### 4.1 The stage table, which is where the item's criterion is written

W14's success criterion is stated inside a *profiled* call — "the tiled
unpack at or below the dense one's 0.647 s in a profiled 284-atom call, and
the staging buffer gone".  `benchmarks/df_block_profile.py --unpacks
kernel,cupy` puts an NVTX range around every stage of every block for all
three variants in one process on one tensor; `nsys stats --report
cuda_gpu_kern_sum --filter-nvtx=<variant>` is the table.  One `get_jk` per
variant, PfPMT, cderi on the host, tile 16, block 116, on an
A100-SXM4-**40GB** node (`DF_WORK_LIST.md` §5.6 — this is not the card §3's
rows were taken on):

| | dense | tiled, CuPy | **tiled, W14** |
|---|---|---|---|
| step 1 | 6.419 s, 170 cuTENSOR GETT | 4.964 s, 19 266 GEMM | **4.979 s**, 19 266 |
| step 2 (`dsyrk`) | 3.497 s | 3.450 s | 3.450 s |
| **unpack** | **0.618 s** `decompress_and_transpose` | **0.786 s** = 0.492 copy + 0.294 scatter | **0.279 s** `unpack_pair_major` |
| Coulomb pass | 0.135 s | 0.132 s | 0.132 s |
| **wall** | **10682.55 ms** | **9420.36** | **8932.60** |

**0.279 s against the dense kernel's 0.618 — the criterion asked for at or
below and the answer is 2.2x below it.**  Nothing else in the table moves:
step 1 is 4.964 against 4.979 s and step 2 agrees to three digits, which is
the check that the two scatters really are the same computation.  The wall
difference, 488 ms, is the unpack difference, 507 ms, to within the noise of
one call — so this stage is now *attributed*, not inferred, which is what
`DF_WORK_LIST.md` §5.7 asks for after W5 spent a round learning it.

**And the gate does not move.**  At taxol the fill is 0.949, `_use_tiled`
declines, and forcing the tiled path is still a loss end to end: **0.94x**
with the kernel against 0.90x without.  The unpack was never the reason that
gate had to be conservative — what sets it is the tiled contraction's flop
rate, 15.0 against 18.2 TF/s at 284 atoms — but `fastdf`'s own comment said
it was, and that sentence was wrong and has been corrected.

### 4.2 The break-even fill moved, and the gate deliberately did not follow it

The cheaper unpack shifts where the tiled path starts losing.  Sweeping the
tile size sweeps the fill, which is the only way to get intermediate points
out of five molecules (`benchmarks/df_block_bench.py --tiles 32,64,128`):

| fill | system / tile | whole block, **kernel** | whole block, CuPy |
|---|---|---|---|
| 55.5 % | HcgC / 16 | **1.22x** | 1.20x |
| 61.9 % | PfPMT / 16 | **1.19x** | 1.10x |
| 67.4 % | HcgC / 32 | **1.23x** | 1.21x |
| 72.3 % | PfPMT / 32 | **1.15x** | 1.08x |
| 78.2 % | HcgC / 64 | **1.12x** | 1.10x |
| 82.7 % | PfPMT / 64 | **1.04x** | 0.98x |
| 89.4 % | HcgC / 128 | **1.04x** | 1.02x |
| 89.9 % | PfPMT / 128 | **1.05x** | 0.98x |
| 94.9 % | taxol / 16 | 0.86x | 0.69x |
| 98.2 % | taxol / 32 | 0.95x | 0.75x |

(The CuPy column is computed from the same round robin's stage times rather
than printed; the PfPMT and taxol rows at tile 16 and 32 come from a job on an
80 GB card and the rest from 40 GB ones, `DF_WORK_LIST.md` §5.6 — the one row
both jobs share, PfPMT / 32, reads 1.15x in each.)

Break-even was "a fill near 0.80" with the CuPy scatter
(`docs/README_fastdfblock.md` §4.2) and is **near 0.90** with the kernel.
`TILE_MAX_FILL` is nevertheless left at **0.75**, and that is a decision
rather than an oversight: at the shipped tile of 16 the five systems measured
sit at 55.5, 61.9, 94.9, 98.3 and 98.9 % and **not one of them lands in the
band the gate would newly open**, so raising it would change no measured run
and would be raising a threshold into territory nothing has been timed in.
The W5 round's own rule applies — a gate that declines a win is the right way
round.  What the band *does* now contain is the two conservative rejects that
write-up recorded at 545 atoms and tile 64; they are still rejects, and they
are still worth 1.12x if anyone wants them (`FASTDF_TILE_MAX_FILL=0.8`).

---

## 5. W15: a localised occupied basis, priced before it was written

W15 asks for the density factor's second index to be localised, on the exact
observation that `vk[i,j] = sum_{k,L} rhok[i,k,L] rhok[j,k,L]` is invariant
under an orthogonal rotation of `k`, so the factor may be replaced for free
and to round-off.  `rhok` is then sparse in `(i, k)` as well as `(i, j)`,
which is what every linear-scaling exchange implementation exploits, and it is
the only route to the shell-granular factor no AO tile can address
(`docs/README_fastdfblock.md` §1.1).

The item attaches a precondition and it is the whole of this section: *"measure
the support of a Cholesky-localised occupied set **before** writing anything,
the way `df_block_probe.py` measures the tile fill.  That probe is an
afternoon and this item is not."*  `benchmarks/df_local_occ_probe.py` is that
probe.  It needs no cderi for its structural half and one for its numerical
half, and it reports every fill twice — once at element granularity, which is
the sparsity that exists, and once at (AO tile x orbital group) granularity,
which is the part a blocked contraction can address.

**The premise is exactly right.**  A pivoted Cholesky of the converged density
to rank `nocc` reproduces it to **1.5e-15** at 284 atoms and costs **0.35 s**,
and the columns are local:

| | atoms | nao | nocc | mean AO support | max | mean atoms touched | max | `|D - f f^T|/|D|` after truncation |
|---|---|---|---|---|---|---|---|---|
| penicillin V | 42 | 430 | 92 | 160 | 254 | 29 of 42 | 42 | 7.0e-4 |
| taxol | 113 | 1123 | 226 | 320 | 684 | 68 of 113 | 109 | 8.3e-4 |
| **PfPMT** | **284** | **2697** | **547** | **266** | **683** | **66 of 284** | **152** | **8.8e-4** |

at a support that discards 1e-5 of each column's squared norm.  **The supports
do not grow with the molecule** — 160, 320, 266 AOs at 42, 113 and 284 atoms —
which is the whole content of "localised", and it means the item's own
falsification criterion, *"falsified if the per-orbital supports at 284 atoms
are not small"*, **does not fire.**  It is the wrong criterion.

### 5.1 Where it dies: the row support is the whole basis

`rhok[i,k,:]` can be nonzero wherever some *stored* AO pair `(i, j)` has `j`
in orbital `k`'s support.  At 284 atoms an orbital's support is 266 AOs on 66
atoms — and the set of AOs sharing a stored cderi pair with any of those 266
is **2697 of 2697**.  `|R|max` is the full basis at every group size the probe
measures **including group 1**, one orbital at a time.

That single fact decides the item, because `vk = rhok rhok^T` is a contraction
over `k` of two objects that are dense in `i`.  Step 2 — the `dsyrk`, **3.45 s
of the 8.93 s call** §4.1 measures — has nothing to exploit at all.  The probe
says so directly, at every truncation, for both large systems: `step2/T` is
0.938 to 1.000 at 284 atoms and **exactly 1.000** at 113.

What is left for step 1, at the granularity a GEMM addresses (tile 16, groups
of 16 orbitals ordered by a recursive bisection of their centroids), against
the **0.619** the block-sparse contraction of W5 already delivers there:

| discarded norm | `|D - f f^T|` | step 1, elementwise | **step 1, tile x group** | step 2, tile x group |
|---|---|---|---|---|
| 1e-4 | 2.8e-3 | 0.182x | **0.436x** | 0.938x |
| 1e-5 | 8.8e-4 | 0.270x | **0.550x** | 0.960x |
| 1e-6 | 2.4e-4 | 0.370x | **0.663x** | 0.986x |
| 1e-8 | 1.8e-5 | 0.570x | **0.838x** | 0.998x |
| 1e-10 | 1.1e-6 | 0.738x | **0.934x** | 1.000x |

**This is `docs/README_fastdfblock.md` §1.1 again, from the other side.**  The
sparsity is a factor of five to twenty at element granularity and a factor of
one to two at the granularity a GEMM can address — and W15 was supposed to be
the way *around* that wall, not a second example of it.  At the loosest
truncation in the table, which costs three digits of the density matrix, step
1 would be 0.436x where the contraction that already ships is 0.619x: a 1.4x,
on the smaller half of the call, for an entirely different algorithm.

The orbital grouping is worth stating because it is the half that works: the
same table at group 16 reads **0.550x** with the orbitals bisected by centroid
and **0.970x** in the order the Cholesky pivots produced them.  Batching
orbitals by common support is, as the item predicted, the hard part — and
doing it perfectly still does not clear the bar.

### 5.2 The oracle: screening `rhok` itself, against the error in `vk`

Everything above is structural — it asks where `rhok` *can* be nonzero.  That
is an upper bound and a loose one, because the cderi's entries decay long
before the screening drops them.  `--rhok` therefore builds a real `rhok` from
real auxiliary blocks, screens it at a magnitude threshold with **no bound and
no cost** — an oracle no implementation can have — and measures the relative
error that discarding the rest makes in `vk` on the same block.  That is the
currency W15's success criterion is written in and it is the only thing that
cannot be inferred from a pattern.

PfPMT, 284 atoms, two blocks of 64 auxiliary functions, `elem` over `(i, k)`
after a maximum over the block, `block` at tile 16 x group 8, `1L` the same
two numbers for a **single** auxiliary function:

| threshold | elem | block | step 2 | `1L` elem | `1L` block | **relK** |
|---|---|---|---|---|---|---|
| 1e-12 | 0.941 | 1.000 | 1.000 | 0.827 | 1.000 | 4.3e-13 |
| **1e-10** | **0.785** | **0.998** | **0.996** | 0.573 | **0.995** | **5.5e-11** |
| 1e-8 | 0.486 | 0.971 | 0.947 | 0.264 | 0.955 | 6.6e-9 |
| 1e-6 | 0.184 | 0.818 | 0.697 | 0.069 | 0.703 | 7.2e-7 |
| 1e-4 | 0.033 | 0.405 | 0.226 | 0.007 | 0.227 | 8.9e-5 |

**Read the `relK` column against `tests/test_df.py`'s tolerance of 1e-11.**
At the threshold that just meets it the oracle keeps **99.8 %** of the blocks
and **99.6 %** of what step 2 would do.  W15's success criterion — "an
exchange matrix at the `tests/test_df.py` tolerance with a contraction cost
that is visibly sub-quadratic" — is not merely hard to reach; it is
**unreachable by any implementation**, because the arithmetic that would have
to be removed is carrying the accuracy.  Halving the blocked cost costs
`relK ~ 1e-4`, which is eight orders of magnitude past what every other module
in this repository holds to.

Two controls make that a statement about the method rather than about the
probe.

* **The canonical factor, screened the same way**, keeps 0.786 of the elements
  at 1e-6 where the Cholesky factor keeps 0.184.  So the localisation is doing
  real work — **4.3x** at element granularity — and the block fill still barely
  moves, 0.980 against 0.908.  The sparsity is there; the blocks are not.
* **A single auxiliary function** rather than a block of 64.  If localising the
  auxiliary index as well were the missing ingredient, one `L` would be far
  sparser in `(i, k)` than the maximum over 64 of them.  Elementwise it is —
  0.573 against 0.785 at 1e-10 — and **at block granularity it is not**: 0.995
  against 0.998.  No auxiliary domain, however compact, concentrates this
  pattern, so that direction is measured rather than left open.

### 5.3 The verdict

**W15's premise holds, its falsification criterion is the wrong one, and the
item fails anyway.**  The rotation is exact and free, the pivoted Cholesky is
cheap and its columns are genuinely local at 284 atoms, and none of that
survives being made into a GEMM: the row support of every localised orbital is
the entire basis, so step 2 gets nothing, and step 1 gets less than the
block-sparse contraction W5 already ships unless three digits of the density
matrix are thrown away.

It is **not** recorded as "needs more work".  An oracle screen is the best any
implementation of this idea can do, and at this repository's tolerance the
oracle removes **0.2 %** of the blocks.  What would have to change for W15 to
come back is the *bar*, not the code: a caller that will accept
`relK ~ 1e-6` — which no entry point in `electrolites/` currently does — would
see roughly 0.82x on step 1 and 0.70x on step 2, about 1.3x on the call,
against the 1.28x §4 already has in hand for a kernel and a gate.

The half of W15 that is worth keeping is in the probe: the localisation
machinery is forty lines, it is exact, and
`benchmarks/df_local_occ_probe.py` will re-measure all of this on any system
with a cached density in about a minute.  If the accuracy bar ever moves, the
measurement is already written.

---

## 6. Tools

```bash
# W14
python benchmarks/df_block_bench.py --cases taxol:0:0 --unpacks kernel,cupy --check
python benchmarks/df_block_bench.py --cases PfPMT:0:-1 --tiles 16,32 \
    --orders native --unpacks kernel,cupy --nl 128
python benchmarks/df_jk_bench.py --xyz benchmarks/molecules/PfPMT.xyz \
    --charge -1 --basis def2-svp --cpu-cderi --tiled 16 \
    --tile-orders native --tile-unpacks kernel,cupy --ndm 1 --reps 2
nsys profile -t cuda,nvtx -c cudaProfilerApi -o $SCRATCH/prof \
    python benchmarks/df_block_profile.py --xyz benchmarks/molecules/PfPMT.xyz \
    --charge -1 --basis def2-svp --cycles 1 --cpu-cderi --tile 16 \
    --unpacks kernel,cupy
nsys stats --report cuda_gpu_kern_sum --filter-nvtx=tiled-kernel $SCRATCH/prof.nsys-rep
python tests/test_df.py --ablate          # both scatters, two tile sizes

# W15
python benchmarks/df_local_occ_probe.py --cases PfPMT:0:-1 \
    --dm $SCRATCH/dfloc_dm --groups 1,8,16,32,64 --hist
python benchmarks/df_local_occ_probe.py --cases PfPMT:0:-1 \
    --dm $SCRATCH/dfloc_dm --groups 8,16,32 --rhok --rhok-blocks 2
```

`nsys` cannot write its report onto the GPFS home (`errno 524`); point `-o` at
`$SCRATCH`.

| switch | effect |
|---|---|
| `FASTDF_TILE_UNPACK=kernel` | default: W14's one-pass CUDA scatter |
| `FASTDF_TILE_UNPACK=cupy` | the transpose into a staging buffer plus two fancy-index scatters it replaced; bit-identical, 4.2x slower at 284 atoms |
| `FASTDF_TILE_MAX_FILL=<f>` | the gate, still 0.75; §4.2 says why it did not follow the break-even up to 0.90 |

