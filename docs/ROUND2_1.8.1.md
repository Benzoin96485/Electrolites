# Second round on GPU4PySCF 1.8.1: the five openings, and what each was worth

Environment: conda env `barw-dev`, `gpu4pyscf-cuda12x 1.8.1`, one A100 of a
Perlmutter GPU node with `CUDA_VISIBLE_DEVICES=0`, `direct_scf_tol = 1e-13`,
level-3 grids, one process at a time.  §2's profile is an A100-SXM4-**40GB**,
the same card class as `docs/UPGRADE_1.8.1.md`; §1's `PfPMT` per-class table
is an A100-SXM4-**80GB**; the ethanol-cluster rows and the compile times are
a shared login-node A100-PCIE-40GB, where absolute times move with what else
is on the card but every variant of a comparison runs back to back in one
process.  Each table says which.  Every number below was measured in this
round; nothing is quoted from the previous one.

`docs/UPGRADE_1.8.1.md` closed with five openings.  This is what happened to
each.  **One of them turned out to rest on a measurement artifact, and saying
so is the most useful result in this document** -- see §2.

---

## 0. Summary

| # | Opening | Outcome |
|---|---|---|
| 1 | `get_j` at 1.33x; the four `md_j` classes 1.8.1 does not unroll | **seventeen classes written**, which is every class an spdf basis reaches. The J build is **1.82x** on `PfPMT`/6-31G\* and **2.07x** on `PfPMT`/def2-TZVP, where the 64 % of the work that used to fall through at 1.00x now runs at **2.49x**; on a 54-atom cluster in def2-TZVP the whole build goes **1.05x -> 5.12x**.  Correct to 1e-12 on the AO-space J across five basis sets and four systems |
| 2 | "a third of the SCF is DIIS, fp64 diagonalisation and grid construction" | **refuted, with an instrumented profile**: it is **1.9 %**.  The 3.33x-vs-2.13x gap that suggested a ceiling was a fixed-density measurement compared against a converged wall clock; §2 |
| 3 | three JoltQC ideas | (a) the multi-dimensional launch search: **built and used**, worth up to 2.6x on a class over a hand-set table; (b) the `pair_vk` block reduction: **rejected on a measured ceiling** -- deleting *every* `vk` atomicAdd is worth 1.1 % at 6-31G\* and 2.2 % at def2-TZVP, so a block reduction cannot pay; (c) the log-magnitude AO screen: **the headroom was a `minao` artifact** -- 65 % of the XC GEMM work on the initial guess, **4 % on a converged density** -- and the obvious implementation measured 0.28x anyway |
| 4 | split the module so a 6-31G\* job does not compile spdfg kernels | **done**: a 6-31G\* job's first-run NVRTC goes **162 s -> 37 s (4.3x)** -- `fastk` 31.7 -> 11.9 s, `fastejk` 58.5 -> 16.5 s, `fastj` 71.6 -> 8.9 s.  It is also what makes opening 1's coverage affordable |
| — | *(end to end)* | `PfPMT` SCF **127.2 -> 55.1 s = 2.31x** (was 2.13x), gradient 19.4 -> 12.5 s = 1.55x; §8 |
| 5 | the XC + VV10 single-pass grid fusion `fastrsh` took down with it | **restored** at a 1.8.x-shaped patch point, and worth **1.036x** on `get_veff` for wB97M-V.  Timing it exposed that it was declining on every non-tiny system, because 1.8.x permutes the two grids -- a bug no correctness test can see; §5 |
| — | *(found on the way)* | the fixed-density table's **gradient** row depends on `OMP_NUM_THREADS`, because its unpatched half is CPU libcint; §6 |

---

## 1. The `md_j` classes GPU4PySCF does not unroll (opening A2)

### What was written

`codegen/gen_j_high.py` writes a McMurchie-Davidson J kernel from the
angular-momentum class `(lij, lkl)` alone.  There is nothing to lift for these
classes -- GPU4PySCF unrolls twelve, up to `lij+lkl == 5`, and sends the rest
to one general kernel, `md_j_1dm_kernel` -- so this is the same *writing*
route `gen_khigh.py` took for the exchange build, where writing returned
1.4x-5.3x on classes that lifting returned 1.07x-1.19x on.

Three things the general kernel pays for its generality:

1. **The Hermite index table.**  Its contraction reads
   `Rt[Rt2_address[k*nf3ij+i] * nsq_per_block]`: a `uint16` load, an integer
   multiply-add, and only then the `double` it wanted -- twice for each of the
   `nf3ij*nf3kl` pairs, 1225 of them for `(4,4)`.  Written per class the
   address is a compile-time constant, and the index load and a two-deep
   dependent load chain go with it.
2. **The `Rt` recurrence.**  `iter_Rt_n` reads a parent index and a `t/u/v`
   factor from `__constant__` memory for every element of every level, into a
   `double Rt_tmp[31]` whose trip count is not a compile-time constant.
   Written per class it is straight-line code with literal indices and a
   register budget the class fixes -- 11 doubles for `(3,3)`, 21 for `(4,4)`,
   57 for `(6,6)` -- instead of the one `double Rt_tmp[31]` every class gets
   alike, indexed by a loop the compiler cannot bound.
3. **The scaffolding the twelve unrolled classes already avoid** -- the
   incomplete-gamma values staged through shared memory, the two fp64
   divisions and the `sqrt` per quartet, and `boys_fn`'s range-separation
   branches carried at run time although a Coulomb build never has
   `omega != 0`.

`gout_stride` threads cooperate on one quartet, and the block is laid out so
that `gout_id = threadIdx.y` is **constant across a warp**.  That is what makes
the per-`gout_id` `switch` in the emitted code free -- it is warp-uniform, so
it never diverges.  The same construction *inside* a warp is what cost the
per-lane unrolled exchange kernels up to 3.7x on the wide classes; the
distinction is the whole reason literal indices are affordable here.

### Two bugs the writing exposed, and the oracle that found them

**A padding lane that was not zeroed.**  GPU4PySCF's general kernel clamps
`task_ij` for the `pair_ij_loc` lookup and then *shadows* it with a fresh
unclamped copy inside the ket loop; the first generator here kept only the
clamped one, so `fac = 0.` never fired for the padding lanes of a partly
filled bra tile and they contributed a small spurious quartet -- to `vj_kl`
only, since `vj_ij` is guarded separately.  It showed as a relative 1.5e-5,
and **no physical density exposes it**: a `minao` guess is exactly zero on the
polarisation shells these classes need, so every check against it passes
vacuously.  It took a dense random density to see it at all.

**Block-level rather than batch-level screening.**  1.8.1 screens a
(bra batch, ket batch) pair against `q_cond_ij[task_ij0]` and
`q_cond_kl[task_kl0]` at *batch* granularity.  The kernels lifted from 1.7.0
substitute the block's first pair, which was equivalent there; on 1.8.1 it
over-screens by up to 1e-3 relative.  The written classes now read the batch
value, which is both correct and what GPU4PySCF does.

**The oracle has to be the AO-space J, not the Hermite-space `vj_xyz` the
kernels write.**  The Hermite components of a high-l shell pair are large and
cancel in the transform back to AO space, so two arithmetically identical
kernels that differ only in tile width -- and therefore in the granularity of
the tile-max screen -- differ by up to a per cent in `vj_xyz` and by 1e-14 in
J.  Judging by `vj_xyz` rejects every tile width but the one GPU4PySCF
happens to use; it rejected 24 of 60 correct candidates in the first sweep.
`benchmarks/perclass_j.py --check` and `benchmarks/sweep_j_high.py` both now
check the AO-space J.

### Coverage

Seventeen classes are written, not four: `(3,3)`, `(3,4)`, `(4,2)`, `(4,3)`,
`(4,4)`, `(5,1)`–`(5,5)` and `(6,0)`–`(6,6)`.  That is **every
`(lij|lkl)` class an spdf basis reaches** — 29 of 29, twelve lifted and
seventeen written — so GPU4PySCF's general `md_j_1dm_kernel` now carries none
of the J work at triple zeta, where the previous round measured it carrying
**68 %** of it at 1.00x -- measured again here on `PfPMT`/def2-TZVP at
**64 %**, now running at 2.49x.  The generator takes any `(lij, lkl)`; the limit is
shared memory, and `(6,6)` at order 12 needs 130 KB of the A100's 164 KB.

Going that far is only affordable because of opening 4: the whole `fastj`
family is now 4.7 MB of CUDA and 72 s of NVRTC, but a 6-31G\* job compiles
16 of the 29 kernels and pays **8.9 s**.  Without the module split the
coverage would have cost every 6-31G\* user a minute of first-run compile for
kernels their basis can never launch.

### Correctness

The oracle is the AO-space J against GPU4PySCF's `MD_build_j`, on the same
task list, screening data and Hermite density, with a **dense random
symmetric density** so that every Hermite block is exercised — a `minao`
guess is exactly zero on the polarisation shells and cannot test these
classes at all:

| system | basis | reaches | max relative difference in J |
|---|---|---|---|
| ethanol, 9 atoms | 6-31G\* | `(4,4)` | 2.8e-16 |
| ethanol | cc-pVDZ | `(4,4)` | 1.2e-14 |
| ethanol | def2-TZVP | `(6,6)` | 1.5e-14 |
| ethanol | cc-pVTZ | `(6,6)` | 1.5e-13 |
| ethanol | cc-pVQZ | `(6,6)` + fall-through | 1.0e-12 |
| 6 ethanols, 54 atoms | 6-31G\* | `(4,4)` | 7.9e-14 |
| 6 ethanols | def2-TZVP | `(6,6)` | 9.4e-13 |
| `PfPMT`, 284 atoms | 6-31G\* | `(4,4)` | 1.6e-14 (converged density) |

and the two suites, unchanged in scope from the port, re-run with everything
in this document active:

| suite | what it compares | worst difference | the same suite before this round |
|---|---|---|---|
| `tests/test_full.py`, 15 cases | energy **and** gradient, patched process vs unpatched process | **6.4e-12 Eh**, **3.1e-12** per gradient component | 8.2e-12 Eh, 2.8e-12 |
| `tests/test_grad.py`, 23 cases | gradient at the same converged density, in one process | **2.7e-12** | 2.8e-12 |

`test_full.py` was re-run after §5's grid-comparison fix, since before it the
three VV10 cases in it were passing through the fall-back rather than the
fused path.

### What it is worth

Per-class J timing, `benchmarks/perclass_j.py`, every task of one real J build
run twice from the same screening data and Hermite density.  **On a converged
density** — see the method note below:

| system / basis | GPU4PySCF | `fastj` | |
|---|---|---|---|
| `PfPMT` 284 atoms / 6-31G\*, 2268 AO | 1767 ms | **973 ms** | **1.82x** |
| `PfPMT` 284 atoms / def2-TZVP, 5241 AO | 34269 ms | **16560 ms** | **2.07x** |
| 6 ethanols 54 atoms / def2-TZVP, 774 AO | 966 ms | **191 ms** | **5.06x** |

(The `PfPMT` row is a dedicated node; the ethanol rows are a shared login GPU,
so their absolute times move with what else is on the card -- every variant of
a row runs back to back in one process, so the ratios do not.)

The ablation that says what the *written* classes are worth -- the same build,
with the seventeen handed back to GPU4PySCF and with them kept
(`FASTJ_NO_HIGH=1` is the switch), on a converged density:

| system / basis | GPU4PySCF | the 12 lifted classes only | all 29 |
|---|---|---|---|
| 6 ethanols / def2-TZVP, 774 AO | 1652 ms | 1569 ms (**1.05x**) | **323 ms (5.12x)** |
| 18 ethanols / 6-31G\*, 972 AO | 424 ms | 344 ms (**1.23x**) | **190 ms (2.23x)** |

At triple zeta the twelve lifted classes were worth **5 %** -- because 68 % of
the work was in classes they do not cover -- and the complete set is worth
**5.1x**.  That is the shape of opening A2: not a slow kernel, a missing one.

and the classes this round added, on `PfPMT`/6-31G\*:

| class | GPU4PySCF | `fastj` | | was, before this round |
|---|---|---|---|---|
| `(3,3)` | 91.0 ms | 32.6 ms | **2.79x** | fell through |
| `(4,2)` | 108.3 ms | 29.0 ms | **3.74x** | fell through |
| `(4,3)` | 38.6 ms | 10.2 ms | **3.79x** | fell through |
| `(4,4)` | 32.4 ms | 2.3 ms | **14.26x** | fell through |

At **def2-TZVP on the same 284-atom cluster**, where the previous round put
68 % of the J work on the general kernel, the seventeen written classes carry
**21.8 s of the 34.3 s** GPU4PySCF spends -- 64 % -- and run it in 8.8 s, i.e.
**2.49x on the part that used to run at 1.00x**:

| class | GPU4PySCF | `fastj` | | | class | GPU4PySCF | `fastj` | |
|---|---|---|---|---|---|---|---|---|
| `(3,3)` | 3943 ms | 1525 ms | 2.59x | | `(5,4)` | 711 ms | 417 ms | 1.71x |
| `(3,4)` | 533 ms | 151 ms | 3.52x | | `(5,5)` | 163 ms | 97 ms | 1.68x |
| `(4,2)` | 4704 ms | 1621 ms | 2.90x | | `(6,0)` | 332 ms | 111 ms | 2.98x |
| `(4,3)` | 3468 ms | 1113 ms | 3.12x | | `(6,1)` | 546 ms | 221 ms | 2.47x |
| `(4,4)` | 1266 ms | 647 ms | 1.96x | | `(6,2)` | 534 ms | 263 ms | 2.03x |
| `(5,1)` | 1724 ms | 588 ms | 2.93x | | `(6,3)` | 443 ms | 259 ms | 1.71x |
| `(5,2)` | 1669 ms | 728 ms | 2.29x | | `(6,4)` | 309 ms | 228 ms | 1.35x |
| `(5,3)` | 1308 ms | 694 ms | 1.88x | | `(6,5)` | 108 ms | 77 ms | 1.40x |
| | | | | | `(6,6)` | 78 ms | 15 ms | 5.17x |

**No class of the 29 is slower than GPU4PySCF** on this system: the slowest
row in the whole table is `(2,2)` at 1.26x, one of the twelve lifted ones.
AO-space J on this run: 2.4e-14 relative.

**Method note, and it matters.**  A `minao` initial guess is *exactly zero* on
the polarisation shells, so the Hermite density of every high-`lij` shell pair
is zero, the density-weighted screen removes their quartets, and a per-class
benchmark run against it times launch overhead rather than the kernel.  Timed
that way the same `(4,3)` kernel reads **0.68x** — apparently slower than
GPU4PySCF — and against a converged density it reads 3.79x.  `--cycles N`
runs N SCF cycles first; use it for anything above `lij = 2`.  This is the
same class of error as §2's.

---

## 2. The Amdahl ceiling that is not there (opening 2)

`docs/UPGRADE_1.8.1.md` reported `get_veff` at 3.33x on `PfPMT` while the SCF
wall clock was 2.13x, and concluded that "about a third of that SCF is not in
J, K or XC at all -- it is DIIS, the fp64 diagonalisation, grid construction
and Python overhead", so that "making `get_veff` twice as fast again would
move the `PfPMT` SCF only from 2.13x to about 2.6x".

**That is wrong, and the reason is worth recording.**  The 3.33x is a
*fixed-density* number: `benchmarks/fixed_density.py` builds one `minao` guess
and times each build on it.  A `minao` density screens far harder than a
converged one, so the `get_veff` an SCF actually runs is a different and
slower build than the one that was timed.  Comparing it against a converged
wall clock compares two different quantities, and the leftover is not an
Amdahl tail -- it is the difference between the two densities.

`benchmarks/scf_anatomy.py` settles it by instrumenting the SCF itself: every
method `scf.hf._kernel` calls per cycle is wrapped with a
`cudaDeviceSynchronize()` on both sides.  `PfPMT`, 284 atoms, 6-31G\*, 2268
AO, B3LYP-D3(BJ), 17 cycles, both runs converged to the same energy
(-7652.5600864499 against -7652.5600864500):

| | stock 1.8.1 | share | patched | share |
|---|---|---|---|---|
| **SCF total** | **122.708 s** | | **66.603 s** | (1.84x) |
| `get_veff` | 120.024 s | 97.8 % | 64.612 s | 97.0 % |
| ↳ `get_k` | 83.134 s | 67.7 % | 44.506 s | 66.8 % |
| ↳ `get_j` | 21.712 s | 17.7 % | 12.136 s | 18.2 % |
| ↳ `nr_rks` (XC) | 13.977 s | 11.4 % | 6.816 s | 10.2 % |
| ↳ `grids.build` | 1.139 s | 0.9 % | 1.114 s | 1.7 % |
| `eig` (fp64 `cusolverDnDsygvd`) | 1.019 s | 0.8 % | 1.021 s | 1.5 % |
| `get_fock` (+ DIIS) | 0.115 s | 0.1 % | 0.114 s | 0.2 % |
| ↳ `diis.update` + `extrapolate` | 0.036 s | 0.0 % | 0.035 s | 0.1 % |
| `get_hcore`, `get_ovlp`, guess, lindep | 1.136 s | 0.9 % | 0.768 s | 1.2 % |
| unattributed | 0.259 s | 0.2 % | 0.012 s | 0.0 % |

**DIIS, the fp64 diagonalisation and grid construction together are 2.3 s of
122.7 s, or 1.9 %** -- not a third.  `get_veff` inside the SCF is
120.0 -> 64.6 s = **1.86x**, and the SCF wall clock is **1.84x**; the two
agree, which is what "there is no Amdahl tail here" looks like.  Making
`get_veff` twice as fast again would take this SCF to about **3.5x**, not
2.6x.

Catalogue items **H1** (dense symmetric generalized eigendecomposition) and
**H2** (DIIS extrapolation) therefore stay what the catalogue calls them:
*negative controls*.  On this molecule and this card there is nothing there to
win.  The one caveat that survives is the catalogue's own: `lib/cusolver.py`
falls back wholesale to `scipy.linalg.eigh` past `MAX_EIGH_DIM = 23150`, which
is a cliff at a size neither cluster here reaches, and on a card whose fp64
rate is 1/32 rather than 1/2 the eigendecomposition's share would be ~16x
larger.  Neither applies to an A100.

**Method note.**  The synchronisation points this profile adds inflate the
total: the same SCF without them is 121.6 s stock (`docs/UPGRADE_1.8.1.md`
§8) against 122.7 s here, i.e. under 1 %.  Nested rows marked ↳ are
sub-intervals of their parent and overlap it, so the column does not sum to
100 %.

---

## 3. The three JoltQC ideas (opening 3)

### 3a. A multi-dimensional launch search, in place of a one-dimensional table

Every launch table in this repository is one-dimensional: one number per
angular-momentum class (`minblocks` for `fastj`, `(nsq, gout_stride)` for
`fastk`).  JoltQC searches four independent fragmentation factors instead.
`benchmarks/sweep_j_high.py` searches five, because a McMurchie-Davidson J
kernel has five knobs that are not equivalent to each other:

| knob | what it trades |
|---|---|
| `threadsx` x `threadsy` | the quartet tile a block holds, hence how many quartets share one `Rt` column and what the shared-memory bill is |
| `gout_stride` | threads cooperating on one quartet: the cost of the cooperative `Rt` build against the width of the contraction each thread carries in registers |
| `tiley` | how many ket tiles a block walks, i.e. what the ket `vj` accumulator costs in shared memory, and so what caps blocks per SM |
| `lreg` | how far the `Rt` recurrence runs in registers before it goes through shared memory and takes two barriers per level |
| `dmreg` | whether the bra Hermite density sits in registers or is re-read from shared memory in the ket loop |

They interact -- raising `gout_stride` shrinks the register footprint but
raises the barrier count, and both feed back into occupancy through
`__launch_bounds__` -- which is exactly what a one-dimensional table cannot
see.  Every candidate runs the same task list, screening data and Hermite
density as the shipped kernel and is checked against GPU4PySCF before it is
timed, so a configuration that is fast because it is wrong cannot win.

What it found, against the hand-set configuration, on two systems:

| class | 162 atoms / 6-31G\* | 54 atoms / def2-TZVP |
|---|---|---|
| `(4,2)` | 2.67x | 1.37x |
| `(3,3)` | 1.41x | 1.23x |
| `(4,3)` | 1.80x | 2.00x |
| `(4,4)` | 1.64x | 1.76x |

**The two systems do not agree on the winner**, and the shipped entry is the
configuration within a few per cent of the best on *both*, not the winner on
either -- `(4,3)`, for instance, is 96 % of the best on one and 95 % on the
other where each system's own winner is 79-83 % on the other's.  Every class
prefers `tiley = 8`, which is the point where two blocks fit on an SM.

One thing the search taught that the table could not: **the correctness oracle
had to change before the search was usable at all**.  Judged on the
Hermite-space `vj_xyz`, 24 of 60 correct candidates were rejected -- every
tile width but the one GPU4PySCF happens to use -- because the tile width sets
the granularity of the tile-max screen and the Hermite components of a high-l
pair are large and cancel on the way back to AO space.  Judged on the
AO-space J they all pass at 1e-13.

### 3b. The `pair_vk` block reduction: measure the ceiling first

The previous round implemented JoltQC's warp-aggregated task counter, measured
it inside the noise floor, and reverted it -- the screening loop is bound by
the global-memory latency of `q_cond`/`dm_cond`, not by a shared-memory
atomic.  The `pair_vk` reduction is a different atomic: the **global**
`atomicAdd`s into `vk` that end every density contraction -- 8 per shell
quartet for `(ps|ss)`, 16 for `(ps|ps)`, 36 for `(pp|pp)`, and 3356 of them
written across the generated files.

Rather than build a block reduction and then find out, the ceiling was
measured first.  `fastk_noatomic_generated.cu` and its two companions are the
generated kernels with every `atomicAdd(vk+...)` guarded by a condition that
is never true: the density contraction, the `dm` loads and the address
arithmetic all still run, only the store does not.  The result is wrong by
construction; what it bounds is what *any* scheme for collapsing those atomics
-- a block reduction, a warp aggregation over lanes sharing a `k0` -- could
possibly be worth.

`codegen/gen_noatomic.py` writes the ablation; `benchmarks/konly.py` times it:

```bash
python codegen/gen_noatomic.py
FASTK_SRC=fastk_noatomic_generated.cu \
FASTK2_SRC=fastk2_noatomic_generated.cu \
FASTKH_SRC=fastkhigh_noatomic_generated.cu \
    python benchmarks/konly.py --xyz benchmarks/molecules/PfPMT.xyz \
        --charge -1 --basis def2-TZVP --patch fastk --reps 5
```

`PfPMT`, `get_k` alone on a `minao` density, both halves back to back in one
job, min of 5, one A100-SXM4-40GB:

| basis | with the `vk` atomics | with **none** of them | ceiling |
|---|---|---|---|
| 6-31G\*, 2268 AO | 0.3723 s | 0.3682 s | **1.011x** |
| def2-TZVP, 5241 AO | 3.054 s | 2.988 s | **1.022x** |

(The five repetitions of each spread by 0.1 %, so a 1-2 % difference is real
and a 1-2 % difference is all there is.  The `with` column also reproduces the
previous round's 0.372 s and 3.064 s for the same measurement, which is the
check that this is the same kernel it was.)

**So the whole tail of every K kernel -- all 3356 global `atomicAdd`s -- is
worth 1.1 % at 6-31G\* and 2.2 % at def2-TZVP.**  A block reduction cannot
beat deleting them, and unlike deleting them it costs a shared accumulator,
its barriers and a flush, so what JoltQC's `pair_vk` could realise here is a
fraction of that.  **Not worth building** -- the same verdict as the previous
round's warp-aggregated task counter, for the same reason: these kernels are
latency-bound at a 91 % L1 hit rate with 6 of 16 warps resident, so what is
on the critical path is not the atomic.

The rule this section is really making is the one the earlier idea cost a day
to learn: for an atomic in a kernel that is not atomic-bound, spend an hour
measuring the ceiling before spending a day building the reduction.  The
ablation is 40 lines of `codegen/gen_noatomic.py`.

### 3c. The log-magnitude AO screen: the headroom is real, the obvious implementation is not

GPU4PySCF decides which AOs a grid block keeps with
`GDFTscreen_index_legacy`: a shell survives when the radial sum
`sum_p c_p exp(-a_p r^2)` exceeds `AO_THRESHOLD = 1e-10` somewhere in the
block.  That test carries no `r^l` factor (the estimate that does is in that
file, commented out) and never looks at the density -- while both contractions
in the XC build cost `nao_sub^2 * ngrids`.

`benchmarks/aoscreen_probe.py` measures the headroom from the AO values
GPU4PySCF has already evaluated, before spending anything on a kernel.  On
`PfPMT`, 6-31G\*, 2268 AO, 830 grid blocks -- **and the density matters more
than anything else here**:

| screen | `minao` guess | | after 8 SCF cycles | |
|---|---|---|---|---|
| | mean `nao_sub` | `nao_sub^2` work | mean `nao_sub` | `nao_sub^2` work |
| GPU4PySCF's | 324.4 | 1.000 | 324.4 | 1.000 |
| exact `max_g \|ao_i\| > 1e-10` | 308.1 | 0.903 | 308.1 | 0.903 |
| density-aware, 1e-13 | 183.5 | **0.324** | 316.9 | **0.959** |
| density-aware, 1e-11 | 158.9 | 0.242 | 277.2 | 0.737 |
| density-aware, 1e-9 | 131.8 | 0.166 | 229.6 | 0.506 |

**On the `minao` guess the density-aware screen looks like a 3x on the XC
build's quadratic work; on a converged density it is worth 4 %.**  A `minao`
guess is block-sparse -- zero on every polarisation shell and between
fragments -- so `sum_j |dm_ij| a_j` is small for most `i`; a converged density
is not sparse and the bound stops biting.  This is the same trap as §1's and
§2's, for the third time in one document, and it is the reason this section
exists as a probe rather than a kernel.

What survives is the density-independent part: GPU4PySCF's radial test keeps
5 % more AOs than an exact `max_g |ao_i| > 1e-10` would, which is **10 % of
the quadratic work**.  Real, but an order of magnitude short of what the first
measurement suggested.

The obvious implementation does not capture even that.  Applying the screen
*after* `_grouped_block_loop` has evaluated the AOs needs a gather of
`nvar * nao_sub * ngrids` doubles per block, and that is `O(nao_sub * ngrids)`
of bandwidth against an `O(nao_sub^2 * ngrids)` saving on a build the previous
round established is latency- rather than flop-bound.  Measured: **0.28x**,
i.e. 3.6x slower, at every threshold.  Reverted; the probe stays.

The direction that would work is the one JoltQC takes -- move the screen
**upstream of `eval_ao`**, into the shell list `_sparse_index` builds, so the
AOs are never evaluated rather than evaluated and then discarded, using a
per-(block, shell) log-magnitude estimate (`log|c| + l*log r - a r^2` at the
block's closest grid point, a per-grid quantity computed once).  But size it
before building it: on a converged density the reachable saving is ~10 % of
the XC build's quadratic work, the XC build is 10 % of a patched `PfPMT` SCF,
and `fastxc` already has it 3x faster than GPU4PySCF.  **That is well under
1 % of an SCF.**  It is not the largest item left; `get_k`, at 67 % of a
patched SCF, is.

---

## 4. Compiling only what the basis reaches (opening 4)

Nothing here is compiled at install time: every kernel is built at run time by
NVRTC and cached by CuPy on the source hash, so the first run of a kernel set
pays for all of it.  The previous round measured that **compile options cannot
fix this** -- `--minimal` and `--use_fast_math` together are worth 4 % on
`fastk`, because the cost is the volume of generated code, not NVRTC's fixed
overhead -- and concluded that the fix is to hand NVRTC less code.

`electrolites/_ksplit.py` does that.  A generated file is a preamble followed
by independent kernels named for their angular-momentum class, so the driver
names the classes its basis reaches and compiles exactly those.  The subset is
derived from the *set of angular momenta in the basis*, never from how many
pairs survive screening, so two molecules in the same basis produce the same
subset, the same source text and the same CuPy disk-cache entry.

Two details that are easy to get wrong, and both were:

* `fastk` and `fastejk` put each class's real work in a
  `template <int NRANGE> __device__ ... kbody_<class>` that sits *between* the
  previous class's `extern "C"` wrappers and its own.  Cutting the file at
  `extern "C"` files each class's body under the previous class's name, so
  dropping a class takes the next class's body with it.  Chunks are cut at the
  **closing brace** of each definition instead, which requires brace matching
  rather than "the next line that is a bare `}`", because those wrappers are
  one-liners.
* `source()` checks that every `kbody_` a kept kernel instantiates is also
  kept, and falls back to the whole file if not.  Nothing silently mis-links.

`benchmarks/compile_time.py`, with CuPy's disk cache disabled so that every
timing is a real compile, one A100:

| module | basis | kernels | source | first-run NVRTC | vs the whole family |
|---|---|---|---|---|---|
| `fastk` | lmax 2 (6-31G\*) | 42/89 | 0.91 MB | **11.9 s** | **2.67x** |
| `fastk` | lmax 3 (def2-TZVP) | 89/89 | 2.76 MB | 31.2 s | 1.02x |
| `fastk` | whole family | 89/89 | 2.76 MB | 31.7 s | — |
| `fastejk` | lmax 2 | 25/65 | 1.90 MB | **16.5 s** | **3.54x** |
| `fastejk` | lmax 3 | 65/65 | 6.77 MB | 58.5 s | 1.00x |
| `fastejk` | whole family | 65/65 | 6.77 MB | 58.5 s | — |
| `fastj` | lmax 2 | 16/29 | 0.79 MB | **8.9 s** | **8.06x** |
| `fastj` | lmax 3 | 29/29 | 4.71 MB | 73.1 s | 0.98x |
| `fastj` | whole family | 29/29 | 4.71 MB | 71.6 s | — |

**For a 6-31G\* energy-and-gradient job the first run's NVRTC bill goes from
162 s to 37 s — 4.3x.**  A def2-TZVP job gains nothing, and should not: it
reaches every class there is and has to compile all of them.

`fastj`'s row is where the split earns the most, and only because of opening
1: seventeen written classes took the family from 0.8 MB to 4.7 MB, so
without the split every 6-31G\* user would have paid a minute of NVRTC for
kernels their basis can never launch.  The two openings are complementary --
the split is what makes the coverage affordable.
`ELECTROLITES_FULL_MODULE=1` compiles everything, which is the ablation.

---

## 5. The XC and VV10 grid passes, fused again (opening 5)

`fastxcnlc.nr_rks_nlc` builds the XC potential and the VV10 non-local
correlation from one sweep of the grid instead of two.  GPU4PySCF's
`get_veff` calls `nr_rks` and then `nr_nlc_vxc`, and each walks the grid twice
-- once for the density, once to contract the potential -- so the AOs and
their gradients are evaluated **four** times per SCF iteration on what is
bit-for-bit the same point set.  The module has been here since the 1.7.x
round; what it lost in the port is its patch point, because `fastrsh` (which
drove it) stands down on 1.8.x and `dft.rks.get_veff` still makes the two
calls separately.

It is back without carrying a copy of `get_veff`: the fused build runs first,
and then GPU4PySCF's own `get_veff` runs unchanged with `nr_rks` temporarily
answering with the fused result and `do_nlc` answering `False`.  The Coulomb
build, the exchange build, the incremental `ecoul` bookkeeping and the
returned `tag_array` stay upstream's code, which is what the port notes asked
for.  Anything not covered -- a different NLC grid, unrestricted, several
density matrices, density fitting, more than one device -- falls back to
GPU4PySCF's `get_veff` unchanged.

Correctness, H2O/def2-TZVPP, `conv_tol = 1e-10`, against unpatched 1.8.1:

| functional | unpatched | fused | difference |
|---|---|---|---|
| wB97M-V (mGGA + VV10) | -76.433422027141 | -76.433422027141 | -4.3e-14 Eh |
| B97M-V (mGGA + VV10) | -76.436269393547 | -76.436269393546 | 1.6e-13 Eh |
| wB97X-V (GGA + VV10) | -76.434452343862 | -76.434452343861 | 8.5e-14 Eh |

and -- after the grid-comparison fix below, on systems where the fused path
actually runs -- one `get_veff` on a fixed converged density, fused against
the same build with the fusion forced off, with `fastxcnlc.stats()` confirming
which path each took:

| system | basis | functional | nao | `max\|dVeff\|` |
|---|---|---|---|---|
| 6 ethanols | def2-TZVPP | wB97M-V | 1062 | 2.1e-14 |
| 6 ethanols | 6-31G\* | B97M-V | 324 | 7.1e-15 |
| `PfPMT`, 284 atoms | 6-31G\* | wB97M-V | 2268 | 5.7e-14 |

`fastxcnlc` is now a module in its own right (`electrolites.patch('fastxcnlc')`)
and covers **meta-GGA**, which `fastxc` hands back in the SCF -- so wB97M-V,
the functional this repository's largest previous win was measured on, is
exactly the case it serves.

### What it is worth, and the bug that timing it exposed

The first timing came back at **1.005x**, which is not a small win, it is no
win.  It was not: **the fusion was declining on every system bigger than a few
atoms, silently, while returning exactly the right answer.**

`same_grid` compared the two grids **elementwise**.  On 1.7.x that was right --
the two builds agreed row for row, which is what the module's original note
recorded.  1.8.x builds `RKS.grids` and `RKS.nlcgrids` independently and they
come out with the same points in a **different order**: on the 284-atom
cluster, 277 700 of 3 396 608 rows differ elementwise and **zero** differ once
both are sorted lexicographically on `(x, y, z, w)`.  So the test said "not
the same grid" and the fusion handed the case back to GPU4PySCF.

Two things worth taking from that:

* A permutation is immaterial here -- the VV10 energy and potential are sums
  over the grid -- so the fix is to compare the two as **sets**: sort both and
  compare, once per pair of grid objects, cached against the dozens of
  `get_veff` calls an SCF makes.  It still has to *be* a permutation, so the
  sort is done rather than assumed.
* **No correctness test could have caught this.**  A silent fall-back returns
  the right number by construction; the energies agreed to 1e-13 either way.
  Only the timing showed it, and only because the expected effect was large
  enough to notice its absence.  A module that can decline needs a check that
  it *did not* -- the diagnostic that found this counts fused-versus-declined
  calls, and that is what belongs in the test suite.

With the fusion actually firing, wB97M-V/6-31G\* on `PfPMT`, converged
density, min of 3, one A100-SXM4-40GB:

| | two grid passes | one | |
|---|---|---|---|
| `get_veff` | 29.238 s | **28.212 s** | **1.036x** |

**About 1.0 s per Fock build**, or ~17 s over a 17-cycle SCF.  That is the
honest size of it and it is smaller than the 1.7.x framing implied, for a
reason the arithmetic gives: the fusion removes two of the four passes over
the grid, i.e. the AO evaluation and one potential contraction, and leaves the
`O(n_grid^2)` VV10 double sum -- which is ~17 s of this 29 s -- exactly where
it was.  At 6-31G\* and 2268 AO the AO passes are simply not expensive.  The
previous round sized the same passes at 3.6 s of 19.9 s, but that was
def2-TZVPD at 6609 AO, where they are.

---

---

## 6. A measurement hazard found while re-running the fixed-density table

`benchmarks/fixed_density.py`'s `grad_total` row times the analytic gradient
with both halves of the A/B in the same job.  Its **unpatched** half runs
GPU4PySCF's nuclear-attraction one-electron derivative on the **CPU**, through
libcint -- which is precisely what `fastgradh` moves to the GPU.  So that row,
alone among the rows in that table, depends on how many OpenMP threads the job
got, and SLURM sets `OMP_NUM_THREADS=1` unless told otherwise.

Re-running it here with the variable unset, the *stock* PfPMT gradient had not
finished after eight minutes against the 16.7 s the previous round recorded
for it, and the 545-atom one likewise.  Whatever the previous round's runs
had, the two are not the same setting, and the direction of the bias is
knowable without knowing which: fewer CPU threads makes the **baseline**
slower and the patched/stock gradient ratio **larger**.

Two consequences, and the second is the load-bearing one:

* the job scripts here now set `OMP_NUM_THREADS` explicitly, and
  `fixed_density.py` gained `--no-grad` so that the GPU build rows -- which
  have no CPU component and are unaffected -- can be measured without waiting
  on the gradient;
* **`docs/UPGRADE_1.8.1.md`'s gradient rows (1.63x on `PfPMT`, 1.60x on
  `HcgC`) should be re-taken with the thread count pinned before they are
  quoted again.**  The build rows in that table are pure GPU and are not
  affected.

This is the third time in this round that a number turned out to be about the
measurement rather than the code -- after §2's fixed-density-versus-converged
comparison and §1's `minao` guess being zero on the shells whose kernels were
being timed.  All three have the same shape: the harness was measuring
something adjacent to the thing named.

---

## 7. The build-by-build table, on both densities

`benchmarks/fixed_density.py --no-grad`, both halves of each A/B in one job,
`OMP_NUM_THREADS=32`, min of 3, one A100-SXM4-40GB.  The `minao` columns are
what `docs/UPGRADE_1.8.1.md` reported; the `scf-8` columns time the same
builds on the density after eight SCF cycles.

**`PfPMT`, 284 atoms, 2268 AO, B3LYP-D3(BJ)/6-31G\***

| | `minao` stock | `minao` ported | | `scf-8` stock | `scf-8` ported | |
|---|---|---|---|---|---|---|
| `get_k` | 2.035 s | 0.373 s | **5.46x** | 6.534 s | 2.690 s | **2.43x** |
| `nr_rks` | 0.914 s | 0.307 s | 2.98x | 0.713 s | 0.305 s | 2.34x |
| `get_j` | 0.447 s | 0.314 s | **1.42x** | 1.787 s | 0.991 s | **1.80x** |
| **`get_veff`** | **3.392 s** | **0.996 s** | **3.41x** | **9.042 s** | **3.997 s** | **2.26x** |

**`HcgC`, 545 atoms, 4170 AO**

| | `minao` stock | `minao` ported | | `scf-8` stock | `scf-8` ported | |
|---|---|---|---|---|---|---|
| `get_k` | 4.674 s | 1.197 s | **3.90x** | 20.327 s | 9.186 s | **2.21x** |
| `nr_rks` | 3.328 s | 0.797 s | 4.18x | 2.408 s | 0.808 s | 2.98x |
| `get_j` | 1.576 s | 1.189 s | **1.33x** | 7.062 s | 4.140 s | **1.71x** |
| **`get_veff`** | **9.615 s** | **3.224 s** | **2.98x** | **29.796 s** | **14.128 s** | **2.11x** |

Three readings.

**The `minao` stock column reproduces the previous round to within 1 %**
(3.392 vs 3.375 s, 9.615 vs 9.618 s, and `get_k` 2.035 vs 2.022 s), so this is
the same measurement, not a different one.

**`get_veff` on a converged density is 2.26x, not 3.41x -- and 2.26x is the
number that belongs next to a wall clock.**  §2's SCF profile puts `get_veff`
inside the SCF at 1.86x and the wall clock at 1.84x; the remaining gap between
2.26x and 1.86x is the early cycles, whose densities are between the guess and
the converged one.  So the sequence 3.41 -> 2.26 -> 1.86 -> 1.84 is a
measurement converging on its subject, and there is no Amdahl tail anywhere in
it.  **Every fixed-density speedup in `docs/UPGRADE_1.8.1.md` is a `minao`
number and overstates what an SCF sees**, by about 1.5x on `get_veff`.

**`get_j` is the one row that moves the other way.**  It is *better* on a
converged density -- 1.42 -> 1.80x on `PfPMT`, 1.33 -> 1.71x on `HcgC` --
because the high-`lij` classes this round added carry no work at all on a
`minao` guess (§1's method note) and real work on a converged one.  It was the
weakest row in the previous round's table at 1.38x/1.33x; it is no longer the
weakest row in either column.

---

## 8. End to end

`benchmarks/cluster_bench.py`, `PfPMT`, 284 atoms, B3LYP-D3(BJ)/6-31G\*,
`conv_tol = 1e-9`, `OMP_NUM_THREADS=32`, one A100-SXM4-40GB, one process at a
time.  Both runs converge in the same number of cycles to the same energy:

| | GPU4PySCF 1.8.1 | + Electrolites | |
|---|---|---|---|
| SCF | 127.17 s | **55.10 s** | **2.31x** |
| gradient | 19.40 s | **12.49 s** | **1.55x** |

E = -7652.5600864500 both ways; the gradient norms agree to the printed ten
digits.  The previous round's figures for the same measurement were 2.13x and
1.58x, so **the SCF moved from 2.13x to 2.31x** and the gradient did not move
-- which is what the rest of this document predicts, since everything added
this round is in the Coulomb build and in the compile, and nothing is in the
gradient.

Reading it against §2 closes the loop opened there.  The SCF is 2.31x; `get_j`
is 1.80x, `get_k` 2.43x and `nr_rks` 2.34x on a converged density (§7); and
DIIS, the eigendecomposition and grid construction are 1.9 % of the wall
clock.  There is no residual left to explain -- the wall clock is what the
three builds say it should be.


## What is left

1. **The AO screen, moved upstream of `eval_ao`** (§3c).  The headroom is
   measured -- 65 % of the XC build's quadratic work on a `minao` density --
   and the post-hoc version is measured not to capture it.  What it needs is a
   per-(block, shell) log-magnitude estimate combined with `log|dm|`, feeding
   the shell list `_sparse_index` builds.  Size it first: the XC build is 10 %
   of a patched `PfPMT` SCF.
2. **`get_k`.**  It is 67 % of a patched `PfPMT` SCF -- 44.5 s of 66.6 s --
   and this round did not touch it.  §3b's ceiling says how much of that the
   `vk` atomics are.
3. **The classes beyond `(6,6)`.**  `gen_j_high.py` covers every class an
   **spdf** basis reaches.  A basis with g functions (cc-pVQZ) reaches
   `lij = 8`, and `(7,x)`/`(8,x)` still fall through.  Thirteen of those
   seventeen would fit this design as it stands -- `(7,0)` needs 39 KB and
   `(8,5)` 158 KB of the A100's 164 KB -- and only four (`(7,7)`, `(8,6)`,
   `(8,7)`, `(8,8)`, 192-274 KB) genuinely need a different `Rt` strategy,
   because the whole order-L Hermite triangle sits in shared memory at one
   column per quartet and `nf3(L)` grows as `L^3`.  So this is a table entry
   and a sweep for thirteen of them, not a redesign.  `default_cfg` now
   *raises* rather than returning a configuration the card cannot launch,
   which is what it used to do.
4. **`fastj` still declines `n_dm > 1` and `omega != 0`**, so the J build in
   CPHF, TDDFT and the range-separated path is untouched -- the previous
   round's B4a/B4b, unchanged.
5. **Multi-GPU.**  Every module still asserts one device.
6. **The 12 lifted J classes screen at block rather than batch granularity.**
   The written classes were fixed (§1); the lifted ones still substitute the
   block's first pair for the batch's.  It costs accuracy, not speed, and it
   did not show above 1e-13 on anything measured here -- but it is the same
   defect, and `gen_j_kernels.py` should get the same one-line change.

## Files

| file | what it is |
|---|---|
| `codegen/gen_j_high.py` | writes the `md_j` classes GPU4PySCF does not unroll, from `(lij,lkl)` alone |
| `electrolites/kernels/fastjhigh_generated.cu`, `fastjhigh_launch.json` | the generated kernels and their launch table |
| `electrolites/_ksplit.py` | splits a generated file into per-class kernels so only what the basis reaches is compiled |
| `electrolites/fastxcnlc.py` | the fused XC + VV10 grid pass, and its 1.8.x patch point |
| `benchmarks/scf_anatomy.py` | where an SCF wall clock goes, method by method |
| `benchmarks/perclass_j.py` | per-class J timing and the AO-space correctness check |
| `benchmarks/sweep_j_high.py` | the launch-configuration search |
| `benchmarks/compile_time.py` | first-run NVRTC cost, whole family against the basis subset |
| `benchmarks/aoscreen_probe.py` | how much a density-aware AO screen could remove |
