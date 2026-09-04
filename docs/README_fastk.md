# fastk — faster exchange-matrix (K) kernels for GPU4PySCF

Direct HF on these clusters is dominated by the exchange build. On an A100-40GB,
HF/6-31G\* on `PfPMT_cluster` spends **75 %** of its GPU time inside
GPU4PySCF's `rys_k_*` kernels and 23 % in the McMurchie–Davidson `md_j_*` J
engine; everything else is under 2 %. `fastk` rewrites the K kernels and leaves
the rest of GPU4PySCF alone.

```python
import sys; sys.path.insert(0, '.../side_test/HF/src')
import fastk                      # patches gpu4pyscf.scf.jk._VHFOpt.get_k
from gpu4pyscf import scf
mf = scf.RHF(mol); mf.kernel()
```

Four generators feed it, in the order they were written.  `gen_kernels.py` and
`gen_k2_kernels.py` **lift** integral arithmetic out of GPU4PySCF's generated
`unrolled_rys_k.cu`; `gen_kgeneral.py` copies its general kernel with the
divisions hoisted; and `gen_khigh.py` **writes** kernels from the
angular-momentum class alone, for the 40 classes an spdf basis reaches that
GPU4PySCF has no unrolled kernel to lift — plus five it does unroll, where the
written kernel measured faster than the lifted one.  The three addenda at the
end of this file cover the last three generators in that order; the results
table immediately below is from the first round only.

## Results (A100-SXM4-40GB, RHF/6-31G\*, `conv_tol=1e-9`, `direct_scf_tol=1e-13`)

| system | atoms | AOs | GPU4PySCF 1.7.0 | fastk | speedup | ΔE |
|---|---|---|---|---|---|---|
| `PfPMT_cluster`, charge −1 | 284 | 2268 | 77.75 s | **49.66 s** | **1.57×** | 1e-10 Eh |
| `HcgC_cluster`, charge −4  | 545 | 4170 | 258.67 s | **175.83 s** | **1.47×** | 3e-10 Eh |

K build alone, at a representative SCF density: **2.03×** (PfPMT) and **1.82×**
(HcgC), with every implemented angular-momentum class faster on both systems. Both runs converge in the same number of iterations to the
same energy; the residual difference is at the level of the SCF convergence
threshold, not of the integrals (per-matrix agreement is ~1e-14 relative).

## What the kernels change

The integral arithmetic is *not* changed: `gen_kernels.py` lifts the Rys 2D
recurrences and the density contraction verbatim out of GPU4PySCF's generated
`gvhf-rys/unrolled_rys_k.cu`, and the Rys root/weight tables are the same
numbers. What changes is the scaffolding around that arithmetic.

1. **Rys roots and weights in registers, not shared memory.** GPU4PySCF stages
   them through a shared `rw` buffer, which forces a `__syncthreads()` *once per
   primitive quartet* in the innermost loop. That barrier costs cycles and, worse,
   stops warps from running ahead of one another. Measured on `rys_k_1000`:
   0.37 instructions issued per scheduler-cycle out of a possible 1.0.
2. **No double-precision division in the inner loop.** GPU4PySCF evaluates
   `rt/(aij+akl)` and `.5/aij` once per Rys root and
   `cicj*ckcl/(aij*akl*sqrt(aij+akl))` once per primitive quartet — up to five
   DP divisions per primitive quartet, each ~20 instructions on sm_80. We keep
   `1/aij` (block-uniform, in shared memory) and `1/akl` (per thread, hoisted out
   of the primitive loop), take one `rsqrt(aij+akl)`, and turn every division
   into a multiply. **This alone was worth 1.77× → 2.03× on the whole K build.**
3. **Block-uniform bra data computed once.** For a given block the bra shell pair
   is fixed, so `aij`, `1/aij`, `aj/aij`, the Gaussian product centre and `cicj`
   depend only on the primitive-pair index. GPU4PySCF recomputes them per thread
   per primitive quartet; we compute them once per bra pair into shared memory.
4. **(ss|ss) skips the Rys root.** Only the weight enters an `(ss|ss)` integral,
   but because GPU4PySCF writes the root through shared memory the compiler
   cannot eliminate it, so it pays an `exp` and a division per primitive quartet
   for a value that is never read.
5. **Occupancy tuned per angular-momentum class.** GPU4PySCF compiles every K
   kernel with `__maxnreg__(128)`, which pins the whole family at 25 % occupancy.
   How many registers a class actually needs is set by its number of `gout`
   accumulators, so `gen_kernels.py` gives each class its own
   `__launch_bounds__`. `(ss|ss)`…`(ds|ss)` run at 64 registers and ~50 %
   occupancy; the wide classes such as `(pp|pp)` (81 accumulators) get 256
   registers and a 64-thread block instead. Choosing this per class rather than
   globally is worth roughly 1.5× on the wide classes and 1.3× on the narrow ones
   relative to a single compromise setting.

## Range separation

`_usable` used to decline every `omega != 0` call, which for a range-separated
hybrid such as ωB97X meant half of the exchange work — GPU4PySCF builds a
full-range *and* a long-range K per SCF iteration — ran on GPU4PySCF's kernels.
It now builds the long-range (erf) operator too, and the whole of the change is
one `fma`: `rsqrt(aij+akl)` becomes `rsqrt(aij+akl + aij*akl/omega^2)`.  The
derivation, the fused `NRANGE == 2` kernels and the `fastrsh` driver that uses
them are in `README_fastrsh.md`.  The short-range (`omega < 0`, `erfc`)
operator still falls through: it needs twice the Rys roots and GPU4PySCF's
`s_estimator` screening.

## Is any of this size-dependent?

Deliberately not. The audit, and how each item is handled:

| ingredient | depends on | handling |
|---|---|---|
| register/occupancy table | GPU + angular-momentum class | fixed per class; a molecule cannot change it. Re-tune with `bench/sweep.sh` for a different GPU. |
| persistent grid width | number of bra shell pairs | `n_blocks = min(4·#SM, npairs_ij)` — a block can only work on one bra pair at a time, so a wider grid would just idle on small molecules. |
| task queue depth | number of ket pairs | `pair_kl` is chunked to `QUEUE_DEPTH`, so the queue never overflows however large the system is. |
| shared bra cache | basis contraction depth, **not** system size | sized for `iprim*jprim ≤ 36` (Pople sets); deeper contractions (cc-pVDZ, ANO) are detected in `_usable` and fall through to GPU4PySCF. |
| implemented classes | basis angular momentum, **not** system size | `f`/`g` shells and the classes GPU4PySCF splits across threads fall through. |
| scratch memory | constant | 56 MB pool + GPU4PySCF's own. |

The two clusters differ by 1.9× in AO count and both get essentially the same
per-class speedups, which is the empirical form of the same statement. The
speedup does drift down slowly with size (2.03× → 1.82× on the K build) because
the classes we do *not* implement grow slightly in relative weight, not because
any mechanism breaks.

`bench/test_correctness.py` exercises the fall-through paths: cc-pVDZ and
cc-pVTZ (contractions deeper than the bra cache, and `f` functions), def2-SVP,
third-row elements, and molecules with fewer bra pairs than the grid is wide.

## Coverage and what is left

*(As of the first round; the three addenda below extend it.  Coverage is now
every class an spdf basis reaches, with nothing falling through — measured at
0.0 % on both systems and both bases.)*

14 of the 25 `(li lj|lk ll)` classes that a 6-31G\* organic system generates are
implemented — the ones GPU4PySCF unrolls with one thread per shell quartet.
They are 78 % of GPU4PySCF's K time. The remaining classes (`(dp|pp)`,
`(dp|dp)`, `(dd|pp)`, …) split their `gout` accumulators across `threadIdx.y`
and stage the 2D integrals through shared memory; the same rewrite is possible
there but needs a warp-shuffle replacement for the shared `rw` buffer.

After this work the J engine is the largest remaining block (≈40 % of the SCF).
It is a genuinely different algorithm (McMurchie–Davidson on primitive shells
with Hermite densities) and profiling shows it at 25 % occupancy like the old K
kernels. GPU4PySCF's fused Rys `RYS_build_jk` is *not* the answer — measured, it
is slower than `md_j` + `rys_k` run separately (8.31 s vs 7.70 s on PfPMT).

> **Correction (later work).** This section originally said the J engine's
> divisions "are already hoisted out of the hot loop". That is wrong: every
> `md_j_*` kernel evaluates `fac/(aij*akl*sqrt(aij+akl))` and
> `aij*akl/(aij+akl)` inside the innermost ket loop. Removing them is most of
> what `fastj` does, and it is worth up to 1.99× per class. See
> `README_fastj.md`.

## Measured and rejected

Recorded so they are not re-tried. All measured on the A100, not reasoned about.

| idea | measurement | verdict |
|---|---|---|
| Load imbalance across the persistent blocks | `sm__cycles_active` min/avg/max within 1.3 % of each other | not a problem; the first `--set full` profile that suggested otherwise was distorted by Nsight Compute's kernel replay |
| Screening scan is too wide (11.9e9 `pair_kl` visits per K build for PfPMT, 23 % accepted) | `q_cond`/`dm_cond` are ~5 MB each and stay in L1 (99.8 % hit rate); the scan is a few ms | not worth sorting `pair_kl` by `q_cond` |
| Primitive-pair screening inside the shell pair | only ~16 % of primitive pairs are negligible at 1e-13 for 6-31G\* | too little to justify dropping terms |
| Replace the `nroots=1` Boys function (`erf`+`exp`) with a Chebyshev table | probe with a free stand-in: `(ss\|ss)` 457→404 ms, `(ps\|ss)` 1081→937 ms | ~13 % of two classes = 1.8 % of the K build; not worth the accuracy risk |
| Fuse J and K into one Rys pass | `RYS_build_jk` 8.31 s vs `md_j`+`rys_k` 7.70 s (PfPMT), 32.9 s vs 27.3 s (HcgC) | GPU4PySCF's split is already the better choice |
| Keep the task queue in scan order, so a consumer warp's 32 tasks are neighbouring shell pairs (warp-aggregated `atomicAdd` instead of one per survivor) | K build 2.928 s → 2.944 s; `k_1000` 15.6 → 15.9 ms | no gain. Nsight says L1 hit rate 91 %, L2 hit rate 90 %, DRAM throughput 0.9 % — the kernel is latency-bound with everything already in cache, so locality is not what is missing |
| More occupancy on `k_1000` (the largest single class): sweep `__launch_bounds__` at 256×3, 256×4, 128×4, 128×6, 64×8 | 15.9, 15.6, 16.7, 18.0, 27.3 ms | flat at best. 80 registers gives 6 warps per scheduler out of 16; forcing 64 registers buys 2 % and nothing else moves |
| Build the short-range operator instead of the full-range one, `hyb*K(erfc) + alpha*K(erf)` in place of `hyb*K(1/r) + (alpha-hyb)*K(erf)` — the same matrix, with `erfc` screening | GPU4PySCF: SR 5.02 s vs full-range 6.01 s (PfPMT). The `erfc` task list is ~2.4× shorter but needs twice the Rys roots, so the work is a wash — and `fastk` accelerates the full-range build 2.05× and the SR build not at all | rejected |
| Wider persistent grid / deeper task queue | `BLOCKS_PER_SM` 2/4/8 → 12.51/11.35/11.34 s; `QUEUE_DEPTH` 32k/64k/128k → 11.35/11.29/11.28 s | 4 blocks per SM and a 64k queue; both are plateaus, not peaks |

## Files

| file | role |
|---|---|
| `fastk.py` | patches `_VHFOpt.get_k`; dispatch, fall-through rules, workspace |
| `fastk_prologue.cu` | Rys roots in registers, task screening, the hand-written `(ss\|ss)` kernel |
| `gen_kernels.py` | lifts GPU4PySCF's unrolled integral bodies into the new scaffolding |
| `fastk_generated.cu` | generated kernels (regenerate with `build_kernels.sh`) |
| `fastk_launch.json` | per-class launch bounds, written by the generator |
| `data/rys_roots_dat.npz` | GPU4PySCF's Rys tables, extracted verbatim |
| `gen_k2_kernels.py` | the classes GPU4PySCF splits across `threadIdx.y` (addendum below) |
| `gen_kgeneral.py` | a faster copy of GPU4PySCF's general kernel, for whatever is left |
| `gen_khigh.py` | writes the classes GPU4PySCF has no unrolled kernel for, from scratch |
| `fastkhigh_generated.cu` | those kernels (regenerate with `build_khigh.sh`) |
| `fastkhigh_launch.json` | their thread scheme and shared-memory needs |
| `../bench/` | `run.py` (end-to-end), `perclass.py` / `perclass_kall.py` (per class), `one_class.py` (one class, every path), `check_k.py`, `sweep.sh`, `sweep_kh.sh`, `coverage.py`, `test_correctness.py` |

Regenerate the kernels against a different GPU4PySCF checkout with

```bash
./build_kernels.sh /path/to/gpu4pyscf/lib/gvhf-rys/unrolled_rys_k.cu
```

---

# Addendum: the four classes GPU4PySCF splits across `threadIdx.y`

The section above covers the 14 classes GPU4PySCF unrolls with one thread per
shell quartet.  A second generator, `gen_k2_kernels.py`, now covers the four
that a 6-31G\* organic system also generates and that GPU4PySCF spreads over
`threadIdx.y`: `(dp|pp)`, `(ds|dp)`, `(dp|ds)` and `(dd|ps)`.  Those keep their
2D integrals in shared memory, so — unlike the first 14 — the Rys roots have to
stay there too.  What changes is the arithmetic around them.

| class | GPU4PySCF | fastk | speedup | | GPU4PySCF | fastk | speedup |
|---|---|---|---|---|---|---|---|
| | *PfPMT (2268 AO)* | | | | *HcgC (4170 AO)* | | |
| `(dp\|pp)` 2111 | 501.3 ms | 309.6 ms | 1.62× | | 1950.3 ms | 1270.8 ms | 1.54× |
| `(ds\|dp)` 2021 | 120.7 ms |  73.0 ms | 1.65× | |  504.0 ms |  330.0 ms | 1.53× |
| `(dd\|ps)` 2210 | 121.7 ms |  76.6 ms | 1.59× | |  471.8 ms |  321.6 ms | 1.47× |
| `(dp\|ds)` 2120 | 167.9 ms | 115.4 ms | 1.46× | |  662.2 ms |  492.0 ms | 1.35× |
| **total** | **911.6 ms** | **574.6 ms** | **1.59×** | | **3588.3 ms** | **2414.4 ms** | **1.49×** |

With these four, the whole K build on `PfPMT` at a representative SCF density
goes from 6013 ms to 2918 ms (**2.06×**); handing them back to GPU4PySCF
(`FASTK_NO_K2=1`) gives 3186 ms (1.88×), so the four classes are worth 1.09× on
the whole K build.  Over the 18 iterations of B3LYP/6-31G\* the K build goes
from 37.1 s to 33.9 s.  On `HcgC` the whole K build is 23205 ms → 12096 ms
(**1.92×**).

Three changes, all of them the same ones that worked for the first 14 classes:

1. **No double-precision division or sqrt in the innermost loop.**  These are
   three-root classes, so GPU4PySCF pays `aj/aij`,
   `cicj/(aij*akl*sqrt(aij+akl))` and `aij*akl/(aij+akl)` once per primitive
   quartet plus `rt/(aij+akl)`, `.5/aij` and `.5/akl` once per root — eleven DP
   divisions and a DP sqrt per primitive quartet.  `1/aij` is cached with the
   bra pair, `1/akl` in one more slot of the ket cache, and a single
   `rsqrt(aij+akl)` supplies the rest.
2. **Block-uniform bra data computed once.**  The bra shell pair is fixed for
   the whole block, so `aij`, `1/aij`, `aj/aij` and the Gaussian product centre
   depend only on the primitive-pair index; GPU4PySCF recomputes them per
   thread per primitive quartet and stages the product centre through a
   two-element shared buffer.  They now go into the same shared cache that
   already holds `cicj`, which also frees that buffer.
3. **A `__launch_bounds__` at all.**  GPU4PySCF compiles these four with no
   register limit; 2 blocks per SM (128 registers) is fastest for all four,
   measured, and the sweep at 1/2/3/4 blocks per SM is reproducible with
   `bench/perclass_k2.py --variants`.

The Rys roots are unchanged in *where* they live (shared, distributed over
`threadIdx.y`, exactly as GPU4PySCF does it), but the evaluation is fastk's
own: the large-argument branch takes one `rsqrt` in place of a `sqrt` and a
division per root.  `omega == 0` is compiled in, which removes the
range-separation branches.

### One thing that had to be copied, not simplified

These kernels run their last lanes one thread-block *past* `ntasks` (with a
zero symmetry factor) so that every `sq_id` lane executes the same number of
`__syncthreads()`.  GPU4PySCF's `_fill_vk_tasks` therefore ends by padding the
task queue with a valid shell pair at `bas_kl_idx[ntasks .. ntasks+nsq-1]`; a
first version of `fill_vk_tasks2` left that out and the lanes read
uninitialised queue memory, which turns into an out-of-range `ksh` and an
illegal address.

The padding then needs somewhere to go.  `pair_kl` is chunked to
`QUEUE_DEPTH`, so `ntasks` can reach `QUEUE_DEPTH` and the padding runs off the
end of the block's slot into the next block's task list.  That is why the queue
stride is now `QUEUE_DEPTH + 256` rather than `QUEUE_DEPTH`.  It is worth
recording how this showed up: `PfPMT` (2268 AOs) was correct to the last digit
and only `HcgC` (4170 AOs) was wrong, by 5e-4 Eh — a bra pair has to accept
almost a whole 65536-pair chunk before the overflow happens, which needs a
large enough molecule.  A single-system check would have passed it.

## The seven classes GPU4PySCF does not unroll

`(ds|dd)`, `(dp|dp)`, `(dp|dd)`, `(dd|pp)`, `(dd|ds)`, `(dd|dp)` and `(dd|dd)`
have no `rys_k_<class>` in `unrolled_rys_k.cu`; they run on GPU4PySCF's general
`rys_k_kernel`, which is 17 % of the exchange time of an ωB97X/6-31G\* run on
`PfPMT`.  `gen_kgeneral.py` lifts that kernel out of
`gvhf-rys/rys_contract_k.cu` and applies the same three changes as above —
reciprocals hoisted (it evaluates `2 + 3*nroots` divisions per primitive
quartet per thread), the block-uniform bra data cached once per shell pair, and
range separation through the same `rsqrt` substitution.  `cuda/build_kgeneral.sh`
compiles it with `nvcc` into `cuda/libmykg.so`, which `fastk` loads through
`ctypes`; `FASTK_NO_GENERAL=1` hands the classes back.

It also had to change how the task queue is indexed.  GPU4PySCF gives each
block the queue slot of the SM it happens to be running on
(`pool + get_smid()*QUEUE_DEPTH`), which is only safe while exactly one block
is resident per SM — an invariant it gets for free from a 255-register
footprint, and which removing arithmetic could silently break.  The copy uses
`fastk`'s persistent grid instead: a fixed number of blocks, each with its own
queue slot, claiming bra pairs from an atomic counter.

**The payoff is 1.07–1.19× per class, 1.10× over the seven** — much less than
the 1.4–1.6× the divisions suggested, and worth about 2 % of the exchange
build.  The reason is visible in `cuobjdump -res-usage`: the kernel still uses
255 registers, because 81 `gout` accumulators are 162 registers on their own,
so it is stuck at one 256-thread block per SM — 2 warps per scheduler.  It is
latency-bound at 12.5 % occupancy, and arithmetic was never what limited it.
Getting more would mean changing the 3×3×3×3 `gout` tile, which is the shape
the `inner_dot` templates and the `c_gxyz_offset` table are built around — or
not using that kernel at all, which is what `gen_khigh.py` below does.  With f
functions in the basis these seven classes grow to forty and carry two thirds
of the exchange, so that is where the next round went; `gen_kgeneral.py`'s copy
is still what catches anything `gen_khigh.py` turns away (a contraction deeper
than the shared bra cache, or `n_dm > 1`).

---

# Addendum: writing the kernels GPU4PySCF does not have (`gen_khigh.py`)

The two generators above *lift* arithmetic out of GPU4PySCF's generated
`gvhf-rys/unrolled_rys_k.cu`.  That file covers 25 angular-momentum classes —
every class an s/p/d basis can produce — so with f functions (def2-TZVP,
cc-pVTZ) there is nothing left to lift: **40 of the 65 classes an f basis
reaches have no unrolled kernel at all**.  They ran on GPU4PySCF's general
`rys_k_kernel`, which was 66 % of the def2-TZVP exchange time and where the
copy in `gen_kgeneral.py` only returned 1.066×.

`gen_khigh.py` writes those 40 kernels from scratch — the Rys 2D recurrences
(VRR, TRR and the two horizontal transfers) and the four exchange contractions,
emitted for one `(li,lj,lk,ll)` with every g-array address a compile-time
constant.  Nothing is copied from GPU4PySCF; the only shared input is
`_c_cartesian_lexical_xyz`, the order PySCF stores cartesian functions in.  The
kernels take the `KARGS2` signature and go in `fastkhigh_launch.json`, so
`fastk` dispatches them through the same path as the `gen_k2_kernels.py`
classes and neither `get_k` nor `get_k_rsh` needed a change.
`FASTK_NO_HIGH=1` hands them back.

It then turned out to beat the *lifted* kernels too.  Against GPU4PySCF's own
unrolled kernels on the six classes `gen_k2_kernels.py` covers (`PfPMT`,
def2-TZVP, long-range):

| class | GPU4PySCF | `gen_k2_kernels.py` (lifted) | `gen_khigh.py` (written) |
|---|---|---|---|
| `2021` | 807 ms | **628 ms** | 932 ms |
| `2111` | 2234 ms | 1854 ms | **785 ms** |
| `2120` | 1238 ms | 1034 ms | **421 ms** |
| `2210` | 726 ms | 604 ms | **331 ms** |
| `3011` | 857 ms | 694 ms | **339 ms** |
| `3110` | 1419 ms | 1169 ms | **553 ms** |
| total | 7281 ms | 5983 ms | **3360 ms** |

Lifting was worth 1.22× over GPU4PySCF on these six; writing them is worth
2.17×.  So five of the six moved over and `2021` stayed, which is why
`build_khigh.sh` generates 45 classes and `gen_k2_kernels.py` now only matters
for `2021` (its other five kernels are still compiled, so `FASTK_NO_HIGH=1`
remains a working ablation).  The one-thread-per-quartet classes of
`gen_kernels.py` were left alone: that design keeps the 2D integrals as well as
the roots in registers, which nothing here can match at low angular momentum.

## Why the general kernel was slow

`ncu --set detailed` on `(ff|ff)`: **86 % L1 throughput, 5.4 % compute, IPC
0.26, 12.5 % occupancy, and 86 % of its shared-memory wavefronts are conflict
replays.**  Three causes, all of which unrolling removes:

1. **The `gout` tile is 3×3×3×3 whatever the class.**  A shell with `nf = 10`
   (f) is padded to 12 and one with `nf = 1` (s) to 3, so `(fs|ff)` does
   `12·3·12·12 / (10·1·10·10) = 5.2×` the contraction work the class needs.
   (`gout_pattern` is meant to drop the padded dimensions, but in GPU4PySCF
   1.7.0's `gvhf-rys/rys_contract_k.cu` it is computed with `>>` where `<<`
   was intended, so only `ll == 0` is ever specialised.)
2. **81 `gout` accumulators are 162 registers**, which pins the kernel at 255
   registers and one 256-thread block per SM; the shared-memory budget then
   forces `nsq_per_block = 1` for the wide classes, so a warp's 32 lanes each
   read a *different* g-array address — that is where the wavefront replays
   come from.  Tiling exactly needs far fewer accumulators, which buys back
   `nsq_per_block`, and with 32 or more shell quartets per block a warp's reads
   are 32 consecutive doubles.
3. **Addresses come from shared `idx_i/j/k/l` arrays**, so each of the three g
   loads per `gout` element carries integer address arithmetic and no two loads
   can be CSE'd.  Unrolled they are constants, and the ten cartesian functions
   of an f shell then read only `li+1 = 4` distinct x addresses.

## One body, not one per lane

The obvious emission — a `switch (gout_id)` with a straight-line body per lane,
which is what GPU4PySCF's own unrolled kernels do — only works while a warp is
one lane wide, that is while the block holds 32 shell quartets.  The wide
classes cannot afford that many, and the switch then diverges 2- to 32-ways
inside every warp.  Measured over 29 classes: it wins **up to 5.6×** on the
classes that keep 32 quartets and **loses up to 3.7×** on the ones that do not,
for no net gain (23.16 s against the general kernel's 23.01 s) and 444 000
lines of CUDA.

So every lane runs the same code instead.  A lane owns all `nfi` cartesian
functions of the bra-i shell for `PER` of the `nfj·nfk·nfl` (j,k,l)
combinations; the (j,k,l) part of each address — the only part that differs
between lanes — is folded into three base pointers computed once, outside every
loop, and what is left inside is `nfi` compile-time offsets.  The lane's block
of combinations is kept aligned to the (j,k,l) odometer (`per_choices`), so
each of j, k and l is a runtime base plus a compile-time delta; that is what
lets the four contractions group their `atomicAdd`s identically for every lane.
31 000 lines for all 40 classes, no divergence, and the same 29 classes now run
in 14.00 s.  (Code size is not free: NVRTC and ptxas take tens of seconds on
the generated file the first time a process builds the module.  CuPy caches the
result in `~/.cupy/kernel_cache`, keyed on the source, so it is paid once per
kernel revision rather than once per run — but it is the reason the per-lane
emission's 444 000 lines would not have been shippable even if it had been
fast.)

## Tuning

`PER` is the only real knob.  A wider `gout` tile costs registers but needs
fewer lanes, and fewer lanes means both more shell quartets per block and less
of the block idling through the recurrences — those are three chains, one per
cartesian direction, so no more than three lanes can ever work on them.  The
generator takes the widest tile that fits `--gout-max`, then the most quartets
per block, snapped so a warp stays inside one lane.  Swept on the sixteen
classes that carry most of the def2-TZVP exchange (`bench/sweep_kh.sh`), against
16.46 s for the general kernel:

| `--gout-max` | time | vs. general |
|---|---|---|
| 42 | 10.53 s | 1.56× |
| 64 | 8.16 s | 2.02× |
| **96** (default) | **8.07 s** | **2.04×** |
| 128 | 8.38 s | 1.96× |

96 is a real optimum, not the end of the range: past it the `gout` tile costs
more in registers than it saves in lanes.  The two independent runs at 96 —
8.071 s and 8.068 s — are the reproducibility of the harness.

The generator also has `--par-hrr`, which spreads the two horizontal transfers
over every lane instead of leaving them on three.  It is **0.97×** at the
shipped setting — one `__syncthreads` per phase per root costs more than the
idling it removes, because widening the tile had already cut the lane count to
2–12 on the classes that carry the time.  It is kept as an option and
`bench/stage21.sh` checks it for correctness, since a class list dominated by
wide quartets would change that balance.

## What it is worth

`PfPMT`/def2-TZVP (5241 AO), the long-range exchange build through the real
`get_k` dispatch, one timed pass at a fixed density:

| | before | after |
|---|---|---|
| GPU4PySCF | 110.52 s | 110.97 s |
| `fastk` | 86.11 s | **49.35 s** |
| speedup | 1.284× | **2.249×** |

which is *past* the 6-31G\* figure the previous round reached (2.13×) — f
functions no longer cost anything in relative terms.  Agreement with GPU4PySCF
is 5.6e-14 relative, the same round-off as every other class, and
`bench/test_correctness_dft.py` passes 33/33.

Per class, against `gen_kgeneral.py`'s copy of the general kernel, the eleven
widest classes (`(ff|ff)` and its neighbours, 1800–10000 `gout`) come in at
**2.22×** together and `(ff|ff)` itself at **5.34×** — the padding they carried
was the largest.
