# B4a: the Coulomb build's four-density cap, and what removing it is worth

Environment: conda env `barw-dev`, `gpu4pyscf-cuda12x 1.8.1`, one
A100-PCIE-40GB shared with other work on a Perlmutter login node,
`direct_scf_tol = 1e-13`, `OMP_NUM_THREADS=8`, one process at a time.  The card
is shared, so absolute times move with what else is on it; **every variant of
every comparison below runs back to back in one process against the same
density, the same task list and the same screening data**, which is the same
practice `docs/ROUND2_1.8.1.md` used on this card.  Nothing here is quoted from
an earlier round.

Catalogue item **B4a** (`instructions/06.gpu4pyscf_kernel_targets.md` §4.B) is
ranked **second** by measured Amdahl share -- 31.3 % of a Hessian -- and
`docs/CATALOGUE_06_STATUS.md` records it as the highest-ranked entry with
nothing done to it.  This is that round.

---

## 0. Summary

| | |
|---|---|
| **The claim** | GPU4PySCF's `MD_build_j` takes at most `DM_BLOCK = 4` densities per launch (`scf/j_engine.py` pins `n_dm` to 4 inside `_md_j_engine_quartets_scheme`; the driver steps `dm_offset += 4`), so a CPHF or Davidson build with 126 densities runs the whole kernel 32 times over -- Hermite `Rt` recurrence included, although `Rt` does not depend on the density at all |
| **The ceiling, measured against GPU4PySCF** | real and large: a four-density pass on PfPMT/6-31G\* costs **5.08 s**, of which only **0.40 s per density** is the contraction.  `R/C = 9.65`, so removing the cap would be worth up to **3.4x** |
| **What was written** | the density dimension carried *inside* the quartet loop: `gen_j_high.py --ndm` emits one kernel per (class, block size), `Rt` built once per quartet and contracted with every density in the block, each shared `Rt` load feeding `ndm` fused multiply-adds instead of one.  Blocks of 4, 8 and 16, over every class an spdf basis reaches |
| **What it is worth** | **1.23x** on the whole J build at `n_dm = 8, 16, 32` and **1.57x** at `n_dm = 4` on a 284-atom cluster; **1.19x** and **1.55x** on a 545-atom one -- all of it from the kernel, none of it from the cap |
| **The result that matters** | **that 3.4x ceiling is not the cap's.**  Measured on *our* kernels, `R/C = 0.37`, so eliminating the four-density cap entirely is worth at most **1.09x** -- and the blocks of 8 and 16 that actually do eliminate it measure **0.90x** and **0.64x** against blocks of 4, because the accumulators they need cost more occupancy than the recurrence they save.  The 3.4x is the per-quartet overhead the cap repeats (the catalogue's §3.6 P1-P5), not the recurrence, and the single-density rewrite had already removed it |
| **What ships** | blocks of **four** -- GPU4PySCF's own block size -- so the gain is the kernel's, honestly labelled.  `--ndm 4,8,16` plus `FASTJ_MDM_BLOCKS` reproduces the ablation |
| **Correctness** | 162 element-wise fixed-density comparisons of the AO-space J against GPU4PySCF, six molecule/basis combinations to f functions, three density shapes, `n_dm` from 1 to 21: worst relative difference **1.5e-13** |

---

## 1. The ceiling, measured on GPU4PySCF first

`benchmarks/ndm_j.py` times one whole J build over every `(i,j,k,l)` task at a
range of `n_dm`, driving GPU4PySCF's own `MD_build_j` through `fastj`'s task
loop so that nothing but the density count changes.  Every density is a copy of
the same matrix, so `dm_cond` -- and therefore the surviving quartet list -- is
identical at every `n_dm`; that is deliberate, and it isolates the block cap
(B4a) from the union-screening problem (B4c), which is a separate entry.

`PfPMT`, 284 atoms, 6-31G\*, 2268 AO, density after one SCF cycle:

| `n_dm` | T (s) | T/`n_dm` (ms) | passes |
|---|---|---|---|
| 1 | 2.199 | 2199 | 1 |
| 2 | 4.303 | 2151 | 1 |
| 4 | 6.396 | 1599 | 1 |
| 8 | 10.014 | 1252 | 2 |
| 9 | 14.276 | 1586 | 3 |
| 12 | 15.477 | 1290 | 3 |
| 16 | 20.082 | 1255 | 4 |
| 32 | 40.215 | 1257 | 8 |

The marginal cost per density is flat at **1.26 ms/AO-pair-set** from `n_dm = 8`
upward, and `T` tracks `ceil(n_dm/4)` and not `n_dm`.  Fitting
`T(n) = ceil(n/4)*R + n*C + L` to the three points that straddle a pass boundary
away from GPU4PySCF's small-`n_dm` specialisations -- `n = 8, 9, 12` --

    R = 3.862 s   C = 0.400 s   L = -0.911 s     R/C = 9.65

`L` comes out negative, which says the three-parameter model is not exact; `R/C`
is what the rest of this document uses and it is robust to that, because it also
falls straight out of the raw table (a pass is 5.08 s, three more densities
inside a pass cost 1.20 s).

**One pass costs as much as 9.65 densities of contraction.**  Removing the cap
would then be worth `1 + R/(4C) = ` **3.4x** on the J build, at any `n_dm` large
enough for the pass count to dominate.  That is what makes B4a look like a
rank-2 target, and it is why this round was attempted.

### Where the time is, class by class

`benchmarks/ndm_j.py --perclass`, same system, GPU4PySCF's kernels, seconds at
three density counts:

| class | n=8 | n=9 | n=12 | `R/C` |
|---|---|---|---|---|
| (2,1) | 1.761 | 2.324 | 2.650 | 4.2 |
| (3,2) | 1.691 | 2.331 | 2.537 | 8.3 |
| (1,1) | 1.111 | 1.741 | 2.138 | 3.8 |
| (1,0) | 0.999 | 1.788 | 1.969 | 12.0 |
| (3,1) | 0.903 | 1.134 | 1.360 | 2.1 |
| (2,2) | 0.829 | 1.059 | 1.252 | 2.6 |
| (2,0) | 0.705 | 1.148 | 1.403 | 4.2 |
| (4,2) | 0.441 | 0.609 | 0.662 | 8.5 |
| (3,0) | 0.379 | 0.498 | 0.569 | 4.0 |
| (3,3) | 0.344 | 0.469 | 0.516 | 6.9 |
| (4,1) | 0.304 | 0.335 | 0.453 | — |
| (0,0) | 0.248 | 0.455 | 0.486 | 18.9 |
| (4,3) | 0.157 | 0.214 | 0.235 | 7.4 |
| (4,0) | 0.103 | 0.122 | 0.154 | 0.8 |
| (4,4) | 0.046 | 0.063 | 0.069 | 8.5 |

Two things came out of this table and both changed the design.  The **low**
classes carry the time -- the top seven are 80 % of the build -- which is the
opposite of the single-density picture at def2-TZVP, where 64 % of the work was
in classes GPU4PySCF does not unroll (`docs/ROUND2_1.8.1.md` §1).  And `(0,0)`
and `(1,0)`, 14 % of the build between them, have the two largest amortisation
ratios, so they had to be generated even though their Hermite tensors have one
and four components and the design's cooperative `Rt` levels are empty for them.

---

## 2. What was written

`electrolites/codegen/gen_j_high.py` gained an `ndm` dimension.  At `ndm = 1`
its output is byte for byte what it was, which is checked by diffing against
`git show HEAD:` on every change; at `ndm > 1` one kernel per (class, block
size) is emitted, named `j_dm<ndm>_<lij>_<lkl>` so that `_ksplit`'s pair-sum
rule still reads the class off the name.

Four things differ from the single-density kernel.

1. **The density loop is inside the quartet loop.**  `Rt` is built once per
   shell-pair quartet and contracted with every density in the block, so the
   recurrence, the Boys function, the geometry and the screening are paid once
   for `ndm` densities instead of once for each.  That is B4a's whole content.
2. **Each `Rt` load feeds `ndm` fused multiply-adds.**  The shared-memory read
   is hoisted out of the density dimension, so arithmetic intensity per shared
   load rises with the block size.  This is a second, separate gain from the
   first and it is the one that would matter if the kernel were
   shared-bandwidth bound.
3. **The `vj_ij` reduction over `ty` is a warp shuffle, not shared memory.**
   With `ndm` accumulators per Hermite slot the shared route would cost
   `ndm*nvj*(log2(TY)+2)` `__syncthreads()` per bra tile -- 160 barriers where
   the single-density kernel has 20.  The shuffle needs the `TY` lanes that
   share a bra pair to sit in one warp, so the multi-density kernels assert
   `threadsx*threadsy == 32`.
4. **The density dimension is a `#pragma unroll` loop behind two macros**, so a
   class costs the same number of *emitted lines* at `ndm = 16` as at `ndm = 1`.
   Written out, `(6|6)` would be 113 000 lines for one kernel; that is the
   mistake the per-lane exchange emission made and `README_fastk.md` records.
   The macro names carry the kernel's own so `_ksplit` can still cut the file
   into independent chunks.

Not changed: the task and tile decomposition, the screening, the symmetry
factors and the Hermite recurrence itself are GPU4PySCF's.

---

## 3. Correctness

`tests/test_j_ndm.py`.  The oracle is GPU4PySCF's own `MD_build_j` on the same
molecule, the same screening data and the same Hermite density -- an
element-wise, fixed-density comparison of the **AO-space** J, which is what the
catalogue's 6 asks for and what `benchmarks/perclass_j.py` explains: the
Hermite components of a high-`l` pair are large and cancel in the transform back
to AO space, so `vj_xyz` is not the thing to compare.  A converged energy is not
used and would not be a signal at all here, because these kernels are only
reached with more than one density, i.e. inside a CPHF or Davidson solve and
never on the SCF's own path.

Three density families, because they exercise different parts of the kernel:

* **identical copies** -- every J must come back the same, which catches a wrong
  `m*dm_size` stride (the failure mode where the kernel reads or writes the
  wrong density) that a single random draw can hide.  The agreement is to
  round-off and not bit for bit, because each density is accumulated by
  `atomicAdd` at its own address and the hardware serialises per address, so the
  summation order differs between them;
* **random symmetric perturbations** of five different magnitudes, so the union
  screening is exercised and a density whose own contribution is tiny still has
  to come out right;
* **atom-local densities**, the shape a CPHF right-hand side actually has.

Six molecule/basis combinations (H2O and ethanol in 6-31G\*, cc-pVDZ, def2-TZVP
and cc-pVTZ -- so up to f functions -- plus SiH4 for a third-row element), three
families, nine values of `n_dm` from 1 to 21 chosen to straddle every block
size, to leave a tail for GPU4PySCF, and to include `n_dm = 1` so that the
single-density path this round did not change is covered too: **162
comparisons, worst relative difference 1.5e-13.**

Two bugs the test found before it passed, both in the test rather than the
kernel, are worth recording because they are easy to repeat: judging the
identical-density case on bit-exactness (it cannot be, see above), and judging
any of it on an *absolute* difference when the test densities are synthetic and
not normalised the way a converged density is.

---

## 4. What it is worth, and why the ceiling in §1 was not the right ceiling

`benchmarks/ndm_j.py --sweep-blocks` runs every density-block set in one
process against the same density and task list.  `PfPMT`, 284 atoms, 6-31G\*:

| `n_dm` | blocks | T (s) | vs GPU4PySCF | launches per class |
|---|---|---|---|---|
| 1 | GPU4PySCF | 2.266 | 1.00x | 1 |
| 1 | (single-density family) | 1.310 | **1.73x** | 1 |
| 4 | GPU4PySCF | 6.575 | 1.00x | 1 |
| 4 | 4 | 4.176 | **1.57x** | 1 |
| 8 | GPU4PySCF | 10.282 | 1.00x | 2 |
| 8 | 4 | 8.371 | **1.23x** | 2 |
| 16 | GPU4PySCF | 20.589 | 1.00x | 4 |
| 16 | 4 | 16.731 | **1.23x** | 4 |
| 16 | 8 | 18.566 | 1.11x | 2 |
| 16 | 16 | 26.050 | 0.79x | 1 |
| 16 | 4,8 | 18.489 | 1.12x | 4 |
| 16 | 4,8,16 | 25.433 | 0.81x | 4 |
| 32 | GPU4PySCF | 41.355 | 1.00x | 8 |
| 32 | 4 | 33.489 | **1.23x** | 8 |
| 32 | 8 | 37.165 | 1.11x | 4 |
| 32 | 16 | 52.134 | 0.79x | 2 |
| 32 | 4,8 | 36.912 | 1.12x | 8 |
| 32 | 4,8,16 | 50.903 | 0.81x | 8 |

Every ratio at `n_dm = 32` reproduces its `n_dm = 16` value to within 1 %, which
is the internal consistency check the design needs: nothing here depends on how
many densities the caller happens to bring.

**The ordering is monotone the wrong way.**  Blocks of 8 halve the number of
passes and are 10 % *slower* than blocks of 4; blocks of 16 quarter the passes
and are 36 % slower.  The cap is not what the time was going into.

### The second system

The catalogue's §3.4 rule -- re-measure on a second, differently-sized system
and count the result only if the ratios agree -- applied to `HcgC`, 545 atoms,
4170 AO, 1.8x the AO count of `PfPMT`, same basis and the same one-cycle
density:

| `n_dm` | GPU4PySCF (s) | blocks of 4 (s) | `HcgC` | `PfPMT` | difference |
|---|---|---|---|---|---|
| 1 | 9.101 | 5.536 | **1.64x** | 1.73x | 5 % |
| 4 | 28.138 | 18.209 | **1.55x** | 1.57x | 1 % |
| 8 | 43.422 | 36.492 | **1.19x** | 1.23x | 3 % |
| 16 | 86.852 | 73.100 | **1.19x** | 1.23x | 3 % |

Every ratio lands within 1-5 % of its `PfPMT` value, which is the empirical
form of "nothing here is tuned to a problem size".  The `n_dm = 12` row of the
`PfPMT` table reads 1.35x rather than 1.23x, and that is GPU4PySCF's numerator
and not our denominator: three of our four-density launches take 12.538 s,
exactly three times the 4.18 s a launch takes everywhere else, while
GPU4PySCF's three passes take 16.878 s against the 15.4 s its own per-pass cost
predicts.  It is not quoted above for that reason.

### The arithmetic, on our kernels rather than GPU4PySCF's

The same two-point fit as §1, applied to the rows above.  One four-density launch
is 4.18 s (4.176 at `n_dm=4`, 8.371/2, 16.731/4 -- the three agree to 0.2 %),
and the single-density family builds the same J in 1.310 s.  So

    R + 1*C = 1.310      R + 4*C = 4.176
    C = 0.955 s          R = 0.354 s          R/C = 0.37

**One caveat on that fit**: the two points are not the same kernel.  `n_dm = 1`
runs the single-density family -- twelve classes lifted out of GPU4PySCF's
`unrolled_md_j.cu` and seventeen written -- and `n_dm = 4` runs the
multi-density one, which is written throughout and configured differently.  The
fit is therefore an estimate of what a *quartet* costs in kernels of this
quality, not of one kernel's own `R` and `C`.  The same numbers on `HcgC` --
5.536 s and 18.209 s -- give `C = 4.224`, `R = 1.312`, `R/C = 0.31`, which is
the same conclusion from a system 1.8x the size.

against GPU4PySCF's `R/C = 9.65`.  **The density-independent part of a quartet
is twenty-six times smaller in these kernels than in GPU4PySCF's.**  It is not
the Hermite recurrence that shrank -- that is the same arithmetic in both -- it
is everything the catalogue's §3.6 P1-P5 lists: the fp64 divisions and the `sqrt` per
primitive quartet, the incomplete-gamma values staged through shared memory with
a `__syncthreads()` in the innermost loop, the block-uniform bra data
recomputed per thread per quartet, the run-time `omega` branch.  GPU4PySCF pays
all of that **once per pass**, so capping the pass at four densities makes it
pay it `n_dm/4` times, and *that* is the 3.4x.  Fixing the overhead and lifting
the cap are two ways of not paying the same bill, and the first one is already
done.

What is left for the cap is `1 + R/(4C) =` **1.09x** (1.08x on `HcgC`), and it
does not grow with `n_dm`.  Blocks of 8 and 16 have to buy that 9 % against
whatever a deeper density dimension costs, and they do not: they lose 10 % and
36 %.  **What exactly they lose it to was not established** -- no Nsight run
was made, and the arithmetic below is a count, not a measurement.  For the
largest class, `(2,1)`, a thread carries roughly 55 doubles of live state at
`ndm = 4` (40 of `vj_ij`, four of the density accumulator, the `gamma_inc`
array and the scalars) and roughly 75 at `ndm = 16`, which is 110 against 150
registers and, on an A100's 65 536 registers per SM, about 18 warps against 13.
The `ndm`-deep `dm_ij_cache` and `vj_kl_cache` grow with the block too --
33 KB against 13 KB of shared memory for that class.  Either could be the
cause; the useful part of the result does not depend on which, because the
ceiling is 9 % whatever it is.  This is the same shape of result as
`docs/ROUND2_1.8.1.md` §3b, where deleting *every* `vk` atomicAdd turned out to
be worth 1.1 %: the ceiling was measured before the thing was built, but here it
was measured on the wrong baseline, and the correction is what the round
produced.

**So the honest label on the 1.23x is "our Coulomb kernel, run four densities at
a time", not "the four-density cap removed".**  The cap is still there; it is
now known to be worth at most 9 %, and the two block sizes that remove it are in
the repository, measured, and not shipped.

### What ships, and what it means for a Hessian

Blocks of four, which is GPU4PySCF's own block size, for every class inside the
emission budget of §5 -- every class a double-zeta polarised basis reaches,
plus `(5,3)` and `(6,2)`.  Anything larger, and any `n_dm` below four, falls
through to GPU4PySCF unchanged.

The catalogue puts the J build at 31.3 % of a Hessian at 42 atoms/def2-SVP.
A 1.23x on that block is `1/(1 - 0.313 + 0.313/1.23) =` **1.06x** on the
Hessian, if that share transports -- and the catalogue's §3.2 warns that it may not, since it
was measured at 430 AO and this was measured at 2268.  **Nothing in this
document is an end-to-end speedup**; it is a kernel measurement, and turning it
into an application number means running the whole Hessian through both codes,
which this round did not do.

---

## 5. Compile time

The multi-density kernels are a separate NVRTC module, built only when a build
with more than one density asks for it, so an ordinary SCF never compiles them
-- the same reasoning as `docs/ROUND2_1.8.1.md` §4, where compiling only the
angular-momentum classes a basis reaches took a 6-31G\* job's first run from
162 s to 37 s.

The name set handed to `_ksplit` deliberately does **not** depend on `n_dm`.  A
Davidson solve calls `get_j` with a different number of trial vectors on every
iteration, and a name set that moved with it would hand NVRTC different source
text each time and recompile the family over and over.

### Why there is an emission budget

Compile time is markedly super-linear in the instruction count inside one basic
block, and the density dimension multiplies that count by `ndm` even though the
source stays the same size.  Per kernel, NVRTC on this card:

| kernel | source | NVRTC |
|---|---|---|
| `j_dm4_2_1` | 33 kB | 1.0 s |
| `j_dm8_3_3` | 112 kB | 8.3 s |
| `j_dm16_3_3` | 133 kB | 19.5 s |
| `j_dm4_4_4` | 242 kB | 28.2 s |
| `j_dm8_4_4` | 276 kB | 45.4 s |
| `j_dm16_4_4` | 348 kB | **92.8 s** |

Emitting every class at every block size costs **413 s** for the 6-31G\* subset
alone -- more than the entire rest of the package.  `MDM_EMIT_BUDGET` caps
`ndm*nf3ij*nf3kl`; a class past it falls through rather than being compiled.
With what ships (blocks of four, budget 4900) the whole multi-density family is
23 kernels, and the first run pays NVRTC once and CuPy's disk cache thereafter.

---

## 6. What is left

1. **The launch configurations are not tuned.**  `NDM_CFG` is empty:
   `default_ndm_cfg` picks `gout_stride` from the register budget and narrows
   the ket tile until shared memory fits, which is a starting point and not an
   optimum.  `benchmarks/sweep_j_high.py` is the tool and
   `gen_j_high.py --vj-budget` is the axis that matters -- it trades `vj_ij`
   registers against threads per quartet, and §4 says the block-size result turns on it.
   That is also the honest way to test §4's conclusion: if a re-tuned `ndm = 8`
   kernel still lost, the 9 % ceiling would be confirmed from a second
   direction.
2. **The classes past the emission budget.**  `(4,4)` upward at blocks of eight
   and sixteen, and `(5,4)` upward at any block size, fall through at
   `n_dm > 1`.  At 6-31G\* `(4,4)` is 0.5 % of the build; at def2-TZVP the high
   classes are most of it, and the single-density round's own experience (`ROUND2` §1) says
   that is where a written kernel is worth the most.  Whether the same is true
   with several densities has **not** been measured.
3. **K in CPHF is untouched** -- catalogue B4b, `fastk.py:175` still declines
   `n_dm != 1`.  §1's argument transfers: `fastk` has had the P1-P5 treatment
   too, so the headroom there should be measured on `fastk`'s kernels and not
   on GPU4PySCF's before anything is written.
4. **The 1.09x itself.**  If it is ever wanted, the way to get it is not a
   bigger density block but a kernel whose accumulators do not grow with it --
   for instance carrying the density dimension on the `gout` axis rather than in
   registers.  That design does not multiply the register file, but it idles
   `ndm-1` of every `ndm` thread groups while `Rt` is being built, which on a
   card where one block fills an SM is not obviously cheaper.  It was not tried.
5. **`fastj` on GPU4PySCF 1.7.x** still declines `n_dm > 1`: the multi-density
   path is on the 1.8.x driver only.

## Files

| file | what it is |
|---|---|
| `electrolites/codegen/gen_j_high.py` | `--ndm` writes the multi-density Coulomb kernels; `--vj-budget` is the register/occupancy axis |
| `electrolites/kernels/fastjmdm_launch.json` | their launch table |
| `electrolites/fastj.py` | `_dm_split`, `_run_mdm`, `_run_ref`: how `n_dm` is covered and what is handed back |
| `benchmarks/ndm_j.py` | the cost model, the per-class table, the block-size ablation |
| `tests/test_j_ndm.py` | the AO-space correctness suite |
