# fastejk2 — `ejk_ip2`, and the seven classes GPU4PySCF unrolls where the entry assumes twenty-five

Catalogue entry **B3**, rank 9 in `instructions/06`'s §5 list.  It names the two
second-derivative ERI kernel families a Hessian's `partial_hess_elec` drives,
`hessian/rhf.py:_partial_ejk_ip2` → `RYS_per_atom_jk_ip2_type12` and
`_type3`, and says:

> `unrolled_ejk_ip2_type12.cu` (14568 lines), `unrolled_ejk_ip2_type3.cu`
> (15696 lines), same shape as B1. … as B1.  Review B §B.5.1: "register
> pressure worse than high-L ERIs at the same nominal basis L", so P3 is worth
> more here than on K

The two files are as long as the entry says.  What is in them is not what the
entry assumes, and §1 is that measurement.

---

## 0. Summary

* **B3 is mostly a *general*-kernel problem.**  GPU4PySCF unrolls **seven**
  angular-momentum classes per ip2 type — `0000 1000 1010 1011 1100 1110 1111`,
  nothing with a d function anywhere — against twenty-five for the exchange
  build.  Everything else runs `rys_contract_jk_ip2.cu`'s general kernel, which
  carries its g-array addresses in shared `idx_i/j/k/l` arrays and its loop
  bounds at run time.  At def2-SVP that is **63 %** of the stage.  §1.
* This module changes two things, each separately switchable, and only one of
  them is on by default:
  * the **launch configuration**, which upstream derives from the
    shared-memory budget alone and never measured — **1.13x** on the whole
    stage for two lines of Python and no compiler.  **On by default.**
  * the **kernels' arithmetic**, lifted from GPU4PySCF's own sources with the
    fp64 divisions folded (P1) and a register cap added (P3) — **1.02x** for
    P1, and **0.58x** with P3, i.e. the register cap the entry asks for is a
    1.75x *regression*.  Correct to 1e-13 either way, but not worth an `nvcc`
    dependency and a vendored header tree, so it is **off unless
    `FASTEJK2_KERNELS=1`**.  §2b has the grid.
* The P3 cap has to be written at the block size the driver launches, and
  finding that out was the round's one real surprise: the *same* 128-register
  cap expressed as `__launch_bounds__(512, 1)` rather than `(256, 2)` computes
  numbers **9x–37x wrong**, with the launch geometry identical.  §4 has the
  ablation grid that isolates it and says what it is not.
* The oracle is unusually clean: the same Python driver, the same task lists,
  the two kernel sets swapped underneath in one process.  §5.
* Rank 9's own position in the ranking survives the round: `ejk_ip2` is 13.9 %
  of a 42-atom Hessian and **3.0 %** of a 113-atom one, so whatever it is worth
  shrinks with system size.  §6 says what is left and why it is a
  `gen_khigh`-sized project for a small entry.

---

## 1. Where `ejk_ip2` actually goes

`benchmarks/ejk_ip2_probe.py` wraps each `RYS_per_atom_jk_ip2_type{12,3}` call
with a `cudaDeviceSynchronize()` on both sides and attributes it to its
angular-momentum class, which the driver derives from `shls_slice` exactly as
the kernel does — so "unrolled" and "general" are read off the class, not
guessed.  Penicillin V/def2-SVP, converged density, one A100-PCIE-40GB:
`_partial_ejk_ip2` = 8.63 s, of which **98.9 %** is in the two kernels.

| group | time/s | share of the stage |
|---|---|---|
| type12, the 7 unrolled classes | 1.892 | 21.9 % |
| type12, the general kernel | **2.494** | **28.9 %** |
| type3, the 7 unrolled classes | 1.283 | 14.9 % |
| type3, the general kernel | **2.862** | **33.2 %** |

The twelve most expensive classes are `type3/2111`, `type12/1111`,
`type3/2110`, `type12/2110`, `type12/2111`, `type3/1111`, `type3/2011`,
`type12/1110`, … — **low** angular momentum with many surviving shell pairs,
not the wide classes.  Two consequences for the entry:

* "as B1" describes 37 % of the stage.  B1's treatment (P1–P5 on an unrolled
  kernel) is exactly right *there*, and `fastejk` measured about 2x from it on
  the gradient — so the whole of that half is worth roughly 1.6 s, **3 % of a
  42-atom Hessian and 0.4 % of a 113-atom one.**
* The other 63 % is the situation `electrolites/codegen/gen_khigh.py` was
  written for on the exchange side, one shell of angular momentum earlier.
  That is a generator project, and §6 sizes it.

## 2. What the module changes

### (a) The launch configuration

`_ip2_quartets_scheme` (`hessian/rhf.py:390`) and its `type3` twin:

```python
counts = (shm_size - ij_prims*8) // (unit*8)
n = min(THREADS, _nearest_power2(counts));  gout_stride = THREADS // n
```

— 256 threads, as many shell quartets as shared memory allows, whatever is left
to `gout_stride`.  Nothing was measured to arrive at that, and the same
question on the exchange side (`benchmarks/sweep_j_high.py`, `gen_khigh.py`'s
`--gout-max`) was worth a factor on several classes.

`benchmarks/ejk_ip2_sweep.py` sweeps it.  Penicillin V/def2-SVP, one process,
GPU4PySCF's kernels throughout:

| threads | quartet halvings | time/s | over stock |
|---|---|---|---|
| 256 | 0 (stock) | 7.429 | 1.00x |
| 256 | **1** | **6.586** | **1.13x** |
| 256 | 2 | 6.575 | 1.13x |
| 128 | 0 | 9.620 | 0.77x |
| 128 | 1 | 8.296 | 0.90x |
| 128 | 2 | 7.849 | 0.95x |
| 512 | 0, 1, 2 | — | fails: *too many resources requested for launch* |

Halving the quartet count moves 128 of the block's threads from the quartet
dimension into `gout_stride`, which is where the contraction's `nfij*nfkl` loop
is spread — the wide ip2 contraction has far more work per quartet than the
exchange build's, so it wants more lanes on it.  The second halving is
level, and 128-thread blocks are worse at every split.  The 512-thread rows are
the P3 story: they fail on stock code because the kernels are compiled at the
255-register ceiling, and 512 × 255 does not fit an SM's register file.

**1.13x for two changed lines of Python, and it took three measurements to
believe it.**  The first sweep had no untimed warm-up, so its stock row carried
the module load, `cudaFuncSetAttribute` and the first touch of `dd_pool` — about
12 % on this stage.  Adding the warm-up then made the change look like 1.00x,
but for a different reason: `benchmarks/ejk_ip2_bench.py` captured
`hrhf._ip2_quartets_scheme` *after* importing `fastejk2`, so its "stock" row was
already carrying the swept scheme and its "+ scheme" row swept it twice.  The
number above is from the fixed 2×2 — kernels and scheme switched independently
off a genuinely unpatched baseline — and it agrees with the unwarmed sweep's
ratio by accident, which is the sort of coincidence that makes a bad
measurement survive.

End to end that is small, and the entry's own ranking says it should be:
`ejk_ip2` is 7.43 s of a 53.4 s penicillin V Hessian after the `fastjk` round,
so 1.13x on it is about **1.6 %** of the run, and 3.0 % of a taxol Hessian
becomes 2.7 %.  It is shipped because it costs two lines of Python, no
compiler, no vendored code and no new failure mode -- not because it changes
what a Hessian costs.

`FASTEJK2_SHIFT` selects the halving; `FASTEJK2_NO_SCHEME=1` is the ablation.

### (b) The kernels

`codegen/gen_ejk_ip2.py` lifts the three GPU4PySCF sources and applies two
transformations, asserting the count of every substitution and then asserting
that **no fp64 division is left** anywhere but the two reciprocals it
introduces:

**P1.**  Per primitive quartet, upstream evaluates

| what | where | how often |
|---|---|---|
| `al / akl` | the ket primitive-pair loop | once per ket pair |
| `aj / aij` | the bra primitive-pair loop | once per primitive quartet |
| `cicj*ckcl / (aij*akl*sqrt(aij+akl))` | " | " |
| `aij*akl / (aij + akl)` (`theta`) | " | " |
| `rt / (aij + akl)` (`rt_aa`) | the **Rys root** loop | once per root |
| `.5/aij` (`b10`), `.5/akl` (`b01`) | " | twice per root |

At `nroots = 5` that is eighteen fp64 divisions and a `sqrt` per primitive
quartet, each about twenty instructions on sm_80.  After: `1/aij` and `1/akl`
are formed once where the exponents are read, one `rsqrt(aij+akl)` supplies
both `1/(aij+akl)` and `1/sqrt(aij+akl)`, and every division in the hot loops
is a multiply — **two divisions and one `rsqrt` per primitive quartet, none per
root**.  Three of the per-root divisions live in the shared
`build_rys_gxyz.cuh` macro, which is transformed rather than copied.

**P3** (`--p3`, and **off by default** for the reason two paragraphs down).
`__launch_bounds__(256, 2)` on all sixteen kernels, where upstream has no
register cap at all.  §4 is the correctness trap that turned up while applying
it.

### What the two transformations are worth

Each build variant timed against GPU4PySCF's kernels **in one process** on one
converged density, with the launch scheme switched independently, `min` of
three after an untimed warm-up (`benchmarks/ejk_ip2_bench.py`, penicillin
V/def2-SVP):

| build | stock | scheme only | kernels only | kernels + scheme |
|---|---|---|---|---|
| **P1 only** — what the generator now writes by default | 7.429 s | 6.586 s (**1.13x**) | 7.267 s (**1.02x**) | 6.408 s (1.16x) |
| **P1 + P3** (`--p3`) | 7.430 s | 6.585 s (1.13x) | 12.726 s (**0.58x**) | 11.778 s (0.63x) |

Both halves of that read against the entry.

* **P1 is worth 1.02x here, against 1.7x–2x on the exchange build.**  The
  divisions really are all there — eighteen and a `sqrt` per primitive quartet
  at five roots — and taking them out changes almost nothing, so these kernels
  are not division-bound.  What they are bound by is visible in §1: 63 % of the
  stage is a kernel whose g-array addresses come out of shared `idx_*` arrays
  and whose loop bounds are run-time values, and no amount of arithmetic
  cleanup reaches that.
* **P3 is a 1.75x regression, where the entry expects it to pay *more* than on
  K.**  The entry's reasoning — "register pressure worse than high-L ERIs at
  the same nominal basis L" — is right about the pressure and backwards about
  the conclusion.  These kernels want all 255 registers: capping them at 128 to
  fit two blocks per SM trades occupancy for spills, and the spills cost more.
  On the exchange side P3 paid because `gen_khigh`'s kernels were *designed* to
  a register budget (exact tiling, `gout` split over lanes) rather than capped
  after the fact.

So what ships from B3 is the two lines of Python, and the kernels stay behind
`FASTEJK2_KERNELS=1` — correct, reproducible, and not on by default.
`--no-p1` and `--no-p3` reproduce the rows above.

## 3. Why this one is built with `nvcc`

Every other module here is compiled at run time by NVRTC through CuPy.  These
kernels cannot be: they are split over three translation units, the dispatchers
take **kernel pointers** (`adjust_threads(rys_ejk_ip2_type12_0000, …)`), and
they need relocatable device code.  So `electrolites/_nvcc.py` builds them into
`libmyejk2.so` with `nvcc` on first use — about 25 s — and caches it under
`~/.cache/electrolites/sm_<arch>/<source hash>/`, the same route `fastk`'s
general exchange kernel takes.  If `nvcc` is not installed the module says so
once and stands down, and the launch-scheme half (a) still applies.

Two consequences worth stating plainly.

* **The support headers have to be vendored**, because the kernels bind to this
  release's C struct layouts and a wheel install carries no CUDA source.  They
  sit in `kernels/cuda/gvhf-rys181/`, separate from the 1.7.0 copies `fastk`
  uses: `BoundsInfo` gained per-group `q_cond`/`s_cond` between the two, so the
  two sets are not interchangeable.
* **This module is therefore version-gated where the others are API-gated.**
  `check_targets` protects against a moved Python entry point; it cannot see a
  changed struct.  `fastejk2.SUPPORTED` lists the releases whose layout the
  vendored headers match, and the module refuses to patch anywhere else and
  says which release it wanted.  That gate is the reason the round did not
  simply copy the sources and hope.

## 4. A register cap has to be written at the block size that is launched

`__launch_bounds__(512, 1)` and `__launch_bounds__(256, 2)` are the *same*
register cap: 65536/(512·1) and 65536/(256·2) are both 128.  One of them is
correct and the other is not.

The full ablation grid, `tests/test_ejk_ip2.py --no-hessian`, 6-31G\* on three
molecules, each variant against **the same driver running upstream's kernels**
in the same process:

| variant | worst relative difference on `ejk` |
|---|---|
| verbatim lift, no transformation at all | **3.5e-15** |
| **P1 only** (`--no-p3`) | **1.8e-13** |
| P3 only, `(512, 1)`, `adjust_threads` live | 3.7e+01 |
| P1 + P3, `(512, 1)`, `adjust_threads` live | 3.7e+01 |
| P1 + P3, `(512, 1)`, `adjust_threads` **disabled** | 3.7e+01 |
| **P1 + P3, `(256, 2)`**, `adjust_threads` disabled | **1.4e-13** |
| P1 + P3, `(256, 2)`, `adjust_threads` live | 6.1e-13 |

What that grid rules out matters as much as what it shows.

* **The lift is sound**: verbatim agrees to 3.5e-15, so nothing about
  vendoring the headers, renaming the entry points or building with `nvcc`
  changes a number.
* **P1 is sound**: 1.8e-13 on its own, and 1.4e-13 together with the cap.
* **The fault is the launch bound alone, and it is not the block shape.**  The
  obvious suspect was `vhf.cuh:158`'s `adjust_threads`, which doubles the
  quartets per block when a kernel uses 128 registers or fewer — dead code on
  stock GPU4PySCF, because without a cap no ip2 kernel is ever under 128, and
  live the moment the cap arrives.  But `(512, 1)` is wrong with that macro
  **disabled** too, i.e. with the launch geometry identical at 256 threads, and
  `(256, 2)` is right with it live.  So two builds with the same register cap
  and the same launch geometry, differing only in the `maxThreadsPerBlock`
  written into the bound, disagree by thirteen orders of magnitude.  **The cause
  is not identified.**

It is recorded because it is a trap for exactly the person the catalogue's B3
entry sends here — the entry asks for P3, and the natural way to write P3
safely, a generous upper bound on the block size, is the wrong way — and
because a wrong answer of this size is silent: every kernel launches, no error
is raised, and only an element-wise comparison against upstream catches it.

§2b then made the whole question moot: P3 is a 1.75x regression here, so the
generator does not write it unless asked (`--p3`), and the shipped sources
carry no register cap and a verbatim `vhf.cuh`.  When `--p3` *is* asked for,
the cap is `(256, 2)` — the block shape `_ip2_quartets_scheme` actually asks
for — and `adjust_threads` is switched off alongside it, not because it is the
cause but so that nothing can change the block shape out from under the bound.
`--keep-adjust-threads` and `--maxthreads/--minblocks` reproduce every row
above.

This is the third time in this repository that applying a catalogue entry's
own prescription produced a *correctness* finding rather than a speed one; the
other two are `docs/README_fastxchess.md` §5 (LDA and GGA factor conventions)
and `docs/CATALOGUE_06_STATUS.md` §6 hazard 6 (the non-reproducible mGGA
`_get_vxc_deriv1`).

## 5. Correctness

The oracle is the **same driver running GPU4PySCF's kernels**, in one process,
on one converged density: `fastejk2.unpatch()` puts the two symbols back, so
the task lists, the screening, the launch schemes and the per-atom contraction
are held fixed and only the arithmetic differs.  That is stronger than a
two-process comparison, and it is the reason this module has the repository's
only `unpatch`.

Two criteria, `tests/test_ejk_ip2.py`: the largest element-wise difference on
the `natm × natm × 3 × 3` `ejk` array relative to its largest element, and the
whole analytic Hessian built on top of it.  The expected agreement is round-off
and not bit for bit, because one `rsqrt(s)` now supplies `1/s` and
`rsqrt(s)·rsqrt(s)` differs from `1/s` by about two units in the last place.
Five molecule/basis pairs to def2-TZVP:

```
worst: ejk 7.023e-13 relative (tol 1e-11), hessian 6.203e-11 absolute (tol 1e-08)
```

The test forces the kernels on (`apply_patch(force=True)`) rather than reading
`FASTEJK2_KERNELS`, since a switch that hides the thing under test would hide
the measurement with it.

## 6. What is left of B3

* The 63 % that is the general kernel needs what `gen_khigh.py` gave the
  exchange build: one exactly-tiled kernel per class with compile-time g-array
  addresses, instead of a 3×3×3×3 tile and shared `idx_*` arrays.  That is a
  generator of the same size as `codegen/gen_ejk.py` — and `gen_ejk.py` already
  writes the *first* derivative of `j·J − k·K` for all 65 classes, so the
  machinery to extend is there.
* It should be weighed against the entry's share, which **falls with system
  size**: 13.9 % of a 42-atom Hessian, 3.0 % of a 113-atom one
  (`docs/HESSIAN_1.8.1.md` §4), because the CPHF cost grows faster than
  `partial_hess_elec` does.  `docs/ROUND4_B4B.md` §7 puts three things ahead of
  it.
* One thing not in the entry and not done here: the ip2 kernels are the only
  place in the Hessian where GPU4PySCF still evaluates `theta_ij = ai*aj/aij`
  in the `cicj` cache loop with a division.  It is once per bra primitive pair
  per block, so P1 leaves it alone deliberately — recorded so the next round
  does not spend time rediscovering that it does not matter.
