# The Hessian's CPHF solve in one atom block, and GPU4PySCF's Krylov solver as GEMMs

`docs/DF_WORK_LIST.md` **W44**, found by W40.  A whole 40 GB UKS Hessian of
taxol⁺ was 1760.7 s against 1482.3 s on the full card, and all of the
difference was the CPHF (`hessian/uhf.py:solve_mo1`): on 40 GB GPU4PySCF
solves it in two atom blocks (64 + 49), each Krylov iteration run to its own
tolerance, fifteen `_get_jk` calls where the full card takes six.  The entry
asked for the patched response's per-right-hand-side memory to be measured,
the atom block sized from it, and whether taxol⁺'s 339 right-hand sides fit
one block at 40 GB.  Success: the whole 40 GB UKS Hessian within 1.05x of
the full card's.  Falsification: the response's AO arrays for all 339 sets
not fitting beside the cderi.

**Done: the whole 40 GB UKS Hessian of taxol⁺ is 1760.7 → 1348.6 s, 1.018x
of the full card's (which goes 1482.3 → 1324.5 s), and the two now agree to
5.2e-10 instead of 1.5e-4** (§3.1).  **And the round found a second, larger thing on the way: GPU4PySCF's
`krylov` itself was 174 s of the full card's 1389 s CPHF.**  Measured with
every piece of `solve_mo1` on its own row (`benchmarks/df_cphf_probe.py`,
§1), the solver's own work -- its projections, one row at a time, and the
stacking of its subspace at the end -- is 12.5 % of the CPHF of a 113-atom
UKS Hessian, and it grows faster than the response does (0.87 s of 18.8 s at
42 atoms).  `electrolites/fastcphf.py` (in `patch_all()`) replaces
`solve_mo1` in both spin cases:

| taxol⁺ UKS CPHF, 113 atoms, def2-SVP | upstream | `fastcphf` |
|---|---|---|
| 80 GB card | 1389.4 s (one block, 6 calls) | **1246.6 s** (1.115x) |
| 40 GB card (ballast 40) | two blocks, 15 calls (1655 s inside W40's Hessian) | **1255.9 s**, one block, 6 calls |
| of which Krylov's own work | 173.7 s | 27.5 s (80 GB) |
| peak above the call's start | +47.8 GB | +44.0 GB (80 GB), +37.9 GB (40 GB) |

| taxol RKS CPHF, 80 GB | 624.2 s | **555.3 s** (1.124x), peak +53.7 → +39.1 GB |
| penicillin V⁺ UKS, 42 atoms | 18.81 s | 18.27 s |
| penicillin V RKS | 7.69 s | 7.59 s |

On the 40 GB card the CPHF is **1.007x of this module's full-card time and
0.90x of upstream's full-card time**: the same six calls, the same Krylov
cycles (339, 336, 336, 201, 80 equations, both cards), the tolerance of one
113-atom block.  The falsification does not fire -- the AO arrays of all 339
sets never have to be held at once, because the response is linear and its
right-hand sides independent (§2).

---

## 1. Where the time went

`benchmarks/df_cphf_probe.py` runs the SCF and `make_h1` once (cached under
`--cache`), then only `solve_mo1`, with `krylov`, `gen_vind`'s closure, the
response, `_get_jk`, the XC response and `cache_xc_kernel` wrapped, and the
device's high-water mark read after every allocation over 64 MB (GPU4PySCF
sends those past CuPy's pool, so the pool's counters do not see them; the
first version of this probe replaced the allocator to count them and in
doing so changed which allocator ran -- a timing and an OOM that were the
probe's, §5).  Upstream, taxol⁺, 80 GB, s:

| | s |
|---|---|
| `solve_mo1` | 1389.35 |
| .. `_get_jk` (W8), 6 calls | 1091.53 |
| .. XC response (W33), 6 calls | 119.64 |
| .. `cache_xc_kernel` | 1.30 |
| .. the rest of `fx` (MO/AO transforms) | 2.16 |
| .. **`krylov`'s own work** | **173.72** |
| .. outside `krylov` and `fx` | 0.74 |

`_get_jk` runs 0.74 s per right-hand side on every call, at W8's rate; what
nothing had timed was the solver.  `lib/cupy_helper.py:krylov` keeps its
trial vectors `xs` and their images `ax` as Python lists of rows, and each
cycle

* projects the new block against every stored row, one row at a time: a
  GEMV over the block and a rank-1 update of the whole block (1.4 GB at 339
  sets) per row -- 1292 rows by the last cycle;
* at the end stacks both lists into contiguous copies (`cupy.asarray(xs)`,
  twice for `xs`) and forms the solution in another loop over rows.

The stacking is also the call's memory peak (+47.8 GB).

## 2. What changed

`electrolites/fastcphf.py`: `hessian/rhf.py:solve_mo1` and
`hessian/uhf.py:solve_mo1`, which every Hessian (density-fitted or direct),
the IR and Raman drivers and `hessian/uks.py`'s class reach through the
module global.  Upstream's arithmetic line for line, except:

1. **The response in groups of right-hand sides, the Krylov iteration in one
   atom block.**  Upstream sizes the atom block by what the response holds
   per set -- AO densities, `vj`, `vk`: `6 nao^2` words unrestricted over 0.3
   of the free memory, `3 nao^2` over 0.5 restricted.  That budget is kept,
   but applied to each **response call** (`_Budget`, `_grouped`): the call is
   split into as many groups of sets as the free memory at that call allows,
   and the Krylov iteration carries every atom in one block, whose own
   arrays are MO-sized (`~10 nmo nocc` words a set; the block is split only
   when those do not fit).  Upstream's tolerance is `conv_tol_cpscf` times
   the block's atoms, so one block is also the full card's tolerance.  At
   40 GB taxol⁺'s calls ran in two or three groups (15 group calls for 6
   response calls); the groups cost `cache_xc_kernel` again (3.5 s
   against 2.4 s) and the response's per-call setup, 9 s over the CPHF.  `FASTCPHF_BLOCK=upstream` is
   upstream's blocks; `FASTCPHF_FXSETS=<n>` caps a group (test-only).
2. **The subspace as blocks** (`krylov`, `_Subspace`): each cycle's block of
   `xs` and `ax` is stored as it comes; the projection is two GEMMs per
   stored block, in upstream's order (block modified Gram-Schmidt), run twice
   (classical Gram-Schmidt with reorthogonalisation -- once leaves 3.5e-11
   between this and upstream's order on glycine⁺, twice 9e-12;
   `FASTCPHF_REORTH=0`); `h = xs ax^T` is accumulated a block row and column
   per cycle; the right-hand side's projections and the solution are a GEMM
   per block; the updates are in place (`cublas.gemm(beta=1)`) rather than
   `x1 -= w.dot(xs)`, which would be an x1-sized `cudaMalloc`/`cudaFree` per
   block (work list §5.8).  `FASTCPHF_KRYLOV=upstream` is upstream's solver.
3. **The subspace on the host when the device is short**: a block stays on
   the device only if what is left still takes the next response call in one
   group, otherwise it goes to page-locked host memory, which the
   projections read back once per cycle.  `FASTCPHF_STORE=auto|device|host`.

A matrix operator, a callback or another inner product is upstream's
`krylov`; `FASTCPHF_OFF=1` is upstream's `solve_mo1`.

## 3. Measurements

One A100-SXM4-80GB of a whole node; the 40 GB card is `--ballast 40` (44.6 GB
free before anything).  `solve_mo1` alone on the cached SCF and `h1mo`, one
call per process (jobs `el_w44a2`, `el_w44c2`, `el_w44d2`, `el_w44l`,
`el_w44b2`).  The table at the top.  Per response call the rate is the same
in every column (0.745-0.75 s a set unrestricted), so the difference is the
solver and, at 40 GB, the blocking.

### 3.1 The whole Hessian

`benchmarks/df_hess_anatomy.py --patch all --rounds 1` (`--ballast 40` for
the 40 GB card), taxol⁺ UKS B3LYP/def2-SVP, jobs `el_w44m`/`el_w44n`; the
"before" columns are W40's (`el_w3941ph`), s:

| | 40 GB before | **40 GB after** | 80 GB before | **80 GB after** |
|---|---|---|---|---|
| **whole Hessian** | 1760.7 | **1348.6** | 1482.3 | **1324.5** |
| partial_hess_elec (.. ip2) | 48.9 (39.4) | 46.4 (37.9) | 43.4 (34.2) | 40.1 (31.3) |
| make_h1 (.. h1) | 55.1 (34.7) | 54.8 (34.5) | 49.3 (29.0) | 49.2 (28.9) |
| solve_mo1 (CPHF) | 1655.0 | **1246.7** | 1387.8 | **1233.5** |
| .. `_get_jk` | 1357.2 (15 calls) | 1083.6 (15 group calls, 6 response calls) | 1089.0 (6) | 1083.2 (9 group calls) |
| .. XC response | 155.1 | 124.2 | 120.1 | 119.0 |
| trace | 108.9223988 | 108.9229005 | 108.9229005 | 108.9229005 |

**The 40 GB Hessian is 1.018x of the full card's** (the entry's bar was
1.05x), and **1.31x faster than it was**; the full card's is 1.12x faster.
The two new Hessians agree to **5.2e-10** Eh/bohr² (the old 40 and 80 GB
ones differed by 1.5e-4, the difference of their atom blocks' Krylov
tolerances, W40 §4.1), and the new full-card one agrees with the old to
3.3e-11: the same equations, solved in one block to one tolerance on both
cards.  What is left of the 1.018x is the ip2 and h1 terms (W42) and the
groups' repeated setup (§2, item 1).

## 4. Correctness

`tests/test_cphf.py` against upstream's `solve_mo1` on the same SCF and
`h1mo`, `mo1` and `mo_e1` relative to the largest element: restricted and
unrestricted, density-fitted and direct, ethanol and the open-shell Cu(II)
complex CuCl₂, in six columns (default, groups, host subspace, host with
groups, one Gram-Schmidt pass, upstream's solver under this module's
blocks): **1e-13–1.4e-11** density-fitted; upstream's solver under these
blocks 2e-15–7e-14.  The direct path with its response in groups is 5e-10–
3.4e-9: its exchange screens on the largest density of the call
(`scf/jk.py`, `condense('absmax', dms)`), so a call on fewer sets keeps
different integrals -- a difference upstream's own atom blocks make as well,
at their Krylov tolerance.  Whole Hessians, patched against
`FASTCPHF_OFF=1`: 2.8e-12 (ethanol RKS), 1.9e-12 Eh/bohr² (ethanol⁺ UKS).
Against upstream in the same process on larger systems: 1.1e-12 relative
(penicillin V⁺, 42 atoms), 1.9e-14 (taxol RKS, 113 atoms).

## 4b. Item 4: the response builds screen at the solve's precision (catalogue B12)

The direct J and K builds screen a shell quartet against `direct_scf_tol`
(1e-13), the SCF's precision, also inside a CPHF converged to
`conv_tol_cpscf`.  While `solve_mo1` runs, `fastcphf` now raises the
threshold of every `_VHFOpt` the response reaches to
`max(direct_scf_tol, min(1e-3 conv_tol_cpscf, 1e-9))`.
`benchmarks/cphf_screen_probe.py` measured the rule on two organic
molecules, taxol and two open-shell transition-metal complexes
(`docs/KERNEL_TARGETS_1.8.1.md` §13.11): at a 1e-6 solve the threshold 1e-9
moves the Hessian by 2.5e-8–1.9e-7 Eh/bohr², against 2.4e-5–3.2e-4 that the
solve's own tolerance moves it, and 1e-7 breaks the iteration.  The Krylov
vectors are scaled to their residuals, so an absolute threshold is already a
relative one that loosens as the solve converges.

| `conv_tol_cpscf` 1e-6, `--patch all` | `solve_mo1`, 1e-13 | 1e-9 |
|---|---|---|
| penicillin V | 30.09 s | 21.03 s |
| taxol | 860.9 s | 431.2 s |
| Ni(II) dithiolene⁻, UKS | 33.52 s | 26.86 s |

`FASTCPHF_SCREEN=off` (and `FASTCPHF_OFF=1`) keep `direct_scf_tol`;
`tests/test_cphf.py` checks the screened direct Hessians against it.

## 5. What the probe taught

GPU4PySCF installs an allocator that sends every allocation over 100 MB
straight to `cudaMalloc` (`set_conditional_mempool_malloc`).  A benchmark
that counts memory by `cupy.cuda.set_allocator(pool.malloc)` does not wrap
that allocator, it *replaces* it: every large array then stays in CuPy's
pool, which has a limit, and the first taxol⁺ run of this probe ran out of
memory in `make_h1` where the Hessian does not, while the smaller runs were
timed on an allocator no Hessian uses.  Wrap `cp.cuda.get_allocator()`, and
read `cudaMemGetInfo` for the peak.

## 6. Tools

```bash
python benchmarks/df_cphf_probe.py --xyz benchmarks/molecules/taxol.xyz \
    --charge 1 --spin 1 --basis def2-svp --xc b3lyp --patch all \
    --cache $SCRATCH/cphf --variants stock,ours [--ballast 40] \
    [--env 'host:FASTCPHF_STORE=host,grp:FASTCPHF_FXSETS=40']
python tests/test_cphf.py [--no-hessian]
```
