# fastj — a faster Coulomb-matrix (J) build for GPU4PySCF

On an A100-40GB, B3LYP/6-31G\* on these clusters spends about 30 % of the SCF
inside GPU4PySCF's McMurchie–Davidson J engine (`md_j_*`), once the exchange
build has been sped up by `fastk`.  `fastj` rewrites those kernels and leaves
the rest of the J engine — the Hermite density transform, the screening, the
tile decomposition, the symmetry factors — exactly as it is.

```python
import sys; sys.path.insert(0, '.../side_test/HF/src')
import fastj                    # patches gpu4pyscf.scf.j_engine._VHFOpt.get_j
```

## Results (A100-SXM4-40GB, `direct_scf_tol=1e-13`)

J kernels alone, at a representative SCF density (`bench/perclass_j.py`):

| class (lij\_lkl) | GPU4PySCF | fastj | speedup | quartets |
|---|---|---|---|---|
| 1\_0 | 353.96 ms | 177.59 ms | **1.99×** | 2.1e10 |
| 2\_0 | 197.94 ms | 104.03 ms | **1.90×** | 8.7e9 |
| 1\_1 | 232.09 ms | 123.14 ms | **1.89×** | 1.7e10 |
| 3\_1 | 112.23 ms |  63.51 ms | 1.77× | 1.5e9 |
| 4\_0 |  14.37 ms |   8.48 ms | 1.69× | 2.2e8 |
| 3\_0 |  55.00 ms |  33.86 ms | 1.62× | 1.9e9 |
| 2\_1 | 257.08 ms | 170.10 ms | 1.51× | 7.2e9 |
| 0\_0 | 102.15 ms |  70.02 ms | 1.46× | 2.5e10 |
| 4\_1 |  20.40 ms |  15.70 ms | 1.30× | 1.8e8 |
| 3\_2 |  83.67 ms |  64.85 ms | 1.29× | 6.3e8 |
| 2\_2 |  92.18 ms |  74.63 ms | 1.24× | 2.2e9 |
| 3\_3, 4\_2, 4\_3, 4\_4 | 282.12 ms | (fall through) | — | 2.3e8 |
| **total** | **1803 ms** | **1188 ms** | **1.52×** | |

The same measurement on `HcgC_cluster` (4170 AOs, 1.8× the size) gives
**7346 ms → 4835 ms = 1.52×**, class for class within a few per cent of the
table above -- the speedups come from per-class properties, not from the size
of the molecule.

End to end, `get_j` including the host-side Hermite transform:
**23.40 s → 16.42 s** over the 18 iterations of B3LYP/6-31G\* on
`PfPMT_cluster` (**1.43×**).  `FASTJ_TIME=1` splits that 16.3 s into 14.6 s of
kernels, 0.50 s of host transform plus upload and 1.01 s of download plus the
reverse transform -- so 91 % of the J build is now kernel time, and the
remaining CPU work is what a GPU port of `Et_dot_dm`/`jengine_dot_Et` would
address.  The J matrix agrees with GPU4PySCF's to
4e-16 relative (`bench/check_j_small.py`), and the converged energy to 1e-10 Eh.

## What the kernels change

The integral arithmetic is *not* changed: `gen_j_kernels.py` lifts the Hermite
(`Rt`) recurrences and the density contraction verbatim out of GPU4PySCF's
generated `gvhf-md/unrolled_md_j.cu`.  What changes is the scaffolding.

1. **No double-precision division or sqrt in the innermost loop.**  GPU4PySCF
   evaluates `fac/(aij*akl*sqrt(aij+akl))` and `aij*akl/(aij+akl)` once per
   shell-pair quartet — two DP divisions and a DP sqrt, each about 20
   instructions on sm_80, for a class such as `(ss|ss)` whose entire useful
   work is one Boys function and one multiply-add.  `1/aij` is uniform over a
   bra tile and `1/akl` over a ket tile, so both are cached in the same shared
   arrays that already hold the Gaussian product centres; one `rsqrt(aij+akl)`
   then supplies both `1/(aij+akl)` and `1/sqrt(aij+akl)`, and every division
   becomes a multiply.
2. **The Boys values live in registers.**  GPU4PySCF stages `gamma_inc`
   through the dynamic shared buffer at stride 256, so each of the 1–56 reads
   per quartet is an LDS.  They are per-thread values, so a register array of
   `order+1` doubles does the same job, and the shared-memory request drops by
   `256*(order+1)` doubles — 6 KB for the low classes and up to 12 KB for the
   high ones, which is what lets the wide classes fit more blocks per SM.
3. **The bra tile's data is read once, not once per quartet.**  The product
   centre, `aij` and `1/aij` do not change inside the ket loop, but GPU4PySCF
   re-reads all four from shared memory on every quartet.  Reading them into
   registers once per bra tile also lets the barrier that guarded them move up
   out of the ket loop, so the ket loop carries no `__syncthreads()` at all.
   This is worth 5–8 % on top of item 1 for every class.
4. **`omega == 0` is compiled in.**  The Coulomb J build never uses
   range separation, but `boys_fn` tests `jk.omega` at run time and carries two
   dead branches, each with its own `sqrt` and scaling loop.
5. **Occupancy is set per angular-momentum class.**  GPU4PySCF compiles the
   whole family with `__maxnreg__(128)`, which pins every class at 2 blocks per
   SM (25 % occupancy).  How many registers a class actually needs is set by
   its `nf3ij` accumulators and `Rt` temporaries — a property of the class, not
   of the molecule — so `gen_j_kernels.py` gives each its own
   `__launch_bounds__`.  `(0,0)`–`(2,0)` run at 64 registers and 4 blocks per
   SM; `(3,2)` and `(4,1)`, which carry 20 and 35 accumulators, need 256 and
   spill badly at anything more than one block per SM.
6. **The incomplete gamma function.**  One `rsqrt` replaces the `sqrt` and the
   two divisions of the large-argument branch; the downward recursion
   multiplies by tabulated reciprocals of half-integers instead of dividing;
   and beyond `t = 36`, where `exp(-t)` is below the double-precision epsilon
   relative to the leading term, the `erf` and the `exp` are skipped
   altogether.

## Is any of this size-dependent?

| ingredient | depends on | handling |
|---|---|---|
| register/occupancy table | GPU + angular-momentum class | fixed per class; a molecule cannot change it. Re-tune with `bench/perclass_j.py --variants` for a different GPU. |
| tile sizes, grid width | GPU4PySCF's own scheme | taken unchanged from `md_j_unrolled`; the grid is `ceil(npairs/tile)` in each direction, as before. |
| shared-memory request | angular-momentum class | computed per class by the generator from the parsed cache layout; the two reciprocal caches are static arrays sized by the class's ket tile. |
| implemented classes | basis angular momentum, **not** system size | `(lij,lkl)` beyond what GPU4PySCF unrolls falls through to its general `md_j_1dm_kernel`. |
| `omega`, `n_dm` | what the caller asked for | checked in `fastj.get_j`; anything else falls through. |

The two clusters differ by 1.8× in AO count and get the same per-class
speedups, which is the empirical form of the same statement.

## Coverage and what is left

12 of the 16 `(lij|lkl)` classes a 6-31G\* organic system generates are
implemented — the ones GPU4PySCF unrolls.  `(3,3)`, `(4,2)`, `(4,3)` and
`(4,4)` go through GPU4PySCF's general `md_j_1dm_kernel`; they are 16 % of the
J kernel time and the same rewrite applies to that kernel, which stages `Rt`
itself through shared memory.

## Files

| file | role |
|---|---|
| `fastj.py` | patches `j_engine._VHFOpt.get_j`; dispatch, fall-through rules |
| `fastj_prologue.cu` | register Boys function, reciprocal tables, the kernel argument list |
| `gen_j_kernels.py` | lifts GPU4PySCF's unrolled J kernels into the new scaffolding |
| `fastj_generated.cu` | generated kernels (regenerate with `build_j_kernels.sh`) |
| `fastj_launch.json` | per-class launch bounds, tiles and shared-memory sizes |
| `fastj_variants.cu` | the same kernels at 1/2/3/4 blocks per SM, for the sweep (`FASTJ_SRC=fastj_variants.cu`) |
| `fastj_nohoist.cu` | the kernels without item 3, kept as the ablation (`FASTJ_SRC=fastj_nohoist.cu`): 1.44× instead of 1.52× |
| `../bench/check_j_small.py` | correctness against GPU4PySCF on small molecules |
| `../bench/perclass_j.py` | per-class timing; `--variants` sweeps the launch bounds |

Regenerate against a different GPU4PySCF checkout with

```bash
./build_j_kernels.sh /path/to/gpu4pyscf/lib/gvhf-md/unrolled_md_j.cu
```
