# Electrolites

Drop-in GPU kernel replacements for [GPU4PySCF](https://github.com/pyscf/gpu4pyscf):
the exchange and Coulomb builds, the XC build, the VV10 non-local correlation,
and the analytic nuclear gradient.

Importing a module monkey-patches one GPU4PySCF entry point in place.  Anything
a module does not cover falls through to GPU4PySCF's own code, so a patched run
computes the same quantity as an unpatched one — measured at **8.2e-12 Eh** on
the energy and **2.9e-12** per gradient component over the suites in `tests/`.

On one A100-SXM4-40GB, GPU4PySCF 1.8.1, B3LYP-D3(BJ)/6-31G\*, level-3 grids,
`OMP_NUM_THREADS=32`:

| | GPU4PySCF 1.8.1 | + Electrolites | |
|---|---|---|---|
| 284-atom cluster, SCF | 127.2 s | **55.1 s** | **2.31x** |
| 284-atom cluster, gradient | 19.4 s | **12.5 s** | **1.55x** |
| 545-atom cluster, one SCF cycle's work (`minao`) | 9.62 s | **3.22 s** | **2.98x** |
| 545-atom cluster, one SCF cycle's work (converged) | 29.8 s | **14.1 s** | **2.11x** |

A build timed on the `minao` initial guess is not the build an SCF runs -- the
guess is far sparser, so it screens harder.  Both rows are given for the
545-atom cluster because the difference is a factor of 1.4 in the ratio; the
284-atom rows are wall clocks and have no such ambiguity.

`docs/UPGRADE_1.8.1.md` has the full tables, the build-by-build breakdown and
the correctness suites.

> **Correction.**  That document says "about a third of that SCF wall clock is
> DIIS, the fp64 diagonalisation and grid construction, which nothing here
> touches".  It is not: instrumenting the SCF itself
> (`benchmarks/scf_anatomy.py`) puts those three at **1.9 %** of it, and
> `get_veff` at 97.8 %.  The third came from comparing a *fixed-density*
> `get_veff` speedup against a *converged* wall clock, which are not the same
> quantity.  `docs/ROUND2_1.8.1.md` §2 has the profile.

## Install

```bash
pip install gpu4pyscf-cuda12x      # or gpu4pyscf-cuda11x; 1.7.0 or 1.8.x
pip install git+https://github.com/Benzoin96485/Electrolites.git
```

Nothing is compiled at install time: every kernel is built at run time by
NVRTC through CuPy, which GPU4PySCF already depends on, and cached by CuPy on
the source hash.  **Only the angular-momentum classes the basis can reach are
compiled**, so a 6-31G\* job's first run pays about 36 s of JIT rather than
98 s; later runs in any process reuse the cache.  A basis with f functions
reaches every class and pays for all of them.
`ELECTROLITES_FULL_MODULE=1` compiles everything, which is the ablation.

One optional library is the exception — a faster copy of GPU4PySCF's *general*
exchange kernel, which needs relocatable device code and so cannot go through
NVRTC.  It is built with `nvcc` on first use and cached in
`~/.cache/electrolites/sm_<arch>/`.  If `nvcc` is not installed, Electrolites
says so once and those angular-momentum classes fall back to GPU4PySCF —
slower, still correct.

## Use

```python
import electrolites
electrolites.patch_all()

from pyscf import gto
from gpu4pyscf import dft

mol = gto.M(atom='h2o.xyz', basis='def2-tzvpp')
mf = dft.RKS(mol, xc='wb97m-v')
mf.conv_tol = 1e-9
e = mf.kernel()
de = mf.nuc_grad_method().kernel()
```

`patch_all()` refuses to patch if the installed GPU4PySCF does not expose an
entry point it replaces, naming which one — it never patches silently.  To see
that check without patching:

```python
electrolites.check_targets()      # {'fastk': None, ...}; a string means missing
electrolites.gpu4pyscf_version()
```

Patch a subset — this is how every ablation in `docs/` was measured:

```python
electrolites.patch('fastk', 'fastxc')          # or patch_all(only=[...])
electrolites.patched()                         # what is active in this process
```

There is no unpatch.  The patches are global, so the benchmarks compare
separate processes; that is the only honest way to A/B them.

## What each module replaces

| module | replaces | covers |
|---|---|---|
| `fastk` | `scf.jk._VHFOpt.get_k` | every angular-momentum class an spdf basis reaches — the 25 GPU4PySCF unrolls and the 40 it does not — for the full-range and the long-range (erf) operator. On 1.8.x it also serves the fused `lr*erf + sr*erfc` request, by way of `erf + erfc == 1` |
| `fastj` | `scf.j_engine._VHFOpt.get_j` | **every `(lij\|lkl)` class an spdf basis reaches** -- the 12 GPU4PySCF unrolls, lifted, and the 17 it does not, written from the class |
| `fastxc` | `dft.numint.NumInt.nr_rks` | LDA and GGA, so hybrids too |
| `fastnlc` | `dft.numint._vv10nlc` | the VV10 / VV10-family `O(n_grid^2)` double sum |
| `fastrsh` | `dft.rks.RKS.get_veff` | range-separated hybrids on GPU4PySCF **1.7.x only**: the two exchange builds fused into one. 1.8.0 does that fusion itself, so this module stands down there |
| `fastxcnlc` | `dft.rks.RKS.get_veff` | the XC potential and the VV10 non-local correlation from **one** sweep of the grid instead of two, on 1.8.x (on 1.7.x `fastrsh` drives it). LDA falls back; GGA and meta-GGA are covered, so wB97M-V is |
| `fastejk` | `grad.rhf._jk_energy_per_atom` | the per-atom derivative of `j*J - k*K`, every class an spdf basis reaches, full-range and long-range |
| `fastgrad` | `grad.rks.Gradients.energy_ee` | range-separated-hybrid gradients: the long-range operator in place of GPU4PySCF's short-range one |
| `fastxcgrad` | `grad.rks.get_exc`, `get_nlc_exc` | the XC (LDA, GGA **and** meta-GGA) and VV10 grid contractions of the gradient |
| `fastgradh` | `grad.rhf.GradientsBase.get_hcore` | the nuclear-attraction part of the one-electron derivative, on the GPU |

**Limits, all of which fall through rather than fail.** `fastk`, `fastj`,
`fastxc` and the four gradient modules are spin-restricted closed-shell only:
an unrestricted calculation reaches them with two density matrices and is
handed back. `fastnlc` covers UKS, because `nr_nlc_vxc` sums the two densities
before the double sum. `fastxc` hands **meta-GGA** back in the SCF, while
`fastxcgrad` covers it in the gradient. Shells with `l > 4` go through
GPU4PySCF's CPU path. Density fitting is untouched. Single GPU only —
GPU4PySCF 1.8.x runs these builds through `multi_gpu.run`, and every module
here asserts one device.

## Tests

```bash
python tests/test_full.py --patch all      # 15 cases: energy and gradient,
                                           # patched process vs unpatched
python tests/test_full.py --patch ''       # ... the unpatched half
python tests/test_grad.py                  # 23 cases: gradient at a fixed
                                           # density, both codes in one process
```

They need a GPU and take a few minutes. The cases span six basis sets to
cc-pVQZ, twelve functionals including three meta-GGAs, `omega < 0`, third-row
elements, a cation and an open-shell radical.

## Benchmarks

`benchmarks/scf_anatomy.py` says where an SCF wall clock goes, method by
method, with a synchronisation point on both sides of each -- use it before
concluding that anything outside J, K and XC matters.
`benchmarks/perclass_j.py` times the Coulomb build one angular-momentum class
at a time and checks the J matrix against GPU4PySCF's;
`benchmarks/sweep_j_high.py` searches the launch configuration of the written
classes; `benchmarks/compile_time.py` reports the first-run NVRTC cost with
and without the module split; `benchmarks/aoscreen_probe.py` measures how much
a density-aware AO screen could remove from the XC build.

`benchmarks/cluster_bench.py` times an SCF and a gradient end to end;
`benchmarks/fixed_density.py` times each build separately on one fixed density,
which is what to use on a large or ill-conditioned system where a converged
wall clock measures the SCF's convergence path rather than the kernels:

```bash
python benchmarks/fixed_density.py --xyz benchmarks/molecules/HcgC.xyz --charge -4
python benchmarks/fixed_density.py --xyz benchmarks/molecules/HcgC.xyz --charge -4 --patch all
```

## Regenerating kernels

`codegen/` holds the generators. `gen_khigh.py`, `gen_ejk.py` and
`gen_j_high.py` write a kernel from the angular-momentum class alone;
`gen_kernels.py`, `gen_k2_kernels.py` and `gen_j_kernels.py` lift the integral
arithmetic out of a GPU4PySCF source file, so they take its path:

```bash
python codegen/gen_khigh.py --table electrolites/kernels/fastkhigh_launch.json \
    > electrolites/kernels/fastkhigh_generated.cu
python codegen/gen_j_high.py > electrolites/kernels/fastjhigh_generated.cu
codegen/build_kernels.sh /path/to/gpu4pyscf/lib/gvhf-rys/unrolled_rys_k.cu
```

`gen_j_high.py` needs no GPU4PySCF source at all -- it writes each
McMurchie-Davidson class from `(lij, lkl)` -- and re-tuning it for another
card is `benchmarks/sweep_j_high.py`, whose winners go into that file's `CFG`.

The launch tables next to the generated sources (`*_launch.json`) are tuned per
angular-momentum class and were measured on an A100; on another card they are a
starting point, not an optimum.

## Environment switches

| variable | effect |
|---|---|
| `ELECTROLITES_NVCC`, `ELECTROLITES_ARCH` | compiler and compute capability for the one `nvcc`-built library |
| `ELECTROLITES_LIBMYKG` | use a prebuilt copy of it instead of building |
| `ELECTROLITES_CACHE` | where to cache the build (default `~/.cache/electrolites`) |
| `FASTK_NO_GENERAL`, `FASTK_NO_K2`, `FASTK_NO_HIGH`, `FASTK_NO_OMEGA`, `FASTK_NO_FUSED` | hand the corresponding class group back to GPU4PySCF (ablations) |
| `FASTNLC_MODE` | `mixed` (default) or `fp64` for the VV10 double sum |
| `FASTJ_TIME`, `FASTXCGRAD_MODE` | per-call timing split; XC-gradient block strategy |

## License

Apache-2.0. Substantial parts of the CUDA are derived from GPU4PySCF, also
Apache-2.0 — see `NOTICE` for exactly which files and how.
