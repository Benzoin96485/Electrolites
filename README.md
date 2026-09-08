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

and, for the analytic **Hessian** this round added (B3LYP/def2-SVP, level-3
grids, `conv_tol_cpscf = 1e-6`):

| | GPU4PySCF 1.8.1 | + Electrolites | |
|---|---|---|---|
| 42-atom Hessian (penicillin V), whole run | 84.3 s | **53.9 s** | **1.56x** |
| ↳ its two XC terms, at fixed density | 15.5 s | **2.11 s** | **7.34x** |
| 113-atom Hessian (taxol), its two XC terms | 117.8 s | **22.2 s** | **5.30x** |

and, for the **response properties** this round added -- whole runs, 42-atom
penicillin V/def2-SVP, level-3 grids, five roots for TDA:

| functional | property | GPU4PySCF 1.8.1 | + Electrolites | |
|---|---|---|---|---|
| PBE | polarizability (CPKS, 3 RHS) | 9.49 s | **7.12 s** | **1.33x** |
| | TDA | 9.16 s | **5.13 s** | **1.79x** |
| B3LYP | polarizability | 16.33 s | **14.26 s** | **1.15x** |
| | TDA | 22.06 s | **17.74 s** | **1.24x** |
| M06-2X | polarizability | 22.53 s | **15.59 s** | **1.45x** |
| | TDA | 32.30 s | **21.75 s** | **1.49x** |
| wB97M-V | polarizability | 154.67 s | **111.26 s** | **1.39x** |
| | TDA | 49.02 s | **38.10 s** | **1.29x** |

The XC response kernel itself is 2.1x-3.7x inside those runs, and 4.65x at
fixed density and `nset` = 5 (1.057 s -> 0.479 s, i.e. 2.21x, on 113-atom
taxol -- the gain falls with system size and `docs/README_fastxcfxc.md` §6
says why).  On the wB97M-V polarizability the VV10 response term carries the
row: 124.7 s -> 87.2 s, 1.43x, and 1.44x-1.49x per call at three sizes --
that one *is* size-independent, and `docs/README_fastfnlc.md` §6 says why the
two modules scale opposite ways.

Every stock/patched pair in that table is measured back-to-back in one script,
because the node is shared: the same stock B3LYP TDA run measured 22.2 s twice
and 43.3 s once, on a node whose load average had gone from 5 to 70.  Only the
ratio inside a pair is meaningful; `docs/RESPONSE_1.8.1.md` §4b has the
method.  Measured that way, the fused J and K build of a **hybrid** does not
move at all under `patch_all()` -- 11.67 -> 11.70 s, 13.56 -> 13.64 s,
19.64 -> 20.07 s -- which is the direct evidence that neither `fastj` nor
`fastk` is on that path.

Two things the profile behind those numbers says, both in
`docs/RESPONSE_1.8.1.md`.  **The split between the grid half and the
two-electron half is a property of the functional**: the XC response kernel is
46-48 % of a PBE response property and 47-48 % of an M06-2X one but only
25-28 % of B3LYP's, and on wB97M-V neither is the target -- **80 %** of that
polarizability is the VV10 response term, which no catalogue entry names.
And **the hybrid's two-electron half is a third code path that no module here
reaches**: the response solve asks `mf.get_jk` for J and K from one *fused*
Rys kernel (`scf/jk.py:401 _VHFOpt.get_jk`) where the SCF calls `get_j` and
`get_k` separately, so `fastj` and `fastk` are not on it and 61-82 % of a
B3LYP response property is stock GPU4PySCF.  The ceiling for closing that,
measured on these kernels, is **2.30x** at 42 atoms and 1.60x at 113.

The Hessian rows are dominated by something none of these modules touches: the
exchange build inside the CPHF solve is 37 % of the 42-atom run after this
round and 54 % of the 113-atom one, and `fastk` hands every multi-density call
back to GPU4PySCF.  `docs/HESSIAN_1.8.1.md` is the profile that says so.

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
| `fastj` | `scf.j_engine._VHFOpt.get_j` | **every `(lij\|lkl)` class an spdf basis reaches** -- the 12 GPU4PySCF unrolls, lifted, and the 17 it does not, written from the class.  Also the **multi-density** build that CPHF, TDDFT and polarizability ask for, four densities to a pass, for every class through `(4,4)` |
| `fastxc` | `dft.numint.NumInt.nr_rks` | LDA and GGA, so hybrids too |
| `fastnlc` | `dft.numint._vv10nlc` | the VV10 / VV10-family `O(n_grid^2)` double sum |
| `fastrsh` | `dft.rks.RKS.get_veff` | range-separated hybrids on GPU4PySCF **1.7.x only**: the two exchange builds fused into one. 1.8.0 does that fusion itself, so this module stands down there |
| `fastxcnlc` | `dft.rks.RKS.get_veff` | the XC potential and the VV10 non-local correlation from **one** sweep of the grid instead of two, on 1.8.x (on 1.7.x `fastrsh` drives it). LDA falls back; GGA and meta-GGA are covered, so wB97M-V is |
| `fast1e` | `gto.int3c1e.get_int3c1e{,_density_contracted,_charge_contracted}` | the three-centre one-electron integrals -- **PCM, QM/MM, ESP, CHELPG** -- every angular-momentum class up to g, contracted with a density, with charges, or neither |
| `fastejk` | `grad.rhf._jk_energy_per_atom` | the per-atom derivative of `j*J - k*K`, every class an spdf basis reaches, full-range and long-range |
| `fastgrad` | `grad.rks.Gradients.energy_ee` | range-separated-hybrid gradients: the long-range operator in place of GPU4PySCF's short-range one |
| `fastxcgrad` | `grad.rks.get_exc`, `get_nlc_exc` | the XC (LDA, GGA **and** meta-GGA) and VV10 grid contractions of the gradient |
| `fastgradh` | `grad.rhf.GradientsBase.get_hcore` | the nuclear-attraction part of the one-electron derivative, on the GPU |
| `fastxchess` | `hessian.rks._get_vxc_deriv1`, `_get_vxc_deriv2` | the two XC terms of an analytic **Hessian**: the same arithmetic in the masked AO space, with the atoms that have no basis function on a grid block skipped -- their contribution is identically zero -- and the atom dimension batched instead of looped in Python.  GGA |
| `fastxcfxc` | `dft.numint.nr_rks_fxc` (the module attribute **and** `NumInt.nr_rks_fxc`) | the XC response kernel -- the grid half of **every response property**: CPKS/polarizability, TDA/TDDFT, NMR, IR/Raman, C6.  One GEMM per grid block instead of one per right-hand side, three custom kernels in place of ten GPU4PySCF launches per right-hand side, and one symmetrisation per call instead of an extra matrix product per block.  LDA and GGA |
| `fastfnlc` | `hessian.rks.nr_rks_fnlc_mo` | the **VV10 non-local-correlation response**, which is 80 % of a wB97M-V polarizability: the ground-state half of it -- including an `O(n_grid^2)` double sum -- computed once per coupled-perturbed solve instead of once per iteration, and the potential contraction batched over the perturbed densities.  Reaches CPKS, NMR, C6 and the analytic Hessian; TDDFT excludes the term itself |

**Limits, all of which fall through rather than fail.** `fastk`, `fastxc` and
the four gradient modules are spin-restricted closed-shell only: an
unrestricted calculation reaches them with two density matrices and is handed
back.  `fastj` takes any number of densities, so it also covers the Coulomb
build inside a CPHF or Davidson solve; what it declines there is the classes
past `(4,4)` and the long-range operator. `fastnlc` covers UKS, because `nr_nlc_vxc` sums the two densities
before the double sum. `fastxc` hands **meta-GGA** back in the SCF, while
`fastxcgrad` covers it in the gradient. `fastxchess` covers GGA and declines
LDA, meta-GGA, the grid-response path and the VV10 Hessian.  `fastxcfxc` covers
LDA, GGA and meta-GGA for a plain-density-matrix response and declines the
`mo1` path (which is the Hessian's own CPHF solve) and UKS.  `fastfnlc`
covers a closed-shell NLC response and declines UKS; its `return_in_mo=True`
path (the Hessian's) gets the cache but keeps GPU4PySCF's potential
contraction.  Two of those have
a reason worth reading before extending it: GPU4PySCF's LDA branch of the same
function uses different factor conventions term by term (they agree with the
GGA branch's -- `docs/README_fastxchess.md` §5 has the algebra), and its
meta-GGA branch is not reproducible run to run. Shells with `l > 4` go through
GPU4PySCF's CPU path. Density fitting is untouched. Single GPU only —
GPU4PySCF 1.8.x runs these builds through `multi_gpu.run`, and every module
here asserts one device.  `fast1e` is the exception: GPU4PySCF does not
distribute the `int3c1e` path, and the module keys its compiled kernels and
its Rys tables on the current device, so it stays correct in a process where
the J and K builds are spread over several.

## Tests

```bash
python tests/test_full.py --patch all      # 15 cases: energy and gradient,
                                           # patched process vs unpatched
python tests/test_full.py --patch ''       # ... the unpatched half
python tests/test_grad.py                  # 23 cases: gradient at a fixed
                                           # density, both codes in one process
python tests/test_j_ndm.py                 # 162 cases: the multi-density
                                           # Coulomb build against GPU4PySCF
python tests/test_xchess.py                # 11 cases: the two XC Hessian
                                           # terms, and the whole analytic
                                           # Hessian, patched vs unpatched
python tests/test_xchess_fd.py             # the analytic Hessian against
                                           # finite differences of the gradient
python tests/test_fxc.py                   # 15 cases: the XC response kernel
                                           # against GPU4PySCF at both hermi
                                           # values, plus a polarizability
                                           # tensor and a TDA spectrum
python tests/test_fnlc.py                  # 8 cases: the VV10 response term,
                                           # bit-identical on the cached path
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
`benchmarks/ndm_j.py` says how the Coulomb build's cost grows with the number
of density matrices, fits the per-pass and per-density parts of it, and A/Bs
the density-block sizes -- use it before assuming that a cap GPU4PySCF pays for
is still worth removing once the kernel under it has been rewritten
(`docs/ROUND3_B4A.md`);
`benchmarks/sweep_j_high.py` searches the launch configuration of the written
classes; `benchmarks/compile_time.py` reports the first-run NVRTC cost with
and without the module split; `benchmarks/aoscreen_probe.py` measures how much
a density-aware AO screen could remove from the XC build.

`benchmarks/hessian_anatomy.py` does the same for an analytic Hessian, stage
by stage -- use it before taking a Hessian target from
`instructions/06`, whose Hessian profile is a 1.7.0 one and whose shares move
by a factor of six between 42 and 113 atoms (`docs/HESSIAN_1.8.1.md`).
`benchmarks/vxc_hess_probe.py` and `benchmarks/xchess_locality.py` say where
the XC Hessian's time goes and how much of its atom loop is arithmetic on
exact zeros; `benchmarks/xchess_bench.py` A/Bs the two terms on one converged
density.

`benchmarks/response_anatomy.py` does the same for a polarizability and a TDA
run -- use it before taking a response-property target from `instructions/06`,
whose two profiles are 1.7.0 ones and whose `eval_ao` share (entry D8, 16.3 %)
is 2.2 % on 1.8.1.  `benchmarks/response_jk_probe.py` answers whether the
response path should split that fused build into `fastj` + `fastk` and
measures the ceiling for rewriting it -- the answer flips sign between 42 and
113 atoms, which is the argument for a fused rewrite rather than a
dispatch; `benchmarks/fnlc_probe.py` and `fnlc_bench.py` do the same for the
VV10 response term; `benchmarks/fxc_bench.py` sweeps the XC
response kernel over the number of right-hand sides, all three arrangements,
with the correctness check inline.

`benchmarks/cluster_bench.py` times an SCF and a gradient end to end;
`benchmarks/fixed_density.py` times each build separately on one fixed density,
which is what to use on a large or ill-conditioned system where a converged
wall clock measures the SCF's convergence path rather than the kernels:

```bash
python benchmarks/fixed_density.py --xyz benchmarks/molecules/HcgC.xyz --charge -4
python benchmarks/fixed_density.py --xyz benchmarks/molecules/HcgC.xyz --charge -4 --patch all
```

## Regenerating kernels

There are two kinds of generator here, and the difference decides what is in
the repository.

`electrolites/codegen/` holds the three that write a kernel from the
angular-momentum class alone, taking nothing from GPU4PySCF: `gen_khigh.py`,
`gen_ejk.py` and `gen_j_high.py` -- the last of which writes two families, the
single-density kernels and, with `--ndm`, the multi-density ones.  **Their
output is not tracked.**  It is 15 MB of straight-line CUDA, it regenerates in
0.8 s byte for byte, and
`electrolites/_gen.py` produces it on first use and caches it under
`~/.cache/electrolites/generated/`, keyed by a hash of the generator and its
launch table so that editing either regenerates.  `ELECTROLITES_GENERATED_DIR`
points at a pre-generated tree instead, for an offline or read-only install or
to share one copy across a cluster.

`codegen/` holds the other three -- `gen_kernels.py`, `gen_k2_kernels.py` and
`gen_j_kernels.py` -- which *lift* the integral arithmetic out of a GPU4PySCF
source file and rewrite the scaffolding around it, so they take its path and
need a checkout an installed user has no reason to have.  Their output stays
tracked:

```bash
codegen/build_kernels.sh /path/to/gpu4pyscf/lib/gvhf-rys/unrolled_rys_k.cu
```

The written families need no such path, and normally need no command either --
they appear when they are first wanted.  To force it, or to fill a cache
ahead of time:

```bash
python -c 'from electrolites import _gen; print(_gen.ensure_all())'
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
| `FASTJ_NO_MDM`, `FASTJ_MDM_BLOCKS` | hand every multi-density Coulomb build back to GPU4PySCF; restrict which density-block sizes are used (the `docs/ROUND3_B4A.md` ablation) |
| `FASTNLC_MODE` | `mixed` (default) or `fp64` for the VV10 double sum |
| `FASTJ_TIME`, `FASTXCGRAD_MODE` | per-call timing split; XC-gradient block strategy |
| `FASTFNLC_CACHE`, `FASTFNLC_BATCH` | hand the two halves of the VV10 response change back (the ablations) |
| `FASTXCFXC_MODE`, `FASTXCFXC_TIME` | `blocked` (default), `nokernel` or `grouped` for the XC response kernel -- the last is catalogue B4d as written and is the slowest of the three; and its per-stage timing split |

## License

Apache-2.0. Substantial parts of the CUDA are derived from GPU4PySCF, also
Apache-2.0 — see `NOTICE` for exactly which files and how.
