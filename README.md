# Electrolites

Drop-in GPU kernel replacements for [GPU4PySCF](https://github.com/pyscf/gpu4pyscf):
the exchange and Coulomb builds and the fused build of both, the XC build, the
VV10 non-local correlation, the density-fitted two-electron path, the analytic
nuclear gradient, and the grid and two-electron terms of an analytic Hessian
and of every response property.

Importing a module monkey-patches one GPU4PySCF entry point in place.  Anything
a module does not cover falls through to GPU4PySCF's own code, so a patched run
computes the same quantity as an unpatched one — measured over the suites in
`tests/` at **7.3e-12 Eh** on the energy and **3.1e-12** per gradient
component, **1.0e-14** relative on an exchange matrix, **2.1e-14** on a fused
Coulomb one and **6.2e-11** on a whole analytic Hessian.

Every number below is measured in this repository, and `docs/` has the round
that produced each one: the stage tables, the ablations, the alternatives that
were measured and rejected, and the ways a measurement on each path goes
silently wrong.

---

## Results

One A100-SXM4-40GB unless a row says otherwise, GPU4PySCF 1.8.1, level-3 grids.

### SCF and gradient — B3LYP-D3(BJ)/6-31G\*

| | GPU4PySCF | + Electrolites | |
|---|---|---|---|
| 284-atom cluster, SCF | 127.2 s | **55.1 s** | **2.31x** |
| 284-atom cluster, gradient | 19.4 s | **12.5 s** | **1.55x** |
| 545-atom cluster, one SCF cycle (`minao` / converged) | 9.62 / 29.8 s | **3.22 / 14.1 s** | **2.98x / 2.11x** |

`docs/UPGRADE_1.8.1.md`, `docs/ROUND2_1.8.1.md`, and `docs/ROUND3_B4A.md` /
`docs/ROUND4_B4B.md` for the multi-density Coulomb and exchange builds.

### Analytic Hessian — B3LYP/def2-SVP

| | GPU4PySCF | + Electrolites | |
|---|---|---|---|
| 42-atom Hessian (penicillin V), whole run | 84.3 s | **53.9 s** | **1.56x** |
| ↳ its two XC terms, at fixed density | 15.5 s | **2.11 s** | **7.34x** |
| 113-atom Hessian (taxol), its two XC terms | 117.8 s | **22.2 s** | **5.30x** |
| 42-atom **DF** Hessian, whole run (80 GB card) | 34.7 s | **15.5 s** | **2.23x** |
| ↳ DF-PBE | 25.6 s | **8.37 s** | **3.06x** |
| 42-atom DF Hessian, def2-TZVPP | **cannot be run** | **46.6 s** | — |
| 113-atom DF Hessian | **cannot be run** | **701 s** | — |
| 113-atom open-shell (UKS) DF Hessian, taxol⁺, 80 GB / 40 GB card | **cannot be run** | **1324.5 s / 1348.6 s** | — |

The last two rows are capacity results, not ratios: GPU4PySCF sizes one
cuTENSOR contraction in the CPHF build from free memory alone and
`createPlan` then refuses it with tens of GB free.  The DF rows' remaining time is mostly that
same CPHF build, which is at 78 % of the card's fp64 peak; the three terms
around it that nothing covered before (the CPHF's grid half, and the `ip2`
and `h1` two-electron terms) are 2.6x, 1.4x and 1.4x inside a 42-atom run.
Upstream's restricted **meta-GGA** Hessian is wrong and irreproducible — two
of its functions contract `fxc` into one of their own operands, which
cuTENSOR gets wrong at five density components — and `fastxchess` repairs
both, and since W28 replaces both meta-GGA XC terms as it does the GGA ones:
a 42-atom DF-TPSS Hessian is **41.5 s → 13.4 s**, and 3.2x on open-shell Ni
and Pt complexes (`docs/README_fastxchessmgga.md`).  The `ip2` term has since gone to 2.3x–2.5x GPU4PySCF's at 113 atoms
with the fitting metric as a Cholesky factor.
`docs/README_fastdfhess.md`, `docs/README_fastdfhessip.md`,
`docs/README_fastdfhessip2.md`, `docs/README_mgga_repro.md`,
`docs/HESSIAN_1.8.1.md`, `docs/README_fastxchess.md`.

### Density-fitted gradient — B3LYP/def2-SVP

| | GPU4PySCF | + Electrolites | |
|---|---|---|---|
| penicillin V/def2-SVP, whole gradient | 1.240 s | **0.858 s** | **1.45x** |
| penicillin V/def2-TZVPP, whole gradient | 3.317 s | **1.651 s** | **2.01x** |
| taxol (113 atoms), whole gradient | 8.172 s | **3.604 s** | **2.27x** |
| **PfPMT (284 atoms), whole gradient** | **cannot be run** | **68.8 s** | — |

Another capacity result: `df/grad/rhf.py` holds the occupied-occupied
transform of the three-centre integrals **twice**, as dense
`(naux, nocc, nocc)` device arrays — 37.7 GB each at 284 atoms, so a 40 GB
card is asked for 75.5 GB.  Neither copy is necessary.
`docs/README_fastdfgrad.md`, `docs/README_fastdfgradoo.md`, and
`docs/README_fastdfplan.md` for the cuTENSOR plan wall three of these modules
met and now ask about instead of retrying.

### Density-fitted SCF

| | GPU4PySCF | + Electrolites | |
|---|---|---|---|
| penicillin V, DF-B3LYP, cderi on the host | 4.57 s | **3.73 s** | **1.22x** |
| penicillin V, DF-**wB97M-V** | 22.01 s | **14.19 s** | **1.55x** |
| taxol, DF-B3LYP, cderi on the host | 25.75 s | **22.39 s** | **1.15x** |
| the cderi build alone, taxol, on the host | 12.76 s | **3.99 s** | **3.20x** |
| the cderi build alone, PfPMT, on the host | 87.1 s | **29.5 s** | **2.95x** |
| `df_jk.get_j` (pure functional), penicillin V | 26.2 ms | **4.27 ms** | **6.14x** |
| ↳ taxol, 113 atoms | 157.3 ms | **29.1 ms** | **5.40x** |
| ↳ PfPMT, 284 atoms | 732.1 ms | **143.7 ms** | **5.10x** |
| ↳ penicillin V, two densities | 130.8 ms | **8.5 ms** | **15.3x** |
| ↳ Ni dithiolene, open shell, two densities | 317.7 ms | **11.7 ms** | **27.2x** |
| whole DF-PBE SCF, PfPMT, 50 cycles, against W9 (which is 1.01x at this size) | 85.1 s | **56.0 s** | **1.52x** |

**RIJONX** — `density_fit(only_dfj=True)`, density-fitted J with exact
exchange — cannot run on GPU4PySCF 1.8.1 at all (`'Mole' object has no
attribute 'shape'`).  Repaired, it is the fastest way this repository builds a
large hybrid, because a Coulomb-only DF build never constructs a cderi:

| PfPMT, 284 atoms, B3LYP/def2-SVP | full DF | **RIJONX** | direct |
|---|---|---|---|
| whole SCF, 18 cycles | 300.5 s | **64.1 s** | 65.4 s |
| before the first Fock matrix | 88.7 s / 86.4 GB | **1.08 s / nothing** | — |

`docs/README_fastdf.md`, `docs/README_fastdfj.md`, `docs/README_fastdfj3c.md`,
`docs/README_fastdfj3ck.md`, `docs/README_fastdfj3cinc.md`, `docs/README_fastdfj3cscr.md`,
`docs/README_fastdfcderi.md`, `docs/README_fastdfstore.md`,
`docs/README_fastdfblock.md`, `docs/README_fastdfunpack.md`,
`docs/README_fastdffp32.md`, `docs/README_fastdfrsh.md`, and
`docs/DF_WORK_LIST.md` for what is left.

### Response properties — penicillin V/def2-SVP, whole runs

| functional | property | GPU4PySCF | + Electrolites | |
|---|---|---|---|---|
| B3LYP | polarizability (CPKS, 3 RHS) | 16.77 s | **8.06 s** | **2.08x** |
| | TDA | 22.21 s | **14.26 s** | **1.56x** |
| B3LYP, 113 atoms | polarizability | 101.08 s | **75.06 s** | **1.35x** |
| PBE | TDA | 9.16 s | **5.13 s** | **1.79x** |
| M06-2X | polarizability | 22.53 s | **15.59 s** | **1.45x** |
| wB97M-V | polarizability | 154.67 s | **111.26 s** | **1.39x** |

The polarizability tensor and the excitation energies are unchanged to every
digit printed.  `docs/RESPONSE_1.8.1.md`, `docs/README_fastjk.md`,
`docs/README_fastxcfxc.md`, `docs/README_fastfnlc.md`.

### Open-shell transition-metal complexes — UKS, B3LYP/def2-SVP

| | GPU4PySCF | + Electrolites | |
|---|---|---|---|
| 21-atom Ni dithiolene (doublet), SCF | 101.9 s | **22.0 s** | **4.64x** |
| ↳ its gradient | 7.84 s | **3.05 s** | **2.57x** |
| 60-atom Cu macrocycle (quartet), SCF | 72.6 s | **25.6 s** | **2.84x** |
| ↳ its gradient | 9.60 s | **4.94 s** | **1.94x** |
| ↳ its DF Hessian's `ip2` / `h1` terms | 10.96 / 4.89 s | **5.93 / 2.90 s** | **1.85x / 1.69x** |
| 113-atom Co(II) sextet, DF Hessian `ip2` term | **cannot be run** | **48.2 s** | — |

83 % of a 28 242-entry CSD-derived set is open-shell, and def2-SVP already
reaches an f shell on a 3d metal.  Several entry points an open-shell run
takes bind their own copies of a function, so patching the restricted one
never reached them; `benchmarks/coverage_probe.py` is what finds that, and it
is worth running before believing any coverage claim here.
`docs/OPENSHELL_TMC.md`, `docs/OPENSHELL_HESSIAN.md`, `docs/OPENSHELL_FXC.md`,
`docs/VV10_OPENSHELL.md`, `docs/README_fastdfuhessip.md`.

### Not a drop-in replacement

`electrolites/dflocal.py` is **robust pair-atomic (PARI) fitting** — a
different approximation, 2e-5 to 8e-5 Eh per atom away from global density
fitting.  It is not in `patch_all()`, `dflocal.density_fit(mf)` is not
`mf.density_fit()`, it warns once on stdout, and a gradient or Hessian of such
an object raises rather than silently differentiating the global fit.  Its
exchange is 2.93x fewer flops than stock at 284 atoms and 4.35x at 545, and
its coefficients are 0.79 GB where global DF stores 86.4.
`docs/README_dflocal.md`, `docs/README_dflocal_k.md`.

---

## Install

```bash
pip install gpu4pyscf-cuda12x      # or gpu4pyscf-cuda11x; 1.7.0 or 1.8.x
pip install git+https://github.com/Benzoin96485/Electrolites.git
# optional: pyscf-properties, the CPU reference of the ssc / nmr_uks tests
pip install "electrolites[properties] @ git+https://github.com/Benzoin96485/Electrolites.git"
```

Nothing is compiled at install time: every kernel is built at run time by
NVRTC through CuPy and cached on the source hash.  Only the angular-momentum
classes the basis can reach are compiled, so a 6-31G\* job's first run pays
about 36 s of JIT rather than 98 s.  `electrolites.warmup` moves that cost off
the first molecule of a dataset run:

```bash
python -m electrolites.warmup --basis def2-tzvp --elements Cu,Cl,C,H,N,O \
    --spin 2 --jobs 8 --cupy-cache /shared/cupy \
    --generated-dir /shared/electrolites/cuda
export CUPY_CACHE_DIR=/shared/cupy ELECTROLITES_GENERATED_DIR=/shared/electrolites/cuda
```

Two optional libraries need relocatable device code and so are built with
`nvcc` on first use (~25 s, cached): a faster copy of GPU4PySCF's general
exchange kernel, and the Hessian's second-derivative ERI kernels.  Without
`nvcc` those two fall back to GPU4PySCF — slower, still correct.
`docs/PRIM_PAIR_AND_JIT.md`.

## Use

```python
import electrolites
electrolites.patch_all()

from pyscf import gto
from gpu4pyscf import dft

mol = gto.M(atom='h2o.xyz', basis='def2-tzvpp')
mf = dft.RKS(mol, xc='wb97m-v')
e = mf.kernel()
de = mf.nuc_grad_method().kernel()
```

`patch_all()` refuses to patch if the installed GPU4PySCF does not expose an
entry point it replaces, naming which one — it never patches silently.

```python
electrolites.check_targets()          # {'fastk': None, ...}; a string means missing
electrolites.patch('fastk', 'fastxc') # a subset; how every ablation is measured
electrolites.patched()                # what is active in this process
```

There is no unpatch: the patches are global, so the benchmarks compare
separate processes.  Every module asserts a **single device** except `fast1e`.

Features GPU4PySCF does not have at all are *added* rather than patched, and
go through `electrolites.enable(...)`: each one first checks that its slot is
still empty upstream, so a GPU4PySCF that grows its own version is noticed
rather than shadowed (`electrolites/features.py`,
`docs/FEATURE_TARGETS_PYSCF_2.13.md`).  Their GPU-against-CPU numbers follow
`docs/CPU_BASELINE.md`.

## What each module replaces

| module | replaces | round |
|---|---|---|
| `fastk` | `scf.jk._VHFOpt.get_k` | `docs/README_fastk.md` |
| `fastj` | `scf.j_engine._VHFOpt.get_j` | `docs/README_fastj.md` |
| `fastjk` | `scf.jk._VHFOpt.get_jk` — the **fused** build the response path takes | `docs/README_fastjk.md` |
| `fastxc` | `dft.numint.NumInt.nr_rks` and `nr_uks` | `docs/README_fastxc.md` |
| `fastnlc` | `dft.numint._vv10nlc` | `docs/README_fastnlc.md` |
| `fastxcnlc` | `RKS.get_veff`, `UKS.get_veff`, `_DFHF.get_veff` — XC and VV10 in one grid sweep | `docs/VV10_OPENSHELL.md` |
| `fastrsh` | `dft.rks.RKS.get_veff` — range-separated hybrids on 1.7.x only | `docs/README_fastrsh.md` |
| `fast1e` | `gto.int3c1e.get_int3c1e*` — PCM, QM/MM, ESP, CHELPG | `docs/README_fast1e.md` |
| `fastejk` | `grad.rhf._jk_energy_per_atom` | `docs/README_fastejk.md` |
| `fastgrad` | `grad.rks.Gradients.energy_ee`, `grad.uks.Gradients.energy_ee`, `hessian.{rks,uks}.partial_hess_elec` | — |
| `fastxcgrad` | `grad.{rks,uks}.get_exc`, `get_nlc_exc` | `docs/README_fastxcgrad.md` |
| `fastgradh` | `grad.rhf.GradientsBase.get_hcore` | — |
| `fastxchess` | `hessian.{rks,uks}._get_vxc_deriv{1,2}` (GGA and meta-GGA), and `hessian.rks`'s in-place `contract` calls | `docs/README_fastxchess.md`, `docs/README_mgga_repro.md`, `docs/README_fastxchessmgga.md` |
| `fastxcfxc` | `dft.numint.nr_{rks,uks}_fxc` — the grid half of every response property — and `hessian.rks.nr_rks_fxc_mo`, a restricted Hessian's | `docs/README_fastxcfxc.md`, `docs/README_fastdfhessip.md` |
| `fastfnlc` | `hessian.{rks,uks}.nr_rks_fnlc_mo` — the VV10 response term | `docs/README_fastfnlc.md` |
| `fastvv10ft` | `libgdft` `VXC_vv10nlc_hess_eval_f_t`, `..._EUW_with_weight1` — the VV10 response and grid-weight pair kernels on the fp64 tensor cores | `docs/README_fastvv10ft.md` |
| `fastvnlch` | `hessian.rks._get_vnlc_deriv1` (and its `uks`/DF bindings) — the VV10 Fock derivative's atom loops as one GEMM per block | `docs/README_fastvv10ft.md` §6 |
| `fastir` | `properties.ir.eval_ir_freq_intensity` — the Hessian's CPHF solution reused for the dipole derivative | `docs/README_fastir.md` |
| `fastnacms` | `tdscf.rhf.TDBase.force_and_nacv` — the force and all NACs through the batched multi-state solver (restricted singlets) | `docs/README_fastnacms.md` |
| `fastnmr` | `properties.shielding.nr_rks` — the NMR shielding's GIAO XC term with the GIAO AOs formed on the GPU; `shielding.get_vxc` — the GIAO two-electron term on the GPU through shifted-basis J/K (`electrolites.giao_jk`) instead of the host `int2e_ig1` | `docs/README_fastnmr.md` |
| `fasttdejk` | `grad.tdrhf._jk_energies_per_atom` — the TDDFT gradient / NAC two-electron derivative for density pairs, a generated kernel family | `docs/README_fasttdejk.md` |
| `fastdfmp2` | `mp.dfmp2_addons.get_df{,u}mp2_energy_pair_intra` — the DF-MP2 / DF-UMP2 pair-energy loops as block GEMMs | `docs/README_fastdfmp2.md` |
| `fastc6` | `properties.c6._solve_full_spectrum`, `calc_c6` — the C6 full-spectrum solve with the ERI and the linear algebra on the device | `docs/README_fastc6.md` |
| `fastqmmm` | `qmmm.itrf.QMMMSCF.energy_nuc`, `QMMMGrad.grad_nuc`, `grad_nuc_mm` — the QM-nucleus–MM-charge terms on the device, cached across SCF cycles | `docs/README_fastqmmm.md` |
| `fastpbcgrad` | `pbc.grad.{krhf,kuhf}.grad_elec`, `pbc.grad.{rhf,uhf}.Gradients.grad_elec`, `pbc.grad.*_stress._get_pp_nonloc_strain_derivatives` — the periodic gradient's local 1e term from one density pass; the stress tensor's non-local PP term batched on the device | `docs/README_fastpbcgrad.md` |
| `fastmcfun` | `dft.mcfun_gpu._make_sph_samples` — the multi-collinear GKS functional on one spin sample per antipodal pair | `docs/README_fastmcfun.md` |
| `fastfftk` | `pbc.df.fft_jk.get_k_kpts` — FFTDF k-point exchange built for one k of each time-reversal pair | `docs/README_fastfftk.md` |
| `fastgridresp` | `hessian.{rks,uks}._get_exc_deriv2_grid_response`, `hessian.rks._get_enlc_deriv2` — the Becke weights' second derivative contracted on the fly, never stored | `docs/README_fastgridresp.md` |
| `fastjkip1` | `libvhf_rys.RYS_build_jk_ip1` (via `hessian.rhf._get_jk_ip1`), `hessian.{uhf,uks}.make_h1` — the Hessian's h1 two-electron kernels without per-launch device queries, both spins from one integral pass | `docs/README_fastjkip1.md` |
| `fastsfxc` | `tdscf._uhf_resp_sf.mcfun_eval_xc_adapter_sf` (and `grad.tduks_sf`'s binding) — the spin-flip multi-collinear XC kernel from libxc's raw derivatives, chain rule in the total/spin basis | `docs/README_fastsfxc.md` |
| `fastpcmbatch` | `gto.int3c1e.int1e_grids` (and where imported), `solvent.hessian.pcm.get_dqsym_dx` — PCM's multi-vector integrals in one pass + GEMM; the charge derivative once per Hessian | `docs/README_fastpcmbatch.md` |
| `fastscfscreen` | `dft.{rks,uks}.{RKS,UKS}.get_veff`, `scf.hf.RHF`, `scf.uhf.UHF` — direct SCF screens at 1e-10 while the density is far off, then rebuilds the Fock matrix tight | `docs/README_fastscfscreen.md` |
| `fastmp2` | `scf.int4c2e.get_int4c2e_ovov` — conventional RMP2's (ia\|jb) as a blocked half transformation (strided-batched GEMMs into the half-transformed tensor), batched over occupied orbitals to fit the card | `electrolites/fastmp2.py` |
| `fastccsd` | `cc.ccsd_incore.CCSDBase.ccsd` — closed-shell CCSD iterations with amplitudes, MO integrals, `ao2mo` accumulators and DIIS on the device, AO integral blocks computed once and cached (device or host) | `electrolites/fastccsd.py` |
| `fastrshresp` | `scf._response_functions._gen_{rhf,uhf}_response` — a range-separated hybrid's response exchange in one pass (`fastk.get_k_rsh`, now also `hermi = 0`), the J+K arrangement chosen per call shape by timing | `electrolites/fastrshresp.py` |
| `fastpbcdf` | `pbc.df.rsdf_builder.compressed_cderi_kk`, `pbc.df.df.GDF.loop` — the GDF k-point cderi on the device instead of mapped host memory when it fits | `electrolites/fastpbcdf.py` |
| `fastejk2` | `hessian.rhf._ip2_quartets_scheme` — the ip2 launch scheme | `docs/README_fastejk2.md` |
| `fastdf` | `df.df_jk.get_jk` — the density-fitted J and K build | `docs/README_fastdf.md`, `docs/README_fastdfblock.md`, `docs/README_fastdfunpack.md` |
| `fastdfj` | `df.df_jk._DFHF.get_jk` — **RIJONX**, which 1.8.1 cannot run | `docs/README_fastdfj.md` |
| `fastdfj3c` | `df.df_jk.get_j` and the three-centre Coulomb engine, its two integral kernels included | `docs/README_fastdfj3c.md`, `docs/README_fastdfj3ck.md` |
| `fastdfcderi` | `df.df._cholesky_eri` — the one-time cderi build — and `df.int3c2e_bdiv.int2c2e`, the two-centre metric integrals under it | `docs/README_fastdfcderi.md`, `docs/README_fastdfstore.md` |
| `fastdfgrad` | `df.grad.{rhf,rks,uhf,uks}.Gradients.jk_energy_per_atom` | `docs/README_fastdfgrad.md`, `docs/README_fastdfgradoo.md` |
| `fastdfhess` | `df.hessian.rhf._get_jk` — the DF Hessian's CPHF build | `docs/README_fastdfhess.md` |
| `fastdfhessip` | `df.hessian.{rhf,uhf}._jk_energy_per_atom` and `_get_veff` — the DF Hessian's `ip2` and `h1` two-electron terms, restricted and unrestricted | `docs/README_fastdfhessip.md`, `docs/README_fastdfhessip2.md`, `docs/README_fastdfuhessip.md`, `docs/README_fastdfh1occ.md`, `docs/README_fastdfhessip3.md` |
| `fastdffp32` | `DF.loop` and the DF Hessian's cderi transfer — carries `FASTDF_FP32=1`, which is inert unless set and **changes the answer** when it is | `docs/README_fastdffp32.md` |
| `fastdfrsh` | `df.df_jk._DFHF.get_veff` for a range-separated hybrid — one cderi tensor where upstream keeps two: exactly, for the long-range-only functionals (ωB97, LC-ωPBE, LC-VV10), and under `FASTDFRSH=fuse`, which **changes the answer** by 1e-7–1e-6 Eh, for the rest | `docs/README_fastdfrsh.md` |
| `guards` | `scf.rohf.ROHF`, `dft.roks.ROKS` (`Hessian`, `Gradients`, `TDA`/`TDHF`/`TDDFT`), `df.df_jk._DFHF.Gradients`/`Hessian`, `qmmm.itrf.QMMMSCF.Hessian` — not a kernel: the entry points that return a **wrong** answer on 1.8.1 (the ROHF/ROKS and QM/MM Hessians) now raise, the ROHF/ROKS gradient goes to PySCF on the CPU with a warning | `docs/FEATURE_TARGETS_PYSCF_2.13.md` §3.2, §9 |
| `fixes` | `dft.{rks,uks,roks}` vSAP, `scf.rohf.ROHF.get_init_guess`, `cc.ccsd_incore.CCSDBase.to_cpu`/`CCSD.__init__`, `mp.dfmp2.DFMP2._kernel_impl`, `mp.dfump2.DFUMP2._kernel_impl` — not a kernel: five 1.8.1 crashes on legal inputs (vSAP guess, ROHF Newton, `CCSD.to_cpu`, open-shell CCSD's error, frozen-core DF-MP2 with `with_t2`) | `docs/FEATURE_TARGETS_PYSCF_2.13.md` §3.3, §9 |
| `planguard` | `lib.cutensor.contraction` — not a kernel: every `contract` in GPU4PySCF and here goes through it, and a contraction cuTENSOR refuses (an output past 2²⁹ elements whose extents do not factor) is cut into pieces it plans instead of raising | `docs/README_fastdfaudit.md` |

`docs/CATALOGUE_06_STATUS.md` maps these onto the kernel-target catalogue,
including the entries that turned out to be closed upstream or reversed.

## Features added to GPU4PySCF

Methods that previously ran only on the CPU (in PySCF or pyscf-properties),
or nowhere, now run on the GPU. Turn them on with `electrolites.enable(name)`
(or `enable_all()`); each first checks that GPU4PySCF still lacks it.

Speed-ups are measured in our test environment: one NVIDIA A100-SXM4-80GB
against one full CPU node (2x AMD EPYC 7763, 128 cores), with PySCF 2.13 or
pyscf-properties as the CPU reference (32 threads for the CAS-based methods,
128 otherwise). Basis def2-SVP unless noted. Every feature is checked against
the CPU code, or against finite differences where there is none
(`tests/`). Full numbers: `docs/FEATURE_TARGETS_PYSCF_2.13.md`.

| feature | what now runs on the GPU | typical speed-up vs the CPU node |
|---|---|---|
| `stability` | `mf.stability()`: RHF/RKS, UHF/UKS (internal and external), ROHF/ROKS (internal) | **41–66x** (Ni dithiolene⁻ and a 113-atom Co(II) complex, UKS B3LYP) |
| `rohf_grad` | ROHF/ROKS analytic gradients, direct and density-fitted | not timed (was a CPU fallback) |
| `sap` | `init_guess='sap'` for RHF/UHF/ROHF and KS | not timed (initial guess) |
| `mp2_rdm` | MP2 one-particle density and frozen natural orbitals, conventional and DF | not timed |
| `ccsd_t` | CCSD(T): the (T) correction on the device | (T) **7–8x**; whole CCSD(T) **9–19x** (glycine/TZVP, alanine dipeptide, ferrocene) |
| `casscf` | CASSCF/CASCI and state-averaged CASSCF: FCI solver, integrals, J/K and the orbital step | **32–60x** per macro-iteration (p-benzyne CAS(10,10) 32x, Ni dithiolene⁻ CAS(10,9) 35x, Cu macrocycle²⁺ CAS(10,9) 60x) |
| `casscf` gradients | CASSCF gradients (single state and state-averaged) and non-adiabatic couplings, density-fitted | gradient **7–9x** (Ni, Cu complexes); SA-2 gradient **10x**, SA-2 NAC **15x** (p-benzyne CAS(10,10)); DF NACs do not exist in PySCF |
| `mcpdft` | MC-PDFT energies (translated LDA/GGA and hybrid on-top functionals, state averaging) | **8–10x** (Ni, Cu complexes, tPBE) |
| `nevpt2` | SC-NEVPT2: 3- and 4-RDM contractions, integrals and all subspaces | **15–24x** (p-benzyne CAS(10,10)–(14,14), ferrocene CAS(10,10)); CAS(14,14) 209 s vs 3201 s |
| `dfmp2_grad` | RI-MP2 nuclear gradient and relaxed density (RHF or hybrid RKS reference) | PySCF has no DF-MP2 gradient; **12–58x** its DF relaxed density alone, **18–113x** its conventional MP2 gradient |
| `dhdft` | double-hybrid energies and analytic gradients (B2PLYP, B2GP-PLYP, PBE0-DH, …) | new: neither PySCF nor GPU4PySCF has them |
| `rpa` | direct RPA correlation energy | **6–15x** (penicillin V, def2-SVP and def2-TZVP) |
| `gw` | G0W0 quasiparticle energies (analytic continuation) | **8–13x** (alanine dipeptide, penicillin V) |
| `pbc_tda` | periodic TDA, Γ point and k-points (kshift 0), RKS/RHF | k-points **31x** (PBE) and **134x** (PBE0), diamond 2×2×2; at Γ, 16- and 54-atom cells run in seconds where PySCF runs out of memory or does not finish |
| `pbc_stability` | periodic RHF/RKS stability analysis, Γ and k-points | > **44x** (diamond 2×2×2 PBE0: 68 s; PySCF unfinished after 50 min) |
| `polar` | static and frequency-dependent polarizability, RHF/RKS and UHF/UKS | **410x** (penicillin V, α(ω)); open-shell cases where pyscf-properties fails to converge |
| `polar` (β) | static first hyperpolarizability, RHF/RKS and UHF/UKS, including the XC kernel-derivative term pyscf-properties omits | **3.5x** (HF) to **42x** (B3LYP) on penicillin V; **10x** alanine dipeptide cation UKS, where pyscf-properties' value is 3x off |
| `raman_uks` | Raman activities and depolarization ratios for UHF/UKS (and RHF/RKS) | new: PySCF has no Raman code; GPU4PySCF's is closed-shell only (alanine dipeptide cation UKS B3LYP: 71 s with the Hessian) |
| `ir_uks` | IR intensities for UHF/UKS (and RHF/RKS) | **24x** (alanine dipeptide cation, UKS B3LYP, including the Hessian) |
| `nmr_uks` | NMR shielding for UHF/UKS; the GIAO two-electron term on the GPU for RKS as well | UKS **1.6–4.1x**, RKS **1.4–3.3x** (22–42 atoms); 10x GPU4PySCF's own host-term path at taxol |
| `ssc` | NMR spin–spin couplings (DSO, PSO, FC, FC+SD), RHF/RKS | **16–67x** (glycine to penicillin V, B3LYP) |
| `cubegen` | density, orbital and electrostatic-potential cube files | **345x** (electrostatic potential, penicillin V/def2-TZVP, 80³ grid) |
| QM/MM | QM/MM Hessian (RHF/RKS) and Gaussian-smeared MM charges | new: PySCF has no QM/MM Hessian; GPU4PySCF returned the vacuum one |
| `lo` | orbital localisation: Foster–Boys, Pipek–Mezey (all PySCF populations), density-fitted Edmiston–Ruedenberg, intrinsic bond orbitals | Boys **4–6x**, PM **14–23x**, IBO **6–37x** (penicillin V, taxol); ER 0.7 s against 32 min on penicillin V |
| `eom` | EOM-IP/EA-CCSD (right and left vectors) and singlet, triplet and spin-flip EOM-EE-CCSD on the GPU CCSD, without the v⁴ intermediate (`mycc.ipccsd()`, `eomee_ccsd_*()`, `eomsf_ccsd()` now work) | IP **7–18x**, EA **11–39x**, EE singlet **49–118x**, triplet/spin-flip **59–74x**, left IP/EA **4–34x** (alanine dipeptide, ferrocene, glycine/TZVP) |
| `cc_lambda` | CCSD Λ equations (`mycc.solve_lambda()`) | **14–43x** (alanine dipeptide, glycine/TZVP, ferrocene) |
| `cc_rdm` | CCSD one- and two-particle density matrices (`mycc.make_rdm1()`, `make_rdm2()`) | 2-RDM **37–128x** (same systems) |
| `adc` | ADC(2) excitation (singlet), ionisation and attachment energies | EE **19x**, IP **2–7x**, EA **5–14x** (alanine dipeptide, penicillin V) |
| `kmp2` | k-point MP2 (KRHF, GDF) | **11–39x** (diamond, 3×3×3 and 4×4×4 meshes); 3x on an 8-atom cell at 2×2×2 |
| `x2c_grad` | spin-free X2C (sfX2C1e) nuclear gradients, RHF/RKS/UHF/UKS; also makes GPU4PySCF's X2C SCF run | **22x** (Pt(II) complex, UKS PBE, all-electron) |
| `ghf_td` | GHF TDA and TDHF (Hartree–Fock) | TDA **4–6x**, TDHF **2x** (alanine dipeptide and penicillin V cations) |

## Tests

```bash
python tests/test_full.py --patch all   # energy and gradient, patched vs not
python tests/test_patch_aliases.py      # no GPU: does every reference to a
                                        # patched function see the patch
ls tests/                               # one suite per module
```

The cases span six basis sets to cc-pVQZ, twelve functionals including three
meta-GGAs, `omega < 0`, third-row elements, a cation, an open-shell radical,
and six open-shell transition-metal complexes with 3d, 4d and 5d metals.

## Benchmarks

`benchmarks/` holds the harness behind every table above; each round's write-up
in `docs/` names the ones it used.  Three are worth knowing about before
measuring anything:

* `benchmarks/coverage_probe.py` — which modules are on the code path of a
  given calculation **at all**.  Three rounds here have shipped a module that
  was loaded, correct and never called; this is the only thing that found them.
* `benchmarks/scf_anatomy.py`, `benchmarks/hessian_anatomy.py`,
  `benchmarks/df_anatomy.py` — where a wall clock actually goes, stage by
  stage, with a synchronisation point on both sides of each.
* `docs/DF_WORK_LIST.md` §5 — the ways a measurement on the density-fitting
  path goes silently wrong, which generalise to the rest.

## Environment switches

Every module's switches are listed with their meanings at the top of its own
file in `electrolites/`, and are what the ablation columns in `docs/` are
measured with.  The ones that are not per-module:

| variable | effect |
|---|---|
| `ELECTROLITES_NVCC`, `ELECTROLITES_ARCH` | compiler and compute capability for the two `nvcc`-built libraries |
| `ELECTROLITES_LIBMYKG`, `ELECTROLITES_LIBMYEJK2` | use prebuilt copies instead of building |
| `ELECTROLITES_CACHE` | where to cache the build (default `~/.cache/electrolites`) |
| `ELECTROLITES_GENERATED_DIR` | a pre-generated tree of the written CUDA, used as is — for an offline or read-only install, or one copy shared across a cluster |
| `ELECTROLITES_FULL_MODULE` | compile every angular-momentum class rather than the ones the basis reaches |
| `ELECTROLITES_MAX_PRIM_PAIR` | largest bra primitive pair a second kernel module is compiled for (default 100, which covers def2-QZVP); **0 is the ablation** |

Two switches change the answer rather than the speed and are off by default:
`FASTDF_FP32=1` (the cderi in single precision) and `FASTEJK2_KERNELS=1` (the
lifted ip2 kernels, which are correct and measured not to pay).

## License

Apache-2.0. Substantial parts of the CUDA are derived from GPU4PySCF, also
Apache-2.0 — see `NOTICE` for exactly which files and how.
