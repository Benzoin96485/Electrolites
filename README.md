# Electrolites

Drop-in GPU kernel replacements for [GPU4PySCF](https://github.com/pyscf/gpu4pyscf):
the exchange and Coulomb builds and the fused build of both, the XC build, the
VV10 non-local correlation, the analytic nuclear gradient, and the grid and
two-electron terms of an analytic Hessian and of every response property.

Importing a module monkey-patches one GPU4PySCF entry point in place.  Anything
a module does not cover falls through to GPU4PySCF's own code, so a patched run
computes the same quantity as an unpatched one — measured over the suites in
`tests/` at **7.3e-12 Eh** on the energy and **3.1e-12** per gradient
component (15 SCF + gradient pairs in separate processes), **1.0e-14** relative
on an exchange matrix and **2.1e-14** on a fused Coulomb one at any number of
densities, and **6.2e-11** on a whole analytic Hessian.

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

and, for the **response properties** -- whole runs, penicillin V/def2-SVP,
level-3 grids, five roots for TDA.  The `fastxcfxc`/`fastfnlc` column is what
the previous round reached; `+ fastjk` is this one, which adds the *fused*
Coulomb-and-exchange build the response path actually calls:

| functional | property | GPU4PySCF 1.8.1 | + fastxcfxc/fastfnlc | | + fastjk | |
|---|---|---|---|---|---|---|
| B3LYP | polarizability (CPKS, 3 RHS) | 16.77 s | 14.26 s | 1.15x | **8.06 s** | **2.08x** |
| | TDA | 22.21 s | 17.74 s | 1.24x | **14.26 s** | **1.56x** |
| B3LYP, 113 atoms | polarizability | 101.08 s | — | — | **75.06 s** | **1.35x** |
| | TDA | 189.66 s | — | — | **148.67 s** | **1.28x** |
| PBE | polarizability | 9.49 s | **7.12 s** | **1.33x** | | |
| | TDA | 9.16 s | **5.13 s** | **1.79x** | | |
| M06-2X | polarizability | 22.53 s | **15.59 s** | **1.45x** | | |
| | TDA | 32.30 s | **21.75 s** | **1.49x** | | |
| wB97M-V | polarizability | 154.67 s | **111.26 s** | **1.39x** | | |
| | TDA | 49.02 s | **38.10 s** | **1.29x** | | |

The polarizability tensor and the excitation energies are unchanged to every
digit printed.  `docs/README_fastjk.md` has the per-kernel table: the fused
build itself is 2.79x at one right-hand side, 1.96x at three and 1.65x at
five, and `docs/ROUND4_B4B.md` §4 says why that ratio falls with the number of
densities and where its floor is.

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
method.  Measured that way, the fused J and K build of a **hybrid** did not
move at all under the previous round's `patch_all()` -- 11.67 -> 11.70 s,
13.56 -> 13.64 s, 19.64 -> 20.07 s -- which was the direct evidence that
neither `fastj` nor `fastk` was on that path.  It is a third code path, and
`fastjk` is now on it: the same row is 11.76 -> 5.37 s and 13.53 -> 10.17 s.

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

The Hessian rows are dominated by the exchange build inside the CPHF solve --
37 % of the 42-atom run and 54 % of the 113-atom one after the `fastxchess`
round, because `fastk` used to hand every multi-density call back to
GPU4PySCF.  It no longer does, and that row is **1.16x** (19.65 -> 16.96 s,
whole Hessian 56.49 -> 53.36 s).  Only 1.16x, and
`docs/ROUND4_B4B.md` §4 is the reason: fitting the build as
`R + n_dm*C` puts the whole of these kernels' advantage in `R`, the
per-quartet integral work, which at `n_dm = 126` is a sixth of it.  The same
kernels are 3.28x at one density.  `docs/HESSIAN_1.8.1.md` is the profile.

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

Every number above is a **closed-shell organic** system in a basis that stops at
d or f.  `docs/OPENSHELL_TMC.md` is what happened when the same modules were
pointed at open-shell **transition-metal complexes** -- 83 % of a 28 242-entry
CSD-derived set is open-shell, and def2-SVP already reaches an f shell on the
metal.  Two entry points that an open-shell ground state takes had no
replacement reaching them and a third declined the case outright, so the XC
half of the SCF, the XC half of the gradient and the whole two-electron half of
the gradient were running GPU4PySCF's own code with every module loaded.  They
are covered now, on one A100 with one GPU visible to the process:

| UKS, B3LYP/def2-SVP, level-3 grids | GPU4PySCF 1.8.1 | + Electrolites | |
|---|---|---|---|
| 21-atom Ni dithiolene (doublet, 327 AO), SCF | 101.9 s | **22.0 s** | **4.64x** |
| ↳ its gradient | 7.84 s | **3.05 s** | **2.57x** |
| 60-atom Cu macrocycle (quartet, 515 AO), SCF | 72.6 s | **25.6 s** | **2.84x** |
| ↳ its gradient | 9.60 s | **4.94 s** | **1.94x** |

The balance inside those runs is not the organic one: the **exchange build is
83-88 % of a UKS SCF** on these systems and the XC build 3-5 %, where on the
clusters above the XC build is a third of it.  So `nr_uks`, the module this
round was written for, is the weakest row in the table at 1.17x-1.38x -- there
is less idle device for it to fill -- while the XC *gradient* term, which was
inert for a namespace reason, is the strongest at 4.5x-10.5x.
`docs/OPENSHELL_TMC.md` §5 has the stage-by-stage table, what is still not
covered, and the corrections the audit made to this README.
`benchmarks/coverage_probe.py` is the tool that made the gaps visible, and it
is worth running before believing any coverage claim here -- as is reading §5's
account of the two measurement setups that produced confident wrong numbers
first.

The round after that took the three **VV10** entries off that list, which are
the ones a wB97M-V, wB97X-V, B97M-V or VV10 calculation of an open-shell
complex meets: the fused XC-and-VV10 grid pass of the SCF, the VV10 term of
the gradient, and the VV10 **response** term -- which on an unrestricted
reference is on the TDDFT path (`tdscf/uhf.py` does not set `exclude_nlc`)
where on a restricted one it is not.  Two of the three gaps were entry points
bound under another name rather than missing kernels.
`docs/VV10_OPENSHELL.md` has the round and four corrections to what
`docs/OPENSHELL_TMC.md` §4 says about the work.  Its measurements, on eight
systems spanning 3 to 60 atoms and both chemical classes: **1.47x-1.63x** on
the response term (a path that was 1.00x by construction), **2.7x-3.6x** on
the gradient term's grid passes -- 1.17x-1.70x on the term as a whole, the
rest of which is the `O(n_grid^2)` double sum -- and, for the SCF's fused
pass, a saving that is linear in the grid and independent of the functional
where `fastxc`'s grouped `nr_uks` saves latency that is neither, so the two
compete rather than compose and §4b says which wins where.

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
reaches every class and pays for all of them — and every def2 set on a
transition metal does, because def2-SVP already puts an f shell on a 3d metal.
`ELECTROLITES_FULL_MODULE=1` compiles everything, which is the ablation.

For a single job that JIT is part of the wall clock.  For a dataset run it
should be the cost of the first molecule and of no other, and
`electrolites.warmup` makes that a step rather than a hope: it compiles exactly
what a given basis will ask for, in worker processes, into a cache directory a
whole cluster can share.

```bash
python -m electrolites.warmup --basis def2-tzvp --elements Cu,Cl,C,H,N,O \
    --spin 2 --jobs 8 --cupy-cache /shared/cupy \
    --generated-dir /shared/electrolites/cuda
export CUPY_CACHE_DIR=/shared/cupy ELECTROLITES_GENERATED_DIR=/shared/electrolites/cuda
```

What it compiles is a property of the basis and the elements, not of the
geometry, so one atom of each element answers the question exactly;
`--xyz` takes a real geometry instead if one is at hand.
`benchmarks/warm_start.py` measures what it takes off a cold first run.

Two optional libraries are the exception, both because they need relocatable
device code and so cannot go through NVRTC: a faster copy of GPU4PySCF's
*general* exchange kernel, and the Hessian's second-derivative ERI kernels
(`fastejk2`).  Each is built with `nvcc` on first use — about 25 s — and cached
in `~/.cache/electrolites/sm_<arch>/<source hash>/`.  If `nvcc` is not
installed, Electrolites says so once and that piece falls back to GPU4PySCF —
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
| `fastj` | `scf.j_engine._VHFOpt.get_j` | **every `(lij\|lkl)` class an spdf basis reaches** -- the 12 GPU4PySCF unrolls, lifted, and the 17 it does not, written from the class.  Also the **multi-density** build that CPHF, TDDFT and polarizability ask for, four densities to a pass, for every class through `(4,4)`.  A **g** shell is past the written set and falls through -- 672 class blocks of a def2-TZVPP build (`docs/OPENSHELL_TMC.md` §4) |
| `fastxc` | `dft.numint.NumInt.nr_rks` **and** `NumInt.nr_uks` | LDA and GGA, so hybrids too -- and, on the unrestricted side only, meta-GGA.  The unrestricted build is the one an open-shell ground state takes, which is 83 % of a transition-metal set (`docs/OPENSHELL_TMC.md`); `dft/uks.py` calls `nr_uks`, so patching `nr_rks` never reached it |
| `fastnlc` | `dft.numint._vv10nlc` | the VV10 / VV10-family `O(n_grid^2)` double sum |
| `fastrsh` | `dft.rks.RKS.get_veff` | range-separated hybrids on GPU4PySCF **1.7.x only**: the two exchange builds fused into one. 1.8.0 does that fusion itself, so this module stands down there |
| `fastxcnlc` | `dft.rks.RKS.get_veff` **and `dft.uks.UKS.get_veff`** | the XC potential and the VV10 non-local correlation from **one** sweep of the grid instead of two, on 1.8.x (on 1.7.x `fastrsh` drives it). LDA falls back; GGA and meta-GGA are covered, so wB97M-V is. Restricted or unrestricted -- `dft/uks.py` binds its own `get_veff`, so patching `dft.rks`'s left an open-shell wB97M-V walking the grid four times per iteration (`docs/VV10_OPENSHELL.md`) |
| `fast1e` | `gto.int3c1e.get_int3c1e{,_density_contracted,_charge_contracted}` | the three-centre one-electron integrals -- **PCM, QM/MM, ESP, CHELPG** -- every angular-momentum class up to g, contracted with a density, with charges, or neither |
| `fastejk` | `grad.rhf._jk_energy_per_atom` | the per-atom derivative of `j*J - k*K`, every class an spdf basis reaches, full-range and long-range, for a **restricted or an unrestricted** reference: the exchange contraction runs over the density matrices and the Coulomb one over their sum.  A **g** shell falls through here too, and more of it -- 26 % of the class blocks of a def2-TZVPP gradient |
| `fastgrad` | `grad.rks.Gradients.energy_ee` | range-separated-hybrid gradients: the long-range operator in place of GPU4PySCF's short-range one |
| `fastxcgrad` | `grad.rks.get_exc`, `get_nlc_exc`, **and `grad.uks.get_exc`, `get_nlc_exc`** | the XC (LDA, GGA **and** meta-GGA) and VV10 grid contractions of the gradient, restricted or unrestricted.  `grad/uks.py` binds its own copies at import, so the restricted patch never reached an open-shell gradient.  The VV10 term takes one or two density matrices and the second case needs no spin index: VV10 is a functional of the total density, so `D_a + D_b` is the whole change |
| `fastgradh` | `grad.rhf.GradientsBase.get_hcore` | the nuclear-attraction part of the one-electron derivative, on the GPU.  **On 1.8.1 a gradient does not call this**: `grad_elec` goes through `_hcore_energy`, and `get_hcore` is reached only from `get_grad_hcore`, i.e. a Hessian's CPHF right-hand side |
| `fastxchess` | `hessian.rks._get_vxc_deriv1`, `_get_vxc_deriv2` **and the `hessian.uks` pair of the same names** | the two XC terms of an analytic **Hessian**: the same arithmetic in the masked AO space, with the atoms that have no basis function on a grid block skipped -- their contribution is identically zero -- and the atom dimension batched instead of looped in Python.  GGA, **restricted or unrestricted**.  `hessian/uks.py` defines its own copies rather than importing the restricted ones, so the restricted patch left every open-shell Hessian on GPU4PySCF's code; the unrestricted pair shares one AO evaluation, one mask, one atom list and one `d rho / d R` between the spin channels, which is why upstream pays **2.38x** for the second channel and this module **1.96x** on the same molecule (`docs/OPENSHELL_HESSIAN.md`) |
| `fastxcfxc` | `dft.numint.nr_rks_fxc` **and `nr_uks_fxc`** (the module attribute **and** the `NumInt` one, for both) | the XC response kernel -- the grid half of **every response property**: CPKS/polarizability, TDA/TDDFT, NMR, IR/Raman, C6.  One GEMM per grid block instead of one per right-hand side, three custom kernels in place of ten GPU4PySCF launches per right-hand side, and one symmetrisation per call instead of an extra matrix product per block.  LDA, GGA and meta-GGA, **restricted or unrestricted**.  `fxc` is a 2x2 block in the spin index, so the unrestricted contraction runs over the pair (spin, component) and has the one kernel of its own that the round needed; interleaving the two spins' densities lets the other three serve both cases unchanged.  On an **unrestricted** reference this is also the Hessian's CPHF grid half -- there is no `nr_uks_fxc_mo`, so `hessian/uks.py` calls this function (`docs/OPENSHELL_FXC.md`) |
| `fastejk2` | `hessian.rhf._ip2_quartets_scheme` and its `type3` twin -- the block shape of the **second-derivative ERI** kernels, which GPU4PySCF derives from the shared-memory budget alone and never measured | **1.13x** on that stage of an analytic Hessian for two lines of Python.  The module also carries a lift of the kernels themselves (every fp64 division out of the primitive and Rys-root loops, a register cap), correct to 1e-13 but measured at 1.02x and 0.58x respectively -- so that half is **off unless `FASTEJK2_KERNELS=1`**.  `docs/README_fastejk2.md` says why, and why the catalogue's B3 entry expects the opposite |
| `fastjk` | `scf.jk._VHFOpt.get_jk` | the **fused** Coulomb-and-exchange build -- the two-electron half of **every response property**, which is a third code path the SCF does not take and neither `fastj` nor `fastk` is on: 61-82 % of a polarizability or a TDDFT run.  Every angular-momentum class an spdf basis reaches, `hermi` 0 and 1, any number of densities, J and K accumulated from one pass over the Rys integrals |
| `fastfnlc` | `hessian.rks.nr_rks_fnlc_mo` **and the `hessian.uks` binding of the same name** | the **VV10 non-local-correlation response**, which is 80 % of a wB97M-V polarizability: the ground-state half of it -- including an `O(n_grid^2)` double sum -- computed once per coupled-perturbed solve instead of once per iteration, and the potential contraction batched over the perturbed densities.  Restricted or unrestricted: `V_munu` is spin-free, so the unrestricted case is the same body on the summed density with an MO-basis answer projected once per spin.  Reaches CPKS, NMR, C6 and the analytic Hessian -- and, on an **unrestricted** reference only, TDA/TDDFT: `tdscf/rhf.py` sets `exclude_nlc = True` and `tdscf/uhf.py` sets nothing, so an open-shell TDDFT pays this term and a closed-shell one does not (`docs/VV10_OPENSHELL.md` §1) |

**Limits, all of which fall through rather than fail.** An **open-shell ground
state** is now covered through the SCF and the gradient: `fastk`, `fastj`,
`fastxc`, `fastnlc`, `fastxcnlc` and `fastejk` all serve it, and `fastxcgrad`
serves both the XC and the VV10 half of the gradient.  On the **response**
side the VV10 term is covered too (`fastfnlc`, through the `hessian.uks`
binding that reaches it), which is what an open-shell TDDFT with wB97M-V
spends most of its time in, and the **analytic Hessian's two XC terms** are
covered on both spin cases (`fastxchess`, through the `hessian.uks` pair).
The **XC response kernel** is covered on both spin cases too
(`fastxcfxc`, `nr_uks_fxc`), which is the grid half of every open-shell
response property *and* -- since there is no `nr_uks_fxc_mo` -- of an
unrestricted Hessian's CPHF solve.  What is left of
`docs/OPENSHELL_TMC.md` §4 that is not a basis-set limit is the VV10 Hessian,
and of its three basis-set limits two are now taken as well: the **bra
primitive-pair bound** no longer fires on any def2 set up to def2-QZVP, and the
first-run JIT is now something `electrolites.warmup` fills ahead of a run
(`docs/PRIM_PAIR_AND_JIT.md`).  **g shells in `fastj` and `fastejk`** are the
one that is open.
`docs/OPENSHELL_TMC.md` has the audit that produced that list,
`docs/VV10_OPENSHELL.md` the three VV10 entries of it that are now taken,
`docs/OPENSHELL_HESSIAN.md` the Hessian XC terms and
`docs/OPENSHELL_FXC.md` the response kernel, what each gap costs, and the correction to what this paragraph used to
claim -- `fastk` never declined an unrestricted density, and
`benchmarks/coverage_probe.py` is how to check a claim of this kind rather than
believe one.  (`fastk` and `fastjk` also take **any** number of *perturbed*
densities, which is a different thing: the density loop is inside the shell
quartet, so a CPHF, TDDFT or polarizability call pays for the integrals once.)
`fastj` takes any number of densities, so it also covers the Coulomb
build inside a CPHF or Davidson solve; what it declines there is the classes
past `(4,4)` and the long-range operator. `fastnlc` covers UKS, because `nr_nlc_vxc` sums the two densities
before the double sum. `fastxc` hands **meta-GGA** back in `nr_rks` -- where
`fastxcnlc` reaches it instead -- and covers it in `nr_uks`, where nothing
else would. `fastxcgrad` covers it in the gradient.  `fastxcnlc`'s fusion
needs the XC and NLC grids to be the same *set* of points and declines
otherwise; `stats()` says whether it fired, because nothing numerical can.
`fastxchess` covers GGA, restricted and unrestricted, and declines
LDA, meta-GGA, the grid-response path and the VV10 Hessian.  `fastxcfxc` covers
LDA, GGA and meta-GGA, restricted and unrestricted.  It declines the
**restricted** `mo1` path -- nothing on 1.8.1 was found to reach it, since a
restricted Hessian's CPHF is served by a different function -- and covers the
**unrestricted** one, which is an open-shell Hessian's CPHF.  `fastfnlc`
covers a restricted and an unrestricted NLC response; its `return_in_mo=True`
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
python tests/test_ejk_ip2.py               # 5 cases: the second-derivative
                                           # ERIs and the whole analytic
                                           # Hessian, our kernels against
                                           # GPU4PySCF's under one driver
python tests/test_uxc.py                   # 17 cases: the unrestricted XC
                                           # build against GPU4PySCF's, on
                                           # transition-metal complexes
python tests/test_unlc.py                  # 8 cases: the unrestricted fused
                                           # XC+VV10 grid pass and the VV10
                                           # gradient term at two densities
python tests/test_ufnlc.py                 # 7 cases: the unrestricted VV10
                                           # response term, and a UKS TDA run
python tests/test_uxcgrad.py               # 10 cases: the unrestricted XC
                                           # gradient term, and the whole
                                           # gradient, ours against theirs
python tests/test_patch_aliases.py         # no GPU: does every reference to a
                                           # patched function see the patch, or
                                           # did some module bind its own copy
                                           # at import time
python tests/test_prim_pair.py             # 10 cases: the wide bra
                                           # primitive-pair module against the
                                           # default one and against
                                           # GPU4PySCF, from 6-31G* (nothing
                                           # to widen) to def2-QZVP
python tests/test_jk.py                    # 6 molecule/basis combinations x
                                           # hermi 0,1 x n_dm 1..8 x three
                                           # density shapes: the multi-density
                                           # exchange build and the fused J+K
                                           # build against GPU4PySCF's
```

They need a GPU and take a few minutes (`test_patch_aliases.py` does not: it
only imports). The cases span six basis sets to
cc-pVQZ, twelve functionals including three meta-GGAs, `omega < 0`, third-row
elements, a cation and an open-shell radical -- and, since
`docs/OPENSHELL_TMC.md`, six open-shell transition-metal complexes from
BOS-TMC-28242 (`benchmarks/molecules/tmc_cases.txt`) with 3d, 4d and 5d metals,
multiplicities to 6, an ECP, and def2-TZVPP, which reaches g.

## Benchmarks

`benchmarks/scf_anatomy.py` says where an SCF wall clock goes, method by
method, with a synchronisation point on both sides of each -- use it before
concluding that anything outside J, K and XC matters.  `--spin` makes it a UKS
run and adds the gradient's stages.
`benchmarks/coverage_probe.py` says which modules are on the code path of a
given calculation at all, and how many angular-momentum class blocks each of
the two-electron ones answered -- use it before believing any statement in
this README about what a module covers, and in particular before assuming that
a module which is loaded is doing anything.
`benchmarks/ejk_ndm.py` times the gradient's two-electron derivative at one
density and at two, both implementations in one process.
`benchmarks/perclass_j.py` times the Coulomb build one angular-momentum class
at a time and checks the J matrix against GPU4PySCF's;
`benchmarks/ndm_j.py` says how the Coulomb build's cost grows with the number
of density matrices, fits the per-pass and per-density parts of it, and A/Bs
the density-block sizes -- use it before assuming that a cap GPU4PySCF pays for
is still worth removing once the kernel under it has been rewritten
(`docs/ROUND3_B4A.md`);
`benchmarks/sweep_j_high.py` searches the launch configuration of the written
classes; `benchmarks/compile_time.py` reports the first-run NVRTC cost with
and without the module split, and `benchmarks/warm_start.py` what warming the
cache ahead of the run takes off it; `benchmarks/prim_pair.py` times the
exchange build and the gradient's two-electron derivative with and without the
wide bra primitive-pair module, in one process, and says which class blocks the
bound moves; `benchmarks/aoscreen_probe.py` measures how much
a density-aware AO screen could remove from the XC build.

`benchmarks/hessian_anatomy.py` does the same for an analytic Hessian, stage
by stage -- use it before taking a Hessian target from
`instructions/06`, whose Hessian profile is a 1.7.0 one and whose shares move
by a factor of six between 42 and 113 atoms (`docs/HESSIAN_1.8.1.md`).
`benchmarks/vxc_hess_probe.py` and `benchmarks/xchess_locality.py` say where
the XC Hessian's time goes and how much of its atom loop is arithmetic on
exact zeros; `benchmarks/xchess_bench.py` A/Bs the two terms on one converged
density.

`benchmarks/ndm_k.py` fits the exchange build as `R + n_dm*C` against both
implementations in one process -- use it before quoting a multi-density
speedup at an `n_dm` the caller does not use, because `C` is the same code in
both and the ratio runs from 3.28x to 1.18x over `n_dm` = 1 to 64
(`docs/ROUND4_B4B.md` §4).  `benchmarks/ndm_screen_probe.py` asks whether the
union-over-densities screening the catalogue's B4c objects to actually costs
anything, on the densities a real solve produces; `benchmarks/ndm_layout_probe.py`
measures the ceiling for transposing the density to `dm[(i,j),i_dm]`, which is
the part of `C` that is memory layout.  `benchmarks/ejk_ip2_probe.py` splits
the Hessian's second-derivative ERI stage by angular-momentum class and by
whether GPU4PySCF has an unrolled kernel for it -- use it before taking
catalogue B3, whose "same shape as B1" describes seven of the sixty-five
classes.

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
codegen/build_jk_kernels.sh /path/to/gpu4pyscf/lib/gvhf-rys/unrolled_rys_jk.cu
```

The second one is the same two generators with `--jk`: the fused build's
kernels differ from the exchange-only ones *only* in the contraction, and
`gen_kernels.assert_same_integrals` checks that claim against the two
GPU4PySCF files before emitting anything, so a release that changes one and
not the other fails the build rather than producing a kernel that computes
something else.

The written families need no such path, and normally need no command either --
they appear when they are first wanted.  To force it, or to fill a cache
ahead of time:

```bash
python -c 'from electrolites import _gen; print(_gen.ensure_all())'
```

`gen_j_high.py` needs no GPU4PySCF source at all -- it writes each
McMurchie-Davidson class from `(lij, lkl)` -- and re-tuning it for another
card is `benchmarks/sweep_j_high.py`, whose winners go into that file's `CFG`.

`codegen/gen_ejk_ip2.py` is a fourth kind: it does not write or lift a kernel
but *transforms* GPU4PySCF's second-derivative ERI sources in place, asserting
the count of every substitution and then asserting that no double-precision
division is left in the primitive or Rys-root loops.  Its output is tracked
because it is compiled by `nvcc`, not NVRTC:

```bash
codegen/build_ejk_ip2.sh /path/to/gpu4pyscf/lib/gvhf-rys
```

The launch tables next to the generated sources (`*_launch.json`) are tuned per
angular-momentum class and were measured on an A100; on another card they are a
starting point, not an optimum.

## Environment switches

| variable | effect |
|---|---|
| `ELECTROLITES_NVCC`, `ELECTROLITES_ARCH` | compiler and compute capability for the one `nvcc`-built library |
| `ELECTROLITES_LIBMYKG` | use a prebuilt copy of it instead of building |
| `ELECTROLITES_CACHE` | where to cache the build (default `~/.cache/electrolites`) |
| `ELECTROLITES_GENERATED_DIR` | a pre-generated tree of the written CUDA, used as is — for an offline or read-only install, or one copy shared across a cluster |
| `ELECTROLITES_MAX_PRIM_PAIR` | largest bra primitive pair a second kernel module will be compiled for (default 100, which covers def2-QZVP). **0 is the ablation**: every class block whose bra pair is longer than 36 falls through, which is what this repository did before `docs/PRIM_PAIR_AND_JIT.md` measured it |
| `FASTK_NO_GENERAL`, `FASTK_NO_K2`, `FASTK_NO_HIGH`, `FASTK_NO_OMEGA`, `FASTK_NO_FUSED` | hand the corresponding class group back to GPU4PySCF (ablations) |
| `FASTJ_NO_MDM`, `FASTJ_MDM_BLOCKS` | hand every multi-density Coulomb build back to GPU4PySCF; restrict which density-block sizes are used (the `docs/ROUND3_B4A.md` ablation) |
| `FASTNLC_MODE` | `mixed` (default) or `fp64` for the VV10 double sum |
| `FASTJ_TIME`, `FASTXCGRAD_MODE` | per-call timing split; XC-gradient block strategy |
| `FASTFNLC_CACHE`, `FASTFNLC_BATCH` | hand the two halves of the VV10 response change back (the ablations) |
| `FASTFNLC_BUDGET` | `fastnlc`'s error-budgeted grid trim applied to the VV10 *response* term's two `O(n_grid^2)` sums. **Off by default** -- `docs/VV10_OPENSHELL.md` §4d has what it is worth and what it perturbs |
| `FASTXCNLC_OFF` | hand every `get_veff` back, which is the ablation for the fused XC-and-VV10 grid pass. Worth running per system: the fusion and `fastxc`'s grouped `nr_rks`/`nr_uks` remove the same latency from opposite directions and do not compose (`docs/VV10_OPENSHELL.md` §4b) |
| `FASTK_NO_NDM` | hand every multi-density exchange build back, which is what `fastk` did before the B4b round (the `docs/ROUND4_B4B.md` ablation) |
| `FASTJK_OFF`, `FASTJK_NO_HIGH`, `FASTJK_NO_LIFTED` | hand the fused build back whole, or only its written / only its lifted classes |
| `FASTEJK2_OFF`, `FASTEJK2_NO_SCHEME`, `FASTEJK2_SHIFT` | hand the Hessian's second-derivative ERI stage back whole, or only its launch-scheme half; and the quartets-per-block halving (1 is what was measured) |
| `FASTEJK2_KERNELS` | **opt in** to the lifted ip2 kernels, which are correct and measured not to pay -- the one switch here that is off by default |
| `ELECTROLITES_LIBMYEJK2` | use a prebuilt copy of the second-derivative ERI library instead of building |
| `FASTXCFXC_MODE`, `FASTXCFXC_TIME` | `blocked` (default), `nokernel` or `grouped` for the XC response kernel -- the last is catalogue B4d as written and is the slowest of the three; and its per-stage timing split |

## License

Apache-2.0. Substantial parts of the CUDA are derived from GPU4PySCF, also
Apache-2.0 — see `NOTICE` for exactly which files and how.
