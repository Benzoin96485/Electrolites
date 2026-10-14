# What PySCF 2.13 computes that GPU4PySCF 1.8.1 does not run on the GPU — the next phase for Electrolites

**Who this is for**: whoever plans the next phase of this repository, and
whoever picks greenfield entries for the LLM-GPU-QM benchmark task suite.

**What it is**: the companion to `docs/KERNEL_TARGETS_1.8.1.md` ("KT" below).
KT catalogued what is *slow* in the features GPU4PySCF 1.8.1 already runs, and
its 31 ranked entries have since been built, measured or rejected (KT §13).
This document catalogues what GPU4PySCF does **not run on the GPU at all**.
That covers every feature of PySCF 2.13.0, and of its official
`pyscf-properties` extension, that GPU4PySCF 1.8.1 lacks, stubs out, runs only
after `to_cpu()`, or runs and gets wrong. It ranks them as targets for new GPU
implementations in this repository. A gap that was already a KT entry (B19,
C6, C13, I6, I7, E3, …) is cited rather than repeated, and carried forward
with whatever is new.

**Scope**: all 35 subpackages of PySCF 2.13.0, molecular and periodic (§2
lists them), plus `pyscf-properties`. Excluded:

- the `md/` and `geomopt/` drivers, which call methods rather than being
  methods;
- `symm/` and `data/` as libraries;
- checkpoint I/O;
- the `*_slow.py` reference implementations;
- the MPI variants.

**What this document is not**: no GPU code was written. The survey has three
parts:

1. a reading of the two source trees;
2. a runnability probe, `benchmarks/feature_gap_probe.py`, added with this
   document, which asks a GPU4PySCF object for each missing feature on
   molecules of two or three atoms and prints what happens, including two
   finite-difference checks;
3. one CPU grid count (S1).

Nothing was timed. Every timing quoted is one already measured in this
repository's `docs/`, with its pointer. **§5's ranking is therefore a
judgment** on demand, GPU fit, reuse and oracle, not on measured shares. §8
lists the CPU baselines and anatomies owed before any entry is built, and
taking them is the first task of the phase (§9).

---

## 0. How to read this

The tag conventions are KT §0's, which are 06 §0's:

- `[V]` — verified against a primary source: the line was read in the tree
  named, the probe was run, or the number was measured in this repository and
  is written in the `docs/` file cited.
- `[?]` — unverified or inferred.  **Never load-bearing.**
- `[X]` — contradicts a claim in circulation (here: in KT or in this
  repository's code); the corrected statement follows.

**Paths**: `P:` is PySCF 2.13.0 as installed in `barw-dev`
(`…/envs/barw-dev/lib/python3.13/site-packages/pyscf/`). `G:` is the
GPU4PySCF 1.8.1 source tree (`/pscratch/sd/w/weiliang/g4psrc/g4p-1.8.1/gpu4pyscf/`,
git `5b284c2`). `electrolites/` is this repository. `KT:B19` is entry B19 of
`KERNEL_TARGETS_1.8.1.md`.

**Status vocabulary** (the "GPU today" column):

| Status | Meaning |
|---|---|
| **absent** | no GPU module, class or method; asking for it raises |
| **stub** | the name exists but is `NotImplemented`, `pass`, or raises |
| **host** | works only after `mf.to_cpu()`, and then the *whole method* runs on the CPU, including the J/K, XC and response builds that are fast on the GPU here |
| **partial** | the named part runs on the GPU and the rest does not |
| **WRONG** | runs on the GPU without an error and returns a wrong number |
| **ahead** | GPU4PySCF has it and PySCF does not |

KT's three rules carry over: a share is not a speedup, a patch target is not a
code path, and a share is not size-independent.  Four more apply to work that
*adds* a feature rather than replacing a kernel:

1. **Two baselines, never pooled.** The baseline depends on what the GPU
   feature replaces:
   - **No GPU path today** ("absent", "stub", "host"): the baseline is
     **PySCF 2.13 on the CPU**, on a whole Perlmutter CPU node (2 × 64-core
     EPYC 7763, `OMP_NUM_THREADS=128`). Every GPU-against-CPU ratio states
     that core count in the same sentence. The LLM-GPU-QM landscape review
     (`02.review_qm_landscape.md`, §"three papers, three different CPU core
     counts") found that published GPU-against-CPU headlines cannot be
     compared because none of them does.
   - **An alternative route to something already on the GPU** (COSX against
     exact K, DF-CASSCF integrals against direct ones): the baseline is
     **the best existing GPU path under `patch_all()`**.

   Neither kind is pooled with KT's GPU-against-GPU ratios (06's B10 rule).
2. **The oracle comes before the speed.** Every new feature is checked against
   the PySCF CPU implementation (§6) on two systems before any timing is
   quoted.  An approximate method (COSX, reduced precision) is opt-in,
   reports its error next to its time, and is never a default.
3. **Generalisation is an acceptance criterion**, as for every round here: a
   second size *and* a second class, i.e. a closed-shell organic molecule and
   an open-shell transition-metal complex from
   `benchmarks/molecules/tmc_cases.txt`.  That set is **all open-shell** (six
   complexes, doublet to sextet), so a restricted-only method (RCCSD(T),
   RMP2 gradients) needs a closed-shell complex added (a low-spin d⁶ or a d¹⁰
   one) before it can meet this rule.
4. **A declined case goes back to the CPU loudly.** When a new module cannot
   handle an input, it hands over to PySCF through `to_cpu()` with a logged
   warning. It never does so silently, and never by computing something else;
   §3.2 has two places where GPU4PySCF 1.8.1 does exactly that today.

**IDs** use new letters so they do not collide with KT's A–K:

| Letter | Area |
|---|---|
| **S** | SCF- and DFT-level |
| **M** | multireference |
| **P** | single-reference correlated and Green's-function |
| **O** | molecular properties |
| **V** | solvation and embedding |
| **X** | periodic |
| **T** | tools |
| **Z** | infrastructure and correctness |

**Diff.** is easy / mod / hard / greenfield and **Contam.** is high / moderate
/ low, both as in KT, and Contam. is `[?]` everywhere. **Demand** is high /
moderate / low and is also `[?]` everywhere. It is a judgment from what the
method is used for and from this repository's own workload: 83 % of the
28 242-entry CSD-derived transition-metal set behind `tmc_cases.txt` is
open-shell (README). It is not a survey of users.

---

## 1. What was surveyed

| Item | Value |
|---|---|
| PySCF | **2.13.0** in `barw-dev`; `pyscf-dispersion` 1.5.0 alongside; `mcpdft` (merged from pyscf-forge) is part of the 2.13 tree `[V]` |
| pyscf-properties | **not installed** in `barw-dev`; the module list (§4 O) was read from `github.com/pyscf/properties` on 2026-10-13, so line numbers there are the repository's, not an installed copy's |
| GPU4PySCF | 1.8.1 source `5b284c2` "1.8.1 release"; the installed wheel matches it on every file the probes touched `[V]` |
| Electrolites | this repository at `f218991` (2026-10-13), every module in `electrolites/` and every file in `docs/` |
| Method | Four read-only passes, one per area: (1) mean field, DFT and tools; (2) multireference and `ao2mo`; (3) correlated and Green's-function methods; (4) solvation, properties and PBC. Each pass diffed P against G file by file and read both trees. Then `benchmarks/feature_gap_probe.py` (§3) ran on the login node's shared A100 80 GB PCIe with water, H₂O⁺, N₂ and a two-atom H₂ cell. The probe results in §3 are this run's. Rows marked "survey probe" in §4 come from the passes' own scratch probes, which are not in the repository |
| Measurements | **none timed**; one CPU grid count (S1) |
| Environment for the probe | `module load cudatoolkit/12.9` before `conda activate barw-dev`. Under the default `cudatoolkit/13.2`, `import gpu4pyscf` fails (`libcusolver.so.11` missing) `[V]` |

---

## 2. The coverage map

Package by package, PySCF 2.13.0 against GPU4PySCF 1.8.1.  Electrolites adds
no feature to this table: it replaces kernels inside features GPU4PySCF
already has.  The two rows where it changes the picture are noted.

| PySCF package | what PySCF provides | GPU4PySCF 1.8.1 | entries |
|---|---|---|---|
| `scf` | RHF/UHF/ROHF/GHF/DHF, stability analysis, addons, CPHF | RHF/UHF/ROHF/GHF SCF, smearing, Newton (RHF/UHF/ROHF). `stability` **stub** (G `scf/hf.py:865`, `uhf.py:324`, `rohf.py:100`, `ghf.py:147`). DHF **absent**. ROHF derivatives stub or **WRONG** | S2, S3, S4, S8, Z1 |
| `dft` | RKS/UKS/ROKS/GKS/DKS, SAP/vSAP guesses, xcfun backend, symmetry-adapted KS | RKS/UKS/ROKS/GKS, DFT+U, CDFT. DKS **absent**. SAP **absent**. vSAP broken by a typo. `rks_symm` absent. An xcfun backend is silently ignored | S4, S8 |
| `sgx` | COSX (seminumerical exchange) energy and gradient | **absent** | S1 |
| `grad` | gradients of every SCF type, TDDFT, CC, CISD, MP2, CASCI/CASSCF/SA-CASSCF, MC-PDFT, DHF | RHF/RKS/UHF/UKS, with DF, TDDFT, +U, CDFT and dispersion. ROHF/ROKS **stub** (asserts). Every correlated gradient **absent** | S3, M6, P3, P4 |
| `hessian` | RHF/RKS/UHF/UKS, `thermo` | the same four. `thermo` works on GPU output. ROHF/ROKS **WRONG** | Z1 |
| `tdscf` | TDA/TDDFT for R/U/G HF and KS, DHF/DKS | R/U HF and KS, plus spin-flip and ris. GHF/GKS/DHF/DKS **absent**. ROHF TDA raises | S6 |
| `x2c` | sfX2C1e SCF, gradient and Hessian; j-adapted X2C SCF, TDDFT and stability | sfX2C1e SCF and X2C1E GSCF. Gradient **stub** (G `x2c/sfx2c1e.py:77-78`). j-adapted classes **absent** | S5 |
| `lo` | Boys, Pipek–Mezey, Edmiston–Ruedenberg, IBO/IAO, NAO, meta-Löwdin, VVO | **absent**; every entry point raises on CuPy input | S7 |
| `soscf` | the generic CIAH solver (used by `lo`) | its own Newton for RHF/UHF/ROHF (ROHF raises) | — |
| `mcscf` | CASCI/CASSCF (+DF, unrestricted, state-average), AVAS/DMET-CAS/APC | **absent** | M2–M5, M9 |
| `fci` | `direct_spin0/1`, `direct_uhf`, `direct_nosym`, selected CI, RDMs, spin operators | **partial**: `contract_2e` only (G `fci/direct_spin1.py`, 192 lines). It breaks inside any CASCI above 400 determinants and inside every CASSCF (§3) | M1 |
| `mrpt` | SC-NEVPT2 (+DF) | **absent** | M7 |
| `mcpdft` | MC-PDFT, L-PDFT, MS/CMS/XMS-PDFT and their gradients | **absent** | M8 |
| `ao2mo` | general four-index transforms | **absent**. G `scf/int4c2e.py:191` builds a dense nao⁴ tensor that nothing outside tests calls; `electrolites/fastmp2.py` builds ovov | Z2 |
| `mp` | RMP2/UMP2/GMP2 and DF variants, RDMs, natural orbitals, gradients | RMP2 and DF-RMP2/DF-UMP2 energies (Electrolites: `fastmp2`, `fastdfmp2`). Conventional UMP2/GMP2 raise. RDMs and gradients **absent** | P3, P5 |
| `cc` | RCCSD/UCCSD/GCCSD, (T), Λ, RDMs, gradients, EOM, DF-CCSD, QCISD(T), CCSDT(Q) | RCCSD energy only (Electrolites: `fastccsd`, device-resident). `ccsd_t`, Λ, EOM, RDMs, gradients **stub** (G `cc/ccsd_incore.py:574-591`). UCCSD **absent**. DF-CCSD silently runs conventional | P1, P2, P6, P7 |
| `ci` | CISD, UCISD and gradients | **absent** | §5.4 |
| `gw` | G0W0 (AC, CD, exact), RPA, URPA | **absent** | P8, P9 |
| `adc` | RADC/UADC IP/EA/EE at (2), (2)-X, (3), CVS | **absent** | P10 |
| `agf2` | AGF2 | **absent** | §5.4 |
| `eph` | electron–phonon coupling | **absent** | P11 |
| `solvent` | PCM family, SMD, ddCOSMO, ddPCM, polarizable embedding, COSMO-RS file | PCM family and SMD, with gradients, Hessians and TDDFT including gradients and NACs. The TDDFT-PCM gradient is **ahead**: PySCF raises `NotImplementedError('PCM-TDDFT Gradients')` (P `solvent/_attach_solvent.py:795`) `[V]`. ddCOSMO/ddPCM/polarizable embedding **absent** | V1, V4, V5 |
| `qmmm` | point and Gaussian MM charges, SCF and gradient, PBC | point charges: SCF, gradient and TDA gradient. Gaussian charges assert. Hessian **WRONG** | V2, V3 |
| `pyscf-properties` | NMR shielding, spin–spin coupling, magnetizability, g-tensor, hyperfine, EFG, ZFS, NSR, rotational g, frequency-dependent and hyper-polarizability | RHF/RKS NMR shielding (with its GIAO two-electron term on the CPU), static RHF/RKS polarizability, RHF/RKS IR/Raman, EDA, C6. Nothing else | O1–O5 |
| `pbc` | SCF/DFT incl. ROKS/GKS, TDSCF, MP2, CC, CI, ADC, GW, EPH, stability, ECP, VV10, spin–orbit X2C | k-point SCF/DFT, GDF/FFTDF/multigrid, gradients and stress, +U, sfX2C. TDSCF/MP2/GW/CC/stability/ECP **absent**. GHF/KGHF empty stubs. VV10 broken at Γ and at k | X1–X7 |
| `tools` | cube files, Molden, FCIDUMP, WFN, `mo_mapping` | none of its own. Molden works; the rest raise on CuPy input | T1 |
| `df` | DF J/K, DF gradients and Hessians, including ROHF/ROKS and CASSCF/MC-PDFT gradients | DF J/K, gradients and Hessians for R/U HF and KS, plus TDDFT gradients and NACs; DF ROHF/ROKS gradients raise; the DF CASSCF and MC-PDFT gradients are **absent** | S3, M6 |
| `nac` | SA-CASSCF and MS-PDFT NACs | none of these, but TDDFT NACs, which PySCF lacks (**ahead**) | M6 |
| `md`, `geomopt`, `symm`, `dispersion`, `data`, `gto`, `lib`, `tddft` | drivers, libraries, data; `tddft` is an alias of `tdscf` | own `md`/`geomopt` drivers; dispersion re-exports PySCF's (KT:F1) | — |

**Reading**:

- **The large gaps are whole method families**, not single operators: all of
  multireference; coupled cluster beyond the RCCSD energy; GW/RPA/ADC;
  excited states and correlation under PBC.
- **The "host" route works for almost everything, at a price.** It moves the
  whole method to the CPU, including the parts this repository has already
  made fast on the GPU.
- **Two places run and are wrong**: the ROHF/ROKS Hessian and the QM/MM
  Hessian (§3.2). They come before any optimisation.

---

## 3. What a user hits today

`benchmarks/feature_gap_probe.py`, run as `python benchmarks/feature_gap_probe.py`
and again with `--patch all` (the same table: no Electrolites module provides
any of these features).  `[V]` for every row.

### 3.1 The four doors, and where each one fails

A user with a GPU4PySCF mean field `gmf` has four ways to ask for a PySCF
feature GPU4PySCF lacks:

| door | what happens | why |
|---|---|---|
| `mol.to_gpu().CCSD` / `.CASSCF` / `.MP2` / `.CISD` | `NotImplementedError` | G `gto/mole.py:449-451` raises for any key containing CI, CC, CAS or MP |
| `gmf.CASSCF`, `gmf.COSX`, `gmf.TDA` (GHF), `gmf.ddCOSMO` | `AttributeError` | G's `SCF` is a `StreamObject` (G `scf/hf.py:695`), not a `pyscf.scf.hf.SCF`, so the methods PySCF binds onto its own classes (P `cc/__init__.py:92`, `mcscf/mc1step.py:1332`) are not there |
| `pyscf.<method>(gmf)` | usually a CuPy implicit-conversion `TypeError`, sometimes an assertion | PySCF's drivers take NumPy arrays and PySCF classes (P `mcscf/mc_ao2mo.py:92`, `cc/ccsd.py:1522`, `sgx/sgx.py:76`, …) |
| `cpu_obj.to_gpu()` | `ModuleNotFoundError: gpu4pyscf.<path>` | P `lib/misc.py:1545` maps `pyscf.X.Y` to `gpu4pyscf.X.Y` and the module does not exist |

The only route that works is `gmf.to_cpu()`, after which the method runs
entirely on the CPU.

The probe, row by row (`G:` / `P:` give the innermost frame that raised):

| ID | call | result |
|---|---|---|
| S1 | `gmf.COSX()` | AttributeError |
| S1 | `pyscf.sgx.sgx_fit(gmf).kernel()` | AssertionError, P `sgx/sgx.py:76` |
| S2 | `gmf.stability()` | TypeError: `'NotImplementedType' object is not callable` |
| S2 | `pyscf.scf.stability.rhf_stability(gmf)` | TypeError (CuPy implicit conversion), P `lib/linalg_helper.py:1416` |
| S3 | GPU `ROHF(...).Gradients().kernel()` | AssertionError, G `grad/rhf.py:424` |
| S3 | GPU `ROHF(...).Hessian()` | **runs; WRONG** (§3.2) |
| S3 | GPU `ROHF(...).TDA().kernel()` | TypeError: `_gen_uhf_response() got an unexpected keyword argument 'singlet'`, G `tdscf/rhf.py:414` |
| S4 | `pyscf DHF(mol).to_gpu()` | ModuleNotFoundError `gpu4pyscf.scf.dhf` |
| S5 | GPU `RKS.sfx2c1e().Gradients().kernel()` | NotImplementedError, G `x2c/sfx2c1e.py:78` |
| S6 | GPU `GHF(...).TDA()` | AttributeError |
| S7 | `lo.Boys(mol, cupy_mo).kernel()`, `lo.ibo.ibo(mol, cupy_mo)` | TypeError (CuPy), P `lo/boys.py:123`, `lo/iao.py:91` |
| S8 | GPU RKS `init_guess='sap'` | AttributeError: no `init_guess_by_sap`, P `scf/hf.py:2022` |
| S8 | GPU RKS `init_guess='vsap'` | AttributeError: no `get_vsap`, P `dft/rks.py:531` |
| T1 | `tools.cubegen.density(mol, f, cupy_dm)` | AttributeError (`ctypes`), P `lib/numpy_helper.py:983` |
| M1 | `mol.to_gpu().CASSCF` | NotImplementedError, G `gto/mole.py:451` |
| M1 | `pyscf.mcscf.CASSCF(gmf, 6, 6).run()` | TypeError (CuPy), P `mcscf/mc_ao2mo.py:92` |
| M2 | CASCI(6,6) on N₂ (400 determinants) with `gpu4pyscf.fci.direct_spin1.FCI` | **runs, ΔE = 2.8e-14** — the GPU kernel is never called (§3.4) |
| M2 | CASCI(8,8) (4 900 determinants), the same solver | ValueError, G `fci/direct_spin1.py:153` |
| M2 | CASSCF(6,6), the same solver | TypeError: `contract_2e() missing … 'link_index'`, P `mcscf/mc1step.py:1121` |
| P1 | GPU `CCSD(gmf).run().ccsd_t()` | TypeError (`ccsd_t = NotImplemented`, G `cc/ccsd_incore.py:575`) |
| P1 | `pyscf.cc.CCSD(gmf).run()` | AssertionError, P `cc/ccsd.py:1522` |
| P2 | GPU `MP2(gmf).run().Gradients()` | NotImplementedError, G `mp/mp2.py:352` |
| P2 | GPU `MP2(GPU UHF)` | NotImplementedError, G `mp/__init__.py:10` |
| P3 | `dft.RKS(mol, xc='B2PLYP')`, CPU **or** GPU | KeyError: libxc has no such name, P `dft/libxc.py:632` |
| P4 | `pyscf.gw.GW(GPU PBE)` / `pyscf.gw.rpa.RPA(GPU PBE)` | TypeError, P `gw/gw_ac.py:776` / P `mp/mp2.py:210` |
| P5 | `pyscf.adc.ADC(gmf)` | TypeError (CuPy), P `adc/radc.py:365` |
| O1 | GPU UKS NMR shielding | ValueError, G `properties/shielding.py:34` |
| O2 | GPU frequency-dependent polarizability | AttributeError: no such entry point |
| V1 | GPU `RKS.ddCOSMO()` | AttributeError |
| V2 | GPU QM/MM Hessian | **runs; WRONG** (§3.2) |
| X1 | GPU Γ-point PBC `RKS.TDA()`; `pyscf.pbc.tdscf.TDA(GPU RKS)` | AttributeError (P `pbc/tdscf/__init__.py:53`) |
| X2 | `pyscf.pbc.mp.KMP2(GPU KRHF)` | TypeError (CuPy), P `pbc/mp/kmp2.py:452` |
| X3 | GPU Γ-point PBC RKS with a VV10 functional | TypeError: `'NotImplementedType' object is not callable`, G `pbc/dft/rks.py:98` |

The probe row IDs are its own grouping, not §4's entry IDs. In particular,
the probe's P2–P5 are §4's P3 (MP2 gradient), P4 (double hybrids), P8/P9
(GW, RPA) and P10 (ADC). The probe's X3 is §4's X6.

### 3.2 Silent wrong answers

These run on the GPU without an error and return the wrong number.  Each was
checked against finite differences of a gradient that is itself correct, with
a control row that passes. The probe's step is 1e-4 bohr at `conv_tol` 1e-12.

| | system | analytic GPU Hessian against FD | control |
|---|---|---|---|
| **ROHF/ROKS Hessian** | H₂O⁺ doublet, STO-3G, the O-z rows; FD of **PySCF CPU** ROHF gradients | **max error 0.509 Eh/bohr²** | GPU UHF Hessian on the same doublet: 4.2e-7 |
| **QM/MM Hessian** | water/6-31G/PBE with ±0.8 e point charges about 3 bohr away, the O-x rows; FD of the GPU QM/MM gradient | **max error 0.0646 Eh/bohr²** | the same with the charges scaled by 1e-3: 1.7e-4 |

- **ROHF/ROKS: cause.** `class ROHF(hf.RHF)` (G `scf/rohf.py:73`) inherits
  the `Hessian` that G `hessian/rhf.py:937` attaches to `scf.hf.RHF`, so the
  RHF formula is applied to an ROHF wavefunction. `nuc_grad_method` is set to
  `NotImplemented` (G `scf/rohf.py:103`), but `Gradients`, `Hessian` and
  `TDA` are not. PySCF's own CPU `ROHF.Hessian()` correctly raises.
- **ROHF/ROKS: DF.** The DF variants fail loudly, so only the direct path is
  silent (survey probe).
- **QM/MM: cause.** Nothing in G `hessian/`, `grad/td*`, `tdscf/` or
  `solvent/` references `mm_mol` (survey grep). The Hessian therefore omits
  the MM charges' second-derivative terms.
- **QM/MM: no CPU oracle.** PySCF has no QM/MM Hessian either.
- **QM/MM: the TDA gradient is fine.** It agrees with FD: −0.2434100 against
  −0.2434099 (survey probe).

Both are guarded in Z1 and implemented in S3 / V2.

### 3.3 Other defects found on the way (not targets)

Each is small.  They belong in `docs/UPSTREAM_BUGS.md` and, where they are
upstream's, in an upstream report.

| # | defect | evidence |
|---|---|---|
| D1 | vSAP guess cannot run. RKS defines no `get_vsap`, and UKS binds it under the misspelled name `get_vasp` | G `dft/uks.py:107`; probe S8 |
| D2 | `ROHF.newton().run()` raises a CuPy implicit-conversion TypeError | G `scf/rohf.py:46` (survey probe) |
| D3 | `CCSDBase.to_cpu()` is broken. It calls `_scf.to_cpu()` on a CPU object, and would import the nonexistent `pyscf.cc.ccsd_incore` | G `cc/ccsd_incore.py:597-599` |
| D4 | GPU CCSD on a UHF reference dies with a bare `IndexError` instead of "restricted only" | P `mp/mp2.py:737` (survey probe) |
| D5 | GPU CCSD on a **DF** reference runs **conventional** CCSD on DF orbitals, with a logged warning; the result is 5.9e-7 Eh from conventional CPU CCSD (not a defect on reflection: §9 status) | G `cc/ccsd_incore.py:612-613` (survey probe) |
| D6 | `gmf.MP2` exists only after `import gpu4pyscf.mp`, and only on RHF | G `mp/mp2.py:375` |
| D7 | DF-MP2 with `frozen` and `with_t2=True` asserts. t2 is allocated with the unfrozen `nocc`/`nmo` | G `mp/dfmp2.py:66-69`, `dfmp2_addons.py:1154` (survey probe) |
| D8 | `mcscf.CASSCF(GPU UHF)` **silently** builds an RHF-type CASSCF; `mcscf.UCASSCF(GPU UHF)` dies on `to_uhf = NotImplemented` | P `mcscf/__init__.py:188,256`; G `scf/hf.py:869` (survey probe) |
| D9 | GPU PBC `GHF`/`KGHF` are `class GHF: pass` | G `pbc/scf/ghf.py:23-24`, `kghf.py:23-24` |
| D10 | latent argument-order bug: the PBC response's singlet branch passes `singlet` into `nr_rks_fxc`'s `fxc` slot and `fxc` into `kpts` | G `pbc/dft/rks.py:323-324` (also `krks.py:253-254`) against the signature at G `pbc/dft/numint.py:505` |
| D11 | `hf_symm` is an alias that silently drops point-group symmetry; `RKS(symmetric mol).to_gpu()` raises `ModuleNotFoundError gpu4pyscf.dft.rks_symm` | G `scf/hf_symm.py` (survey probe) |
| D12 | setting `_numint.libxc = xcfun` is silently ignored; the GPU always resolves libxc IDs | G `dft/numint.py:1903-1914` (survey probe) |
| D13 | `[X]` this repository's `electrolites/fastdfmp2.py:24` docstring says `with_t2` is "on by default"; GPU4PySCF's default is `False` | G `mp/dfmp2_addons.py:57` |

### 3.4 Corrections to KT

| KT | claim | correction |
|---|---|---|
| §12.4, §5.1 runnability walls | "1.8.1's GPU FCI solver cannot run inside a CASCI at all" | `[X]` **Too strong.** It runs and is exact (ΔE 2.8e-14) while the CAS has ≤ 400 determinants: P `fci/direct_spin1.py:682` then diagonalises the P-space directly and never calls `contract_2e`. Above 400, or with `davidson_only=True`, the Davidson hands it a 1-D vector and it fails (G `fci/direct_spin1.py:153`). **Every CASSCF** fails regardless: P `mcscf/mc1step.py:1121` calls `contract_2e` without `link_index`, which G makes a required argument (`:150`, `:192`). The kernel itself is right: a 2-D call agrees with PySCF to 1.8e-15 (closed shell) and 4.4e-16 (nα ≠ nβ) (survey probe) |
| §3.3 last-but-one row | COSX "greenfield", filed under "Measured and rejected here — do not re-propose" | `[X]` It was never measured, so it is not a rejection. It is S1 here |
| §4 J, last line | "k-point VV10 … absent" | `[X]` Also broken at **Γ**. The Γ path warns and then calls `nr_nlc_vxc = NotImplemented` (G `pbc/dft/rks.py:85-98`, `pbc/dft/numint.py:841`); the k-point path raises (G `pbc/dft/krks.py:63-64`). PySCF's own PBC VV10 is the molecular double integral over one cell, with no lattice sum, so there is no periodic oracle in PySCF either (X6) |

---

## 4. The catalogue

Columns:

- **PySCF 2.13**: where the cost is.
- **GPU today**: the §0 status, with its evidence.
- **Carries it**: what already exists on the GPU, in GPU4PySCF or in
  Electrolites.
- **To build**: the new work.

Oracles are in §6.

### S. SCF- and DFT-level

| ID | Feature | PySCF 2.13 | GPU today | Carries it | To build | Diff. | Contam. | Demand |
|---|---|---|---|---|---|---|---|---|
| **S1** | **COSX (seminumerical exchange) energy, then gradient** | `sgx_fit` (P `sgx/sgx.py:36-110`). The default `pjs=False` makes both J and K seminumerical; `pjs=True` gives DF-J plus P-junction-screened K (`:96-101`). Level-2 LKO grids (`:389-390`). Hot loop `get_k_only` (P `sgx/sgx_jk.py:1043-1133`), per grid block: AOs; F = (w·ao)(proj·D); **G[ν,g] = Σ_λ (νλ\|g) F[λ,g]** in C (`SGXnr_direct_k_drv`, `:1342-1416`); then K = aoᵀG. Incremental ΔD builds, rebuilt every 5 cycles (`sgx.py:129,183-230`). **2 918 grid points per atom** at level 2, against 12 370 for the level-3 DFT grid (penicillin V/def2-SVP, PySCF CPU grid build, measured here `[V]`). Gradient: P `sgx/grad/rhf.py:288-494` (`int1e_grids_ip`, LKO weight and overlap-fit derivatives) | **absent** (§3.1) | GPU `eval_ao`; DF-J (`fastdfj`, 5–6x GPU4PySCF's); `fast1e`'s int3c1e Rys/Boys code, far-field branch and launch tuning (`electrolites/fast1e.py:415-500`, `_plain_chunk`); grouped GEMMs. **Neither** existing contraction of `int1e_grids` (charge-contracted, density-contracted: G `gto/int3c1e.py:307,393`) is the one K needs, and the plain one stages (ngrids, nao, nao) through pinned host memory (`:211-305`) | a **K-contracted int3c1e** kernel, G[ν,g] = Σ_λ(νλ\|g)F[λ,g]: shell-pair × grid-tile blocks, F in shared memory, pair and \|F\| screening, no atomics. LKO grids: the GPU partition kernel knows only Becke and Stratmann (G `dft/gen_grid.py:194-201`), so build them on the host once or add the scheme. ΔD builds. Short-range ω for RSH (the GPU int3c1e asserts ω ≥ 0, G `gto/int3c1e.py:213`). Gradient: the ip variant and the `s1` form | mod (unscreened energy); hard (screened, gradient) | moderate (GauXC's sn-K is public, BSD) | high (RIJCOSX is ORCA's hybrid default) |
| **S2** | **SCF stability analysis** (KT:C13) | P `scf/stability.py`: RHF/UHF/ROHF internal (`:283,432,468`); RHF→UHF and real→complex external (`:325-430`); UHF→GHF (`:503-640`); GHF (`:176`); DHF (`:231`). A Davidson over the orbital Hessian, about one TDA iteration per step | **stub** (§2 row `scf`); PySCF's driver on `gmf` raises (§3.1); **host** works | `gen_g_hop_rhf/rohf/uhf` (G `scf/soscf.py:34,70,106`); `gen_response` with `singlet` None/True/False and `hermi` 0/1/2 (G `scf/_response_functions.py`); the Davidson `eigh` (G `tdscf/_lr_eig.py:44`); every response kernel Electrolites already speeds up (`fastjk`, `fastk` at `hermi=2`, `fastxcfxc`, `fastrshresp`) | glue (≈150 lines `[?]`). GHF/GKS also need `gen_g_hop_ghf` (the GPU Newton raises for GHF, G `scf/soscf.py:714`) and S6's GKS `fxc` | easy | low | **high**: every UKS solution in the TMC set should be checked `[?]` |
| **S3** | **ROHF/ROKS gradient; Hessian** (KT:B19) | Gradient = the UHF gradient with a focka/fockb energy-weighted density (P `grad/rohf.py:28-56`, `roks.py`, `df/grad/rohf.py`). **PySCF has no ROHF/ROKS Hessian** (P `hessian/` holds rhf, rks, uhf, uks) | gradient **stub** (asserts); TDA raises; Hessian **WRONG** (§3.2) | G `grad/uhf.py:35-71` `grad_elec` (needs the stacked tag that G `dft/roks.py:31-37` already builds for `get_veff`), `grad/uks.py`, the DF versions; Electrolites' UHF/UKS gradient modules (reach them: `coverage_probe`) | gradient glue; Hessian greenfield (ROHF CPHF) | easy (gradient); greenfield (Hessian) | low | moderate |
| **S4** | four-component DHF/DKS (+DF, TD-DHF/DKS, DHF gradient) | P `scf/dhf.py`: LLLL/SSLL/SSSS spinor ERIs (`:913,939,968`; SSSS on by default `:483`), Gaunt/Breit (`:623-640`); DF `df/r_incore.py:61-62`; DKS `dft/r_numint.py`; gradient `grad/dhf.py:231-270`. ERI-bound on a complex 2-spinor basis; SSSS carries l+1 on all four centres | **absent** | the real-basis σ·p identity reduces SSLL to the `ipvip1` class: G has 3c2e `ipvip1` (`df/int3c2e.py:1190,1311`), so a DF Dirac–Coulomb is existing kernels + complex GEMM + `sph2spinor` `[?]` | direct SSSS needs new fourth-derivative J/K classes | greenfield / hard | low | low (X2C covers most uses), but the **cleanest greenfield benchmark** in S |
| **S5** | sfX2C1e gradient and Hessian | P `x2c/sfx2c1e_grad.py`: per atom, 3 × several (2n_unc)³ GEMMs/solves plus `_get_r1` (`:50-89,149-215`), i.e. O(natm·n_unc³); ip 1e integrals on the uncontracted basis (`:109-126`). Hessian `sfx2c1e_hess.py:115-176`, O(natm²·n_unc³) `[?]` | gradient **stub**; the Hessian object builds and its `kernel` raises | strided-batched GEMM / cuSOLVER; a GPU `pvp`-type kernel exists in PBC X2C (G `pbc/x2c/x2c1e.py:316-318`, KT:K3) | per-atom linear algebra batched over atoms × 3; GPU `ippnucp`/`ipprinvp`-type integrals | mod | low | moderate (all-electron heavy-element geometries; the Pt complex of KT §12.4) |
| **S6** | GHF/GKS TDDFT (and X2C-GHF spin–orbit TDDFT) | P `tdscf/ghf.py:53-97`, `gks.py:169-176` (mcol/ncol `fxc`). PySCF has no GHF/GKS gradients either | **absent**; no `_gen_ghf_response` in G; `numint2c.nr_fxc` raises (G `dft/numint2c.py:545-547`) | complex spin-blocked J/K (G `scf/ghf.py:36-123`); `_lr_eig`; `mcfun_gpu.eval_xc_eff` to deriv 2; `fastmcfun`'s antipodal-pair halving of the 770 samples (KT §13.15) | GHF response (glue); a GKS `fxc` contraction (grid kernel, complex four-component ρ₁) | easy–mod (GHF); mod–hard (GKS) | low | low–moderate |
| **S7** | orbital localization | Boys (P `lo/boys.py`). Pipek–Mezey **recomputes `orth_ao('meta_lowdin')` on every `atomic_pops` call** (P `lo/pipek.py:76`) and holds `projR` (natm × nmo², `:67-80`). IBO is a **pure-Python triple loop** (i, j<i, atom) per sweep (P `lo/ibo.py:148-200`). Edmiston–Ruedenberg feeds nmo rank-1 densities into a direct `get_jk` each iteration (P `lo/edmiston.py:31-37`) | **absent**; CuPy input raises | cuBLAS; DF cderi (ER as Σ_P B_pq B_ii); meta-Löwdin is already measured small (1.0–3.7 s at ≤ 284 atoms, KT §12.4) | device CIAH; parallel round-robin Jacobi sweeps for IBO/PM; DF-ER | easy–mod | low–moderate | moderate (local correlation, embedding, bonding analysis) |
| **S8** | initial guesses and SCF addons | SAP guess (P `scf/hf.py:738-760,801-845`: int3c2e against SAP charges); vSAP (grid); `mom_occ`; `remove_linear_dep_` | SAP **absent**; vSAP broken (D1); `mom_occ` and `remove_linear_dep_` fail on CuPy (survey probe) | `contract_int3c2e_auxvec` (as G `x2c/sfx2c1e.py:174` uses it); the grid code | one call each; the D1 fix | easy | low | moderate. The two TMCs that do not converge in 50 cycles (KT §5.1 runnability walls) are the cases a better guess is for `[?]` |

### M. Multireference

Nothing in GPU4PySCF beyond `fci/direct_spin1.py`, and nothing in Electrolites
(its only FCI code is `benchmarks/path_anatomy.py`'s `feature_fci`).

| ID | Feature | PySCF 2.13 | GPU today | Carries it | To build | Diff. | Contam. | Demand |
|---|---|---|---|---|---|---|---|---|
| **M1** | **a device FCI solver** (`direct_spin1` complete: σ, Davidson, RDMs, multi-root) | P `fci/direct_spin1.py:87-148` → C `FCIcontract_2e_spin1`; Davidson; RDMs via C `FCIrdm12_drv` (P `fci/rdm.py:123-160`). σ ≈ 2·npair²·Nα·Nβ flop: ≈ 6e12 at CAS(16,16) (arithmetic) | **partial**: σ only, and correct. It is **34–39x** the CPU σ at CAS(16,16) (KT §12.4). Every call rebuilds and uploads the address tables (G `:131-148,158`), uploads `eri` and `ci0` (`:151,156`) and returns `out.get()` (`:187`). `uint16` addresses cap Nα, Nβ ≤ 65 535 (`:134,144`), so CAS(18,18) is the largest closed shell. It fails on 1-D vectors and without `link_index` (§3.4) | the σ kernel; `_lr_eig.eigh` | a PySCF-compatible signature; device-cached link tables; device Davidson and `make_hdiag`; multi-root σ (nroots × 32 GEMM columns per tile); RDM1/RDM12/`trans_rdm12` as GEMMs; 32-bit addresses. At CAS(18,18) one vector is 18.9 GB, so a 12-vector subspace needs host staging or a two-vector Olsen solver. Hand back NumPy at the end: P `mcscf/mc1step.py:445` replaces a non-NumPy CI vector by a bool | mod | moderate–high (G's CUDA is public; GANSU has FCI) | high (CAS(14–18) double-shell 3d spaces) |
| **M2** | CASCI/CASSCF driver and the **direct** active-space integrals | CASCI: `get_h2eff` → `ao2mo.full` (P `mcscf/casci.py:1107-1125`). CASSCF `_ERIS` (ppaa, papa, vhf_c) rebuilt **every macro-iteration** (P `mcscf/mc1step.py:353,435`) by `mc_ao2mo`. That goes outcore whenever `_scf._eri` is absent (P `mcscf/mc_ao2mo.py:274-291`), with Python shell-pair loops and HDF5 round trips (`:194-250`). 2·nmo²·ncas² of storage: 4.1 GB at nmo 1000, ncas 16 | **absent** (§3.1; D8) | Two routes, both verified against P (survey probes): (a) (pq\|uv) = Cᵀ J[C_u C_vᵀ] C over ncas² densities with GPU `get_jk(hermi=0)` matches ppaa to 4.6e-10, and papa via K to 4.8e-10; (b) `fastmp2.get_int4c2e_ovov(mol, C_act, C_all)` returns papa to 6.2e-10. GPU and libcint integrals differ at ~5e-10, so the cross-code CASCI tolerance is 1e-9 | a `CASSCF` subclass that keeps P's macro/micro control flow and overrides `ao2mo`, `gen_g_hop`, `update_jk_in_ah`, `update_casdm`, `get_jk`; a four-coefficient `fastmp2` (C1C2\|C3C4) for ppaa (Z2); h1eff from the GPU `get_veff` | mod | moderate (TeraChem's AO-direct CASSCF is closed) | high |
| **M3** | **DF-CASSCF integrals** | P `mcscf/df.py:302-396`: the full (L\|pq) transform every macro-iteration, **written to HDF5 as naux·nmo² doubles** (`fxpp`, `:349`) and read back (`:383-386`), which is 32 GB at nmo 1000, naux 4000; papa GEMM 2·naux·(nmo·ncas)² | **absent**: G `df/df.py:272` `ao2mo = NotImplemented`; a GPU DF object is not routed to DFCASSCF (P `mcscf/__init__.py:190`) | `get_j3c_ovl_gpu_bdiv` (G `mp/dfmp2_addons.py:871`) and `decompose_j3c_gpu` (`:501`) build B[u,p,L] for any coefficient pair, in fp64 (DF-MP2's default is fp32); DF `get_jk` and `fastdf` for vhf_c | papa = BBᵀ; ppaa blocked over p; no nmo²·naux tensor | easy–mod | low–moderate | **high**: the production route for 50–150-atom TMCs |
| **M4** | the CASSCF orbital step on the device | `gen_g_hop` has Python per-orbital loops (P `mcscf/mc1step.py:63-75,104-108`). **Each AH iteration does one two-density J/K** (`:194,1021`; ≤ 30 per micro-step), and `gorb_update` does another (`:97`) | **absent** | `_davidson_cc` (G `scf/soscf.py:308`); `fastjk` (n_dm densities, one integral pass); DF `get_jk` | the per-orbital loops as batched GEMMs | mod | moderate | (with M2/M3) |
| **M5** | state averaging | P `mcscf/addons.py:896-1030`; RDMs root by root (`:982-990`) | **absent** | M1 | σ and RDMs batched over roots | easy (after M1) | moderate | high (ligand-field and spin manifolds, 5–20 roots) |
| **M6** | **(SA-)CASSCF gradients and NACs** | Direct: per atom, **every** `int2e_ip1` integral, unscreened, contracted with the AO active 2-RDM (P `grad/casscf.py:125-131`); `grad/casci.py` adds an RHF Z-vector; SA-CASSCF via a Lagrangian CG (P `grad/lagrange.py:130`); NACs P `nac/sacasscf.py:140`. DF: P `df/grad/casscf.py` with `solve_df_rdm2` (P `df/grad/casdm2_util.py:84`), `int3c2e_ip1` (`:449`), `int2c2e_ip1` (`:290`). All on the host | **absent** | **DF, nearly a drop-in**: G `df/grad/rhf.py`'s `_jk_energy_per_atom` builds Γ^P_μν from a factorised density and calls the general `sum_ejk_int3c2e_ip1` (`:158`) and `int2c2e_ip1_per_atom`. Putting d_{P,uv} and C_act in gives the active 2-RDM term, and `fastdfgrad` already speeds that function up. `fastcphf` serves the Z-vector. **Direct**: `fasttdejk`'s density-pair ip1 family covers the core/active terms | a hook and transition-RDM variants (DF). Direct: the 2-RDM term as ncas(ncas+1)/2 J-gradient pairs, or a native Γ-contracted ip1 kernel | mod (DF); hard (direct) | low–moderate (TeraChem's is closed) | high (geometry optimisation, conical intersections; pairs with `fastnacms`) |
| **M7** | **SC-NEVPT2** | Two regimes. **Large CAS**: dm123 plus two 4-RDM × ERI contractions in C (`_contract4pdm`, `NEVPTkern_*`, P `mrpt/nevpt2.py:796-798`), O(N_det·ncas⁶) `[?]`, then ~60 host `einsum`s up to ncas⁸. **Large molecule**: Sijrs needs **cvcv** (core × virtual)² from `ao2mo.outcore.general` (`:385`), MP2-like N⁵ with o²v² memory; DF variant P `mrpt/dfnevpt2.py` (cvcv via HDF5) | **absent** | cvcv: `fastmp2` (occupied → core, virtual → virtual) or `fastdfmp2`-style DF. M1's RDM kernels | the **4-RDM contraction kernel** (string gather + GEMM): greenfield | hard | **low** | high (dynamic correlation on CASSCF for TMC spin states) |
| **M8** | MC-PDFT family (MC-PDFT, L-PDFT, MS/CMS/XMS-PDFT, gradients) | P `mcpdft/`: E_ot over the CPU `block_loop` (`otfnal.py:200`); on-top density as an ngrid·ncas⁴ GEMM (`otpd.py:115-121`); translated functionals through CPU libxc; veff builds a **grid papa** (`pdft_eff.py:123`); gradients use CPU `eval_ao` (P `grad/mcpdft.py:292`) | **absent** | GPU numint `eval_ao`/`eval_rho`, GPU libxc, cuBLAS | the on-top contraction and translated-functional glue; veff; gradients | easy (energy); mod (veff, gradients) | **low** | moderate–high (TMC spin states `[?]`) |
| **M9** | the rest | UCASSCF: `half_e1` with `nocc`, MP2-like (P `mcscf/umc_ao2mo.py:78,112`); `direct_uhf`; selected CI (irregular sparse string lists); AVAS/DMET-CAS/APC (host linear algebra; fail only on CuPy input) | **absent** | M1–M3; `fastmp2` per spin | as the lines say; AVAS & co. need only a `.get()` shim | mod / greenfield / easy | low | low |

**M2 and M3, when built**, cut the cost that dominates per macro-iteration
today on the host. M1 cuts the CI step. M6–M8 then sit on top of them. A
minimal GPU CASSCF in build order is §9 Phase 2.

### P. Single-reference correlated and Green's-function methods

| ID | Feature | PySCF 2.13 | GPU today | Carries it | To build | Diff. | Contam. | Demand |
|---|---|---|---|---|---|---|---|---|
| **P1** | **RCCSD(T)** | P `cc/ccsd_t.py:34-131`. `_sort_eri` builds `vvop` = v²·o·(o+v) doubles (`:135-175`), in HDF5 unless incore. Loops over a ≥ b virtual blocks, each into the C kernel `CCsd_t_contract` (`:74`, source not shipped). Cost ≈ 2·o³v⁴ flop `[?]`, about 2o CCSD-ladder iterations. At alanine dipeptide/def2-SVP (o = 39, v = 161) that is ≈ 8×10¹³ flop, about 5x the 14 iterations `fastccsd` now finishes in 10.1 s (KT §13.27) — so once CCSD is on the device, (T) is the step `[?, arithmetic]` | **stub** (`ccsd_t = NotImplemented`). `pyscf.cc.ccsd_t.kernel` on GPU ERIs fails (G has no `ovvv`; its ovvv/vvvv are AO-direct). **host** works (7e-15 against CPU on water, survey probe) | `fastccsd`'s device t1/t2 and ovoo/oovv/ovvo, and its cache of **all AO integral blocks — dropped at the end of the solve** (`electrolites/fastccsd.py:658`); `fastmp2`'s half transform (two virtual quarters give (ia\|bc) per occupied batch); cuTENSOR | **ijk-driven** triples: W_abc from v × v × v² DGEMMs on the fp64 tensor cores (PySCF's abc layout gives thin (o², v)·(v, o) GEMMs); one reduction kernel for the six-permutation energy; keep the AO cache into (T); frozen core; ovvv streamed by k-batch when o·v³ does not fit | mod | **high** (ByteQC, Apache-2.0, ships multi-GPU CCSD(T); NWChem; ExaChem/TAMM) | **high** |
| **P2** | riders on P1 and `fastccsd`: QCISD(T), CCD, BCCD; **DF-CCSD** | QCISD(T) is a copy of the (T) driver with its own kernel (P `cc/qcisd_t.py:35-126`); CCD zeroes t1 (P `cc/ccd.py:28-33`). DF-CCSD (P `cc/dfccsd.py`) assembles vvvv from vvL·vvLᵀ every iteration (`:101-110`) | **absent**; DF runs conventional, with a warning (D5) | P1; `fastccsd`; `fastdf*` cderi | DF-CCSD: no `GINTfill`, so none of `fastccsd`'s f-shell 37 GB local-memory wall (KT §13.27), plus a DF ovvv for (T) | easy–mod (riders); mod (DF) | moderate | moderate |
| **P3** | **DF-MP2 relaxed density and gradient** (KT:I7, first half) | **PySCF has no DF-MP2 gradient at all**: every DF class raises (P `mp/dfmp2.py:177-178` and the U/native variants). The conventional `grad/mp2.py` builds an AO 2-PDM per shell block and recomputes `int2e_ip1` (`:86-100`); the Z-vector goes through `cphf.solve` | **stub** (G `mp/mp2.py:351-352`) | G `df/grad/rhf.py:158`: the general three-index density into `sum_ejk_int3c2e_ip1`; `int2c2e_ip1_per_atom`; `fastdfgrad`; `fastdfmp2`'s per-block amplitudes; CPHF and the response path. **No new integral kernel** `[?]` | Γ^P_ia and the relaxed density; the Lagrangian; the Z-vector | mod | moderate | **high** (RI-MP2 geometries). Once built it is **ahead** of PySCF |
| **P4** | **double-hybrid DFT** (KT:I7, second half) | **no support in PySCF**: `B2PLYP` → KeyError (P `dft/libxc.py:632`, CPU and GPU, §3.1). The manual recipe is DFT with scaled exchange plus scaled MP2. `grad/mp2.py` on an RKS reference is **not** a valid double-hybrid gradient (HF exchange and `get_veff` in its Z-vector) | **absent**; a manual B2PLYP on G (DF-RKS + DF-MP2) runs and lands 8.9e-7 Eh from the CPU recipe (survey probe) | P3; `fastdfmp2`'s os/ss components | energies: a parameter table and os/ss scaling (front end); gradients: P3 + `fxc` in the Z-vector + the XC gradient | easy (energy); mod (gradient) | moderate | high `[?]` |
| **P5** | MP2 RDMs and frozen natural orbitals; conventional UMP2/GMP2 | P `mp/mp2.py:151-173` (`make_rdm1`), `:217-272` (`make_fno`) | **absent**: G binds only `ao2mo`/`init_amps` (G `mp/mp2.py:365-371`); UMP2/GMP2 raise; D7 | `fastdfmp2` writes t = g/D per block, so P_ij and P_ab are two more GEMMs | as stated | easy | moderate | moderate (FNO-CCSD(T) with P1) |
| **P6** | EOM-CCSD (IP, EA, EE) and CCSD Λ | P `cc/eom_rccsd.py`: host Davidson; **EA stores the full Wvvvv** (v⁴, P `cc/rintermediates.py:180-188`); EE costs an o²v⁴ ladder per root per iteration. Λ costs about one CCSD solve (P `cc/ccsd_lambda.py`) | **stub** | `fastccsd`'s AO-direct ladder (which removes the v⁴ store); `_lr_eig` | IP first, then EE/EA; Λ | easy–mod (IP); mod–hard (EE/EA) | moderate (ExaChem/TAMM) | moderate |
| **P7** | UCCSD / UCCSD(T) / DF-UCCSD | P `cc/uccsd.py` (three amplitude spin blocks), `uccsd_t.py` (four C kernels) | **absent** | P1's kernels | a whole device UCCSD first (≈ 3x the `fastccsd` work `[?]`) | hard | moderate | moderate (open-shell thermochemistry) |
| **P8** | dRPA / URPA | P `gw/rpa.py:43` (40 frequencies): per frequency an naux²·ov SYRK and a determinant; `get_ov_blk` re-reads ovL from HDF5 every frequency | **absent** | G's device `cderi_ovl` (`mp/dfmp2_drivers.py:80-82`) | nothing new: cuBLAS SYRK, a Cholesky log-determinant | easy | moderate–low | moderate |
| **P9** | G0W0 (AC, CD), UGW | P `gw/gw_ac.py`: the **full** Lpq[naux, nmo, nmo] on the host (`:913-916`), 100 frequencies; per frequency a Π SYRK, an naux³ inverse and a Qmn GEMM. CD adds complex per-pole terms | **absent** | as P8 | build only L_ia and L[:, orbs] (≈ 50 GB for the full Lpq at 1 200 AOs `[?]`); batch the frequencies | mod | moderate | moderate–high |
| **P10** | ADC(2), ADC(2)-X, ADC(3) | P `adc/radc_ee.py:913-919`: the vvvv ladder applied to every Davidson vector; DF `adc/dfadc.py:69-84` | **absent** | `fastccsd`'s device tensors and AO-direct ladder | RADC(2) first | mod; hard (ADC(3), UADC-EE) | low | moderate (photochemistry) |
| **P11** | electron–phonon coupling (molecular) | P `eph/rhf.py:35-50`: the Hessian's `make_h1`, `solve_mo1`, `hess_elec` | **absent** | GPU equivalents in MO conventions (G `hessian/rhf.py:54,422,640`); `fastjkip1` | glue, plus a density-side-only flag in `fastjkip1` (G's `_get_jk_ip1` sums both halves) | mod | low | low |

### O. Molecular properties

`pyscf-properties` modules: efg, esr, freq, gtensor, hfc, infrared,
magnetizability, nmr, nsr, polarizability, rotational_gtensor, ssc, zfs, plus
`pbc/prop/{efg, polarizability}`. Most have rhf/rks/uhf/uks variants, and nmr,
ssc and efg also have dhf. **None has a GPU counterpart.**

GPU4PySCF's own `properties/` coverage is restricted-only:

- NMR shielding: RHF/RKS, GIAO, no RSH, meta-GGA or NLC (G
  `properties/shielding.py:40,120-122`).
- Polarizability: static only.
- IR and Raman: RHF/RKS, although G `hessian/uks.py` exists.
- EDA: restricted only.

| ID | Feature | PySCF / pyscf-properties | GPU today | Carries it | To build | Diff. | Contam. | Demand |
|---|---|---|---|---|---|---|---|---|
| **O1** | **the GIAO two-electron term** (KT:C6) **and UKS NMR** | `int2e_ig1` via `_vhf.direct_mapdm` on the host (G `properties/shielding.py:139-150`). After `fastnmr` it is **5.9 / 38.7 / 1175.0 s of 8.6 / 44.3 / 1207.5 s** NMR runs at 22 / 42 / 113 atoms, i.e. 69–97 % (`docs/README_fastnmr.md`) `[V]` | **partial**: RKS on the GPU except this term; UKS raises (G `properties/shielding.py:34`) | the ip1 Rys machinery (the ig factor is an origin shift, KT:C6); `ucphf`; `hermi=2` response; `fastnmr` | a GPU (ig1) Rys J/K kernel; a spin-resolved `fastnmr` and the UKS driver | hard (kernel); easy–mod (UKS) | **low** | high |
| **O2** | **NMR spin–spin coupling** | `ssc`: DSO via `int3c1e_iprinv` per nucleus pair; PSO via `int1e_prinvxp` per nucleus and a singlet `hermi=2` CPHF; FC from AO values at the nuclei and a **triplet** CPHF (`gen_response(singlet=False)`); FC+SD via `int1e_sa01sp`. About 3·N_nuc + N_nuc (or 6·N_nuc) right-hand sides | **absent** | multi-RHS CPHF, which is exactly what `fastcphf`, multi-density `fastjk`/`fastk` and `fastxcfxc` (triplet `fxc`) win at; `int3c1e_ipip` with nuclei as grid points; `eval_ao` at the nuclei | rinv-type 1e kernels batched over nuclei (KT:C9's family) | mod | **low** | high (organic NMR prediction `[?]`) |
| **O3** | frequency-dependent polarizability, first hyperpolarizability; the UKS gates on IR, Raman and polarizability | `polarizability/rhf.py`: `hyper_polarizability`, `cphf_with_freq` (SciPy `newton_krylov`), `polarizability_with_freq` | **absent** (static RHF/RKS only, G `properties/polarizability.py:52-90`); IR/Raman assert restricted (G `properties/ir.py:40`, `raman.py:210`) | `ABBA_krylov_solver` with `'shifted_linear'` (G `tdscf/_krylov_tools.py:708`); RKS/UKS `gen_response`; G `hessian/uks.py` | drivers; β needs `kxc` (a third XC derivative, `fastxcfxc`-like) | easy–mod; mod (β) | low–moderate | moderate |
| **O4** | magnetizability | `int2e_gg1`, `int2e_g1g2` through `jk.get_jk(…, 's4', 9)`; CPHF from NMR | **absent** | O1's kernel family | second-order GIAO J/K | hard | low | low–moderate |
| **O5** | EPR and the rest: hyperfine, g-tensor, EFG/Mössbauer, ZFS, NSR, rotational g | Hyperfine: FC (AOs at the nucleus) and SD (`int1e_ipiprinv`) are cheap; PSO needs 3 RHS per nucleus. g-tensor: two-electron spin–orbit integrals (`int2e_p1vxp1`, …) through `direct_mapdm`. EFG: 1e per nucleus. ZFS: two-electron spin–spin dipolar | **absent** | `int3c1e_ipip`; UHF response | hyperfine FC+SD and EFG: drivers; g-tensor and ZFS: greenfield two-electron kernels | easy (HFC FC+SD, EFG); hard (g-tensor, ZFS) | low | moderate for open-shell TMCs `[?]` |

ECD rotatory strengths are Im(μ·m) of transition moments G already computes
(`tdscf/rhf.py:493-549`). That is trivial and not a kernel target.

### V. Solvation and embedding

| ID | Feature | PySCF 2.13 | GPU today | Carries it | To build | Diff. | Contam. | Demand |
|---|---|---|---|---|---|---|---|---|
| **V1** | ddCOSMO / ddPCM (SCF, gradient, TDDFT gradient) | P `solvent/ddcosmo.py`: per cycle `make_phi` (int3c2e on cavity points, `:406-460`), a per-atom `block_loop` (`:490-509`), and a dense (natm·nlm)² solve **every call** (`:531`) | **absent** | `int1e_grids` and `fastpcmbatch`; `eval_ao`; one cuSOLVER LU instead of a solve per call | small Ylm projection kernels | mod | low | **low**: GPU4PySCF's PCM family covers the same use |
| **V2** | **QM/MM Hessian** (MM-charge terms) | none in PySCF | **WRONG** (§3.2) | `int3c1e_ipip.py`; the PCM Hessian's charge-contracted `h1` path (`fastpcmbatch`) | the MM charges' `ipip`/`ip` second derivatives into `hess_elec` and `h1` | easy–mod | low | moderate (QM/MM frequencies) |
| **V3** | Gaussian-smeared MM charges | P `qmmm/itrf.py:158-195` (SCF), `:324-437` (gradient) | hcore already passes exponents (G `qmmm/itrf.py:150-153`), but `energy_nuc` asserts point charges (`:97`) and so does the gradient (`:221`) | `int1e_grids(charge_exponents=)`, `int1e_grids_ip*` | an erf nucleus–MM term | easy | low | moderate |
| **V4** | polarizable embedding (CPPE) | two `int3c2e_ip1` passes on the polarizable sites per cycle (P `solvent/pol_embed.py:386-405`); induced dipoles solved in CPPE on the CPU; no gradients | **absent** (`cppe` not installed) | `int1e_grids_ip1`, density- and dipole-contracted (G `gto/int3c1e_ip.py:468`); KT:F9's batching | a GPU induced-dipole solver | mod | low–moderate | low–moderate |
| **V5** | TDDFT-PCM gradient: three dense builds where one will do (extends KT:F11) | — (PySCF has none) | **ahead**, but `grad_solver` runs three times per gradient (via `pcmobj.grad`, G `solvent/pcm.py:717-724`, and G `solvent/tdscf/pcm.py:199,203`; NACs three more, `:279,288,289`). Each run rebuilds dense dF, dA and [3, n, n] dD, dS for IEF-PCM, SMD and SS(V)PE (G `solvent/grad/pcm.py:575-579,651-655`) | the upstream code | one build serving all three, since the term is bilinear | easy | low | moderate |

### X. Periodic

| ID | Feature | PySCF 2.13 | GPU today | Carries it | To build | Diff. | Contam. | Demand |
|---|---|---|---|---|---|---|---|---|
| **X1** | **Γ-point TDA/TDDFT** (RKS/UKS) | P `pbc/tdscf/rhf.py:206-230`: the molecular `gen_vind` with `exxdiv=None`, plus velocity-gauge oscillator strengths with the GTH non-local commutator (`:140-170`) | **absent** (§3.1). The molecular GPU TDA on a Γ object fails because the PBC `gen_response` takes one density (G `pbc/dft/rks.py:317`, `reshape(1,1,nao,nao)`) | `gen_response` (G `pbc/dft/rks.py:285`, written for SOSCF); `nr_rks_fxc` (G `pbc/dft/numint.py:505`); multigrid_v2 `fxc`; the GPU Davidson; FFTDF/GDF/RSJK K | a multi-density `vind`; the D10 fix; `exxdiv`; the velocity gauge | easy–mod | low–moderate | high (no GPU path anywhere in PySCF's ecosystem `[?]`) |
| **X2** | **k-point TDA/TDDFT** (kshift = 0; then kshift ≠ 0) | P `pbc/tdscf/krhf.py:237-287`: per trial vector, nk transition densities built with Python `einsum` k-loops, then the response. K dominates for hybrids, as KT:J3 | **absent**. The k-point `gen_response` exists at kshift = 0 only (G `pbc/scf/khf.py:781-790`). The GDF kshift J/K are stubs that raise (G `pbc/df/df_jk.py:126-127,260-262`) | `fastfftk` already takes nset > 1 (time reversal holds for transition densities, checked); `fastpbcdf` | k-blocked transition-density GEMMs; a batched-RHS FFTDF K; GDF kshift J/K | mod; mod–hard (kshift ≠ 0) | low | high |
| **X3** | PBC stability analysis | P `pbc/scf/stability.py:56-147` | **absent** (TypeError, G `pbc/scf/khf.py:654`) | X1 / X2; S2 | a driver | easy–mod (after X1) | low | moderate |
| **X4** | k-point MP2 | P `pbc/mp/kmp2.py:111-143`: a ki·kj·ka loop of `einsum('Lia,Ljb->iajb')`, O(nk³·o²v²·naux); `Lov` built on the host per k-pair (`:156-226`) | **absent** (§3.1) | GDF k-pair cderi on the device (`fastpbcdf`); the `fastdfmp2` pair-GEMM pattern | a k-blocked complex `Lov` transform; batched pair GEMMs reduced on the device | mod | moderate | moderate |
| **X5** | PBC G0W0 | P `pbc/gw/krgw_ac.py:162-181`: Π per k, per frequency, per k-shift | **absent** | GDF cderi, ZGEMM, batched inverses | as P9 with k | hard | moderate | moderate |
| **X6** | periodic VV10 / rVV10 | no lattice sum in PySCF (§3.4) | **broken** at Γ and k | `fastvv10ft`/`fastnlc` ideas; the FFT mesh | a Román-Pérez–Soler FFT kernel (truly periodic); the oracle is an external code (e.g. Quantum ESPRESSO rVV10) | mod–hard (greenfield) | low | moderate |
| **X7** | PBC ECP | P `pbc/gto/ecp.py:31-75`: a one-time lattice-summed ECP | raises `NotImplementedError('ECP in PBC SCF')` (G `pbc/scf/hf.py:252-253`) | G `gto/ecp.py` and the BvK supercell machinery | the lattice-summed ECP and its gradient | mod | low | moderate (one-time cost; it decides whether heavy elements run at all) |

Not targets now:

- GHF/KGHF (empty stubs, D9);
- spin–orbit X2C;
- k-point symmetry;
- ROKS/GKS;
- MDF;
- periodic CC, CI and ADC;
- `eph_fd`.

### T. Tools

| ID | Feature | PySCF 2.13 | GPU today | Carries it | To build | Diff. | Contam. | Demand |
|---|---|---|---|---|---|---|---|---|
| **T1** | cube files: density, orbitals, MEP (KT:E3); CuPy shims for FCIDUMP, WFN, `mo_mapping` | P `tools/cubegen.py:96-101`: `eval_gto` per 8 000-point block; MEP by CPU int3c2e per 600 points (`:198-205`) | **absent** (raises on CuPy, §3.1) | GPU `eval_ao`; `fast1e`'s density-contracted `int1e_grids` **is** the MEP | drivers | easy | low | moderate (every visualisation workflow) |

`hessian/thermo.py` works on the GPU Hessian, which is returned as NumPy, so
it is not a target `[V]` (survey probe).

### Z. Infrastructure and correctness

| ID | Item | Why | Diff. |
|---|---|---|---|
| **Z1** | **guards for the silent wrong answers**, then the §3.3 defects | §3.2: the ROHF/ROKS `Hessian`, `Gradients` and `TDA` must raise until S3 exists; the QM/MM Hessian must raise until V2 does. Ship the guards as an Electrolites module and report them upstream | easy |
| **Z2** | **a general device ao2mo**, (C1C2\|C3C4), direct and DF | Serves CASCI and CASSCF (M2), NEVPT2's cvcv (M7), CCSD(T)'s ovvv (P1), FCIDUMP (T1) and UCASSCF (M9). `fastmp2._half_into` already takes two coefficient pairs (`electrolites/fastmp2.py:91-114`); its last two GEMMs (`:221-224`) need a second pair. The DF version is `get_j3c_ovl_gpu_bdiv` in fp64 | mod |
| **Z3** | **delivery: modules that *add* a feature** | Electrolites so far *replaces*: `_TARGETS` in `electrolites/__init__.py` names the GPU4PySCF attribute each module overwrites and fails loudly if it moved. A feature module needs the inverse. It **asserts that the attribute is absent or `NotImplemented`** in GPU4PySCF (so an upstream implementation is noticed, not shadowed). It then (a) attaches methods to G's classes (`RHF.stability`, `RHF.CASSCF`, `RHF.COSX`, …), (b) registers a `gpu4pyscf.<path>` module where PySCF's `to_gpu()` will look for it (P `lib/misc.py:1545`), and (c) falls back to `to_cpu()` with a warning (§0 rule 4) | easy |
| **Z4** | one device eigensolver / Krylov layer | Stability (S2), FCI (M1), EOM (P6), ADC (P10), PBC TDDFT (X1/X2) and frequency-dependent response (O3) all need a device Davidson or a shifted linear solver. G has `_lr_eig.eigh`, `_krylov_tools.ABBA_krylov_solver` and `soscf._davidson_cc`; reuse rather than add a fourth | easy |
| **Z5** | a CPU-baseline harness | `benchmarks/cpu_baseline.py`: the same molecules, functionals and flags as `path_anatomy.py`, on PySCF 2.13 on a CPU node, recording the core count. Owed before any §5.1 timing (§0 rule 1) | easy |

---

## 5. Ranked

Two orderings, as KT §5, because they barely overlap; plus the glue items that
need no new kernel, and what not to do now.  **No entry has a measured share
yet**, so every rank is a judgment on four inputs. In order of weight they are:

1. **demand**, in this repository's workload and generally;
2. **GPU fit**: GEMM- or integral-bound arithmetic, rather than host-side
   irregular work;
3. **reuse**: how much of the GPU work already exists here;
4. **oracle cleanliness and cost.**

§8's measurements can reorder it, and §5.1 is re-ranked when they are taken,
as KT's was.

### 5.1 Kernel and method targets — where the optimisation work is

| Rank | Entry | Reaches | Why this rank |
|---|---|---|---|
| 1 **done** (P1, P5; P2 open) | **P1** RCCSD(T) (+ **P5** FNO, **P2** DF-CCSD) | every CCSD(T) single point | Once `fastccsd` has made CCSD 2.4–10.9x faster (KT §13.27), (T) — still CPU-only — is ≈ 2o ladder iterations of work, so it is **the** step of a CCSD(T) run `[?]`. Every input but ovvv is already on the device, and the AO blocks ovvv needs are computed and then thrown away (`fastccsd.py:658`). The work is DGEMM-bound on fp64 tensor cores. The oracle is exact and deterministic. Highest demand of any correlated method. Contamination is high (ByteQC), which is fine for throughput but poor for the benchmark track. **Measured (§11.2)**: CPU (T) is 76–91 % of today's fastest route. **Built (§12.2)**: GPU (T) 7.1–7.9x the CPU node, ≤ 6e-15 Eh from PySCF |
| 2 **done** (M1, M3, M5, M2; M4's J/K) | **M1 → M3 → M4 → M5** a GPU (DF-)CASSCF | CASCI, CASSCF, SA-CASSCF; then M6–M8 | The largest capability gap, on this repository's own workload (open-shell TMCs). Every piece but the FCI Davidson/RDMs exists and is verified: the σ kernel at 34–39x, DF B[u,p,L], ppaa/papa from J/K and from `fastmp2`. PySCF's DF-CASSCF writes naux·nmo² to disk every macro-iteration and does a J/K per AH step on the host. **Measured (§11.3)**: J/K + `_ERIS` 85–93 % at CAS ≤ (10,9), FCI 29 % at (14,13), no gain from more CPU cores **Built (§13)**: DF-CASSCF 20–38x a CPU node per macro-iteration on two open-shell TMCs, 8.8–15.6x on p-benzyne; energies ≤ 1e-9 from PySCF |
| 3 → §5.4 (gate failed, §14.3) | **S1** COSX energy (gradient after) | every hybrid SCF at TZ/QZ or with diffuse functions, opt-in | The one new **integral kernel** in this plan with throughput upside on the SCF itself, and the place KT left off: exact K is 66.8 % of a patched SCF and at the machine (KT §2.1). It is an approximation and an *alternative route*, so its baseline is `fastk`, and its rank is gated on §8 item 1's crossover. Rank 3 if it wins at def2-TZVP; drop to §5.4 if it does not win anywhere `fastk` runs. **Measured (§11.1)**: CPU COSX loses to GPU exact K by 20x at SVP but only 3.7x at TZVPD and 2.6x on open-shell TZVP; kept for TZVP+/diffuse, error 3e-4–1.2e-3 Eh **Gate (§14.3)**: an unscreened GPU sn-K on GPU4PySCF's int3c1e kernel costs 42x the exact K at penicillin V/TZVPD (131 s against 3.1 s per build) -- slower even than the CPU COSX. Not built |
| 4 **done** | **P3 + P4** DF-MP2 gradient; double hybrids | RI-MP2 and double-hybrid geometries | **Ahead of PySCF**, which has neither. No new integral kernel: G's general three-index derivative contraction, `fastdfgrad`, `fastdfmp2` and CPHF carry it. Double-hybrid energies are a front end on the same code **Built (§14)**: DF-MP2 gradient 1.2 s (alanine dipeptide/SVP) to 80 s (taxol/SVP) on one A100, 19-58x PySCF's DF-MP2 relaxed density alone; double hybrids (B2PLYP & co.) with analytic gradients |
| 5 | **O1** GIAO two-electron kernel (KT:C6) + UKS NMR; **O2** spin–spin coupling | every NMR run; open-shell NMR; J-couplings | O1 is the **measured** remainder of NMR after `fastnmr`: 69–97 % (`README_fastnmr.md`), the only entry here with a share already in hand. O2 is a many-RHS triplet + PSO CPHF, the batching this repository is best at. Both are low contamination |
| 6 | **X1 → X2** PBC TDA/TDDFT | excited states of solids and surfaces | No GPU path exists. `gen_response`, `fxc`, the Davidson and `fastfftk`'s multi-density K are in place; batching K over nroots·nk² pays more here than in the SCF |
| 7 | **M6** DF-(SA-)CASSCF gradients and NACs | geometry optimisation, conical intersections | Nearly a hook on G's DF gradient machinery, which `fastdfgrad` already speeds up. After rank 2 |
| 8 | **M8** MC-PDFT, then **M7** NEVPT2 | dynamic correlation on CASSCF | MC-PDFT is about one XC evaluation once CASSCF is on the GPU. NEVPT2 is the cleanest high-value greenfield kernel (4-RDM contraction) and the cvcv half is `fastmp2`'s. After rank 2 |
| 9 | **P8 → P9** RPA, then G0W0 | correlation energies; quasiparticle energies | Pure cuBLAS/cuSOLVER on G's device `cderi_ovl`. The design rule is never to build the full Lpq |
| 10 | **S5** sfX2C1e gradient (Hessian after) | all-electron heavy-element geometries | Per-atom dense linear algebra, which maps to strided-batched GEMM; needs GPU `pnucp`-type ip integrals |
| 11 | **S6** GHF/GKS TDDFT | spin–orbit and non-collinear excitations | A new complex four-component `fxc` grid kernel; GHF-TDHF is nearly glue |
| 12 | **S7** IBO / PM / ER localisation | local correlation, embedding, analysis | Host-bound Python sweeps (IBO's triple loop, PM's uncached `orth_ao`), and ER via DF |
| 13 | **X4** k-point MP2 | periodic correlation | GEMM-bound O(nk³); extends `fastdfmp2` and `fastpbcdf`; the `Lov` memory wall is the interesting part |
| 14 | **P6** EOM-CCSD + Λ; **P10** ADC(2) | excited / ionised states at CC quality | On `fastccsd`'s ladder and the shared Davidson (Z4) |
| 15 | **O3** frequency-dependent α, β | NLO, dispersion | The Krylov solver exists; β needs `kxc` |

### 5.2 Glue — no new kernel, do alongside

Each of these reuses only what is already on the GPU, and each closes a door
that users hit today (§3).

| Order | Entry | What it closes |
|---|---|---|
| 1 | **Z1** guards + **D1–D13** | two silent wrong answers; a dozen crashes |
| 2 | **Z3** delivery, **Z5** CPU baseline harness | the mechanism and the measurement every §5.1 entry needs |
| 3 **done** | **S2** stability analysis | the check every open-shell TMC solution should get, currently CPU-only: **measured 53–126x the GPU SCF it follows (§11.4)**; first item of Phase 1. **Built (§12.1)**: 41–66x the CPU node |
| 4 **done** | **S3** ROHF/ROKS gradient (§12.3) | ROKS geometries (ΔSCF, radicals) |
| 5 **done** | **S8** SAP guess (§12.4), vSAP fix | a better guess for the TMCs that do not converge (`[?]` until tried) |
| 6 | **V3**, **V2** QM/MM Gaussian charges, QM/MM Hessian | the assertion; the wrong Hessian |
| 7 | **V5** one TDDFT-PCM gradient build | 3x on that term `[?]` |
| 8 | **T1** cube files on the GPU; CuPy shims (`lo`, `tools`, AVAS) | every visualisation and active-space selection workflow |
| 9 | **O3** UKS IR/Raman/polarizability drivers | open-shell IR and Raman |
| 10 | **X3** PBC stability (after X1) | — |

### 5.3 By benchmark cleanliness (low contamination)

For the LLM-GPU-QM track. Greenfield results are **never pooled into speedup
statistics** with replacement kernels (06's B10 rule):

| Entry | Why it is clean | Oracle |
|---|---|---|
| **M7** NEVPT2's 4-RDM × ERI contraction | no public GPU SC-NEVPT2 located `[?]` | P `mrpt.NEVPT` per-subspace energies |
| **O1** GIAO `(ig1)` two-electron J/K | no public GPU GIAO two-electron integrals `[?]`; and a measured 69–97 % share | P `int2e_ig1` element-wise; shieldings |
| **O2** spin–spin coupling | no public GPU J-coupling `[?]` | pyscf-properties `ssc` |
| **M3/M8** DF-CASSCF integrals; MC-PDFT on-top terms | no public GPU DF-CASSCF or MC-PDFT located `[?]` | P `mcscf.df._ERIS`; P `mcpdft` E_ot |
| **S4** DF Dirac–Coulomb | no public GPU 4c code located `[?]` | P DF-DHF |
| **S6** GKS multi-collinear `fxc` | recent and niche | P `tdscf.gks` |
| **X6** periodic rVV10 | greenfield; even PySCF has none | an external periodic code |
| **P3** DF-MP2 gradient | moderate (RI-MP2 gradient theory is public; the GPU codes are closed) | finite differences; P `dfmp2_native.make_rdm1(relaxed=True)` |
| **P1** CCSD(T) | **high** — throughput track only | P `ccsd_t.kernel` |
| **S1** COSX | moderate (GauXC) — and **the** probe for silently degrading chemistry (02 review rank 5) | P `sgx_jk` at a fixed grid, element-wise |

### 5.4 Not recommended now

| entry | reason |
|---|---|
| conventional CC / CISD / MP2 gradients | a non-separable four-centre derivative contraction (an AO 2-PDM of nao⁴/2 on disk in PySCF, P `grad/ccsd.py:80`); the DF route (P3) gets the same users |
| CCSDT, CCSDTQ, GCCSD(T), CISD, AGF2, `gw_exact` | small-molecule or niche; T3 alone is o³v³/6, ≈ 85 GB at o = 20, v = 200 `[?]` |
| UCASSCF, selected CI (M9) | low demand / irregular greenfield |
| ddCOSMO/ddPCM (V1) | the PCM family covers it |
| full 4c DHF direct SSSS (S4) | X2C covers the users; keep S4's DF form as a benchmark only |
| P11 EPH, PBC CC/CI/ADC/GW | niche, or too far from what exists |
| S1 COSX / sn-K on the GPU | §14.3: the one GPU kernel for its integrals (GPU4PySCF's `GINTfill_int3c1e`) makes an unscreened sn-K build 33-240x the exact K it would replace, and 16x the CPU COSX build at penicillin V/TZVPD. Reopen only with a new grid-integral kernel ≥ 40x faster, with screening |

---

## 6. Correctness oracles

KT §6 / 06 §6 apply: element-wise against libcint/PySCF in fp64; finite
differences for derivatives; integrated quantities only for anything
approximate. Additions for the new classes:

| Target class | Oracle | Why it holds |
|---|---|---|
| any glue on existing GPU pieces (S2, S3 gradient, S8, P5, O3 drivers) | the PySCF CPU result on `mf.to_cpu()`, to 1e-10 (eigenvalues, energies) / 1e-8 (gradients) | same equations, both in fp64 |
| CCSD(T) and riders (P1, P2) | P `ccsd_t.kernel(mycc, eris, t1, t2)` **on the same t1/t2** to 1e-10 Eh | removes CCSD convergence from the comparison; (T) is non-iterative and deterministic |
| FCI / CASCI / CASSCF (M1–M5) | P `direct_spin1` energies to 1e-10 and RDMs to 1e-12 at fixed integrals; then CASSCF energies to **1e-9** (GPU and libcint integrals differ at ~5e-10) with `conv_tol` 1e-11; `g_orb` and `h_op(x)` on random x element-wise; test nα = nβ and nα ≠ nβ, with and without `davidson_only` | fixed integrals isolate the solver; the orbital step is checked before the macro loop |
| DF-CASSCF (M3, M6) | P `mcscf.df._ERIS` arrays with the same auxbasis to 1e-11; DF-CASSCF gradients to 1e-9; FD | identical DF approximation |
| NEVPT2, MC-PDFT (M7, M8) | per-subspace energies (Sr, Si, Sijrs, …) / E_ot on the same grid, to 1e-10 | each term has a CPU twin |
| COSX (S1) | P `sgx_jk.get_k_only` at a fixed density on **the same grid**, element-wise; its `debug=True` path (int3c2e on a fake molecule) for the exact-integral reference; then the error against exact K reported next to every timing | exact at a fixed grid; the approximation is the grid's, measured separately |
| DF-MP2 gradient, double hybrids (P3, P4) | FD of the DF-MP2 / double-hybrid energy; P `dfmp2_native.make_rdm1(relaxed=True)` to 1e-9; P's conventional MP2 gradient to ~1e-5 (DF error) | PySCF has no DF-MP2 gradient, so FD is the real oracle |
| GW / RPA (P8, P9) | P `GWAC` quasiparticle energies to 1e-8 on the same frequency grid; P `RPA` e_corr to 1e-9 | same quadrature |
| GIAO / SSC / properties (O1–O5) | pyscf-properties on `mf.to_cpu()` (installed in a separate environment, not `barw-dev`); integrals element-wise against libcint | every component has a CPU twin |
| QM/MM Hessian, ROHF Hessian (V2, S3) | FD of the (correct) gradient, as `feature_gap_probe.py` already does | no PySCF Hessian exists for either |
| PBC TDDFT, KMP2 (X1–X4) | P `pbc.tdscf` / `pbc.mp.KMP2` on `mf.to_cpu()`, per k, element-wise | |

---

## 7. Quarantine: unverified, to re-check

| # | Claim | Status |
|---|---|---|
| Q1 | every **Demand** cell | `[?]` by definition; §5's order rests on it, so §8's measurements and any user input outrank it |
| Q2 | P1: (T) ≈ 2o CCSD iterations; 5x `fastccsd`'s CCSD at alanine dipeptide | arithmetic from flop counts, not a measurement; §8 item 2 |
| Q3 | S1: COSX beats `fastk` at TZ/QZ and may lose at DZ | the deciding question for rank 3; §8 item 1 |
| Q4 | P3: "no new integral kernel" | read from G `df/grad/rhf.py:158`'s generality; not built |
| Q5 | M7: NEVPT2's 4-RDM contraction is O(N_det·ncas⁶) | the C source is not shipped; inferred |
| Q6 | S4: SSSS ≈ 10x LLLL; DF-DHF from existing `ipvip1` kernels | inferred |
| Q7 | P9: full Lpq ≈ 50 GB at 1 200 AOs | arithmetic |
| Q8 | pyscf-properties line references | read on GitHub, not run here |
| Q9 | contamination tiers | `[?]` exactly as 06 §0: no recall probe was run |
| Q10 | S8: a SAP guess helps the two non-converging TMCs | untried |
| Q11 | "≈150 lines" (S2), "≈ 3x the work" (P7) | estimates |
| Q12 | the "survey probe" rows | run by the survey passes in a scratch directory; the load-bearing ones (§3.1, §3.2, the CASCI correction) are in `benchmarks/feature_gap_probe.py` and were re-run for this document |

---

## 8. Measurements owed, in the order they would change §5

All of these fit an interactive allocation. Run the CPU side on a CPU node
(`salloc -C cpu --qos=interactive`, `OMP_NUM_THREADS=128`) and the GPU side on
one A100 with `patch_all()`. When `/pscratch` is down, use `-L cfs` and keep
files under `~/vibeqc/_el_jobs/`. Each measurement uses two sizes, and a TMC
where the method allows one.

1. **COSX crossover** (gates §5.1 rank 3). PySCF SGX (`pjs=True`, level 2) on
   the CPU node, against the patched GPU exact-K SCF. Systems: penicillin V
   and taxol at def2-SVP, def2-TZVP and def2-TZVPD, plus Ni dithiolene⁻ UKS.
   Record the SGX energy error against exact K. Also an upper bound for a GPU
   sn-K: the CPU SGX K-build's integral count against `fastk`'s quartet count.
2. **CCSD(T) split** on the CPU node: glycine/def2-TZVP and alanine
   dipeptide/def2-SVP, CCSD against (T) wall time; against `fastccsd`'s CCSD.
   This also times today's only hybrid route (GPU CCSD → host → CPU (T)).
3. **DF-CASSCF anatomy** on the CPU node: CAS(10,10) and CAS(14,14) on Ni
   dithiolene⁻ and the Cu macrocycle at def2-SVP, with 5-root SA. Per
   macro-iteration: `_ERIS`, the J/K in AH, FCI, and the rest.
4. **Stability on the host route**: UKS internal stability via `to_cpu()`, on
   Ni dithiolene⁻ and the Co sextet, against the GPU SCF time. This is the
   penalty S2 removes.
5. **NEVPT2 anatomy** on item 3's systems: the 4-RDM contractions against
   cvcv.
6. **The NMR remainder at UKS, and SSC** (pyscf-properties in a separate
   environment): glycine and alanine dipeptide.
7. **PBC TDA on the CPU**: diamond8 Γ and 2×2×2, FFTDF and GDF.
8. **RPA / G0W0 on the CPU**: alanine dipeptide/def2-SVP and penicillin V.
9. **The reference for a DF-MP2 gradient**: the conventional MP2 gradient on
   the CPU (the only PySCF reference) and the cost of an FD DF-MP2 gradient,
   at glycine and alanine dipeptide.
10. **Localisation on the CPU**: IBO, PM and Boys at taxol and PfPMT.

---

## 9. The next phase: work plan

Each item ships the way KT §13's did:

- an `electrolites/<module>.py`;
- `tests/test_<module>.py` against the §6 oracle on two systems;
- a `docs/README_<module>.md` with the measured table (two sizes, and a second
  class where the method allows one);
- a `coverage_probe` / `feature_gap_probe` row showing it is reached;
- the row in §5 marked **done**, with what is left.

The phases are ordered by dependency, not by calendar.

**Phase 0 — correctness and the means to measure** (before anything else)

- **Z1**: guard the ROHF/ROKS `Hessian`/`Gradients`/`TDA` and the QM/MM
  `Hessian` so they raise. Fix D1, D3, D4, D5 (raise instead of the silent
  conventional fallback), D7 and D13. Add §3.2/§3.3 to
  `docs/UPSTREAM_BUGS.md`.
  - *Accept*: `feature_gap_probe.py` shows no `WRONG` row.
- **Z3**: the feature-registry mechanism, with its "absent upstream"
  assertion. **Z5**: `benchmarks/cpu_baseline.py`.
- **§8 items 1–4.** These decide whether S1 stays at rank 3, give P1 its
  share, give M1–M5 their per-iteration anatomy, and give S2 its penalty.
- Add a closed-shell transition-metal complex to `benchmarks/molecules/` (§0
  rule 3).

*Status (2026-10-14)* — Phase 0 done. The taxol/def2-TZVPD GPU reference
did not converge in 150 cycles (§11.1):

| item | what was done | evidence |
|---|---|---|
| **Z1** guards | `electrolites/guards.py`, in `patch_all()`. The ROHF/ROKS `Hessian` and `TDA`/`TDHF`/`TDDFT` raise with the reason (PySCF has neither). The ROHF/ROKS `Gradients`, direct and DF, go to PySCF's gradient on the CPU with a warning (`cpu_fallback`), agreeing with it to 6e-8. The QM/MM `Hessian` raises; `undo_qmmm().Hessian()` still gives the vacuum one. | `tests/test_guards.py` (13 checks); `feature_gap_probe.py --patch all`: the two `WRONG` rows are now `raises` |
| defects | `electrolites/fixes.py`, in `patch_all()`. **D1** a device vSAP (PySCF's matrix to 2.8e-14 on the same grid; RKS/UKS/ROKS reach the minao energies to 1e-12). **D2** ROHF Newton (the guess is now a device array; 2.4e-12 from the plain SCF). **D3** `CCSD.to_cpu()` (its (T) matches PySCF to 5e-11). **D4** a clear error for open-shell CCSD. **D7** frozen-core DF-RMP2/UMP2 with `with_t2` (asserted before; now 5e-10 from PySCF's DF-MP2). **D13** the docstring. | `tests/test_fixes.py` (11 checks) |
| D5 | **left as is** `[X]` of §3.3's wording: the DF→conventional CCSD substitution is not silent. GPU4PySCF logs "DF-CCSD not available. Run the standard CCSD." at the default verbosity, and the calculation it runs is a well-defined one. Raising would break working inputs. DF-CCSD is P2 | — |
| **Z3** feature interface | `electrolites/features.py`: `register` / `enable` / `check_absent` (an added feature's slot must be missing, `None` or `NotImplemented` upstream, else `enable` refuses); `attach` (refuses to shadow); `register_gpu_module` (PySCF's `to_gpu()` then finds the GPU class); `cpu_fallback`; `not_available`. Exported as `electrolites.enable`, `enable_all`, `check_absent` | `tests/test_features.py` (9 checks, on a throw-away feature) |
| **Z5** CPU baseline | `docs/CPU_BASELINE.md` (the protocol) and `benchmarks/cpu_baseline.py` (features `scf`, `sgx`, `ccsd_t`, `casscf`, `nevpt2`, `stability`; stage times; the node, both BLAS libraries and thread counts recorded). Smoke-tested on two- and three-atom systems | — |
| closed-shell TMC | `benchmarks/molecules/tmc_ferrocene.xyz` (idealised D5h), added to `tmc_cases.txt`. RHF/def2-SVP converges (221 AOs, gap 0.46 Eh) | — |
| §8 items 1–4 | **run 2026-10-13/14**: results and what they change in §5 are **§11**. Scripts and environments are in `~/vibeqc/_el_jobs/phase0/` (`docs/CPU_BASELINE.md` §8) | §11 |

Found on the way: PySCF 2.13's wheel runs its C kernels on a **bundled
OpenBLAS 0.3.3**, not on NumPy's 0.3.30. `docs/CPU_BASELINE.md` §3 says how
that is labelled, and what is owed before a (T) or FCI ratio is quoted in a
summary.

**Phase 1 — the cheap capability items, and CCSD(T)**

- **S2** stability analysis (RHF/UHF/ROHF/RKS/UKS, internal and external).
- **S3** ROHF/ROKS gradient. **S8** SAP and vSAP. **P5** MP2 RDMs and FNO.
- **P1** RCCSD(T), keeping `fastccsd`'s AO cache into the triples.
  - *Accept*: (T) to 1e-10 against P on the same amplitudes, on glycine/TZVP,
    alanine dipeptide/SVP and the new closed-shell complex. Report the
    GPU-against-CPU-node ratio with the core count. Report the ratio against
    today's GPU-CCSD + CPU-(T) hybrid.
- Then **P2**'s DF-CCSD if item 2 shows the f-shell `GINTfill` wall still
  binds.

**Phase 2 — a GPU CASSCF, DF first**

1. **M1**: the device FCI solver (signature, link tables, Davidson, multi-root
   σ, GEMM RDMs, 32-bit addresses).
2. **M3**: the DF `_ERIS` (`get_j3c_ovl_gpu_bdiv` in fp64, papa = BBᵀ, ppaa
   blocked over p).
3. **M4**: the orbital step (vectorised `gen_g_hop`, J/K through `fastjk` or
   DF `get_jk`, `_davidson_cc`).
4. **M5**: state averaging.
5. **M2**: the direct route via Z2's four-coefficient `fastmp2`.

*Accept*: CASSCF energies to 1e-9 against PySCF with the same auxbasis, at
CAS(10,10) and CAS(14,14), on an organic diradical and on Ni dithiolene⁻ and
the Cu macrocycle; GPU against the CPU node per macro-iteration.

*Status (2026-10-14)* — Phase 2 done: §13. The `casscf` feature is PySCF's
CASSCF/CASCI with the FCI solver (σ, Davidson, RDMs), the `_ERIS` and every
J/K on the device, DF and exact. Left: M4's host loops (`gen_g_hop`,
`solve_approx_ci`), and CAS above 65 535 strings per spin.

**Phase 3 — the kernels with an open crossover question**

- **S1** COSX, *if* §8 item 1 shows a crossover inside the range users run.
  Start with the unscreened K-contracted kernel and element-wise agreement
  with PySCF at a fixed grid; then screening, ΔD builds, RSH, and the
  gradient.
- **P3 + P4**: the DF-MP2 relaxed density and gradient, double-hybrid energies
  and gradients.

*Status (2026-10-14)* — Phase 3 done: §14. P3 and P4 built (`dfmp2_grad`,
`dhdft`). S1's gate measured and failed, so S1 moves to §5.4.

**Phase 4 — on top of Phase 2, and the independent tracks**

- **M6** DF-(SA-)CASSCF gradients and NACs; **M8** MC-PDFT; **M7** NEVPT2.
- **O1** the GIAO two-electron kernel and UKS NMR; **O2** spin–spin coupling.
- **X1 → X2** PBC TDA/TDDFT (with D10 fixed first).
- **P8 → P9** RPA, then G0W0.

§5.2's glue items run alongside every phase. They need no kernel work, and
each closes a crash or a host-only route that users hit today.

---

## 10. Reproducing the survey

The runnability table (§3.1, §3.2):

    module unload cudatoolkit; module load cudatoolkit/12.9
    conda activate barw-dev
    python benchmarks/feature_gap_probe.py                 # GPU4PySCF 1.8.1 as shipped
    python benchmarks/feature_gap_probe.py --patch all     # the same table
    python benchmarks/feature_gap_probe.py --only S3,V2    # just the wrong-answer rows

The SGX grid count (S1), on any node:

    python -c "from pyscf import gto, dft, sgx; \
      m = gto.M(atom='benchmarks/molecules/penicillinV.xyz', basis='def2-svp'); \
      s = sgx.SGX(m); s.build(level=2); g = dft.gen_grid.Grids(m); g.level = 3; g.build(); \
      print(s.grids.coords.shape[0] / m.natm, g.coords.shape[0] / m.natm)"

The coverage map (§2) is a diff of the two trees' file lists:

    P=$CONDA_PREFIX/lib/python3.13/site-packages/pyscf
    G=/pscratch/sd/w/weiliang/g4psrc/g4p-1.8.1/gpu4pyscf
    for d in $(ls $P); do echo "$d: $(ls $G/$d 2>/dev/null | wc -l) files in G"; done

Every other `file:line` is a reading of P (2.13.0) or G (1.8.1, `5b284c2`). The
line numbers will move with either release, and the claim beside each one
says what to look for.

---

## 11. Phase 0 measurements (2026-10-13/14): §8 items 1–4

Taken by `docs/CPU_BASELINE.md`'s protocol, with these settings:

- **CPU side**: `benchmarks/cpu_baseline.py` on a whole Perlmutter CPU node
  (2 × EPYC 7763), PySCF 2.13 wheel, at the thread count §11.0 chose.
- **GPU side**: `benchmarks/path_anatomy.py --patch all` on one
  A100-SXM4-80GB.
- **Common**: `conv_tol` 1e-9, level-3 grids, all-electron.
- **Records**: JSON in `~/vibeqc/_el_jobs/phase0/results/`, logs beside them,
  tables by `phase0/compile.py`.

Each table states its gate: the agreement between the GPU and CPU numbers.
No GPU feature was built, so the ratios below are of **today's** routes.
They say how large each §5.1 target is, not what any target will deliver.

### 11.0 Threads, and three ways the first launch went wrong

**Thread scaling** on a 128-core node (`item0_threads_cpu.sh`):

| case | 32 threads | 64 | 128 |
|---|---|---|---|
| CCSD(T), glycine/def2-SVP | 17.5 s | 19.2 s | 23.7 s |
| CCSD(T), glycine/def2-TZVP | 164 s | **138 s** | 173 s |
| CCSD(T), alanine dipeptide/def2-SVP | 854 s | **707 s** | 932 s |
| CCSD(T), ferrocene/def2-SVP | 2338 s | 1988 s | **1866 s** |
| COSX SCF, penicillin V/def2-SVP | 102 s | 75 s | **72 s** |
| DF-CASSCF(8,7), Ni dithiolene⁻ | 1341 s | 1392 s | 1295 s |

**Choice**: CC at the fastest per system (64, 64, 128); SCF, COSX and
stability at 128; DF-CASSCF at **32**.

PySCF's DF-CASSCF does not scale past 32 threads. At 64 its J/K got faster
(731 → 509 s) and its `_ERIS` slower (430 → 750 s): the `_ERIS` step writes
naux·nmo² to HDF5 and reads it back every macro-iteration. On a CPU node, then,
more cores do not help the host CASSCF at all.

**Three launch hazards**, each of which made numbers that were then
discarded:

1. **`salloc` with a script runs the script on the login node.** It must be
   `salloc … srun …`. Both `phase0/*_env.sh` now refuse to run on a login
   node.
2. **`OMP_PROC_BIND=spread` pins NumPy's OpenBLAS pthreads to one core.**
   Binding the master thread does this, because those threads inherit its
   affinity: the harness then saw 2 visible CPUs instead of 256. Process
   binding is left to `srun --cpu-bind=cores`.
3. **`pkill -f` / `pgrep -f` on a pattern also matches the shell issuing it.**
   Twice it revoked a fresh allocation.

`docs/CPU_BASELINE.md` §2–3 now say all three.

### 11.1 Item 1 — COSX against the GPU's exact exchange (S1, rank 3)

COSX here is PySCF's SGX with `pjs=True` (DF-J, P-junction-screened K) and
level-2 grids, on the CPU node at 128 threads. The comparison is the patched
GPU exact-K SCF. The COSX error is measured against the GPU exact energy,
which agrees with PySCF's own exact SCF to 6e-10 Eh (penicillin V/SVP, the
one CPU exact run).

| system / basis | nao | CPU COSX SCF (cycles) | CPU COSX K, total | GPU exact SCF (cycles) | GPU `get_k`, total | K per build, CPU COSX / GPU exact | COSX error |
|---|---|---|---|---|---|---|---|
| penicillin V / def2-SVP | 430 | 72.5 s (13) | 20.7 s | 4.4 s (14) | 1.1 s | 20x | −4.2e-4 Eh |
| penicillin V / def2-TZVP | 858 | 171.6 s (14) | 68.3 s | 18.3 s (14) | 10.7 s | 6.4x | −4.7e-4 |
| penicillin V / def2-TZVPD | 1074 | 233.7 s (14) | 116.8 s | 45.6 s (14) | 31.9 s | **3.7x** | −4.7e-4 |
| taxol / def2-SVP | 1123 | 394.8 s (14) | 229.6 s | 13.7 s (15) | 6.8 s | 36x | −1.0e-3 |
| taxol / def2-TZVP | 2228 | 1509.7 s (15) | 1003.4 s | 134.6 s (14) | 95.0 s | 9.8x | −1.2e-3 |
| Ni dithiolene⁻ UKS / def2-SVP | 327 | 149.7 s (37) | 27.5 s | 11.3 s (35) | 4.6 s | 5.7x | −2.9e-4 |
| Ni dithiolene⁻ UKS / def2-TZVP | 689 | 382.0 s (49) | 111.0 s | 48.4 s (39) | 34.0 s | **2.6x** | −2.7e-4 |

**Taxol/def2-TZVPD** (nao 2795; no CPU COSX run) **does not converge on
the GPU**:

- **50 cycles**: 2286 s, `get_k` 1722 s, unconverged.
- **150 cycles** (`max_cycle` 150): 5938 s, `get_k` 4481 s, still
  unconverged. Its energy wanders at the 2e-5 Eh level (−2930.665299 after
  50 cycles, −2930.665279 after 150).

This is a convergence finding, not a timing row: diffuse functions on 113
atoms defeat the default DIIS. It needs a stability check (S2), level
shifting or SOSCF before it can serve as a reference. The one number it does
give is a GPU exact-K build of **~30 s per Fock build** at this size: 4481 s
over ~150 builds, i.e. 1722 s over 50 in the first pass, ~34 s, and less in
the later, tighter-screened builds. That is the number a GPU sn-K would have
to beat here.

**Reading:**

- **Who is faster.** Today's GPU exact K beats 128 CPU cores running COSX
  everywhere, by 2.6–36x per Fock build.
- **The gap shrinks with the basis**: 20x → 6.4x → 3.7x from SVP through
  TZVP to TZVPD on penicillin V, and 5.7x → 2.6x on the open-shell complex.
  That is the scaling argument for sn-K, measured.
- **What a GPU COSX would need.** It has to beat the CPU COSX by more than
  these ratios to win at all. At the ~10x a GPU kernel typically gets over a
  full CPU node `[?]`:
  - it would **lose at def2-SVP**;
  - it would **break even around def2-TZVP**;
  - it would **win 2.5–4x at def2-TZVPD** and on open-shell def2-TZVP.
- **Accuracy.** The approximation costs 2.7e-4 to 1.2e-3 Eh in the absolute
  energy (0.2–0.75 kcal/mol), and the error grows with the system.

**Consequence for §5.1 rank 3**:

- **Kept**, with a narrowed scope: diffuse and triple-ζ-or-larger bases, and
  large open-shell systems.
- **The first build step** must measure a GPU sn-K against `fastk` at
  penicillin V/def2-TZVPD before anything else is built on it.
- **The error budget** must be reported on relative energies, not only on
  absolute ones.

### 11.2 Item 2 — CCSD and (T) (P1, rank 1)

The CPU side is PySCF RCCSD followed by `ccsd_t`, all-electron, at the best
thread count. The GPU side is `fastccsd`'s CCSD, re-run at the CPU's
`conv_tol` 1e-9 (`path_anatomy.py --cc-conv-tol 1e-9`, added for this).

| system / basis | nao | CPU CCSD | CPU (T) | GPU CCSD (cycles) | CCSD: CPU / GPU | **(T)'s share** of today's fastest route (GPU SCF + CCSD, then CPU (T)) | gate: \|ΔE_corr\| |
|---|---|---|---|---|---|---|---|
| glycine / def2-TZVP | 185 | 102.2 s (64 thr) | 35.3 s | 6.3 s (16) | 16x | **76 %** | 2.5e-8 |
| alanine dipeptide / def2-SVP | 200 | 554.1 s (64) | 151.9 s | 12.4 s (19) | 45x | **91 %** | 2.7e-9 |
| ferrocene / def2-SVP | 221 | 1450.1 s (128) | 413.9 s | 66.1 s (24) | 22x | **85 %** | 3.6e-8 |

**The gate is 1e-7 and not 1e-8.** Independently converged CCSD energies at
`conv_tol` 1e-9 with PySCF's default `conv_tol_normt` 1e-5 differ by up to
3.6e-8 between the two codes. The amplitudes, not the integrals, set that
floor: the SCF energies agree to 4e-10. `docs/CPU_BASELINE.md` §5 now says
so. (T) on identical amplitudes keeps its 1e-10.

The first GPU pass ran at PySCF's default 1e-7, and its ferrocene e_corr was
1.7e-6 off: a default-tolerance run is not a valid comparison.

**Reading:**

- **(T) dominates today's fastest route.** Once CCSD is on the GPU, the CPU
  (T) is **76–91 %** of the fastest route that exists today. Q2's estimate
  (arithmetic) is now a measurement: P1 stays **rank 1**.
- **(T) on the CPU node alone takes 5.6–12x as long as the whole GPU CCSD**
  that precedes it. A GPU (T) at ~10x over a CPU node (the CCSD got 16–45x
  here) would take today's 47–485 s route down to ~15–112 s `[?]`.
- **These CPU (T) times use the wheel's bundled OpenBLAS 0.3.3**
  (CPU_BASELINE §3). A source-built PySCF comparison is owed before a (T)
  ratio is quoted in a summary.

### 11.3 Item 3 — DF-CASSCF anatomy on the CPU (M1–M5, rank 2)

PySCF DF-CASSCF from ROHF orbitals, def2-SVP, 32 threads. There is no GPU
side, because no GPU CASSCF exists (§3).

| system, CAS | wall | DF J/K (all calls) | ..of it in the AH step | `_ERIS` rebuilds | FCI solver (σ) | J/K + `_ERIS` | FCI |
|---|---|---|---|---|---|---|---|
| Ni dithiolene⁻, (8,7) | 1341 s | 731 s | 588 s | 430 s | 0.8 s | **87 %** | 0.1 % |
| Ni dithiolene⁻, (10,9) | 1960 s | 1111 s | 912 s | 621 s | 3.6 s | **88 %** | 0.2 % |
| Ni dithiolene⁻, (10,9) SA-5 | 1779 s | 775 s | 622 s | 735 s | 8.5 s | **85 %** | 0.5 % |
| Ni dithiolene⁻, (14,13) | 4996 s, not converged | 1409 s | 1159 s | 807 s | 1428 s (σ 1317 s) | 44 % | **29 %** |
| Cu macrocycle²⁺, (10,9) | 2821 s | 1908 s | 1475 s | 716 s | 0.6 s | **93 %** | 0.0 % |

Caveats on two rows:

- **Cu macrocycle**: the ROHF reference did not converge in 100 cycles (on
  the CPU). The CASSCF converged from those orbitals, so the anatomy stands,
  but the energy is not a reference.
- **Ni (14,13)**: stopped at PySCF's macro-iteration limit unconverged. Its
  per-iteration split is still valid.

**Reading:**

- **Up to CAS(10,9)**, 85–93 % of a host DF-CASSCF is the J/K and the `_ERIS`
  rebuild that **M3 + M4** move onto GPU kernels that already exist. Neither
  scales with CPU cores (§11.0).
- **At CAS(14,13)** the FCI σ build grows to 29 %: σ grows as C(14,7)², the
  integrals do not. That makes **M1** — a device FCI solver whose σ kernel
  is already 34–39x the CPU's at CAS(16,16) (KT §12.4) — as necessary as
  M3/M4.
- **Rank 2 stands**, and its build order (M1 → M3 → M4 → M5) is confirmed.
  The Amdahl bound for M3 + M4 alone is about 7–14x at CAS ≤ (10,9).

### 11.4 Item 4 — stability analysis on the host route (S2)

PySCF UKS internal + external stability (`mf.stability(external=True)`) on
the CPU node at 128 threads, against the GPU UKS SCF it would follow. The
GPU and CPU SCF energies agree to 3e-9 (Ni) and 7e-9 Eh (Co).

| system | CPU SCF (cycles) | CPU stability: internal + external | GPU SCF (cycles) | **stability / GPU SCF** | result |
|---|---|---|---|---|---|
| Ni dithiolene⁻ / def2-SVP (21 atoms) | 120.5 s (37) | 320.5 + 275.9 = **596 s** | 11.2 s (35) | **53x** | internally stable, externally unstable |
| Co(II) sextet / def2-SVP (113 atoms) | 2639 s (78) | 3529.5 + 4467.6 = **7997 s** | 63.7 s (58, at `max_cycle` 150) | **126x** | internally stable, externally unstable |

**Reading:**

- **Checking a solution costs far more than finding it.** Today, checking a
  GPU UKS solution for stability takes 53–126 times as long as finding it,
  2.2 hours for the 113-atom complex.
- **The work is already on the GPU in another form.** It is two Davidson
  solves on orbital-Hessian products, and those products are J/K and XC
  response builds that this repository already runs on the GPU.
- **S2 should open Phase 1**: it is easy glue on measured-large time.
- **A second finding**: GPU4PySCF's default `max_cycle` 50 does not converge
  this complex (58 cycles on the GPU, 78 on the CPU), the same wall KT §5.1
  recorded for Ni-dithiolene⁻/def2-TZVPP.

### 11.5 What §5 and §9 take from this

| §5.1 / §5.2 entry | before | after the measurements |
|---|---|---|
| P1 CCSD(T), rank 1 | (T) "the step" `[?]` | **measured**: 76–91 % of today's fastest route; rank 1 confirmed |
| M1–M5 GPU CASSCF, rank 2 | J/K per AH step, disk `_ERIS` `[?]` | **measured**: 85–93 % J/K + `_ERIS` at CAS ≤ (10,9); FCI 29 % at (14,13); flat in CPU cores; order M1 → M3 → M4 → M5 confirmed |
| S1 COSX, rank 3 | gated on the crossover | CPU COSX loses to GPU exact K by 2.6–36x per build, the gap closing with the basis. **Kept for TZVP+/diffuse/large open-shell**; first build step is a GPU sn-K against `fastk` at penicillin V/def2-TZVPD |
| S2 stability, glue #3 | "CPU-only" | **measured**: 53–126x the GPU SCF; moved to the **first item of Phase 1** |

Still owed:

- a converged taxol/def2-TZVPD reference (§11.1; needs SOSCF or level
  shifting);
- the wheel-against-source-build BLAS comparison for (T) and FCI
  (CPU_BASELINE §3);
- §8 items 5–10.

---

## 12. Phase 1, built (2026-10-14)

Five features. Each follows the same pattern:

- added through `electrolites.features` (`enable(name)`), each refusing to
  load if GPU4PySCF has filled its slot;
- each with its test suite against PySCF 2.13 on the CPU;
- all eight Phase 0/1 suites pass on a compute node, under `patch_all()`
  where it applies (`~/vibeqc/_el_jobs/phase0/logs/p1_tests.log`).

Timings are taken on one A100 (40 or 80 GB, as stated) against
§11's CPU-node records, by `docs/CPU_BASELINE.md`.

### 12.1 S2 — stability analysis (`electrolites/stability.py`, `enable('stability')`)

**What it is.** PySCF's `scf/stability.py`, term by term:

- RHF/RKS internal, real → complex and RHF → UHF;
- UHF/UKS internal, real → complex and UHF → GHF;
- ROHF/ROKS internal.

The orbital-Hessian products are **batched**: a block of trial vectors is one
`gen_response` call over a stack of transition densities. The eigensolver is
a block Davidson on device vectors.

**Correctness** (`tests/test_stability.py`). Seven cases: water RHF, B3LYP and
DF-PBE0; N₂ at 2.0 Å (unstable in all three senses); H₂O⁺ UHF, UKS and ROHF.
At Davidson tol 1e-8 on both sides, every lowest eigenvalue agrees with
PySCF's to < 1e-5, and every stable/unstable verdict agrees. The rotated N₂
orbitals take a UHF SCF 0.020 Eh below the RHF.

**Found on the way** `[X]`: PySCF's UKS UHF → GHF "Hessian" is **not
symmetric**. With its collinear XC kernel on spin-mixing densities,
|H − Hᵀ| = 0.023 on H₂O⁺/B3LYP. PySCF's symmetric Davidson therefore
returns a tolerance-dependent number: −0.07138 at its default tol and
−0.07270 at 1e-8. The operator's actual lowest real eigenvalue, from the
dense matrix, is **−0.07269**. This solver treats that one operator as
non-symmetric and returns −0.072690, which is what the test checks.

**Timed** (`benchmarks/stability_bench.py`, A100-80GB, `patch_all()`; the
CPU side is §11.4, 128 cores):

| system | GPU SCF (cycles) | **GPU stability, internal + external** | CPU stability (§11.4) | ratio | verdicts |
|---|---|---|---|---|---|
| Ni dithiolene⁻ / def2-SVP, UKS B3LYP | 13.7 s (35) | 7.8 + 6.9 = **14.7 s** | 596 s | **41x** | (stable, unstable) = CPU's |
| Co(II) sextet / def2-SVP, UKS B3LYP (113 atoms) | 65.6 s (58) | 84.4 + 36.9 = **121.3 s** | 7997 s | **66x** | (stable, unstable) = CPU's |
| penicillin V / def2-SVP, RKS B3LYP | 6.9 s (14) | 4.9 + 4.3 = **9.2 s** | — | — | (stable, stable) |

Checking a solution's stability now costs about as much as the GPU SCF that
found it (1.1–1.9x), where it cost 53–126x.

### 12.2 P1 — the closed-shell (T) correction (`electrolites/ccsd_t.py`, `enable('ccsd_t')`)

**What it is.** `CCSD.ccsd_t()` on GPU4PySCF's CCSD object.

- **Loop order.** PySCF's energy, rearranged to a loop over occupied triples
  i ≥ j ≥ k. For each (i, j) and a batch of k, the six occupied orders of W
  are strided-batched cuBLAS GEMMs on views of (ia|bc), (ia|jk) and t₂. Two
  CUDA passes form the permutation sum X and the energy.
- **Integrals.** From a generalisation of `fastmp2`'s half transformation
  (`get_int4c2e_ovpq`).
- **Fock matrix.** From one GPU Fock build, so a non-canonical f_vo enters as
  in PySCF.

**Correctness** (`tests/test_ccsd_t.py`). On identical amplitudes and
orbitals, it agrees with PySCF's `ccsd_t` to **≤ 6e-15 Eh**, across closed
shells, frozen core, frozen core + virtuals, and the batched-k path.

The first version assumed the summand symmetric in the occupied indices
alone. It is not, and that version was 10 % off on water (caught by a numpy
prototype). Each canonical triple now sums its six orders explicitly.

**Timed** (`benchmarks/ccsd_t_bench.py`, A100-80GB, on `fastccsd`'s
amplitudes at `conv_tol` 1e-9; the CPU side is §11.2):

| system / basis | o / v | **GPU (T)** (integrals + triples) | CPU (T), best threads | ratio | whole CCSD(T), GPU | whole CCSD(T), CPU node | |Δe_T| vs CPU |
|---|---|---|---|---|---|---|---|
| glycine / def2-TZVP | 20 / 165 | **4.6 s** (1.5 + 3.1) | 35.3 s (64) | **7.7x** | 14.8 s | 138 s | 5.7e-8 (CCSD-limited) |
| alanine dipeptide / def2-SVP | 39 / 161 | **21.4 s** (1.2 + 20.2) | 151.9 s (64) | **7.1x** | 38.2 s | 707 s | 1.8e-9 |
| ferrocene / def2-SVP | 48 / 173 | **52.6 s** (6.2 + 46.4) | 413.9 s (128) | **7.9x** | 124.1 s | 1866 s | 4.1e-8 (CCSD-limited) |

The (T) differences against the CPU runs are the CCSD amplitudes'
differences (§11.2's 1e-7 gate), not the (T) kernel's. Against today's
fastest route (GPU CCSD + CPU (T), 47–485 s), the all-GPU CCSD(T) is
**3.1–4.4x**.

**Profile and what is left** (`ELECTROLITES_T_PROFILE`, alanine dipeptide):

- **The fix that mattered.** The first build spent 101 of 116 s in the W
  GEMMs. CuPy's `matmul` with a broadcast operand copies it per batch
  element. Replacing every contraction with a strided-batched cuBLAS call on
  views took that to 13.4 s.
- **What is left.**
  - The GEMMs are 13.4 s, against an fp64 tensor-core floor of ~4.5 s for the
    o³v⁴ work.
  - The two energy passes are 6.8 s, memory-bound. Their permuted reads of
    W and X are uncoalesced.
  - A tiled energy kernel and larger GEMM batches (several (i, j) at once)
    are the next steps.
- **Memory.** It holds o·v³ for (ia|bc) and seven k-batch × v³ blocks.

### 12.3 S3 — ROHF/ROKS gradients (`electrolites/rohf_grad.py`, `enable('rohf_grad')`)

**What it is.** PySCF's ROHF/ROKS gradient — the UHF/UKS gradient with the
Roothaan energy-weighted density and a UHF-shaped density tag — on
GPU4PySCF's UHF/UKS gradient classes, direct and density-fitted. It replaces
`guards`' CPU fallback when enabled (`guards` defers to it).

**Correctness** (`tests/test_rohf_grad.py`, also with `--patch-all`).
Against PySCF's ROHF/ROKS gradients, to **≤ 2.2e-8 Eh/bohr**, across:

- ROHF, ROKS PBE, ROKS B3LYP, DF-ROKS B3LYP and DF-ROHF;
- H₂O⁺ (doublet) and CH₂ (triplet).

The analytic gradient agrees with a finite difference to 9e-6 (the step's
error).

### 12.4 S8 — the SAP initial guess (`electrolites/sap.py`, `enable('sap')`)

**What it is.** PySCF's Gaussian-fit SAP. V_sap is one
`contract_int3c2e_auxvec` call against all atoms' SAP charges, and the guess
follows PySCF's recipe for RHF, UHF (with `init_guess_breaksym`) and ROHF.

**Correctness** (`tests/test_sap.py`):

- V_sap agrees with PySCF's `make_sap` to 1.2e-12, and the guess densities
  to 1e-12.
- An SCF from it lands on the minao energy (water RKS, H₂O⁺ UKS, Ni
  dithiolene⁻ UKS). The Ni complex takes 35 cycles against 37: no
  convergence rescue on this case (Q10 stays open for the two
  non-converging TMCs).

**Also fixed.** The generalised `eigh` the guess would have used fails on a
compute node, though not on the login node (cause not isolated). The guess
uses canonical orthogonalisation instead, as does `fixes`' vSAP.

### 12.5 P5 — MP2 1-RDM and frozen natural orbitals (`electrolites/mp2_rdm.py`, `enable('mp2_rdm')`)

**What it is.** PySCF's `make_rdm1` (unrelaxed, spin-traced) and `make_fno`
on GPU4PySCF's conventional RMP2 and DF-MP2: the occupied and virtual blocks
as two GEMMs over the device amplitudes, with the MP2 re-run with `with_t2`
when none are stored.

**Correctness** (`tests/test_mp2_rdm.py`), against PySCF on the same
orbitals (the DF case against PySCF's DF-MP2), with and without frozen core:

- the 1-RDM to 2e-14 (MO basis) and 2e-13 (AO basis);
- the FNO frozen lists identical, and the active-virtual projectors to 2e-11.

With P1 this gives FNO-CCSD(T): `frozen, no = pt.make_fno(...)` →
`CCSD(mf, frozen=frozen, mo_coeff=no)` → `ccsd_t()`.

### 12.6 What §5 takes from Phase 1

| entry | status | measured |
|---|---|---|
| **P1** rank 1 | **built** | (T) 7.1–7.9x a CPU node; whole CCSD(T) 9.3–18.5x; the route GPU-CCSD + CPU-(T) is gone. Left: GEMM efficiency (3x headroom) and the energy pass |
| **S2** glue #3 | **built** | 41–66x the CPU node's analysis; now 1.1–1.9x the GPU SCF |
| **S3** glue #4 | **built** | GPU ROHF/ROKS gradients, direct and DF; the CPU fallback is no longer reached when enabled |
| **S8** glue #5 | **built** (SAP; vSAP was Phase 0's D1) | no convergence gain on the one TMC tried |
| **P5** | **built** | prerequisite for FNO-CCSD(T) |

---

## 13. Phase 2, built (2026-10-14): a GPU CASSCF

One feature, `casscf` (`electrolites.enable('casscf')` binds `mf.CASSCF` and
`mf.CASCI` on GPU4PySCF mean fields; `electrolites.casscf_gpu.CASSCF` /
`CASCI` are the same without the binding). It covers §9's items M1, M3, M4's
J/K, M5 and M2:

| file | what |
|---|---|
| `electrolites/fci_gpu.py` | M1: the FCI solver |
| `electrolites/casscf_gpu.py` | M3/M4/M2: the integrals, the J/K, and the assembly onto PySCF's CASSCF |
| `electrolites/casscf.py` | the feature binding |
| `tests/test_casscf_gpu.py` | 17 checks against PySCF 2.13 |
| `benchmarks/casscf_bench.py` | timing, with three CPU cross-checks (`--cpu-ref`, `--casci-check`, `--polish`) |
| `benchmarks/molecules/pbenzyne.xyz` | the organic diradical (idealised D2h p-benzyne) |

Timings are on one A100-SXM4-80GB against the CPU node of §11.3 (2× EPYC
7763, DF-CASSCF at 32 threads, `docs/CPU_BASELINE.md`), all def2-SVP from
(RO)HF orbitals.

### 13.1 What it is

**The object is PySCF's.** `CASSCF(mf, ncas, nelecas)` returns PySCF's own
`mcscf.CASSCF` (or `DFCASSCF`) built on `mf.to_cpu()`, with a mixin ahead of
it in the MRO. The macro/micro iteration, the augmented-Hessian solver, the
convergence tests, state averaging (`state_average_`), `frozen`, `natorb`,
the results and their attributes are PySCF's code. What moved to the device:

- **The FCI solver (M1)**, `fci_gpu.FCISolver`, a `direct_spin1.FCISolver`:
  - σ is GPU4PySCF's two kernels behind PySCF's signature (optional
    `link_index`, 1-D or 2-D vectors), with the address tables built once per
    (norb, nelec) and kept on the device.
  - The RDMs (`make_rdm1/12`, `trans_rdm1/12`): one gather kernel builds
    T[pq, I] = (E_pq c)[I] block by block, then dm1 = T c and
    ⟨E_pq E_rs⟩ = T Tᵀ are GEMMs; PySCF's `reorder_rdm` follows.
  - The Davidson runs on device vectors (`kernel`/`eig` overrides). It keeps
    `lib.davidson1`'s rules — |Δe| < tol and |r| < √tol, the subspace size
    `max_space + 4(nroots − 1)`, restart from the Ritz vectors, the
    level-shifted diagonal preconditioner taken at the lowest root for every
    root — so it takes the same path as PySCF's (13.4 says why that matters).
    PySCF's `kernel_ms1` still makes `hdiag`, the pspace step and the initial
    guess.
- **DF `_ERIS` (M3)**: three-centre integrals from GPU4PySCF's
  `get_j3c_ovl_gpu_bdiv`, fitted by the Cholesky factor of the metric (fp64),
  then ppaa, papa = B Bᵀ, j_pc, k_pc in blocks of p on the device. No HDF5 —
  PySCF's DF route writes naux·nmo² to disk and reads it back every
  macro-iteration (§11.3).
- **Every J/K (M4's)**: the AO-basis fitted tensor L[μ,ν,P] is built once and
  kept (when 2.5·nao²·naux·8 B fits in 30 % of free memory). J is two GEMVs;
  K for a symmetric density goes through its eigendecomposition — the CASSCF
  densities (core; the AH step's core/active rotations) are low rank — as
  K = Σ_P M_P s M_Pᵀ with M_P = L_P U. Otherwise GPU4PySCF's DF J/K.
- **The exact-integral route (M2)**, taken for a non-DF mean field (as
  `mcscf.CASSCF` does) or with `density_fit=False`:
  - ppaa and papa come from one batch of GPU4PySCF direct J and K over the
    active-pair densities D_uv = c_u c_vᵀ (u ≤ v): J[D_uv + D_vu] = 2(μν|uv)
    and K[D_uv] = (μu|νv).
  - Every other J/K is GPU4PySCF's direct J/K.
  - j_pc and k_pc, which PySCF reads only for the orbital step's diagonal
    preconditioner (`mc1step.py:150`), come from the fitted integrals, so
    the converged result does not depend on them.
  - The first build transformed the four-centre integrals
    (`fastmp2.get_int4c2e_ovpq`, two passes). The J/K form is 2.8x faster
    on the TMCs: 8.4 → 3.0 s per macro-iteration on Ni, 33.6 → 12.3 s on Cu.

**What is not on the device**: `gen_g_hop`'s Python loops and the AH
Davidson (PySCF's, on nmo² vectors), `solve_approx_ci`'s host σ calls (one
transfer each), and the CI vectors between macro-iterations. 13.4 says what
they cost.

### 13.2 Correctness

`tests/test_casscf_gpu.py` (17 checks, all pass on the login node; PySCF on
the same orbitals and the same auxiliary basis):

| check | worst difference |
|---|---|
| σ, rdm12 (both orderings), trans_rdm12; CAS(6,(3,3)) … CAS(9,(5,4)), random integrals | 2.4e-15 relative |
| device Davidson, 1–3 roots, CAS(10,(5,5)), (10,(5,4)), (11,(6,5)) | 3.6e-14 Eh |
| N₂/6-31G CAS(8,8) DF-CASSCF; CAS(6,6) DF-CASCI | 6.7e-13; 8.0e-13 |
| N₂ SA-4 DF-CASSCF from the same start | 1e-12 (states 5e-9: a degenerate Π pair) |
| O₂ triplet CAS(6,8) DF-CASSCF (ROHF) | 6.5e-13 |
| the same three without DF (M2) | 7e-14 – 8e-13 |
| `gpu_mf.CASSCF` = `CASSCF(gpu_mf)`, DF and not | 1e-9 gate |

**At scale**, three cross-checks, all in `casscf_bench.py`:

- **Independent runs.** GPU and CPU (§11.3's runs, which started from DF-ROHF
  orbitals, not the GPU's) land on the same minimum: Ni (10,9) **7.9e-13**,
  p-benzyne (14,14) **3.3e-10**, p-benzyne (10,10) exact **2.7e-11**, Cu
  exact **2.6e-10** (the CPU's from its unconverged ROHF).
- **Polish.** PySCF's CPU CASSCF is restarted from the GPU's converged
  orbitals and CI vectors at `conv_tol` 1e-10. It stops in 1 macro-iteration
  and moves the energy by **4e-11 – 1.0e-10** wherever the GPU ran at 1e-10
  (Ni SA-5, the exact route on all four systems). From GPU runs at PySCF's
  1e-9 the move is 0.8–1.1e-9 in 3–4 macro-iterations: that is the
  convergence tolerance, not a difference between the codes.
- **Same-orbital CASCI** (`--casci-check`, round 6). At the GPU's final
  orbitals, the GPU and CPU CASCI energies, both with CI tolerance 1e-12,
  agree to **1.8e-12 – 1.2e-11** on all six DF systems: p-benzyne (10,10)
  and (14,14); Ni (10,9), SA-5 and (14,13); Cu. This is the
  implementation's agreement with convergence taken out. The CASSCF's own
  `e_tot` is up to 9e-9 above it at (14,13), because PySCF solves the last
  CI only to `fcisolver.conv_tol` 1e-8.

**The 1e-9 acceptance of §9 is met** on every system. Converged-energy
checks hold on p-benzyne (10,10) and (14,14), Ni dithiolene⁻ (10,9) and
SA-5, and the Cu macrocycle (10,9), DF and exact. The same-orbital check
holds on all six, including Ni (14,13), whose CASSCF does not converge to
1e-9 on either code (13.4).

### 13.3 Timed

**DF-CASSCF, GPU against the CPU node per macro-iteration** (round 4, final
code; `conv_tol` 1e-9 except SA-5 at 1e-10; the CPU side is §11.3 plus two
new p-benzyne runs, `cpu_cas10_pb`/`cpu_cas14_pb`; macro-iterations counted
as `_ERIS` builds on both sides):

| system, CAS | GPU: macro, wall | **GPU s / macro** | CPU s / macro (macro) | **ratio** | GPU J/K, `_ERIS`, rest per macro |
|---|---|---|---|---|---|
| p-benzyne (10,10) | 35, 12.4 s | **0.35** | 5.47 (39) | **15.6x** | 0.10, 0.03, 0.22 |
| p-benzyne (14,14) | 30, 122.7 s | **4.09** | 35.8 (25) | **8.8x** | 0.10, 0.03, 3.96 |
| Ni dithiolene⁻ (10,9) | 112, 241.6 s | **2.16** | 42.9 (45) | **19.9x** | 0.99, 0.20, 0.97 |
| Ni dithiolene⁻ (10,9) SA-5 | 46, 67.3 s | **1.46** | 41.7 (42) | **28.6x** | 0.52, 0.11, 0.83 |
| Ni dithiolene⁻ (14,13) | 59, 369.7 s | **6.27** | 97.4 (51, unconverged) | **15.5x** | 0.97, 0.15, 5.15 |
| Cu macrocycle²⁺ (10,9) | 33, 135.0 s | **4.09** | 155.5 (17) | **38.0x** | 1.81, 0.36, 1.92 |

Whole runs are not compared: the two sides start from different orbitals,
so they take different numbers of macro-iterations (Ni (10,9): 112 on the
GPU, 45 on the CPU, to the same energy). On the Cu macrocycle they even end
at different minima (13.4).

**Exact integrals (M2)** (round 4/5; `conv_tol` 1e-10; the CPU side is new,
`cpu_cas10_pb_exact`, `cpu_cas10_ni_exact`, `cpu_cas10_cu_exact`, 32
threads, all with the ERIs in core):

| system, CAS | GPU: macro, wall | **GPU s / macro** | CPU s / macro (macro) | ratio | GPU J/K, `_ERIS` per macro |
|---|---|---|---|---|---|
| p-benzyne (10,10) | 33, 27.2 s | 0.82 | 0.75 (38) | 0.9x | 0.21, 0.42 |
| p-benzyne (14,14) | 35, 153.9 s | 4.40 | — | — | 0.20, 0.66 |
| Ni dithiolene⁻ (10,9) | 110, 764.0 s | 6.95 | 24.5 (51, unconverged) | 3.5x | 3.29, 2.84 |
| Cu macrocycle²⁺ (10,9) | 26, 477.0 s | 18.35 | 115.9 (34) | 6.3x | 6.72, 10.28 |

Reading:

- **DF route.** On the TMCs it is **20–38x per macro-iteration**, past
  §11.3's Amdahl estimate of 7–14x for M3 + M4 alone. The FCI step went to
  the device too (M1), and the CPU's J/K + `_ERIS` was not just moved but
  replaced: no HDF5, and a J/K that reuses the fitted tensor.
- **The organic diradical gains less** (8.8–15.6x). Its integrals are cheap
  on the CPU as well (104 AOs). At (14,14) the GPU macro-iteration is 97 %
  CI work: σ (11.8 M determinants), RDMs, and the host side of PySCF's
  `solve_approx_ci` and `gen_g_hop`.
- **Exact route.** It is not faster than the CPU for a small molecule:
  PySCF holds the 104-AO ERIs in memory, and the ERIs-in-core CASSCF is
  0.75 s per macro-iteration. On the TMCs the CPU's in-core ERIs are still
  cheaper than its DF route (24.5 against 42.9 s on Ni). The GPU's
  exact-route cost is the direct J/K, about 20 builds per macro-iteration at
  0.01–0.4 s each, plus the `_ERIS` J/K batch. **For a GPU CASSCF, DF is the
  fast route**: 3.2x (Ni) and 4.5x (Cu) faster per macro-iteration than the
  GPU's exact one.

### 13.4 How it got there, and what was found

**Rounds** (A100-80GB):

| round | change | Ni (10,9) s / macro | p-benzyne (14,14) s / macro | Ni (14,13) s / macro |
|---|---|---|---|---|
| 1 | GPU DF integrals, GPU4PySCF DF J/K, GPU σ/RDMs, PySCF Davidson on the host | 3.37 | — | — |
| 2 | J/K from the cached fitted tensor (low-rank K) | 2.53 | 8.36 | 16.4 |
| 3 | Davidson on the device | 2.22 | 4.16 | 6.28 |
| 4 | Davidson made PySCF's exact rule set; exact `_ERIS` through J/K | 2.16 | 4.09 | 6.27 |

- **Round 2 → 3: from the profile.** cProfile of p-benzyne (14,14), 6
  macro-iterations: of 58.5 s, `lib.davidson1`'s host work took about 30 s —
  `_outprod_to_subspace` 12.5 s, `_normalize_xt_` 8.6 s, and 9.0 s of σ
  results copied to the host. Moving the vectors to the device halved the
  macro-iteration.
- **What the (14,14) macro-iteration is now** (round 3 profile, 5
  macro-iterations, 19.9 s):
  - device synchronisation waits (`.get`) 6.0 s, mostly σ;
  - the RDM GEMMs 5.3 s (29 calls);
  - PySCF's `mc1step` Python 3.1 s;
  - J/K 0.5 s.
  
  The next steps, at this CAS size, are `solve_approx_ci` on the device and
  M4's vectorised `gen_g_hop`.

**Found on the way:**

- **Davidson rule parity is a correctness matter.** The first device
  Davidson applied each root's own level shift and a `max_space + nroots`
  subspace. Its energies matched PySCF's to 1e-13 on every non-degenerate
  test. On one run of the N₂ SA-4 test, with its degenerate Π pair, it
  converged to a solution 3.2e-3 Eh away. With PySCF's rules (preconditioner
  at e₀, `max_space + 4(nroots − 1)`) it matches to 1e-12 on five
  consecutive runs and on both J/K paths. `[X]` That one failure was not
  reproduced in isolation, so its trigger is not pinned down.
- **An Electrolites bug, fixed: `fastjk` crashed on an all-zero density.**
  `compat.tril_pair_mappings` reduced an empty array (every shell pair
  screened out) and raised `ValueError`. The CASSCF AH step asks for that
  when a trial vector has no core–external block. It is now an empty pair
  list, and the build returns zeros. `tests/test_jk.py` gained the check,
  and it fails without the fix.
- **The Cu macrocycle has many CASSCF minima a few mEh apart.** Seven GPU
  DF runs from the GPU ROHF ended at five energies: −2558.0910 (twice),
  −2558.0914, −2558.1155, −2558.1192 and −2558.1360 (twice); an eighth
  (round 6) ended at −2558.1154. The CPU's
  −2558.1154 came from its own, unconverged, ROHF. The selection is
  sensitive to tolerances and round-off in the first 20 macro-iterations.
  One `conv_tol` 1e-10 run sat 301 macro-iterations at −2558.09100 without
  converging; the same settings converged in 35 in a separate job. Every end
  point checks out against PySCF from the same orbitals (polish ≤ 3e-9). The
  exact route, a different Hamiltonian, reached −2558.115845 twice,
  identically. So Cu energies are not comparable between independent runs,
  and §11.3's CPU Cu energy is not a reference.
- **Ni (14,13) is flat.** GPU runs at 1e-9 stop at −4104.183761604 to
  −4104.183761613. PySCF restarted from there still lowers the energy by
  2.2e-8 over 50 macro-iterations without converging, as §11.3's own CPU run
  did not converge in 50. This is PySCF's convergence test on a flat
  surface, not a GPU difference.
  - A GPU run at 1e-10 went 301 macro-iterations without converging
    (−4104.183761616).
  - At fixed orbitals the GPU and CPU agree to 1.2e-11 (13.2).

### 13.5 What is left

- **M4 proper**: `gen_g_hop`'s loops, `solve_approx_ci` and the AH vectors
  on the device. They are most of the macro-iteration at CAS(14,14)
  (13.4), and a small share for the TMCs at (10,9).
- **CAS beyond 16 bits of strings.** GPU4PySCF's address tables are
  16-bit, so above 65 535 strings per spin (e.g. CAS(18,18)) σ falls back to
  PySCF's (logged), and so does the Davidson. M1's 32-bit addresses were
  not needed for this plan's sizes and are not done.
- **Memory.** The cached fitted tensor is 2.5·nao²·naux·8 B: 5 GB for the Cu
  macrocycle. Past 30 % of free memory the J/K goes back to GPU4PySCF's DF
  J/K, which is correct and about 2x slower (round 1 against round 2).
- **Not covered**: UCASSCF (M9), `newton_casscf` (untested; it uses the
  same hooks), `fix_spin_` (it runs on PySCF's Davidson by design), and
  NEVPT2/MC-PDFT/gradients (M6–M8, Phase 4).

### 13.6 What §5 takes from Phase 2

| entry | status | measured |
|---|---|---|
| **M1 → M3 → M4 → M5** rank 2 | **built** (M4 = its J/K; the host loops are left) | DF-CASSCF 20–38x a CPU node per macro-iteration on two open-shell TMCs at (10,9)–(14,13), 8.8–15.6x on an organic diradical; energies to ≤ 1e-9 against PySCF where both converge |
| **M2** | **built** | correct to 1e-10; 3.5x (Ni) and 6.3x (Cu) the CPU node's in-core exact CASSCF per macro-iteration, 0.9x on the 104-AO organic. The GPU's DF route is the fast one: 3.2–4.5x faster per macro-iteration than its exact route on the TMCs |
| **M6–M8** | unblocked | they sit on this object (`mc.ci`, `mc.mo_coeff`, the RDMs on the device) |

---

## 14. Phase 3, built (2026-10-14): DF-MP2 gradient, double hybrids, and the COSX gate

Two features and one negative result:

| file | what |
|---|---|
| `electrolites/dfmp2_grad.py` | P3: `DFMP2.Gradients()`, `DFMP2.make_rdm1_relaxed()` (feature `dfmp2_grad`) |
| `electrolites/dhdft.py` | P4: `DoubleHybrid(mol, 'B2PLYP')`, energies and gradients (feature `dhdft`) |
| `tests/test_dfmp2_grad.py`, `tests/test_dhdft.py` | 8 + 4 checks |
| `benchmarks/dfmp2_grad_bench.py` | GPU timing with a stage split |
| `benchmarks/snk_gate.py` | S1's gate |
| `benchmarks/cpu_baseline.py` | new features `mp2grad`, `dfmp2` |

Timings are on one A100-SXM4-80GB, one process per card, after a warm-up
run. The CPU side is the §11 node (2× EPYC 7763, 128 threads).

### 14.1 P3 — the DF-MP2 relaxed density and gradient (`enable('dfmp2_grad')`)

**What it is.** The closed-shell RI-MP2 gradient. **PySCF has none**: every
PySCF DF-MP2 class raises in `nuc_grad_method`, and only `dfmp2_native` has
the relaxed density. GPU4PySCF's `DFMP2.Gradients` is a stub raising
`NotImplementedError`. Supported:

- an RHF reference, exact or density-fitted;
- frozen core;
- spin-component scaling (`mp.ps`, `mp.pt`, PySCF's names);
- through P4, a global-hybrid or pure RKS reference.

**Derivation.** It was checked first as a numpy prototype, to 4e-11 against
finite differences. In outline:

- E2 = Σ t̃ (ia|jb)_RI, with t̃ = (c_os + c_ss) t − c_ss tᵀ.
- The Hylleraas form gives dE2 = P·dF + Σ D2 d(ia|jb), with:
  - P_ij = −2 Σ t t̃ and P_ab = 2 Σ t t̃;
  - D2 = 2t̃;
  - the three-index two-particle density Γ^P_ia = Σ_jb D2 c^P_jb on the
    raw auxiliary index.
- The orbital response:
  - occupied-occupied and virtual-virtual rotations take the symmetric
    gauge (−½S¹), which is valid because the non-canonical Hylleraas
    functional is invariant to them;
  - frozen-active rotations follow from keeping the orbitals canonical
    (non-iterative, as `dfmp2_native` does);
  - the virtual-occupied ones go through one Z-vector.
- The result is a relaxed density P_rel and an energy-weighted W.
- The gradient is the sum of:
  - the SCF's own gradient at D;
  - h¹ and S¹ contracted with P_rel and W;
  - the skeleton cross term P_rel·(J − ½a_x K)¹[D], plus P_rel·V_xc¹[D] on
    a KS reference;
  - the non-separable 2 Σ (μν|P)¹ Γ^P_μν − Σ (P|Q)¹ Γ̃_PQ.

**On the GPU.**

- **Amplitudes.** Fitted (ia|P) come from GPU4PySCF's bdiv int3c2e. The
  amplitudes are blocked over the first occupied index and never stored
  whole; per block:
  - K is one GEMM;
  - t and t̃ come from a division kernel and a tiled transpose kernel
    (shared-memory 32×32: 2.2 s → 0.07 s per five taxol-sized blocks
    against CuPy's 4-D transpose);
  - P_vv is one 2-D GEMM per occupied orbital (CuPy's batched matmul with a
    transposed operand copies it — the same trap as §12.2's (T));
  - P_oo is a batched GEMM;
  - Γ is one GEMM;
  - E2 is read off Γ at the end (½ Σ Γ·(ia|P)) instead of a per-block
    reduction.
- **Y and Z** (the Lagrangian's Γ-integral contractions) come from one pass
  over the AO three-index integrals (GPU4PySCF's evaluator), not from MO
  integrals per orbital block: 10.2 → 1.2 s at penicillin V/TZVP.
- **The non-separable derivative** is GPU4PySCF's `sum_ejk_int3c2e_ip1` —
  the kernel of its DF J/K gradient — fed Γ as a general pair density, plus
  `int2c2e_ip1_per_atom`.
  - The kernel's normalisation was measured element-wise against PySCF's
    `int3c2e_ip1`/`ip2`/`int2c2e_ip1`: ½ on the 3c part, −1 on the 2c part.
  - The two MO indices are contracted larger-first, which keeps the pair
    density at nao²·min(ni, nj).
- **The SCF cross term on a DF reference.**
  - J goes by polarisation (any rank costs the same).
  - K goes through the same three-index kernel, with D's occupied factor
    on one side and P_rel in the MO basis on the other.
  - The first build polarised the full J/K gradient instead. That
    factorises a full-rank P_rel, which ran out of memory at taxol and took
    13 s at penicillin V/TZVP.
- **The Z-vector** is preconditioned CG through the SCF's own response
  (`mf.gen_response`), to |r| < 1e-7 |b|. GPU4PySCF's DF response builds
  stagnate near 3e-8 relative; that is their round-off.

**Correctness** (`tests/test_dfmp2_grad.py`, 8 checks, all pass on the
login node and on a compute node):

| check | worst |
|---|---|
| three-index derivative contraction against PySCF's ip integrals (random Γ) | 4.8e-13 relative |
| water/cc-pVDZ, all 9 components, against Richardson FD of the GPU DF-MP2 energy | 2.2e-10 |
| the same with frozen core | 2.0e-10 |
| formaldehyde/def2-SVP, frozen core, 4 components | 5.8e-10 |
| relaxed density against PySCF `dfmp2_native` (exact RHF), with and without frozen core | 1.6e-7 (PySCF's side, below) |
| the feature binding; a range-separated reference raises | — |

**Found on the way** `[X]`: PySCF's `dfmp2_native` relaxed density is good
only to about 1e-7.

- Its `cphf.solve`, at `cphf_tol` 1e-12, leaves a 2e-7 error in the
  Z-vector.
- Solving the same equations exactly (a dense solve in the prototype) moves
  the prototype's gradient from 4e-8 to 4e-11 against finite differences.
- The 1.6e-7 in the table is that error, not this code's.

**Timed** (frozen chemical core, DF-RHF with the default JK-fit basis, RI with
PySCF's default mp2fit basis):

| system / basis | nao / naux | GPU SCF | GPU SCF grad | GPU DF-MP2 energy | **GPU DF-MP2 gradient** | CPU: PySCF conventional MP2 gradient | CPU: PySCF `dfmp2_native` relaxed density alone |
|---|---|---|---|---|---|---|---|
| alanine dipeptide / SVP | 200 / 648 | 1.2 s | 0.3 s | 0.3 s | **1.2 s** | 27.7 s (23x) | 23.0 s (19x) |
| ferrocene / SVP | 221 / 763 | 2.4 s | 0.6 s | 0.6 s | **2.4 s** | 42.7 s (18x) | 27.7 s (12x) |
| ferrocene / TZVP | 415 / 1092 | 1.8 s | 0.9 s | 0.9 s | **4.6 s** | — | 73.6 s (16x) |
| penicillin V / SVP | 430 / 1418 | 2.0 s | 0.7 s | 0.6 s | **4.5 s** | 506.5 s (113x) | 262.3 s (58x) |
| penicillin V / TZVP | 858 / 2131 | 3.1 s | 1.4 s | 1.9 s | **15.3 s** | — | — |
| taxol / SVP | 1123 / 3690 | 12.1 s | 2.7 s | 10.9 s | **80.4 s** | — | — |

Reading:

- **There is no like-for-like CPU number.** PySCF has no DF-MP2 gradient.
  - Its analytic MP2 gradient is the conventional one, a different
    approximation, 18-113x slower.
  - Its DF route can produce only the relaxed density, which alone is
    12-58x slower than the whole GPU gradient.
  - A CPU DF-MP2 gradient today means finite differences: 6N energies.
- **The gradient costs 4-8x the GPU DF-MP2 energy.**
- **What the taxol gradient is** (80 s):
  - amplitudes, P and Γ: 27 s, 2.5x the energy's K GEMMs, as the extra
    Γ and P GEMMs make it;
  - the Z-vector: 22.5 s, 17 DF response builds;
  - the DF K cross term: 13.7 s;
  - everything else: under 5 s each.
- **Optimisation rounds** at penicillin V/TZVP: 39.9 s → 19.7 s (Y/Z
  through AO integrals; K cross term by the three-index kernel) → 16.8 s
  (batched P GEMMs) → 15.3 s (fused t̃, E2 from Γ, 2-D GEMMs for P_vv).
  Taxol: out of memory → 111 → 80 s.
- **What is left.**
  - The Z-vector's response builds (one DF J/K each), and the
    K cross term's (pi|P).
  - P_oo as one GEMM after a transpose kernel (now a batched GEMM over
    nv·ni small products, 1.6-1.9 s per five taxol blocks).

### 14.2 P4 — double hybrids (`electrolites.dhdft`, `enable('dhdft')`)

**What it is.** Neither PySCF nor GPU4PySCF has them. `B2PLYP` is a
`KeyError`: the wheel's libxc lacks its double-hybrid entries, and so do the
`HYB_GGA_XC_*` names. `DoubleHybrid(mol, name)` is:

- a GPU RKS with the functional's DFT part (a PySCF xc string);
- then GPU4PySCF's DF-MP2 on its orbitals, scaled per spin component:
  E = E_KS + c_os E_os + c_ss E_ss.

Built in: B2PLYP, B2GP-PLYP, mPW2PLYP, PBE0-DH and PBE-QIDH, with their
literature parameters. Any (xc, c_os, c_ss) works as a custom functional.

**The gradient** is §14.1's code on the RKS. Three things differ from RHF:

- the response in the Z-vector and in every G[·] of the Lagrangian is the
  KS one (`gen_response` with fxc and the scaled exact exchange);
- the K cross term is scaled by a_x;
- a new term, P_rel·V_xc¹[D] (`xc_cross_grad`): the AO derivatives of
  V_xc's basis functions contracted with P_rel, plus those of ρ_D under
  fxc·ρ_P contracted with D. It uses GPU4PySCF's own KS-gradient XC
  kernels, has no grid response (as its KS gradient), and covers LDA and
  GGA.

Meta-GGA, range-separated and NLC parts raise.

**Correctness** (`tests/test_dhdft.py`):

| check | worst |
|---|---|
| B2PLYP/water energy against the CPU recipe (PySCF DF-RKS + 0.27 PySCF DF-MP2, same grid and basis sets) | 1.7e-10 |
| B2PLYP gradient, all 9 components, against Richardson FD on a (99, 590) grid | 2.3e-7 |
| PBE0-DH, frozen core, 4 components | 3.1e-8 |
| a custom LDA double hybrid (0.5 HF + 0.5 Slater, 0.75 VWN5; c_os 0.25, c_ss 0.20), 4 components | 8.4e-8 |

The 2.3e-7 is the KS gradient's own: GPU4PySCF's RKS gradient of the same
DFT part, on the same grid, sits 2.3e-7 from its finite difference too
(grid response). The double-hybrid terms add nothing measurable to it.

**Timed** (DF-RKS, default grids, frozen core):

| system / functional | GPU SCF | GPU DH energy (MP2 part) | **GPU DH gradient** |
|---|---|---|---|
| alanine dipeptide / SVP / B2PLYP | 2.0 s | 0.3 s | **2.0 s** |
| ferrocene / SVP / PBE0-DH | 3.2 s | 0.7 s | **3.1 s** |
| penicillin V / SVP / B2PLYP | 3.1 s | 0.7 s | **7.6 s** |
| penicillin V / TZVP / B2PLYP | 5.3 s | 1.8 s | **16.4 s** |

The KS response makes the Z-vector 2-5x the RHF one: each build also
evaluates fxc on the grid. Everything else is as in §14.1. There is no CPU
double-hybrid gradient to compare with.

### 14.3 S1 — the COSX gate (not built)

§11.1 left S1 at rank 3 on one condition: a GPU seminumerical exchange
would have to beat `fastk`'s exact K at triple-ζ/diffuse sizes. The gate
measured the floor of such a build with the GPU kernel that exists for its
integrals.

- **The build.** K_μν = Σ_g w_g φ_μ(g) Σ_λ A_νλ(g) F_λ(g), with
  A_νλ(g) = (ν|1/|r−g||λ) from GPU4PySCF's `GINTfill_int3c1e` (the PCM/QM-MM
  kernel). It runs device-resident in grid slices, on PySCF's SGX level-2
  grids. The contraction is a batched GEMV plus a GEMM.
- **No screening**, so this is the lower bound of an unscreened sn-K.
- **The K it produces** sits 6e-5 to 2e-4 from the exact K: the expected
  COSX error, which sanity-checks the contraction.

`benchmarks/snk_gate.py`, one A100-80GB:

| system / basis (SGX grid) | nao | grid points | **sn-K build** (of it: the A integrals) | exact K (`fastk`) | ratio |
|---|---|---|---|---|---|
| penicillin V / SVP (pruned) | 430 | 122 576 | 19.4 s (19.1 s) | 0.08 s | 240x slower |
| penicillin V / TZVP (pruned) | 858 | 123 008 | 82.1 s (80.5 s) | 1.09 s | 75x slower |
| penicillin V / TZVPD (pruned) | 1074 | 123 648 | 130.6 s (123.5 s) | 3.14 s | **42x slower** |
| penicillin V / TZVPD (unpruned) | 1074 | 304 952 | 310.9 s (299.6 s) | 3.14 s | 99x slower |
| Ni dithiolene⁻ / TZVP (pruned) | 689 | 67 776 | 31.3 s (29.2 s) | 0.96 s | 33x slower |

**Reading.**

- **The integrals are 95-98 % of the build.** The kernel runs at
  5.8e8 grid-point × pair integrals per second at TZVPD (7.1e10 in 124 s).
- **Even the CPU beats it.** The CPU COSX K build of §11.1 at the same size
  is 8.3 s (116.8 s over 14 builds, 128 cores, screened, P-junction).
- **What would be needed.** A GPU sn-K would have to evaluate those
  integrals more than 40x faster before any screening to tie with `fastk`,
  and more still to win.
- **Verdict.** That is a new kernel project with an uncertain payoff, not
  S1's "a K-contracted kernel on what exists". **S1 moves to §5.4.**
  `fastk` stays the exact-exchange route on the GPU at every size measured,
  diffuse TZVPD included.

### 14.4 What §5 takes from Phase 3

| entry | status | measured |
|---|---|---|
| **P3** rank 4 | **built** | the RI-MP2 gradient PySCF lacks: 1.2-80 s, 4-8x the GPU energy; 12-58x PySCF's relaxed density alone, 18-113x its conventional MP2 gradient; FD to ≤ 6e-10 |
| **P4** rank 4 | **built** | double-hybrid energies (to 1.7e-10 against the CPU recipe) and gradients (FD to the KS gradient's own grid-response floor) |
| **S1** rank 3 | **gate failed → §5.4** | an unscreened GPU sn-K is 33-240x the exact K and 16x the CPU COSX at TZVPD |
