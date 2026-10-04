# What is left to optimize in GPU4PySCF 1.8.1, after Electrolites

**Who this is for**: whoever picks the next round of work in this repository,
and whoever is choosing entries for the LLM-GPU-QM benchmark task suite.

**What it is**: a second edition of the kernel-target catalogue
`LLM-GPU-QM/instructions/06.gpu4pyscf_kernel_targets.md` (called "the
catalogue" or "06" below), re-taken against the **1.8.1** source and against
what this repository has already done.  The catalogue is an input and is not
edited; where this survey contradicts it, the contradiction is recorded here
(§3, §7).  `docs/CATALOGUE_06_STATUS.md` is the per-entry record of what was
done; this document is the list of what is **not** done, plus the features
06 never surveyed.

**Scope**: every quantum-chemistry feature GPU4PySCF 1.8.1 ships — SCF of
every reference type, analytic derivatives of every order, response
properties, excited states and their derivatives, population and real-space
analysis, solvation, QM/MM, dispersion, ECP and relativistic Hamiltonians —
**and**, unlike 06, the methods 06 excluded: MP2/DF-MP2, CCSD, FCI, the PM6
semi-empirical code and periodic boundary conditions (§4 I–K).  Excluded:
the **molecular-dynamics** (`md/`) and **geometry-optimization**
(`geomopt/`) engines, which drive the methods above rather than being
methods; the energies and derivatives they call are in scope.

**What this document is not**: no optimization was performed.  The survey
itself (§1–§4, §6–§9) took no measurement: every number there is one already
measured in this repository's `docs/` (with the pointer), or one in 06
(marked as such), and everything else is a reading of the source.  The
eleven profiles §8 said were owed were then taken (2026-10-03) and are in
**§10**; **§5 is ranked on them**, and §3.4, §7 and §8 say which of their
items those profiles closed.

---

## 0. How to read this

The conventions are the catalogue's (06 §0), unchanged:

- `[V]` — verified against a primary source: the line was read in the 1.8.1
  tree for this survey, or the number was measured in this repository and is
  written in the `docs/` file cited.
- `[?]` — unverified or inferred.  **Never load-bearing.**
- `[X]` — contradicts a claim in circulation (in 06, or in this repository's
  own `docs/`); the corrected statement follows.

The two hard rules carry over verbatim in spirit:

1. **A share is a target-selection input, not a speedup.**  A component made
   K times faster moves the whole run by at most `1/(1-f+f/K)`.
2. **Nothing here is a speedup claim.**  The numbers this document quotes are
   this repository's own A/B timings, not `harness/evaluate.py` runs.

Three rules this repository learned on top of 06, which every entry below has
been read against (`docs/CATALOGUE_06_STATUS.md` §6):

- **A patch target is not a code path.**  Three rounds shipped a module that
  was loaded, correct and never called.  An entry's *caller* is named, and
  `benchmarks/coverage_probe.py` is the check before any coverage claim.
- **A share measured on GPU4PySCF is not the share left after Electrolites.**
  §2 quotes post-patch shares wherever they exist.
- **A share is not size-independent.**  Hessian entries reorder by a factor of
  six between 42 and 113 atoms (`docs/HESSIAN_1.8.1.md` §4); the open-shell
  transition-metal balance is the opposite of the organic one
  (`docs/OPENSHELL_TMC.md` §5).  Rows give both sizes where both exist.

**IDs.**  An entry that survives from 06 keeps its 06 ID (`A5`, `B9`, …).  A
new entry continues the numbering of its section (`A14`, `B12`, …).  §4 I, J
and K are new sections for the scope 06 excluded.  **Difficulty** follows 06:
easy / mod / hard / greenfield.  **Contam.** is training-data contamination
as 06 defines it (high = GPU4PySCF's long-public CUDA, moderate = public but
less trodden or recent, low = essentially nothing public to recall); it is
`[?]` everywhere, exactly as in 06 §0, and only matters for the benchmark
track.

**Status vocabulary** (as `docs/CATALOGUE_06_STATUS.md`): **open** = nothing
here touches it; **partial** = Electrolites covers part, the uncovered part is
named; **upstream** = closed in GPU4PySCF between 1.7.0 and 1.8.1;
**rejected** = measured here not to pay (not re-proposed; §3.3).

---

## 1. What was surveyed

| Item | Value |
|---|---|
| GPU4PySCF source | `/pscratch/sd/w/weiliang/g4psrc/g4p-1.8.1`, git `5b284c2` = **"1.8.1 release"**; the installed wheel in `barw-dev` reports `__version__ = '1.8.1'` `[V]` |
| Electrolites | this repository at `dd85215` (2026-10-03), every module and every file in `docs/` |
| Line numbers | all `file:line` references are **1.8.1** lines, relative to `gpu4pyscf/`, unless prefixed `electrolites/` |
| Method | five read-only passes over the source, one per area (SCF; derivatives; response and excited states; solvation, QM/MM, dispersion, ECP, analysis; methods outside 06's scope), each also reading this repository's modules for what they decline.  The load-bearing claims were re-read for this document (`fastgradh` vs 1.8.1 `grad/rhf.py:199-215`; `fastxc.py:503-505`; `scf/hf.py:150-167`; `properties/ir.py:41,91-97`; `properties/shielding.py:139-150`; `tdscf/uhf.py:750-754`; `lib/gdft/vv10.cu:286-364,760`; `hessian/rhf.py:470-472`; `hessian/rks.py:2009-2010`; `grad/uks.py:95-107`; `mp/dfmp2_addons.py:1157-1167`; `pbc/dft/multigrid_v2.py:1876-1895`) |
| Measurements | **none new**.  Hardware for the quoted ones: one A100 (SXM4-40GB, SXM4-80GB or PCIe-40GB, as the cited document says) |

---

## 2. Where the time is now, after `patch_all()`

The catalogue's §2 profiles are stock 1.7.0.  These are the post-Electrolites
profiles this repository has already measured on 1.8.1, which are the right
input for choosing what is left.  `↳` rows are sub-intervals of their parent.

### 2.1 Direct SCF — PfPMT (284 atoms), B3LYP/6-31G\*, 17 cycles (`docs/ROUND2_1.8.1.md` §2) `[V]`

| stage | stock | patched | patched share |
|---|---|---|---|
| **SCF total** | 122.7 s | **66.6 s** | 100 % |
| ↳ `get_k` | 83.1 s | 44.5 s | **66.8 %** |
| ↳ `get_j` | 21.7 s | 12.1 s | 18.2 % |
| ↳ `nr_rks` (XC) | 14.0 s | 6.8 s | 10.2 % |
| `eig` + DIIS + grids + hcore/guess | 2.3 s | 2.3 s | 3.4 % |

And the device's idle time over a whole patched SCF (`docs/DIRECT_SURVEY.md`
§1, `nsys`): the GPU is **90–94 % busy**; J and K run at 99 % busy with no
GPU4PySCF two-electron kernel left on the path; the idle time is the XC
build's — **34–38 % of `nr_rks`** is device idle (197k kernel launches, 864
`cudaFree`, 108 `cudaGetDeviceProperties` over PfPMT's SCF).  On open-shell
transition-metal complexes the balance is different again: `get_k` is
**84–88 %** of a stock UKS SCF and `nr_uks` 3–5 % (`docs/OPENSHELL_TMC.md` §5).

**Reading**: the direct SCF's two-electron builds are at the machine.  What is
left is (a) arithmetic that has to become *less or cheaper* (A20), and (b) the
XC build's host overhead and the kernel under it (A5, A15, A14).

### 2.2 Direct Hessian — B3LYP/def2-SVP (`docs/HESSIAN_1.8.1.md` §2–4) `[V]`

| entry | penicillin V (42 atoms) patched share | taxol (113 atoms) patched share |
|---|---|---|
| K in CPHF (`get_k`, `n_dm ≤ 3·natm`) | **37.3 %** | **54.9 %** |
| J in CPHF | 23.5 % | 24.5 % |
| `ejk_ip2` (B3) | 13.7 % | 3.0 % |
| `nr_rks_fxc_mo` | 10.1 % | 6.8 % |
| `jk_ip1` | 7.0 % | 2.5 % |
| XC Hessian (B5, done) | 4.1 % | 1.7 % |

(`fastjk`/`fastk` since took the K row 1.16x at 42 atoms — `docs/ROUND4_B4B.md`
§2 — and the round bounded what is left of it: `T = R + n_dm·C` with `C`, the
per-density contraction, **identical in both codes** and 94 % of the K build
at `n_dm = 126`.)  On an open-shell transition-metal UKS Hessian the order is
different: `ejk_ip2` **29.1 %** and `jk_ip1` **26.8 %** of a patched Ni
dithiolene Hessian (`docs/OPENSHELL_HESSIAN.md` §4c).

On a VV10 functional none of the above is the largest term:
`nr_rks_fnlc_mo` is **83.3 %** of a wB97M-V CPHF solve on alanine dipeptide
(112.88 s of 135.56 s), and `_get_vnlc_deriv1` is 31.31 s of `make_h1`'s
39.07 s (`docs/README_fastfnlc.md` §4b) `[V]`.

### 2.3 Density-fitted Hessian (`docs/DF_WORK_LIST.md` W44–W47, `docs/README_fastcphf.md`) `[V]`

Taxol⁺ UKS B3LYP/def2-SVP, whole Hessian 1324.5 s (80 GB card) after W44:
the DF CPHF exchange `_get_jk` is **1083.2 s = 82 %**, at 78 % of the card's
fp64 peak and 0.745 s per right-hand side — it cannot be rearranged (W8); only
the *number* of right-hand-side calls (339+336+336+201+80+339) and their
*precision* are open (W46).

### 2.4 Response properties — penicillin V/def2-SVP (`docs/RESPONSE_1.8.1.md` §2b, §4b; README) `[V]`

After `patch_all()` (with `fastjk`): B3LYP polarizability 16.77 → 8.06 s,
TDA 22.21 → 14.26 s.  Stock splits by functional, which sets what is left:

| functional | property | XC response | J + K | VV10 response |
|---|---|---|---|---|
| PBE | polarizability / TDA | 45.6 % / 47.7 % | 47.9 % / 40.5 % | — |
| B3LYP | polarizability / TDA | 24.8 % / 28.1 % | 71.4 % / 61.5 % | — |
| M06-2X | polarizability / TDA | 48.4 % / 47.0 % | 50.0 % / 45.3 % | — |
| wB97M-V | polarizability / TDA | 6.4 % / 34.3 % | 12.7 % / 58.8 % | **80.6 %** / excluded |

The wB97M-V polarizability's VV10 term is 124.7 → 87.2 s after `fastfnlc`
(1.43x) and is still the largest term of that run; 64.7 % of one call of it
is the single kernel `VXC_vv10nlc_hess_eval_f_t` (`docs/README_fastfnlc.md`
§4).

### 2.5 What has never been profiled

Every one of the following is a shipped 1.8.1 feature with **no profile in
06 or in this repository**, so every entry under it in §4 is "measure first":
TDDFT/TDA **gradients** and **non-adiabatic couplings** (incl. spin-flip and
the FSSH batch), **NMR** shielding, **IR** and **Raman**, **EDA**, **C6**,
TDDFT-ris, spin-flip TDA, the **PCM Hessian** and **SMD**, **QM/MM**
gradients, **ECP**-bearing systems, the restricted **meta-GGA SCF** (06 Q13),
**HSE06** after Electrolites, the **UKS range-separated gradient**, **GKS /
X2C**, **DF-MP2**, **CCSD**, **FCI**, **PM6**, and every **PBC** path.  §8
lists the profiles owed, in the order they would change the ranking.

**Since profiled (§10)**: TDDFT gradients and NACs (not spin-flip, not the
FSSH batch), NMR, IR, the PCM Hessian, the restricted meta-GGA SCF, HSE06
after Electrolites, the UKS range-separated gradient, DF-MP2, CCSD, and the
k-point hybrid SCF.  The rest of the list is still unprofiled (§5.2).

---

## 3. 06 against 1.8.1 and against Electrolites

### 3.1 Entries closed since 06 was written

| 06 entry | closed by | evidence |
|---|---|---|
| A2, B1, B2, B4b, B4d, B5, C1, D1, F2, G3 | Electrolites | `docs/CATALOGUE_06_STATUS.md` §1–4 (B5 now also meta-GGA and UKS; B4d now also the Hessian's `nr_rks_fxc_mo`, W16 — `electrolites/fastxcfxc.py:91`) |
| A3 (host Hermite transform) | upstream 1.8.0 | `scf/j_engine.py:139-142` `_dm_to_Rt`/`_Rt_to_dm` on the device; a residual D2H→H2D round trip remains at `j_engine.py:224`, and `electrolites/fastj.py:489` inherits it — small `[V]` |
| B6 (`hess_nuc_elec`) | upstream | `hessian/rhf.py:200 _hess_nuc_without_ecp` `[V]` |
| B11 (DF-Hessian aux integrals on CPU) | upstream | `df/hessian/rhf.py:737` GPU `int2c2e`; `:1147-1185` `e_int2c2e_ip2`; no `intor` left `[V]` |
| **D3** (third XC derivative on CPU libxc) | **upstream** — `CATALOGUE_06_STATUS.md` still lists it open `[X]` | no `LIBXC_ON_GPU` anywhere in 1.8.1; `grad/tdrks.py:389,441`, `grad/tduks.py:410`, `nac/tdrks_grad_nacv.py:148` call GPU `ni.eval_xc_eff(..., deriv=3)` `[V]`.  The per-block structure around the call is still slow: that is D9 below |
| **D7** (spin-flip XC on CPU) | **mostly upstream** `[X]` | `tdscf/_uhf_resp_sf.py:192-195` goes to the CPU only for a PySCF `NumInt`; the `ni.to_cpu()` is gone `[V]`.  What remains is the 20-sample multicollinear quadrature (D13) |
| B8 (Hessian 1e integrals on CPU) | **half upstream** | first derivatives on the GPU (`grad/rhf.py:211-212`, `hessian/rhf.py:634`); the second derivatives are still CPU libcint (`hessian/rhf.py:158-159`, `:636-637`) — still an Amdahl tail `[V]` |

### 3.2 Entries still open, with their 1.8.1 location

| 06 entry | 1.8.1 location `[V]` | carried to |
|---|---|---|
| A1 short-range K | `lib/gvhf-rys/rys_roots.cu` `else` branch; Electrolites serves HSE by decomposition (`electrolites/fastk.py:709-719`) | A17 (re-framed) |
| A4 restricted mGGA in the SCF | `electrolites/fastxc.py:503-505` accepts `('LDA','GGA')` only for RKS; `nr_uks` accepts MGGA (`:490-492`) | A14 |
| A5 AO kernel | `lib/gdft/nr_eval_gto.cu:363-400` (one shell per `blockIdx.y`), `:394`, `:499` (run-time `nprim`), `:1804-1812` (one launch per contraction pattern per block) | A5 |
| A7 RSH fusion, high-`l` classes | `electrolites/fastk.py:661-675` (`IMPLEMENTED2`, one pass per operator) | A7 |
| A8 Rys roots | `rys_roots.cu:24` run-time `nroots`, `:42` division | reached only by classes that fall through; no entry |
| A9 union screening | `scf/jk.py:419,562,712` `condense('absmax', dms)` | rejected as B4c (1.14–1.18x) |
| A10 libxc dispatch | `lib/gdft/libxc.cu:29-92` (ADD macros, `_add_out`), `:418-475`; `dft/libxc.py:128-144` allocates output + scratch per label per call; `_memset_*` is a synchronous `cudaMemset` on the default stream | A15 |
| A11 Becke locality | `lib/gdft/gen_grids.cu:73,87`; default `original_becke` (`dft/gen_grid.py:452-453`) | tail: `grids.build` 1.7 % of a patched PfPMT SCF; re-measure at 500+ atoms |
| A12 ECP screening | `gto/ecp.py:80,101` `# TODO: Add screening`; full `np.meshgrid` cross product on the host `:75-114`; one launch per group `:166-182` | F8 |
| B3 `ejk_ip2` | 7 unrolled classes per type file (`unrolled_ejk_ip2_type12.cu:11`); general kernel in `rys_contract_jk_ip2.cu` | B3 |
| B4a J-engine `n_dm` cap | `scf/j_engine.py:299-301`; `md_contract_j.cu:1008` | bounded at 1.09x (`docs/ROUND3_B4A.md` §4); no entry |
| B7 ECP Hessian | `hessian/rhf.py:757-791` | F8 |
| B9 grid response | `lib/gdft/gen_grids.cu:317,342,476,514`; `hessian/rks.py:2009-2010` | B9 |
| C2 NMR | `properties/shielding.py:171-173,203,240-244` | C6–C9 |
| C3 IR/Raman | `properties/ir.py:105-106`; `raman.py:67-69` | C10 (the real cost is elsewhere) |
| C4 EDA | `properties/eda.py:217,234,581,832`, `:284-325` | C4 |
| C5 C6 | `properties/c6.py:23-110` — **mechanism changed**: full A/B via `get_ab`, then CPU `cholesky`/`eigh` on `2·nov` | C5 |
| D4 `get_ab` in-core | `tdscf/rhf.py:135,180`; `uhf.py:171,236,478,538` | D4 (now on a user path through C5) |
| D5 TDDFT+PCM pivot sync | `tdscf/rhf.py:91-96`, `uhf.py:108-113` — only inside `get_ab`'s `add_solvent_`, not the Davidson path (`solvent/tdscf/pcm.py:136`) | folded into D4 |
| D6 ris | `tdscf/ris.py:382,453,518` | D6 |
| E1–E4 | `scf/hf.py:923-926` (`analyze` → `to_cpu()`), `:903-913` (`dip_moment` via host), `tools/` (no real-space driver), `pop/esp.py:52-102`, `qmmm/chelpg.py:65-99` | E1–E5 |
| F1, F7 dispersion | `dispersion/dftd3.py`, `dftd4.py`, `gcp.py` re-export `pyscf.dispersion` | F1 |
| F4 PCM gradient/Hessian host loops | `solvent/hessian/pcm.py:66-148,363-368,506-530,637-665` (partly cleaned by #809) | F9–F11 |
| F5 external field | `qmmm/external_field.py:93,162,177` | not a target (once per run, O(nao²)) |
| F6 QM/MM nuclei–MM | `qmmm/itrf.py:97,102-103,221,227-230,246-249` | F12 |
| G4 multi-GPU | `lib/multi_gpu.py:99` | G5 |

### 3.3 Measured and rejected here — do not re-propose

Each is a measurement, kept so that it is not rediscovered.  Full table:
`docs/CATALOGUE_06_STATUS.md` §5; DF ones: `docs/DF_WORK_LIST.md` §2.

| idea | measured | where |
|---|---|---|
| per-lane `switch (gout_id)` unrolling | no net gain, 444 000 lines | 06 §3.3 |
| `vk` block reduction (JoltQC `pair_vk`) | ceiling 1.1 % / 2.2 % | `ROUND2_1.8.1.md` §3b |
| JoltQC warp-aggregated task atomics | 0.995–0.999 | `UPGRADE_1.8.1.md` §9 |
| post-hoc log-magnitude AO screen | 0.28x; headroom was a `minao` artifact | `ROUND2_1.8.1.md` §3c |
| fused grid pass without batching | 0.54x | 06 §3.6 |
| bigger grid blocks (4096 → 32768) | 4x slower | `DIRECT_SURVEY.md` §2.1 |
| 8 or 16 densities per Coulomb pass | 0.90x / 0.64x; cap worth ≤ 1.09x | `ROUND3_B4A.md` §4 |
| per-density screening (B4c) | 1.14–1.18x ceiling; densities are not atom-local | `ROUND4_B4B.md` §5 |
| transposed multi-density layout | 0.30–0.38x | `ROUND4_B4B.md` §7 |
| `ejk_ip1` occupancy caps | 1.70x → 0.42x | `README_fastejk.md` |
| `ip2` P1 / P3 | 1.02x / 0.58x | `ROUND4_B4B.md` §6 |
| grouped GEMM in XC Hessian / XC response | 1.1 % / 0.72 s vs 0.22 s | `README_fastxchess.md` §1, `README_fastxcfxc.md` §2 |
| AO cache across response iterations (D8) | 2.2 % of the run | `RESPONSE_1.8.1.md` §3.1 |
| `lr_eigh` preallocation (D2) | 0.19 GB over a run | `RESPONSE_1.8.1.md` §5 |
| splitting the fused response J+K | 0.70–0.93x | `RESPONSE_1.8.1.md` §4 |
| `fastfnlc` cache on a Hessian; VV10 grid trim | 1.03x; 6–8 % | `README_fastfnlc.md` §4b, §4d |
| local `nocc` in the CPHF grid route | falsified by count | `README_fastxcfxclocal.md` |
| DF: AO-batched first contraction, own unpack kernel, blocked triangular GEMM, localized occupied index, density screen in the 3c kernels | 1.00x / 2.65x slower / 0.95x / 1.000x / off | `DF_WORK_LIST.md` §2, W15, W20, W26 |
| int3c1e error-budget screen (P11) | 21–33 % of tasks, all on the cheap branch | `README_fast1e.md` |
| PCM D/S matrix-free (06 F3), SCF path | off the default path; 0.026 s of 141 s | `README_fast1e.md` |
| eigensolver, DIIS (06 H1/H2) | 1.9 % together with grids | `ROUND2_1.8.1.md` §2 |
| seminumerical exchange (COSX) | greenfield; `sem/` is semi-empirical, not seminumerical | `DF_WORK_LIST.md` "Not on this list" |

### 3.4 Corrections and coverage defects found by this survey

These are not optimization targets; they are things that are wrong *now*, and
two of them could cost time on the patched path.

| # | finding | evidence | consequence |
|---|---|---|---|
| R1 | **`fastgradh` may be a regression on 1.8.1** `[X]` | 1.8.1's own `grad/rhf.py:199-215 get_hcore` builds `int1e_ipkin` on the GPU (`pbc.gto.int1e`) and `int1e_ipnuc` through the int3c2e ip1 path (`:217-240`) `[V]`; `electrolites/fastgradh.py:45` replaces it with CPU `mol.intor('int1e_ipkin', comp=3)` `[V]`.  The module was written against 1.7.0, where both were CPU | the patched Hessian's `make_h1` (reached via `get_grad_hcore`, `grad/rhf.py:396-398`; twice for UHF) pays CPU libcint that stock does not.  Not measured: A/B it, then retire or narrow the module.  **Measured (§10.7): not a regression — 0.06 s against stock's 0.14 s, 0.1–0.3 % of a Hessian; keep the module** |
| R2 | **`get_hcore` is not "CPU" on 1.8.1** `[X]` | `scf/hf.py:150-167`: nuclear attraction via `contract_int3c2e_auxvec`, kinetic via `pbc.gto.int1e.int1e_kin`, ECP via `get_ecp` — all GPU `[V]` | `docs/DF_WORK_LIST.md` §3.1 and `docs/README_fastdf.md` §3.1 label a 21.8 % / 13.9 % `get_hcore` row "on the CPU".  The *number* was measured; the *label* is not what the code does.  Same stage is 0.77 s of a 284-atom direct SCF (`ROUND2_1.8.1.md` §2).  Unexplained until profiled (first-call JIT/module load is the obvious suspect `[?]`) — §8 item 9.  **Profiled (§10.9): first-call initialisation, 0.46–0.52 s cold against 0.10 s warm, independent of thread count** |
| R3 | `ROUND2_1.8.1.md` §6's "unpatched nuclear-attraction derivative on the CPU" vs the 1.8.1 source | `grad/rhf.py:242-293 _grad_nuc_without_ecp` and `:217-240` are GPU `[V]`, yet §6 *measured* the stock PfPMT gradient depending on `OMP_NUM_THREADS` | a measurement and a source reading disagree; something on the stock gradient path is CPU and is not the term §6 names.  Profile (§8 item 9) — do not explain it by inspection.  **Profiled (§10.9): does not reproduce on 1.8.1 — the stock PfPMT gradient is 19.79 s at 1 and at 16 threads, with no CPU integral in its profile** |
| R4 | `CATALOGUE_06_STATUS.md` rows stale | D3 and D7 closed upstream (§3.1); B4d's Hessian half done by W16; "`_get_vnlc_deriv1` is 23.1 % of `make_h1`" — the underlying numbers (31.31 s of 39.07 s, `README_fastfnlc.md` §4b) make it **80 %** of `make_h1`; 23.1 % is its ratio to the 135.56 s CPHF | read §2.2 here instead |
| R5 | **UKS TDDFT ignores `exclude_nlc`** `[V lines; consequence ?]` | `tdscf/uhf.py:752-754` `gen_response(mo_coeff=None, mo_occ=None, hermi=hermi)` drops `with_nlc`; the restricted `tdscf/rhf.py:393,408-415` honours it; `grad/tduks.py:395` runs without NLC | an open-shell VV10 TDDFT energy includes the VV10 response while its gradient does not, and every Davidson iteration pays the VV10 term (`fastfnlc` covers it, 1.47–1.63x).  A semantics fix to report upstream, not a speedup |
| R6 | CCSD skips shell pairs with `l > LMAX_ON_GPU = 4` (`cc/ccsd_incore.py:281,396,472`; `scf/int4c2e.py:27`) `[V]` | atomic bases are limited to g (`l = 4`) in 1.8.1's README, so this cannot fire on a supported basis | not a bug today; it becomes one the day h shells are allowed |
| R7 | `nac/tdrks_grad_nacv.py:148` uses `spin=0` on a doubled triplet density where `grad/tdrks.py:441` uses `spin=1` | dead code: `:255` raises on non-singlets `[V]` | none now |


---

## 4. The catalogue

Column **Electrolites** says what of the entry this repository already
reaches and what it hands back (the gating condition, with its line).
**Share** is a measured number with its document, or `not measured [?]`.

### A. Single-point energy (SCF)

| ID | Target | Evidence (1.8.1) | Mechanism | Electrolites | Share | Direction | Diff. | Contam. |
|---|---|---|---|---|---|---|---|---|
| **A5** | **the grid AO-evaluation kernel** (06 rank 3, the kernel half) | `lib/gdft/nr_eval_gto.cu:363-400`, `:463-505`: one shell per `blockIdx.y`, run-time `nprim` bound (`:394`, `:499`), no per-point radial cut, atom-centred `rx,ry,rz,rr` recomputed per shell; `:1804-1812` one launch per contraction pattern per grid block `[V]` | the `exp` loop cannot unroll; far points pay full `exp` sets; ~4 300 `_cart_kernel_deriv1` launches per `nr_rks` call on PfPMT (`DIRECT_SURVEY.md` §2.1) | **partial**: the repetition is gone (`fastxc`, `fastxcnlc`); the kernel is untouched | 06: 18–21 % of stock SCF (1.7.0); `eval_ao` at deriv 2 is **26 %** of the patched XC-gradient block (`README_fastxcgrad.md` "What is left"); patched XC build is 10.2 % of a PfPMT SCF | template on `nprim`; skip `exp` past `-a_min r² < ln(cut)` (bitwise-safe below the cut); one block per (atom, grid tile) over all its shells; one launch per block over all patterns from a task table | mod | high |
| **A7** | RSH fusion for the high-`l` (`gen_khigh`) classes | `electrolites/fastk.py:661-675` — "These still take one pass per operator" `[V]` | P5's payoff never collected on the widest classes | open (the one-thread-per-quartet classes are fused) | 06 sizes it ~9 % of K for RSH; not re-measured `[?]` | carry the fused-range accumulation into the generated classes | easy | high |
| **A14** | **restricted meta-GGA XC in the SCF** (TPSS, M06-2X, r2SCAN, SCAN, MN15) | `electrolites/fastxc.py:503-505` accepts `('LDA','GGA')` for RKS, while `nr_uks` accepts MGGA (`:490-492`); stock `dft/numint.py:427-544` is two passes, per-block `_tau_dot` + `contract` (`:528-532`); with VV10, `electrolites/fastxcnlc.py:163-212` fuses but walks `ni.block_loop` one block at a time `[V]` | P7+P8 never applied to the most common restricted mGGA case — though the open-shell twin has it | **open** for RKS mGGA without VV10; fused-not-grouped with VV10 | **not measured — this is 06 Q13**; analogues: mGGA gradient 3.98–4.46x (`README_fastxcgrad.md`), mGGA response is 47–48 % of a stock M06-2X response property (`RESPONSE_1.8.1.md` §2b) | port `_nr_uks_grouped`'s tau terms to `_nr_rks_grouped`; group `nr_rks_nlc` | **easy** (the code exists for UKS) | high |
| **A15** | **`nr_rks` host overhead and libxc dispatch** (06 A10, re-sized) | `DIRECT_SURVEY.md` §1–2: 80 `cudaFree` per call around libxc, ten 3.3 ms `cudaGetDeviceProperties` per call in `lib/cupy_helper/grouped_gemm.cu:38-46` / `grouped_dot.cu:42`; `dft/libxc.py:128-144` allocates output + scratch `(2, np, size)` per label per call; `lib/gdft/libxc.cu:29-92` one `_add_out` per output per component, synchronous `cudaMemset` on the default stream `[V]` | device idle 34–38 % of `nr_rks`; allocator synchronisation (the DF rounds' recurring finding, `DF_WORK_LIST.md` §5.8) | **partial**: `fastxc` groups the GEMMs, inherits the rest | bounded by `nr_rks`'s idle: **1.16 s of 16.1 s (7 %)** taxol, **2.7–3.0 s (4.5–5 %)** PfPMT (`DIRECT_SURVEY.md` §2) | cache device properties; preallocate and reuse libxc outputs; accumulate components in place; stream-ordered memsets | easy–mod | moderate |
| **A17** | **native short-range (`erfc`) K for HSE06** (06 A1, re-framed) | `electrolites/fastk.py:709-719` serves `lr·erf + sr·erfc` as `sr·(full) + (lr−sr)·(erf)` screened on a sibling full-range `q_cond` (`_full_range_opt`, `:564-589`); stock screens the short-range operator with `s_cond` `[V]` | for HSE (lr = 0) the patched path builds **two dense** K matrices where stock builds **one screened** short-range K; the short-range operator's locality is thrown away `[?]` | "served" by decomposition; `hermi != 1` falls back | **not measured** — it may be slower than stock at 284+ atoms `[?]`; §8 item 5 | `erfc` Rys roots in the P1–P5 kernels with an `s_estimator` screen; drops the sibling `_VHFOpt` | mod | high |
| **A18** | K classes with fewer bra pairs than SMs | `DIRECT_SURVEY.md` §2.4: `n_blocks = min(432, npairs_ij)`, 2–29 blocks on PfPMT's small groups | under-filled launches; depends on chemistry (singleton shell groups), not size — likely larger on transition-metal complexes | open | **1.056x** on PfPMT 6-31G\* `get_k`, 1.02x def2-SVP, 0 on taxol (login node only) | split the ket list across streams | easy | high |
| **A19** | J classes beyond `(6,6)` (g shells) and g-shell `ejk` classes | `ROUND2_1.8.1.md` "What is left" 3; `OPENSHELL_TMC.md` §4 item 7: g shells are 26 % of class blocks at def2-TZVPP on 3d metals | g-function bases fall through to GPU4PySCF's general kernels | open | not measured `[?]` | 13 of 17 classes are a table entry and a sweep; `(7,7)`, `(8,6)`, `(8,7)`, `(8,8)` need a new `Rt` strategy | mod | high |
| **A20** | **less or cheaper arithmetic in direct K** | `get_k` is **66.8 %** of a patched PfPMT SCF (`ROUND2_1.8.1.md` §2) and runs 99 % busy with every class already this repository's (`DIRECT_SURVEY.md` §1) | at the machine: rearrangement is exhausted (§3.3); only precision or a looser per-cycle threshold is left | none (fp32 exists only for the DF cderi, `README_fastdffp32.md`) | **66.8 %** of a patched SCF; 84–88 % of a stock UKS TMC SCF | fp32 Rys for quartets whose Schwarz bound is below a budget (06 P10's argument), or incremental Fock builds with a cycle-dependent threshold; fp64 near convergence | hard | low–moderate |
| **A21** | GKS multicollinear XC | `dft/numint2c.py:101-207` two passes per block, 8 `add_sparse` per block; `dft/mcfun_gpu.py:24` `spin_samples=770`, libxc at `deriv+1` per sample batch (`:84`, `:195-235`); locally-collinear raises (`numint2c.py:124`) `[V]` | ~770 polarized libxc evaluations per grid point per cycle | none | not measured `[?]` | symmetry-reduced or adaptive sampling; fuse the reductions; group blocks | mod | low |
| **A22** | basis sets past g run J/K on the CPU | `scf/jk.py:49` `LMAX = 4`, `:343-346`, `_vhf.direct_mapdm` at `:541,687,829`; `scf/j_engine.py:41,86-89,226-241`; `df/int3c2e_bdiv.py:46` `[V]` | h shells go to host libcint | `electrolites/fastk.py:704-706`, `fastj.py:662` fall back | not measured `[?]`; outside the README's supported range (atomic bases to g) | GPU classes for `l = 5` from the generators | hard | low–moderate |

### B. Analytic derivatives (gradients, Hessians, CPHF)

| ID | Target | Evidence (1.8.1) | Mechanism | Electrolites | Share | Direction | Diff. | Contam. |
|---|---|---|---|---|---|---|---|---|
| **B12** | **the per-density contraction `C` in the CPHF exchange build** | `lib/gvhf-rys/unrolled_rys_k.cu:165` `for i_dm` innermost; `T = R + n_dm·C` with `C` **1.01x–1.16x** between the two codes (`ROUND4_B4B.md` §4) `[V]` | 94 % of the K build at `n_dm = 126`; `4·nf²` `atomicAdd`s per quartet per density — the only unexamined half of 06 Q4 | the integrals (`R`) are done; `C` is identical code | `get_k` **37.3 % / 54.9 %** of a 42/113-atom patched Hessian (`HESSIAN_1.8.1.md` §4) | (a) measure the atomic traffic (`ncu` is unavailable on this machine — `README_fast1e.md`; needs another route); (b) **inexact-Krylov relaxation**: loosen the integral threshold on later CPHF iterations as the residual falls — the direct-path twin of W46 `[?]` | hard | high |
| **B13** | **`VXC_vv10nlc_hess_eval_f_t`, the VV10 response/Hessian pair kernel** | `lib/gdft/vv10.cu:286-369` off-diagonal kernel: untiled `j` loop, fp64 division; `:760` `n_trial_per_thread = 6` with `blockIdx.y` over trial groups — so each `(i, j)` pair term (~40 flops + a division) is evaluated once per **six** trial vectors, `ceil(ntrial/6)` times in all (57 times at `ntrial = 3·natm`, 113 atoms) `[V]`.  Callers `hessian/rks.py:930` (`_get_enlc_deriv2`), `:2243` (`_get_vnlc_deriv1`), `:3806` (`nr_rks_fnlc_mo`) `[V]` | O(ngrids²·ntrial/6) pair evaluations where O(ngrids²) evaluations + an O(ngrids²·ntrial) contraction suffice; no tiling, no screen, no fp32 | none: `fastfnlc` calls the same `.so` entry (`electrolites/fastfnlc.py:417`).  **`CATALOGUE_06_STATUS.md` §7 item 7's obstacle — "exists only compiled" — is gone: the CUDA source is in the 1.8.1 tree** | `nr_rks_fnlc_mo` **83.3 %** of a wB97M-V CPHF; `f_t` **64.7 %** of one call; VV10 response **80.6 %** of a stock wB97M-V polarizability, still the largest term after `fastfnlc` (`README_fastfnlc.md` §1, §4, §4b) | evaluate the pair kernel once per `(i, j)` tile and contract against all trials (a GEMM with the kernel built on the fly); `fastnlc`'s MUFU reciprocal, fp32 beyond 10 Bohr (P10) and error-budget trim (P11) | mod | moderate (public only since 1.8.0) |
| **B14** | VV10 Hessian host side: `_get_vnlc_deriv1`, `_get_enlc_deriv2` | `hessian/rks.py:2381` (RKS), `:2341` (UKS): `for i_atom` inside the NLC block loop, ~10 contractions per atom per block; dense `(natm, 3, ngrids)` at `:2187-2189,2240,872-873,927-928`; the O(ngrids²) `UWABCE` recomputed in each (`:842`, `:2160`) `[V]` | B5's launch-bound atom loop, on the VV10 grid; O(natm·ngrids) memory | none (declined by `fastxchess`; open in `VV10_OPENSHELL.md` §5 item 3) | inside `_get_vnlc_deriv1`'s 31.31 s of 39.07 s `make_h1`; the split against B13 is not measured `[?]` | `fastxchess`'s batching of the atom axis; share the ground-state `UWABCE` with `fastfnlc`'s cache | mod | low |
| **B15** | **`_get_jk_ip1`, the direct Hessian's `h1` two-electron term** | kernel `lib/gvhf-rys/unrolled_rys_jk_ip1.cu:134` (division/sqrt per primitive), `:159` (run-time `omega`), `:167` (per-root division), no launch bounds; unrolled only to d.  Driver `rys_contract_jk_ip1.cu:1705,1807,1903,2010` call `cudaGetDeviceProperties` (+ a `cudaMemset`) per class launch.  `hessian/rhf.py:463` rebuilds `_VHFOpt` per call; `:472` `assert n_dm == 1`, so UKS calls it per spin (`hessian/uhf.py:181-182`, `uks.py:168-169,186-187`) `[V]` | B1's anti-patterns before `fastejk` (P1–P5), plus host overhead, plus a 2x integral recomputation for open shells | none, either spin | **7.0 % / 2.5 %** of a patched 42/113-atom RKS Hessian; **26.8 %** of a patched Ni-dithiolene UKS Hessian (`OPENSHELL_HESSIAN.md` §4c); 1.40 s device idle, 188 gaps > 1 ms (`DIRECT_SURVEY.md` §1) | (a) query properties once, reuse the SCF's `vhfopt` — easy; (b) one call at `n_dm = 2` (J on the total density, K per spin); (c) emit `vj`/`vk` ip1 classes from `gen_ejk`/`gen_khigh` | mod (a: easy) | high |
| **B3** | `ejk_ip2`'s **general** kernel | 7 unrolled classes per type file (`unrolled_ejk_ip2_type12.cu:11`); the rest in `rys_contract_jk_ip2.cu`'s general kernel `[V]` | padded tiles, run-time indices — the P6 problem `gen_khigh` solved for K | `fastejk2` changes the launch scheme only (1.13x); P1/P3 measured not to pay | general kernel = **63 %** of `ejk_ip2` at def2-SVP (`ROUND4_B4B.md` §6); `ejk_ip2` 13.7 % / 3.0 % of a patched 42/113-atom Hessian, **29.1 %** of a patched UKS TMC Hessian | exact-tile generator per class (extend `gen_ejk` to second derivatives) | hard | moderate |
| **B16** | the short-range half of RSH derivatives on stock code | `grad/uks.py:95-107` sets `omega = -omega` ("Prefer computing the SR part"); the Hessian does the same (`hessian/rks.py:76-89`, roots doubled at `rhf.py:399-400`); `electrolites/fastgrad.py:68-77` rewrites it as `K − K(erf)` for `grad.rks` only, and `fastejk.py:158` declines `omega < 0` `[V]` | the short-range operator doubles the Rys roots and runs the stock kernel | none for UKS gradients or either Hessian's ip2 | RKS analogue: the 2e derivative **1.77x** on PfPMT wB97M-V/def2-TZVPD (`README_fastejk.md`); UKS not measured `[?]` | port `fastgrad`'s identity to `grad.uks` and to the Hessians' ip2; fuse the two passes | easy | high |
| **B9** | grid-response weight derivatives | `lib/gdft/gen_grids.cu:317` first derivative loops every atom (`:342`), no screen; the off-diagonal second derivative (`:476`, `B` loop `:514`) is one thread per (point, G, H) looping B → **O(ngrids·natm³)**; `hessian/rks.py:2009` materialises `natm²·9·ngrids` per 4096-point block and `:2010` asserts it is below 2³¹ — which fails above ~240 atoms by arithmetic `[V; the 240 is computed, not run]`; `grad/rks.py:506-507,680-681` two host syncs per block | 06 A11's locality problem at third order, plus an int32 wall | declined by `fastxcgrad` and `fastxchess` | not measured; off by default `[?]` | fuse each derivative kernel with its contraction so `d2w` is never stored; skip B where the switching function saturates | mod | low |
| **B17** | DF CPHF: the right-hand-side count and precision (`DF_WORK_LIST.md` W46), the solver's own time (W45), the 40 GB chunk count (W47) | `DF_WORK_LIST.md` §4 W45–W47 `[V]` | W46: the Krylov tolerance is 1e-6 per atom, yet every image is fp64 | `fastdfhess`, `fastcphf` on the path | `_get_jk` **82 %** of taxol⁺'s UKS DF Hessian; solver 27.5 s of 1246.6 s after W44 | as written in W45–W47 | mod–hard | low |
| **B18** | dense `s1ao` per atom | `hessian/rhf.py:100-110`, `:708-713`; `uhf.py:97-114`; copied into `electrolites/fastcphf.py:475-480,596-600` `[V]` | O(natm·nao²·nocc) where O(nao²·nocc) suffices | none | not measured; < 0.2 s at 113 atoms by arithmetic `[?]` — a tail | use rows `p0:p1` | easy | low |
| **B19** | ROHF/ROKS gradients and Hessians do not exist | `scf/rohf.py:103`, `dft/roks.py:44` `nuc_grad_method = NotImplemented` `[V]` | missing, not slow | n/a | n/a | gradient from the UHF machinery + an ROHF energy-weighted density; Hessian greenfield — **must not be pooled into speedup statistics** (06 B10's rule) | mod / greenfield | low |

06 **B7** (ECP Hessian) and **B8** (CPU second-derivative 1e integrals) are
under F8 and §3.1 respectively.

### C. Response properties

| ID | Target | Evidence (1.8.1) | Mechanism | Electrolites | Share | Direction | Diff. | Contam. |
|---|---|---|---|---|---|---|---|---|
| **C10** | **IR intensities solve the Hessian's CPHF twice** | `properties/ir.py:41` `hessian_obj.kernel()`, then `:91-97` a second `make_h1` and `solve_mo1`, under upstream's own `# TODO: compact with hessian method, which can save one time cphf solve` `[V]`; `hessian/rhf.py:54-86 hess_elec` already accepts `mo1`/`mo_e1`/`h1mo` | the whole 3·natm-RHS CPHF and `make_h1` repeated | indirect: the Hessian modules speed both copies | not measured for IR; **derived** from `HESSIAN_1.8.1.md` §2–3: `make_h1 + solve_mo1` are 80 % / 94 % of a 42/113-atom Hessian, so the duplicate is ~44–48 % of an IR run `[?]` | solve once, pass `mo1` into `hess_elec`, reuse it for the dipole derivative | **easy** | low |
| **C11** | Raman by finite differences | `properties/raman.py:81-93`: a Hessian plus `polarizability_derivative_numerical_dEdE` ≈ 19 SCF + response runs `[V]` | numerical third derivative | indirect | not measured `[?]` | analytic polarizability derivative (greenfield; inherits C10's CPHF) | greenfield | low |
| **C6** | **NMR: the GIAO two-electron term on the CPU** | `properties/shielding.py:139-150` `_vhf.direct_mapdm('int2e_ig1', …, dm0.get(), 3, …)` under `# TODO: this can be modified into jk, not _vhf` `[V]` | O(N⁴) single-node libcint, three components, needed even for a pure functional (J) | none | not measured; likely the dominant NMR cost at size `[?]` | a GPU `(ig1)` Rys kernel — the ig factor is an origin shift on the ip1 machinery | greenfield–hard | **low** |
| **C7** | NMR XC: GIAO AOs evaluated on the CPU per grid block | `shielding.py:91`, `:101-102` `eval_gto('GTOval_ig'/'GTOval_ipig', coords.get())` over the full `nao`; Python `for idirect` at `:94`, `:107` `[V]` | a D2H/H2D round trip and a CPU AO evaluation per block, ×9 components | none | not measured `[?]` | form ig/ipig on the GPU from `eval_ao` deriv 1–2 times `(R_μ × r)`; no new integrals | easy–mod | low |
| **C8** | NMR CPHF exchange called badly | `shielding.py:51`, `:225` `jk.get_jk(mol, dm1, with_j=False)`; `scf/jk.py:66-67` rebuilds `_VHFOpt` each call; `hermi` defaults to 0 on an antisymmetric density, so `jk.py:443-445` stacks transposes (2x densities) `[V]` | per iteration: an option rebuild, doubled densities, a wasted J | reached by `fastjk` (which also doubles at `hermi = 0`) | not measured `[?]` | `mf.get_k(mol, dm1, hermi=2)` | easy | low |
| **C9** | NMR per-atom CPU 1e integrals | `shielding.py:237-244` `set_rinv_origin` + `int1e_prinvxp` / `a01gp` / `giao_a11part` per atom `[V]` | natm CPU libcint calls | none | not measured `[?]` | GPU rinv-type integrals batched over atoms (the `int3c1e` family) | mod | low |
| **C12** | RSH response: J+K and the long-range K in two passes | `scf/_response_functions.py:72-75,94-97,171-174`; `grad/tdrks.py:100-105` `[V]` | the second pass repeats screening and density loads | both passes patched (`fastjk`, `fastk`), not fused; `fastk.get_k_rsh` fuses at `hermi = 1` only | fusion worth 1.12–1.16x of K in the SCF (`README_fastrsh.md`); wB97M-V TDA J+K is 58.8 % (`RESPONSE_1.8.1.md` §2b) | fused `J + a·K + b·K_LR` in `fastjk`, incl. `hermi = 0` | mod | high |
| **C4** | EDA | `properties/eda.py:217,234,581,832` CPU `int1e_ovlp`/`int1e_kin`; `:666,673,698` multipoles; `:284-325` 4-deep fragment loop `[V]` | O(n_frag²–n_frag⁴) Python, CPU integrals | none | not measured `[?]` | GPU 1e integrals exist (`pbc.gto.int1e`); batch the fragment pairs | mod | low |
| **C5** | C6 coefficients — **mechanism changed** | `properties/c6.py:23-26` full A/B via `td.get_ab`, then CPU `np.linalg.cholesky`/`eigh`/`eig` on `2·nov` (`:41-110`) `[V]` | inherits D4's CPU ERI, plus an O(nov³) host eigensolve | none | not measured `[?]` | Casimir–Polder from imaginary-frequency CPKS on the GPU response path (which Electrolites already accelerates) | mod | low |
| **C13** | stability analysis is missing | `scf/hf.py:865`, `uhf.py:324`, `rohf.py:100` `NotImplemented` `[V]` | missing | n/a | n/a | a Davidson on `gen_response` — the pieces exist | greenfield | low |

06 **C1** is done; **C3**'s CPU dipole integrals are trivial — its real cost
is C10/C11.

### D. Excited states and their derivatives

| ID | Target | Evidence (1.8.1) | Mechanism | Electrolites | Share | Direction | Diff. | Contam. |
|---|---|---|---|---|---|---|---|---|
| **D9** | **TDDFT gradient / NACV XC kernel: per-matrix loops inside the grid loop** | `grad/tdrks.py:382-428` up to 4 `eval_rho` + 4 `fmat_` (each `_dot_ao_ao` + `_gga_grad_sum_` + `add_sparse`) per block on deriv-2 AOs; `nac/tdrks_grad_nacv.py:166-224` a Python `for i in range(n_batch)` inside the block loop; the function runs 2–3 times per state (`grad/tdrks.py:88,293`; `nac/tdrks.py:290,292,559`), each re-running `eval_ao`, `eval_rho2` and libxc `[V]` | `fastxcfxc`'s per-set GEMM problem plus repeated whole-grid passes | **none** — `fastxcgrad` patches only `get_exc` | **never profiled**; the analogous batching was 3.2–3.8x inside a TDA run (`RESPONSE_1.8.1.md` §4b) | stride-batch the densities (f1vo, f1oo, k1ao, states) into one GEMM per block as `fastxcfxc` does; merge the separate passes | mod | moderate |
| **D10** | **TDDFT gradient / NAC two-electron derivative** | `grad/tdrhf.py:293-420` `_jk_energies_per_atom` (`RYS_per_atom_jk_ip1_sum`, multi-density pairs), from `grad/tdrks.py:271-280` (twice for RSH), `nac/tdrks_grad_nacv.py:646-656`, `grad/tduks_sf.py:500` `[V]` | the ground-state ip1 family at several density pairs, without the P1–P5 treatment | **none** — `fastejk` patches only `grad.rhf._jk_energy_per_atom` (`electrolites/fastejk.py:365-366`) | never profiled; the ground-state analogue is the `ejk` 21–30 % of a gradient (06 §2.2) | extend `fastejk`'s kernels to `n_dm` (dm1, dm2) pairs with j/k factors; fuse full and long range | mod | moderate |
| **D11** | DF TDDFT gradient memory wall | `df/grad/tdrhf.py:423-424` **two** `(naux, n_o, n_o)` arrays per density pair, re-allocated at `:459-461`; TDDFT passes 3 pairs `[V]` | the wall `fastdfgradoo` removed for the ground state, ×3–6 | none | one such array is 37.7 GB at 284 atoms (`README_fastdfgradoo.md`), so ~226 GB asked `[?]` | as `fastdfgradoo`: stream the auxiliary index, hold one array per pair or none | mod | moderate |
| **D12** | `force_and_nacv`: one Z-vector CPHF per state pair | `tdscf/rhf.py:478-489` loops `td_nac.kernel(states=(i,j))`; a batched `get_nacv_multi` exists (`nac/tdrks_grad_nacv.py:247`, solve at `:503-535`) but is singlet-only and labelled experimental (`:255`) `[V]` | `n_pairs` serial CPHF solves at `nset = 1`, where the multi-density kernels gain least (`ROUND4_B4B.md` §4) | per call via `fastjk`/`fastxcfxc` | not measured `[?]` — this is the FSSH inner loop | route through the batched solve; extend it to triplets | easy–mod | low |
| **D13** | spin-flip multicollinear XC | `tdscf/_uhf_resp_sf.py:72-93` libxc on `ngrids × 20` samples, full 2×2(×2) spin `fxc`/`kxc` materialised and one slice used; called **per grid block** in the SF gradient (`grad/tduks_sf.py:393`) `[V]` | 20x libxc work; 4–8x of the output discarded | response half probably through `fastxcfxc` (`_uhf_resp_sf.py:156` calls `ni.nr_rks_fxc`) — **not probed** `[?]`; gradient half none | not measured `[?]` | integrate over samples inside one kernel without materialising; evaluate only the needed spin blocks | mod–hard | low |
| **D4** | `get_ab`'s in-core A/B (with D5) | `tdscf/rhf.py:135,180` CPU `mol.intor('int2e_sph')` + four O(N⁵) transforms; `uhf.py:171,236,478,538`; D5's per-cavity-point pivot sync (`rhf.py:91-96`) lives only here `[V]` | now on a user path (C5); GPU `scf/int4c2e.py:191 get_int4c2e` / `:325 get_int4c2e_ovov` exist unused here | none | not measured `[?]` | use the GPU ovov builder; keep the LU on the device | easy–mod | low |
| **D6** | TDDFT-ris | `tdscf/ris.py:382,453,518` aux/vir/occ chunk loops `[V]` | GEMM-dominated already | none | not measured `[?]` | measure first | mod | low |

06 **D1** is done; **D2**, **D8** are rejected (§3.3); **D3** and most of
**D7** are closed upstream (§3.1).

### E. Real-space and population analysis

| ID | Target | Evidence (1.8.1) | Mechanism | Diff. | Contam. |
|---|---|---|---|---|---|
| **E1** | `analyze()` / Mulliken / meta-Löwdin | `scf/hf.py:923-926` `return self.to_cpu().analyze(...)`; `:859-860` `mulliken_pop = NotImplemented` `[V]` | whole object to the host; O(nao³) CPU meta-Löwdin | easy–mod | low |
| **E2** | `dip_moment` / `quad_moment` | `scf/hf.py:903-913` `hf_cpu.dip_moment(mol, cupy.asnumpy(dm), …)` `[V]` | CPU multipole integrals + D2H of the density | easy | low |
| **E3** | **real-space grid properties: still absent** | `tools/` holds `ase_interface.py`, `method_config.py`, `__init__.py` only `[V]` | no cube, MO-on-grid, ELF/LOL/NCI, Laplacian or real-space ESP driver; the pieces (`get_rho*`, `nr_eval_gto.cu` deriv 0–4) exist | greenfield | **low** |
| **E4** | ESP / RESP fitting-point setup | `pop/esp.py:52-84` Python `unit_surface`; `:102` scipy `distance_matrix` per atom × 4 scales `[V]` | host setup; the integral is `fast1e`'s | easy | low |
| **E5** | CHELPG grid generation | `qmmm/chelpg.py:65-99`: the full bounding-box cube at 0.3 Å with a host `natm × npts` `cdist` before filtering `[V]` | host memory wall: ~8e9 doubles for a 1000-atom box by arithmetic `[?]` | easy | low |

Nothing in E has a measured share; at ≤ 113 atoms these are tails.  E3 is a
benchmark candidate (low contamination, greenfield), not a throughput target.

### F. Solvation, QM/MM, dispersion, ECP

| ID | Target | Evidence (1.8.1) | Mechanism | Electrolites | Share | Direction | Diff. | Contam. |
|---|---|---|---|---|---|---|---|---|
| **F9** | **multi-RHS `int3c1e`: the PCM Hessian's `h1` and the PCM response** | `solvent/hessian/pcm.py:1271-1278`: `for i_atom … for i_xyz: int1e_grids(…, charges=dqdx[i_atom,i_xyz,:])` = **3·natm full-cavity passes**, twice in UHF (`:1360-1361`) with the same density; `gto/int3c1e.py:528-530,541-543` one density / one charge vector per pass, reached from `PCM._B_dot_x` (`solvent/pcm.py:750-770`) in CPHF, TDDFT-PCM and polarizability; each density call does a D2H + CPU Hermite pre-pass (`int3c1e.py:412-428`), kept by `electrolites/fast1e.py:535-546` `[V]` | the same (pair, point) integrals recomputed per right-hand side | **partial**: `fast1e` speeds each pass (1.49x charge, 1.98x density at taxol, `README_fast1e.md`) but does not batch | not measured; 339 calls × 75.8–112.9 ms ≈ **26–38 s** at taxol for `h1` alone, by arithmetic from `README_fast1e.md`'s per-call times `[?]` | one pass contracting `n_rhs` vectors; or materialise plain `int3c1e` per grid chunk and GEMM with `Q[ngrid × 3natm]`; Hermite pre-pass on the device | mod | low–moderate |
| **F10** | **PCM Hessian dense O(ngrids·natm²) tensors** | `solvent/hessian/pcm.py:63-64` `d2F, d2A = zeros([ngrids,natm,natm,3,3])`; `:170` `d2Sii`; `get_d2D_d2S` (`:308`, kernel `lib/solvent/pcm.cu:555-628`) 9·n² each; unconditional for IEF-PCM, SS(V)PE, SMD (`:747,805,921`); C-PCM falls back only when short of memory (`:722-735`) `[V]` | the switching function is local, the storage is dense | none | not measured; by shape, taxol `d2F`/`d2A`/`d2Sii` **10.1 GB each**, valinomycin (168 atoms) `d2F + d2A` ≈ 70 GB — an OOM on 80 GB `[?]` | neighbour lists for dF/d2F; extend upstream's C-PCM matrix-free `pcm_contract_d2s_offdiagonal` (`pcm.cu:630`) to d2D; vectorise the natm² loops (`:506-530,637-665`) | mod–hard | low |
| **F11** | PCM gradient/Hessian: dense dD/dS by default, intermediates rebuilt, host `fakemol` integrals | `solvent/grad/pcm.py:180-206` `[3,n,n]` dD and dS (default for IEF/SMD, `:573-576`); `get_dqsym_dx` rebuilt 3 times in one Hessian (`hessian/pcm.py:433`, before `:550`, `:1271`); `intor_cross(int2c2e*, fakemol_nuc, fakemol)` on the host at `solvent/pcm.py:602-606`, `smd.py:439-441`, `grad/pcm.py:433-448`, `hessian/pcm.py:393-439,1155-1162` `[V]` | 6·n² memory (5.8 GB at taxol); repeated work; closed-form erf(ξr)/r on the CPU | none (06 F3 was rejected for the SCF path only) | 06: PCM 5.3 % of a stock taxol gradient (1.7.0); Hessian not measured | compute `dqdx` once; tiled matrix-free dS/dD with P10's fp32 far field (`erf → 1`); a GPU kernel for the nucleus–cavity term | easy (caching) / mod | low |
| **F12** | SMD | `solvent/smd.py:324,400-401` `sasa_ng = 590`, always the dense IEF path (`:406-424`); CDS through Fortran `mnsol_interface` (`:263-302`); CDS Hessian by **6·natm finite differences** (`solvent/hessian/smd.py:44-53`) `[V]` | ~2x PCM's points, ~4x its n² memory `[?]`; single-core CDS | none | not measured `[?]` | the CuPy CDS in `smd_experiment.py` exists; analytic CDS Hessian; lowmem electrostatics | mod | low |
| **F8** | **ECP: no screening, and duplicate derivative integrals** (06 A12 + B7) | `gto/ecp.py:80,101` unscreened, host `np.meshgrid` (`:75-114`); `get_ecp_ipip` computed twice per Hessian (`hessian/rhf.py:767,778` and `:161-162`); `get_ecp_ip` twice in `get_grad_hcore` (`grad/rhf.py:214`, `:409`), per spin in UHF; dense `[n_ecp_atm,9,nao,nao]` (`ecp.py:302`) `[V]` | O(nbas²·n_ecp) tasks where O(nbas·n_ecp) survive; 2–4x recomputation | none (`fastgradh` also calls `get_ecp_ip`) | not measured (needs heavy elements; the Pt complex of `OPENSHELL_TMC.md` would do) `[?]` | cache (bitwise-safe); distance screen in `make_tasks`; contract with the density in-kernel | easy (cache) / mod | **low** |
| **F13** | QM/MM gradient's MM terms | `qmmm/itrf.py:184` `int1e_grids_ip1` (charges, no `intopt`), `:213` `int1e_grids_ip2`; host numpy `n_mm × natm` at `:102-103,227-230,246-249`; `charge_model == 'point'` asserts `:97,221`; `qmmm/pbc/itrf.py:718-760` materialises Tija/Tijab/Tijabc per QM-atom block `[V]` | with 10⁴–10⁵ point charges nearly all in the far field; the `_ip` kernels are not `fast1e`'s | none (`fast1e` patches the three non-derivative drivers only) | not measured `[?]` | carry `fast1e`'s far-field collapse into `g1e_ip*` / `*_save_u2` (`README_fast1e.md` "What is left"); fused multipole-Ewald kernel; F6 to the device | mod (F6 part easy) | low |
| **F14** | register pressure in the charge-contracted `int3c1e` kernels | `ptxas -v`: `charge_contracted_kernel_expanded<2,2>` and `density_contracted_kernel_general<8>` at 255 registers with spills (`README_fast1e.md` "What is left") `[V]` | `g[3·NROOTS·(LI+1)·(LJ+1)]` live across the contraction; 12.5 % occupancy `[?]` | `fast1e` leaves it | the four `expanded<LI,LJ>` kernels are **43 %** of what is left after `fast1e` | build `g` one Rys root at a time | mod | low |
| **F1** | D3/D4/GCP on the CPU, and the dispersion Hessian by finite differences | `dispersion/dftd3.py`, `dftd4.py`, `gcp.py` re-export `pyscf.dispersion`; `hessian/dispersion.py:48-92` 6·natm CPU gradient calls, each rebuilding the model `[V]` | O(N²–N³) single core; the Hessian 6·natm-fold | none | not measured; irrelevant at 42/113 atoms (06 Q7) `[?]` | first reuse the model across displacements; then analytic two-body D3(BJ) on the GPU | easy / greenfield | **low** |

### G. Infrastructure

| ID | Target | Evidence | Why it matters |
|---|---|---|---|
| **G5** | **Electrolites on more than one GPU** | every module except `fast1e` asserts one device (`electrolites/fastk.py:477`, `fastj.py:478,662`, `fastxc.py:490,504`, `fastdf.py:809`, `fastxchess.py:1034`) `[V]`, while upstream distributes `get_jk`, `nr_rks`, `ejk`, `ip2`, `jk_ip1` with `multi_gpu.run` (`scf/jk.py:517`, `dft/numint.py:571-577`, `grad/rhf.py:161`, `hessian/rhf.py:379,571`) `[V]`; `DF_WORK_LIST.md` W2 | any `num_devices > 1` run is **entirely stock**.  Upstream's `lib/multi_gpu.py:99` `for i in num_devices:` defect (06 G4) is still there, in dead code |
| **G6** | per-launch device queries and memsets in upstream's C drivers | `cudaGetDeviceProperties` per class launch in `rys_contract_jk_ip1.cu:1705,1807,1903,2010`, `rys_contract_jk_ip2.cu:1262,1360`; per call in `grouped_gemm.cu:38-46`, `grouped_dot.cu:42` `[V]`; measured 3.3 ms each (`DIRECT_SURVEY.md` §2.2) | one fix (query once per device, cache) reaches B15, B3, A15 at once.  It is in upstream's `.so`, so it needs either a rebuilt library or Electrolites' own driver |
| **G7** | coverage of the feature paths no module is on | §2.5 | a coverage probe (`benchmarks/coverage_probe.py`) run on each §2.5 feature before any of their entries is taken; the three earlier silent misses are the reason |

### H. Negative controls (unchanged from 06)

H1 (dense `eigh`), H2 (DIIS), H3 (`GDFTcontract_rho`) stand: `eig` + DIIS +
grids are **1.9 %** of a stock PfPMT SCF (`ROUND2_1.8.1.md` §2).  The
`eigh` cliff now sits at `lib/cupy_helper.py:1366-1375` (`MAX_EIGH_DIM =
23150`) `[V]`, and **GHF/GKS doubles the dimension**, so a 12 000-AO GKS hits
it.  **cuEST** (`lib/cuest_wrapper.py`: opt-in, DF only, no RSH `:4451-4452`,
no Hessian `:4617`, K from occupied orbitals only, asserting a PSD density
`:4555-4565`) has no callers inside `gpu4pyscf` `[V]`; it is closed source
(06 tier C2) and is a **timing reference**, not a target.  This repository has
never been benchmarked against it.

### I. Correlated wavefunction methods (outside 06's scope)

| ID | Target | Evidence (1.8.1) | Mechanism | Diff. | Contam. |
|---|---|---|---|---|---|
| **I1** | **DF-MP2 pair-energy loop** | `mp/dfmp2_addons.py:1159-1167`: `for i … for j ≤ i`: `cderi_ovl[i] @ cderi_ovl[j].T`, then `float(e_bi1)`, `float(e_bi2)` `[V]` | nocc²/2 small GEMMs and **two host syncs per pair** — 06 P9 at nocc² scale | easy | moderate–high |
| **I2** | DF-MP2 on one device | `mp/dfmp2.py:74` always `dfmp2_kernel_one_gpu`; `dfmp2_drivers.py:58,79-80` hold `[nocc, nvir, naux_cart]` (Cartesian auxiliary count) on one GPU; a host-staged multi-GPU driver (`:180`) is called only by tests `[V]` | the largest DF-MP2 is one card's memory | mod | moderate |
| **I3** | conventional MP2 ovov | `scf/int4c2e.py:325-366` legacy `GINTfill_int2e`, whole class-pair blocks (`:417`), 8 `einsum` per block; full `ovov` on the device (`:343`); restricted only `[V]` | O(o²v²) device memory | mod | moderate |
| **I4** | CCSD amplitude update on the host | `cc/ccsd_incore.py:75-76` t1/t2 uploaded per iteration; host `lib.einsum`/`np.einsum` at `:87,90,98-100,118-119,131-145`; `.get()` at `:84,105,108,125,128,331`; PySCF DIIS; the constructor calls `mf.to_cpu()` (`:610-611`) `[V]` | O(o²v²–o³v²) terms on the CPU, PCIe both ways every iteration | mod | high |
| **I5** | CCSD AO-direct vvvv | `ccsd_incore.py:288`, `:390` all four-centre ERIs recomputed every iteration with the legacy kernel; `Ht2` `nocc²·nao²·16 B` with a hard error (`:161-165`); host `nao²·nocc²` arrays (`:272-273`) `[V]` | recomputation + device and host memory walls | hard | moderate–high |
| **I6** | FCI host Davidson | `fci/direct_spin1.py:187` `return out.get()`; address tables rebuilt per call (`:158`), `uint16` string cap (`:134`); only `contract_2e` on the GPU `[V]` | CI vectors cross PCIe twice per iteration | mod | moderate |
| **I7** | MP2 gradients; double hybrids | `mp/mp2.py:351-352` `Gradients` raises; no double-hybrid path `[V]` | missing — greenfield; never pooled with speedups | greenfield | moderate |

### J. Periodic boundary conditions (outside 06's scope)

| ID | Target | Evidence (1.8.1) | Mechanism | Diff. | Contam. |
|---|---|---|---|---|---|
| **J1** | GDF k-point cderi in host mapped memory | `pbc/df/rsdf_builder.py:483` `empty_mapped`; streamed per J/K build (`pbc/df/df.py:192-206`, `df_jk.py:228-243`) `[V]` | 06 A13's PCIe problem, per k-point | mod | moderate |
| **J2** | GDF metric `eigh`, serial over k | `rsdf_builder.py:60` `PREFER_ED = True` (default since 1.8.0), `:614-617`, `:175-177` one `cp.linalg.eigh` per unique k `[V]` | nk serial naux³ solves on the default path | easy–mod | low |
| **J3** | FFTDF exchange (KHF default) | `pbc/df/fft_jk.py:170-203` k1×k2 Python loops × nao/32 FFT batches; `:153` AOs for all k on the full grid; no time-reversal symmetry (RSJK has it, `scf/rsjk.py:507-522`) `[V]` | O(nk²) serial launches; nk·ngrids·nao memory | mod–hard | moderate |
| **J4** | multigrid stress by strain finite differences | `pbc/dft/multigrid_v2.py:1876-1895`: 3×3×2 = **18 full `evaluate_density_on_g_mesh` passes** `[V]` | numerical where analytic is standard | mod | low |
| **J5** | GTH non-local PP: k-points on the CPU, and its strain derivative | `pbc/gto/pseudo/pp_int.py:135` CPU `_int_vnl` for k ≠ Γ (and `pbc/grad/pp.py:43-45`); the Γ path ends in `.get()` (`:64`, `:157`); `pbc/grad/rks_stress.py:416-472` 18 strains × nk × natm Python with CPU `eval_gto` `[V]` | host libcint lattice sums; host round trips | easy–mod | low |
| **J6** | GTH local part 2 on the CPU (GDF/AFTDF) | `rsdf_builder.py:909`, `pbc/df/aft.py:103` `_IntPPBuilder(...).get_pp_loc_part2()` `[V]` | one-time CPU 3c1e lattice sum | mod | low |
| **J7** | PBC gradient `hcore` re-evaluates every AO per atom | `pbc/grad/krhf.py:93-95` per-atom loop over `hcore_deriv` (`:196-215`), which reruns `block_loop` over the whole grid for every k; the multigrid-v2 branch (`:78-86`) already does it as one G-space contraction `[V]` | O(natm·nk·ngrids·nao) AO evaluations | easy–mod | low |
| **J8** | KHF `eig` serial over k | `pbc/scf/khf.py:560-568` `[V]` | nk/2 serial generalized `eigh` per cycle | easy | low |

PBC TDDFT, k-point VV10 (`pbc/dft/krks.py:64`) and spin-orbit PBC X2C are
absent (greenfield).

### K. Semi-empirical and relativistic Hamiltonians (outside 06's scope)

| ID | Target | Evidence (1.8.1) | Mechanism | Diff. | Contam. |
|---|---|---|---|---|---|
| **K1** | PM6 all-pairs two-centre ERIs and J/K | `sem/gto/mole.py:205-208` every `i > j` pair, no distance cutoff (the `cutoff = 10` attribute is used only by the overlap/resonance term, `sem/integral/fock.py:220`, `sem/integral/hcore2c1e.py:456`); `lib/sem/fock.cu:118-170` one block per pair with atomics `[V]` | O(natm²) memory and work; NDDO decays as 1/R so a plain cutoff is not exact | mod–hard | **low** |
| **K2** | PM6 gradients are pairwise finite differences | `sem/grad/rhf.py:45` (h = 1e-5), `:71` 6 displaced pair energies per pair, `:171` a memory TODO, `:210` the displacement loop `[V]` | 6x recomputation of every pair integral; the changelog calls them "analytical" `[X]` | mod | low |
| **K3** | molecular X2C W integrals on the CPU | `x2c/x2c.py:170,232` `xmol.intor('int1e_spnucsp')`; `x2c/sfx2c1e.py:176,220` `int1e_pnucp`; atomic-X via PySCF per element (`x2c.py:525-536`).  The PBC sfX2C already has a GPU `pvp` kernel (`pbc/x2c/x2c1e.py:316-318`) `[V]` | one-time libcint on the uncontracted basis | easy–mod | low |


---

## 5. Ranked

Two orderings, as 06 §5 requires, because they barely overlap.  **This
section was re-ranked on 2026-10-03 on the §8 measurements**, which are in
§10; every share below that cites §10 was measured after that date and every
other one is quoted from the document named.

### 5.1 By measured value (throughput)

Ranking = measured share **after Electrolites** × how much of an existing,
measured treatment transfers × how clean the oracle is × how cheap the change
is.  **Only entries with a measured or directly derived share are ranked**;
the rest are in §5.2 and §5.3.  "Share" is of the whole run of the feature
named, after `patch_all()`, unless the cell says otherwise.

| Rank | Entry | Features it reaches | Share it acts on (measured) | Why this rank |
|---|---|---|---|---|
| 1 | **B13** VV10 pair kernel `f_t` (+ **B14**) | every VV10 functional's polarizability, Hessian, frequencies, IR | `nr_rks_fnlc_mo` **83.3 %** of a wB97M-V CPHF; VV10 response 80.6 % of a stock wB97M-V polarizability, the largest term left after `fastfnlc` | the largest single share anywhere in this repository's profiles; three of `fastnlc`'s measured techniques (MUFU, P10, P11 — 3.92x together on the ground-state sum) transfer; the source is now in the tree; the redundant `ntrial/6` re-evaluation is structural, not tuning |
| 2 | **C10** IR's duplicate CPHF | IR intensities (and every workflow that asks for them with frequencies) | the duplicate `make_h1` + `solve_mo1` is **41.8 %** of a 42-atom IR run and 31.6 % of a 22-atom one, growing with size (§10.2); removing it is worth **1.72x / 1.46x** on the whole run | easy, and exact (bitwise: the same equations, solved once); upstream marks it TODO.  Q2's derived 44–48 % was the right size, slightly high |
| 3 | **A14** restricted meta-GGA XC in the SCF | every restricted TPSS, r2SCAN, SCAN, M06-2X, MN15 SCF (and everything after it) | `nr_rks` is **53–62 %** of a patched pure-mGGA SCF and **34–41 %** of a patched M06-2X one, 42 and 113 atoms, and `fastxc` hands **every** call back to stock (§10.3); per call it is 2.6x the patched GGA build of the same molecule | the code exists for UKS; bounded by Amdahl at **1.3–1.45x** of a pure-mGGA SCF if the GGA treatment's 2.06x transfers.  Broad reach, easy, an element-wise oracle |
| 4 | **C7** NMR: GIAO AOs evaluated on the CPU per grid block | every NMR shielding | **72–78 %** of a patched NMR run at 22 and 42 atoms (126 s of 171 s at 42 atoms), B3LYP and PBE alike (§10.4) | easy–mod: ig/ipig are `eval_ao` deriv 1–2 times `(R_μ × r)`, no new integrals.  NMR is otherwise untouched by Electrolites (1.02–1.16x).  Was ranked below C6 by guess; measured, it is 3–6x C6 at these sizes |
| 5 | **A17** native short-range K for HSE06 | every HSE06 (and any `erfc`-only RSH) SCF and its derivatives | patched `get_k` is **66 % / 72 %** of a patched HSE06 SCF at 113 / 284 atoms, and the decomposition gives it **1.01x / 1.05x** over stock where B3LYP's K gets 2.54x (§10.5) | not a regression (Q5 refuted) but the locality of `erfc` is thrown away from ~100 atoms on.  If a native kernel reached B3LYP's 2.5x, the 284-atom SCF gains ~1.7x.  mod |
| 6 | **I1** DF-MP2 pair-energy loop | every DF-MP2 | **73 % / 81 %** of the DF-MP2 step at 42 / 113 atoms, 46–49 % of SCF + MP2 at 113 atoms; grows with size (§10.10) | easy (batch the pairs into GEMMs, drop the two host syncs per pair), exact to round-off |
| 7 | **B12** the per-density exchange contraction `C` | every direct Hessian, frequencies, IR — **and TDDFT** | `get_k` **37–55 %** of a patched Hessian, 94 % of it `C`; the multi-density response `get_jk` is **71 %** of a patched 113-atom TDDFT + gradient + NAC run, where `fastjk` falls from 3.2x at 42 atoms to **1.33x** (§10.1) | the largest share in both the Hessian and the TDDFT paths and it grows with size; but no treatment transfers (inexact-Krylov is a hypothesis), so it ranks below the certainties |
| 8 | **B16** short-range half of RSH derivatives, UKS | UKS range-separated gradients (and both Hessians' ip2) | the 2e derivative is **88 % / 96 %** of a patched UKS ωB97X gradient (penicillin V⁺ / Ni dithiolene⁻), 18–28 % of SCF + gradient; exactly half its class blocks are stock — the ω < 0 pass (§10.8) | easy: `fastgrad`'s RKS identity, measured there at 1.99x on the same term, ported to `grad.uks` |
| 9 | **D9 + D10** TDDFT gradient / NAC XC kernel and 2e derivative | TDDFT and TDA gradients, NACs, the FSSH inner loop | together **28–33 %** of a patched pure-functional TDDFT gradient + NAC run (42 and 113 atoms) and 13–25 % of a hybrid one; untouched by every module (patched = stock time) (§10.1) | the two structures `fastxcfxc` (3–4x) and `fastejk` (1.7x) already beat on neighbouring paths; bounded at ~1.14x of the run for B3LYP, more for pure functionals.  mod |
| 10 | **J3** FFTDF k-point exchange | every k-point hybrid on the default `KRKS` back end | `fft_jk.get_k_kpts` is **94–97 %** of a k-point PBE0 SCF; ~nk^2.2 from nk = 8 to 27 (§10.11) | the largest share on the periodic path and on its default back end; no measured treatment transfers (time-reversal symmetry, batched FFTs).  mod–hard |
| 11 | **B15** `jk_ip1` (step a first) | every direct Hessian | **26.8 %** of a patched UKS TMC Hessian; 7.7 % of a patched 42-atom RKS one on an otherwise idle node (§10.7) | step (a) is a host-overhead fix measured elsewhere at 3.3 ms per query — and this stage doubled when the node's CPUs were shared (§10.7), the signature of a host-bound stage; (c) is `fastejk`'s P1–P5, measured 1.70x on the sibling kernel |
| 12 | **B3** `ejk_ip2`'s general kernel | every direct Hessian, esp. transition metals | 63 % of `ejk_ip2`; `ejk_ip2` **29.1 %** of a patched UKS TMC Hessian, 13.7 % of a 42-atom RKS one | P6's "write, don't lift" measured 2.48x on K's general classes; a generator project |
| 13 | **F9 / F11** PCM response and `h1` | PCM Hessians, TDDFT-PCM, PCM polarizabilities | PCM-specific terms **28–30 %** of a patched 42-atom PCM Hessian: `_B_dot_x` in the CPHF **12.6–14.8 %**, the solvent second derivative 10–11 %, the 3·natm `int1e_grids` loop only **1.2–1.5 %**, `get_dqsym_dx` ×3 1–2 % (§10.6) | the multi-RHS batching F9 proposed belongs on `_B_dot_x`, not on the `h1` loop the catalogue named; F11's caching is easy and small |
| 14 | **A5** the AO kernel (+ **A15**) | every DFT SCF, gradient, response | XC 10.2 % of a patched SCF, 34–38 % of it device-idle; `eval_ao` 26 % of the patched XC gradient | the A15 half is easy and bounded at 4.5–7 % of an SCF; the kernel half is mod |
| 15 | **A20** cheaper direct K | every hybrid SCF | **66.8 %** of a patched SCF | the largest share in the SCF, but nothing measured transfers and it changes the answer; ranked by confidence, not size |
| 16 | **B17** DF CPHF RHS count/precision (W46) | DF Hessians | 82 % of a 113-atom UKS DF Hessian | already specified in `DF_WORK_LIST.md` with a falsification criterion |
| 17 | **I5 / I4** CCSD AO-direct vvvv and host amplitude update | CCSD | AO-direct `ovvv`/`vvvv` **48–53 %** of a 22-atom CCSD; host-side einsum/reshape/DIIS ≈ 22 % and `cudaMalloc` ≈ 16 % (2931 real allocations) by cProfile (§10.10) | the allocator half is this repository's recurring cheap fix (`DF_WORK_LIST.md` §5.8); the rest is hard |
| 18 | **C12** fused RSH response J+K | RSH polarizability, TDDFT | J+K 58.8 % of a wB97M-V TDA; fusion 1.12–1.16x of K in SCF | as `fastrsh`, on `fastjk`'s kernels |
| 19 | **J1** GDF k-point exchange | k-point hybrids with `density_fit()` | `df_jk.get_k_kpts` **50–54 %**; the one-time cderi 39–43 % (§10.11) | J2 (metric `eigh` per k) is 2–8 % and is a tail |
| 20 | **A18** under-filled K classes | SCF, esp. TMCs | 1.056x on PfPMT `get_k` (login node) | small but measured; owes a compute-node re-run |
| 21 | **A7** RSH fusion, high-`l` classes | RSH SCF | ~9 % of K (06's estimate) | easy |

**Not a target any more** (closed by §10): R1 `fastgradh` (it is 2–3x
*faster* than stock on its row, and the row is 0.1–0.3 % of a Hessian,
§10.7); R2 the DF `get_hcore` row (a first-call initialisation, 0.46–0.52 s
cold, 0.10 s warm, §10.9); R3 the OMP-dependent stock gradient (not
reproduced on 1.8.1: PfPMT 19.79 s at 1 and at 16 threads, §10.9); F10 as a
*time* entry (0.3 % of a PCM Hessian) — it stays as a **memory** wall, below.

**Runnability walls measured by §10** (not throughput, but they decide
which systems a user can run at all):

* **F10**: at taxol (113 atoms, 11 274 cavity points) `get_d2F_d2A` holds
  **22.0 GB** and `get_d2D_d2S` **18.3 GB** of device memory, and an IEF-PCM
  Hessian needs both — beyond a 40 GB card; at 284 atoms the PCM build itself
  is out of memory on 40 GB (§10.6).
* **TDDFT**: `_lr_eig` sizes a dense `(2·max_space)²` device matrix from the
  *host* budget `max_memory`; at 80 GB it is 31 GB for penicillin V and OOMs a
  40 GB card in stock and patched alike (§10.1).  And the default
  `cphf_max_cycle = 50` does not converge taxol/PBE's Z-vector (200 does).

### 5.2 Measure first — probably large, never profiled

§8's eleven measurements moved ten entries into §5.1.  What is still
unprofiled, in the order a single profile would most likely change §5.1:

1. **D12** `force_and_nacv` — one Z-vector CPHF per state pair; §10.1 puts
   the Z-vector at 23–40 % of a single pair, so `n_pairs` of them is the FSSH
   inner loop.  Easy–mod.
2. **B14 against B13** — the split of `_get_vnlc_deriv1`'s 31 s between the
   host atom loop and the pair kernel.
3. **D13** spin-flip multicollinear XC, and whether `fastxcfxc` is on the
   spin-flip response path at all.
4. **F12** SMD (the CDS Hessian by 6·natm finite differences) and **F8** ECP
   (needs a heavy-element system; the Pt complex of `OPENSHELL_TMC.md`).
5. **B9** grid-response Hessian, including the `:2010` int32 assert (Q4).
6. **J4, J5, J7** — the periodic gradient and stress paths.
7. **A19, A21, K1/K2, C4, C5, D6** — g-shell classes, GKS, PM6, EDA, C6
   coefficients, TDDFT-ris.

### 5.3 By benchmark cleanliness (low contamination)

The catalogue's four (A12, F1, C2, E3) are all still open; this survey adds
to them.  Low contamination means essentially no public GPU source to recall:

| Entry | Why it is clean | Oracle |
|---|---|---|
| **C7** GIAO AOs on the GPU (new, §10.4) | no public GPU GIAO AO evaluation `[?]`; and it is now also the largest NMR share | PySCF `eval_gto('GTOval_ig')` element-wise; shielding tensors |
| **C6** GIAO `(ig1)` two-electron integrals | no public GPU implementation `[?]`; 20–22 % of a 42-atom NMR run and growing as ~nao^2.7 (§10.4) | PySCF `prop.nmr` CPU, element-wise |
| **F8** ECP screening (06 A12) | carries its own TODO; no public GPU ECP screen | exact: unscreened result to 1e-14 |
| **F9/F10** PCM response batching and sparse d2F | PCM Hessians on GPU are GPU4PySCF-only and recent | upstream's dense path; finite differences |
| **B13** VV10 response kernel | public only since 1.8.0 (06 tier C1, "the best discriminating tier") | the stock `.so` element-wise |
| **K1/K2** PM6 pair integrals and analytic derivatives | PM6 on GPU is new in 1.7.1/1.8.0 | MOPAC; the current FD gradient |
| **J4** analytic multigrid stress | new in 1.8.0; FD upstream | the FD stress |
| **E3** real-space properties | greenfield, no incumbent — never pooled into speedup statistics | PySCF `tools.cubegen` |
| **F1** D3/D4 on GPU | no public GPU port located `[?]` | simple-dftd3 / dftd4 to 1e-12 |

**The one-sentence version** (06's, still true, now with a second
exception): the most valuable throughput targets (§5.1) and the cleanest
benchmark targets (§5.3) barely overlap — except **B13**, which is both the
largest measured share left and a C1-tier kernel, and **C7**, which is both
three quarters of an NMR run and essentially uncontaminated.

---

## 6. Correctness oracles

06 §6's table applies unchanged (element-wise against libcint/PySCF in fp64;
`Σ w ρ = N`; finite differences for Hessians; integrated quantities only for
anything approximate; fixed-density matrix-by-matrix on ill-conditioned
systems).  Additions for the classes 06 did not cover:

| Target class | Oracle | Why it holds |
|---|---|---|
| anything that removes a duplicate computation (C10, F8 caching, B15 n_dm=2, F11 `dqdx` once) | **bitwise** against stock | the arithmetic is the same; any difference is a bug |
| TDDFT gradients / NACs (D9–D12) | stock `_contract_xc_kernel` / `_jk_energies_per_atom` outputs at fixed X/Y, element-wise; then finite differences of excitation energies | fixed X/Y removes the Davidson's own tolerance from the comparison |
| VV10 response (B13) | the stock `f_t` output element-wise in fp64; with P10/P11 on, the integrated Hessian/polarizability with the error bound stated beside the measured error | 06 §3.6 P11's practice |
| NMR (C6–C9) | PySCF CPU `prop.nmr` shielding tensors; integrals element-wise against libcint | GIAO integrals have a CPU reference for every component |
| PCM Hessian (F9–F11) | the dense upstream path at small natm; finite differences of the PCM gradient; `solvent/tests/test_pcm_hessian.py` | the dense path is exact where it fits |
| cheaper arithmetic (A20, B17) | the converged energy / Hessian within `conv_tol` / `conv_tol_cpscf` of fp64, **plus** per-cycle matrix error against fp64 | as `README_fastdffp32.md`: an approximation is opt-in and reported with its error |
| post-HF (I) | PySCF CPU `e_corr` to 1e-10; t-amplitude norms | |
| PBC (J) | PySCF CPU matrices per k, element-wise; FD stress | |
| PM6 (K) | MOPAC heats of formation; the current FD gradient | |

---

## 7. Quarantine: unverified, to re-check

| # | Claim | Status |
|---|---|---|
| Q1 | Every "not measured" entry in §4 | `[?]` by definition.  §5.2's order is a judgment, not a measurement |
| Q2 | C10's 44–48 % | **closed, §10.2**: measured 41.8 % at 42 atoms (31.6 % at 22) — the right size, slightly high |
| Q3 | F9's 26–38 s and F10's 10 GB / 70 GB | **closed, §10.6**: F10's tensors measured at 22.0 GB (d2F + d2A) and 18.3 GB (d2D + d2S) at taxol, i.e. the shape arithmetic holds; F9's `h1` loop is 1.1 s at 42 atoms, so 26–38 s at taxol was an over-estimate (~6–8 s by scaling `[?]`) |
| Q4 | B9's "fails above ~240 atoms" | arithmetic on the `:2010` assert, never run |
| Q5 | A17: HSE06 patched may be slower than stock | **refuted, §10.5**: never slower; but K gains only 1.01–1.05x from 113 atoms on |
| Q6 | R1: `fastgradh` is a regression on 1.8.1 | **refuted, §10.7**: its row is 2–3x faster than stock, and 0.1–0.3 % of a Hessian |
| Q7 | R2/R3: the `get_hcore` "CPU" row and the OMP-dependent stock gradient | **closed, §10.9**: R2 is first-call initialisation (0.46–0.52 s cold, 0.10 s warm); R3 does not reproduce on 1.8.1 (PfPMT 19.79 s at 1 and 16 threads) |
| Q8 | B13's `ceil(ntrial/6)` re-evaluation | read from the launch geometry (`vv10.cu:760-764`, `blockIdx.y` over trial groups); whether the compiler or the cache hides any of it is not measured |
| Q9 | contamination tiers | `[?]` exactly as 06 §0 says: no recall probe was run |
| Q10 | 06's §2 shares quoted here (18–21 % AO evaluation; PCM 5.3 % of a gradient) | stock **1.7.0** numbers; 1.8.1's `eval_ao` is several times faster (`RESPONSE_1.8.1.md` §3.4) |

---

## 8. Measurements owed, in the order they would change §5

**All eleven were taken on 2026-10-03; the results are §10.1–§10.11, in the
same order, and §5 is ranked on them.**  The list is kept as it was written.

All on a compute node (the login node's shared card overstated host-bound
stretches 2x, `DIRECT_SURVEY.md`), stock against `patch_all()` back to back
in one job, both of a differently-sized pair of systems (06 §3.4), with
`benchmarks/coverage_probe.py` first so that a share is never claimed for a
module that is not on the path.

1. **A TDDFT-gradient and NAC anatomy** (penicillin V and taxol, B3LYP and
   PBE, 5 roots): the shares of D9, D10 and the Z-vector CPHF.
2. **An IR anatomy**: confirm or correct Q2.
3. **The restricted meta-GGA SCF** (TPSS, M06-2X, r2SCAN): 06 Q13, i.e. A14.
4. **An NMR anatomy**: C6 against C7–C9.
5. **HSE06 patched against stock** at 42, 113 and 284 atoms: A17 / Q5.
6. **A PCM Hessian anatomy** (IEF-PCM and C-PCM): F9–F11, Q3.
7. **`fastgradh` on and off** inside a 42-atom Hessian: R1 / Q6.
8. **A UKS range-separated gradient**: B16's share.
9. **The stock gradient's OMP-dependent stage and the DF `get_hcore` row**
   under `nsys`: R2, R3, Q7.
10. **DF-MP2 and CCSD** on one mid-sized molecule: I1, I4, I5.
11. **A k-point hybrid** (GDF and FFTDF): J1, J3.

---

## 9. Reproducing the survey

There is nothing to run: this document is a source reading of
`/pscratch/sd/w/weiliang/g4psrc/g4p-1.8.1` at `5b284c2` against this
repository at `dd85215`.  To re-check an entry, open its `file:line` in that
tree (the line numbers are 1.8.1's and will move in 1.8.2), and read the
Electrolites gating line it names.  Every quoted number has its `docs/` file
and section beside it; the commands that produced those numbers are in that
file.

§10's measurements are reproduced with (from `benchmarks/`, one GPU, the
`barw-dev` environment; `--patch none` and `--patch all` back to back):

    python path_anatomy.py --feature tdgrad --xyz molecules/penicillinV.xyz --basis def2-svp --xc b3lyp --patch all
    python path_anatomy.py --feature ir     --xyz molecules/penicillinV.xyz --basis def2-svp --xc b3lyp --patch all
    python path_anatomy.py --feature scf --no-grad --xyz molecules/taxol.xyz --basis def2-svp --xc tpss --patch all
    python path_anatomy.py --feature nmr    --xyz molecules/penicillinV.xyz --basis def2-svp --xc b3lyp --patch all
    python path_anatomy.py --feature scf --no-grad --xyz molecules/PfPMT.xyz --charge -1 --basis '6-31g*' --xc hse06 --patch all
    python path_anatomy.py --feature pcmhess --solvent IEF-PCM --xyz molecules/penicillinV.xyz --basis def2-svp --xc b3lyp --patch all
    python pcm_d2_probe.py --xyz molecules/taxol.xyz
    python path_anatomy.py --feature hess   --xyz molecules/penicillinV.xyz --basis def2-svp --xc b3lyp --patch <all, or all but fastgradh>
    python path_anatomy.py --feature grad   --xyz molecules/tmc_ni_dithiolene.xyz --charge -1 --spin 1 --basis def2-svp --xc wb97x --patch all
    OMP_NUM_THREADS=1 python path_anatomy.py --feature grad --xyz molecules/PfPMT.xyz --charge -1 --basis '6-31g*' --xc b3lyp --patch none --cprofile grad.prof
    python path_anatomy.py --feature scf --no-grad --first-call --density-fit --xyz molecules/penicillinV.xyz --basis def2-svp --xc b3lyp --patch none
    python path_anatomy.py --feature dfmp2  --xyz molecules/taxol.xyz --basis def2-svp --patch all
    python path_anatomy.py --feature ccsd   --xyz molecules/alanine_dipeptide.xyz --basis def2-svp --patch all --cprofile ccsd.prof
    python kpt_anatomy.py --cell diamond8 --kpts 2,2,2 --df fftdf

with the other molecules, functionals and flags as §10's tables name them.
The raw logs are under `/pscratch/sd/w/weiliang/kt181/logs/` (purge-eligible).

---

## 10. The §8 measurements (2026-10-03)

Everything in this section was measured for §8 and nothing else in the
document was re-measured.  **Hardware and method**: one Perlmutter GPU node
(`nid001077`), A100-SXM4-**40GB**, GPU4PySCF 1.8.1 in `barw-dev` against
`cudatoolkit/12.9`, `OMP_NUM_THREADS=16` unless a row says otherwise,
`direct_scf_tol = 1e-13`, `conv_tol = 1e-9`, level-3 grids, def2-SVP unless
named.  Stock (`--patch none`) and `patch_all()` are separate processes run
back to back on the same card.  The four cards ran four different
measurements at once; §10.7 found that this inflates **host-bound** stages
(one stage doubled), so the two results that rest on a host-bound stage were
re-run on an otherwise idle node and agree (§10.4, §10.7).  Each number is one
run, not a minimum of several.

The harness is `benchmarks/path_anatomy.py` (one script for every molecular
feature: each stage wrapped with a device synchronisation on both sides, and
a **coverage** count of every Electrolites entry point reached, of every call
a module handed back to its saved stock original, and of the class blocks
taken by our kernels against GPU4PySCF's), `benchmarks/kpt_anatomy.py` for
the periodic path and `benchmarks/pcm_d2_probe.py` for the PCM memory wall.
Rows prefixed `↳` are sub-intervals of their parent; shares are of the whole
run (SCF included) unless a row says otherwise.

### 10.1 TDDFT gradient + NAC (§8 item 1; D9, D10, Z-vector)

TDDFT (RPA), 5 roots, gradient of state 1 and the NAC between states 1 and 2.

| system / functional | stock → patched | TDDFT eigen | Z-vector CPHF ×2 | **D9** `_contract_xc_kernel` ×5 | **D10** `jk_energies_per_atom` (grad + NAC) | response J/K |
|---|---|---|---|---|---|---|
| penicillin V / B3LYP | 89.0 → 36.4 s | 30.0 % | 25.1 % | 3.17 s = **8.7 %** | 6.08 s = **16.7 %** | `get_jk` 45.7 % |
| penicillin V / PBE | 42.8 → 27.9 s | 20.9 % | 28.1 % | 3.34 s = **12.0 %** | 5.78 s = **20.7 %** | `get_j` 34.7 % |
| taxol / B3LYP | 335.6 → 254.3 s | 52.8 % | 22.8 % | 10.42 s = 4.1 % | 22.40 s = 8.8 % | `get_jk` **71.2 %** |
| taxol / PBE ‡ | 147.7 → 103.2 s | 19.8 % | 39.9 % | 9.47 s = **9.2 %** | 18.87 s = **18.3 %** | `get_j` 40.6 % |

(patched shares.)  **D9 and D10 take the same time patched and stock**
(3.50 / 3.17 s and 3.06 / 3.06 s at penicillin V) — no module is on either,
as the source reading said.  `fastjk` and `fastxcfxc` are on the Davidson and
the Z-vector; `fastjk`'s response build goes 52.95 → 16.60 s (3.2x) at 42
atoms but only 241.2 → 181.2 s (**1.33x**) at 113, where it is 71 % of the
run.  With PBE, `fastj`'s multi-density J hands 315 of 588 (42 atoms) and
252 of 462 (113 atoms) class blocks to stock `_run_ref`.

‡ taxol/PBE's Z-vector does **not converge in 50 iterations** (GPU4PySCF's
default `cphf_max_cycle`), stock or patched; the row is with 200.  And
`_lr_eig` sizes a dense `(2·max_space)²` **device** matrix from `max_memory`,
a host budget: at the 80 GB `_common.build_mol` sets, penicillin V asks for
31 GB and OOMs a 40 GB card in both builds.  The harness sets
`td.max_memory = 4000` (PySCF's default).

### 10.2 IR (§8 item 2; C10, Q2)

B3LYP.  "Duplicate" = `eval_ir_freq_intensity` minus the `hessian.kernel()`
inside it: the second `make_h1` + `solve_mo1` + the dipole derivative
(`make_h1` and `solve_mo1` each count **2** calls).

| system | stock: run / duplicate | patched: run / duplicate | duplicate share, patched | gain from removing it |
|---|---|---|---|---|
| alanine dipeptide (22) | 47.6 s / 17.7 s | 18.65 s / 5.89 s | 31.6 % | 1.46x |
| penicillin V (42) | 167.7 s / 68.4 s | 87.63 s / 36.63 s | **41.8 %** (44.4 % of the IR driver) | **1.72x** |

Q2's derived 44–48 % is confirmed in size and slightly high.

### 10.3 Restricted meta-GGA SCF (§8 item 3; A14, 06 Q13)

| system / functional | SCF stock → patched | `nr_rks` patched | its share, patched | stock fall-backs |
|---|---|---|---|---|
| penicillin V / TPSS | 6.63 → 5.53 s | 3.10 s | **56.1 %** | 16 of 16 |
| penicillin V / r2SCAN | 6.85 → 6.33 s | 3.34 s | **52.8 %** | 16 of 16 |
| penicillin V / M06-2X | 14.40 → 7.57 s | 3.09 s | 40.9 % | 14 of 14 |
| taxol / TPSS | 18.32 → 14.20 s | 8.80 s | **62.0 %** | 18 of 18 |
| taxol / r2SCAN | 17.12 → 13.43 s | 8.16 s | **60.8 %** | 17 of 17 |
| taxol / M06-2X | 30.37 → 19.74 s | 6.67 s | 33.8 % | 14 of 14 |

`fastxc`'s `nr_rks` is reached on every cycle and hands **every** call to
`_nr_rks_orig`: the whole speedup is J and K.  For scale, the patched taxol
PBE `nr_rks` is 0.19 s per cycle against 0.49 s (TPSS) and 0.48 s (r2SCAN).

### 10.4 NMR (§8 item 4; C6 against C7–C9)

| system / functional | run stock → patched | **C7** `shielding.nr_rks` (GIAO AOs on the CPU) | **C6** `int2e_ig1` (libcint) | C8 `jk.get_jk` | CPHF |
|---|---|---|---|---|---|
| alanine dipeptide / B3LYP | 46.4 → 39.9 s | 30.2 s = **75.8 %** | 5.2 s = 13.0 % | 2.5 % | 1.7 % |
| alanine dipeptide / PBE | 37.8 → 37.8 s | 29.5 s = **78.1 %** | 4.6 s = 12.2 % | 2.6 % | 0.4 % |
| penicillin V / B3LYP | 190.6 → 177.6 s | 128.6 s = **72.4 %** | 38.3 s = 21.6 % | 2.0 % | 1.6 % |
| penicillin V / PBE | 173.4 → 169.2 s | 126.3 s = **74.6 %** | 36.7 s = 21.7 % | 0.9 % | 0.3 % |
| penicillin V / B3LYP, idle node | — → 170.7 s | 126.3 s = 74.0 % | 34.1 s = 20.0 % | 2.1 % | 1.7 % |

From 200 to 430 AOs C7 grows ~nao^1.9 and C6 ~nao^2.7, so C6 would overtake
C7 near 2000 AOs by extrapolation `[?]`.  C9's per-atom integrals are inside
the < 1 % the wrapped rows leave over.

### 10.5 HSE06 patched against stock (§8 item 5; A17, Q5)

| system | `get_k` stock → patched | SCF stock → patched | patched `get_k` share |
|---|---|---|---|
| penicillin V (42), def2-SVP | 11.04 → 3.29 s (3.36x) | 16.29 → 6.59 s | 49.9 % |
| taxol (113), def2-SVP | 15.22 → 15.10 s (**1.01x**) | 27.69 → 22.85 s | **66.1 %** |
| PfPMT (284), 6-31G\*, charge −1 | 60.93 → 58.19 s (**1.05x**) | 100.43 → 80.79 s | **72.0 %** |
| PfPMT, B3LYP, same card, for scale | 81.44 → 32.12 s (2.54x) | 120.49 → 54.97 s | 58.4 % |

Patched is never slower (Q5 refuted; energies agree to 1e-8 Eh).  Stock's
short-range screen makes stock HSE06's K 25 % cheaper than stock B3LYP's at
284 atoms; the decomposition makes patched HSE06's K **1.81x** dearer than
patched B3LYP's.

### 10.6 PCM Hessian (§8 item 6; F9–F11, Q3)

B3LYP, Lebedev order 29, ε = 78.36.

| penicillin V (42, 4834 cavity points), patched | IEF-PCM | C-PCM |
|---|---|---|
| whole run, stock → patched | 166.2 → 90.7 s | 157.5 → 73.9 s |
| `PCM._B_dot_x` inside the CPHF (F9, response half) | 11.43 s = **12.6 %** | 10.94 s = **14.8 %** |
| solvent second derivative `PCM.hess` | 10.16 s = 11.2 % | 7.24 s = 9.8 % |
| ↳ `analytical_hess_qv` | 5.97 s | 5.44 s |
| `analytical_grad_vmat` in `make_h1` | 4.14 s = 4.6 % | 3.95 s = 5.4 % |
| ↳ 3·natm `int1e_grids` (F9, `h1` half; 126 calls) | 1.13 s = **1.2 %** | 1.14 s = 1.5 % |
| ↳ `get_dqsym_dx`, called 3 times (F11) | 1.83 s = 2.0 % | 0.87 s = 1.2 % |
| `get_d2F_d2A` + `get_d2D_d2S` (F10) | 0.24 s | 0.26 s |
| PCM-specific total | **28 %** | **30 %** |

Alanine dipeptide (2729 points): 25 % (IEF-PCM).  `fast1e` is on the path
(622 calls).  **Q3, time**: the 3·natm `h1` loop is 1.1 s at 42 atoms, so the
arithmetic's 26–38 s at taxol is an over-estimate; scaling by calls × points
gives ~6–8 s `[?]`.  **Q3, memory** (`pcm_d2_probe.py`, read from the
device's free memory because GPU4PySCF's own allocator bypasses CuPy's pool
counters):

| system | cavity points | `get_d2F_d2A` holds | `get_d2D_d2S` holds |
|---|---|---|---|
| alanine dipeptide | 2 729 | 0.33 GB | 1.07 GB |
| penicillin V | 4 834 | 1.63 GB | 3.37 GB |
| taxol | 11 274 | **21.97 GB** | **18.30 GB** |
| PfPMT (284) | — | the PCM build itself is out of memory on 40 GB | |

A whole taxol IEF-PCM Hessian on the 40 GB card ran out of memory earlier,
in the vacuum `make_h1`'s `_get_jk_ip1`, so the d2 wall could not be reached
end to end on this card.

### 10.7 `fastgradh` on and off in a 42-atom Hessian (§8 item 7; R1, Q6)

Penicillin V B3LYP Hessian, three processes: stock, `patch_all()`, and
`patch_all()` without `fastgradh`.  `grad.get_hcore` (via
`get_grad_hcore` in `make_h1`): **0.14 s stock, 0.06 s with `fastgradh`,
0.18 s without**; Hessian 52.68 s with, 53.27 s without; traces agree to
1e-10.  `fastgradh` is not a regression, and its row is 0.1–0.3 % of the
Hessian either way.  On an idle node: Hessian 83.34 → 45.80 s (1.82x), the row
0.14 → 0.05 s.

Measurement hazard: in the concurrent run `jk_ip1` — patched by nothing —
read 3.75 s (stock), **8.12 s** (all) and 6.49 s (all but `fastgradh`); on
the idle node it is 3.96 / 3.93 s.  It is host-bound enough (B15's
per-launch device queries) to double under CPU contention.

### 10.8 UKS range-separated gradient (§8 item 8; B16)

ωB97X; penicillin V⁺ (doublet, 42 atoms) and the Ni dithiolene anion
(doublet, 21 atoms), with neutral penicillin V RKS for comparison.

| system | 2e derivative stock → patched | its share of the patched gradient | of SCF + gradient | fastejk class blocks ours / stock |
|---|---|---|---|---|
| penicillin V⁺ UKS | 3.75 → 3.32 s (1.13x) | **88 %** | 18.4 % | 462 / 462 |
| Ni dithiolene⁻ UKS | 18.29 → 12.97 s (1.41x) | **96 %** | 27.7 % | 1649 / 1649 |
| penicillin V RKS | 3.74 → 1.88 s (1.99x) | 84 % | 21.5 % | 924 / 0 |

The UKS gradient calls `_jk_energy_per_atom` twice and exactly the ω < 0
(short-range) pass goes to stock — the split the source reading predicted.

### 10.9 The stock gradient's thread dependence and the DF `get_hcore` row (§8 item 9; R2, R3, Q7)

Profiled with cProfile (host self time per function) rather than `nsys`,
because the question was what runs on the host.

* **R3**: stock B3LYP/6-31G\* gradient at `OMP_NUM_THREADS` 16 and 1 —
  penicillin V **1.74 / 1.70 s**, PfPMT **19.79 / 19.79 s** (`_jk_energy_per_atom`
  86 % of it).  The top host self-time entries are GPU launches and
  synchronisations (`scf/jk.py:730 proc`, `grad/rhf.py:96 proc`, `.get()`);
  there is no CPU integral on the path.  `ROUND2_1.8.1.md` §6's
  thread-dependent stock gradient does **not** reproduce on 1.8.1.
* **R2**: DF-B3LYP penicillin V, stock: the **first** `get_hcore` in a fresh
  process takes 0.46–0.52 s and the second **0.10 s**, at either thread
  count; the difference is module import and CUDA module loading.  The
  `DF_WORK_LIST.md` §3.1 row is a first-call cost (13 % of a 3.8 s DF SCF),
  not a CPU integral.

### 10.10 DF-MP2 and CCSD (§8 item 10; I1, I4, I5)

| DF-MP2 (RHF reference, default RI basis) | MP2 step | pair-energy loop (I1) | 3c2e + transform | I1 share of SCF + MP2 |
|---|---|---|---|---|
| penicillin V (42) | 1.29 s | 0.94 s = **73 %** of the step | 0.11 s | 19–29 % |
| taxol (113) | 17.23 s | 13.99 s = **81 %** of the step | 2.28 s | 46–49 % |

| CCSD (`ccsd_incore`, RHF/def2-SVP) | CCSD | `update_amps` | ↳ AO-direct `ovvv`/`vvvv` (I5) | `run_diis` |
|---|---|---|---|---|
| glycine (10, 95 AO), patched | 7.70 s | 6.48 s | 4.63 s = 43.5 % | 2.7 % |
| alanine dipeptide (22, 200 AO), stock | 163.2 s | 139.9 s | 85.8 s = **47.8 %** | 7.1 % |
| alanine dipeptide, patched | 175.8 s | 151.8 s | 94.2 s = **52.5 %** | 6.9 % |

cProfile of the 22-atom run: host numpy/PySCF einsum, reshapes and the
h5py-backed DIIS ≈ 39 s (**≈ 22 %**, I4), `cudaMalloc` 28.5 s over 2931
real allocations (**≈ 16 %**), `.get()` 25 s.  Electrolites touches only the
SCF in both (CCSD 1.00x).

### 10.11 k-point hybrid (§8 item 11; J1, J2, J3)

Diamond, PBE0/gth-dzvp/gth-pade; the 2-atom primitive cell (26 AO) and the
8-atom cubic cell (104 AO).

| cell, k mesh, back end | SCF | exchange | one-time cderi | metric `eigh` per k (J2) | XC |
|---|---|---|---|---|---|
| 2-atom, 2×2×2, FFTDF | 8.28 s | **J3** 5.36 s = 64.7 % | — | — | 10.1 % |
| 2-atom, 3×3×3, FFTDF | 71.64 s | **J3** 67.35 s = **94.0 %** | — | — | 2.6 % |
| 8-atom, 2×2×2, FFTDF | 276.69 s | **J3** 267.84 s = **96.8 %** | — | — | 1.8 % |
| 2-atom, 3×3×3, GDF | 7.18 s | **J1** 3.62 s = 50.4 % | 38.7 % | 8.2 % | 22.8 % |
| 8-atom, 2×2×2, GDF | 15.59 s | **J1** 8.21 s = 52.7 % | 41.8 % | 1.7 % | 30.4 % |

FFTDF exchange grows ~nk^2.2 from 8 to 27 k-points (the k1×k2 loops), and
GDF is 10–18x faster than the default FFTDF on the same cell.  The first run
in the batch (2-atom GDF, stock) carried 12 s of first-call initialisation in
`get_hcore` and is not quoted.  Electrolites is inert on the periodic path
(same times, same energies, no entry point reached).
