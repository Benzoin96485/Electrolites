# Status of `instructions/06.gpu4pyscf_kernel_targets.md` in this repository

What this repository has done, per catalogue ID, as of commit `06eb0a8`
(GPU4PySCF **1.8.1**, one A100).  The catalogue is the file
`LLM-GPU-QM/instructions/06.gpu4pyscf_kernel_targets.md`; its §4 is the target
list and its §5 the ranking.  This file is a **record**, not a re-measurement:
every number quoted here is one this repository measured and wrote down in
`docs/`, and the pointer to where is given in every row.

Three cautions carry over from the catalogue itself and apply to everything
below.

- The catalogue's own line numbers and several of its shares are **1.7.0**
  (catalogue §7 Q1).  This repository now runs on 1.8.1, and one entry (**A3**)
  was closed by upstream in the interval rather than by anything here.
- The catalogue's §5 ranking is built from **§2's** profiles — 42/113-atom
  molecules at def2-SVP.  This repository measures on 284/545-atom clusters at
  6-31G\*–def2-TZVPD.  Catalogue §3.2 shows those two regimes **invert** the
  ordering of K against XC.  Where a row below says "done", it means done and
  measured *in this repository's regime*; where the two regimes disagree the
  row says so.
- Nothing here is a speedup claim in the catalogue §0 sense.  These are this
  repository's own A/B timings, not `harness/evaluate.py` runs with a bootstrap
  confidence interval.

Status vocabulary: **done** = implemented, measured, and correctness-checked;
**partial** = implemented for some of the entry's scope, with the uncovered
part named; **open** = nothing here touches it; **upstream** = GPU4PySCF closed
it; **rejected** = tried and measured not to pay, with the measurement given;
**n/a** = out of this repository's scope (a negative control, or a greenfield
item with no incumbent).

---

## 1. The ranking, top to bottom

Catalogue §5's ranked list is the actionable part.  This is that list with a
status column.

| Rank | Project | Status | Where |
|---|---|---|---|
| 1 | **G1 + B2** — grouped GEMM in the XC gradient | **done** (gradient); G1 **partial** as infrastructure | `fastxcgrad`; `docs/README_fastxcgrad.md` |
| 2 | **B4a** — the J engine's multi-density block cap | **partial, and its premise corrected** | `fastj` now covers `n_dm > 1`; `docs/ROUND3_B4A.md` |
| 3 | **A5** — the grid AO-evaluation kernel, and the second pass | **partial**: the second pass is gone, the kernel is untouched | `fastxc`, `fastxcnlc`; `docs/ROUND2_1.8.1.md` §5 |
| 4 | **B1** — the §3.6 treatment in `ejk_ip1` | **done** | `fastejk`; `docs/README_fastejk.md` |
| 5 | **A4** — `nr_rks` for UKS and mGGA | **partial**: mGGA covered in the *gradient*, open in the SCF; UKS open | `fastxcgrad` covers mGGA; `fastxc.py:65` declines it |
| 6 | **F2 + F3** — spatial screening in `int3c1e`, PCM D/S | **open** | — |
| 7 | **B5** — batch the atom dimension in the XC Hessian | **open** | — |
| 8 | **B4b + B4c** — K in CPHF, per-density screening | **open** | `fastk.py:175` declines `n_dm != 1` |
| 9 | **B3** — `ejk_ip2` | **open** | — |
| 10 | **B6** — batch `hess_nuc_elec` | **open** | — |
| 11 | **A3** — the J engine's host-side Hermite transform | **upstream**: 1.8.0 moved it to the GPU | `docs/UPGRADE_1.8.1.md` §6 |
| 12 | **B4d + D8** — batching and AO repetition on the response path | **open** | — |
| 13 | **D2** — preallocate `lr_eigh`'s trial space | **open** | — |
| — | **A12 / F1 / C2 / E3** — the low-contamination four | **open** (all four) | — |

**The highest-ranked entry with nothing done to it is now rank 3, A5.**  Rank
2, B4a, was taken in `docs/ROUND3_B4A.md`; what came back is a 1.23x on the
Coulomb build at `n_dm > 1` and a correction to the entry's premise, since the
part of it that is really about the four-density cap measures **1.09x** once
the per-quartet overheads are fixed.  Ranks 2, 8 and 12 are one family: every one of them is a
build with `n_dm > 1`, and this repository's SCF modules all decline that.

---

## 2. Section A — single-point energy (SCF)

| ID | Status | What was done, and what is left |
|---|---|---|
| **A1** short-range (`erfc`) K | **open** | `fastk.py:171` `if omega < 0: return False`, unchanged.  HSE06 still falls through, and catalogue §3.7 keeps it in the suite for exactly that reason.  1.8.x asks for `lr*erf + sr*erfc` in one call, and `_get_k_dispatch` (`fastk.py:634-647`) decomposes it as `sr*(full range) + (lr-sr)*(long range)` through `erf + erfc == 1` — so the **fused RSH request** is served, but the standalone short-range operator is not, and the decomposition is not the kernel A1 asks for |
| **A2** the unrolled-free high-`lij+lkl` `md_j` classes | **done** | `codegen/gen_j_high.py` writes **seventeen** classes from `(lij,lkl)` alone — every class an spdf basis reaches.  J build **1.82x** on `PfPMT`/6-31G\* and **2.07x** on `PfPMT`/def2-TZVP, where the 64 % of the work that fell through at 1.00x now runs at **2.49x**; 1.05x → **5.12x** on a 54-atom def2-TZVP cluster.  `docs/ROUND2_1.8.1.md` §1.  **Left**: the classes beyond `(6,6)` that a g-function basis reaches — thirteen of seventeen are a table entry and a sweep, four need a different `Rt` strategy (ROUND2 "What is left" 3) |
| **A3** host-side Hermite transform | **upstream** | 1.8.0's `_dm_to_Rt`/`_Rt_to_dm` replaced the host-side `Et_dot_dm`/`jengine_dot_Et`.  `docs/UPGRADE_1.8.1.md` §6.  Nothing for this repository to do |
| **A4** `nr_rks`'s UKS / non-Hermitian / multi-density / mGGA paths | **partial** | `fastxc.py:65` still asserts `dms.ndim == 2` and declines mGGA in the SCF.  **mGGA is covered in the gradient** by `fastxcgrad` (3.98x–4.46x on the two mGGA columns, `docs/README_fastxcgrad.md`), which is direct evidence *against* catalogue §3.5's "the mGGA XC build is not a gap worth closing" — but that evidence is from the gradient's contraction, not the SCF's, so catalogue **Q13 is still open**: nobody has measured whether the SCF mGGA build is launch-bound at small `nao_sub`.  UKS is untouched |
| **A5** the grid AO-evaluation kernel | **partial** | The **repetition** is addressed: `fastxc` fuses GPU4PySCF's two grid passes into one (P7+P8), and `fastxcnlc` fuses the XC and VV10 passes on 1.8.x, worth **1.036x** on `get_veff` for ωB97M-V (`docs/ROUND2_1.8.1.md` §5).  The **kernel** — `nr_eval_gto.cu`'s run-time `nprim` bound, absent per-point radial screening, per-shell recomputation — is untouched.  `docs/README_fastxcgrad.md` §312 measures `eval_ao` at deriv=2 as **26 %** of the gradient's XC block and names it as A5, not B2 |
| **A6** VV10 gradient and Hessian | **partial** | `fastxcgrad` replaces `grad.rks.get_nlc_exc`, so the VV10 **gradient** grid contraction is covered.  The VV10 **Hessian** (`hessian/rks.py _get_enlc_deriv2`) is open |
| **A7** extend the RSH fusion to the high-`l` classes | **open** | `get_k_rsh` (`fastk.py:583-588`) fuses only the `IMPLEMENTED` one-thread-per-quartet classes; the comment on the `IMPLEMENTED2` branch says it outright — *"These still take one pass per operator"*.  Catalogue sizes it at ~9 % of K for RSH functionals |
| **A8** Rys roots: run-time `nroots`, shared memory, fp64 division | **done for the kernels here** | P1/P2/P5 are applied throughout `fastk`, `fastejk` and the written `gen_khigh` family: roots in a register array rather than shared memory, `rsqrt(fma(aij*akl, inv_om2, aij+akl))` as the sole source of all three scalings (`gen_khigh.py:676-680`).  GPU4PySCF's own `rys_roots.cu` is unchanged — anything still falling through pays it |
| **A9** screening and task generation | **open** | The union-over-densities `dm_cond` is untouched, and `fastj`'s multi-density kernels take it as it is, so they screen exactly what GPU4PySCF screens.  Per-density screening is B4c and is a separate, algorithmic change |
| **A10** libxc dispatch and reduction | **open** | Still `[?]` in the catalogue and still not separately instrumented here |
| **A11** locality in the Becke partition weights | **open** | Catalogue Q6: measured at 0.1 % on 42/113 atoms and never measured at 500+.  This repository's `benchmarks/scf_anatomy.py` on `PfPMT` puts grid construction inside the 1.9 % that DIIS, the diagonalisation and grid construction share (`docs/ROUND2_1.8.1.md` §2), so it is still a tail at 284 atoms |
| **A12** ECP task generation has no screening | **open** | One of the catalogue's four low-contamination entries |
| **A13** the density-fitting (RI) path | **open** | Density fitting is untouched by every module here (`README.md`, "Limits") |

### Two SCF-side results this repository added that the catalogue does not have

- **The "Amdahl ceiling" at 1.9 %, not a third.**  `docs/UPGRADE_1.8.1.md`
  claimed DIIS, the fp64 diagonalisation and grid construction were about a
  third of the SCF wall clock.  Instrumenting the SCF itself
  (`benchmarks/scf_anatomy.py`) puts those three at **1.9 %** and `get_veff` at
  **97.8 %**.  The error was comparing a *fixed-density* `get_veff` speedup
  against a *converged* wall clock.  `docs/ROUND2_1.8.1.md` §2.
- **Compile time as a first-class cost** (catalogue **G3**).  Splitting the
  generated files so only the angular-momentum classes the basis can reach are
  compiled takes a 6-31G\* job's first-run NVRTC from **162 s to 37 s (4.3x)**.
  `docs/ROUND2_1.8.1.md` §4.  The catalogue lists G3 as a constraint on
  generator work; this is a measurement of it and a fix.

---

## 3. Section B — analytic derivatives

| ID | Status | What was done, and what is left |
|---|---|---|
| **B1** `ejk_ip1` | **done** | `fastejk` replaces `grad.rhf._jk_energy_per_atom` with the full P1–P5 treatment, every class an spdf basis reaches, full-range and long-range.  `docs/README_fastejk.md`.  Spin-restricted closed-shell only |
| **B2** the XC gradient: batch the GEMMs | **done, and past what the catalogue asked** | `fastxcgrad`.  The catalogue asks for P8 (grouped GEMM); this went further and **reformulated** the contraction so the intermediate matrices are never formed — 2 GEMMs per block where GPU4PySCF issues 10 for GGA, 4 where it issues 19 for mGGA.  `get_exc` at fixed density: **3.43x–4.46x** on 284/545 atoms at 6-31G\*, **2.22x–2.88x** at def2-TZVPD.  `docs/README_fastxcgrad.md` |
| **B3** `ejk_ip2` | **open** | The Hessian is untouched here |
| **B4a** J engine multi-density block cap | **partial** | The cap is still live on 1.8.1 (`scf/j_engine.py:299-301` pins `n_dm` to 4; `MD_build_j` steps `dm_offset += 4`), and `fastj` now has multi-density kernels: **1.23x** on the whole J build at `n_dm` = 8, 16 and 32 on a 284-atom cluster and **1.19x** on a 545-atom one, 1.57x/1.55x at `n_dm` = 4.  But that gain is the kernel's, not the cap's: measured on GPU4PySCF the cap looks worth 3.4x, measured on *our* kernels it is worth **1.09x**, because the 3.4x is the per-quartet overhead (P1–P5) the cap repeats and the single-density rewrite had already removed it.  Blocks of 8 and 16 were written, measured at **0.90x** and **0.64x** against blocks of 4, and are not shipped.  `docs/ROUND3_B4A.md` |
| **B4b** K in CPHF | **open** | `fastk.py:175` `if n_dm != 1: return False` |
| **B4c** per-density screening | **open** | Inherits A9 |
| **B4d** `nr_rks_fxc` | **open** | The common entry point for Hessian, TDDFT, polarizability and NMR; no module here patches it |
| **B5** the XC Hessian's Python atom loop | **open** | — |
| **B6** `hess_nuc_elec` | **open** | — |
| **B7** `hess_nuc_elec_ecp` | **open** | — |
| **B8** Hessian 1e integrals on the CPU | **open** | Catalogue flags this as an Amdahl tail (0.1 % at 42 atoms) listed for re-measurement, not for fixing |
| **B9** grid-response kernels | **open** | `fastxcgrad` **declines** the grid-response path and hands it to GPU4PySCF (`docs/README_fastxcgrad.md`) |
| **B10** third and higher derivatives | **n/a** | Greenfield, no incumbent |
| **B11** DF-Hessian auxiliary integrals on the CPU | **open** | Density fitting untouched |

One item outside the catalogue's B list was done: **`fastgradh`** puts the
nuclear-attraction part of the one-electron derivative on the GPU.  The
catalogue measures the CPU one-electron integrals at 8–13 % of a gradient
(`docs/README_fastejk.md`'s own profile) without giving them an ID.

---

## 4. Sections C–H

| Section | Status |
|---|---|
| **C** response properties (C1 polarizability/CPKS, C2 NMR/GIAO, C3 IR/Raman, C4 EDA, C5 C6) | **all open.**  Every one of them routes through `nr_rks_fxc` and/or a multi-density `get_jk`, i.e. through B4a/B4b/B4d |
| **D** excited states (D1 Davidson, D2 `lr_eigh`, D3 third XC derivative on CPU, D4 `get_ab`, D5 TDDFT+PCM, D6 ris, D7 spin-flip, D8 AO re-evaluation) | **all open**, same reason |
| **E** real-space and population analysis (E1–E4) | **all open.**  E3 is greenfield and one of the catalogue's four low-contamination entries |
| **F** dispersion, solvation, fields, embedding (F1–F7) | **all open.**  F2 is rank 6 and reaches PCM, QM/MM, ESP, CHELPG and `hess_nuc_elec` through one kernel |
| **G1** extend `_grouped_block_loop` everywhere | **partial**: it reached `grad.rks.get_exc` (B2) and, in a different form, the SCF's `nr_rks` (`fastxc`).  It has **not** reached `nr_rks_fxc` (B4d) or `hessian.rks._get_vxc_deriv1/2` (B5) |
| **G2** fusing small memory-bound kernels | **partly rejected on a measured ceiling.**  Deleting *every* `vk` `atomicAdd` — the ceiling for a block reduction, not an implementation of one — is worth **1.1 %** at 6-31G\* and **2.2 %** at def2-TZVP, so a block reduction cannot pay.  `docs/ROUND2_1.8.1.md` §3b.  This is the same Amdahl warning the catalogue attaches to G2 |
| **G3** NVRTC/ptxas compile time | **done**: 162 s → 37 s, above |
| **G4** the multi-GPU layer | **open** and deliberately so: every module here asserts a single device |
| **H1–H3** negative controls | **n/a** by construction |

---

## 5. Things this repository tried and measured *not* to pay

These belong in a status record because re-attempting them is the main way
effort gets wasted.  Each is a measurement, not an opinion.

| Idea | Measured outcome | Where |
|---|---|---|
| Per-lane `switch (gout_id)` unrolling for the wide K classes | wins up to 5.6x on classes that keep 32 quartets, loses up to 3.7x on those that do not; **no net gain** (23.16 s vs 23.01 s) and 444 000 lines of CUDA | catalogue §3.3 |
| The `pair_vk` block reduction (JoltQC) | ceiling measured first: deleting every `vk` atomicAdd is **1.1 % / 2.2 %** | `docs/ROUND2_1.8.1.md` §3b |
| The log-magnitude AO screen, post-hoc | the headroom was a `minao` artifact — **65 %** of the XC GEMM work on the initial guess, **4 %** on a converged density — and the implementation measured **0.28x** | `docs/ROUND2_1.8.1.md` §3c |
| A fused single grid pass *without* batching | **0.54x** — P7 is negative on its own and only pays with P8 | catalogue §3.6 |
| Carrying 8 or 16 densities through one Coulomb pass instead of 4 | **0.90x** and **0.64x** against four, and the ceiling for removing the cap altogether is **1.09x** once the per-quartet overheads are fixed | `docs/ROUND3_B4A.md` §4 |

The AO screen's headroom is **not** zero: `docs/ROUND2_1.8.1.md`'s "What is
left" item 1 keeps it open as a per-(block, shell) log-magnitude estimate moved
*upstream* of `eval_ao`, which is a different change from the one that measured
0.28x.  That is catalogue **A5**(b) by another route.

---

## 6. Two measurement hazards found here, which constrain any future entry

Both are the same kind of finding as catalogue §3.7's warning, and both are
reasons a number in a table can be wrong while every correctness test passes.

1. **A build timed on the `minao` initial guess is not the build an SCF runs.**
   The guess is far sparser, so it screens harder.  On the 545-atom cluster the
   difference is a factor of **1.4** in the reported ratio, and it is what made
   the AO screen look like a 65 % opportunity when it is a 4 % one.
   `README.md`, `docs/ROUND2_1.8.1.md` §3c, §6.
2. **The fixed-density table's gradient row depends on `OMP_NUM_THREADS`**,
   because its unpatched half is CPU libcint.  `docs/ROUND2_1.8.1.md` §6.
3. **A ceiling measured on GPU4PySCF is not the ceiling for a kernel this
   repository has already rewritten.**  The Coulomb build's four-density cap
   makes GPU4PySCF repeat a per-pass cost worth 9.65 densities of contraction,
   which reads as a 3.4x opportunity; the same cost in these kernels is worth
   0.37 densities, so the real opportunity is 1.09x.  Any catalogue entry whose
   Amdahl share is "what GPU4PySCF wastes" has to be re-measured against
   whatever has already replaced that code.  `docs/ROUND3_B4A.md` §4.

Catalogue §3.4's size-independence rule — re-measure on a second,
differently-sized system and count the result only if the ratios agree — is
followed throughout `docs/`, and every table there gives both systems.

---

## 7. What the next round should take

By catalogue §5's ranking, filtered to what is open:

1. **A5** (rank 3, 18–21 % of SCF) — the kernel itself, now that the
   repetition is gone.  B4a, rank 2, was taken in `docs/ROUND3_B4A.md`; what is
   left of it there is a measured **1.09x**, so the rest of the multi-density
   family (B4b, B4c, B4d) should have its ceiling measured on **this
   repository's** kernels before anything is written — measuring it on
   GPU4PySCF's overstates it by a factor of three, which is that round's main
   result.
2. **A4**'s SCF mGGA path (rank 5), which first needs catalogue **Q13**
   measured: is the SCF mGGA build launch-bound at small `nao_sub`?  This
   repository's own mGGA gradient result (4.46x on M06-2X) is a reason to
   expect yes, but it is not that measurement.
3. **F2** (rank 6, 13.9 % of SCF at PCM defaults), where P10 and P11 from
   `fastnlc` apply verbatim.

For the LLM-GPU-QM task suite rather than for throughput, the catalogue's own
advice stands unchanged: **A12, F1, C2, E3** are worth more than ranks 1–13
because they are the least contaminated, and all four are open.
