# Status of `instructions/06.gpu4pyscf_kernel_targets.md` in this repository

What this repository has done, per catalogue ID, as of the `fastjk` round
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
| 6 | **F2 + F3** — spatial screening in `int3c1e`, PCM D/S | **F2 done, and its premise corrected; F3 rejected** | `fast1e`; `docs/README_fast1e.md` |
| 7 | **B5** — batch the atom dimension in the XC Hessian | **done** (GGA), and its stated mechanism corrected | `fastxchess`; `docs/README_fastxchess.md` |
| 8 | **B4b + B4c** — K in CPHF, per-density screening | **B4b done, on both of its entry points, and its ceiling measured; B4c rejected on a measured premise** | `fastk` (multi-density), `fastjk` (fused); `docs/README_fastjk.md`, `docs/ROUND4_B4B.md` |
| 9 | **B3** — `ejk_ip2` | **partial, and both of the entry's expectations reversed**: what pays is the launch scheme the entry does not mention (**1.13x**, two lines of Python), while P1 measures 1.02x and P3 — which the entry says is worth *more* here than on K — is a **1.75x regression** | `fastejk2`; `docs/README_fastejk2.md`, `docs/ROUND4_B4B.md` §6 |
| 10 | **B6** — batch `hess_nuc_elec` | **upstream**: 1.8.1 replaced the per-charge loop | `docs/HESSIAN_1.8.1.md` §1 |
| 11 | **A3** — the J engine's host-side Hermite transform | **upstream**: 1.8.0 moved it to the GPU | `docs/UPGRADE_1.8.1.md` §6 |
| 12 | **B4d + D8** — batching and AO repetition on the response path | **B4d done** on the response path, and the entry's prescription corrected; **D8 re-sized and rejected** | `fastxcfxc`; `docs/README_fastxcfxc.md`, `docs/RESPONSE_1.8.1.md` |
| 13 | **D2** — preallocate `lr_eigh`'s trial space | **rejected on a measured size**: 0.19 GB of copying over a whole 22 s TDA run | `docs/RESPONSE_1.8.1.md` §5 |
| — | **A12 / F1 / C2 / E3** — the low-contamination four | **open** (all four) | — |

**The highest-ranked entry with nothing done to it is now rank 3, A5.**
Ranks 6, 7 and 12 were taken in `docs/README_fast1e.md`,
`docs/README_fastxchess.md` and `docs/README_fastxcfxc.md`; ranks 10 and 11 had
been closed upstream and rank 13 measured too small to take.  Rank 2, B4a, was
taken in `docs/ROUND3_B4A.md`, with a correction to its premise: the part of it
that is really about the four-density cap measures **1.09x** once the
per-quartet overheads are fixed.  Rank 8 was taken in
`docs/README_fastjk.md` and `docs/ROUND4_B4B.md`, and it is the largest single
result here so far — a B3LYP polarizability on penicillin V goes 1.15x → **2.08x**
and a five-root TDA 1.24x → **1.56x**, because the fused `get_jk` the response
path actually calls was a third code path that no module was on.

Ranks 2, 8 and 12 are one family: every one of them is a build with `n_dm > 1`,
and the round that closed 8 also bounded the family.  Fitting the exchange
build as `T = R + n_dm·C` puts **all** of the gain in `R`, the per-quartet
integral work: `R` is 3.55x better here, `C` — the density contraction and its
atomicAdds — is **1.01x**, the same code.  So the same kernels are worth 2.8x
at the `n_dm` a response solve asks for and 1.16x at the `n_dm = 126` a
Hessian's CPHF asks for, and no further work on the integral side can change
that.  `C` is what §7 now nominates.

**A13 is not in §5's ranking at all**, and the round that took it (`fastdf`,
`docs/README_fastdf.md`) ended with its largest result outside the module it
was written for.  The module itself is the first here to end in a mostly
negative result worth stating as one.  A density-fitted run is a *fourth* two-electron code
path that no module here was on, so the coverage gap was real; but the build
on that path is already at **71 % of memory bandwidth and 15.8/17.8 TF/s of an
A100's 19.5 TF/s** stage by stage, which is not what any of the direct-SCF
rounds found.  The only structural saving left in it is that a symmetric
exchange matrix need not have both of its triangles computed — worth
**1.11x–1.17x** at a thousand basis functions and nothing at four hundred,
because cuBLAS's `dsyrk` gives the factor of two back there.  Everything past
that has to make the arithmetic smaller rather than rearrange it, and
`benchmarks/df_fill_probe.py` sizes what is available: the screening stores
**22 %** of the AO-pair space at 284 atoms and the contraction is dense over
all of it.  `docs/DF_WORK_LIST.md` ranks the nine candidates that follow from
that, including the two that are capacity fixes rather than speedups.

> **W1 of that list has since been taken, and it changes the conclusion above
> without contradicting it** (`electrolites/fastdfj.py`,
> `docs/README_fastdfj.md`).  The *contraction* is still at the machine's
> limits — nothing below rearranges it.  What the round found is that at 284
> atoms the contraction should not be happening at all: `only_dfj=True`
> (ORCA's RIJONX — fitted J, exact K) builds **no cderi**, was **broken in
> GPU4PySCF 1.8.1** by an argument-order bug, and once repaired runs the whole
> SCF of the 284-atom case in **64.1 s** against full DF's **300.5 s**,
> **4.69x**, level with the direct path.  88.7 s of full DF's time is
> `_cholesky_eri` alone, for a tensor RIJONX never makes.  Two further
> measurements came with it: the three-centre Coulomb engine
> (`df/j_engine_3c2e.py`), which a *pure* DF functional already runs on, is
> the **fastest** Coulomb build measured here at every size — 1.67x the direct
> four-centre one at 284 atoms — and `_DFHF.get_veff` rebuilds the Fock matrix
> from the whole density every cycle, worth **1.33x** once an `only_dfj`
> object drives its own class's instead.

What the round *did* find is a coverage bug of the kind
`docs/OPENSHELL_TMC.md` and `docs/VV10_OPENSHELL.md` each found once before.
`density_fit()` builds its class as `set_class(mf, (_DFHF, cls))`, so
`_DFHF.get_veff` precedes `RKS.get_veff` and `UKS.get_veff` in the method
resolution order and **`fastxcnlc` was inert on every density-fitted run**.
A DF wB97M-V walked the grid twice per iteration with `patch_all()` loaded,
and `nr_nlc_vxc` is 77-80 % of such a run.  Patching that third binding takes a DF wB97M-V SCF on penicillin V from
16.0 s to 14.2 s, **1.13x** -- and the whole patched set from 22.0 s to
14.2 s, 1.55x, the rest of which `fastnlc` was already reaching.  Three
rounds, three patch targets that were not code paths; the only thing that has
ever found one is `benchmarks/coverage_probe.py`.

---

## 2. Section A — single-point energy (SCF)

| ID | Status | What was done, and what is left |
|---|---|---|
| **A1** short-range (`erfc`) K | **open** | `fastk.py:171` `if omega < 0: return False`, unchanged.  HSE06 still falls through, and catalogue §3.7 keeps it in the suite for exactly that reason.  1.8.x asks for `lr*erf + sr*erfc` in one call, and `_get_k_dispatch` (`fastk.py:634-647`) decomposes it as `sr*(full range) + (lr-sr)*(long range)` through `erf + erfc == 1` — so the **fused RSH request** is served, but the standalone short-range operator is not, and the decomposition is not the kernel A1 asks for |
| **A2** the unrolled-free high-`lij+lkl` `md_j` classes | **done** | `codegen/gen_j_high.py` writes **seventeen** classes from `(lij,lkl)` alone — every class an spdf basis reaches.  J build **1.82x** on `PfPMT`/6-31G\* and **2.07x** on `PfPMT`/def2-TZVP, where the 64 % of the work that fell through at 1.00x now runs at **2.49x**; 1.05x → **5.12x** on a 54-atom def2-TZVP cluster.  `docs/ROUND2_1.8.1.md` §1.  **Left**: the classes beyond `(6,6)` that a g-function basis reaches — thirteen of seventeen are a table entry and a sweep, four need a different `Rt` strategy (ROUND2 "What is left" 3) |
| **A3** host-side Hermite transform | **upstream** | 1.8.0's `_dm_to_Rt`/`_Rt_to_dm` replaced the host-side `Et_dot_dm`/`jengine_dot_Et`.  `docs/UPGRADE_1.8.1.md` §6.  Nothing for this repository to do |
| **A4** `nr_rks`'s UKS / non-Hermitian / multi-density / mGGA paths | **partial** | `fastxc.py:65` still asserts `dms.ndim == 2` and declines mGGA in the SCF.  **mGGA is covered in the gradient** by `fastxcgrad` (3.98x–4.46x on the two mGGA columns, `docs/README_fastxcgrad.md`), which is direct evidence *against* catalogue §3.5's "the mGGA XC build is not a gap worth closing" — but that evidence is from the gradient's contraction, not the SCF's, so catalogue **Q13 is still open**: nobody has measured whether the SCF mGGA build is launch-bound at small `nao_sub`.  UKS is untouched |
| **A5** the grid AO-evaluation kernel | **partial** | The **repetition** is addressed: `fastxc` fuses GPU4PySCF's two grid passes into one (P7+P8), and `fastxcnlc` fuses the XC and VV10 passes on 1.8.x, worth **1.036x** on `get_veff` for ωB97M-V (`docs/ROUND2_1.8.1.md` §5).  The **kernel** — `nr_eval_gto.cu`'s run-time `nprim` bound, absent per-point radial screening, per-shell recomputation — is untouched.  `docs/README_fastxcgrad.md` §312 measures `eval_ao` at deriv=2 as **26 %** of the gradient's XC block and names it as A5, not B2 |
| **A6** VV10 gradient and Hessian | **partial** | `fastxcgrad` replaces `grad.rks.get_nlc_exc`, so the VV10 **gradient** grid contraction is covered.  The VV10 **Hessian** (`hessian/rks.py _get_enlc_deriv2`) is open.  The VV10 **response** -- `hessian/rks.py nr_rks_fnlc_mo`, which the catalogue does not list anywhere and which is **80 % of a wB97M-V polarizability** -- is `fastfnlc`, 1.43x over a real CPKS solve; see §4 |
| **A7** extend the RSH fusion to the high-`l` classes | **open** | `get_k_rsh` (`fastk.py:583-588`) fuses only the `IMPLEMENTED` one-thread-per-quartet classes; the comment on the `IMPLEMENTED2` branch says it outright — *"These still take one pass per operator"*.  Catalogue sizes it at ~9 % of K for RSH functionals |
| **A8** Rys roots: run-time `nroots`, shared memory, fp64 division | **done for the kernels here** | P1/P2/P5 are applied throughout `fastk`, `fastejk` and the written `gen_khigh` family: roots in a register array rather than shared memory, `rsqrt(fma(aij*akl, inv_om2, aij+akl))` as the sole source of all three scalings (`gen_khigh.py:676-680`).  GPU4PySCF's own `rys_roots.cu` is unchanged — anything still falling through pays it |
| **A9** screening and task generation | **open, and now bounded** | The union-over-densities `dm_cond` is untouched: `fastj`'s and `fastk`'s multi-density kernels and `fastjk` all screen exactly what GPU4PySCF screens.  Per-density screening is B4c, and it is **measured at 1.14x–1.18x on the contraction, not the algorithmic gain the entry claims** — `docs/ROUND4_B4B.md` §5 |
| **A10** libxc dispatch and reduction | **open** | Still `[?]` in the catalogue and still not separately instrumented here |
| **A11** locality in the Becke partition weights | **open** | Catalogue Q6: measured at 0.1 % on 42/113 atoms and never measured at 500+.  This repository's `benchmarks/scf_anatomy.py` on `PfPMT` puts grid construction inside the 1.9 % that DIIS, the diagonalisation and grid construction share (`docs/ROUND2_1.8.1.md` §2), so it is still a tail at 284 atoms |
| **A12** ECP task generation has no screening | **open** | One of the catalogue's four low-contamination entries |
| **A13** the density-fitting (RI) path | **partial, and two of its four claims closed upstream** | `fastdf` replaces `df/df_jk.py:get_jk`, the build every RI/DF calculation's two-electron half goes through -- a *fourth* code path that `fastj`, `fastk` and `fastjk` are all off.  **1.09x-1.41x** on that build with the cderi on the device and **1.64x-4.53x** with it on the host -- the regime the entry's own headline is about, where GPU4PySCF streams the whole tensor across PCIe every cycle and this module holds as much of it resident as fits.  Whole DF SCF runs: 1.12x-1.22x on B3LYP and **1.55x** on wB97M-V/B97M-V, the second of which is not this module at all (see below).  Of the entry's four pieces of evidence, the CPU `int2c2e` and the `synchronize` that defeated the prefetch were **closed upstream** in 1.8.x, the "two fancy-index scatters" are a CUDA kernel on 1.8.1, and the RSH duplication does not happen.  What the entry does not mention is where the time actually is, and `docs/README_fastdf.md` §3.4 measures those three stages at **71 % of memory bandwidth and 15.8/17.8 TF/s of an A100's 19.5** -- there is no rearrangement left.  **Left**: the arithmetic is dense where the screening stores only 22 % of the AO-pair space at 284 atoms (§3.6) -- **measured since, and it is not worth the 4.5x that number suggests -- or anything**: a GEMM addresses AO tiles, not pairs, and the tile fill is 69.4 % at 284 atoms and 98.2 % at 113, because the pair fill is the product of an atom-pair factor a tile can reach and a shell-structure factor at one-to-five-AO granularity that no AO tile can.  `fastdf`'s first contraction can run block-sparse above a measured fill (1.09x on the exchange block at 284 atoms and 1.20x at 545, but **0.85x-1.02x on a whole `get_jk` call** over five paired readings, so it is opt-in and off by default, and a loss below), `docs/README_fastdfblock.md`; the one-time cderi build, which is **39 %** of a 15-cycle taxol DF SCF; and the DF Hessian's own MO-basis CPHF build, a fifth code path.  **Since**: `fastdfj` repairs `only_dfj=True`, which does not run at all on 1.8.1, and a 284-atom hybrid is then **4.69x** faster than full DF end to end with no cderi built (`docs/README_fastdfj.md`).  **And since**: `fastdfgrad` takes the **DF gradient's two-electron derivative**, which is 72-97 % of a density-fitted gradient depending on the basis.  Two things there, and neither is what W3 predicted: the three-centre integrals in its first loop are ones the SCF's cderi already holds (`aux_coef @ aux_coef.T == metric` makes the reuse exact), and the derivative kernel is launched with four shell pairs per thread block where one is **1.26x-3.15x** faster.  The contraction — which this entry's own text, and W3 after it, pointed at — is 2.7-14.4 % of the function and already runs at up to 71 % of fp64 peak (`docs/README_fastdfgrad.md`).  **And since that**: `fastdfcderi` takes the entry's own "one-time cderi build", which is **1.14x-1.83x** with the tensor on the device and **2.72x-3.20x** with it on the host (and **2.95x** at 284 atoms, 87.1 s -> 29.5 s), and a 113-atom host-resident DF SCF **1.37x** around it.  Three things, and the largest is one no reading of the source would have found: **65 %** of a host-resident 113-atom build — and 41.5 s of a 72.5 s 284-atom one — is `cudaHostAlloc` page-locking the cderi's host buffer at 2.09 GB/s, where pre-allocated memory registered with `cudaHostRegister` reaches 8.6 GB/s -- it is the allocation that is slow, not the page-locking.  The other two are the metric contraction as a sparse apply and a triangular *multiply* (**1.9x-2.1x**; the triangular *solve* the work list proposed is a loss, cuBLAS's `dtrsm` never reaching 0.66 of its `dgemm`) and the same shell-pair aggregation constant again (**1.19x-1.50x** on the block loop).  `docs/README_fastdfcderi.md` |

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
| **B3** `ejk_ip2` | **partial: the launch scheme done, the P1–P5 treatment applied and measured not to pay** | Profiled rather than assumed (`benchmarks/ejk_ip2_probe.py`): `_partial_ejk_ip2` is 8.63 s on penicillin V and 98.9 % of it is in the two ip2 kernels, but **63 % of that is GPU4PySCF's *general* kernel** — it unrolls seven ip2 classes (`0000`…`1111`, nothing with a d function) against twenty-five for the exchange build, so "as B1" covers 37 % of the stage.  `fastejk2` then applied the treatment to that 37 % rather than estimating it: **P1 (every fp64 division out of the primitive and root loops) is 1.02x, and P3 (a 128-register cap) is 0.58x** — a 1.75x regression where the entry says P3 should pay more here than on K, because these kernels want all 255 registers and capping them buys occupancy with spills.  What does pay is not in the entry: the block shape, which `_ip2_quartets_scheme` derives from the shared-memory budget alone, **1.13x**.  `docs/README_fastejk2.md` |
| **B4a** J engine multi-density block cap | **partial** | The cap is still live on 1.8.1 (`scf/j_engine.py:299-301` pins `n_dm` to 4; `MD_build_j` steps `dm_offset += 4`), and `fastj` now has multi-density kernels: **1.23x** on the whole J build at `n_dm` = 8, 16 and 32 on a 284-atom cluster and **1.19x** on a 545-atom one, 1.57x/1.55x at `n_dm` = 4.  But that gain is the kernel's, not the cap's: measured on GPU4PySCF the cap looks worth 3.4x, measured on *our* kernels it is worth **1.09x**, because the 3.4x is the per-quartet overhead (P1–P5) the cap repeats and the single-density rewrite had already removed it.  Blocks of 8 and 16 were written, measured at **0.90x** and **0.64x** against blocks of 4, and are not shipped.  `docs/ROUND3_B4A.md` |
| **B4b** K in CPHF | **done, on both entry points** | Two of them, and the entry names only one.  (a) `jk._VHFOpt.get_k` at `n_dm = 3·natm`, which the Hessian's CPHF calls: `fastk` no longer declines it, and the same change makes `hermi = 0` work.  **1.16x** on that row, 56.49 → 53.36 s on a 42-atom Hessian, the trace agreeing to 2.8e-11 relative.  (b) `jk._VHFOpt.get_jk`, the **fused** build every response property calls and which no module here was on: `fastjk`, 65 classes, `hermi` 0 and 1, **2.79x/1.96x/1.65x** at `nset` = 1/3/5 and **2.08x** on a whole polarizability.  `docs/README_fastjk.md` |
| **B4c** per-density screening | **rejected on a measured premise** | The entry's mechanism — "each `dm1[A]` has support only near atom A" — is false on 1.8.1, whose CPHF solve runs in the **MO** basis and builds its AO densities as `C·mo1·C_occ^T` from delocalised canonical orbitals.  A perfect per-density screen leaves 87 % of the (quartet, density) work at 42 atoms and 98 % at 10.  `benchmarks/ndm_screen_probe.py`, `docs/ROUND4_B4B.md` §5 |
| **B4d** `nr_rks_fxc` | **done on the response path; open on the Hessian's, and re-sized there** | Two different functions on 1.8.1, and they want different treatments.  (a) The **response** path -- CPKS, TDA/TDDFT, NMR, IR/Raman, C6 -- still goes through `dft/numint.py:1339 nr_rks_fxc`, which is **23.2 %** of a polarizability and **28.4 %** of a five-root TDA run on 1.8.1 (`docs/RESPONSE_1.8.1.md`).  `fastxcfxc` replaces it for LDA and GGA: **2.93x/3.97x** at `nset = 3` and **3.35x/4.65x** at `nset = 5` (`hermi` 1/0) at 42 atoms, **1.43x/1.91x** and **1.58x/2.21x** at 113, and the entry's prescription is corrected -- see the row below.  `docs/README_fastxcfxc.md`.  (b) The **Hessian's** CPHF solve is a different function, `hessian/rks.py:3486 _nr_rks_fxc_mo_task`, which contracts to the MO basis per block group.  Instrumented rather than inferred (catalogue Q3 infers ~11 % of a Hessian): **6.5 %** on penicillin V, **5.9 %** on taxol.  Its per-`nset` loop looks like B5's launch problem but is not -- the arithmetic is `nset x nao_sub^2 x ngrids`, about 6.0 TFLOP per call on penicillin V, and it does that in 0.91 s = **6.6 TFLOP/s**, a third of the card's fp64 peak.  A function at a third of peak is not launch-bound, so batching its sets is worth the difference between a 214x214x4096 GEMM and a large one -- not what batching bought in B5.  `docs/HESSIAN_1.8.1.md` |
| **B5** the XC Hessian's Python atom loop | **done** (GGA) | `fastxchess` replaces `_get_vxc_deriv1` and `_get_vxc_deriv2`: the two terms together **7.34x** on penicillin V and **5.30x** on taxol at fixed density, which is 18.2 % of a 42-atom Hessian turned into 4.1 % and takes the whole run 84.26 s → 53.93 s (1.56x with the rest of Electrolites, of which `fastj` was already 1.24x).  The catalogue's prescription for it — "the same change as G1, generalized", i.e. batch the grid blocks so libxc is called once per group — is measured **not** to be where the time is: `eval_xc_eff` is 1.5 % of `_get_vxc_deriv1`, against 42 % of the XC gradient where `fastxcgrad` found it.  The atom loop is 90 % on penicillin V and 93 % on taxol, and what makes it expensive is that `d rho / d R_A` is supported only where atom A's basis functions are, so most of it is arithmetic on exact zeros.  `docs/README_fastxchess.md`.  LDA and meta-GGA are declined, both for reasons found in upstream — see §5 and §6 below |
| **B6** `hess_nuc_elec` | **upstream**: gone in 1.8.1 | `hessian/rhf.py:200 _hess_nuc_without_ecp` replaced the per-charge loop, and the `# TODO: It is computing one charge at a time` comment with it.  It does not appear in either 1.8.1 profile in `docs/HESSIAN_1.8.1.md`.  The catalogue ranks it 10th at 4.5 % of a Hessian; there is nothing left to do |
| **B7** `hess_nuc_elec_ecp` | **open** | — |
| **B8** Hessian 1e integrals on the CPU | **open** | Catalogue flags this as an Amdahl tail (0.1 % at 42 atoms) listed for re-measurement, not for fixing |
| **B9** grid-response kernels | **open** | `fastxcgrad` **declines** the grid-response path and hands it to GPU4PySCF (`docs/README_fastxcgrad.md`) |
| **B10** third and higher derivatives | **n/a** | Greenfield, no incumbent |
| **B11** DF-Hessian auxiliary integrals on the CPU | **upstream** | The entry's four `auxmol.intor` calls are gone on 1.8.1: `df/hessian/rhf.py` takes `int2c2e` and `int2c2e_ip1` from `df/int3c2e_bdiv.py` (which forwards to `pbc/df/int2c2e.py`'s GPU kernels) and the second derivatives from `libvhf_rys.e_int2c2e_ip2` (`:1144-1185`).  Nothing for this repository to do, and the same correction A13's first row needs |

One item outside the catalogue's B list was done: **`fastgradh`** puts the
nuclear-attraction part of the one-electron derivative on the GPU.  The
catalogue measures the CPU one-electron integrals at 8–13 % of a gradient
(`docs/README_fastejk.md`'s own profile) without giving them an ID.

---

## 4. Sections C–H

| Section | Status |
|---|---|
| **C** response properties (C1 polarizability/CPKS, C2 NMR/GIAO, C3 IR/Raman, C4 EDA, C5 C6) | **C1 done, all three of its stages**: its XC half is `fastxcfxc` (2.10x on `nr_rks_fxc` inside a real B3LYP CPKS solve), its VV10 half -- 80.6 % of the run on an NLC functional -- is `fastfnlc` (1.43x), and **its J+K half, which was 71 % of a B3LYP run at 42 atoms and 82 % at 113 and which no module reached, is now `fastjk` (2.19x)**.  A whole B3LYP polarizability on penicillin V goes 16.77 s → **8.06 s (2.08x)**, against 1.15x before this round.  C3, C4, C5 inherit all three halves and are otherwise **open**; C2 (NMR/GIAO) is **open** and is one of the catalogue's four low-contamination entries.  `docs/README_fastjk.md`, `docs/RESPONSE_1.8.1.md` |
| **D** excited states (D1 Davidson, D2 `lr_eigh`, D3 third XC derivative on CPU, D4 `get_ab`, D5 TDDFT+PCM, D6 ris, D7 spin-flip, D8 AO re-evaluation) | **D1 done, both halves**: its `nr_rks_fxc_st` half is `fastxcfxc` (3.84x inside a real TDA run) and its `get_jk` half, 61 % of the run, is `fastjk` (1.33x at `nset = 5`, `hermi = 0`, which the kernel sees as `n_dm = 10`).  A five-root TDA on penicillin V goes 22.21 s → **14.26 s (1.56x)**, against 1.24x before, with the five excitation energies unchanged to nine decimals.  **D2 rejected** on a measured size, **D8 re-sized from 9.7 % to 1.2 % and rejected** -- both in `docs/RESPONSE_1.8.1.md` §5 and `docs/README_fastxcfxc.md` §5.  D3-D7 **open** |
| **E** real-space and population analysis (E1–E4) | **all open.**  E3 is greenfield and one of the catalogue's four low-contamination entries |
| **F** dispersion, solvation, fields, embedding (F1–F7) | **F2 done** in `fast1e` — the `int3c1e` kernels that reach PCM, QM/MM, ESP and CHELPG, 1.57x–1.77x per SCF cycle over 20–168 atoms.  **F3 rejected**: the matrix-free PCM D/S products it names are off the default path in 1.8.1 and `get_D_S` runs once per geometry (0.026 s of a 141 s Taxol SCF).  F1, F4–F7 **open**.  `fast1e` patches only the three `int3c1e` drivers, so the `int3c1e_ip`/`_ipip` kernels behind the PCM **gradient and Hessian** are untouched |
| **G1** extend `_grouped_block_loop` everywhere | **partial, and its premise now has two counterexamples.**  It reached `grad.rks.get_exc` (B2) and, in a different form, the SCF's `nr_rks` (`fastxc`), where P8 was measured as the largest single effect.  It has **not** reached `hessian.rks._get_vxc_deriv1/2` (B5), where the batching the entry names is **1.1 %** of the function (`docs/README_fastxchess.md` §1), and it is implemented for `nr_rks_fxc` (B4d) and measured **slower than leaving GPU4PySCF's issue order alone** -- 0.72 s against 0.22 s at `nset = 20` (`docs/README_fastxcfxc.md` §2).  P8 pays where a grid block issues a few GEMMs and then waits; it does not where the block already has `nset` independent GEMMs back to back, and there `grouped_gemm`'s host-side pointer table costs more than the batching saves.  The catalogue calls G1 "the highest-leverage single piece of infrastructure work"; on the evidence here it is the highest-leverage piece **for the gradient**, and one of three places it was prescribed |
| **G2** fusing small memory-bound kernels | **partly rejected on a measured ceiling.**  Deleting *every* `vk` `atomicAdd` — the ceiling for a block reduction, not an implementation of one — is worth **1.1 %** at 6-31G\* and **2.2 %** at def2-TZVP, so a block reduction cannot pay.  `docs/ROUND2_1.8.1.md` §3b.  This is the same Amdahl warning the catalogue attaches to G2 |
| **G3** NVRTC/ptxas compile time | **done**, and re-measured with the entry's premise corrected: the 162 s → 37 s split fires on a Pople basis and **not** on a def2 one, where every class is reached anyway.  A def2 transition-metal first run pays **186-198 s** of NVRTC, which is **86-89 % of the whole run**, against 42 s for 6-31G\*.  `electrolites.warmup` moves that out of the run and divides it over workers (**1.8x-3.0x** at eight).  `docs/PRIM_PAIR_AND_JIT.md` §4c |
| **G4** the multi-GPU layer | **open** and deliberately so: every module here asserts a single device, except `fast1e`, which keys its kernels and tables on the current device because the `int3c1e` path is not distributed upstream |
| **H1–H3** negative controls | **n/a** by construction |

### Three response-side results this repository added that the catalogue does not have

- **On an NLC functional the response property's cost is neither of the two
  stages the catalogue lists.**  C1 names `get_jk` and `nr_rks_fxc`.  For
  wB97M-V they are 14.1 % and 5.6 % of a polarizability, and
  `hessian.rks.nr_rks_fnlc_mo` -- the response of the VV10 term, which has no
  catalogue ID -- is **80.6 %**.  Taken in `docs/README_fastfnlc.md`:
  **1.44x-1.49x** per warm call and 1.43x over a whole solve, of which the
  exact part is that the ground-state half of
  the function, including an `O(ngrids^2)` double sum, was being recomputed on
  every one of the 19 CPKS iterations.  What is left is one more
  `O(ngrids^2)` fp64 kernel that only exists compiled.
- **The same term dominates a wB97M-V analytic Hessian, which no profile in
  `docs/` or in the catalogue could see.**  `benchmarks/hessian_anatomy.py
  --cphf-only` on alanine dipeptide with wB97M-V: `nr_rks_fnlc_mo` is
  **83.3 %** of the CPHF solve (112.88 s of 135.56 s) and
  `_get_vnlc_deriv1` is 23.1 % of `make_h1`, against 11.9 % for the
  multi-density J and K build that `docs/HESSIAN_1.8.1.md` -- a B3LYP profile
  -- ranks first.  Catalogue **A6** lists the VV10 Hessian as an entry with no
  share attached; this is the share, and it is the largest one in any Hessian
  profile taken here.  `docs/README_fastfnlc.md` §4b.
- **The split between the grid half and the two-electron half is a property of
  the functional, not of the response solver.**  The catalogue profiles B3LYP
  and ranks from that.  Measured on the same molecule: `nr_rks_fxc` is
  **46-48 %** of a PBE response property and **47-48 %** of an M06-2X one
  against 25-28 % for B3LYP, and on a pure functional the two-electron half
  goes through `mf.get_j` -- the McMurchie-Davidson engine `fastj` patches --
  rather than the fused Rys kernel of hazard 5.
  `docs/RESPONSE_1.8.1.md` §2b.

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
| Grouping the grid blocks in the XC **Hessian**, which is what the catalogue asks for in B5 | `eval_xc_eff(deriv=2)` on all 519 680 penicillin V grid points in one call is **26.7 ms**; the same points in 127 calls is **119.3 ms**.  The whole of the gain the catalogue names is 93 ms of an 8.2 s function — 1.1 % | `docs/README_fastxchess.md` §1 |
| Sub-dividing the 4096-point grid block to sharpen the XC Hessian's atom screen | at 64-point chunks `natm_local/natm` falls only from 0.745 to 0.683 (taxol 0.626 → 0.537), because GPU4PySCF's AO mask cutoff of 1e-10 keeps a Gaussian tail alive far past where it matters.  9–17 % for a per-chunk atom list and a gather | `docs/README_fastxchess.md` §4 |
| Grouping the grid blocks in the XC **response** kernel, which is what the catalogue asks for in B4d and G1 | **0.72 s against 0.22 s** at `nset = 20` on penicillin V — slower than leaving GPU4PySCF's one-block-at-a-time issue order alone, because the response path already has `nset` independent GEMMs in flight per block and `grouped_gemm` pays a host-side pointer table per `(block, set)` pair.  Kept in the module as `FASTXCFXC_MODE=grouped` so the comparison stays runnable | `docs/README_fastxcfxc.md` §2 |
| Caching the AOs across response iterations (catalogue D8 route a) | the call count the entry rests on is exact and reproduces — 2921 `eval_ao` calls in a polarizability — but they are **2.2 %** of the run on 1.8.1, not 16.3 %.  A cache would be 4.2 GB on a 42-atom molecule to save 0.35 s of 16 s | `docs/RESPONSE_1.8.1.md` §3.1 |
| Preallocating `lr_eigh`'s trial space (catalogue D2) | the reallocation is real and still there on 1.8.1, and it is **0.19 GB over a whole 22 s TDA run** at 42 atoms, 1.3 GB at 113 — under a millisecond either way.  The entry's premise needs `nov ~ 10⁶` | `docs/RESPONSE_1.8.1.md` §5 |
| Splitting the response path's fused J+K into `j_engine.get_j` + `jk.get_k` | **0.70x–0.93x** on stock code at every `nset`: J is nearly free inside the fused Rys kernel (0.52 s fused against 0.51 s for K alone).  With Electrolites patched the single-density row is **2.30x** the other way, which is what makes a *fused* multi-density rewrite the target rather than a split | `docs/RESPONSE_1.8.1.md` §4 |

The AO screen's headroom is **not** zero: `docs/ROUND2_1.8.1.md`'s "What is
left" item 1 keeps it open as a per-(block, shell) log-magnitude estimate moved
*upstream* of `eval_ao`, which is a different change from the one that measured
0.28x.  That is catalogue **A5**(b) by another route.

---

## 6. Measurement hazards found here, which constrain any future entry

All are the same kind of finding as catalogue §3.7's warning: reasons a
number in a table can be wrong while every correctness test passes.

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

4. **A Hessian entry's share is not a molecule-independent number, and the
   catalogue's §5 ranking of the Hessian entries is built from one 42-atom
   profile.**  Between 42 and 113 atoms the exchange build inside CPHF (B4b)
   goes from 24.0 % to **47.3 %** of the wall clock while the XC Hessian (B5)
   goes from 18.2 % to 7.7 % and `ejk_ip2` (B3) from 8.8 % to 2.6 %.  The
   catalogue ranks B5 seventh and B4b eighth; at 113 atoms the order is the
   other way round by a factor of six.  Catalogue §3.4's size-independence
   rule applies to the choice of *target*, not only to the speedup.
   `docs/HESSIAN_1.8.1.md` §4.
5. **A patch target is not a code path.**  `fastj` patches
   `j_engine._VHFOpt.get_j` and `fastk` patches `jk._VHFOpt.get_k`, which is
   where the *SCF* builds J and K.  The response path calls `mf.get_jk`, which
   lands in `jk._VHFOpt.get_jk` — a **fused** Rys kernel that builds both from
   one integral pass — so neither module was on it, and a polarizability or a
   TDDFT run was stock GPU4PySCF in 61–82 % of its wall clock even with
   `patch_all()`.  Nothing failed and no test noticed; the modules simply were
   not called.  Before an entry's Amdahl share is claimed as reached, the
   entry's *caller* has to be checked, not only the function name.
   `docs/RESPONSE_1.8.1.md` §3.2.  Closed by `fastjk`, which patches that third
   path; the hazard is recorded because the *reason* it went unnoticed for two
   rounds — every test passed and every module was silent — has not changed.
   `benchmarks/ndm_screen_probe.py` walked into it again from a third
   direction: its CPKS and TDA stages return no rows at all, because they
   instrument `get_k` and those paths never call it.

8. **A speedup that falls with `n_dm` is not an implementation that degrades.**
   Fitting the exchange build as `T = R + n_dm·C` and solving for both halves
   separately (`benchmarks/ndm_k.py`) shows `C` identical between GPU4PySCF's
   kernels and these (1.01x at 42 atoms, 1.16x at 113): the P1–P5 treatment
   applies to integrals and not to the contraction.  A ratio quoted at one
   `n_dm` therefore says almost nothing about another — 3.28x at `n_dm = 1`,
   1.18x at 64, on the same kernels and the same molecule — and an entry in
   this family has to be quoted with the `n_dm` its caller actually uses.
   `docs/ROUND4_B4B.md` §4.
6. **GPU4PySCF's meta-GGA `_get_vxc_deriv1` is not reproducible run to run.**
   Called twice on the same input it differs by **3.0e-5**, and the resulting
   Hessian by 1.7e-5; every LDA and GGA case gives exactly zero.  That is a
   different statement from `hessian/rks.py`'s own warning that the MGGA
   Hessian is grid-sensitive — this is at a fixed grid.  Any meta-GGA Hessian
   work needs this understood first, because until then there is no reference
   to converge against.  `tests/test_xchess.py`'s `self` column.
7. **A factor 2 that looks missing and is not.**  GPU4PySCF's LDA and GGA
   branches of `_get_vxc_deriv1` apply different factors to `d rho / d R` —
   the GGA branch has `_make_dR_rho1`'s `rho1 *= 2`, the LDA branch says "*2
   is not applied because + c.c. in the end" — and the two then do the same
   two contractions.  They still agree, because the GGA branch also carries a
   compensating `wv2[:,0] *= .5` on the density component only.  Recorded
   because it cost an hour to chase and because it is the reason LDA is not a
   special case of the GGA code with `ncomp = 1`: the factor conventions
   differ term by term.  `docs/README_fastxchess.md` §5 has the algebra and
   `tests/test_xchess_fd.py` an LDA row that checks it.

Catalogue §3.4's size-independence rule — re-measure on a second,
differently-sized system and count the result only if the ratios agree — is
followed throughout `docs/`, and every table there gives both systems.

---

## 7. What the next round should take

The `fastjk` round moved the top of this list: it is no longer a catalogue
entry.

1. **`C`, the multi-density contraction** — not an entry, and now the largest
   measured thing between this repository and a fast Hessian.  It is 94 % of an
   `n_dm = 126` exchange build, it is identical code in both implementations,
   and `get_k` is 32 % of a 42-atom Hessian after the round and over half of a
   113-atom one.  Two candidates, in the order the evidence supports:
   **per-density screening**, which §5's own probe bounds at 1.14x–1.18x and
   which grows with system size, and the **`atomicAdd` traffic**, which is now
   the only unexamined half of catalogue Q4 — the other half, the density
   layout, was measured this round and **rejected**: transposing to
   `dm[(a*nao+b)*n_dm + i_dm]` is 0.30x–0.38x, three times worse, because it
   trades a stride between densities for a stride between quartets and there
   are four orders of magnitude more quartets.  `docs/ROUND4_B4B.md` §4 and §7.
2. **A5** (rank 3, 18–21 % of SCF) — the kernel itself, now that the
   repetition is gone.
3. **A4**'s SCF mGGA path (rank 5), which first needs catalogue **Q13**
   measured: is the SCF mGGA build launch-bound at small `nao_sub`?  This
   repository's own mGGA gradient result (4.46x on M06-2X) is a reason to
   expect yes, but it is not that measurement.
4. ~~**B4d**, the response half of rank 12~~ — taken in
   `docs/README_fastxcfxc.md`, with the entry's own prescription (G1's grouped
   GEMM) measured **slower** than leaving the issue order alone, and with
   **D8, its other half, re-sized from 16.3 % to 2.2 % and rejected**.  What
   pays there is batching along `nset`, which is a stride rather than a list
   of pointers.  ~~**B5** (rank 7)~~ — taken in `docs/README_fastxchess.md`, and, like F2
   before it, with a correction to the entry: the mechanism the catalogue
   names for B5 (batch the grid blocks, as G1 does) is 1.1 % of the function,
   while the mechanism that carries it (most of the atom loop is arithmetic
   on exact zeros, because `d rho / d R_A` lives only where atom A's basis
   does) is not in the entry at all.  Rank 6, **F2**, was taken in
   `docs/README_fast1e.md`, and what came back was a correction to that entry
   as well as a 1.71x: the mechanism is not the missing screen but the far
   field being evaluated with the near-field formula — 92 % of (primitive
   pair, cavity point) tasks — while a rigorous error-budget screen in the P11
   style turns out to be worth only 21 % of tasks, and those come off the
   branch the patch already made cheap.
5. ~~**B4b**~~ — taken in `docs/README_fastjk.md` and `docs/ROUND4_B4B.md`,
   and it was the largest single result here so far: a whole B3LYP
   polarizability 1.15x → **2.08x**, a five-root TDA 1.24x → **1.56x**, the
   Hessian's `get_k` row 1.16x.  Three things that round settled, each of which
   changed what should be done rather than only how fast something is:

   * **It is two entry points.**  `jk._VHFOpt.get_k` at `n_dm = 3·natm` for the
     Hessian, and the *fused* `jk._VHFOpt.get_jk` for every response property.
     The second was a third code path no module was on (hazard 5 in §6).
   * **Fused beats split at every size, now that the K kernels take
     `n_dm > 1`.**  The previous round had refused to choose between the two
     arrangements because the sign flipped between molecules; a fused rewrite
     removes the choice and exceeds the 2.30x ceiling that probe measured.
   * **The entry's mechanism has a floor, and it is arithmetic.**  `T = R +
     n_dm·C` with `C` identical in both codes (1.01x): the whole family's
     asymptote in `n_dm` is 1.01x–1.16x, so `C` — not the integrals — is what
     the next round has to attack.  That is now item 1 below.

6. ~~**B3** (rank 9)~~ — taken in `docs/README_fastejk2.md`, and, like F2 and
   B5 before it, with a correction to the entry; here two corrections and a
   correctness finding.  (a) The entry's "same shape as B1, so the P1–P5
   treatment transfers" covers 37 % of the stage: GPU4PySCF unrolls **seven**
   ip2 classes against twenty-five for the exchange build, so at def2-SVP 63 %
   of `ejk_ip2` runs on a *general* kernel.  (b) Applied to that 37 %, **P1 is
   1.02x and P3 is 0.58x** — the register cap the entry singles out as worth
   *more* here than on K is a 1.75x regression, because these kernels want all
   255 registers and a cap buys occupancy with spills.  What pays is the block
   shape, which the entry does not mention: **1.13x** for two lines of Python.
   (c) The correctness finding: the same register cap written as
   `__launch_bounds__(512, 1)` rather than `(256, 2)` — same cap, same launch
   geometry — makes these kernels compute numbers 9x–37x wrong, for a reason
   not identified.  What is left of the entry is the general kernel, a
   `gen_khigh`-sized generator project for a share that falls from 13.9 % of a
   42-atom Hessian to 3.0 % of a 113-atom one.

7. **`VXC_vv10nlc_hess_eval_f_t`** -- not a catalogue entry, and the largest
   single kernel left on any response path measured here: **64.7 %** of the
   VV10 response term, which is 80 % of a wB97M-V polarizability.  It is an
   `O(ngrids^2)` fp64 double sum and `fastnlc`'s three measured techniques
   (a MUFU-seeded reciprocal, single precision beyond 10 Bohr, an
   error-budgeted grid trim -- 3.92x together on the ground-state sum) apply
   to it.  The obstacle is that it exists only inside `libgdft.so`, so it
   needs either GPU4PySCF's CUDA source or a re-derivation from the paper its
   docstring cites, with the `.so` as the oracle.
   `docs/README_fastfnlc.md` §4.

For the LLM-GPU-QM task suite rather than for throughput, the catalogue's own
advice stands unchanged: **A12, F1, C2, E3** are worth more than ranks 1–13
because they are the least contaminated, and all four are open.
