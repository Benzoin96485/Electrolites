# The density-fitting path: what is done, what was rejected, and what is left

A work list for whoever picks this up next.  It is written to be read without
the session that produced it: every number in it is one measured in this
repository, with the command that produces it again, and every dead end is
recorded with the measurement that killed it so nobody spends a day
rediscovering it.

Companion documents: `docs/README_fastdf.md` is the round that produced this
list (what changed, why, and the full tables); `docs/README_fastdfj.md` is the
round that closed **W1**, which is where the biggest number on this page now
is; `docs/README_fastdfgrad.md` closed **W3**, `docs/README_fastdfcderi.md`
**W4**, `docs/README_fastdfblock.md` **W5**, `docs/README_fastdffp32.md`
**W7**, `docs/README_fastdfhess.md` **W8** (whose result is that its own
target cannot be made faster, and whose *finding* is W16, W17 and W18),
`docs/README_fastdfj3c.md` **W9** (whose findings are W19 and W20),
`docs/README_fastdfstore.md` closed **W11**, **W12** and **W13** together
(they are three items on one function, and its finding is W22),
and **W6** took two rounds --
`docs/README_dflocal.md` is the two measurements it asked for first and
`docs/README_dflocal_k.md` the implementation and the cost estimate;
`docs/CATALOGUE_06_STATUS.md` §2 row **A13** is where this sits in
`instructions/06`'s catalogue.

---

## 0. Preconditions

* GPU4PySCF **1.8.1**.  Several of the catalogue's claims about this path are
  1.7.0 claims that 1.8.x closed — check the source before trusting any line
  number in `instructions/06`.  `docs/README_fastdf.md` §1 has the four that
  were audited.
* One A100 per process.  Everything in `electrolites/` asserts a single
  device; §3 W2 is about that being a limitation of *this* repository rather
  than of GPU4PySCF.
* Measurements go on a **whole node** (`--qos=debug` on Perlmutter, 29 min,
  4 GPUs requested and one used).  §5 says why the login node is unusable for
  this and what else will silently poison a number.

---

## 1. What is already done — do not redo

`electrolites/fastdf.py` replaces `df/df_jk.py:get_jk`, which `df/df.py:117`
is the only caller of.  That one target covers the DF SCF, the DF response
solve and a DF Hessian's CPHF at once.  Four changes, all measured:

1. **The exchange contraction as `dsyrk`** above `SYRK_MIN_NAO = 800` basis
   functions.  1.11x on the build at 1007 functions, 1.18x at two densities
   there, 1.41x at two densities on taxol.  Below the threshold `dsyrk` is
   *slower* than the GEMM it replaces despite half the arithmetic (0.96x at
   430), which is why the threshold exists.
2. **The first contraction left alone** — see §2, this is a result.
3. **The Coulomb pass as an explicit GEMV/GEMM pair** rather than two
   `contract` calls.  1.09x on the build at one density, nothing at two.
4. **A host-resident cderi only partly streamed**: as many auxiliary rows kept
   on the device as free memory allows, the rest streamed with GPU4PySCF's
   double-buffered prefetch.  1.64x–4.53x on the build, and the payoff is
   close to linear in the fraction that fits (25/50/75/100 % resident =
   1.21x/1.55x/2.66x/4.51x on a 5.5 GB tensor).

Separately, `electrolites/fastxcnlc.py` now patches **`df_jk._DFHF.get_veff`**
as well as the two `RKS`/`UKS` ones.  `density_fit()` builds its class as
`set_class(mf, (_DFHF, cls))`, so `_DFHF.get_veff` precedes both in the method
resolution order and that module was **inert on every density-fitted run**.
A DF wB97M-V SCF on penicillin V: 21.99 s stock → 16.02 s with `patch_all()`
as it stood before (that part is `fastnlc`, which was always reachable) →
**14.19 s** with the new binding.  `nr_nlc_vxc` is 77–80 % of such a run.

Correctness bar for anything added here: `tests/test_df.py` compares
element-wise against GPU4PySCF in one process on one cderi tensor and reaches
**5.4e-16 (J) / 3.6e-15 (K)** relative.  Do not accept a change that needs a
looser one without saying why in the same commit.

---

## 2. What was measured and rejected — do not redo these either

| tried | measured | why |
|---|---|---|
| the first contraction as a cuBLAS batch over the **AO** index | **1.00x** at 430 functions, **slower** at 1007 | `nao` batches of a `(nocc × nL)` output is the small-matrix regime cuBLAS's batched path is weakest in; cuTENSOR sees one large contraction and runs it at **15.8 TF/s** of an A100's 19.5 |
| an `(L,i,j)` unpack kernel of our own (CUDA, written and correct) | **2.65x slower** than GPU4PySCF's — 148 GB/s against 393 | the transposed half of a symmetric fill is coalesced in GPU4PySCF's `(i,j,L)` layout and is not in that one |
| a blocked triangular GEMM in place of `dsyrk` | 0.95x of the full GEMM at `n = 1123`; only wins at `n = 2000` | — |
| raising/lowering the auxiliary block size | 256 best on three systems of four, 128 6 % better on the fourth | GPU4PySCF's own cap stands |

Both of the first two are still runnable (`FASTDF_STEP1=bgemm`, and the kernel
is in the git history) — a rejected alternative that can be re-run is worth
more than one described in prose.

**The headline negative result.**  The DF build on 1.8.1 is already near the
machine stage by stage: the unpack moves 1.93 GB per block at **1.43 TB/s**
(71 % of the card), and the two contractions run at **15.8** and **17.8 TF/s**
against an fp64 peak of 19.5.  That is the opposite of what every direct-SCF
round in this repository found, and it is why this path is worth 1.1x from
rearrangement and not 3x.  **Anything past that has to make the arithmetic or
the storage smaller, not tidier.**  §3 is that list.

---

## 3. Where the time and the memory actually are

### 3.1 Time

`benchmarks/df_anatomy.py`, penicillin V, DF-B3LYP, level-3 grids, 14 cycles,
unpatched, cderi on the device:

| stage | def2-SVP (nao 430) | def2-TZVPP (nao 1007) |
|---|---|---|
| `with_df.get_jk`, of which | 27.2 % | 42.0 % |
| ↳ the **one-time** cderi build | 12.4 % | 15.5 % |
| ↳ the 14 recurring builds | 14.8 % | 26.5 % |
| `nr_rks` (XC — `fastxc` already covers it) | 34.6 % | 28.1 % |
| `get_hcore` (one-time, CPU) | 21.8 % | 13.9 % |

With the cderi on the **host** the build's share roughly doubles (43.9 %, of
which 22.8 % is the one-time build).  On **taxol/def2-SVP**, host-resident,
15 cycles, `_cholesky_eri` alone is **39.4 %** of the SCF.

The **gradient** inverts with basis size: `df/grad/rhf.py:_jk_energy_per_atom`
is **29 %** of a def2-SVP DF gradient and **83 %** of a def2-TZVPP one.

> **Corrected upward since.**  Those two came from `df_anatomy.py`, which
> wrapped the module-level function and so missed the `with_df.reset()` and
> the `SortedMole` + `Int3c2eOpt` rebuild the *method* does on every call.
> Timed at the method — where the work actually starts —
> `benchmarks/df_grad_probe.py --whole` puts it at **72 %** of a def2-SVP
> gradient, **93 %** of a def2-TZVPP one and **97 %** of a 113-atom
> def2-SVP one.  `df_anatomy.py` now wraps the method.

### 3.2 Memory

`benchmarks/df_fill_probe.py` (no GPU time; the screening is a property of the
geometry and the two basis sets).  def2-SVP with its own jkfit set:

| | atoms | nao | naux | stored pairs | fill | cderi |
|---|---|---|---|---|---|---|
| glycine | 10 | 95 | 471 | 4 623 | 101.4 % | 0.02 GB |
| alanine dipeptide | 22 | 200 | 974 | 18 148 | 90.3 % | 0.14 GB |
| Ni dithiolene | 21 | 327 | 1 992 | 36 704 | 68.4 % | 0.58 GB |
| penicillin V | 42 | 430 | 2 191 | 60 845 | 65.7 % | 1.07 GB |
| taxol | 113 | 1 123 | 5 598 | 319 589 | 50.6 % | 14.3 GB |
| PfPMT | 284 | 2 697 | 13 365 | 808 156 | **22.2 %** | **86.4 GB** |

42 → 284 atoms is 6.8x in atom count and **81x** in tensor — about
O(N^2.5): the screening holds `npair` to roughly N^1.5 but `naux` is linear
and the two multiply.

### 3.3 The 284-atom case actually runs, and that reorders everything

`_cholesky_eri`'s host branch calls `empty_mapped` **without consulting
`max_memory`**, so the ceiling is the node's DRAM and not the 80 GB the
benchmarks assume.  Measured, PfPMT/def2-SVP on one A100-SXM4-40GB in a node
with 251 GB of DRAM (`benchmarks/df_probe.py --xyz .../PfPMT.xyz --charge -1
--basis def2-svp --cycles 1 --reps 1`):

| | |
|---|---|
| cderi | **86.41 GB, host-resident**, `blksize` 128, 105 blocks |
| `get_jk` J+K, one call | **12.52 s** |
| ↳ K only | 12.38 s |
| ↳ J only | 3.28 s |
| `DF.loop` alone, `unpack=False` | 3.27 s |
| `DF.loop` alone, `unpack=True`, blk 32 / 64 | 3.28 s / 3.35 s |
| step 1 per block (`contract('Lij,njk->nikL')`) | 55.9 ms |
| step 2 per block, GEMM / `dsyrk` | 59.3 ms / **32.9 ms** |
| step 1 per block, cuBLAS batch over `i` | 62.2 ms |

Four things follow, and three of them contradict what the smaller systems
suggested.

**It is not a capacity wall at this size.**  A 284-atom density-fitted
calculation builds and runs.  *(At the next size up it is: `HcgC`/def2-SVP,
545 atoms, has 2 094 539 stored pairs and 24 722 auxiliary functions, so its
global cderi is **414 GB** against the node's 251 — measured in
`docs/README_dflocal.md` §1, and the reason W6 is no longer demoted.)*  GPU4PySCF's README limit ("~168 atoms with
def2-tzvpd, bounded by CPU memory") is about a smaller-memory configuration;
with 251 GB of host DRAM the binding constraint here is *speed*.

**The PCIe stream is at hardware speed and is fully hidden behind the exchange
contractions.**  86.41 GB in 3.27 s is **26.4 GB/s** — a saturated gen4 x16
link — and `unpack=True` costs the same as `unpack=False` to within 2 %, so
the unpack is free relative to the transfer.  More importantly
`get_jk(J+K) = 12.52 s ≈ K only = 12.38 s`, not `12.1 + 3.27`: at `nocc = 547`
the contractions are large enough that the double-buffered prefetch hides the
whole tensor behind them.  **So `fastdf`'s item 4 buys nothing for J+K here.**
Its 1.64x–4.53x was measured where the stream is *exposed* — smaller systems,
where the contractions are too short to hide it.  A J-**only** call is the
opposite: 3.28 s, exactly the transfer, entirely PCIe-bound.

**The two contractions are the whole cost**: 105 × (55.9 + 59.3) ms = 12.1 s
of the 12.38 s K build.  `dsyrk` is **1.80x** on step 2 at this size (32.9 ms
against 59.3), which is `fastdf`'s largest single margin anywhere and puts the
K build at about **1.29x**.  The cuBLAS batch over `i` loses again (62.2
against 55.9), a fourth system agreeing with §2.

**And DF is losing to the direct path at this size.**  Fifteen cycles of a
12.5 s build is ~190 s in the two-electron half alone, before `_cholesky_eri`.
This repository's direct-SCF path runs the *whole* SCF of the same molecule
(6-31G\*, nao 2268 against 2697) in **55.1 s** patched.  The bases differ, but
not by enough to explain the gap.  That is the strongest single argument on
this page and it is what re-ranked §4.

> **Measured since, in the same basis.**  W1 ran all three to convergence on
> this molecule in def2-SVP, 18 cycles each: full DF **300.5 s**, the direct
> path **65.4 s**, and RIJONX — density-fitted J, exact K — **64.1 s**.  The
> gap this paragraph inferred is 4.6x and real, and 88.7 s of it is
> `_cholesky_eri` alone.  `docs/README_fastdfj.md` §3.

---

---

## 4. The work list

Ranked by (value × confidence) / cost, **after** §3.3.  Each item gives what
to change, the evidence it rests on, what would count as success, and what
would falsify it.

**W4 is done, and W11, W12 and W13 are what it found.**  The cderi build is
1.14x–1.83x faster with the tensor on the device and **2.72x–3.20x** with it
on the host (`docs/README_fastdfcderi.md`), and the item's headline lead —
the metric contraction as a sparse gather and a triangular *solve* — is
rejected by the falsification criterion the item itself attached, then
rescued by a triangular *multiply*.  What the round found rather than closed
is that the single largest stage of a large build is neither integrals nor
arithmetic: `cudaHostAlloc`, which page-locks the host buffer at 2.09 GB/s
and is 41.5 s of a 72.5 s 284-atom build.  Made 4.1x faster here; the
remaining 10 s, the host store and `_decompose_j2c`'s latency are **W12**,
**W11** and **W13**.

**W8 is done, it is a capacity item rather than a speed one, and W16, W17
and W18 are what it found.**  `df/hessian/rhf.py:_get_jk` — the fifth
two-electron code path — turns out to be **78 % of an A100's fp64 peak**
already and **15 % of the Hessian it sits in**, and three rearrangements of
it were written and measured at 0.90x–1.00x.  What the round is for is that
the function **does not run at all** past a certain shape: cuTENSOR refuses
the plan for its dominant contraction with tens of GB free, so a 42-atom
def2-TZVPP DF Hessian and a 113-atom def2-SVP one cannot be run by GPU4PySCF
and can be by `electrolites/fastdfhess.py`.  The round also produced the
first anatomy of a density-fitted Hessian in this repository, and that
anatomy says the time is in three functions no entry on this page named
(`docs/README_fastdfhess.md`).

**W9 is done, and W19 and W20 are what it found.**  The three-centre Coulomb
build a *pure* density-fitted functional runs on is **2.32x** at 10 atoms,
1.78x at 42, 1.13x at 113 and 1.03x at 284 at one density -- and **4.45x** at
two densities, **2.55x-2.73x** from four to thirty-two, and **13.2x** on a
21-atom open-shell 3d complex, where def2-SVP reaches an f shell on the
auxiliary set and `df_jk.get_j`'s engine choice for `n_dm > 1` is worst.  A
whole DF-PBE SCF is **1.16x** at 42 atoms against a `patch_all()` baseline.
**None of it is arithmetic**: every kernel launched is GPU4PySCF's own with
GPU4PySCF's own arguments, and `tests/test_dfj3c.py` compares element-wise at
**1.6e-16**.  The two things that carry it are that the call re-uploads about
thirty host arrays per Coulomb matrix, and that `Int3c2eOpt.build` gives the
auxiliary-vector kernel **372 thread blocks on a 108-SM card** with the
angular-momentum groups differing in size by seventy times.  What the round
found rather than closed is that after it the build is **85 % two integral
kernels at 42 atoms and 97 % at 284** (`docs/README_fastdfj3c.md`), and the
three things still inside them are **W19**; the fourth, which is not inside
them, is **W20**.

**W3 is done, and W10 is what it found.**  The DF gradient's two-electron
derivative is 1.75x–2.59x faster and the gradient around it 1.24x–2.27x
(`docs/README_fastdfgrad.md`), from two things the entry did not predict —
integrals the SCF's cderi already holds, and a launch constant.  What it could
*not* do is run at 284 atoms: `j3c_oo` is a dense `(naux, nocc, nocc)` array,
32.0 GB there, and neither upstream nor this repository batches it.  That is
**W10**.

**W10 is done, and W21 is what it found.**  The array is 37.7 GB rather than
32.0 — the allocation is on the Cartesian sorted auxiliary dimension, not the
spherical one — and GPU4PySCF holds **two** of them.  Neither is necessary:
the second is never allocated, because the only step that needs the whole
auxiliary index at once is the metric multiply and that one slices the
*occupied* index and runs in place; and the first goes to page-locked host
memory when it still will not fit.  The **284-atom DF-B3LYP gradient now
runs**, 68.8 s whole and 64.5 s for the two-electron derivative that is 96 %
of it, where GPU4PySCF cannot run it at all.  What the item priced in —
extra passes over the cderi — is not needed at all, because the array was
already built a block at a time and only ever *held* whole; slicing is free
(0.991x at eight slices) and what costs is the host store (1.087x at 42
atoms, 1.294x at 113, and exactly four PCIe traversals plus one
page-locking).  `docs/README_fastdfgradoo.md`.  Its finding is that freeing
the array makes the ip1 loop's block bigger and re-opens the cuTENSOR plan
wall **W8** hit from the other side, which is **W21**.

**W6 is done, in two rounds, and the second contradicts the first's
headline.**  Robust (Dunlap) fitting is mandatory — non-robust pair-atomic
fitting collapses variationally by **16.5 Eh** on butane — and the
coefficients fall from **N^2.52 to N^1.53** (`docs/README_dflocal.md`).  The
implementation is `electrolites/dflocal.py` and it says three things the first
round could not.  Pair-atomic exchange **does** factorise: it is one pass
against `fastdf`'s two, and what changes is `fastdf`'s *second* contraction,
not its first.  Its cost is `2 nocc / nlambda` with `nlambda` a quantity that
*grows* — 1.09x at 284 atoms unscreened — until the fitting coefficients are
screened, after which `nlambda` is flat with size and the ratio is **2.93x at
284 atoms and 4.35x at 545**.  And the storage number is about the *fit*: the
robust exchange build touches the same 86.4 GB global DF does, unless the fit
residual is stored instead, which is measured (**31.5x at 113 atoms**) and not
implemented.  `docs/README_dflocal_k.md`.

**W7 is done, and what it cost was not the cast.**  The cderi in single
precision is exactly half the bytes and exactly half the PCIe, the
arithmetic stays in double, and the switch is off by default
(`docs/README_fastdffp32.md`).  What the item priced as "technically the
least work on this list" took three patches, because a narrow tensor has
**no fall-through**: `fill_symmetric` and `transpose_write` take a `void*`
and are compiled for `double`, so every path that reaches the tensor either
knows about the mode or reads past the end of it and returns a plausible
wrong answer.  The error is the **Coulomb** energy's rather than the
exchange matrix's the item asks about, it is flat at **3e-8 – 1e-7 Eh per
atom** for organics and **1.3e-6 – 2.6e-6** on a 3d transition metal — where
for FeCl4(-) it is the same size as the fitting error itself — and what the
mode is really for is neither: at 284 atoms the tensor stops being 86.4 GB
on the host and becomes **43.2 GB on the card**, and at 545 atoms a 414 GB
tensor that does not fit in a 251 GB node becomes a 207 GB one that builds
in 129 s.

**W5 is done, and W14 and W15 are what it found.**  The block-sparse exchange
contraction is **1.21x on a whole 284-atom `get_jk` call** and ships on behind
a measured tile-fill gate.  It read 0.90x until Nsight Systems found a 763 MB
temporary allocated per block whose `cudaFree` drained the prefetch pipeline
(`docs/README_fastdfblock.md` §4.1).  The 4.5x the item priced was never
available — the pair fill it quoted is the product of an atom-pair factor a
tile can address and a shell-structure factor no AO tile can, and the tile
fill is 69.4 % at 284 atoms — and its falsification criterion fires at 113
atoms exactly as written.

The ordering before the 284-atom run put the capacity items first, because the
size ladder in §3.2 looked like a wall.  It is not one on this hardware, so
the list now leads with the three that attack the 12.5 s per call §3.3
measures: **W1** removes the cderi from the hybrid path entirely, **W2** puts
it on four devices at once, **W5** stops the contraction doing 4.5x the
arithmetic it needs to.  W6, the only asymptotic fix, dropped from second to
sixth for the same reason and will come back at 500+ atoms.

> **W1 is done, and it changes what the rest of this list is for.**  A
> 284-atom hybrid no longer has to build a cderi: RIJONX runs the whole SCF in
> **64.1 s** against full DF's **300.5 s**, and is level with the direct path.
> So W2, W5, W6 and W7 are no longer the way to make a *large hybrid* fast —
> they are the way to make the **full-DF** build fast, which is still the only
> option when the fitted exchange is wanted (it is the more accurate of the
> two approximations, `docs/README_fastdfj.md` §5) and is still what every
> *pure* functional, every gradient and every Hessian on this path uses.  W3
> and W4 were untouched by W1 — **both are now done**, and W4 inherited a
> measurement from W3 (the shell-pair aggregation, which is worth 1.19x–1.50x
> on a whole build).  W9 is worth *less*: §3.1 of
> `docs/README_fastdfj.md` measures the three-centre Coulomb engine at 1.67x
> the direct four-centre build at 284 atoms, so the path it names is already
> the faster of the two.  *(**W9 is now done and that ranking was half
> right.**  The engine is good and nothing in that round changed its
> arithmetic; what nobody had done was time the **stages** of the call around
> it, and at ten atoms three quarters of a Coulomb build is not integrals at
> all.  1.03x-2.32x at one density, 1.5x-4.5x at more than one, and 13.2x on
> an open-shell 3d complex where GPU4PySCF picks the wrong engine
> altogether.)*

### W1 — `only_dfj` fixed, and RIJONX measured — **done**

`electrolites/fastdfj.py`, `tests/test_dfj.py`, `benchmarks/dfj_probe.py`,
write-up in `docs/README_fastdfj.md`.  It was the cheapest item here and it is
the largest result on the page.

**The fix.**  `_DFHF.get_jk` passed `mol` where `DF.get_jk` expects the
density (`df/df_jk.py:213`), so every `only_dfj=True` run died with `'Mole'
object has no attribute 'shape'`.  One line, copied from the GHF branch forty
lines above that makes the same call correctly.  `FASTDFJ_OFF=1` puts the
failure back.

**What it is worth**, PfPMT (284 atoms, def2-SVP, B3LYP, one A100-SXM4-40GB,
`patch_all()`):

| | full DF | **RIJONX** | direct |
|---|---|---|---|
| `get_jk(J+K)`, per call | 10.64 s | **4.01 s** | 4.51 s |
| ↳ J | 3.19 s | **0.74 s** | 1.23 s |
| ↳ K | 10.51 s | **3.28 s** | 3.28 s |
| one-time | **88.7 s** | **1.08 s** | — |
| stored | **86.4 GB**, host | **none** | none |
| **whole SCF**, 18 cycles | **300.5 s** | **64.1 s** | 65.4 s |

* **Success criterion cleared**: the bar was a two-electron half under §3.3's
  **12.5 s**; it is **4.01 s**, i.e. **2.65x** against the same call measured
  in the same process.  End to end, **4.69x**.
* **Falsification criterion cleared by 4.3x**: the bar for the three-centre J
  engine was **3.28 s**, full DF's J-only call.  It does it in **0.736 s** —
  and beats the *direct* four-centre Coulomb build as well (1.23 s, 1.67x).
  `df/j_engine_3c2e.py` is much better than this list assumed.
* **The exchange is the direct path's to the third digit** (3.277 against
  3.278 s), as it must be — the same `scf/jk.py` call.

**And a second result the fix exposed.**  `_DFHF.get_veff` rebuilds from the
*whole* density every cycle — correct for a cderi contraction, which does not
screen, and wrong for a four-centre exchange build, which does.  With it,
RIJONX was *cheaper per call* than the direct path at 113 atoms and *slower
over a whole SCF*.  `fastdfj` hands an `only_dfj` object to its own class's
`get_veff` instead, so the exchange is built from `D - D_last`: **1.33x** at
284 atoms, **1.31x** at 113, same energy to 8.7e-11 Eh in the same number of
cycles.  That change also puts a RIJONX run back on `fastxcnlc`'s fused
XC-and-VV10 sweep, which `_DFHF.get_veff` was keeping it off.

**Where the crossing is.**  Full DF still wins below about a hundred atoms in
def2-SVP — 1.67 s against 3.46 s on penicillin V — and loses from taxol up:
22.73 against **14.98** s at 113 atoms and 300.5 against **64.1** s at 284.

**Left open**, stated in `docs/README_fastdfj.md` §6: the **gradient** of an
`only_dfj` object is still the full-DF derivative, because nothing in
`df/grad/` reads the flag, and the module warns once rather than pretending
otherwise.  Making it consistent needs the exchange term from `fastejk` and a
Coulomb term GPU4PySCF has only for the cderi route.

**And one thing this closes rather than opens.**  W9 asks about the pure-
functional Coulomb engine; the same measurement now sizes it.
`df/j_engine_3c2e.py` builds J at 284 atoms in **0.736 s** against the direct
four-centre build's 1.228 s — it is the *faster* of the two, at every size
measured.  A pure DF functional is already on it and builds no cderi (checked:
a DF-PBE run leaves `_cderi` at `None` and `fastdf` at zero calls), so W9 is
about a path that is in good shape, not one that is losing.

### W2 — Let `fastdf` run on more than one device

*(§3.3 raised this one: four devices hold 160 GB, so PfPMT's 86.4 GB goes
resident, which removes the 3.28 s transfer **and** quarters the 12.1 s of
contractions.)*

GPU4PySCF **already** distributes the cderi: `_cholesky_eri` splits the
auxiliary index across devices and `get_jk` runs `proc()` per device through
`multi_gpu.run`.  Four A100s on one node is 160 GB of device memory, which is
past PfPMT's 86.4 GB.  Every module in `electrolites/` — including `fastdf` —
declines when `multi_gpu.num_devices != 1`, so a multi-GPU DF run is correct
and entirely stock.  **That is this repository's limitation, not GPU4PySCF's.**

* **Do**: make `fastdf` per-device rather than single-device.  Its four items
  are all local to one device's slice; the block loop, the resident prefix and
  both cuBLAS calls should need the device id and nothing else.
* **Success**: `fastdf` answers calls with `num_devices > 1`, and the
  correctness suite passes with 2 and 4 visible GPUs.
* **Watch**: `_resident_prefix` caches on the `dfobj`, which is shared across
  the threads `multi_gpu.run` spawns.  Key it per device or it will hand one
  device another device's rows.

### W3 — The DF gradient's two-electron derivative — **done**

`electrolites/fastdfgrad.py`, `tests/test_dfgrad.py`,
`benchmarks/df_grad_probe.py`, `benchmarks/df_grad_bench.py`,
`benchmarks/df_grad_ip1_sweep.py`, `benchmarks/df_grad_metric_probe.py`,
write-up in `docs/README_fastdfgrad.md`.

**The profile this item asked for contradicts the paragraph that motivated
it.**  A traced clone of the function, stage by stage, puts **76–85 %** of it
in two Rys kernels — the derivative three-centre integrals at 52–59 % and the
plain ones at 24–26 % — at 42 atoms in two basis sets and at 113 atoms.  The
"two fancy-index scatters" this entry singles out are **1.0–1.5 %** of it and
run at 644–1821 GB/s against an A100's 1555.  All four contractions together
are 2.7 %/5.5 %/14.4 % and run at 5–17 TF/s, the largest of them at 71 % of
the machine.  §2's lesson held a second time, and the `(L,i,j)` note above is
answered by arithmetic rather than by an experiment: a *perfect* version of
every contraction in the function is worth under 5 % at 42 atoms.

**What the table pointed at, and what it is worth.**  Two things, neither a
rearrangement:

1. **A quarter of the function is integrals the SCF already computed.**  The
   first loop evaluates `(pq|P)` from scratch; the **cderi** holds exactly
   those, and `jk_energy_per_atom` deletes it with a `with_df.reset()` on its
   first line.  `_decompose_j2c`'s own matrix satisfies `aux_coef @
   aux_coef.T == metric` — the very metric the gradient rebuilds — so the
   pseudo-density comes out of the existing tensor **exactly**, and the
   half-transform it costs is one the function was doing anyway (on a
   *smaller* basis: the cderi is contracted and spherical, 1007 functions
   against the gradient's decontracted Cartesian 1146).  **1.22x–1.46x** on
   the function.
2. **The derivative kernel is launched four shell pairs deep where one is
   faster.**  `df/grad/rhf.py:141` aggregates at `nsp_per_block*4`.  Swept
   over `{1,2,4,8,16,32}` and four `gout_width`s, **1** is the fastest of the
   six in every row of every table, by a margin that grows: **1.257x** on the
   kernel at 42 atoms/def2-SVP, **1.769x** at def2-TZVPP, **3.153x** at 113
   atoms.  Same integrals, same order, identical answer in every cell of the
   grid — and `gout_width` above 54 is not a preference but a **ceiling**:
   the kernel's answer changes there, by 89 at 72 and 1.8e3 at 108.

Together, one `jk_energy_per_atom` call at fixed density, round robin in one
process:

| | stock | **fastdfgrad** | | whole gradient |
|---|---|---|---|---|
| penicillin V, def2-SVP, B3LYP | 905.1 ms | **518.4 ms** | **1.75x** | 1.240 → **0.858 s**, 1.45x |
| penicillin V, def2-TZVPP, B3LYP | 2826.8 ms | **1314.7 ms** | **2.15x** | 3.317 → **1.651 s**, 2.01x |
| penicillin V, def2-SVP, ωB97X-V | 910.9 ms | **519.5 ms** | **1.75x** | 3.406 → **2.743 s**, 1.24x |
| taxol, def2-SVP, B3LYP (113 atoms) | 7369.0 ms | **2850.1 ms** | **2.59x** | 8.172 → **3.604 s**, 2.27x |
| Ni dithiolene⁻ doublet, UKS (21 atoms) | 793.4 ms | **556.8 ms** | 1.43x | 1.254 → **0.982 s**, 1.28x |
| Cu macrocycle²⁺ quartet, UKS (60 atoms) | 1868.5 ms | **973.1 ms** | **1.92x** | 2.664 → **1.771 s**, 1.50x |

Patched against stock, element-wise in the same process: **1.5e-9 to 7.0e-8
relative** per atom and component.  That is not the reuse:
`benchmarks/df_grad_metric_probe.py` takes GPU4PySCF's *own* `j3c_oo` and
merely associates the metric contraction the other way round, and it moves by
**more** — 7.8e-07 against the term's 5.6e-08 at 42 atoms, 2.3e-05 against
4.1e-06 at 113.  The fitting metric is conditioned at 2.3e9 and 1.9e10 on
those two systems, and both disagreements track that rather than the reuse.

**Two smaller results.**  Reusing the SCF's `Int3c2eOpt` — the same object
`DF.build` makes, thrown away one line before the gradient rebuilds it — is
worth 1.14x at 42 atoms/def2-SVP and nothing past it.  And building the
Coulomb term of the pseudo-density on the compressed pair list, which does
**strictly less work** than upstream's dense arrangement, is **0–1.4 %
slower** on all five systems: it ships off, behind `FASTDFGRAD_JPAIR=1`.

**A range-separated functional's two calls both reuse**, which needed
upstream's `reset()` — which empties `_rsh_df` — to become a partial release.
Without that the long-range call finds nothing and the whole-gradient ratio
reads 1.12x instead of 1.24x.

* **Success criterion met as stated**: the item asked for "a stage table with
  each stage against its roofline, then whatever that table points at".  The
  table is `docs/README_fastdfgrad.md` §2; what it pointed at was not on this
  page.
* **And a capacity wall the round found rather than closed.**  At 284 atoms
  the DF gradient does **not run** on one A100-SXM4-40GB, patched or not:
  `_jk_energy_per_atom` allocates `j3c_oo` as a dense `(naux, nocc, nocc)`
  array, which is `13365 × 547² × 8` = **32.0 GB**, and dies a few lines
  later in `cp.asarray(auxmol.ctr_coeff)`.  `fastdfgrad`'s `dm_oo` is the
  same array, so it dies in the same place.  The SCF at that size *does* run
  — §3.3 — because `_cholesky_eri` streams the cderi from the host; the
  gradient has no equivalent, and batching the occupied index is **W10**
  below.
* **And one finding that belongs to W4, not here.**  The `nsp_per_block*4`
  aggregation is not only in the gradient: the same constant is in
  `df/int3c2e_bdiv.py:176`, which is what `_cholesky_eri` builds the cderi
  through, and a `*16` is in `_j_energy_per_atom` and in
  `Int3c2eOpt.contract_dm`.  Swept on the plain integral kernel, `1` is
  **1.24x** at 42 atoms/def2-SVP, **1.46x** at def2-TZVPP and **3.19x** at
  113 atoms against upstream's `*4` — 1567.8 ms → 491.4 ms on taxol.  W4 is
  the cderi build, `_cholesky_eri` is **39.4 %** of a 15-cycle taxol DF SCF
  (§3.1), and this is the cheapest thing anyone has measured on it.

### W4 — The cderi build — **done**

`electrolites/fastdfcderi.py`, `tests/test_dfcderi.py`,
`benchmarks/df_cderi_probe.py`, `benchmarks/df_cderi_bench.py`, write-up in
`docs/README_fastdfcderi.md`.

**The stage table this item should have started from.**  A traced clone of
`_cholesky_eri`, checked against the real function on every run, splits the
block loop as

| | atoms | `eval_j3c` | metric contraction | store |
|---|---|---|---|---|
| glycine, def2-SVP | 10 | 91.7 % | 8.2 % | — |
| penicillin V, def2-SVP | 42 | 63.9 % | 36.1 % | — |
| taxol, def2-SVP | 113 | 52.0 % | 48.0 % | — |
| taxol, def2-SVP, host | 113 | 39.4 % | 36.5 % | 24.0 % |
| PfPMT, def2-SVP, host | 284 | 24.5 % | **59.5 %** | 16.0 % |

so the metric contraction does grow into the dominant term, as this entry
guessed.  What the entry did not have is the stage that is not in the loop at
all: **`empty_mapped`**, the `cudaHostAlloc` that page-locks the cderi's host
buffer, is **6.8 s of a 10.5 s** host-resident taxol build and **41.5 s of a
72.5 s** PfPMT one — bigger than the integrals and the contraction together,
and about 40 s of the **88.7 s** §3.3 measures for this function at 284
atoms.

**The first lead is rejected by its own falsification criterion.**  This
entry proposes `j3c · C` as a sparse gather followed by a `trsm`, and says it
is falsified if `dtrsm` runs at less than half of `dgemm`.  Swept on
synthetic matrices from 256 to 5598 auxiliary functions, cuBLAS's `dtrsm`
reaches **1.1 → 10.9 TF/s** against `dgemm`'s 12.4 → 19.0 and never exceeds
**0.66** of it; on real blocks the whole route is **0.37x–1.76x**, a loss
below about 2000 auxiliary functions.  The sparse half is fine — the
coefficients are 0.010–0.28 % dense, about 1.5 per column, and a one-pass
kernel applies them in 2.7 ms where the dense GEMM takes 43.7 (penicillin V).
It is the solve that gives the factor back.

**With a triangular *multiply* it pays.**  `L^{-1}` formed once and
explicitly — which is no more conservative than upstream, whose
`_decompose_j2c` solves for `C L^{-T}` with *more* right-hand sides — lets
cuBLAS's `dtrmm` run the per-block half at **15.4–16.8 TF/s**, GEMM
throughput for half the arithmetic.  The contraction is then **2.08x** at 42
atoms, **2.05x** at 21, **1.94x** at 113 and **1.90x** at 284.  Below
`npair × naux_sorted × naux = 4e10` the setup costs more than it saves and
the dense product is kept; that threshold is measured, not guessed.

**The second lead is worth what the sweep said.**  `sp_mult = 1` on the whole
build is 1.00x at 10 atoms, 1.06x at 21, **1.19x** at 42/def2-SVP, **1.34x**
at def2-TZVPP, **1.50x** at 113 and **1.18x** at 284, `1` is the best of
`{1,2,4,8}` everywhere, and the cderi is identical at every value.  The
`*16` in `_j_energy_per_atom` and `Int3c2eOpt.contract_dm` is still unswept.

**And the allocation, which no lead named**, is ordinary memory page-locked
with `cudaHostRegister` instead: **2.09 → 8.6 GB/s**, taxol's 14.31 GB in
1.70 s instead of 6.84 and PfPMT's 86.41 GB in 10.1 s instead of 41.3.  It is
the allocation that is slow and not the page-locking — registering
pre-allocated memory reaches 8.6 GB/s on ordinary 4 kB pages, and huge pages
change that by 0.3 % — and pinning does not parallelise across threads
either.

**What it is worth**, `_cholesky_eri` itself, every variant round robin in
one process:

| | cderi | GPU4PySCF | patched | |
|---|---|---|---|---|
| Ni dithiolene, def2-SVP (21 atoms) | device | 116.8 ms | **102.1 ms** | 1.14x |
| penicillin V, def2-SVP (42) | device | 166.4 ms | **128.1 ms** | 1.30x |
| penicillin V, def2-TZVPP | device | 589.8 ms | **339.6 ms** | **1.74x** |
| taxol, def2-SVP (113) | device | 3208 ms | **1751 ms** | **1.83x** |
| penicillin V, def2-SVP | host | 918.3 ms | **337.1 ms** | **2.72x** |
| taxol, def2-SVP | host | 12764 ms | **3990 ms** | **3.20x** |
| PfPMT, def2-SVP (284) | host | **87.1 s** | **29.5 s** | **2.95x** |

and on the SCF the build sits in, second of two rounds: 1.05x (penicillin
V/def2-TZVPP), **1.11x** (taxol, cderi on the device) and **1.37x** (taxol,
cderi on the host — 24.27 s → 17.69 s).  The build is a one-time cost, so a
module on it is worth the share it holds: 8 % of a small def2-SVP SCF and
53 % of one whose cderi lives on the host, which is every large molecule.

* **Success criterion met as stated**: the item asked for `_cholesky_eri`
  measured before and after on taxol.  12764 ms → 3990 ms host-resident,
  3208 ms → 1751 ms device-resident.  And the 284-atom row is the one number
  on this page that can be checked against something written before the
  round: §3.3 measures this function at **88.7 s** inside a full SCF, and the
  bench's unpatched column reads **87.1 s**.
* **Watch, as this entry says**: `gout_width` is a ceiling and was left at
  GPU4PySCF's 54.  Only `sp_mult` was swept.
* **Three things the round found rather than closed**, all in §4's new
  items: the host store (**W11**), the pinned buffer that is still 10 s at
  284 atoms (**W12**), and `_decompose_j2c`'s 20–62 ms of cuSOLVER latency at
  the small end (**W13**).

### W5 — A block-sparse exchange contraction — **done, 1.21x, and the last 30 % of it was an allocator**

`electrolites/_dfblock.py` and `FASTDF_STEP1` in `electrolites/fastdf.py`,
probes in `benchmarks/df_block_probe.py` and `benchmarks/df_block_bench.py`,
write-up in `docs/README_fastdfblock.md`.

**The item's premise was wrong, and the measurement that shows it is cheaper
than the implementation was.**  This entry priced the prize at the **4.5x**
reciprocal of §3.2's 22.2 % pair fill.  A GEMM cannot address a pair; it
addresses a tile, and the fraction of `nao x nao` covered by AO tiles holding
at least one stored pair is **69.4 %** at 284 atoms and **66.1 %** at 545 —
a ceiling of 1.44x and 1.51x on the first contraction, not 4.5x, and
**98.2 %** at 113 atoms, which is no ceiling at all.

The reason is the one fact this list did not have.  **The cderi's sparsity is
the product of two factors and only one of them is spatial:**

| | atoms | pair fill | = atom-pair fill | x AO density inside a surviving atom block |
|---|---|---|---|---|
| taxol | 113 | 50.6 % | 90.3 % | 56.0 % |
| PfPMT | 284 | 22.2 % | 52.0 % | 43.7 % |
| HcgC | 545 | 16.5 % | 45.0 % | 38.5 % |

A tile of 32 AOs is three and a half atoms, so it can reach the atom-pair
factor and nothing else.  The second factor is shell structure — which
exponent and which angular momentum — at one-to-five-AO granularity, and no
AO tile of any size addresses it.  Four AO reorderings were measured (`rcm`,
`rcb`, `morton`, and a shell-class-major one); **the best is worth three
points of fill and the worst minus two.**

**The item's own falsification criterion fires.**  At taxol's fill the tile
contraction is **0.91x** of the dense one and the whole exchange block
**0.72x**, so the acceptance criterion — faster at 113 atoms *and* at 284 —
cannot be met.

**What was built**, above a measured crossover: the first contraction
**1.31x** at 284 atoms in isolation and **1.38x** at 545, the exchange block
**1.09x** and **1.20x**, and a whole `get_jk` call at 284 atoms — the only
end-to-end size that runs — **10825 ms → 8916 ms, 1.21x** at tile 16.

**That last number was 0.90x until a profiler was pointed at it**, and the
difference was not the algorithm.  `RowCompressed.unpack` allocated a 763 MB
temporary per block; CuPy's pool went to the driver for it; and `cudaFree`
**synchronises the device**, so the double-buffered prefetch was drained 114
times a call and 2.33 s of an 11.73 s call was GPU idle while the tiled path
was already doing 1.13x less GPU work.  Allocating the staging buffer once
per pattern fixes it.  Three lessons, all in §5's rules now: a kernel
benchmark cannot see an allocator, `cudaFree` is a synchronisation point, and
a cached buffer must not be sized from free memory.

**Tile 16, not 32.**  End to end they are 1.21x and 1.07x where the kernel
benchmark puts them within 0.03x: once the per-block allocation is gone what
is left is the fill, and 61.9 % against 72.3 % is the whole difference.

It ships **on** (`FASTDF_STEP1=auto`), gated on the measured tile fill
(`FASTDF_TILE_MAX_FILL`, default 0.75) and **not** on the system size,
because a size gate is demonstrably wrong: taxol in def2-TZVPP has 2636 basis
functions against PfPMT's 2697 in def2-SVP and a tile fill of **98.9 %**
against 72.3 %.  Going up a basis set moves the two factors in opposite
directions — the atom-pair fill *rises* (52.0 → 63.5 %) as the diffuse
functions reach further, the within-block density falls (43.7 → 31.3 %) — and
the tile fill, which is the only one a tile sees, does not move.

* **Correctness** is `tests/test_df.py --ablate`, which now runs the tiled
  contraction at two tile sizes: **3.0e-16 (J) / 1.5e-14 (K)**, and the
  tiled and dense contractions are bit-identical up to 2697 AOs.
* **Two things it found rather than closed**: the tiled unpack is a CuPy
  scatter where GPU4PySCF's is a CUDA kernel, and after the allocator fix it
  is 0.752 s against 0.647 at 284 atoms — **W14**, now worth 0.1 s a call
  rather than the 6 s it appeared to be; and the route to the factor a tile
  cannot see is a localised occupied basis, which `vk = rhok rhokᵀ` admits
  *exactly* — **W15**.
* **And three measurement rules**, §5.7 to §5.9, all learned the hard way
  here.
* **One thing it does not deliver** that the entry promised: the dense
  `(nao, nao, blksize)` buffer does shrink by `fill` (7.45 → 4.61 GB at 284
  atoms), but the density factor has to be gathered onto the tile rows and
  that array is as large as the saving at 545 atoms.  Net footprint is a
  wash.

### W6 — Local / pair-atomic fitting — **done**

`electrolites/dflocal.py`, `tests/test_dflocal.py`,
`benchmarks/df_local_k_probe.py`, `benchmarks/df_local_k_bench.py`, write-up
in `docs/README_dflocal_k.md`.  The two conditions this item attached before
any code were measured first and are `docs/README_dflocal.md`: the fit has to
be **robust** (non-robust pair-atomic fitting collapses variationally by
16.5 Eh on butane and the collapse grows with the molecule), and a pair domain
holds **118-130** auxiliary functions on every system from 10 to 545 atoms.
What the item asked for next was the implementation and, inside it,
*"estimating the cost of PARI-K"*.  Both are done, and three of the four
results contradict something this entry used to say.

**It does factorise, and the density can be contracted first.**  The robust
expansion `eri = B c^T + c B^T - T c^T`, `T = c J`, has a symmetric third
term, so it splits evenly between the first two: with **`M = B - T/2` folded
once at build time** the exchange is `vk = M X + (M X)^T`, **one** pass where
global DF needs two, and `T` is never formed at run time.  Contracting the
occupied factor first then gives `vk[m,l] = sum_{i,Q} rhok[m,i,Q] G[i,Q,l]`
in which `rhok` is *exactly* the intermediate `fastdf` builds.  So what does
not factorise is `fastdf`'s **second** contraction, not its first, and this
entry's sentence about it was wrong in the useful direction.

**What it costs is `2 nocc / nlambda`, and `nlambda` is not `nlocal`.**
`nlambda` — how many AOs overlap an auxiliary function's own atom — is the
width of the exchange intermediate's second index, and unlike `nlocal` it
**grows**, like N^0.68 against `nocc`'s N^0.99.  Unscreened, PARI-K is
**0.42x** the arithmetic of stock global DF-K at 10 atoms, **1.09x** at 284
and **1.36x** at 545.  But a `lambda` at the edge of the integral screening
carries a fitting coefficient of 1e-8, and dropping it is an error in the
exchange matrix rather than in the fit: measured on the matrix itself, a
threshold of **1e-6** leaves 48 % of `nlambda` at taxol and changes `K` by
**1.2e-7** relative, against the 5.4e-4 the approximation is already away from
global DF.  Screened, `nlambda` is **236, 407, 374, 465** from 42 atoms to
545 — *flat*, which is the property the whole item wanted — and PARI-K is
**2.93x** stock global DF-K at 284 atoms and **4.35x** at 545.

**The storage number does not carry to the exchange build as it stands, and
does once the residual is thresholded.**  A robust expansion needs `(mn|Q)`
for every stored pair against *every* auxiliary function, not only the domain
ones, so the tensor a PARI exchange build touches is the same 86.4 GB at 284
atoms that global DF stores — the 110x is a number about the *fit*.  Splitting
`B = R + cJ` and storing only the residual recovers it: at 113 atoms a
threshold costing **2.9e-5** relative on `K` leaves **0.45 GB against 14.31**,
within 1.4x of the coefficient-count bound, and the ratio grows with the
molecule (2.6x, 5.3x, 9.5x, 31.5x from 10 to 113 atoms).  **The container is
not written**, and it is the largest thing left here: it is the one shape in
which a 545-atom robust density-fitted calculation is possible at all, where
global DF needs 414 GB against a node's 251.

**It is size-extensive, and the per-atom error is a range, not a constant.**
The `nao^4` reference of the round before this one stopped at C4H10; this
implementation takes the same homologous series to **C20H42, 62 atoms**, and
the error per atom is flat and falling — 1.90e-5 at CH4, 2.51e-5 at C4H10,
**2.22e-5** at C20H42.  What moves is composition: 3.3e-5 on glycine, 6.2e-5
on penicillin V, **7.4e-5 on taxol**.  So the number to carry is **2e-5 to
8e-5 Eh per atom**, and the single 2.5e-5 an alkane series gave is the bottom
of it.  A `pair+3.0` domain is 1e-7 per atom and about eight times `nlambda`,
so it is an accuracy reference and not an operating point.

* **Measured, at the last size where the tensor is device-resident** (taxol,
  113 atoms, one A100-SXM4-80GB, one side per process): `get_jk` **1.20x**
  stock global DF and **1.39x** with the screened `lambda` list; the build
  1.16x *faster* than upstream's and 1.69x slower than this repository's own
  `fastdfcderi`, which is implementation and not algebra — the fold it does in
  place of the global metric contraction is 49x fewer flops.
* **And at 284 atoms it is 2.0x slower, for a reason that is the point.**
  There the tensor is 86.4 GB and host-resident on both sides, and the call
  is **PCIe-bound**: 12.6 s of a 24.6 s `get_jk` is 86.4 GB off the host at
  6.8 GB/s, against the 26.4 GB/s GPU4PySCF's mapped, double-buffered
  prefetch reaches and hides entirely (§3.3).  The contraction itself is
  10.6 s for 7.3e13 flops.  So the flop ratio cannot show at the size this
  page is written around until the tensor stops crossing PCIe — which is the
  residual container above — and two of the three things in the way are
  already written in `electrolites/fastdfcderi.py`.
* **And where it can live is settled by construction.**  It is not in
  `patch_all()`, `density_fit()` here is not `mf.density_fit()`, it warns
  once, and `nuc_grad_method`/`Gradients`/`Hessian` on such an object raise.
* **Left**: the sparse residual container above; the host-memory paths
  `fastdfcderi` already has (`cudaHostRegister` for the store, a prefetched
  stream for the read), which are 24.9 s of a 43.6 s build and 12.6 s of a
  24.6 s call at 284 atoms; **gradients**, which need the non-variational
  fit's extra terms; and **range separation**, which a Coulomb-metric fit has
  no form for and which is most of what this repository is benchmarked on.

### W7 — fp32 cderi storage — **done**

`electrolites/_dffp32.py` (the mode), `electrolites/fastdffp32.py` (the two
consumers that are not `get_jk`), `tests/test_dffp32.py`,
`benchmarks/df_fp32_probe.py`, `benchmarks/df_fp32_bench.py`, write-up in
`docs/README_fastdffp32.md`.  `FASTDF_FP32=1`, off by default, named in
`README.md`'s limits paragraph.  Every condition this item attached is met
and three of its sentences turn out to be wrong in an interesting direction.

**"Technically the least work on this list" is wrong, and the reason is
worth carrying to anything else that changes a storage format.**  The cast
is four lines.  What is not is that **a narrow tensor has no
fall-through**: `fill_symmetric` and `transpose_write` take a `void*` and
are compiled for `double`, so handed a `float32` tensor they do not fail,
they read past the end of it and return a plausible wrong answer.  Three
paths reach the tensor — `df_jk.get_jk`, `DF.loop` (which is how the DF
gradient and the DF Hessian's CPHF reach it) and one line of
`df/hessian/rhf.py:_get_jk` — and all three had to be covered before the
switch could exist.  Every call `fastdf` *declines* now raises rather than
handing the tensor back to GPU4PySCF.

**The error is the Coulomb energy's, not the exchange matrix's.**  This
entry asks for the error "in the total energy and in the exchange matrix".
Measured at a fixed density, the Coulomb energy's share is **five to ten
times** the exchange energy's in every row of the ladder, and it is the
whole of the total-energy difference — the SCF's own relaxation contributes
nothing, as a variational energy should, and the iteration count never
moves.  Organics: **2e-7 – 2e-5 Eh**, flat at **3e-8 – 1e-7 Eh per atom**
over an alkane series from CH4 to C24H50 and 1.70e-5 Eh at taxol; the
matrices themselves at **8e-9 – 6e-8 relative**, which is single precision
and no amplification; gradients at **5e-9 – 7e-7 Eh/bohr**.

**And it is not uniformly cheap across chemistry**, which is the result
that decides how the switch is described.  On an open-shell 3d complex the
per-atom error is **1.3e-6 – 2.6e-6 Eh**, twenty to sixty times the
organics', because the largest cderi elements are core-region integrals and
the rounding is relative to those.  For **FeCl4(-)** the narrowing error is
**1.30e-5 Eh against a fitting error of 1.301e-5** — single precision costs
as much as density fitting itself.  For organics the same ratio is 6x–151x
in the other direction.  A Pt complex, whose core is an ECP, is back at the
organics' level, so it is the hard 3d core and not the open shell.

**"2x on PCIe" is exactly right, and mostly already taken.**  The link runs
at **26.7 GB/s in every configuration** — both precisions, every block
size, matched byte counts — so halving the bytes halves the transfer and
nothing more; a J-only call, which is the PCIe-bound one, is **1.90x–1.98x**
at 42 and 113 atoms and **2.60x** at 284.  But a J+K call at these sizes is
**1.10x–1.72x**, because §3.3's finding holds: the prefetch already hides
most of the transfer behind the exchange contractions.  And on a
*device-resident* tensor there is no transfer to halve at all — J+K is
**0.99x**, J-only **0.67x**, and what the mode buys there is the memory.

**What it is really for is capacity.**  At 284 atoms the double tensor is
86.4 GB and cannot go on an 80 GB card; the single one is **43.2 GB and
does**, which is a configuration GPU4PySCF cannot reach: the build goes from
29.1 s to **13.2 s** and a J-only call from 2.65 s to **0.19 s**.  At
**545 atoms** (HcgC) the double tensor is 414 GB against a node's 251 and
**cannot be built at all**; the single one is **207 GB and builds in
130 s**.  That is the same wall W6 is about, reached from the other side and
for one bit of precision rather than a different approximation.

**Left**: the error is J's, so a mixed scheme that keeps the Coulomb pass
in double (or keeps a narrow *residual* alongside a coarse double part)
would buy most of the storage at a tenth of the error, and nothing here
tries it; and `DF.loop` allocates its two staging buffers **per call**
(1.3 GB at taxol), which the ablation in `docs/README_fastdffp32.md` §3.2
prices at 6 % of the stream when the pool is warm and much more when it is
not — that one is upstream's and is §5.8 and §5.11 again.

### W8 — The DF Hessian's own CPHF build — **done, and it is a capacity item, not a speed one**

`electrolites/fastdfhess.py`, `tests/test_dfhess.py`,
`benchmarks/df_hess_probe.py` (the anatomy),
`benchmarks/df_hess_bench.py` (the A/B),
`benchmarks/df_hess_krylov_probe.py` (the memory),
`benchmarks/df_hess_cutensor_limit.py` (the plan boundary), write-up in
`docs/README_fastdfhess.md`.

`df/hessian/rhf.py:_get_jk` is a **fifth** two-electron code path: it builds J
and K directly in the MO basis and does not go through `df_jk.get_jk`, so
`fastdf` is not on it.  This entry assumed there was a factor in it.  There
is not, and the round found something else.

**The function is already at the machine, and the entry never said how big it
was.**  One call is 15.53 Tflop in **1015 ms** on one A100-SXM4-80GB —
**15.2 TF/s of 19.5, 78 % of fp64 peak** — its dominant contraction at
**81 %**, and each of its four contractions is within **1.14x** of a plain
`dgemm` at its own shape, two of them *faster* than one.  §2's headline for
`df_jk.get_jk` holds here too, and was reached independently.  Three
rearrangements were written and measured: grouping the algebra by the
perturbed density so the several-GB intermediate is written once and read
once is **0.90x–0.94x**; giving the two contractions that write into `vk` one
accumulator each in their own GEMM's layout is **0.91x–1.00x**; upstream's
ordering through this module's loop is **0.97x–1.00x**.  All three are still
runnable (`FASTDFHESS_ARRANGE`).

**And it is 15 % of the Hessian it sits in** — 5.19 s of a 34.6 s 42-atom
DF-B3LYP/def2-SVP Hessian, 6.3 % of a 22-atom def2-TZVPP one, 24 % after
`patch_all()`.  Even deleting it is 1.18x.

**What the round is for is that the function does not run past a certain
shape.**  GPU4PySCF sizes the density batch from free memory alone and hands
the whole batch to one cuTENSOR contraction; past about 5e8 output elements
`cutensor.createPlan` refuses `Lpq,snqi->snpiL` outright with
**`CUTENSOR_STATUS_NOT_SUPPORTED`**, raised out of `estimateWorkspaceSize`,
so it is the *descriptor* that is rejected and free memory does not help.
A **42-atom def2-TZVPP** DF Hessian dies there with 73.7 GB free on an 80 GB
card and 34.0 GB free on a 40 GB one; a **113-atom def2-SVP** one dies with
65.5 GB free.  Capping the batch and halving-and-retrying behind the cap
makes both run: **4.05 s** and **110.3 s** per call, at 16.2 and 15.7 TF/s.
The cap costs **0.7 %** at 42 atoms on an 80 GB card and is a small *gain* on
a 40 GB one.

**The "Leave space for storing Krylov subspace vectors" comment
(`:1258-1273`) is guarding 0.37 GB with 31 GB.**  Measured, a CPHF converges
in **five** Krylov cycles and its subspace is 0.37 GB at 42 atoms, against a
cderi of 0.99 GB, a whole-Hessian peak of about 2 GB, and a
0.4-of-free-memory reserve of 31 GB.  The module computes the reserve
instead.  **And the residency does not matter for the exchange call at all**:
with the tensor forced onto the host that call is **1017 ms against 1024**
device-resident, because the prefetch hides the stream behind the
contractions (§3.3 again).  It matters **9.0x** for a Coulomb-only call
(81.8 ms against 9.1), which is what a pure functional's CPHF takes —
so this entry's standing advice, *`FASTDF_RESIDENT_FRAC=0` is the setting for
a DF Hessian*, is right about the memory and irrelevant to the speed of a
hybrid's CPHF.

**Left**: the plan boundary is a measured number with no model behind it —
bisected, the largest planable `dm_batch` is 530 at (nao 430, nocc 92), 224 at
(478, 39), 22 at (1007, 92) and 8 at (1123, 226), which is not a clean
function of the shape; and a partially resident cderi for the Coulomb-only
branch, where the 9.0x is.

**What the round found rather than closed is W16, W17 and W18**: the first
anatomy of a density-fitted Hessian in this repository says the time is in
three functions no entry on this page names.

### W9 — The pure-functional DF Coulomb build — **done**

`electrolites/fastdfj3c.py`, `electrolites/_j3c.py`, `tests/test_dfj3c.py`,
`benchmarks/df_j3c_{probe,bench,sweep}.py`, write-up in
`docs/README_fastdfj3c.md`.

The entry called this "a `fastj`-shaped generator project rather than a driver
rewrite".  **It is the other way round.**  The kernels were left untouched and
the round is entirely in the driver, because the thing nobody had done was
time the *stages* of `df_jk.get_j`: unpatched, the two integral kernels are
**24 %** of a 10-atom Coulomb build, 63 % at 22 atoms, 85 % at 42, 93 % at 113
and 97 % at 284.  Everything else is host-side setup and density
transformations that do not grow.

**What it is worth**, one density, def2-SVP, A100-SXM4-40GB:

| | GPU4PySCF | fastdfj3c | |
|---|---|---|---|
| glycine, 10 atoms | 3.275 ms | **1.411 ms** | **2.32x** |
| alanine dipeptide, 22 | 7.057 ms | **3.240 ms** | **2.18x** |
| penicillin V, 42 | 26.23 ms | **14.70 ms** | **1.78x** |
| taxol, 113 | 157.5 ms | **139.6 ms** | 1.13x |
| PfPMT, 284 | 731.4 ms | **708.3 ms** | 1.03x |
| penicillin V, def2-TZVPP | 69.07 ms | **53.49 ms** | **1.29x** |

and **more than one density**, which is every response property of a pure
density-fitted functional: **4.45x** at two, 2.73x at four, 2.58x at eight and
**2.55x at thirty-two** on penicillin V; 2.65x/1.61x/1.50x on taxol.  On an
open-shell 3d complex it is **13.2x** (Ni dithiolene, 21 atoms, 317.8 ms →
24.0 ms) and **6.7x** (Cu macrocycle, 60 atoms).  A whole DF-PBE SCF is
**1.16x** at 42 atoms and 1.05x at 113 against a `patch_all()` baseline.

**Five host-side items, no arithmetic.**  The largest two:

* `Int3c2eOpt.build` makes **one** thread-block range per angular-momentum
  group, and the auxiliary-vector kernel's grid is
  `(groups × auxiliary-shell batches)` — **372 blocks on a 108-SM card** at 42
  atoms, with the groups differing in size by seventy times.  Both kernels
  read their `(li,lj)` from the first pair of their range and stride the rest,
  so a range may be any sub-interval of a group: this is a launch
  configuration change, the same shape as W3's `nsp_per_block*4`.  The swept
  optimum moves by **thirty-two times** across the size ladder, so the block
  size is *derived* from the molecule and the basis and not fixed — within
  1.7 % of the swept optimum on seven systems in three bases, where a fixed
  1024 would have given up 29 % on glycine.  The *other* kernel is left alone:
  splitting is monotonically worse for it, down to 0.37x.
* `df_jk.get_j` sends every `n_dm > 1` call to the **Rys** `int3c2e_bdiv`
  engine, which carries the density index inside one launch where the
  McMurchie–Davidson engine loops and rebuilds the integrals per density.
  That is the wrong choice at every count up to 32, by 1.36x–2.54x, and two
  densities is the worst because the Rys kernel batches densities in fours.

The other three are the per-call uploads (thirty host-to-device copies and six
NumPy cumsums per Coulomb matrix, in `SortedGTO`'s four recontraction kernels
and the two contractions), the Hermite scratch (§5.8's `cudaFree`), and
`ao_loc`.  All are cached per **device** as well as per object, because
`multi_gpu.run` hands one `SortedMole` to a thread per device — W2's trap,
avoided rather than found.

**Two ideas measured and rejected**, both kept runnable: a smaller
shared-memory budget (the launch asks for **91 KB per block**, one block per
A100 SM, and cutting it is monotonically slower — these kernels want shell
pairs in flight, not blocks per SM) and one launch per angular-momentum class
with that class's own shared memory (**0.42x–0.44x**: thirty small grids
serialise on the legacy default stream).  And one measured and *not taken*:
32 auxiliary shells per block instead of GPU4PySCF's 16 is worth 0.9–1.7 % of
one kernel, and moving it would invalidate the seven-system audit of the block
size it feeds.

**What it found rather than closed is W19 and W20.**

### W10 — Batch the DF gradient's `(naux, nocc, nocc)` array — **done**

*(Found by W3, which could not run at 284 atoms because of it.)*

`electrolites/fastdfgrad.py` (`_OOStore`, `_metric_pass`, `_fill_from_j3c`),
`tests/test_dfgrad.py` §5, `benchmarks/df_grad_oo_sweep.py`, write-up in
`docs/README_fastdfgradoo.md`.

**The success criterion is met.**  A 284-atom DF-B3LYP gradient completes on
one A100-SXM4-40GB: the whole analytic gradient in **68.8 s**, of which
`jk_energy_per_atom` is 66.1 s — **96 %**, which is the 97 % §1 measures on
taxol and is why this item was the only thing in the way.  GPU4PySCF cannot
run it at all.  Priced against the same function without the cderi reuse
(itself only runnable because of this round), **80.8 s against 64.5**, and
the two agree to 2.0e-6 Eh/Bohr on an `|ejk|max` of 52.9 — **3.8e-8
relative**, which is where the 1.5e-9 at 42 atoms and 7.0e-8 at 113 of
`docs/README_fastdfgrad.md` §5 were heading.

**And the array is bigger than this entry said.**  32.0 GB is
`13365 × 547² × 8`, and 13365 is `mf.with_df.naux`, the *spherical*
auxiliary dimension.  The allocation is on `int3c2e_opt.auxmol.nao`, the
**Cartesian sorted** one, which is **15762**: `j3c_oo` is **37.7 GB**,
`dm_oo` is a second one, and what `_jk_energy_per_atom` asks a 40 GB card
for is **75.5 GB**.

**What the item got wrong is the useful part.**  It prices the slicing at
"extra passes over the cderi", and there are none.  The array is *built* one
auxiliary block at a time in both routes — that is what the block loop
already is — and it was only ever *held* whole.  The one step that needs the
whole auxiliary index at once is `coef @ oo`, and that one slices the
**occupied** index and runs **in place**.  So the second copy is gone
unconditionally, the ip1 loop is untouched, and the slicing is free: one
slice against eight is 1.000x at 42 atoms and 0.991x at 113.

**The falsification therefore does not fire, and the threshold it asks for
is not the one that was needed.**  What costs is not the slicing but *where
the array lives*: a host-resident store is 1.087x at 42 atoms and 1.294x at
113, and that is exactly four traversals of it over PCIe plus one
page-locking (47.0 ms predicted against 45.3 measured, 721 against 719.5).
Walked up a ballast ladder on taxol until the device store would no longer
allocate, the device store is ahead at **every** rung — 1.34x with 22 GB
withheld, 1.18x with 24, 1.04x with 24.5 — so there is no crossover to place
a threshold at.  `FASTDFGRAD_OOFRAC` is set at 0.8 and the thing that decides
the hard cases is that the allocation is attempted and falls back to the host
on `OutOfMemoryError`.

**Three things only the host store needs, and all three pay**: the
double-buffered prefetch under the ip1 kernel (1.029x / 1.038x), page-locking
at all (1.077x / 1.206x), and page-locking with `cudaHostRegister` rather
than `cudaHostAlloc` (**1.200x / 1.437x**) — **W12**'s 8.6 GB/s against 2.09
turning up in a second module, on an array a thirtieth the size of the one it
was measured on.

**And what the round found rather than closed is W21.**  Freeing the second
array makes `mem_avail` larger, the ip1 loop's block bigger, and the gradient
reachable by the *same* cuTENSOR plan wall `docs/README_fastdfhess.md` found
on the DF Hessian: `CUTENSOR_STATUS_NOT_SUPPORTED` with tens of GB free.  A
halve-and-retry around the ip1 loop is the fix, and pinning the block says
why it has to be: the refusal does not depend on free memory and is **not
monotonic in the block** — 1096 is refused where 1240, 1456, 2048 and 3072
are accepted — so no element cap of the kind W8 carries can describe it.

### W11 — The host store, and the stream it runs on — **done, and it is not a host thread**

*(Found by W4, which measured it rather than closed it.)*

`electrolites/_dfstore.py` (`HostStore` and its two kernels), the host branch
of `electrolites/fastdfcderi.py`, `tests/test_dfcderi.py` (all three stores on
every host-resident case), `benchmarks/df_store_probe.py`, write-up in
`docs/README_fastdfstore.md` §2.

**The premise of this entry was wrong and its conclusion was exactly right.**
The entry called `libvhf_rys.transpose_write` "a synchronous host-side
transpose-and-copy on one thread" and asked for the write's GB/s first, on
the theory that "it is the transpose, not the transfer".  Disassembled,
`transpose_write` calls `cudaHostGetDevicePointer` and launches a CUDA kernel
— grid `(naux,1,1)`, block `(512,1,1)` — **on the default stream**.  There is
no host loop: it moves a 2.05 GB block in **117.8 ms at 17.37 GB/s**, and that
number is the same to four digits at `OMP_NUM_THREADS` of 1, 4, 16 and 32.

**And it is not slow, either.**  What sets the kernel's rate is how long a
contiguous run it writes into host memory, and that run is the block's *pair*
count: 45683 pairs is a 365 kB row and gets 17.37 GB/s, 111000 pairs is
888 kB and gets **24.61**, against a plain `cudaMemcpy` of the same bytes at
26.28.  At 284 atoms upstream's store is already at 94 % of the link.  What
it is, at every size, is **synchronous** — which is the one thing this entry
got right and the only thing worth fixing.

**So the fix is the entry's own, plus one half it does not name.**  The write
goes on a `non_blocking` stream of its own, with **two** destination blocks so
that the block being written is not the block being computed and an event
that lets batch `n+2` reuse batch `n`'s buffer — and it is handed to a **copy
engine** rather than to the SMs: the transpose is done on the device into a
staging slice (2.7 ms of 117, 744 GB/s, free) and what is left is a
`cudaMemcpy2DAsync` at 24.36 GB/s against the kernel's 17.37.

On the block loop of the real function, which the module now times for itself:

| block loop, cderi on the host | GPU4PySCF | upstream's kernel, own stream | + the copy engine |
|---|---|---|---|
| penicillin V, def2-SVP (42 atoms) | 181.4 ms | 182.5 ms (0.99x) | **163.8 ms (1.11x)** |
| taxol, def2-SVP (113 atoms) | 2649.0 ms | 2631.9 ms (1.01x) | **2385.3 ms (1.11x)** |
| PfPMT, def2-SVP (284 atoms) | 28768.4 ms | 28763.7 ms (1.00x) | **23787.1 ms (1.21x)** |

and inside the whole module, where the rest of the loop is faster too, the
store is worth **1.16x** at 42 atoms, **1.19x** at 113 and **1.41x** at 284
(`all` 119.7 / 1406.3 / 11589.7 ms against `all -store` 139.2 / 1675.6 /
16365.5).  The cost is one `(batch_pairs, naux)` buffer — 2.05 GB at 113
atoms, 11.9 GB at 284 — and the object declines the arrangement, with a
reason, if `get_avail_mem` says it will not fit.

**The half that is not named is the half that pays, and a synthetic loop says
the opposite.**  Moving upstream's own kernel to the store stream and
changing nothing else is 1.00x on the real call: a store done by the SMs has
nothing to overlap with, because `fill_int3c2e` and the metric contraction
are what the SMs are already doing.  A *synthetic* block loop — a spin kernel
of a known duration standing in for the integrals — reads it the other way,
1886 ms against the staged route's 1937 at 113-atom shapes and 10262 against
12520 at 284-atom ones, because a spin kernel occupies thread blocks without
wanting the issue slots the integrals want.  §5.7 for the third time.  Both
routes are kept runnable (`FASTDFCDERI_STORE=direct`, `=stage`).

**Two more things measured and not used.**  Splitting the staged route's
transposes and copies over two streams, so the copy engine is never behind a
transpose, is *slower* (2177 against 1937 ms at 113-atom shapes).  And the
staging slice has to be taken along the **auxiliary** index, not the pair
index: a slice of pairs makes every row of the 2-D copy as narrow as the
slice (45 kB at a 256 MB budget) and a copy that narrow does not reach the
link, which cost 400 ms of a 2.6 s loop.  The same asymmetry is why the
*tiled* transpose is the wrong kernel to point at host memory at all —
writing 32-double runs it gets **6.48 GB/s** at 284-atom shapes, a quarter of
upstream's.

**And a measurement rule this round had to learn** (now §5.14): the round
robin in `benchmarks/df_cderi_bench.py` drifts across the columns of one
round on the *host* branch.  An `all` column repeated in the last position
read **2218–2259 ms against 1479–1575** in three separate jobs, with an
identical configuration.  A few hundred milliseconds between two adjacent
columns there is not an item.  The same three jobs read the **block loop** of
those two columns as 1406.3 / 1410.2 ms — four digits apart — so the drift is
entirely outside the loop, and W11 is priced on the loop, which
`electrolites/fastdfcderi.py` times for itself and reports as
`stats()['loop_s']`.

### W12 — A pinned-buffer pool — **done, and the range-separated half of this entry is not true**

*(Also from W4, which made the allocation 4.1x faster and did not remove it.)*

`electrolites/_dfpin.py`, the host branch of `electrolites/fastdfcderi.py`,
`tests/test_dfcderi.py` (the buffer built twice, by address as well as by
counter), `benchmarks/df_store_probe.py --pool`,
`benchmarks/df_cderi_bench.py --steps`, write-up in
`docs/README_fastdfstore.md` §3.

**The success criterion is met exactly as written.**  Taxol's 14.32 GB host
tensor, built four times in one process: `cudaHostAlloc` 6709/6692/6860/6841
ms (2.09–2.14 GB/s), `cudaHostRegister` 1622/1641/1645/1722 (8.31–8.83), and
`cudaHostRegister` with the pool **1662 / 0.1 / 0.0 / 0.0 ms**.  The second
and later builds of one process pay nothing.

**On `_cholesky_eri` itself**, cderi on the host, every column round robin
in one process — and read the `all` column as a *repeat* and `all/cold` as a
first build, because the pool is the one thing in this round a single SCF
cannot use:

| | GPU4PySCF | `pin` (W4) | `all/cold` | | `all` | |
|---|---|---|---|---|---|---|
| penicillin V, def2-SVP (42 atoms) | 961.6 ms | 2.47x | 341.5 ms | 2.82x | **169.5 ms** | **5.67x** |
| taxol, def2-SVP (113) | 12035.4 ms | 2.55x | 3464.8 ms | 3.47x | **1574.7 ms** | **7.64x** |
| PfPMT, def2-SVP (284) | 91278.6 ms | 2.18x | 24024.3 ms | **3.80x** | **12037.6 ms** | **7.58x** |

`docs/README_fastdfcderi.md` §6.2 measured the last of those at 2.95x with
W4 alone.

**End to end, three `DF.build()` calls on displaced geometries** — which is
the shape of a geometry-optimisation step loop, and the only caller that
pays the page-locking more than once — taxol/def2-SVP, cderi on the host:

| | step 1 | step 2 | step 3 | total |
|---|---|---|---|---|
| GPU4PySCF | 9.63 s | 9.59 s | 9.58 s | 28.80 s |
| `cudaHostRegister` (W4) | 4.59 | 4.57 | 4.46 | 13.62 |
| + the pool (W12) | 4.57 | **2.84** | **2.80** | **10.21** |
| the whole module | 3.32 | **1.83** | **2.24** | **7.40 s** |

**The falsification is what shaped the policy.**  The pressure that matters
is *host* memory, not `get_avail_mem`'s device memory, so a buffer is
retained only while `MemAvailable` stays above the larger of 8 GB and a
tenth of the node, the pool is capped at 60 % of the node (or
`FASTDFCDERI_POOL_GB`), and the least recently released buffer is evicted
first.  The reuse window is narrow on purpose — capacity between the request
and 1.25x it — so that a 14 GB build cannot be handed the 86 GB buffer of
the last one and hold it for the rest of the process.

**But the range-separated half of this entry is wrong.**  `DF.range_coulomb`
does build a second `DF` object and does page-lock a second buffer, and it
**keeps the first**: `df/df.py:245` stores the range-separated object in
`self._rsh_df[key]` and both live until the SCF ends.  The two tensors are
alive at the same time, so there is nothing for a pool to hand over and it
correctly does nothing there.  That is worth recording for its own sake: at
284 atoms a range-separated density-fitted SCF holds **two** 86.41 GB host
tensors, 173 of the node's 251 GB, and nothing warns.  That is a capacity
item, not a speed one, and it is **W22**.

### W13 — `_decompose_j2c`'s latency at the small end — **done, and it is none of the three**

*(From W4's stage table.)*

`electrolites/_dfj2c.py`, `tests/test_dfcderi.py` §W13,
`benchmarks/df_j2c_probe.py`, write-up in `docs/README_fastdfstore.md` §4.
`fastdfcderi` now also replaces `df/int3c2e_bdiv.py:int2c2e`, which
`df/df.py:602` looks up on the module and so reaches upstream's own
`_decompose_j2c` as well as this repository's.

**The item asks which of `int2c2e`, `potrf` and the solve the 20–62 ms is.
It is none of them.**

| | naux | `_decompose_j2c` | `int2c2e` | of which `Int2c2eOpt()` | the kernel | `cholesky` | `dtrsm` |
|---|---|---|---|---|---|---|---|
| glycine, def2-SVP | 471 | 18.9 ms | 17.1 | **14.1** | 2.9 | 0.70 | 0.40 |
| alanine dipeptide | 974 | 20.2 | 17.4 | **14.3** | 2.9 | 1.67 | 0.92 |
| Pt amine | 1358 | 42.4 | 38.1 | **32.3** | 5.9 | 2.51 | 1.46 |
| Ni dithiolene | 1992 | 35.4 | 29.3 | **23.9** | 5.3 | 3.71 | 2.67 |
| penicillin V | 2191 | 30.6 | 23.2 | **19.5** | 4.1 | 4.42 | 3.16 |
| taxol | 5598 | 65.3 | 25.2 | **15.0** | 10.4 | 14.3 | 26.0 |

70–80 % of the stage at the small end is the `Int2c2eOpt` object `int2c2e`
constructs per call and drops when it returns, and it is **nearly flat in
system size** — 14.1 ms at `naux` 471 and 15.0 at 5598 — which is the
signature of launch latency rather than of work.  Inside it, `from_cell`
costs **0.00 ms** (the auxiliary basis is already sorted), `_scale_sp_ctr_coeff`
0.12–0.21 and `PBCIntEnvVars` 0.14; the whole of the rest is **one loop**
building the shell-pair index table with about eight individual cupy calls
per pair of angular-momentum classes — 66 pairs for glycine at 13.79 ms, 171
for the Pt amine at 31.57, 0.21 ms a pair either way.

Built with numpy and moved in one transfer it is **0.83–1.92 ms**, a
10x–17x, and the probe checks on every row that it is the same table.  (At
**284 atoms** that reverses: the numpy index arithmetic grows with the number
of shells where upstream's launch latency grows with the number of
angular-momentum classes, and building the table on the host is 17 ms
*slower* there — 0.07 % of a first build, and paid back by the cache on every
later one.)  The
optimiser is then also **kept**, keyed on a fingerprint of `_atm`, `_bas`
and `_env` with the one word that holds `omega` left out — `mol.omega` *is*
`_env[PTR_RANGE_OMEGA]`, so a naive fingerprint would rebuild for the second
of the two builds a range-separated functional makes, and leaving it out is
safe because `fill_int2c2e` takes `omega` as an explicit argument (measured:
an optimiser built at ω = 0 reproduces GPU4PySCF's matrices at ω = ±0.3 to
**0.0**).

**The CPU factorisation the item asks about is rejected by six to
twenty-six times**, with both transfers charged to it: LAPACK `dpotrf`
4.38 ms against cuSOLVER's 0.70 at naux 471, 55.22 against 4.42 at 2191, and
373.32 against 14.26 at 5598 — and the gap *grows* with the order, which is
the opposite of what a latency argument predicts.  The copy, the `tril` and
the `dev_info[0]` read around `potrf` are 2–8 % of `cholesky`, so there is
nothing there either.

**This is no longer the least valuable item on this page**, which is what
the entry called it: it is worth 14–32 ms of a 30–170 ms build at the small
end, and the same 14–32 ms again in every gradient and every Hessian that
calls `_decompose_j2c` at a geometry already seen.

### W14 — A CUDA kernel for the tiled unpack

*(Found by W5, which measured it rather than closed it.)*

The block-sparse contraction reaches its row-compressed buffer in two kernels
— a transpose into a staging buffer and a CuPy scatter — where GPU4PySCF's
dense unpack is one hand-written `decompress_and_transpose`.  Profiled inside
a real call at 284 atoms that is **0.457 + 0.295 = 0.752 s against 0.647**,
so it is now 16 % behind rather than the four times it appeared to be before
the allocator fix of `docs/README_fastdfblock.md` §4.1.  In
`df_block_bench.py`'s isolated timing it still reads 7.8 ms against 2.2 per
block, which §5.7 explains.

**This is a small item now.**  It is kept because it also removes the staging
buffer — `npair * blk` words, 0.67 GB at 284 atoms — which is the larger half
of what it is worth.

* **Do**: the same kernel GPU4PySCF has, with a destination table instead of
  a `divmod`.  `RowCompressed` already computes the two destination offsets
  per stored pair (`electrolites/_dfblock.py`), and the access pattern is the
  same shape as upstream's — contiguous in `L`, one row of `nL` doubles per
  pair — so §2's "2.65x slower than GPU4PySCF's" result, which was about an
  `(L,i,j)` *layout*, does not apply.
* **Success**: the tiled unpack at or below the dense one's 0.647 s in a
  profiled 284-atom call, and the staging buffer gone.  Worth perhaps 0.1 s
  of an 8.9 s call, plus the block size the freed memory buys back.
* **Falsified if**: the scatter is bandwidth-bound.  1.65 GB in the profiled
  0.295 s of scatter is 5.6 GB/s of *useful* traffic, far under the dense
  kernel's 1.43 TB/s, so it is not.

### W15 — Localised occupied orbitals

*(Found by W5, which measured the factor a tile cannot reach.)*

`vk[i,j] = sum_{k,L} rhok[i,k,L] rhok[j,k,L]` is **exactly** invariant under
an orthogonal rotation of the density factor's second index — `rhok -> rhok U`
leaves the sum over `k` alone — so the factor may be replaced by a localised
one for free and to round-off.  `rhok` is then sparse in `(i, k)` as well as
`(i, j)`, which is what every linear-scaling exchange implementation exploits
and the only route to W5's §1.1 second factor that does not need a
shell-granular GEMM.

* **Do**: read the occ-RI-K / linear-scaling-K literature first.  A pivoted
  Cholesky of the density matrix gives localised vectors and is exact; the
  hard part is not the localisation but batching orbitals into groups with a
  common support so the GEMMs stay large.
* **Success**: an exchange matrix at the `tests/test_df.py` tolerance with a
  contraction cost that is visibly sub-quadratic on the §3.2 ladder.
* **Falsified if**: the per-orbital supports at 284 atoms are not small —
  measure the support of a Cholesky-localised occupied set *before* writing
  anything, the way `df_block_probe.py` measures the tile fill.  That probe is
  an afternoon and this item is not.
* The AO-permutation machinery in `electrolites/_dfblock.py` is reusable here:
  it is the half of the problem that already exists and is tested.

### W16 — `nr_rks_fxc_mo`, the CPHF's grid half, which `fastxcfxc` is not on

*(Found by W8's anatomy, which is the first profile of a density-fitted
Hessian in this repository.)*

`electrolites/fastxcfxc.py` replaces `dft/numint.py:nr_rks_fxc`, which is what
a TDDFT or polarizability response solve calls.  A **Hessian's** CPHF calls
`hessian/rks.py:nr_rks_fxc_mo` — a different function, in a different module,
with the MO-basis transformation folded in — and that row does not move under
`patch_all()`: **5.36 s unpatched, 5.40 s patched**, on a 42-atom
DF-B3LYP/def2-SVP Hessian.  It is then the **single largest stage of a
patched density-fitted Hessian, 25 %** of 21.24 s.

* **Do**: read `hessian/rks.py:nr_rks_fxc_mo` against `dft/numint.py:nr_rks_fxc`
  first — `fastxcfxc`'s kernel may transplant, in which case this is a
  binding job and not a kernel one, which is the shape `fastxcnlc`'s
  `_DFHF.get_veff` finding had (§1).  The unrestricted twin is
  `hessian/uks.py`'s.
* **Success**: the row moves by what `fastxcfxc` is worth elsewhere
  (2.1x–3.7x inside a response run), i.e. a patched 42-atom DF Hessian at
  about 18 s rather than 21.2.
* **Falsified if**: the two functions differ in more than the basis the
  density is expressed in — the `_mo` variant evaluates the perturbed density
  from `mo1` on the grid rather than from a density matrix, and if that is
  where its time is then `fastxcfxc`'s kernel is not the thing to move.

### W17 — `df/hessian/rhf.py:_jk_energy_per_atom`, the DF Hessian's ip2 term

*(Also from W8's anatomy.)*

**21 %** of a patched 42-atom DF Hessian (4.50 s of 21.24), and **13 %**
unpatched.  It is the two-electron half of `partial_hess_elec` and it is a
**sixth** code path: it is not `df/grad/rhf.py:_jk_energy_per_atom`, which
`fastdfgrad` replaces (that one is the ip1 gradient term), and it is not
`hessian/rhf.py:_partial_ejk_ip2`, which `fastejk2` replaces (that one is the
direct path's).  Nothing in this repository is on it.

* **Do**: profile it the way W3 profiled the gradient's — stage by stage, and
  do not assume the contraction is the cost.  `fastdfgrad`'s two findings
  (integrals the SCF's cderi already holds, and the `nsp_per_block*4` launch
  constant) both apply to a function of this shape and neither was
  predictable from the source.
* **Success**: the row moves by what `fastdfgrad` was worth on its sibling,
  1.75x–2.59x on the function.
* **Falsified if**: the derivative integrals dominate and are already at the
  machine, which is what `docs/README_fastdfgrad.md` found for the
  contraction and not for the kernel.

### W18 — `df/hessian/rhf.py:_get_veff`, the DF Hessian's h1 term

*(Also from W8's anatomy.)*

**10 %** of a patched 42-atom DF Hessian (2.13 s), the two-electron half of
`make_h1`, and a **seventh** code path.  Smaller than W17 and of the same
shape — both build a density-fitted quantity per atom out of
`Int3c2eOpt` — so the two are one project rather than two, and W17 is the
half worth measuring first.

### W19 — The three-centre Coulomb kernels themselves

*(Found by W9, which measured them rather than changed them.)*

After W9 the Coulomb build of a pure density-fitted functional is **85 % two
integral kernels at 42 atoms and 97 % at 284**, and all three of the things
inside them need the CUDA source
(`gvhf-md/contract_int3c2e.cu`).  This is the `fastj`-shaped generator project
the old W9 entry described; what has changed is that it is now the *only*
thing left on this path, and that its two halves are known to cost the same.

* **Six double-precision divisions and a square root per (shell pair,
  auxiliary shell).**  `contract_int3c2e_kernel` recomputes every shell pair's
  Gaussian product centre — `xij = (ai*ri[0] + aj*rj[0]) / aij` and its two
  partners — inside both the auxiliary-primitive loop and the pair loop,
  although it is a property of the pair alone; then `theta = aij*ak/(aij+ak)`
  and `fac = ck/(aij*ak*sqrt(aij+ak))`.  `fastj` removed exactly this family
  from the four-centre MD J kernel by caching `1/aij` beside the product
  centre and letting one `rsqrt(aij+akl)` supply both reciprocals
  (`docs/README_fastj.md`).  Here the pair data is a `(npair, 4)` device array
  built once at `Int3c2eOpt.build()` — which is the shape `electrolites/_j3c.py`
  already is.
* **The Hermite index table is a global-memory read per term.**  The
  contraction is `vj_xyz[k] += Rt[p1_ij[k*nf3ij + i] * nsp] * dm_ij` with
  `p1_ij` a `uint16_t*` into `Rt2_kl_ij`, and the kernels template on `lk`
  alone — so `nf3ij`, `order` and every index is a runtime value.  Templating
  on `lij` as well makes them compile-time constants.
* **`omega == 0` is a runtime test** inside `boys_fn`, on the innermost path.

* **Do**: `codegen/gen_j_kernels.py` is the precedent and the 1.8.1 source is
  the input.  Start with the pair data, which is the cheapest of the three and
  needs no templating.
* **Success**: the two kernels move by what `fastj`'s scaffolding was worth on
  the four-centre one.
* **Falsified if**: the low angular-momentum classes — which hold most of the
  shell pairs — turn out to be bandwidth-bound on the Hermite density rather
  than instruction-bound.  Profile one class before writing a generator;
  W9 did not, because it never touched the kernels.
* **Not** a route to the `nsp_per_block`/shared-memory trade: W9 swept that
  from the host and it is monotone the way GPU4PySCF has it
  (`docs/README_fastdfj3c.md` §8).

### W20 — The three-centre Coulomb build does not screen on the density

*(Also from W9.)*

`J` is linear in `D`, so `J(D) = J(D_last) + J(D - D_last)` **exactly**, and
`D - D_last` near convergence is small almost everywhere.
`scf/j_engine.py`'s four-centre MD J engine screens on `qd_ij_max`, a
density-weighted Schwarz bound.  The three-centre one has **no density
screening of any kind**: it walks every stored shell pair against every
auxiliary shell on every cycle, 330 612 × 5 817 of them at 284 atoms.

* **Do**: filter `shl_pair_idx` on `max|ΔD_ij| · q_ij` and rebuild `pair_loc`
  for the filtered list, which is a host-side change of exactly the kind W9 is
  made of — and hand the *difference* Coulomb matrix back to an accumulating
  driver.  `_DFHF.__init__` sets `direct_scf = False`, so the driver half is
  `fastdfj`'s §1.2 in a second costume.
* **Success**: the pair count falls through an SCF, and the last cycles of a
  284-atom DF-PBE run cost a fraction of the first.
* **Falsified if**: the filtered list costs more to rebuild each cycle than
  the pairs it removes save — it is a *different* list per cycle, so it gives
  back exactly the caching W9's §2.1 is about, and `Int3c2eConstants` would
  need a two-tier cache.  Measure the pair count first: a probe over an SCF
  trajectory is an afternoon and this item is not.

### W21 — Two modules have now hit the same cuTENSOR plan wall

*(Found by W8 on the DF Hessian's CPHF build and by W10 on the DF gradient's
ip1 loop, from opposite directions.)*

`cutensor.createPlan` answers `CUTENSOR_STATUS_NOT_SUPPORTED` with tens of GB
free, and the call fails rather than running slowly.
`docs/README_fastdfhess.md` is where it stops a 42-atom def2-TZVPP DF Hessian
and a 113-atom def2-SVP one from running at all; `docs/README_fastdfgradoo.md`
§4.1 is where it reappears in the gradient's `contract('iqr,pi->pqr')` the
moment W10 stops reserving for an array that is no longer allocated and the
block grows.  W8 guards it with a measured element cap and a halve-and-retry;
W10 carries only the retry, because it measured the boundary and **a cap
cannot describe it**.

**What the measurement says** (taxol/def2-SVP on an 80 GB card, the ip1
block pinned with `FASTDFGRAD_BLK`, `docs/README_fastdfgradoo.md` §4.1):

* it does **not** depend on free memory — a block of 1096 is refused at 70.3,
  62.2, 54.2, 50.2 and 46.2 GB free alike, and 1240 is accepted at all five;
* it is **not monotonic in the block** — 8, 128, 512, 1240, 1456, 2048 and
  3072 are all accepted and only **1096** is refused, and the largest
  accepted puts **4.3e9 elements** in the output, twice 2³¹;
* so no threshold of the form "keep the output under N elements" describes
  it, and what clears a refusal is *changing the extent*, which is what
  halving does.

* **Do**: re-run `benchmarks/df_hess_cutensor_limit.py` with the block pinned
  and the memory varied independently, the way §4.1 of the gradient write-up
  does, and find out whether W8's `MAX_PLAN_ELEMENTS = 5e8` is a real
  boundary or a coincidence of the shapes it was bisected on.  If it is a
  coincidence, the cap should go and the retry is the whole answer in both
  modules.
* **Success**: one statement about what cuTENSOR actually refuses, and one
  fewer constant in the repository.  It is also the first thing to re-run on
  a different cuTENSOR or a different card — §5 of this page has a rule about
  constants measured on one machine.
* **Falsified if**: the refused extents turn out to be reproducible and
  predictable after all (1096 = 2³·137, and a large prime factor in the
  extent is the obvious thing to test next), in which case a cheap test
  beats a retry and both modules should carry it.

### W22 — A range-separated DF run holds two host cderi tensors at once

*(Found by W12, which went looking for a pool and found there is nothing to
pool.)*

`DF.range_coulomb` builds a second `DF` object for the long-range operator
and **keeps** it: `df/df.py:245` stores it in `self._rsh_df[key]` and both
objects live until the SCF ends.  So a range-separated density-fitted run
holds two cderi tensors of the same size at the same time — at 284 atoms
that is **two 86.41 GB host tensors, 173 of the node's 251 GB**, and nothing
warns.  §3.2 sizes the tensor; this entry is about there being two of them.

It is a capacity item rather than a speed one, and it is the reason W12's
pool does nothing for a range-separated functional: there is no released
buffer to hand over.

* **Do**: find out whether the second tensor has to be a second *tensor*.
  The long-range build differs from the full-range one only in `omega`, and
  `electrolites/_dfj2c.py` already shows that the two share an
  `Int2c2eOpt` exactly; whether they can share storage is a different
  question and needs the three-centre integrals looked at, not assumed.
  Failing that, the honest minimum is that `DF.build` should *say* how much
  host memory the pair will take before it takes it.
* **Success**: a wB97X-D3 DF SCF at 284 atoms that does not need 173 GB of
  host memory, or that says it does.
* **Falsified if**: the two tensors genuinely share no structure, in which
  case the item is a warning message and nothing more.

### Not on this list

**Seminumerical exchange (sn-K / COSX)**, which would avoid the cderi
entirely, is greenfield: GPU4PySCF 1.8.1 has no such module (`gpu4pyscf/sem`
is semiempirical methods — 1c2e/2c2e integrals and MNDO-family parameters —
not seminumerical exchange).  Worth knowing before someone goes looking for it.

---

## 5. Measurement rules for this path

Four things poisoned numbers in the round that produced this document.  All
four are cheap to avoid and none is detectable from the number itself.

1. **The login node is unusable.**  The same call measured 42 ms and 26 ms an
   hour apart on a shared login GPU.  Use `--qos=debug` and take the whole
   node.
2. **Time variants round robin, not one after the other.**  `df_jk_bench.py`
   does this; a competing job's load drifts over the minutes a row takes and a
   sequential schedule puts all of that drift on one variant.
3. **The first process of a Slurm job pays a cold start.**  The same unpatched
   DF-B3LYP run measured **10.803 s** as a job's first process and **4.054 s**
   as its seventh — a stock-then-patched ordering inside one job would read
   that as 2.66x.  Run every end-to-end pair **twice** and quote the second.
4. **Do not install a timer on the `DF` instance.**  `DF.range_coulomb` hands
   a range-separated build a `self.copy()`, and an instance-level wrapper is
   copied with it while still closed over the *original* object's bound
   method — so the copy's `build()` fills the original's `_cderi` and the
   copy's stays `None`.  Every wB97M-V DF run died in `get_jk` with
   `'NoneType' object is not subscriptable`, patched and unpatched alike.
   Wrap the class.

5. **Take the whole node and then hide three of its GPUs.**  `--qos=debug`
   gives four, `multi_gpu.num_devices` is then 4, and every module in
   `electrolites/` asserts a single device and hands every call back — a whole
   job reads as "no change" with nothing else wrong, and `stats()` says
   `declined_reasons: {'more than one device': N}`.  `export
   CUDA_VISIBLE_DEVICES=0` in the job script is the fix
   (`docs/OPENSHELL_TMC.md` §5 records the same trap).  It also changes what
   is being measured: on four devices GPU4PySCF splits the cderi, so a
   taxol/def2-SVP tensor reads 3.58 GB per device rather than the 14.3 GB
   §3.2 quotes.
6. **The debug nodes are not all the same card.**  The rows in
   `docs/README_fastdfblock.md` were taken on A100-SXM4-**80GB** nodes, where
   `fastdf._blksize` chooses 241 auxiliary functions at 284 atoms against the
   128 §3.3 records on a 40 GB one.  Fix the block size explicitly in any
   comparison that is meant to be read next to an older table.

7. **A kernel benchmark cannot see an allocator.**  W5's
   `df_block_bench.py` reuses one device-resident block, so its timed region
   allocates nothing; the real `get_jk` allocated a 763 MB temporary per
   block.  The benchmark read 1.09x and the call read 0.90x, and no amount of
   re-timing either would have found the difference.  When a kernel ratio and
   a call ratio disagree, profile — `benchmarks/df_block_profile.py` is the
   NVTX harness for this path and `docs/README_fastdfblock.md` §4.1 is what
   it found.
8. **`cudaFree` is a synchronisation point.**  A per-iteration temporary in a
   loop that relies on a prefetch does not cost its own allocation time, it
   costs the overlap: 114 of them drained GPU4PySCF's double-buffered stream
   114 times and left 2.33 s of an 11.73 s call idle.  Anything allocated
   inside a block loop belongs outside it.
9. **A cached buffer must not be sized from free memory.**  `_blksize` derives
   the auxiliary block from `get_avail_mem()` on every call, so a cached
   buffer sized to that block is reallocated whenever a neighbouring variant's
   footprint moves.  `_tiles` now fixes the block at the first call and
   `get_jk` takes it back.  (This looked like the explanation for the 0.90x
   and was not — it made that measurement *slower*.  It is kept because it is
   right, not because it paid.)

10. **Two tensors of the same size do not fit on one card, and the loser goes
   to the host silently.**  A converged `_DFHF` object holds a 14.3 GB cderi
   at 113 atoms; building a second tensor of the same size in the same
   process puts *it* on the host, and nothing warns.  Measured, that turned a
   1.2x difference into a 6.4x one.  Run one variant per process and pass the
   *same* converged density to both (`benchmarks/df_local_k_bench.py --only`,
   `--dm-cache`), and print where each tensor actually lives.
11. **`fill_symmetric` allocates its output if you let it.**  Without `out=`
   it makes a `(nao, nao, nL)` zeroed buffer per call: **57 %** of a 113-atom
   exchange pass, against a written volume that is 40 ms of the card's
   bandwidth.  This is §5.8 in a second costume — the allocation is not the
   cost, the `cudaFree` that ends it is.

12. **A module that imports another module's kernels imports its patches.**
   `electrolites/dflocal.py` wanted three kernels out of
   `electrolites/fastdfcderi.py`; importing that module patches
   `_cholesky_eri`, so a benchmark that imports the first silently measures
   the *patched* build in its "stock" column.  The symptom is a patched and
   an unpatched column agreeing to three digits.  Shared kernels belong in a
   module that patches nothing — `electrolites/_dfaux.py` is that.

8. **Nsight Systems cannot write its report onto `$HOME`.**  `nsys profile`
   runs, produces the `.qdstrm`, and then fails the import with
   `errno 524` on the GPFS global home; the profile is not lost but it is not
   readable either.  Point `-o` at `$SCRATCH`, or recover an existing
   `.qdstrm` with `Nsight_Systems/host-linux-x64/QdstrmImporter -i` from a
   directory on Lustre.

13. **A benchmark that imports a module imports its patches -- including its
   own.**  §5.12 is about one module importing *another*'s kernels;
   `benchmarks/df_j3c_probe.py` did it to itself.  It grew a module-level
   `from electrolites import fastdfj3c` so that a patched run could charge the
   Hermite transforms to the right row, and that turned every "GPU4PySCF's
   own" column it printed into a patched one -- silently, because the patched
   numbers are plausible.  The import belongs inside the `--patch` branch.  The
   symptom, again, is a stock column that agrees with the patched one.

14. **A round robin is not enough on the host branch of the cderi build.**
   §5.2's rule — time the variants round robin rather than one after the
   other — stops a *drift over minutes* from landing on one variant.  It does
   not stop a drift within one round, and there is one:
   `benchmarks/df_cderi_bench.py` on a host-resident taxol build reads
   **1479–1499 ms** for its `all` column and **2218–2238 ms** for the same
   configuration repeated in the last position, in two separate jobs.  Nothing
   about the columns between them explains it, and it is not the pool (the
   counters say every one of those columns reused the same buffer).  The
   consequence is practical: on that branch a difference of a few hundred
   milliseconds between two adjacent columns is not an item, and W11 — which
   is worth exactly that much — had to be priced on the **block loop**, which
   `electrolites/fastdfcderi.py` now times for itself and reports as
   `stats()['loop_s']`.  Add a repeat of the column you care about in the last
   position before believing any small difference here.

15. **`int2c2e` is replaced at the module level, so the "unpatched"
   reference is not unpatched.**  W13 replaces
   `df/int3c2e_bdiv.py:int2c2e`, and the reference every test and benchmark
   here uses — `DFMOD._cholesky_eri` saved before patching — looks that
   function up on the module at *call* time.  Both `tests/test_dfcderi.py`
   and `benchmarks/df_cderi_bench.py` now force `FASTDFCDERI_J2C=stock`, and
   every other switch back to GPU4PySCF's own value, around the reference.
   This is §5.13 in a third costume and the symptom is the same: a stock
   column that agrees with the patched one.

And one that is not about timing: **a patch target is not a code path.**
Three rounds in this repository have now shipped a module that was loaded,
correct, and never called — `dft/uks.py`, `hessian/uks.py`, and now
`df_jk._DFHF.get_veff`.  The only thing that has ever found one is
`benchmarks/coverage_probe.py`.  Run it, with `--density-fit`, before
believing any coverage claim in this document or in `README.md`.

---

## 6. Tools

```bash
python tests/test_df.py --ndm 1,2,4          # the correctness bar
python benchmarks/df_fill_probe.py --basis def2-svp     # §3.2, no GPU time
python benchmarks/df_anatomy.py   --xyz benchmarks/molecules/taxol.xyz \
    --basis def2-svp [--patch all] [--cpu-cderi]        # §3.1
python benchmarks/df_probe.py     --xyz ... --cycles 6  # §2's roofline table
python benchmarks/df_jk_bench.py  --xyz ... --ndm 1,2,4 \
    --blksizes 64,128,256,512 [--cpu-cderi --resident-fracs 0,0.25,0.5,1.0]
python benchmarks/coverage_probe.py --xyz ... --density-fit [--only-dfj]
python tests/test_dfj.py [--basis def2-tzvpp]           # W1's correctness bar
python benchmarks/dfj_probe.py    --xyz ... --scf --scf-ablate   # W1

python tests/test_dfgrad.py [--basis def2-tzvpp] [--no-fd]       # W3 and W10
python benchmarks/df_grad_probe.py --xyz ... --whole             # W3's profile
python benchmarks/df_grad_bench.py --xyz ... --whole --with-rebuild
python benchmarks/df_grad_ip1_sweep.py --xyz ...   # the launch configuration
python benchmarks/df_grad_metric_probe.py --xyz ... [--omega 0.3]

python benchmarks/df_grad_oo_sweep.py --xyz ... --reps 2 --extras   # W10
python benchmarks/df_grad_oo_sweep.py --xyz ... --stores device,host \
    --slices auto --ballast 0,8,14,17,19,20.5,21.5,22,23,24,24.5
    # the threshold, measured by withholding memory rather than guessed
python benchmarks/df_grad_bench.py --xyz benchmarks/molecules/PfPMT.xyz \
    --charge -1 --basis def2-svp --cycles 1 --reps 1 \
    --variants stock,fastdfgrad --with-rebuild      # the 284-atom gradient
python benchmarks/df_grad_bench.py --xyz ... \
    --variants stock,fastdfgrad,'old sizing','old fill buf'
    # the two ablations that account for W10 being faster than W3's table

python tests/test_dfcderi.py [--basis def2-tzvpp]        # W4, W11, W12, W13
python benchmarks/df_cderi_probe.py --xyz ... [--cpu-cderi] [--no-check]
python benchmarks/df_cderi_probe.py --xyz ... --skip-metric --sp-mults 4 \
    --trmm-ladder 256,512,1024,2048,3072,5598   # where dtrsm stops paying
python benchmarks/df_cderi_bench.py  --xyz ... [--cpu-cderi] [--scf]
python benchmarks/df_cderi_bench.py  --xyz ... --cpu-cderi --steps 3
    # W12: `DF.build()` again and again, as an optimisation step loop

python benchmarks/df_j2c_probe.py --cases glycine,tmc_pt_amine,penicillinV
    # W13: `_decompose_j2c` taken apart, and the host factorisation rejected
python benchmarks/df_store_probe.py --naux 5598 --blk 45683 --nblk 7 \
    --threads 1,4,16,32 --compute-ms 250 --pool
    # W11 and W12: the store's rates, the thread sweep that says there is no
    # host loop at all, the overlap, and the page-locked buffer
```

```bash
python benchmarks/df_block_probe.py --breakdown --hist   # W5's pattern, no GPU
python benchmarks/df_block_probe.py --basis def2-tzvpp   # why the gate is fill
python benchmarks/df_block_bench.py --check --cases taxol:0:0,PfPMT:0:-1
python benchmarks/df_jk_bench.py --xyz ... --tiled 32 --tile-orders native
nsys profile -t cuda,nvtx -c cudaProfilerApi -o $SCRATCH/p \
    python benchmarks/df_block_profile.py --xyz ... --cpu-cderi --tile 16
python benchmarks/df_block_profile.py --stats $SCRATCH/p.nsys-rep  # the reports

python benchmarks/df_local_probe.py --domains pair,pair+3.0   # W6, no GPU
python benchmarks/df_local_accuracy.py --cases c1,c2,c3,c4    # W6's energies

python tests/test_dflocal.py                             # W6's bar
python benchmarks/df_local_k_probe.py                    # nlambda, no GPU
python benchmarks/df_local_k_probe.py --lam --cases taxol:0:0
python benchmarks/df_local_k_probe.py --lam-fit --cases PfPMT:0:-1,HcgC:0:0
python benchmarks/df_local_k_probe.py --residual --residual-k --cases taxol:0:0
python benchmarks/df_local_k_bench.py --no-jk --scf --alkanes 1,2,4,8,16,20
python benchmarks/df_local_k_bench.py --only pari --anatomy \
    --dm-cache $SCRATCH/dm --cases taxol:0:0     # one side per process, §5.10
```

```bash
python tests/test_dffp32.py [--basis def2-tzvpp] [--no-scf]   # W7's bars
python benchmarks/df_fp32_probe.py --cases glycine,penicillinV --exact
python benchmarks/df_fp32_probe.py --alkanes 1,2,4,8,16,24    # does it grow
python benchmarks/df_fp32_probe.py --cases tmc_fecl4:-1:5 --exact
python benchmarks/df_fp32_bench.py --xyz ... --dm-cache $SCRATCH/fp32dm \
    [--cpu-cderi --resident-frac 0] [--scf] [--pcie] [--loop-ablate]
python benchmarks/df_fp32_bench.py --xyz ... --only fp32 --stages loop,jk,j \
    --guess-dm --cpu-cderi        # the sizes where one tensor is the budget
```

```bash
python tests/test_dfj3c.py [--basis def2-tzvpp] [--no-scf]        # W9's bar
CUDA_VISIBLE_DEVICES=0,1 python tests/test_dfj3c.py --no-scf  # the device key
python benchmarks/df_j3c_probe.py --cases glycine,penicillinV,taxol  # W9's §3
python benchmarks/df_j3c_probe.py --cases penicillinV --shapes    # no GPU time
python benchmarks/df_j3c_probe.py --cases ... --patch fastdfj3c   # the after
python benchmarks/df_j3c_bench.py --cases ... [--ndm 1,2,4] [--scf] [--patch ...]
python benchmarks/df_j3c_sweep.py --cases ...     # the block size, and the two
python benchmarks/df_j3c_sweep.py --cases ... --shm-fracs 1.0,0.5 --per-class \
    --ksh-chunks 4,8,16,32,64                     # ideas W9 measured and rejected
python benchmarks/coverage_probe.py --xyz ... --xc pbe --density-fit
    # `fastdf` reads 0 on a pure functional and `fastdfj3c` one call per cycle
```

`dflocal`'s switches (`DFLOCAL_DOMAIN`, `DFLOCAL_STORE`, `DFLOCAL_BLK`,
`DFLOCAL_LAMBDA_TOL`, `DFLOCAL_KMODE`, `DFLOCAL_FOLD`) are listed at the top
of `electrolites/dflocal.py`.  It is **not** in `patch_all()` and
`dflocal.density_fit(mf)` is not `mf.density_fit()`: it is a different
approximation and says so once, on stdout.

`fastdfcderi`'s switches (`FASTDFCDERI_OFF`, `FASTDFCDERI_METRIC`,
`FASTDFCDERI_SPMULT`, `FASTDFCDERI_MIN_WORK`, `FASTDFCDERI_PIN`,
`FASTDFCDERI_HUGEPAGE`) are listed at the top of
`electrolites/fastdfcderi.py` and are what every ablation column in
`docs/README_fastdfcderi.md` is measured with.

`FASTDF_FP32=1` is W7's, and it is the one switch here that changes the
answer: `electrolites/_dffp32.py` owns it, `electrolites/fastdffp32.py`
carries the two consumers that are not `get_jk`, and
`docs/README_fastdffp32.md` is what it costs and what it buys.  With it set
there is no fall-through — a call `fastdf` declines raises, because
GPU4PySCF's own build would read the narrow tensor as a wide one.

`fastdfj3c`'s switches (`FASTDFJ3C_OFF`, `FASTDFJ3C_NO_CTR`,
`FASTDFJ3C_NO_OPT`, `FASTDFJ3C_NO_MDM`, `FASTDFJ3C_SP_CHUNK_AUX`,
`FASTDFJ3C_SP_CHUNK_DM`, `FASTDFJ3C_KSH_CHUNK`, `FASTDFJ3C_MDM_MAX_NDM`,
`FASTDFJ3C_TARGET_BLOCKS`, `FASTDFJ3C_CHUNK_MIN`, `FASTDFJ3C_CHUNK_MAX`,
`FASTDFJ3C_SHM_FRAC`, `FASTDFJ3C_PER_CLASS`) are listed with their meanings at
the top of `electrolites/fastdfj3c.py`.  The last two are the two ideas
`docs/README_fastdfj3c.md` §8 measured and rejected, kept runnable rather than
described.

`fastdfgrad`'s switches (`FASTDFGRAD_OFF`, `FASTDFGRAD_CDERI`,
`FASTDFGRAD_JPAIR`, `FASTDFGRAD_SPMULT`, `FASTDFGRAD_GOUT`,
`FASTDFGRAD_INTOPT`, and W10's `FASTDFGRAD_OOSTORE`, `FASTDFGRAD_OOFRAC`,
`FASTDFGRAD_OOSLICE`, `FASTDFGRAD_OOPREFETCH`, `FASTDFGRAD_OOPIN`,
`FASTDFGRAD_J3CBATCH`, `FASTDFGRAD_OOSIZING`, `FASTDFGRAD_FILLBUF`,
`FASTDFGRAD_BLK`) are
listed with their meanings at the top of `electrolites/fastdfgrad.py`.  The
last two are ablations rather than settings: they put back what W3's code did
and are how `docs/README_fastdfgradoo.md` §5 accounts for the difference
between the two rounds.

`fastdf`'s switches (`FASTDF_OFF`, `FASTDF_STEP1`, `FASTDF_STEP2`,
`FASTDF_JPASS`, `FASTDF_SYRK_MIN_NAO`, `FASTDF_BLK`, `FASTDF_RESIDENT_FRAC`,
and W5's `FASTDF_TILE`, `FASTDF_TILE_ORDER`, `FASTDF_TILE_MAX_FILL`)
are listed with their meanings at the top of `electrolites/fastdf.py` and are
what every ablation column in `docs/README_fastdf.md` is measured with.
