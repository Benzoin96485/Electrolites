# The density-fitting path: what is done, what was rejected, and what is left

A work list for whoever picks this up next.  It is written to be read without
the session that produced it: every number in it is one measured in this
repository, with the command that produces it again, and every dead end is
recorded with the measurement that killed it so nobody spends a day
rediscovering it.

Companion documents: `docs/README_fastdf.md` is the round that produced this
list (what changed, why, and the full tables); `docs/README_fastdfj.md` is the
round that closed **W1**, which is where the biggest number on this page now
is; `docs/CATALOGUE_06_STATUS.md` §2 row **A13** is where this sits in
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
calculation builds and runs.  GPU4PySCF's README limit ("~168 atoms with
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
> and W4 are untouched by W1.  W9 is worth *less*: §3.1 of
> `docs/README_fastdfj.md` measures the three-centre Coulomb engine at 1.67x
> the direct four-centre build at 284 atoms, so the path it names is already
> the faster of the two.

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

### W3 — The DF gradient's two-electron derivative

`df/grad/rhf.py:_jk_energy_per_atom`: **83 %** of a def2-TZVPP DF gradient,
29 % at def2-SVP — the one uncovered stage that grows with basis size instead
of shrinking.  Nothing in this repository is on it.

It has the same shape as the SCF build and two of the same problems, plus one
the SCF build does not have: the dense `(i,j,L)` expansion is built by **two
fancy-index scatters** (`j3c[j_addr,i_addr] = j3c[i_addr,j_addr] = ...`)
rather than by a kernel, and it happens twice — once to form `j3c_oo` and once
to compress the pseudo-density back onto the pair list with `cp.take`.

* **Do**: profile it first, stage by stage, the way
  `benchmarks/df_probe.py` does the SCF build.  §2's lesson applies: do not
  assume the contraction is the cost.
* **Success**: a stage table with each stage against its roofline, then
  whatever that table points at.
* **Note**: unlike the SCF build, here the `(L,i,j)` layout *would* turn both
  contraction pairs into clean batched GEMMs — §2's rejection of that layout
  was measured on the SCF build's unpack, which is a different kernel and a
  different input layout.  Re-measure, do not inherit the conclusion.

### W4 — The cderi build

`df/df.py:_cholesky_eri` is **39.4 %** of a 15-cycle taxol DF SCF and 12–23 %
of the shorter ones.  Untouched.

One concrete lead: the metric contraction `j3c.dot(aux_coef)` forms `L^{-1}`
explicitly (`_decompose_j2c` solves the triangular system against the
contraction coefficients) and then multiplies by the dense result — a dense
`npair × naux_sorted × naux` GEMM.  A triangular solve is half the arithmetic,
and the auxiliary contraction coefficients it is folded with are **0.06 %
dense** (measured: 3 412 non-zeros in a 2 590 × 2 191 matrix), so the two can
be separated cheaply.

* **Do**: `j3c · C` as a sparse gather, then `trsm` against the Cholesky
  factor, instead of one dense GEMM against `C · L^{-T}`.
* **Success**: measure `_cholesky_eri` before and after on taxol.
* **Falsified if**: `dtrsm` at these shapes runs at less than half of `dgemm`,
  which would eat the factor of two.  cuBLAS's triangular solve is not as well
  tuned as its GEMM — the same trap `dsyrk` fell into in §1 item 1.

### W5 — A block-sparse exchange contraction

*(§3.3 raised this one too, and it is now the largest lever on DF-K itself.)*

The prize is §3.2's fill: **4.5x** the arithmetic at 284 atoms, growing with
system size.  §3.3 measures the two contractions at **12.1 s of a 12.38 s** K
build there, so unlike at 42 atoms there is nothing else left to blame.  It
also removes the dense `(nao, nao, blksize)` work buffer, which is 7 GB at
PfPMT's measured `blksize` of 128 and 14 GB at 256.

This is a real project, not a rearrangement: the pattern has to be built from
the shell-pair list, the unpack has to write tiles rather than a dense block,
and the second contraction (`vk = R Rᵀ`) stays dense either way — so the
ceiling is a bit under half the build, times the fill factor.

* **Do**: AO tiles (32 or 64) rather than shell pairs, so the block GEMMs are
  a usable shape; a tile-pair pattern computed once per `dfobj`; a tiled
  unpack; then either `cuSPARSE` BSRMM per auxiliary function or a custom
  batched GEMM over present tile pairs.
* **Success**: step 1 faster than the dense contraction at 113 atoms *and*
  at 284, with the crossover reported.
* **Falsified if**: the tile GEMMs at 50 % fill (taxol) are slower than one
  dense contraction at 15.8 TF/s — likely, which is why the acceptance
  criterion names two sizes.  This may be a large-system-only win.

### W6 — Local / pair-atomic fitting

**The only item on this list that changes the storage exponent** — but §3.3
demoted it: at 284 atoms on a 251 GB node capacity is not what binds, speed
is.  Its value here is that `naux_local` also shrinks the *contractions*, not
only the tensor.  It becomes urgent again at 500+ atoms or on a node with less
DRAM.  Fit each AO
pair in the auxiliary functions on its own two atoms (plus near neighbours)
instead of all of them: storage goes from `npair × naux` to
`npair × naux_local` with `naux_local` independent of system size.  On
PfPMT's 808 156 pairs at, say, 300 auxiliary functions per pair that is
**1.9 GB against 86.4 GB**.

* **It is a different approximation, not a faster route to the same number.**
  The fitting error is not the global one, some variants are non-variational,
  and GPU4PySCF has none of it.  Anything built here must be benchmarked on
  *energies* against global DF, not only on wall clocks.
* **Do**: read the PARI / local-RI literature before writing anything; decide
  robust vs non-robust fitting first, because it changes the whole data
  structure.
* **Success**: a total energy within a stated tolerance of global DF on the
  §3.2 ladder, and a storage curve that is visibly sub-quadratic.

### W7 — fp32 cderi storage

Exactly 2x on storage and 2x on PCIe, and the arithmetic can stay fp64 by
accumulating in double.  Technically the least work on this list.

**It changes the answer**, and this repository's correctness tables are at
1e-14.  So it is a measured, explicitly-labelled switch or it is nothing:
report the error it introduces in the total energy and in the exchange matrix
on the §3.2 ladder, default it off, and say so in `README.md`'s limits
paragraph.

### W8 — The DF Hessian's own CPHF build

`df/hessian/rhf.py:_get_jk` is a **fifth** two-electron code path: it builds J
and K directly in the MO basis and does not go through `df_jk.get_jk`, so
`fastdf` is not on it.  Same `(i,j,L)` expansion, same cuTENSOR contractions.

It also moves the cderi to the host mid-run on purpose (`:1258-1273`, "Leave
space for storing Krylov subspace vectors") — so `fastdf`'s item 4 would take
back half of what it just freed.  Anyone working here should know that
`FASTDF_RESIDENT_FRAC=0` is the setting for a DF Hessian until something
smarter exists.

### W9 — The pure-functional DF Coulomb build

`df/j_engine_3c2e.py`, a McMurchie–Davidson three-centre kernel in
`libvhf_md`.  A DF-PBE run spends its two-electron time here and reaches
nothing in this repository.  It is a `fastj`-shaped generator project rather
than a driver rewrite, and W1 makes it matter for hybrids too.

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
```

`fastdf`'s switches (`FASTDF_OFF`, `FASTDF_STEP1`, `FASTDF_STEP2`,
`FASTDF_JPASS`, `FASTDF_SYRK_MIN_NAO`, `FASTDF_BLK`, `FASTDF_RESIDENT_FRAC`)
are listed with their meanings at the top of `electrolites/fastdf.py` and are
what every ablation column in `docs/README_fastdf.md` is measured with.
