# fastdf — the density-fitted J and K build, and a fourth code path

Catalogue entry **A13**, "the density-fitting (RI) path".  Its status in
`docs/CATALOGUE_06_STATUS.md` has been **open** since the repository started,
with one line of justification: *"Density fitting is untouched by every module
here."*  That is exactly right and it is worth saying what it means.  A
`density_fit()` run does not take any of the three two-electron code paths
`fastj`, `fastk` and `fastjk` are on.  It takes a fourth:

| caller | entry point | what it computes | patched by |
|---|---|---|---|
| direct SCF (`scf/hf.py:get_veff`) | `j_engine._VHFOpt.get_j`, `jk._VHFOpt.get_k` | four-centre J, four-centre K | `fastj`, `fastk` |
| Hessian CPHF | the same two, at `n_dm = 3·natm` | " | `fastj`, `fastk` |
| every response property | `jk._VHFOpt.get_jk` | fused four-centre J+K | `fastjk` |
| **every DF/RI calculation** | **`df/df_jk.py:get_jk`** | **J and K from the three-index cderi tensor** | **`fastdf`** |

`df/df.py:117` is the only caller of that last function, and it looks it up on
the module, so one patch target covers the DF SCF, the DF response solve and a
DF Hessian's CPHF at once.  Everything upstream of it — the three-centre
integrals, the metric decomposition, the screening — is GPU4PySCF's and stays
GPU4PySCF's.

---

## 0. Summary

* `fastdf` replaces `df/df_jk.py:get_jk`, the build **every** RI/DF
  calculation's two-electron half goes through -- the SCF, the response solve
  and a DF Hessian's CPHF.  Restricted or unrestricted, `hermi` 0 or 1, any
  number of densities, the cderi on the device or on the host, full-range or
  long-range.
* With the cderi **on the device**: **1.09x-1.41x** where it applies and
  1.00x-1.01x where it does not, and which of those a row gets is decided by a
  measured size threshold rather than by hope (§3.2).
* With the cderi **on the host** -- the regime catalogue A13 is about --
  **1.64x-4.53x**, and what it is worth is close to the fraction of the tensor
  that fits on the device (§3.3).  Whole SCF runs: **1.15x-1.22x** there and
  **1.03x-1.12x** with the tensor on the device (§3.5).
* **Two of A13's four pieces of evidence were closed upstream** in 1.8.x and a
  third is wrong on 1.8.1 (§1).  What is left of the entry is the two
  contractions it does not mention.
* **The build is already near the machine's limits**: 71 % of memory bandwidth
  on the unpack and 15.8/17.8 TF/s of an A100's 19.5 on the two contractions
  (§3.4).  That is the opposite of what the direct-SCF rounds found, and it is
  why this round is worth 1.1x and not 3x.  Three rearrangements were written,
  measured and rejected, one of them a CUDA kernel (§3.4).
* **The one structural saving left is to make the arithmetic smaller.**  The
  screening stores **22 %** of the AO-pair space at 284 atoms and the
  contraction is dense over all of it (§3.6).
* **The largest result of the round is not in this module.**
  `density_fit()` puts `_DFHF.get_veff` ahead of `RKS.get_veff` and
  `UKS.get_veff` in the method resolution order, so `fastxcnlc` was inert on
  **every** density-fitted run and a DF wB97M-V walked the grid twice per
  iteration.  Patching that third binding as well takes a DF wB97M-V SCF on penicillin V
  from 16.0 s to **14.2 s (1.13x)**, and the whole patched set from 22.0 s to
  14.2 s (**1.55x**) -- `nr_nlc_vxc` is 77-80 % of such a run unpatched, and
  `fastnlc` was already reaching part of it (§3.5, §5).
* `benchmarks/df_anatomy.py` also found and this round fixed a *measurement*
  bug of its own: a timer installed on the `DF` **instance** is copied by
  `range_coulomb`, so every range-separated DF run died in `get_jk` with
  `'NoneType' object is not subscriptable` -- patched and unpatched alike.
  Wrap the class.

---

## 1. What is left of catalogue A13 on 1.8.1

The entry's evidence is a **1.7.0** reading (catalogue §7 Q1 says so of the
whole document), and `df/df.py` was largely rewritten in the interval.  Every
one of its four line references points somewhere else in 1.8.1.  Taking them
in order:

| A13's claim | 1.7.0 evidence | on 1.8.1 |
|---|---|---|
| `df/df.py:105` `j2c_cpu = auxmol.intor('int2c2e')` — the auxiliary 2-centre integrals on the **CPU**, "a GPU version exists in `scf/int2c2e.py`" | the line exists | **closed upstream.** `_decompose_j2c` (`df/df.py:602`) calls `int3c2e_bdiv.int2c2e`, which forwards to `pbc/df/int2c2e.py`'s GPU kernel.  `auxmol.intor` appears nowhere in 1.8.1's `df/df.py` |
| `df/df.py:200` a `cupy.cuda.Device().synchronize()` after every `yield`, "turning the `buf_prefetch` above it into a blocking copy" | the line exists | **closed upstream.** `DF.loop`'s host-resident branch (`df/df.py:187-217`) is a two-stream double buffer with two events and no device-wide synchronise; line 200 is now `compute_event.record(comput_stream)` |
| `:195-196` "expand the packed cderi into `nao²` with two fancy-index scatters per block" | the lines exist | **half survives.**  The scatters are gone — 1.8.1 calls `fill_symmetric`, a CUDA kernel.  The **expansion** survives, and its output layout is what item 2 below is about |
| `df/df.py:145` "duplicates the whole cderi object for RSH" | the line exists | **overstated on 1.8.1.**  `range_coulomb` (`df/df.py:245`) makes `self.copy().reset()`, and `reset` sets `_cderi = None`, so nothing is duplicated.  A range-separated functional does end up with *two* tensors, each built separately, which is a real cost but not the one the entry names |

And the headline mechanism — *"the three-index tensor lives on the host and
crosses PCIe every iteration"* — is conditional on 1.8.1, not unconditional:
`_cholesky_eri` (`df/df.py:517-524`) puts the cderi on the **device** whenever
it fits and falls back to host memory only when it does not.  The README limit
the entry quotes ("~168 atoms with def2-tzvpd basis, bounded by CPU memory")
is the size at which that fallback fires.  Item 3 below is about that regime
and only that regime.

So of A13 as written, two claims were closed upstream, one is a layout
question rather than a scatter question, and one is wrong on 1.8.1.  What is
*left* of the entry is real, and it is the two contractions the entry does not
mention at all.

---

## 2. What the module changes

Only the arrangement of the work changes.  The cderi tensor, the metric, the
screening, the pair list and every symmetry factor are GPU4PySCF's, which is
why §4's correctness numbers are at round-off rather than at a tolerance.

### 2.1 The exchange contraction is a rank-k update, not a GEMM

GPU4PySCF builds K in two contractions per auxiliary block (`df_jk.py:536-541`):

```python
contract('Lij,njk->nikL', cderi, factor_l, out=rhok)     # step 1
contract('nikL,njkL->nij', rhok, rhok,  beta=1, out=vk)  # step 2
```

Step 2 contracts `rhok` **with itself**.  Its result is therefore symmetric by
construction, and the general GEMM computes both of its triangles.  Every
ground-state density reaches this branch: `factorize_dm` returns a single
factor and no second one whenever the density carries `mo_coeff`, which an SCF
density, a UKS pair of densities and every `mo_coeff`-tagged perturbed density
do.  A `dsyrk` is the same answer for half the flops.

No layout change is needed for it: `rhok` is `(n, nao, nocc, nL)`, so for each
density the `(k,L)` pair is already the contiguous trailing block that the
update contracts over, and `rhok[n]` is a plain `(nao, nocc*nL)` row-major
matrix.  The two triangles are recombined once per call, for `nao²` work.

An **untagged** density — the `hermi = 0` densities a TDDFT build produces —
factorises into two different tensors and its K is not symmetric.  That branch
keeps a general GEMM, and is checked separately in §4.

### 2.2 The first contraction is left where it is

This is the one the layout argument points at, and it is the one that does not
move.  `DF.loop` unpacks the packed cderi into a `(nao, nao, blk)` buffer and
yields `out.transpose(2,0,1)` (`df/df.py:183-185`).  That view has strides
`(1, nao·nL, nL)`: the auxiliary index is the *innermost* one, so in step 1's
batch over `L` the matrix `cderi[L]` has a unit stride in **neither**
dimension and neither does the output slice.  No cuBLAS call can express that,
and cuTENSOR serves it with a general permuting contraction instead.

The objection is right and the fix does not pay.  Batching the same arithmetic
over **i** rather than over `L` makes every operand a proper matrix without
changing the layout at all: `work[i]` is a row-major `(j, L)` matrix, so

```
out[i,L,k] = sum_j work[i,j,L] · factor[j,k]
```

is one `dgemmStridedBatched` with `m = nocc`, `n = nL`, `k = nao`, `nao`
batches, and an output already in the `(i, L, k)` order §2.1's rank-k update
contracts over — no transpose, no second unpack.  It is carried as
`FASTDF_STEP1=bgemm` and §3.4 measures it: **1.00x** at one density and
**slower** at 1007 basis functions, because `nao` batches of a `(nocc × nL)`
output is the small-matrix regime cuBLAS's batched path is weakest in while
cuTENSOR sees one large contraction and runs it at 15.8 TF/s.  At two
densities it is faster on three systems out of four, by 5–23 %, and slower on
the fourth; that is not a rule, so the default stays with cuTENSOR and the
switch is there to flip.

### 2.3 The Coulomb pass is an explicit GEMV

`auxvec[n,L] = sum_p dm[n,p] cderi[L,p]` and then
`vj[n,p] += sum_L auxvec[n,L] cderi[L,p]` are the whole Coulomb build,
and the packed block is a plain matrix in both of the layouts it arrives
in -- the auxiliary index is contiguous when the tensor lives on the
device (`_cholesky_eri` returns a transpose) and the pair index is when it
has been staged from the host.  So each pass is one `dgemv` at a single
density and one `dgemm` at several, in place of GPU4PySCF's
`contract('np,Lp->nL')` and `contract('nL,Lp->np')`.  Same arithmetic,
same operands, different summation order over the auxiliary index -- which
is why §4's Coulomb agreement is `3e-16` relative and not exactly zero.
It is worth **1.09x** on the whole build at one density and nothing at
two, where cuTENSOR already issues a GEMM (§3.2).

### 2.4 A cderi that does not fit is only partly streamed

Past the size at which `_cholesky_eri`'s test fails, GPU4PySCF keeps the whole
tensor in host memory and copies the whole tensor across PCIe on every SCF
cycle.  The all-or-nothing is an artefact of deciding once, for the whole
tensor, at build time.  This module keeps as many auxiliary rows resident on
the device as the free memory allows — once, on the first call, reused by
every later one — and streams only the remainder, with GPU4PySCF's
double-buffered prefetch kept for that remainder.  The traffic per cycle falls
by whatever fraction fits.

`FASTDF_RESIDENT_FRAC` is the share of free memory it may take (default 0.5);
`0` restores GPU4PySCF's behaviour exactly, which is how §3's `no-resident`
column is measured.

**One case wants `0`.**  A DF Hessian moves the cderi to the host *on
purpose* -- `df/hessian/rhf.py:1258-1273`, with the comment "Leave space for
storing Krylov subspace vectors" -- and this item would take half of what it
just freed back, on the response calls that follow.  Nothing in this module
can tell that context from an ordinary host-resident one, so the knob is the
answer and not a heuristic; a DF Hessian that runs out of memory with this
loaded should be re-run with `FASTDF_RESIDENT_FRAC=0` before anything else is
suspected.

---

## 3. What it is worth

All of §3 is measured on **one A100-SXM4-80GB with nothing else on it**
(`--qos=debug`, a whole node), GPU4PySCF 1.8.1, B3LYP, the basis set's own
jkfit auxiliary set.  Every row is a `benchmarks/df_jk_bench.py` row: both
implementations in **one process** on **one cderi tensor**, at a converged
density, and timed **round robin** rather than one after the other -- the
login node measured the same call at 42 ms and 26 ms an hour apart, and a
round-robin schedule is the only way a ratio survives that.

### 3.1 Where a density-fitted run's time goes

`benchmarks/df_anatomy.py`, penicillin V, DF-B3LYP, level-3 grids, 14 cycles,
unpatched, the cderi on the device:

| stage | def2-SVP (nao 430) | def2-TZVPP (nao 1007) |
|---|---|---|
| `with_df.get_jk` — the DF J and K build, of which | 27.2 % | 42.0 % |
| ↳ `with_df.build` — the cderi, **once** | 12.4 % | 15.5 % |
| ↳ the recurring build, 14 calls | 14.8 % | 26.5 % |
| `nr_rks` — the XC build, which `fastxc` already covers | 34.6 % | 28.1 % |
| `get_hcore` — the one-electron integrals, **once**, on the CPU | 21.8 % | 13.9 % |
| DIIS, the eigensolve, `make_rdm1`, the grids | 4.8 % | 4.4 % |

and, for the same molecule with the cderi forced onto the **host**, which is
what happens past the capacity threshold, the build's share roughly doubles:
`with_df.get_jk` 43.9 %, of which 22.8 % is the one-time cderi build and
21.1 % the fourteen recurring ones.

and of the **gradient** at def2-TZVPP, `df/grad/rhf.py`'s two-electron
derivative is **83.1 %** (3.34 s of 4.02 s) and the XC term 9.4 %.  At
def2-SVP the same two are 29 % and 52 %.

Three things follow.  A DF run is *not* the direct-SCF picture with J and K
swapped out: the grid half is as *big as or bigger than* the two-electron half
and this repository already had it, so this module's target is the 15–27 % the
recurring build is (21 % with the tensor on the host).  The **one-time** cderi
build is 12–23 % and nothing here touches it, which is most of why §3.5's end-to-end ratios are so much smaller
than §3.2's per-call ones on a 14-cycle run.  And the **gradient** inverts
with basis size: at def2-TZVPP five sixths of it is one uncovered function,
which makes `df/grad/rhf.py:_jk_energy_per_atom` the largest single DF target
left in a *gradient* (§5).  In an **SCF** that place belongs to the one-time
cderi build: §3.5 measures `_cholesky_eri` at 39.4 % of a 15-cycle taxol run.

### 3.2 The build, with the cderi on the device

| system | nao | naux | n_dm | GPU4PySCF | `fastdf` | |
|---|---|---|---|---|---|---|
| penicillin V/def2-SVP | 430 | 2191 | 1 | 18.03 ms | **16.46 ms** | **1.09x** |
| | | | 2 | 31.72 ms | 31.35 ms | 1.01x |
| | | | 4 | 48.39 ms | 48.11 ms | 1.01x |
| penicillin V/def2-TZVPP | 1007 | 2191 | 1 | 83.31 ms | **75.23 ms** | **1.11x** |
| | | | 2 | 131.47 ms | **111.58 ms** | **1.18x** |
| taxol/def2-SVP | 1123 | 5598 | 1 | 442.88 ms | **405.68 ms** | **1.09x** |
| | | | 2 | 1120.74 ms | **795.53 ms** | **1.41x** |
| Ni dithiolene/def2-SVP (**UKS**) | 327 | 1992 | 2 (spin) | 22.38 ms | 22.07 ms | 1.01x |

The ablation columns say which item each row is.  At 430 and 327 basis
functions `_use_syrk` keeps GPU4PySCF's GEMM, so item 1 contributes nothing
and what is left is the Coulomb pass: `j=tensor` measures 17.82 ms against
`fastdf`'s 16.46 at one density, which is the whole of that row's 1.09x, and
nothing at two densities where the pass is a GEMM rather than a GEMV in both.
At 1007 and 1123 functions item 1 is the row: forcing `step2=gemm` measures
79.24 ms against 75.23 at def2-TZVPP and 435.81 against 405.68 on taxol.

The 430-function rows are 1.00x on item 1 **by design, not by accident**.
Forced on there, `step2=syrk` measures 17.12 ms against the GEMM's 16.49 at
one density -- **0.96x** -- because cuBLAS's `dsyrk` is less well tuned than
its `dgemm` and below about a thousand functions gives the factor of two in
arithmetic back.  `SYRK_MIN_NAO` is where `_use_syrk` switches over and
`FASTDF_STEP2` forces either branch.

### 3.3 The same build with the cderi on the host

This is the regime catalogue A13 names, and it is where the round is worth
something.  `--cpu-cderi` forces it at sizes where the tensor would have fit,
so the *ratio* is what a system past the threshold sees and the *size* is not:

| system | nao | n_dm | GPU4PySCF | `fastdf` | | item 4 alone |
|---|---|---|---|---|---|---|
| penicillin V/def2-SVP | 430 | 1 | 56.46 ms | **15.76 ms** | **3.58x** | 51.62 -> 15.76 |
| | | 2 | 56.41 ms | **30.55 ms** | **1.85x** | 56.44 -> 30.55 |
| penicillin V/def2-TZVPP | 1007 | 1 | 297.01 ms | **65.52 ms** | **4.53x** | 267.85 -> 65.52 |
| taxol/def2-SVP | 1123 | 1 | 668.06 ms | **407.13 ms** | **1.64x** | 631.81 -> 407.13 |

The last column is the `no-resident` ablation: everything else on, the whole
tensor streamed.  It is within a few per cent of the stock column in every
row, which is the point -- items 1 and 2 are irrelevant to a build that is
waiting on PCIe, and item 4 is the whole of these numbers.

**What each resident row is worth**, at penicillin V/def2-TZVPP (a 5.5 GB
tensor), with `--resident-fracs` holding a fixed share of the *tensor* rather
than a share of the memory, so the curve is one molecule's:

| resident | time | |
|---|---|---|
| 0 % (what GPU4PySCF does) | 295.9 ms | 1.00x |
| 25 % | 243.7 ms | 1.21x |
| 50 % | 191.2 ms | 1.55x |
| 75 % | 111.1 ms | 2.66x |
| 100 % | 65.6 ms | 4.51x |

**A bound on all of that, measured after the fact.**  These numbers are what
item 4 is worth where the PCIe stream is *exposed*.  It is not always: at 284
atoms (PfPMT/def2-SVP, an 86.4 GB host-resident tensor) the transfer runs at
26.4 GB/s -- a saturated gen4 x16 link -- and the exchange contractions at
`nocc = 547` are long enough to hide **all** of it, so
`get_jk(J+K) = 12.52 s` against `K alone = 12.38 s` rather than the sum.  Item
4 buys nothing there, and a J-**only** call at the same size is the opposite
extreme: 3.28 s, which is exactly the transfer and nothing else.  So the rule
is not "the fraction that fits" on its own -- it is that, times how much of
the stream was not already hidden.  `docs/DF_WORK_LIST.md` §3.3 has the run.

Within the regime where it *is* exposed, the curve is not quite linear -- the
last quarter is worth more than the first -- but it is close enough that the
useful reading is the simple one: **what this is worth is the fraction of the
tensor that fits**.  A system just past
the capacity threshold gets most of it; one at three times the device's memory
gets a third of it.  `FASTDF_RESIDENT_FRAC` is the share of *free* memory the
prefix may take (default 0.5), and `0` restores GPU4PySCF's behaviour exactly,
which is how the `no-resident` column above is measured.

### 3.4 The things that were tried and do not pay

`benchmarks/df_probe.py` times one auxiliary block of the def2-TZVPP case
(`nao = 1007`, `nocc = 92`, `nL = 256`, nine blocks) stage by stage:

| stage | per block | what that is |
|---|---|---|
| unpack, GPU4PySCF's `fill_symmetric` | 1.35 ms | 1.93 GB moved = **1.43 TB/s**, 71 % of the card |
| step 1, `contract('Lij,njk->nikL')` | 3.02 ms | 4.78e10 flop = **15.8 TF/s** |
| step 2, `contract('nikL,njkL->nij')` | 2.68 ms | the same flop count = **17.8 TF/s** |
| step 2 as `dsyrk` | **1.74 ms** | half the flops, **1.54x** |
| step 1 as a cuBLAS batch over `i` | 3.60 ms | **0.84x** -- slower |

Two conclusions, and the second is the more useful one.

**The layout objection in §2.2 is correct and its fix does not pay.**  Step 1
really cannot be a cuBLAS call in GPU4PySCF's layout.  Batching over the AO
index instead makes it one -- and it is *slower*, because `nao` batches of a
`(nocc x nL)` output is the small-matrix regime cuBLAS's batched path is
weakest in, while cuTENSOR sees one large contraction and runs it at 15.8
TF/s.  Two further variants were measured and rejected with it: an `(L,i,j)`
unpack of our own, which at penicillin V/def2-SVP is **2.65x slower** than
GPU4PySCF's (148 GB/s against 393) because the transposed half of a symmetric
fill is coalesced in GPU4PySCF's layout and is not in that one; and a blocked
triangular GEMM in place of `dsyrk`, which at `n = 1123` measures 0.95x of the
full GEMM and only beats it at `n = 2000`.

**The build is already near the machine's limits stage by stage** -- 71 % of
memory bandwidth on the unpack, 15.8 and 17.8 TF/s against an A100's 19.5
TF/s fp64 peak on the two contractions.  That is the real finding of this
round and it is the opposite of what the direct-SCF rounds found: there is no
rearrangement left to collect.  The one structural saving is the factor of two
that a symmetric result does not need to compute twice, which item 1 takes;
everything beyond it has to make the arithmetic *smaller*, which §3.6 sizes.

One qualification on the cuBLAS step 1, because it is not a clean loss.  Where
the call carries **two** densities -- every UKS ground state, and every
closed-shell call a solve makes with a pair of right-hand sides -- it is
faster on four systems of five: the Ni dithiolene 17.9 ms against 22.1 and,
measured again in another job, 17.4 against 22.1; the Cu macrocycle 48.3
against 50.4; taxol 757 ms against 795; penicillin V/def2-SVP 27.6 ms against
31.4.  It is slower on the fifth (penicillin V/def2-TZVPP, 122 ms against
112).  The two systems it helps most are the **open-shell transition-metal**
ones, which is the chemical class `docs/OPENSHELL_TMC.md` says 83 % of a
real transition-metal set belongs to -- so a rule keyed on the density count
would help exactly where it matters and regress the one large-basis row.  Three of four at spreads of 5-23 % with run-to-run
variation of about 5 % is not a rule, so the default stays with cuTENSOR;
`FASTDF_STEP1=bgemm` is there because a rejected alternative that can still be
run is worth more than one described in a document.

---

### 3.5 End to end, and why the ratios are so much smaller

`benchmarks/df_anatomy.py`, whole SCF wall clocks, penicillin V/B3LYP,
level-3 grids, 14 cycles.  Each triple is run **twice** and the second
repetition is the one quoted: the first process of a Slurm job pays a cold
start that has nothing to do with the patch and everything to do with the
filesystem -- the same unpatched def2-SVP run measured **10.803 s** as the
job's first process and **4.054 s** as its seventh.  A stock-then-patched
ordering inside one job would have read that as 2.66x.

| run | GPU4PySCF | + `fastdf` | | + `patch_all()` | |
|---|---|---|---|---|---|
| def2-SVP, cderi on the device | 4.054 s | 3.951 s | 1.03x | **3.621 s** | **1.12x** |
| def2-TZVPP, cderi on the device | 6.172 s | **5.788 s** | **1.07x** | 6.237 s | 0.99x |
| def2-SVP, cderi on the **host** | 4.568 s | — | — | **3.731 s** | **1.22x** |
| taxol/def2-SVP, cderi on the **host**, 15 cycles | 25.750 s | — | — | **22.394 s** | **1.15x** |

Three things to read off that, and the third is a correction to this
repository's own habit.

**The recurring build gets what §3.2 and §3.3 say it gets** -- 1.19x at
def2-TZVPP (1.638 s of stock against 1.373 s), 1.74x on the host-resident
def2-SVP row, 1.28x on taxol.  What dilutes it is everything else in an SCF:
the **one-time cderi build** is 15.5 % of the def2-TZVPP run and **39.4 %** of
the taxol one and nothing here touches it, and the **XC build** is another
15-30 % and `fastxc` already had it.

**`patch_all()` is worse than `fastdf` alone on the def2-TZVPP row**, in both
repetitions, and the anatomy says where: `_cholesky_eri` costs 1.078 s with
everything loaded against 0.772 s with only `fastdf` -- the cderi build sizes
its batches from the free memory it finds, and the other modules have taken
some -- and `nr_rks` costs 1.866 s against 1.715 s.  Neither is this module's
doing and neither is a correctness problem, but a DF run at a large basis is
one of the few places where loading everything is not free, and
`electrolites.patch('fastdf', 'fastxcnlc', 'fastnlc')` is the subset that pays
there.

**A VV10 functional is a different picture entirely.**  On the same molecule,
DF/def2-SVP, two repetitions each, second quoted:

| functional | GPU4PySCF | + `patch_all()` | |
|---|---|---|---|
| wB97M-V | 21.993 s | **14.194 s** | **1.55x** |
| B97M-V | 22.832 s | **14.701 s** | **1.55x** |

`nr_nlc_vxc` is **77-80 %** of those runs unpatched, and in the patched ones
the row is gone.  **How much of that is this round** takes a third column,
because the previous rounds were not entirely absent from a density-fitted
VV10 run: `fastnlc` replaces `dft.numint._vv10nlc`, the `O(n_grid^2)` double
sum, and `_DFHF.get_veff` does reach that name through `nr_nlc_vxc`.  Running
`patch_all()` with `FASTXCNLC_OFF=1` is exactly what this repository did on a
DF run before this round:

| wB97M-V, penicillin V/def2-SVP | wall clock | against stock |
|---|---|---|
| GPU4PySCF 1.8.1 | 21.993 s | — |
| + `patch_all()` as it stood before this round | 16.018 s | 1.37x |
| + the `_DFHF.get_veff` binding this round added | **14.194 s** | **1.55x** |

So the fused grid pass is worth **1.13x** on top of what was already there,
and the coverage gap it closed was worth reporting whatever its size: the
module was loaded, the run was correct, and nothing said it was doing nothing.

### 3.6 What the dense expansion costs, and what is left

The exchange build unpacks the packed cderi into a dense `(nao, nao)` block
per auxiliary function and contracts it densely.  How much of that block is
structurally zero is a property of the geometry and the two basis sets alone,
and `benchmarks/df_fill_probe.py` measures it without running anything
(def2-SVP, the basis' own jkfit set):

| molecule | natm | nao | naux | stored pairs | `nao(nao+1)/2` | fill | cderi |
|---|---|---|---|---|---|---|---|
| glycine | 10 | 95 | 471 | 4 623 | 4 560 | 101.4 % | 0.02 GB |
| alanine dipeptide | 22 | 200 | 974 | 18 148 | 20 100 | 90.3 % | 0.14 GB |
| Ni dithiolene (open shell) | 21 | 327 | 1 992 | 36 704 | 53 628 | 68.4 % | 0.58 GB |
| penicillin V | 42 | 430 | 2 191 | 60 845 | 92 665 | 65.7 % | 1.07 GB |
| taxol | 113 | 1 123 | 5 598 | 319 589 | 631 126 | 50.6 % | 14.3 GB |
| PfPMT | 284 | 2 697 | 13 365 | 808 156 | 3 638 253 | **22.2 %** | 86.4 GB |

Two things follow, and they are the same thing seen twice.

**The dense contraction does 1/fill times the arithmetic it needs to**, and
that factor grows with the system: 1.5x at 42 atoms, 2.0x at 113, **4.5x** at
284.  Neither item 1 nor item 2 touches it -- they rearrange the dense work,
they do not make it smaller.  A block-sparse exchange build, over shell-pair
or AO-tile blocks, is what would, and this table is the case for writing one.
It is also why that is a round of its own and not a paragraph here: the
pattern has to be built, the unpack has to write tiles rather than a dense
block, and the second contraction (`vk = R R^T`) stays dense either way.

**The same screening is what bounds the system size**, which is the limit
GPU4PySCF's README states and catalogue A13 quotes.  At 284 atoms/def2-SVP the
packed tensor is 86 GB: past the device, and past the 80 GB host allowance the
benchmarks here run with, so that calculation does not run at all in this
scheme.  The direct-SCF path in this repository is measured at 284 and 545
atoms; the density-fitted one cannot reach either.  Item 3 moves the boundary
between "on the device" and "streamed", not the boundary between "runs" and
"does not".

---

## 4. Correctness

`tests/test_df.py` compares this build against GPU4PySCF's **element-wise on
the AO matrices**, both implementations in one process on one cderi tensor, and
then runs a whole DF-B3LYP energy and gradient against the unpatched pair.

Six molecule/basis cases (up to f functions, one of them open shell), both
density factorisations, `hermi` 0 and 1, `n_dm` 1, 2 and 4, and **every
internal configuration** -- the rank-k branch has to be forced on these
molecules, because all of them are below `SYRK_MIN_NAO` and the default
configuration would never take it:

```
worst relative difference: J 5.4e-16, K 3.6e-15      (tolerance 1e-11)
ok  DF-b3lyp/def2-svp   ethanol (spin=0): dE = 5.7e-14 Eh, |dg| = 1.7e-10
ok  DF-b3lyp/def2-svp   CH3     (spin=1): dE = 0.0    Eh, |dg| = 6.3e-16
ok  DF-wb97m-v/def2-svp ethanol (spin=0): dE = 1.1e-13 Eh, |dg| = 4.7e-09
ok  DF-wb97m-v/def2-svp CH3     (spin=1): dE = 0.0    Eh, |dg| = 7.7e-10
```

The two wB97M-V rows are there because they are the only thing in this suite
that reaches `_DFHF.get_veff`'s fused pass (§5), and their gradient column is
two orders of magnitude looser than the B3LYP one: a VV10 gradient carries the
`O(n_grid^2)` double sum, and these are two *separately converged* densities,
not one density through two code paths.  `4.7e-09` against a `1e-8` tolerance
is that and not a kernel disagreement -- the fixed-density comparison above it
is at `3.6e-15`.

The Coulomb matrix is not bit-identical the way `fastj`'s is, and the reason is
worth stating: the Coulomb pass is the same two contractions over the same
operands, issued as an explicit GEMV/GEMM pair instead of through cuTENSOR, so
the summation order over the auxiliary index differs.  `3.0e-16` relative is
that and nothing else -- with `FASTDF_JPASS=tensor` the difference is exactly
zero.

`tests/test_unlc.py`, which is the direct-SCF fused grid pass this round
extended, still reads 8 cases + 1 whole-SCF check, 0 failed, with
`fused_u = 157`: adding the density-fitted binding did not move the two that
were already there.

Separately, both cderi residencies and both packed layouts are checked --
`use_gpu_memory=False` stages the tensor from the host in the *opposite*
layout (the pair index contiguous rather than the auxiliary one), which is a
different branch of both the Coulomb pass and the block iterator -- and the
mixed resident/streamed split of item 4 is checked at every block size against
the tensor it is a partition of.

---

## 5. What it does not cover

Four of these are *other DF code paths*, not declines: a call that reaches
them never reaches `df_jk.get_jk` at all, so `fastdf.stats()` shows nothing
and `benchmarks/coverage_probe.py --density-fit` is the way to see it.

| not covered | where it is | why it matters |
|---|---|---|
| a **pure** functional's DF Coulomb build | `df/j_engine_3c2e.py`, reached from `df/df.py:113` when `with_k` is false and no cderi exists | a DF-PBE run never builds a cderi and never calls this function; `fastdf` reports **zero calls** on one.  It is a McMurchie-Davidson three-centre kernel in `libvhf_md`, i.e. a `fastj`-shaped generator project rather than a driver rewrite |
| the **cderi build** itself | `df/df.py:_cholesky_eri`, and `Int3c2eOpt.build` under it | one-time, and §3.1 measures what that is worth at each size.  The metric contraction inside it, `j3c.dot(aux_coef)`, forms `L^{-1}` explicitly and then multiplies by it; a triangular solve is half the arithmetic, and the auxiliary contraction coefficients it is folded with are 0.06 % dense |
| the **DF gradient's** two-electron derivative | `df/grad/rhf.py:_jk_energy_per_atom` | §3.1 measures it at **29 % of a DF gradient at def2-SVP and 83 % at def2-TZVPP** -- the largest single uncovered DF target, and the one that grows with the basis rather than shrinking.  It has the same shape as this module's: a dense `(i,j,L)` expansion, here built by two *fancy-index scatters* rather than by a kernel, and two cuTENSOR contractions per block, twice over |
| the **DF Hessian's** CPHF build | `df/hessian/rhf.py:_get_jk`, a fifth code path | it builds J and K directly in the MO basis and does not go through `df_jk.get_jk`.  It also moves the cderi to the host mid-run (`:1273`), which is the regime item 4 is about |

One more gap this round found, in a *different* module, and closed.
`density_fit()` builds its class as
`lib.set_class(dfmf, (_DFHF, mf.__class__))`, so `_DFHF.get_veff`
(`df/df_jk.py:270`) comes before `rks.RKS.get_veff` **and** `uks.UKS.get_veff`
in the method resolution order, and `fastxcnlc` -- which patches those two --
was never on a density-fitted run at all.
`benchmarks/coverage_probe.py --density-fit` read `(-)` on that row for every
functional (it now reads 14 calls on a 13-cycle DF wB97M-V, with
`fastxcnlc.stats()` reporting `fused: 14` -- the row itself has no `stock`
counter, because this wrapper calls the function it replaced on *every* call
by design and counting that would read `ours = 0` on a run where every call
fused), and a DF wB97M-V or B97M-V calculation therefore walked the grid
twice per iteration where a direct-SCF one walks it once.  It is the same
class of bug `docs/OPENSHELL_TMC.md` found on `dft/uks.py` and
`docs/VV10_OPENSHELL.md` on `hessian/uks.py`: *a patch target is not a code
path*, and only a counter finds it.  `fastxcnlc` now patches
`_DFHF.get_veff` as well, restricted and unrestricted; it declines the ROHF
branch of that function, which rewrites the density into a two-row array
before it calls `nr_uks`, and the GHF branch, which calls neither of the two
names the wrapper redirects.

And one path that does not reach this module because, on stock GPU4PySCF, it
does not run at all: **`only_dfj=True` is broken in 1.8.1**.  `_DFHF.get_jk`
calls `self.with_df.get_jk(mol, dm, hermi, with_k=False)` (`df/df_jk.py:211`)
where `DF.get_jk`'s first argument is the density, so a
`density_fit(only_dfj=True)` run dies with `'Mole' object has no attribute
'shape'` before it reaches anything.  Upstream's own bug, unrelated to
anything here, and it matters because "density-fit J only, exact K" is the
configuration a reader of the catalogue entry would most likely try first.

> **Since this was written**, `fastdfj` repairs that branch, and the
> configuration turns out to be the fastest of the three ways this repository
> can build a large hybrid's two-electron half.  `docs/README_fastdfj.md` is
> the module and the measurement; `docs/DF_WORK_LIST.md` W1 is the item it
> closes.  It is still true that the repaired path does **not** reach `fastdf`
> -- a J-only DF build never constructs a cderi, so there is nothing for this
> module to contract.

Within `df_jk.get_jk` itself, what falls through to GPU4PySCF (and is counted
by `stats()['declined_reasons']`):

* a **complex** density -- the GHF/DHF spin-free build stacks real and
  imaginary factors and asks for `hermi = 0`;
* **more than one device**, as every module here does;
* the two **asymmetric density factorisations** (`_aggregate_dm_factor_l`),
  which only the TDDFT gradient and non-adiabatic-coupling drivers produce.

And one thing that is covered but is worth stating because the arithmetic
differs: an **untagged** density -- every `hermi = 0` right-hand side a TDDFT
build produces -- factorises through an SVD into two different tensors, its
exchange matrix is not symmetric, and item 1 does not apply to it.  Those
calls keep the general GEMM and take only what items 3 and 4 give them.  So is
a **Coulomb-only** call on an existing cderi -- nothing is unpacked and no
exchange tensor is formed, so only items 3 and 4 reach it, and it has a branch
of its own rather than a fall-through so that it does.

---

## 6. What comes next

`docs/DF_WORK_LIST.md` is the ranked work list this round ends in — nine
candidates with the evidence each rests on, an acceptance criterion and a
falsification condition for each, what was already tried and measured not to
pay, and the four ways a measurement on this path goes silently wrong.  Two of
the nine are **capacity** fixes rather than speedups (a one-line upstream bug
that unlocks RIJONX, and letting these modules run on more than one device);
one is the only route that changes the storage exponent (local/pair-atomic
fitting) and is a different approximation rather than a faster path to the
same number.

---

## 7. Reproducing

```bash
python tests/test_df.py --ndm 1,2,4
python benchmarks/df_anatomy.py --xyz benchmarks/molecules/taxol.xyz \
    --basis def2-svp [--patch all]
python benchmarks/df_probe.py  --xyz benchmarks/molecules/taxol.xyz \
    --basis def2-svp --cycles 6
python benchmarks/df_jk_bench.py --xyz benchmarks/molecules/taxol.xyz \
    --basis def2-svp --cycles 6 --ndm 1,2,4 --blksizes 64,128,256,512
python benchmarks/coverage_probe.py --xyz benchmarks/molecules/taxol.xyz \
    --basis def2-svp --density-fit
python benchmarks/df_fill_probe.py --basis def2-svp     # no GPU time at all
```

The ablations, all of which are read off `df_jk_bench.py`'s columns:

```bash
FASTDF_OFF=1                 # hand every call back to GPU4PySCF
FASTDF_STEP2=syrk|gemm|auto  # the exchange contraction
FASTDF_SYRK_MIN_NAO=<n>      # where `auto` switches over (default 800)
FASTDF_STEP1=tensor|bgemm    # the first contraction
FASTDF_JPASS=blas|tensor     # the Coulomb pass
FASTDF_BLK=<n>               # the auxiliary block size
FASTDF_RESIDENT_FRAC=<f>     # item 4's share of free memory; 0 disables it
FASTDF_FP32=1                # store the cderi in single precision.  Not an
                             # ablation of this module: it changes the
                             # answer, and `docs/README_fastdffp32.md` is
                             # what it costs (DF_WORK_LIST W7)
FASTXCNLC_OFF=1              # and, for the VV10 rows, the fused grid pass
```
