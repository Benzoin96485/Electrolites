# fastdfj3c — the Coulomb build a pure density-fitted functional runs on

`docs/DF_WORK_LIST.md` **W9**.  It closes.

W9 is the one entry on that page whose target was already known to be *good*.
`docs/README_fastdfj.md` §3.1 timed GPU4PySCF's three-centre McMurchie–Davidson
Coulomb engine against the direct four-centre build at three sizes and found it
**1.67x–2.43x faster at every one**, and wrote: "`df/j_engine_3c2e.py` is a
McMurchie–Davidson kernel nobody in this repository had profiled; it is very
good."  W9 was therefore ranked last of the open items and described as being
"about a path that is in good shape, not one that is losing".

That is still true of the *integral arithmetic*, and this round does not change
a line of it.  What it changes is everything around it, and the reason there
was anything to change is that **nobody had timed the stages of that call**.
Timed stage by stage, GPU4PySCF's Coulomb build spends a large and
size-dependent fraction of itself re-deriving arguments it derived on the
previous cycle, and launches its larger kernel with a few hundred thread blocks
on a 108-SM card.

---

## 0. Summary

*(Every number on this page is measured on one A100 of a whole Perlmutter GPU
node, GPU4PySCF 1.8.1, def2-SVP with its own jkfit auxiliary set unless another
basis is named, and variants are timed round robin inside one process —
`docs/DF_WORK_LIST.md` §5.)*

* **The Coulomb build of a pure density-fitted functional is 1.03x-2.32x
  faster at one density** -- 2.32x at 10 atoms, 1.78x at 42, 1.13x at 113 and
  1.03x at 284, because what is removed is host work and the kernels take over
  as the molecule grows (§3, §4).  A whole DF-PBE SCF against a `patch_all()`
  baseline is **1.16x** at 42 atoms.
* **And 1.5x-4.5x when more than one density is asked for**, which is every
  response property of such a functional -- **4.45x** at two densities, where
  GPU4PySCF's engine pays for a batch of four and uses two (§6).
* **On an open-shell 3d complex it is 5.8x-13.2x**, because def2-SVP reaches an
  f shell on the metal's auxiliary set and that is where the engine GPU4PySCF
  picks for multiple densities is worst: 318 ms for a 21-atom Coulomb matrix
  against 24 ms (§6.1).
* **It changes no arithmetic.**  Every CUDA kernel launched is GPU4PySCF's own
  with GPU4PySCF's own arguments, so the bar is round-off and not a tolerance
  argued from the physics.  `tests/test_dfj3c.py` compares element-wise against
  an unpatched binding **in the same process**, on the same option objects and
  the same density.
* **The largest single item is a launch geometry, not a kernel.**
  `Int3c2eOpt.build` makes exactly **one** thread-block range per
  angular-momentum group, and the auxiliary-vector kernel's grid is
  `(groups × auxiliary-shell batches)` — six by sixty-two on penicillin V, i.e.
  **372 thread blocks**, with the groups differing in size by seventy times.
  Both kernels read their group's `(li,lj)` from the *first* pair of their
  range and stride the rest, so a range may be any sub-interval of a group:
  splitting one is a launch configuration change and not a kernel change.  §5.
* **The engine GPU4PySCF picks for more than one density is the wrong one.**
  `df_jk.get_j` uses the McMurchie–Davidson engine only when it is handed a
  *single* density and sends every other call to the Rys `int3c2e_bdiv` engine
  (`df/df_jk.py:594`).  That looks right — the Rys engine carries the density
  index inside one kernel launch where the MD engine loops over densities in
  Python and rebuilds the integrals for each — and measured it is backwards:
  looping the faster engine wins even though it repeats the integral work.  §6.
* **Two ideas the shared-memory request suggests are measured and rejected**,
  and kept runnable: a smaller shared-memory budget, and one launch per
  angular-momentum class.  §8.
* **What is left is inside the kernels**, which are now almost the whole call
  -- 85 % at 42 atoms and 97 % at 284.  §9 says which three things, and why
  none of them is reachable from the host.

---

## 1. What this call is, and who reaches it

`df/df.py:111`:

```python
def get_jk(self, dm, hermi=1, with_j=True, with_k=True, ...):
    if not with_k and self._cderi is None:
        return df_jk.get_j(self, dm, hermi), None
```

A **pure** functional never asks for exchange, so a density-fitted PBE, BLYP or
TPSS run returns on that line on every cycle and **never builds a cderi at
all**.  `fastdf`, which rearranges the cderi contraction and is the largest
module on the density-fitting path, reads **zero calls** on such a run — that
is not an inference, `benchmarks/coverage_probe.py --density-fit --xc pbe`
counts it.  The same function is the Coulomb half of every **RIJONX** run
(`fastdfj`), and the Coulomb build of every **response** property of a
density-fitted pure functional, where `scf/_response_functions.py` asks for one
density per right-hand side.

The call is nine stages:

```
dm --> mol.apply_C_mat_CT          decontract the density
   --> Int3c2eOpt.contract_dm      rho_P = sum_ij (ij|P) D_ij
   --> auxmol.apply_CT_dot
   --> two triangular solves with the Coulomb metric
   --> auxmol.apply_C_dot
   --> Int3c2eOpt.contract_auxvec  vj_ij = sum_P (ij|P) rho_P
   --> mol.apply_CT_mat_C
```

and the two `contract_*` are each a host-side setup, a Hermite transformation
kernel and one integral kernel out of `libgvhf_md`.
`benchmarks/df_j3c_probe.py` times all nine.

One thing about that shape is worth saying before any number, because it is
what ranks the two halves against each other and nothing in
`docs/DF_WORK_LIST.md` says it: **the three-centre integrals are computed
twice per Fock build**.  `contract_dm` evaluates every `(ij|P)` to contract it
with the density, the metric solve happens, and then `contract_auxvec`
evaluates every `(ij|P)` *again* to contract it with the fitted coefficients.
That is the price of storing no cderi -- the full-DF route pays it once, into
86.4 GB at 284 atoms (`docs/DF_WORK_LIST.md` §3.2), and then contracts twice
against the stored tensor.  It is also why the two kernels come out at
roughly the same cost in every row of §3, and why no rearrangement can fuse
them: the second pass needs the output of the metric solve, which needs the
first.

---

## 2. What the module changes

Six things, none of them arithmetic.  `electrolites/fastdfj3c.py` has them in
order, with the switch that turns each off.

### 2.1 The per-call constants are uploaded once

`contract_dm` and `contract_auxvec` between them copy the shell-pair list, the
block offsets, the Hermite pair offsets and the auxiliary `ao_loc` to the
device on **every** call, re-run a `np.cumsum` over the auxiliary shells and
rebuild the 16-shell load-balancing split in a Python loop.  Every one of those
is fixed at `Int3c2eOpt.build()`.  They move there
(`electrolites/_j3c.py`).

### 2.2 The recontraction tables are uploaded once

`SortedGTO.c_ao_loc` is a **property**: it recomputes a NumPy cumsum and copies
it to the device every time it is read, and each of the four recontraction
kernels (`CT_dot_mat`, `C_dot_mat`, `mat_dot_C`, `mat_dot_CT`) reads it plus
four more host arrays.  One `get_j` reads six of those methods, so **thirty
host-to-device copies and six NumPy cumsums per Coulomb matrix**.  All five
arrays are written once in `SortedGTO.from_mol` and never again.

This one is not only W9's.  The same four methods are how the cderi build
applies its contraction coefficients (`docs/README_fastdfcderi.md`), how
`scf/hf.py:159` builds the nuclear attraction, and how `x2c` transforms its
operators.

### 2.3 The scratch is allocated once

Both contractions build a Hermite work buffer per call and free it again.
`docs/DF_WORK_LIST.md` §5.8 is this repository's own lesson that such a
temporary does not cost its allocation, it costs the `cudaFree` that ends it,
which is a synchronisation point.  The three buffers the caller never sees are
kept on the option object; the two that are returned are not.

### 2.4 The Hermite transformations stop re-uploading `ao_loc`

`scf/j_engine.py:_dm_to_Rt` and `_Rt_to_dm` take the pair list and the pair
offsets as device arrays and then copy `mol.ao_loc` from the host themselves.
This module calls the same two CUDA kernels with the cached copy; the direct
path's own `_dm_to_Rt` is untouched.

### 2.5 The auxiliary-vector kernel is launched with enough thread blocks

§5.  The density-contraction kernel is deliberately left alone, and §5 is also
why.

### 2.6 More than one density goes to the McMurchie–Davidson engine too

§6.

---

## 3. Where the time was

`benchmarks/df_j3c_probe.py`, unpatched, one density, at fixed converged
density.  Each stage is synchronised and charged to itself, so the column sums
to slightly more than an unwrapped call; the probe prints both.

Milliseconds per call, A100-SXM4-40GB, minimum of five.  The instrumented
column sums to a little more than the unwrapped call because each stage is
synchronised; both are printed.

| stage | glycine 10 | alanine 22 | penicillin V 42 | taxol 113 | PfPMT 284 |
|---|---|---|---|---|---|
| `mol.apply_C_mat_CT` | 0.537 | 0.548 | 0.553 | 0.705 | 2.612 |
| `contract_dm`, host setup | 0.285 | 0.302 | 0.311 | 0.386 | 1.611 |
| ↳ `dm_to_Rt` | 0.438 | 0.419 | 0.521 | 0.674 | 1.015 |
| ↳ **the integral kernel** | 0.346 | 1.548 | 7.119 | 75.386 | **395.6** |
| `auxmol.apply_CT_dot` | 0.284 | 0.306 | 0.300 | 0.328 | 0.384 |
| the metric, two triangular solves | 0.240 | 0.345 | 0.654 | 4.332 | 12.731 |
| `auxmol.apply_C_dot` | 0.290 | 0.298 | 0.297 | 0.476 | 0.784 |
| `contract_auxvec`, host setup | 0.366 | 0.381 | 0.401 | 2.196 | 2.538 |
| ↳ `aux_to_Rt` | 0.017 | 0.018 | 0.018 | 0.020 | 0.025 |
| ↳ **the integral kernel** | 0.738 | 4.631 | 16.213 | 74.075 | **312.4** |
| ↳ `Rt_to_dm` | 0.419 | 0.406 | 0.424 | 0.654 | 1.371 |
| `mol.apply_CT_mat_C` | 0.576 | 0.654 | 0.655 | 0.743 | 1.559 |
| | | | | | |
| **whole call**, unwrapped | **3.35** | **8.55** | **26.13** | **157.2** | **730.0** |
| the two integral kernels | **24 %** | **63 %** | **85 %** | **93 %** | **97 %** |

**The share that is not integrals runs from three quarters to three per
cent.**  That is the whole shape of this round: at ten atoms a Coulomb build
is two integral kernels and about thirty host-to-device copies, and the copies
cost three times the kernels; at 284 atoms the kernels are everything.  Nine of
the twelve rows above are **flat in system size** -- `apply_C_mat_CT` is 0.537
ms at ten atoms and 2.6 ms at 284, `aux_to_Rt` is 17 µs at every size, the two
host setups are a third of a millisecond each until they are not -- because
they are launch and transfer latency, not work.

**Two rows are worth reading twice.**  `apply_C_mat_CT` and `apply_CT_mat_C`
together are 1.11 ms on a *ten-atom* molecule, which is a third of its entire
Coulomb build; each is two CUDA kernels, and between them they read the same
five host arrays ten times and recompute a NumPy cumsum six times (§2.2).  And
the **metric** — the pair of triangular solves everyone would look at first —
is 5.3 % at ten atoms and **1.7 %** at 284; there is nothing there, which is
worth recording because an explicit inverse in place of the two solves is the
obvious idea and it would buy under two per cent.

---

## 4. What it is worth

`benchmarks/df_j3c_bench.py`.  Six bindings of the same call, in one process,
round robin:

One density, def2-SVP, at a converged density, minimum of eight.
A100-SXM4-40GB.

| | `stock` | `ctr` | `opt` | `blk` | `mdm` | **`fast`** |
|---|---|---|---|---|---|---|
| glycine, 10 atoms | 3.275 ms | 1.545x | 1.431x | 1.629x | 1.000x | **2.320x** |
| alanine dipeptide, 22 | 7.057 ms | 1.137x | 1.120x | 1.850x | 0.997x | **2.178x** |
| penicillin V, 42 | 26.232 ms | 1.032x | 1.028x | 1.724x | 1.000x | **1.784x** |
| taxol, 113 | 157.491 ms | 1.013x | 1.018x | 1.121x | 1.002x | **1.128x** |
| PfPMT, 284 | 731.442 ms | 1.006x | 1.007x | 1.029x | 1.003x | **1.033x** |
| penicillin V, def2-TZVPP | 69.071 ms | 1.008x | 1.007x | 1.281x | 1.000x | **1.291x** |

Worst relative difference against `stock` over the whole table: **1.6e-15**.

**The margin falls with size, and §3 says why.**  At ten atoms the Coulomb
build is mostly not integrals, so removing the host work is most of what there
is to remove; at 284 atoms it is 98 % integrals and there is nothing left for
this module to take.  That is the opposite of the shape most entries on
`docs/DF_WORK_LIST.md` have, and it is worth saying plainly: **this is a
small-and-medium-molecule result on the single-density path**, and the size
where it stops paying is between 42 and 113 atoms.

**Which item carries it also moves with size.**  At ten atoms the
recontraction tables are worth 1.55x on their own, because a ten-atom Coulomb
build is thirty host-to-device copies and a dozen kernel launches and the
copies cost more than the kernels (§3).  From twenty atoms up it is the launch
geometry -- 1.85x at 22 atoms, 1.72x at 42, 1.12x at 113 and 1.03x at 284 --
and by 113 atoms the two host items are worth one per cent each.  `mdm` is
1.00x on every row here, as it must be: at one density it **is** GPU4PySCF's
engine choice, and §6 is the column it is for.

The `blk` and `ctr` columns do not compose additively (1.63 x 1.55 is not
2.32) because they overlap in the same call; what is quoted is what each is
worth *alone* against `stock`, which is the form an ablation has to take when
the items are not independent.

### 4.1 Whole runs

A whole density-fitted **PBE** SCF, GPU4PySCF everywhere else, each variant run
twice in one process and the second quoted (`docs/DF_WORK_LIST.md` §5.3):

| | stock | fast | | energy |
|---|---|---|---|---|
| penicillin V, 42 atoms | 2.300 s | **2.135 s** | 1.078x | 4.6e-13 Eh |
| taxol, 113 atoms | 7.935 s | **7.623 s** | 1.041x | 2.0e-11 Eh |

**A 1.78x per call is 1.08x end to end**, and the reason is not this module:
the Coulomb build is a *small share of a pure-functional SCF* at these sizes --
`benchmarks/df_anatomy.py` puts `with_df.get_jk` at 27 % of a 42-atom DF-B3LYP
run and `nr_rks` at 35 %, and a pure functional has no exchange to pay for at
all, so the grid half dominates further.  The row a user actually has is the
one against a patched baseline, where `fastxc` has already taken the grid
half; that is this one:

| against a patched baseline (`fastxc`, `fastnlc`, `fastxcnlc`, `fastdf`) | stock | fast | |
|---|---|---|---|
| penicillin V, 42 atoms | 1.755 s | **1.508 s** | **1.164x** |
| taxol, 113 atoms | 6.438 s | **6.151 s** | 1.047x |
| PfPMT, 284 atoms *(unpatched baseline)* | 85.706 s | 84.864 s | 1.010x |

The energies agree to 9.1e-13, 1.8e-12 and 3.1e-10 Eh, in the same number of
cycles.  **1.16x on a whole 42-atom DF-PBE run** is the number a user of this
repository sees, and it is larger than the unpatched-baseline row because
`fastxc` has already taken the grid half that was hiding the Coulomb build.

---

## 5. The launch geometry, which is the largest item

`Int3c2eOpt.build` ends with

```python
self.shl_pair_offsets = np.cumsum(
    [0] + [len(x) for x in shl_pair_idx], dtype=np.int32)
```

— one block range per `(li, lj)` group, however many shell pairs that is.  On
penicillin V/def2-SVP the six groups hold 11 306, 10 150, 3 200, 2 446, 1 362
and 159 shell pairs.

The two kernels use that list differently:

| | grid | blocks, penicillin V | blocks, taxol |
|---|---|---|---|
| `contract_int3c2e_dm` | (auxiliary shells × groups) | 946 × 6 = **5 676** | 2 454 × 6 = **14 724** |
| `contract_int3c2e_auxvec` | (groups × auxiliary-shell batches) | 6 × 62 = **372** | 6 × 155 = **930** |

372 thread blocks on a 108-SM A100, of which 62 carry seventy times the work of
another 62.

Both kernels read `li`/`lj` from `bas_ij_idx[shl_pair0]` — the **first** pair of
their range — and then stride, and the strided loop is correct for any
sub-interval: threads past the end take `fac = 0`.  So the group may be cut
into pieces, and that is a change to an argument and not to a kernel.

`benchmarks/df_j3c_sweep.py` sweeps it:

`contract_auxvec`, milliseconds and the ratio against GPU4PySCF's whole
groups, at a converged density.  The `contract_dm` column is the same sweep on
the other kernel, and is why that one is left alone:

| shell pairs per block | 0 (upstream) | 8192 | 4096 | 2048 | 1024 | 512 | 256 | 128 |
|---|---|---|---|---|---|---|---|---|
| **glycine**, 10 atoms | 1.063 | 0.999 | 0.999 | 1.001 | 1.068 | 1.240 | **1.495** | 1.311 |
| **Ni dithiolene**, 21, open shell | 10.029 | 1.000 | 1.250 | 1.350 | 1.763 | **2.039** | 2.022 | 1.500 |
| **penicillin V**, 42 | 16.507 | 1.356 | 1.655 | 2.511 | 2.648 | **2.699** | 2.661 | 1.715 |
| **Cu macrocycle**, 60, open shell | 31.171 | 1.981 | 2.255 | 2.309 | 2.364 | **2.368** | 2.348 | 1.541 |
| **penicillin V**, def2-TZVPP | 38.815 | 1.410 | 1.603 | 1.607 | 1.605 | 1.630 | **1.633** | 1.163 |
| **taxol**, 113 | 74.596 | 1.223 | 1.235 | **1.241** | 1.241 | 1.235 | 1.224 | 0.772 |
| **PfPMT**, 284 | 313.867 | 1.054 | **1.055** | 1.054 | 1.051 | 1.043 | 1.031 | 0.647 |
| `contract_dm`, penicillin V | 7.514 | 0.995 | 0.995 | 0.982 | 0.939 | 0.860 | 0.719 | 0.422 |
| `contract_dm`, taxol | 75.748 | 0.987 | 0.976 | 0.952 | 0.911 | 0.822 | 0.675 | 0.390 |
| `contract_dm`, PfPMT | 396.167 | 0.981 | 0.960 | 0.929 | 0.876 | 0.799 | 0.652 | 0.374 |

Worst relative difference across the whole table: **2.3e-15**; the splits are
different summation orders of the same sum and nothing else.

**The two kernels want opposite things, as the entry predicted.**
`contract_dm` is monotonically *worse* for every split, down to 0.37x at 128 —
its grid already carries the auxiliary shells, so a split buys it nothing and
costs it a warp reduction and an `atomicAdd` per block.  It keeps GPU4PySCF's
whole groups.

**And the optimum for the other one moves by a factor of thirty-two across the
ladder**, which is why the shipped block size is derived and not fixed.  256
shell pairs per block is best at ten atoms and is *0.65x* at 284; 8192 is best
at 284 and is 1.00x — no gain at all — at ten.  The rule
(`Int3c2eConstants.auto_chunk`) aims at a **total block count**, because what
the split buys is thread blocks and the kernel already has
`groups x auxiliary-shell batches` of them:

```
want_sp = ceil(4000 / auxiliary-shell batches)
chunk   = largest power of two <= (shell pairs / want_sp),  clamped to [256, 8192]
```

which picks 256 on glycine and the Ni complex, 512 on the Cu macrocycle, 1024
at def2-TZVPP, 4096 on taxol and 8192 on PfPMT — **within 1.7 % of the swept
optimum in all seven rows**, where a fixed 1024 would have given up 29 % on
glycine and 13 % on the Ni complex.  All three constants are switches.

**The gain itself falls with size and that is not the block size's fault.**
At 284 atoms upstream already launches 2 196 blocks and the whole split is
worth 1.05x; at 42 atoms it launches 372 and the split is worth 2.70x.  The
open-shell transition-metal rows are there because a threshold tuned on
organics of one size is exactly the kind that does not generalise; these two
have ten angular-momentum groups instead of six and an f shell on the metal,
and the rule picks the swept optimum for both.

### 5.1 And the other half of the same geometry, which does not move

`contract_auxvec` also splits the **auxiliary** shells, 16 at a time within
each angular-momentum group, and that is the knob that trades the same
parallelism the other way: a larger batch repeats each shell pair's Gaussian
product centre fewer times (it is computed outside the auxiliary loop), a
smaller one gives more blocks.  Swept at the shipped shell-pair block size:

`contract_auxvec`, at the shipped shell-pair block size, relative to 4:

| auxiliary shells per block | 4 | 8 | **16** (upstream) | 32 | 64 |
|---|---|---|---|---|---|
| penicillin V, 42 atoms | 6.623 ms | 1.054x | 1.068x | **1.078x** | 1.067x |
| taxol, 113 atoms | 66.000 ms | 1.062x | 1.092x | 1.111x | **1.118x** |

**This knob is left at GPU4PySCF's 16 although 32 is measurably better**, and
the reason is worth stating rather than hiding: it is worth 0.9 % on penicillin
V and 1.7 % on taxol *of one kernel*, i.e. under 0.7 % of the call, and moving
it changes `ksh_blocks`, which is the input the shell-pair block rule above was
tuned against on seven systems.  Re-auditing that rule to collect half a per
cent is not a trade worth making; the number is recorded so that whoever does
re-audit it knows which way to go.  `FASTDFJ3C_KSH_CHUNK` is the switch.

`FASTDFJ3C_KSH_CHUNK` is kept as a switch and its default is GPU4PySCF's.

---

## 6. The engine GPU4PySCF picks for more than one density

`df/df_jk.py:594`:

```python
intopt = dfobj.intopt                      # the Rys `int3c2e_bdiv` engine
if dm.ndim == 2 or len(dm) == 1:
    ...
    intopt = dfobj.j_engine                # the McMurchie-Davidson engine
```

The argument for that is real: `int3c2e_bdiv`'s two kernels take `n_dm`
and carry the density index **inside** one launch, so the integrals are
computed once and contracted against every density, where `j_engine_3c2e`
loops over densities in Python and rebuilds the integrals for each.  On paper
the Rys engine should win for `n_dm > 1` by exactly the factor the MD engine
wastes.

Measured, it loses anyway:

| penicillin V, 42 atoms | `n_dm` | GPU4PySCF | `mdm` (the engine alone) | **`fast`** |
|---|---|---|---|---|
| | 1 | 26.2 ms | 1.000x | 1.784x |
| | **2** | 130.9 ms | 2.543x | **4.450x** |
| | 4 | 155.0 ms | 1.553x | **2.730x** |
| | 8 | 289.0 ms | 1.465x | **2.577x** |
| | 16 | 573.1 ms | 1.461x | **2.571x** |
| | 32 | 1126.1 ms | 1.445x | **2.551x** |
| taxol, 113 atoms | **2** | 738.3 ms | 2.377x | **2.646x** |
| | 4 | 889.7 ms | 1.450x | **1.610x** |
| | 8 | 1658.1 ms | 1.356x | **1.503x** |

**The Rys engine is 1.36x–2.54x slower per density than the engine that
rebuilds the integrals for each**, at every count measured up to 32, and it
does not catch up: both curves are linear in `n_dm`, so there is no crossing
to find and `FASTDFJ3C_MDM_MAX_NDM` -- the switch that would send large counts
back -- does not fire by default.

**Two densities is the sharpest row and the reason is a batch size.**
`int3c2e_bdiv.contract_auxvec` reserves shared memory for `dm_batch_size = 4`
and steps the density index in fours, so **two densities cost 84 % of what
four cost** (130.9 against 155.0 ms) -- an unrestricted response solve, which
asks for one density per spin per right-hand side, pays for four and uses two.
Looping a faster engine twice beats that by 4.45x.

### 6.1 Where it is largest: an open-shell 3d complex

The same table on two transition-metal complexes -- a different chemical class
from the organics above, and the one this repository's rule
(`docs/DF_WORK_LIST.md`) says to check a threshold against.  def2-SVP reaches
an **f shell** on the metal's auxiliary set, which no organic def2-SVP case
here does.  `n_dm = 1` means one *unrestricted* density, i.e. two matrices:

| | `n_dm` | GPU4PySCF | `mdm` | **`fast`** |
|---|---|---|---|---|
| Ni dithiolene, 21 atoms, doublet | 1 (2 matrices) | 317.8 ms | 8.976x | **13.247x** |
| | 4 (8 matrices) | 1110.2 ms | 8.351x | **12.300x** |
| Cu macrocycle, 60 atoms, quartet | 1 (2 matrices) | 418.1 ms | 4.205x | **6.743x** |
| | 4 (8 matrices) | 1410.1 ms | 3.635x | **5.846x** |

**318 ms for a 21-atom Coulomb matrix** is what GPU4PySCF's Rys engine costs
when the auxiliary basis reaches f, and the McMurchie--Davidson engine does the
same thing in 35 ms.  Note what is *not* affected: a plain unrestricted **SCF**
never reaches this, because `dft/uks.py:67` and `df/df_jk.py:314` sum the two
spin densities before the Coulomb call and hand over a single matrix.  What
reaches it is every **response** property of such a system, and every caller
that passes the two spins separately.

The `ctr`, `opt` and `blk` columns are 1.00x on these four rows, as they must
be: with the engine unchanged the call is the Rys kernel and nothing this
module touches is in it.

---

## 7. Correctness

There is no tolerance to argue here: the module launches GPU4PySCF's kernels
with GPU4PySCF's arguments, so anything above round-off is a bug.  Every
comparison in `tests/test_dfj3c.py` is made **in one process**, by swapping the
bound functions, so both sides see the same molecule, option objects and
density.

| | reference | worst, def2-SVP | worst, def2-TZVPP |
|---|---|---|---|
| `get_j`, one density | the same engine, GPU4PySCF's binding | **1.6e-16** | **2.2e-16** |
| `get_j`, two and four densities | GPU4PySCF's **Rys** engine | 3.4e-13 | 3.8e-13 |
| `get_hcore` | GPU4PySCF's binding | **0.0** | **0.0** |
| a whole SCF, seven functional/spin cases | GPU4PySCF's binding | ≤ 1.7e-13 Eh | ≤ 5.7e-14 Eh |
| a DF-B3LYP gradient | GPU4PySCF's binding | 6.0e-11 Eh/Bohr | 3.9e-11 |

relative to the largest element of the reference, over ethanol and CH3,
`hermi` 0 and 1, restricted and unrestricted.  Every SCF converged in the same
number of cycles as its reference.

The one comparison that is allowed to be more than round-off is the
multi-density row, because that one compares **two different integral
algorithms** — this module's McMurchie–Davidson build against GPU4PySCF's Rys
one — and it is held to the same bar anyway.

Also checked:

* a whole **pure**-functional DF SCF and a whole **hybrid** one (which builds a
  cderi, so the cached recontraction tables go through `_cholesky_eri` as
  well), restricted and unrestricted, to the same energy in the same number of
  cycles;
* a **RIJONX** run, the other consumer of this Coulomb path;
* a **gradient**, because `get_hcore` and the cderi build reach the same four
  `SortedGTO` methods by a different route;
* `get_hcore` alone, element-wise;
* a **rebuild**: the constants are keyed on the option object's
  `int3c2e_envs`, and a geometry optimisation calls `build()` again on an
  object that already has them.  A cache that did not invalidate would return
  the previous geometry's Coulomb matrix, which no energy check would catch;
* the caches are keyed on the **device** as well as on the object.
  `multi_gpu.run` hands one `SortedMole` to a thread per device and a CuPy
  array carries the device it was allocated on, so a cache keyed on the object
  alone would give one device another device's pointers —
  `docs/DF_WORK_LIST.md` W2 records the same trap on `fastdf`'s
  `_resident_prefix`, where it was found rather than avoided.  That check is
  skipped, loudly, unless more than one GPU is visible;
* `tests/test_patch_aliases.py` carries the seven entry points this module
  replaces, because four of them are methods on `SortedGTO` that four other
  GPU4PySCF modules reach.

```bash
python tests/test_dfj3c.py
python tests/test_dfj3c.py --basis def2-tzvpp
CUDA_VISIBLE_DEVICES=0,1 python tests/test_dfj3c.py --no-scf   # the device key
```

---

## 8. What was measured and rejected

Two ideas that the shared-memory request suggests, and that the card does not
want.  Both are still runnable, which is worth more than a sentence of prose
(`docs/DF_WORK_LIST.md` §2).

| tried | measured | why |
|---|---|---|
| **a smaller shared-memory budget** (`FASTDFJ3C_SHM_FRAC`).  Both schemes size the per-block shell-pair count from `scf/jk.py:SHM_SIZE` and then ask for the widest class's requirement -- **91 KB** for the density contraction and **81 KB** for the auxiliary-vector build -- on *every* block of the grid.  An A100 SM has 164 KB, so that is **one block per SM**, 256 of 2048 threads | **0.98x at half the budget, 0.80x at a quarter** for the density contraction and **0.95x / 0.76x** for the auxiliary-vector build; monotone, at both 42 and 113 atoms | these kernels want shell pairs *in flight*, not blocks per SM: `nsp_per_block` is the number of shell pairs a block carries concurrently, and halving it to double the occupancy is a straight trade the occupancy loses |
| **one launch per angular-momentum class** (`FASTDFJ3C_PER_CLASS=1`), each asking only for the shared memory that class lays out -- 4 KB for `(lij 0, lk 0)` against the 81 KB the whole basis is charged | **0.44x** at 42 atoms and **0.42x** at 113 -- thirty launches take 2.3x the time one does | the C entry point launches on the legacy default stream, so thirty small grids run one after another instead of one large grid filling the card.  The occupancy each of them gains is worth less than the grid-level parallelism all of them lose.  It would need the launcher to take a stream |

The machinery for the second is kept (`Int3c2eConstants.pair_offsets_by_group`
and `ksh_batches_by_group`), because a stream-aware launcher would make it
worth re-running, and because the per-class shared-memory arithmetic is the
thing anyone rewriting these kernels needs first.

---

## 9. What is left

After this round the Coulomb build is **85 % two integral kernels at 42 atoms
and 97 % at 284** (§3), and all three of the
things still inside them are out of reach from the host.

1. **Six double-precision divisions and a square root per (shell pair,
   auxiliary shell).**  `contract_int3c2e_kernel` recomputes the Gaussian
   product centre of every shell pair --
   `xij = (ai*ri[0] + aj*rj[0]) / aij` and its two partners -- inside both the
   auxiliary-primitive loop and the pair loop, so once per auxiliary shell
   block, although it is a property of the pair alone.  It then evaluates
   `theta = aij*ak/(aij+ak)` and `fac = ck/(aij*ak*sqrt(aij+ak))`, two more
   divisions and a root.  `fastj` removed exactly this family from the
   four-centre McMurchie--Davidson J kernel by caching `1/aij` alongside the
   product centre and letting one `rsqrt(aij+akl)` supply both reciprocals
   (`docs/README_fastj.md`); here the pair data would be a `(npair, 4)` device
   array built once at `Int3c2eOpt.build()`.  At the low angular-momentum
   classes -- which hold most of the shell pairs -- that arithmetic is
   comparable to the contraction it wraps.
2. **The Hermite index table is read from global memory once per term.**  The
   contraction is
   `vj_xyz[k] += Rt[p1_ij[k*nf3ij + i] * nsp_per_block] * dm_ij` with `p1_ij`
   a `uint16_t*` into `Rt2_kl_ij`, and the kernels template on `lk` alone --
   so `nf3ij`, `order` and every one of those indices is a runtime value.
   Templating on `lij` as well makes them compile-time constants.  That is the
   `fastj`-shaped generator W9 named, and it is `2*LMAX+1` by `L_AUX_MAX+1`
   classes of two kernels.
3. **`omega == 0` is a runtime test** inside `boys_fn`, on the innermost path,
   although a plain Coulomb build never takes the other branch.

And one that is not about the kernels at all: **the Coulomb build does not
screen on the density.**  `J` is linear in `D`, so `J(D) = J(D_last) +
J(D - D_last)` exactly, and `D - D_last` near convergence is small almost
everywhere.  `scf/j_engine.py`'s four-centre MD J engine screens on
`qd_ij_max`, a density-weighted Schwarz bound; the three-centre one has **no
density screening of any kind** -- it walks every stored shell pair against
every auxiliary shell on every cycle.  A filtered pair list is a host-side
change to `shl_pair_idx` and `pair_loc`, which is the shape this round is
already in, but it is a *different* pair list per cycle and so gives back the
caching of §2.1; whether that trade pays is a measurement nobody has made.

---

## 10. Tools

```bash
python tests/test_dfj3c.py [--basis def2-tzvpp] [--no-scf]
python benchmarks/df_j3c_probe.py --cases glycine,penicillinV,taxol
python benchmarks/df_j3c_probe.py --cases penicillinV --shapes   # no GPU time
python benchmarks/df_j3c_probe.py --cases penicillinV --ndm 1,2,4
python benchmarks/df_j3c_bench.py --cases ... [--scf] [--ndm 1,2,4] [--patch ...]
python benchmarks/df_j3c_sweep.py --cases ... [--ksh-chunks 4,8,16,32,64]
python benchmarks/coverage_probe.py --xyz ... --basis def2-svp --xc pbe \
    --density-fit        # `fastdf` reads 0 and this reads one call per cycle
```

Switches, which are what every ablation column above is measured with:

| | |
|---|---|
| `FASTDFJ3C_OFF=1` | hand every call back to GPU4PySCF |
| `FASTDFJ3C_NO_CTR=1` | keep GPU4PySCF's recontraction kernels (§2.2) |
| `FASTDFJ3C_NO_OPT=1` | keep GPU4PySCF's two contractions (§2.1, §2.3, §2.4) |
| `FASTDFJ3C_NO_MDM=1` | keep GPU4PySCF's engine choice (§6) |
| `FASTDFJ3C_SP_CHUNK_AUX=<n>` | shell pairs per block of the auxiliary-vector kernel; 0 is GPU4PySCF's whole group (§5) |
| `FASTDFJ3C_SP_CHUNK_DM=<n>` | the same knob on the other kernel, where the measured answer is to leave it at 0 |
| `FASTDFJ3C_KSH_CHUNK=<n>` | auxiliary shells per block; 16 is GPU4PySCF's and is the measured optimum (§5.1) |
