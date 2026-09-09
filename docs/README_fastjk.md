# fastjk — the fused Coulomb-and-exchange build, and the third code path

Catalogue entry **B4b**, rank 8 in `instructions/06`'s §5 list, paired there
with **B4c**.  The entry names "K in CPHF: multi-density reads and atomics" and
points at `unrolled_rys_k.cu`'s `i_dm` loop.  Two things about that turn out to
matter more than the entry says, and they split this round in two:

1. **There are two multi-density exchange builds in 1.8.1, not one.**  An
   analytic Hessian's CPHF solve calls `jk._VHFOpt.get_k` directly
   (`hessian/rhf.py:857`, `n_dm = 3·natm`); every *response property* calls
   `jk._VHFOpt.get_jk`, a **fused** kernel that builds J and K from one pass
   over the Rys integrals.  `fastk` used to decline both — its
   `if n_dm != 1: return False` — and no module here patched `get_jk` at all.
   `fastk` now serves the first (see §7); this module is the second.
2. **The two regimes want different things.**  At the `n_dm = 1`–10 a response
   solve asks for, the integrals dominate and the P1–P5 treatment is worth
   1.4–2.8x.  At the `n_dm = 126`–339 a Hessian's CPHF asks for, the density
   contraction dominates and the same treatment is worth 1.16x.  §6 measures
   both halves of that split, and §8 is what the entry's other half (B4c) turns
   out to be worth once measured.

---

## 0. Summary

* `fastjk` replaces `scf/jk.py:_VHFOpt.get_jk` for **every angular-momentum
  class an spdf basis reaches** and for both `hermi = 1` (CPKS, NMR, IR,
  Raman, C6) and `hermi = 0` (every TDDFT/TDA call).  Nineteen classes run one
  thread per shell quartet, six spread their `gout` over `threadIdx.y` — both
  groups lifted from `gvhf-rys/unrolled_rys_jk.cu` — and forty-five are written
  from `(li,lj,lk,ll)` alone.
* Against GPU4PySCF's own fused kernel, both in one process on one density,
  penicillin V/def2-SVP: **2.79x** at `nset = 1`, **1.96x** at 3, **1.65x** at
  5 with `hermi = 1`; **2.30x**, **1.53x**, **1.36x** with `hermi = 0`.
* End to end, whole runs: a B3LYP polarizability on penicillin V goes
  **16.77 s → 8.06 s (2.08x)** and a five-root TDA **22.21 s → 14.26 s
  (1.56x)**, against 1.15x and 1.24x for the same two runs before this round.
  The polarizability tensor and the five excitation energies are unchanged to
  every digit printed.
* **The fused arrangement is now the right one at every size, which it was
  not before.**  `docs/RESPONSE_1.8.1.md` §4 measured a split into
  `fastj` + `fastk` at 2.30x of the fused kernel at one density and 0.73x at
  three, and concluded that a threshold between the two would be tuned to one
  molecule.  A fused rewrite removes the choice: it beats *both* arrangements
  at every row measured here.
* **Nothing about the SCF changes.**  `scf/hf.py:get_veff` asks for `get_j`
  and `get_k` separately, so this module is not on the SCF path at all and
  cannot slow it down.

---

## 1. The third code path

Three different builds of the two-electron matrices exist in GPU4PySCF 1.8.1,
and until this round `Electrolites` was on two of them:

| caller | entry point | kernel | patched by |
|---|---|---|---|
| SCF (`scf/hf.py:get_veff`, `dft/rks.py:get_veff`) | `j_engine._VHFOpt.get_j` | McMurchie–Davidson `MD_build_j` | `fastj` |
| " | `jk._VHFOpt.get_k` | Rys `RYS_build_k` | `fastk` |
| Hessian CPHF (`hessian/rhf.py:_get_jk_mo`) | the same two, at `n_dm = 3·natm` | the same two | `fastj`; `fastk` **only since this round** |
| **every response property** (`scf/_response_functions.py`) | **`jk._VHFOpt.get_jk`** | **fused `RYS_build_jk`** | **`fastjk`** |

The last row is `docs/CATALOGUE_06_STATUS.md` §6's hazard 5 — *a patch target is
not a code path*.  A B3LYP polarizability or TDA run with `patch_all()` was
stock GPU4PySCF in 61–82 % of its wall clock, with no test failing and no
module reporting anything, because the modules simply were not called.

## 2. Why fused rather than split

The fused kernel exists because J is nearly free once the integrals have been
computed for K.  On stock code that is decisive: `docs/RESPONSE_1.8.1.md` §4
measured the fused build at 0.522 s against 0.679 s for a separate J plus a
separate K at one density, and the gap widens with `nset`.

Splitting only looked attractive because `fastk` at one density was 3.3x
GPU4PySCF's K while `fastj` was 2.6x its J — and that advantage evaporated at
`n_dm > 1`, where `fastk` fell through entirely.  With the multi-density
exchange kernels of §7 in place the split is real but modest, and a fused
rewrite is better than either arrangement:

penicillin V/def2-SVP, one converged density with realistic perturbed
densities, `min` of three, everything in one process
(`benchmarks/response_jk_probe.py --patch all`):

| `hermi` | `nset` | **`fastjk`** | GPU4PySCF fused | `fastj` + `fastk` | ↳ `fastj` | ↳ `fastk` | `fastjk` over stock |
|---|---|---|---|---|---|---|---|
| 1 | 1 | **0.181** | 0.505 | 0.225 | 0.067 | 0.157 | **2.79x** |
| 1 | 3 | **0.276** | 0.541 | 0.430 | 0.222 | 0.208 | **1.96x** |
| 1 | 5 | **0.377** | 0.623 | 0.522 | 0.256 | 0.267 | **1.65x** |
| 0 | 1 | **0.229** | 0.526 | 0.253 | 0.067 | 0.183 | **2.30x** |
| 0 | 3 | **0.433** | 0.661 | 0.515 | 0.219 | 0.296 | **1.53x** |
| 0 | 5 | **0.645** | 0.880 | 0.672 | 0.254 | 0.418 | **1.36x** |

Two things to read off it.  The fused kernel wins over the split at every row
(the split is 1.24x–1.56x slower at `hermi = 1`), so **the arrangement no
longer has to be chosen per molecule** — which was the objection
`docs/RESPONSE_1.8.1.md` §4 raised against a threshold. And the ratio falls
with `nset`, for the reason §6 gives: what a rewrite of these kernels buys is
the per-quartet integral work, and the share of the total that is per-quartet
work falls as `1/n_dm`.

## 3. What the kernels are

Three families, all with the same signature and the same dispatch path, and all
carrying the density loop *inside* the shell quartet so that `n_dm` densities
cost one integral pass:

| family | classes | written by | shape |
|---|---|---|---|
| `jk_<class>` | 19 (`(ss\|ss)` through `(fs\|ds)`) | `codegen/gen_kernels.py --jk` | one thread per shell quartet, 2D integrals and Rys roots in registers |
| `jk2_<class>` | 6 | `codegen/gen_k2_kernels.py --jk` | `gout` over `threadIdx.y`, 2D integrals in shared memory |
| `jkh_<class>` | 45 | `electrolites/codegen/gen_khigh.py --jk` | the compact one-body-per-lane scheme, written from the class |

The first two **lift** GPU4PySCF's own integral arithmetic out of
`unrolled_rys_jk.cu` and rewrite only the scaffolding around it, exactly as
`fastk`'s two lifted families do out of `unrolled_rys_k.cu`.  That works because
the two GPU4PySCF files' Rys arithmetic is *textually identical* — the same
`gout` registers, the same root loop, character for character — and only the
contraction differs.  `gen_kernels.assert_same_integrals` checks that rather
than trusting it, so a future release that changes one file and not the other
fails the build instead of silently producing a kernel that computes something
else.

What the scaffolding changes is `fastk`'s P1–P5 list, unchanged:

* **P1.** One `rsqrt` and two cached reciprocals in place of five fp64
  divisions and a `sqrt` per primitive quartet.  `1/aij` comes from the bra
  cache, `1/akl` from the ket, and one `rsqrt(fma(aij*akl, inv_om2, aij+akl))`
  supplies the geometric prefactor, `rt_aa`, `b10` and `b01`.
* **P2.** The Rys roots in registers for the one-thread-per-quartet classes,
  which removes the `__syncthreads()` GPU4PySCF executes once per *primitive
  quartet* to protect its shared `rw` buffer.
* **P3.** A per-class `__launch_bounds__`.  GPU4PySCF compiles these with no
  register cap at all.
* **P4.** The block-uniform bra data (`aij`, `1/aij`, `aj/aij`, the Gaussian
  product centre, `cicj`) computed once per bra shell pair instead of once per
  thread per primitive quartet.
* **P5.** `omega` compiled in, so the range-separation branches of the root
  evaluation disappear at `omega = 0`.

The forty-five written classes get the same treatment plus the one that matters
most above `(pp|pp)`: **exact tiling and compile-time g-array addresses.**
GPU4PySCF's fused build has no unrolled kernel past `(fs|ds)`; everything else
goes to a general kernel whose `gout` tile is 3×3×3×3 whatever the class and
whose addresses come from shared `idx_i/j/k/l` arrays.
`docs/README_fastk.md` §4 measures what that costs on the exchange side.

## 4. The J half of the contraction

Six contractions per shell quartet instead of four:

```
vk[i,l] += Σ_jk gout·dm[j,k]      vk[j,l] += Σ_ik gout·dm[i,k]
vk[i,k] += Σ_jl gout·dm[j,l]      vk[j,k] += Σ_il gout·dm[i,l]
vj[i,j] += Σ_kl gout·dm[l,k]      vj[k,l] += Σ_ij gout·dm[j,i]
```

The index order of the two J density reads is GPU4PySCF's own and is not
cosmetic: at `hermi = 0` the caller does not symmetrise the density, it *stacks
the transposed densities* (`jk.py:443`) so that a tril-only quartet enumeration
reaches both triangles, and `dm[l,k]` and `dm[j,i]` are then different numbers
from `dm[k,l]` and `dm[i,j]`.  The written classes emit those two terms grouped
the same way the four exchange terms are: one `atomicAdd` per output element
per density, with the summands a compile-time subset of the lane's `PER`
combinations, so one body still serves every lane.

Screening changes with them.  J reads two shell-pair blocks of the density that
K does not, so the task filter gains two `dm_cond` tests
(`_fill_vjk_tasks` against `_fill_vk_tasks` in `create_tasks.cu`, reproduced as
`fill_vjk_tasks`/`fill_vjk_tasks2` in `kernels/fastjk_prologue.cu`).  The fused
filter keeps a **superset** of what the exchange-only filter keeps, so a fused
build cannot drop a quartet that either separate build would have kept.

## 5. Correctness

The oracle is GPU4PySCF's own `RYS_build_jk` on the same molecule with the same
screening data — an element-wise, fixed-density comparison of the AO-space J and
K, which is what the catalogue's §6 asks for on an integral kernel.  A converged
energy would not be a useful signal: this entry point is only reached inside a
CPHF or Davidson solve, never on the SCF's own path.

`tests/test_jk.py`, six molecule/basis combinations to cc-pVTZ, `hermi` 1 and 0,
`n_dm` 1 to 8, three density shapes (identical copies, random perturbations of
five different magnitudes, atom-local blocks):

```
worst relative difference: K 1.040e-14, fused J 2.050e-14, fused K 1.040e-14
```

against a 1e-12 tolerance.  The identical-copies case additionally requires the
`n_dm` matrices to come back the same to round-off, which is what catches a
wrong per-density stride — the failure mode a single random draw can hide.

End to end, on penicillin V/def2-SVP: the polarizability tensor's trace and
Frobenius norm agree to all thirteen digits printed
(6.175651783180e+02, 3.691631074435e+02), and the five TDA excitation energies
to all nine decimals.

## 6. What it is worth end to end, and why the ratio falls with `n_dm`

Whole runs, penicillin V/def2-SVP, level-3 grids, `conv_tol = 1e-9`, CPKS
tolerance 1e-7, one A100-PCIE-40GB (`benchmarks/response_anatomy.py`).  Stock
and patched are separate processes run back to back, for the shared-node reason
`docs/RESPONSE_1.8.1.md` §4b gives.

| stage | polarizability, stock | + Electrolites | | TDA (5 roots), stock | + Electrolites | |
|---|---|---|---|---|---|---|
| **total** | **16.768 s** | **8.062 s** | **2.08x** | **22.207 s** | **14.264 s** | **1.56x** |
| the matrix–vector product | 16.135 | 7.410 | 2.18x | 20.232 | 12.245 | 1.65x |
| ↳ J + K response (`get_jk`) | 11.758 | 5.374 | **2.19x** | 13.529 | 10.168 | **1.33x** |
| ↳ XC response (`nr_rks_fxc*`) | 4.352 | 2.000 | 2.18x | 6.376 | 1.728 | 3.69x |

The J+K row is this module; the XC row is `fastxcfxc` from the previous round.
Before this round the same two runs were 1.15x and 1.24x, with the J+K row
flat to ±0.6 % — which is the measurement that made §1's point.

The same two properties on taxol (113 atoms, `nao = 1123`), which is the
second size catalogue §3.4 requires:

| stage | polarizability, stock | + Electrolites | | TDA (5 roots), stock | + Electrolites | |
|---|---|---|---|---|---|---|
| **total** | **101.082 s** | **75.060 s** | **1.35x** | **189.658 s** | **148.668 s** | **1.28x** |
| ↳ J + K response (`get_jk`) | 82.719 | 62.281 | **1.33x** | 165.572 | 136.724 | **1.21x** |
| ↳ XC response (`nr_rks_fxc*`) | 17.295 | 11.722 | 1.48x | 20.896 | 8.966 | 2.33x |

Both sizes gain and the ratio falls with size, by a factor the same cost model
accounts for: §7's fit puts `R_stock/R_ours` at 3.55x on penicillin V and
2.36x on taxol, and at the `nset` a response solve uses that difference is most
of the gap between 2.08x and 1.35x.  The polarizability trace and Frobenius
norm agree to thirteen digits at this size too
(1.601787534941e+03, 9.365475710724e+02) and the excitation energies to nine
decimals.

**Why the polarizability gains more than the TDA run.**  A CPKS solve asks for
`nset` = 3, then 2, then 1 as the three dipole directions converge, all at
`hermi = 1`; a five-root Davidson asks for `nset = 5` at `hermi = 0`, which the
kernel sees as `n_dm = 10`.  The cost model is

    T(n_dm) = R + n_dm·C + L

with `R` the per-quartet integral work (the Rys roots, the 2D recurrences, the
geometry, the screening), `C` the per-density contraction and its atomicAdds,
and `L` a launch constant.  These kernels change `R` and leave `C` where it
was, so the ratio is bounded by

    (R_stock + n·C) / (R_ours + n·C)  ->  1  as  n -> infinity.

That is not a defect of the implementation; it is what the entry's mechanism
can reach.  §7 is the same statement at the other end of the `n_dm` range, and
§8 is the measurement of the one part of rank 8 that was supposed to attack `C`
itself.

## 7. The other half: `get_k` at `n_dm = 126`

`fastk` no longer declines a multi-density call.  The change is small — the
`for (int i_dm = 0; i_dm < n_dm; ++i_dm)` loop that GPU4PySCF wraps its
contraction in, restored around ours in all three families, with the density
offset widened to `size_t` because `i_dm * nao * nao` overflows a signed int at
339 densities on a 2268-AO basis — and it also, as a side effect, makes
`hermi = 0` work: `fastk` was declining that only because the caller doubles
`n_dm` to reach both triangles.

What it is worth on an analytic Hessian, penicillin V/def2-SVP,
`conv_tol_cpscf = 1e-6`, everything else patched, the two runs back to back in
one script (`FASTK_NO_NDM=1` is the ablation, i.e. this repository before the
round):

| stage | before | after | |
|---|---|---|---|
| **whole Hessian** | **56.49 s** | **53.36 s** | **1.06x** |
| `solve_mo1` | 41.09 | 38.33 | 1.07x |
| ↳ `get_k`, `n_dm` ≤ 126 | **19.65** | **16.96** | **1.16x** |
| ↳ `get_j`, `n_dm` ≤ 126 | 13.14 | 13.12 | 1.00x |
| `..ejk_ip2` (catalogue B3) | 7.45 | 7.43 | 1.00x |

The two Hessians agree to 1.2e-9 absolute on the trace
(4.181647004638e+01 against 4.181647004519e+01), which is 2.8e-11 relative and
is the same `atomicAdd`-and-CPHF floor `docs/HESSIAN_1.8.1.md` reports.

**1.16x, against 1.96x for the same kernels at `nset = 3`.**  That is §6's
`R/(R + n·C)` seen from the other end, and `benchmarks/ndm_k.py` measures both
halves directly rather than inferring them.  Fitting `T = base + n_dm·C` on one
fixed density, both implementations in one process:

| | GPU4PySCF | ours | |
|---|---|---|---|
| penicillin V, `base` | 0.4019 s | 0.1133 s | **3.55x** |
| penicillin V, `C` | 0.02775 s/density | 0.02759 s/density | **1.01x** |
| taxol, `base` | 1.0181 s | 0.4306 s | 2.36x |
| taxol, `C` | 0.48109 s/density | 0.41634 s/density | 1.16x |

**`C` is the same code in both.**  The P1–P5 treatment applies to integrals and
does not touch the contraction, so the asymptote of this entry as `n_dm → ∞` is
1.01x at 42 atoms and 1.16x at 113 — and every ratio quoted anywhere in this
file is `(base_stock + n·C)/(base_ours + n·C)` interpolating between 3.3x and
that.  The measured series runs 3.28x, 2.90x, 2.35x, 1.76x, 1.43x, 1.26x, 1.18x
at `n_dm` = 1, 2, 4, 8, 16, 32, 64.

**The second size, for this half, is the fit and not a whole run.**
Catalogue §3.4 asks for a differently-sized system, and the taxol column of
`benchmarks/ndm_k.py` above is it: 1.75x at `n_dm = 1` falling to 1.20x at 32,
`base` 2.36x, `C` 1.16x, i.e. the same shape one size up.  A whole 113-atom
Hessian A/B is **not** in this round -- it is a 25-minute run per half and the
pair was lost twice to node conditions -- so the only end-to-end Hessian number
here is the 42-atom one.  What `docs/HESSIAN_1.8.1.md` §4 implies for the larger
size is that `get_k` is over half the wall clock there, so 1.16x on it would be
about 1.08x on the run; that is an inference, not a measurement, and is written
here as one.

That settles catalogue **Q4** — "at large `n_dm` the bottleneck moves to
uncoalesced density reads and `atomicAdd` contention", listed as `[?]` inferred
from code shape and never confirmed — in the sense that decides what to do
next: at `n_dm = 126` the contraction is 94 % of the build, so **no treatment
of the integrals can be worth more than 1.06x there.**  Which half of the
contraction it is, the strided reads or the atomics, is half-answered:
`benchmarks/ndm_layout_probe.py` measures the density-major layout
`dm[(i,j),i_dm]` — which would turn 126 cache lines per density-matrix element
into eight — at **0.30x–0.38x**, i.e. three times *worse*, because it trades a
stride between densities for a stride between quartets and there are four
orders of magnitude more quartets than densities.  `docs/ROUND4_B4B.md` §7 has
that measurement; the `atomicAdd` traffic is what is left of Q4.

## 8. B4c: the entry's premise, measured

Rank 8's other half, **B4c**, is the one the catalogue calls "algorithmic — the
hardest and the most valuable".  Its claim: `scf/jk.py`'s `dm_cond` is one
`nbas × nbas` table, the max over all `n_dm` densities, so the screening is a
union, and "each `dm1[A]` has support only near atom A, yet every density pays
the full quartet list".

The claim about the code is exactly right.  The claim about the densities is
not.  `benchmarks/ndm_screen_probe.py` takes the densities a **real** solve
hands to `get_k`, builds each one's own `dm_cond`, and counts, over the
quartets the union screen keeps, how many densities would survive a screen
built from their own table:

| system | call | `n_dm` | union quartets | (quartet, density) kept | fraction | cost-weighted |
|---|---|---|---|---|---|---|
| glycine/6-31G\*, Hessian | 1 | 30 | 3.19e5 | 9.38e6 | 0.979 | 0.990 |
| " | 2 | 27 | 3.19e5 | 8.39e6 | 0.975 | 0.989 |
| penicillin V/def2-SVP, Hessian | 1 | 126 | 6.02e7 | 6.61e9 | **0.870** | 0.877 |
| " | 2 | 123 | 5.81e7 | 6.08e9 | **0.850** | 0.858 |

So a *perfect* per-density screen would remove 12–15 % of the contraction at 42
atoms and 1–2 % at 10 atoms — **1.14x–1.18x on the contraction, not an
algorithmic gain**, and that is a ceiling with no implementation cost charged
against it.

The reason is structural, and it is why the entry's premise is wrong rather
than merely optimistic.  1.8.1's CPHF solve runs **in the MO basis**:
`_get_jk_mo` builds its AO densities as `C·mo1·C_occ^T`, and the canonical MOs
are delocalised, so `dm1[A]` has support wherever the occupied and virtual
orbitals do — not near atom A.  The entry describes the *right-hand sides*
`h1`, which are atom-local, and not the perturbed densities the iterations
actually contract.  The trend across the two sizes (0.98 → 0.87) says the
ceiling grows slowly with system size, so it is worth re-measuring at 500+
atoms before the entry is closed, but not worth building at these sizes.

Two further notes from the same probe.  The `cphf` and `tda` stages of it
return **no rows at all**, because those paths never call `get_k` — they call
`get_jk`, which is §1 again, observed from a third direction.  And a
per-density screen would not be free: it needs an `n_dm × nbas × nbas` float
table (20 MB at penicillin V's size, 340 MB at taxol's) and one extra load per
(quartet, density) inside the contraction, against the 12–15 % it could save.

## 9. Limits, all of which fall through rather than fail

* **Single GPU.**  `num_devices > 1` hands the call back; GPU4PySCF 1.8.x
  spreads this build over devices with `multi_gpu.run` and every module here
  asserts one device.
* **The short-range (erfc) operator** is handed back, as in `fastk` — it needs
  twice the Rys roots and the `s_estimator` screening.  A range-separated
  hybrid's response path asks `get_jk` for the *full-range* operator and takes
  its long-range exchange from a separate `get_k`, so `omega < 0` does not
  arise there in practice; `omega > 0` is served through the same single `fma`
  `fastk` uses.
* **`l > 3` on any centre** goes to GPU4PySCF's general fused kernel, one class
  at a time.  Unlike `fastk` there is no faster copy of that kernel here yet.
* **The `h_shls` CPU path** (`l > LMAX`) is handed back whole.
* **`FASTJK_OFF=1`** hands every call back, `FASTJK_NO_HIGH=1` only the
  written classes and `FASTJK_NO_LIFTED=1` only the lifted ones; those three
  are the ablations behind §2's table.
