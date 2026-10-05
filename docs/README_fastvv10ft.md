# fastvv10ft, fastvnlch — the VV10 Hessian and response terms (catalogue B13 + B14)

`docs/KERNEL_TARGETS_1.8.1.md` §5.1 rank 1.  The VV10 response pair kernel
`VXC_vv10nlc_hess_eval_f_t` was **71 %** of a patched wB97M-V Hessian at 42
atoms, and Electrolites took that Hessian only 1.12x (§11.2).  This round
replaces the kernel, its grid-weight twin, and the host-side loops around
them.  Measured on one A100-SXM4-80GB, every pair of numbers below back to
back on the same node:

| wB97M-V/def2-SVP, level-3 grids | before this round (Electrolites) | after | |
|---|---|---|---|
| **Hessian, alanine dipeptide (22 atoms)** | 141.5 s (stock 165.5 s) | **45.4 s** | **3.12x** (3.65x on stock) |
| **Hessian, penicillin V (42 atoms)** | 932.6 s | **237.3 s** | **3.93x** |
| **Hessian, Ni dithiolene⁻, UKS doublet (21 atoms)** | 406.2 s | **218.3 s** | **1.86x** |
| **polarizability (CPKS solve), penicillin V** | 93.3 s (stock 150.1 s) | **51.7 s** | **1.80x** (2.90x on stock) |
| **polarizability (CPKS solve), taxol (113 atoms)** | 698.0 s | **359.3 s** | **1.94x** |

The open-shell row is lower because its Hessian is not VV10-bound: after this
round its largest terms are `get_k` (73.6 s) and `jk_ip1` (44.4 s), not the
VV10 kernels (31 s).

Four changes, two modules and two additions to `fastfnlc`:

1. **`fastvv10ft`: the response pair kernel `f_t` on the fp64 tensor cores**
   (§1–§3) — 5.5–6.1x on the kernel at a Hessian's `3 natm` trial densities,
   1.7–2.2x at a polarizability's 3.
2. **`fastvv10ft`: the grid-weight-derivative kernel `EUW_with_weight1`**, the
   same construction (§4) — 50.1 s → 14.5 s in the 42-atom Hessian.
3. **`fastfnlc`: the perturbed densities and the MO projection batched over
   the densities** (§5) — the `return_in_mo` path a Hessian's CPHF takes had
   kept GPU4PySCF's per-density loops.
4. **`fastvnlch`: B14, `_get_vnlc_deriv1`'s atom loops as one GEMM per grid
   block** (§6).

---

## 1. The kernel, and why it was 71 % of a Hessian

For every pair of grid points `(i, j)` the VV10 response needs four kernel
values, and contracts them with the weighted trial densities
`Xr = w ρ_t`, `Xg = w γ_t`:

```
f_ρ[t,i] = Σ_j K_ρρ(i,j) Xr[t,j] + K_ργ(i,j) Xg[t,j]
f_γ[t,i] = Σ_j K_γρ(i,j) Xr[t,j] + K_γγ(i,j) Xg[t,j]
```

GPU4PySCF 1.8.1 (`lib/gdft/vv10.cu:286-369`, launch `:760`) gives each thread
one `i` and **six** trials and loops over every `j`, so every pair's ~60
operations and one fp64 division are evaluated `ceil(ntrial/6)` times — 21
times at a 42-atom Hessian's `ntrial = 126` — and the contraction (4 FMA per
pair and trial) runs on the fp64 SIMT pipe.  At `ntrial = 126` that is
`2.0e11 pairs × (21 × ~60 + 126 × 8) ≈ 4.5e14` flops: 89 s, i.e. the SIMT fp64
peak.  (Q8 of §7 asked whether the compiler or the cache hides the
re-evaluation: it does not.)

**The restructuring is a GEMM whose left operand is built on the fly.**  In an
`mma.m8n8k4.f64` lane `l` holds `A[l>>2][l&3]`; with rows = `i` points and the
k index = `j` points, *the lane that evaluates pair `(i0 + l>>2, j0 + l&3)`
holds exactly that pair's A-fragment element* for each of the four MMAs

```
Cρ += K_ρρ · Xr + K_ργ · Xg        Cγ += K_γρ · Xr + K_γγ · Xg
```

so the kernel values never touch shared memory; only `Xr`, `Xg` (four `j` by
eight trials per fragment) are staged, with a row stride of 4 mod 16 doubles so
a fragment load is two conflict-free wavefronts.  A warp holds `8·MT` points
and `8·NTW` trials, so each pair is evaluated once per `8·NTW` trials instead
of once per 6.  The one division becomes a MUFU.RCP64H seed and two Newton
steps (`fastnlc`'s RCP=1).  Expressions are GPU4PySCF's, regrouped
(`kernels/fastvv10ft.cu:pair_kernel`): `K_ρρ = r_i t_j − r_j ΦP_j + Φ`,
`K_γρ = b_i r² t_j`, `K_ργ = b_j r² t_i`, `K_γγ = b_i b_j r⁴ ΦD`.

The diagonal (`i = j`) term is GPU4PySCF's kernel verbatim.  All four
callers in 1.8.1 (`hessian/rks.py:930, 1188, 2243, 3806`) and `fastfnlc`
look the symbol up on `numint.libgdft` at call time, so the module sets that
one ctypes attribute; a drop-in `_entry` takes the same raw pointers.

## 2. Kernel A/B

`benchmarks/vv10ft_bench.py`, penicillin V/def2-SVP/wB97M-V, level 3,
446 839 NLC points kept, the caller's ground-state arrays, both
implementations in one process, element-wise against GPU4PySCF:

| `ntrial` | GPU4PySCF | fastvv10ft, all fp64 | | + fp32 beyond 10 Bohr (default) | | max rel. err (default) |
|---|---|---|---|---|---|---|
| 3 (polarizability) | 4.025 s | 2.382 s | 1.69x | **1.837 s** | **2.19x** | 9.5e-11 |
| 32 | 25.00 s | 4.579 s | 5.46x | **4.012 s** | **6.23x** | 7.8e-11 |
| 64 | 46.37 s | 8.16 s | 5.68x | | | |
| 126 (Hessian, `3 natm`) | 89.00 s | 15.00 s | 5.93x | **14.51 s** | **6.13x** | 1.1e-10 |

(the fp64 column is 1e-15 against GPU4PySCF everywhere.)  The second size,
taxol (113 atoms, 1 139 586 points kept):

| `ntrial` | GPU4PySCF | fastvv10ft, all fp64 | |
|---|---|---|---|
| 3 | 25.22 s | 15.33 s | 1.65x |
| 32 | 160.65 s | 29.82 s | 5.39x |
| 339 (its Hessian's `3 natm`) | not run (~1700 s by the 32-trial rate) | 290.4 s | ~5.9x |

i.e. the same ratios as on penicillin V to within 3 %: the kernel's cost is
`ngrids² × (evaluations + MMAs)` on both sides, so the ratio depends on the
trial count and not on the grid.

**Tile choice.**  Swept (`--tiles`): `MT, NTW, NW` = m-tiles per warp, trial
tiles per warp, warps per block.  The rule is one trial group when `ntrial`
fits, the smallest group that holds it otherwise: `(2,1,4)` for ≤ 8 trials,
`(1,2,8)` ≤ 16, `(1,4,8)` ≤ 32, `(1,8,8)` beyond, which is within 2 % of the
best of nine tiles at 3, 32, 64 and 126 trials.  The waste is in padded trials
(66 trials in groups of 64 cost as much as 128): it depends on `ntrial` only.
At `ntrial = 126` the kernel runs 2.05e14 MMA flops in 15 s, **13.7 TF of the
A100's 19.5 TF fp64 tensor peak**; the pair evaluation (fp64 SIMT) is what it
does not hide.

## 3. Single precision beyond 10 Bohr

`fastnlc`'s P10, transferred: points Morton-sorted, per (warp, 32-point tile)
bounding boxes, and a (warp, tile) pair whose boxes are more than
`FASTVV10FT_RCUT` (10 Bohr) apart evaluates the kernel values in fp32 (MUFU
reciprocal) — the MMA stays fp64.  The branch is warp-uniform.  It pays where
the evaluation is the cost (few trials: 1.69x → 2.19x at 3) and little where
the MMA is (126: 5.93x → 6.13x).  Error against GPU4PySCF, element-wise,
penicillin V:

| `RCUT` | 3 trials | 32 | 126 | time at 3 |
|---|---|---|---|---|
| ∞ (all fp64) | 8e-16 | 1e-15 | 1.4e-15 | 2.382 s |
| 15 Bohr | 7.6e-12 | 9.8e-12 | | 2.132 s |
| **10 Bohr (default)** | **9.5e-11** | **7.8e-11** | **1.1e-10** | **1.837 s** |
| 6 Bohr | 4.2e-10 | 4.0e-10 | 6.2e-10 | 1.515 s |
| 4 Bohr | 1.0e-9 | 1.1e-9 | | 1.406 s |

Unlike `fastnlc`'s energy, this is a response operator, so the error is also
measured on what it feeds: the alanine-dipeptide Hessian with the default
against stock differs by **1.6e-11** relative and its 60 harmonic frequencies
by **< 0.0001 cm⁻¹**; with `RCUT = ∞` the same Hessian differs from stock by
3.2e-7 (0.0008 cm⁻¹), which is GPU4PySCF's own run-to-run scatter — two
stock Hessians of water in one process differ by up to 5.6e-7
(`tests/test_vv10ft.py` prints that beside every comparison).  The
polarizabilities agree to 4e-16.  `FASTVV10FT_RCUT=inf` is the all-fp64
kernel.

## 4. `EUW_with_weight1`

`_get_vnlc_deriv1` always computes the grid response (`grid_response = True`
is hard-coded, `hessian/rks.py:2053`, for the reason its comment gives), and
part of that is `VXC_vv10nlc_hess_eval_EUW_with_weight1` (`vv10.cu:504`):
three pair kernels `E_ij = −ρ_j/(g_ij g_ji s)`, `U_ij`, `W_ij = U_ij r²`
contracted with the `3 natm` grid-weight derivatives, with the same
six-per-thread structure.  It was **50.1 s of the 42-atom Hessian** once `f_t`
was fixed.  Same construction, three MMAs per (pair quad, eight derivatives):
**50.1 s → 14.5 s** (3.5x; its MMA count per pair is ¾ of `f_t`'s and its
evaluation is cheaper, so the evaluation share is larger).  Element-wise
against GPU4PySCF: 2–4e-15 all-fp64; with the default fp32 far field up to
2.2e-9 on random weight derivatives (its terms carry `ρ_j` rather than a
density derivative, so more of it is far field) — the Hessian numbers of §3
include it.

## 5. `fastfnlc`: the densities and the MO projection, batched

With `f_t` at 15 s, the rest of `nr_rks_fnlc_mo` in a 42-atom CPHF was **98.7 s**
of 185.3 s.  `fastfnlc`'s batched contraction (its §3) had been declined on
the `return_in_mo=True` path — the one a Hessian's CPHF takes — and both paths
kept GPU4PySCF's per-density `eval_rho` loop: `n_dm × nblocks` pairs of small
GEMMs.  Two changes:

* **the perturbed densities for the whole batch**: with `S = D + Dᵀ`,
  `ρ = ½ ao₀·(S ao₀)` and `∇ρ = ∇ao·(S ao₀)` is exactly `eval_rho(..., hermi=0)`,
  and `S ao₀` for every density is one GEMM per block
  (`FASTFNLC_RHO_BATCH=0` is the ablation);
* **the MO projection batched**: the weighted-AO GEMM of `fastfnlc` §3 now
  also serves `return_in_mo=True`, followed by `C[idx]ᵀ (V Cocc[idx])` for all
  densities as two contractions (`FASTFNLC_MO_BATCH=0`).

`nr_rks_fnlc_mo` in the 42-atom Hessian: 185.3 s → **100.8 s**, of which 86.6 s
is now `f_t`.  The polarizability gains from the first change too.  The `mo`
path is no longer bit-identical (it reassociates the same sums):
`tests/test_fnlc.py` and `tests/test_ufnlc.py` now check it to round-off
(≤ 1.2e-14 over 15 cases) and still check bit-identity with both switches off.

## 6. `fastvnlch`: B14

`_get_vnlc_deriv1` has two grid-block loops, and inside each block a loop over
the atoms issuing ~10 contractions per atom (5 in the grid-response loop) into
a dense `(natm, 3, nsub, nocc)` block, projected onto the MO basis per block.
The per-atom sums regroup by the AO factor they end on:

```
dF[p,(A,d),j] = Σ_g B_c[p,g] L_c[g,(A,d),j],    B = [μ, ∇μ·∇ρ, ∂xμ, ∂yμ, ∂zμ]
L_μ  = f_ρ^A w_μ + 2 f_γ^A (∇ρ·w∇μ) + 2 Σx ∂(∇xρ)/∂A · f_γ w∂xμ
L_dd = 2 f_γ^A w_μ        L_x = 2 ∂(∇xρ)/∂A · f_γ w_μ
```

so a block is **one GEMM** `(nsub, 5 ng) × (5 ng, 3natm·nocc)` (the
grid-response loop: two components), `L` built by broadcasting and chunked
over the atoms to stay under `FASTVNLCH_LBYTES` (256 MB); the result is
accumulated in the AO basis and projected once.  Everything else in the
function is GPU4PySCF's code, copied from 1.8.1.  Both spin cases and the
density-fitted Hessians (which bind the name at import) are covered:
`tests/test_vnlch.py`, 5 cases, ≤ 2.6e-15.  42 atoms: the function's time outside the
four `libgdft` kernels 24 s → **5.5 s** (37.5 s total, of which `f_t` 14.2 s,
`EUW` 14.5 s).

## 7. End to end

`benchmarks/path_anatomy.py --feature hess --xc wb97m-v --basis def2-svp`,
level 3, stages in seconds:

| | ala. dipeptide, before | after | penicillin V, before | after |
|---|---|---|---|---|
| **Hessian** | **141.5** | **45.4** | **932.6** | **237.3** |
| `f_t` (all callers) | 88.2 | 22.6 | 671.8 | 115.0 |
| `make_h1` | 28.0 | 12.4 | 175.4 | 46.4 |
| `solve_mo1` | 96.9 | 24.4 | 647.4 | 156.0 |
| ↳ `nr_rks_fnlc_mo` | 90.3 | 17.7 | 591.7 | 100.8 |
| `enlc_deriv2` | 12.7 | 4.9 | 91.8 | 16.8 |
| `get_k` + `get_j` | 7.8 | 7.7 | 55.6 | 55.7 |

The J/K rows, which this round does not touch, are the same to 0.2 %, which is
the check that the node's load did not move.  The stepwise ablation on the
42-atom Hessian: `f_t` alone 932.6 → 376.0 s; + §5 → 292.6 s; + §4 and §6
→ 237.3 s.

Ni dithiolene⁻ (UKS doublet, 21 atoms, Ni at def2-SVP): 406.2 → 218.3 s;
`f_t` 162.8 → 28.2 s.  Hessian traces equal to 9 digits in every pair.

## 8. Correctness

| test | what | result |
|---|---|---|
| `tests/test_vv10ft.py` part 1 | `f_t` element-wise, 5 cases (3 functionals, a third-row atom, grid level 1), `ntrial` 1, 5, 8, 13, 70 | fp64 ≤ 9.2e-16; default ≤ 9.5e-11 |
| | `EUW_with_weight1`, `nder` 3, 13, 70 | fp64 ≤ 3.5e-15; default ≤ 2.2e-9 |
| part 2 | RKS and UKS wB97M-V Hessian, wB97X-V Hessian, polarizabilities, symbol swapped in-process | ≤ 5.7e-7 Hessian (stock's own scatter 1e-8–5.6e-7), ≤ 4.4e-16 polarizability |
| `tests/test_vnlch.py` | `_get_vnlc_deriv1`, RKS ×3, UKS, DF | ≤ 2.6e-15 |
| `tests/test_fnlc.py`, `tests/test_ufnlc.py` | batched density and projection | ≤ 1.2e-14; bit-identical with both off; UKS TDA 3.6e-14 Eh |
| `benchmarks/hess_save.py --compare` | alanine dipeptide frequencies, stock vs patched | ≤ 0.0001 cm⁻¹ (default), 0.0008 cm⁻¹ (all fp64, = stock's scatter) |

## 9. What depends on size and class

* **Trial count**: the tile and the gain (1.7x at 3, 6x at 126) are set by
  `ntrial`, which is `3 natm` in a Hessian and 3 in a polarizability; the rule
  in §2 picks the tile from it.
* **Grid size**: none of the kernel's choices depends on it; the kernel's ratio on
  taxol (2.6x the points) is within 3 % of penicillin V's (§2).
* **Spin**: the kernel sees the total density; the UKS Hessian reaches it
  (10 calls) and the batched host code covers both spins (`test_vnlch` UKS).
* **The far-field split** depends on the molecule's extent: a compact system
  (water, the Ni complex) has few far pairs and runs essentially all fp64; it
  never changes which code runs, only the precision of distant pairs.
* **Memory**: the reordered, weighted copy of the trial densities is chunked
  at 128 trials (`FASTVV10FT_CHUNK`); `fastvnlch`'s operand at 256 MB.

## 10. What is left

* **Hessian**: at 42 atoms it is now `f_t` 48 % (115 s, MMA-bound at 13.7 of
  19.5 TF), J/K 23 %, `EUW` 6 %, the rest of the VV10 terms ~10 %.  The next
  step on `f_t` would be 3xTF32 tensor cores for the far field (2.7x the fp64
  tensor rate at fp32 accuracy), which this round did not try.
* **Polarizability**: `nr_rks_fnlc_mo` is still 79 % of the taxol CPKS solve
  (268 s of 359 s, 11.7 s a call): at 3 trials `f_t` is bound by the pair
  *evaluation*, which the MMA does not touch, and the fp32 far field takes
  it 1.69x → 2.19x.  The same 3xTF32 route, or `fastnlc`'s grid trim (P11,
  `FASTFNLC_BUDGET`, off by default for the reason `README_fastfnlc.md` §4d
  gives), are what is left.

(The polarizability rows are the CPKS solve, with the SCF — 17.4 s and 74.3 s,
the same in both columns — excluded.)

## Files

| file | what |
|---|---|
| `electrolites/kernels/fastvv10ft.cu` | `f_t` and `EUW_with_weight1` kernels, fp64 and mixed |
| `electrolites/fastvv10ft.py` | the two drop-in entry points, the sort, the tile rule |
| `electrolites/fastvnlch.py` | `_get_vnlc_deriv1` with batched atom loops (B14) |
| `electrolites/fastfnlc.py` | §5's two changes |
| `benchmarks/vv10ft_bench.py` | §2–§3 |
| `benchmarks/hess_save.py` | the frequency comparison |
| `tests/test_vv10ft.py`, `tests/test_vnlch.py` | §8 |

## Environment switches

| variable | effect |
|---|---|
| `FASTVV10FT_OFF=1` | leave both `libgdft` symbols alone |
| `FASTVV10FT_RCUT` | fp32 beyond this box separation (Bohr); default 10, `inf` = all fp64 |
| `FASTVV10FT_MT`, `_NTW`, `_NW`, `_JT`, `_CHUNK` | tile override, j tile, trials per launch |
| `FASTVNLCH_OFF=1`, `FASTVNLCH_LBYTES` | B14 off; size of its batched operand |
| `FASTFNLC_RHO_BATCH=0`, `FASTFNLC_MO_BATCH=0` | §5's ablations |
