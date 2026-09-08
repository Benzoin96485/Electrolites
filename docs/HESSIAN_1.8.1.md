# Where an analytic Hessian's wall clock goes on GPU4PySCF 1.8.1

The catalogue's Hessian profile (`instructions/06` §2.3) is a **1.7.0**
profile, and 1.7.0 → 1.8.1 rewrote the part of the Hessian it is mostly about.
Before taking any Hessian entry from the catalogue's §5 ranking, the shares
have to be re-measured; §5's ordering of the Hessian entries is built from that
table.  `benchmarks/hessian_anatomy.py` is that re-measurement.  Everything
below is one A100, GPU4PySCF 1.8.1, B3LYP/def2-SVP, `grids.level = 3`,
`conv_tol = 1e-9`, `conv_tol_cpscf = 1e-6` (GPU4PySCF's own default), on the
catalogue's own two molecules — penicillin V (42 atoms, 430 AOs) and taxol
(113 atoms, 1123 AOs), built from SMILES with RDKit ETKDGv3 + MMFF94 and kept
in `benchmarks/molecules/`.

Every row is wrapped with a `cudaDeviceSynchronize()` on both sides.  Rows
prefixed `..` are sub-intervals of their parent and overlap it, so a column
does not sum to its parent; the `accounted` line sums only the top-level rows.

---

## 1. What 1.8.1 changed about the Hessian

Three of the catalogue's Hessian entries are about code that no longer exists
in this form:

| catalogue | 1.7.0 | 1.8.1 |
|---|---|---|
| **B6** `hess_nuc_elec`: "one charge at a time", 4.5 % of a Hessian, "carries its own TODO; smallest change in the list" (rank 10) | `hessian/rhf.py:746`, an outer `for j_atom` passing one charge to `int1e_grids_ip1ip2` | **gone.**  `hessian/rhf.py:200 _hess_nuc_without_ecp` replaces it, and the `# TODO: It is computing one charge at a time` comment with it.  It does not appear in either profile below |
| **B4a/B4b** the multi-density J and K builds in CPHF | `solve_mo1` built AO-basis perturbed densities and called `get_jk` | the CPHF solve is in the **MO basis**: `solve_mo1` → `gen_vind` → `get_veff_resp_mo` → `_get_jk_mo`, which is where `get_j`/`get_k` are now called, and `nr_rks_fxc_mo` alongside them |
| **B4d** `nr_rks_fxc`: "per-block GEMMs × nset, plus a Python loop", "the ~25 s unattributed inside the Hessian's CPHF is inferred to be here" (catalogue Q3) | `dft/numint.py:1305 nr_rks_fxc` | `hessian/rks.py:3486 _nr_rks_fxc_mo_task`, which contracts to the MO basis at the end of each block group.  Instrumented rather than inferred below: it is **6.5 %** of a penicillin V Hessian and **5.9 %** of a taxol one, not the ~25 s (11 %) the catalogue infers |

## 2. Penicillin V, 42 atoms, 430 AOs

`stock` is GPU4PySCF 1.8.1.  `+ Electrolites` is `patch_all()` **without**
`fastxchess` — i.e. what this repository had before this round; the only module
of the eleven that reaches a Hessian at all is `fastj`, through the
multi-density Coulomb build in CPHF.  `+ fastxchess` adds this round's module.

| stage | stock | share | fastxchess only | + Electrolites | share | all twelve | share |
|---|---|---|---|---|---|---|---|
| **TOTAL** | **84.26 s** | 100 % | **70.87 s** | **68.12 s** | 100 % | **53.93 s** | 100 % |
| `partial_hess_elec` | 16.54 | 19.6 % | 8.93 | 16.54 | 24.3 % | 8.95 | 16.6 % |
| `..ejk_ip2` (B3) | 7.42 | 8.8 % | 7.41 | 7.47 | 11.0 % | 7.39 | 13.7 % |
| `..exc_deriv2` | 8.33 | 9.9 % | 1.16 | 8.63 | 12.7 % | 1.14 | 2.1 % |
| `....vxc_diag` | 0.37 | 0.4 % | 0.37 | 0.39 | 0.6 % | 0.37 | 0.7 % |
| `....vxc_deriv2` (**B5**) | 7.86 | 9.3 % | **0.70** | 8.14 | 11.9 % | **0.68** | 1.3 % |
| `make_h1` | 11.81 | 14.0 % | 5.90 | 11.84 | 17.4 % | 5.61 | 10.4 % |
| `..vxc_deriv1` (**B5**) | 7.53 | 8.9 % | **1.53** | 7.75 | 11.4 % | **1.55** | 2.9 % |
| `..jk_ip1` | 3.92 | 4.6 % | 4.03 | 3.79 | 5.6 % | 3.76 | 7.0 % |
| `solve_mo1` | 55.82 | 66.2 % | 55.96 | 39.65 | 58.2 % | 39.28 | 72.8 % |
| `..nr_rks_fxc_mo` (B4d) | 5.46 | 6.5 % | 5.47 | 5.42 | 8.0 % | 5.45 | 10.1 % |
| `..get_j`, n_dm ≤ 126 (B4a) | 28.99 | 34.4 % | 28.99 | 12.67 | 18.6 % | 12.66 | 23.5 % |
| `..get_k`, n_dm ≤ 126 (**B4b**) | 20.20 | 24.0 % | 20.43 | 20.50 | 30.1 % | 20.12 | 37.3 % |

84.26 → 53.93 s is **1.56x** on the whole Hessian.  The two single-module
columns say where it comes from: `fastxchess` on its own is **1.19x**
(70.87 s) and the other eleven together are **1.24x** (68.12 s, all of it
`fastj`'s multi-density Coulomb build), and since they act on disjoint parts
of the run the two compose.  The XC Hessian's two terms go from 15.39 s to
2.23 s (**6.9x**), which turns 18.2 % of the wall clock into 4.1 %; the
Hessians agree to 2e-11 on the trace.  What is left is the same thing the
113-atom profile says louder: **`get_k` and `get_j` are 61 % of it**, and
`get_k` has had nothing done to it.

`fastj`'s multi-density Coulomb build is **2.29x** here (28.99 → 12.67 s at
`n_dm` = 126), which is worth noting against `docs/ROUND3_B4A.md`: that round
measured 1.19–1.23x on the whole J build at `n_dm` = 8, 16 and 32 on a
fixed density, and the Hessian's CPHF runs it at `n_dm` = 126.

## 3. Taxol, 113 atoms, 1123 AOs

| stage | stock | share | + Electrolites + fastxchess | share | |
|---|---|---|---|---|---|
| **TOTAL** | **1519.0 s** | 100 % | **1313.8 s** | 100 % | **1.16x** |
| `partial_hess_elec` | 96.33 | 6.3 % | 44.78 | 3.4 % | 2.15x |
| `..ejk_ip2` (B3) | 39.13 | 2.6 % | 39.12 | 3.0 % | 1.00x |
| `..exc_deriv2` | 55.90 | 3.7 % | 4.41 | 0.3 % | 12.7x |
| `....vxc_diag` | 1.24 | 0.1 % | 1.29 | 0.1 % | 0.96x |
| `....vxc_deriv2` (**B5**) | 54.09 | 3.6 % | **2.55** | 0.2 % | **21.2x** |
| `make_h1` | 96.46 | 6.3 % | 53.77 | 4.1 % | 1.79x |
| `..vxc_deriv1` (**B5**) | 62.39 | 4.1 % | **19.92** | 1.5 % | **3.13x** |
| `..jk_ip1` | 33.47 | 2.2 % | 33.23 | 2.5 % | 1.01x |
| `solve_mo1` | 1325.81 | 87.3 % | 1214.88 | 92.5 % | 1.09x |
| `..nr_rks_fxc_mo` (B4d) | 89.63 | 5.9 % | 89.29 | 6.8 % | 1.00x |
| `..get_j`, n_dm ≤ 339 (B4a) | 434.11 | 28.6 % | 321.58 | 24.5 % | 1.35x |
| `..get_k`, n_dm ≤ 339 (**B4b**) | 718.75 | 47.3 % | 720.73 | **54.9 %** | 1.00x |

The two Hessians agree to 2e-11 on the trace and the Frobenius norm
(1.108497169052e+02 against 1.108497169054e+02), which is the same
`atomicAdd`-and-CPHF floor the unit tests see.

Two rows to read together.  `fastxchess` takes the XC Hessian's two terms from
116.5 s to 22.5 s (**5.18x**), which is 7.7 % of the wall clock turned into
1.7 %.  And `get_k` does not move, because `fastk` declines `n_dm != 1` — so
after this round **the exchange build inside CPHF is 55 % of a 113-atom
Hessian**, more than everything else here put together.

## 4. What the two sizes say about the catalogue's ranking

The catalogue ranks the Hessian entries by their share of a **42-atom**
Hessian.  Between 42 and 113 atoms the shares move a long way, and in opposite
directions:

| entry | share at 42 atoms | at 113 atoms | catalogue's number (1.7.0, 42 atoms) |
|---|---|---|---|
| **B4b** K in CPHF | 24.0 % | **47.3 %** | 23.7 % |
| **B4a** J in CPHF | 34.4 % | 28.6 % | 31.3 % |
| **B5** the XC Hessian | 18.2 % | 7.7 % | 14.8 % |
| **B3** `ejk_ip2` | 8.8 % | 2.6 % | 8.0 % |
| **B4d** `nr_rks_fxc_mo` | 6.5 % | 5.9 % | (inferred ~11 %) |

Three things follow.

1. **The catalogue's shares reproduce on 1.8.1 at 42 atoms**, within a couple
   of points, everywhere except B6 (gone) and B4d (over-inferred).  Its §2.3
   table is a good 1.8.1 table too, which is not obvious given §1 above.
2. **B4b is the Hessian's largest entry at any size, and it grows.**  At 113
   atoms the exchange build inside CPHF is nearly half the wall clock.  The
   catalogue ranks it 8th, behind B5 (7th), on the strength of a 42-atom
   profile; on a 113-atom one the order is the other way round by a factor of
   six.  This repository's `fastk` declines `n_dm != 1`
   (`fastk.py:175`), so all of it is untouched.
3. **B5 and B3 shrink with size**, because the CPHF cost grows as
   `n_dm x nao^2`-ish while a grid term grows as `natm x ngrids`.  A Hessian
   entry's share is not a molecule-independent number, and neither is the
   ranking built from it — catalogue §3.4's size-independence rule applies to
   the *targets*, not only to the speedups.

And what the round leaves, which is the same conclusion twice over:

| entry | 42 atoms, after | 113 atoms, after |
|---|---|---|
| **B4b** K in CPHF | **37.3 %** | **54.9 %** |
| **B4a** J in CPHF (`fastj`, 2.29x / 1.35x) | 23.5 % | 24.5 % |
| **B3** `ejk_ip2` | 13.7 % | 3.0 % |
| **B4d** `nr_rks_fxc_mo` | 10.1 % | 6.8 % |
| **B5** the XC Hessian (this round) | 4.1 % | 1.7 % |
| `jk_ip1` | 7.0 % | 2.5 % |

Every remaining Hessian target is now in the CPHF solve, and one of them is
half of it.

