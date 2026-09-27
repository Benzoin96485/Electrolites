# The meta-GGA XC derivatives of a Hessian: B5's screen, a fifth row, and a kernel the GGAs needed too

`docs/DF_WORK_LIST.md` **W28**, found by W23.  `fastxchess` (catalogue B5)
replaced `hessian/rks.py:_get_vxc_deriv1` and `_get_vxc_deriv2` for GGAs and
declined everything else, so a meta-GGA Hessian ran GPU4PySCF's own code for
both.  On the 42-atom DF-TPSS/def2-SVP Hessian W23 repaired, those two rows
(`_get_vxc_deriv1`, and `_get_exc_deriv2` which is `_get_vxc_deriv2` plus
`_get_vxc_diag`) were 73 % of the run.  The entry asked whether B5's two
mechanisms carry over, and said it was falsified if the time is in the
per-block `eval_xc_eff` instead.  It is not, they do, and both spin cases are
covered; `electrolites/fastxchess.py` is the module, extended.

---

## 0. Summary

* **The falsification does not fire.**  Upstream's meta-GGA branches staged
  on penicillin V (§1): the atom loop is **92 %** of `_get_vxc_deriv1` and
  **92 %** of `_get_vxc_deriv2`; `eval_xc_eff` at `deriv=2` is **1.2 %** and
  **1.3 %**.  A meta-GGA's third derivative is dearer than a GGA's, but it is
  called once per block and the atom loop runs 42 times per block.
* **Every tau term folds into a contraction B5 already had** (§2), so the
  meta-GGA path is the GGA one with a fifth density component rather than a
  second loop: `d rho / d R`'s fifth row comes from the same second-derivative
  gather; in `_get_vxc_deriv1` upstream's `for j in range(1, 4)` block is a
  matrix and its own transpose added together — the same matrix twice — so it
  is one GEMM of inner dimension `4 ngrids` shared with the GGA term; in
  `_get_vxc_deriv2` the nine per-atom tau contractions are a fifth row of the
  atom-independent `W[x]`.
* **One thing the profile found that is not about meta-GGAs** (§3): with the
  GEMMs at 16 TF/s, forming their `aow`/`mow` operands with a cuTENSOR
  contraction (`'ypg,ayg->apg'`, a batch index cuTENSOR does not treat as
  one) wrote them at **139 GB/s** and cost as much as the GEMMs.  A
  register-tiled kernel writes them at **766 GB/s**.  It is 1.4x–2.0x on
  `_get_vxc_deriv1` from 21 atoms up, for meta-GGAs *and GGAs* — B5's own B3LYP number moves
  7.67 → 0.78 s on penicillin V (1.50 s before) and 19.8 → 9.75 s on taxol.
* **The two functions, meta-GGA, against GPU4PySCF** (§4): `_get_vxc_deriv1`
  **10.2x–10.8x** from 22 to 42 atoms, 8.7x at 113, 5.8x in def2-TZVPP,
  10.0x–13.5x on open shells (Ni, Pt, Cu complexes, a radical cation);
  `_get_vxc_deriv2` **9.1x–27x** from 21 atoms up.  At 113 atoms
  `_get_vxc_deriv1` is now its two GEMMs at **17.6 TF/s** (73 % of the
  call).  Five functionals (TPSS, r2SCAN, M06-L, M06-2X, and wB97M-V in the
  tests) move identically, which is what a change
  that never touches the functional should do.  The success criterion — the
  two rows move by what B5 was worth on GGAs (5.07x and 12.94x at 42 atoms) —
  is met and passed on every system from 22 atoms up; at 5–10 atoms it is
  3.7x–5.2x, where GGAs got about the same.
* **Whole DF Hessians** (§5): penicillin V DF-TPSS **41.5 s → 13.4 s
  (3.10x)**, def2-TZVPP 67.4 → 41.7 s (1.62x), alanine dipeptide DF-r2SCAN
  10.9 → 3.39 s (3.21x), and the Ni(II) dithiolene and Pt(II) triplet UKS
  DF-TPSS **3.22x** and **3.24x**.  Traces agree to 1e-9 (the ip2 kernel's
  atomics).
* **Correct against PySCF on the CPU, not only against GPU4PySCF** (§6):
  1e-12 (`deriv1`) and 1e-14 (`deriv2`) relative on eight restricted and
  unrestricted cases, and the finite-difference Hessian test now has
  meta-GGA rows.  One of them, NH₃/r2SCAN, is 1.1e-2 off the finite
  difference — by the same amount with GPU4PySCF's branches (§6.1).
* **What is left** (§7) is not in these two functions any more: in
  def2-TZVPP the CPHF's `nr_rks_fxc_mo` is **68 %** of the Hessian, and its
  meta-GGA call is 3.9x the B3LYP one.

Everything is one A100-SXM4-80GB of a whole node (`--qos=debug`, four chains
on four GPUs), GPU4PySCF 1.8.1, def2-SVP unless a row says otherwise,
level-3 grids, `conv_tol_cpscf = 1e-6`.  Function tables are
`benchmarks/xchess_bench.py`, both implementations in one process on one
converged SCF.  The "before" of a whole Hessian is `patch_all()` with
`FASTXCHESS_MGGA=0`, which hands meta-GGAs back to GPU4PySCF (with W23's
repair, which stays on).

---

## 1. Where the time was

`benchmarks/mgga_xchess_probe.py` replays both of upstream's MGGA branches
with a synchronisation after every stage.  Penicillin V, TPSS, 127 blocks;
upstream's own functions take 15.54 s and 14.22 s in the same process
(`_get_vxc_diag` 0.60 s):

| stage | `_get_vxc_deriv1` | | `_get_vxc_deriv2` | |
|---|---|---|---|---|
| expand the masked AOs to `nao` | 0.56 s | 3.1 % | 0.34 s | 2.0 % |
| `eval_rho2` | 0.12 | 0.6 % | 0.12 | 0.7 % |
| **`eval_xc_eff`, `deriv=2`** | **0.22** | **1.2 %** | **0.22** | **1.3 %** |
| `v_ip` / `ipip` (GGA + tau) | 0.27 | 1.5 % | 0.54 | 3.1 % |
| atom loop: `_make_dR_rho1` + `fxc` | 7.63 | 42.1 % | 7.65 | 44.0 % |
| atom loop: the GGA terms | 2.70 | 14.9 % | 3.31 | 19.0 % |
| atom loop: **the tau terms** | 6.43 | **35.5 %** | 4.47 | **25.7 %** |
| atom loop: scatter | — | | 0.56 | 3.2 % |
| total (the replay) | 18.12 s | | 17.40 s | |

`eval_xc_eff` on the whole grid in one call is 10.6 ms against 240 ms in 127
calls, so catalogue G1's grouping would be worth 0.23 s of 30 s.  The atom
loop is 92–95 %, and the tau terms alone are a quarter to a third: upstream
spends as much on them as on everything else the atom loop does.  B5 applies.

---

## 2. What changed

`electrolites/fastxchess.py`, with `nc` (4 or 5 density components) through
every helper; `_setup` takes `xctype == 'MGGA'` unless `FASTXCHESS_MGGA=0`.
The factors are upstream's, and they differ between the two functions: the
tau row of the perturbed potential is scaled by 1/4 in `_get_vxc_deriv1`,
which adds the term and its transpose, and by 1/2 in `_get_vxc_deriv2`, which
adds it once.

1. **`tau` and `d tau / d R`** (`_rho`, `_dR_rho1`).  `tau = 1/2 sum_j
   (d_j ao) . (D d_j ao)` from the `c = D . ao[:4]` the module already forms;
   `_make_dR_rho1`'s fifth row, `1/2 sum_j (d_s d_j ao)(D d_j ao)` on atom
   A's AOs, from the same second-derivative gather the gradient rows use,
   against `c[1:4]` instead of `c[0]`, and the atom dimension through the same
   indicator GEMM.  Upstream: nine `_contract_rho` launches per atom and
   block for this row.
2. **`_get_vxc_deriv1`'s tau term is one slab of an existing GEMM**
   (`_deriv1_gemms`).  Upstream adds, per atom, direction and `j = 1..3`,
   `(ao_j w_4) . mo_j^T` and `((mo_j w_4) . ao_j^T)^T` — which are the same
   matrix.  So the term is `2 sum_j (mo_j w_4) . ao_j^T`, and it has the
   shape of the GGA half's `mow . ao_0^T`: stacked as
   `[mow | 2 mo_j w_4] . [ao_0 | ao_1 | ao_2 | ao_3]^T` it is one GEMM of
   inner dimension `4 ngrids` per atom block, where upstream issues eight
   GEMMs and sixteen scalings per atom and direction.
3. **`_get_vxc_deriv2`'s tau term is a fifth row of `W[x]`** (`_deriv2_t`).
   Upstream's nine `contract('jg,xjg->xj', ao_mask[d_k d_x], ...)` per atom
   are `sum_k (d_k d_x ao)(D d_k ao) . wv_4`, and `sum_k (d_k d_x ao)(D d_k
   ao)` does not depend on the atom, so it is the fifth component of the
   `W[x]` that B5 already builds once per block, and the whole term is one
   more row of the same contraction.
4. **`ipip`'s tau part is one contraction per block** (`_ipip_block`):
   upstream's three `_d1d2_dot_` calls on the second derivatives (nine GEMMs
   of inner dimension `ngrids`) as `'kypg,kxqg->xypq'` over a gather of
   `ao[_H]`.
5. **`v_ip`'s tau part** is `grad/rks.py:_tau_grad_dot_` in the masked AO
   space, added to the GGA term before the scatter.

The unrestricted pair shares everything with a spin index, as B5's did:
the tau rows of both spins' `d rho / d R` go through `_wv2_u`'s spin-coupled
`fxc` together.

---

## 3. The kernel, which the GGAs needed too

After items 1–5, `_get_vxc_deriv1` on penicillin V/TPSS was 2.19 s, and the
stages (`mgga_xchess_probe.py --fast-stages`) put 1.61 s of it in
`_deriv1_gemms`.  Replayed step by step (`benchmarks/xchess_gemm_probe.py`):

| step of `_deriv1_gemms` | cuTENSOR `aow`/`mow` | kernel, one row per thread | kernel, register-tiled |
|---|---|---|---|
| `aow = sum_y ao_y w_y` | 0.636 s | 0.208 s | **0.115 s** |
| `mow` and the tau slabs | 0.416 s | 0.120 s | 0.132 s |
| GEMM 1, `aow . mo_0^T` | 0.137 s | 0.136 s | 0.135 s |
| GEMM 2, inner dimension `4 ngrids` | 0.498 s | 0.498 s | 0.496 s |
| `_get_vxc_deriv1` | 2.48 s | 1.77 s | **1.70 s** |

The two GEMMs are 10.15 Tflop in 0.63 s, **16.0 TF/s** of the card's 19.5 —
there is nothing left in them.  The operands were the problem: `aow` is
88.3 GB of writes over the call, and cuTENSOR's `'ypg,ayg->apg'` wrote it at
139 GB/s, because `g` is a mode of all three operands and the library does
not treat it as a batch.  `_scale_sum` is a small CUDA kernel: one thread per
grid point holds that point's weights for four rows of `a` in registers and
walks eight AOs, so every value of `ao` is read once per four outputs — 766
GB/s.  With the tau slabs it writes `mow`'s four slabs in one pass.
`FASTXCHESS_SCALESUM=0` puts the contraction back; the bench's "cuTENSOR
aow/mow" column is that ablation.

It is not a meta-GGA change — B5's GGA path forms the same two operands the
same way, and B5's write-up had found the allocation around them (§3b there)
but not their rate.  `_get_vxc_deriv1`, the kernel against the contraction,
same process:

| | B3LYP | TPSS |
|---|---|---|
| glycine, 10 atoms | — | 0.96x |
| alanine dipeptide, 22 | — | 1.44x |
| penicillin V, 42 | **1.92x** (1.496 → 0.779 s) | 1.51x |
| penicillin V, def2-TZVPP | — | 1.55x |
| taxol, 113 | **2.03x** (19.81 → 9.75 s) | 1.51x |
| Ni(II) dithiolene⁻ / Pt(II) triplet / Cu(II) macrocycle, UKS | — | 1.37x / 1.46x / 1.62x |

GGAs gain more because their GEMM 2 is a quarter of the meta-GGA one, so the
operands were a larger share.  On a whole 42-atom DF-B3LYP Hessian it is
15.27 → 14.53 s (two pairs, 1.05x), `_get_vxc_deriv1` 1.50 → 0.78 s.
`_get_vxc_deriv2` does not form these operands and is unchanged.

---

## 4. The functions

`benchmarks/xchess_bench.py`, final code, min of `--reps`; taxol's GPU4PySCF
column is from an earlier job of the same session (`--skip-orig` in the
final one: its upstream `_get_vxc_deriv1` alone is three minutes):

| system | 2S | xc | `_get_vxc_deriv1` before → after | | `_get_vxc_deriv2` before → after | |
|---|---|---|---|---|---|---|
| glycine, 10 atoms | 0 | TPSS | 0.971 → 0.188 s | 5.17x | 0.957 → 0.199 s | 4.81x |
| FeCl₄⁻ sextet, 5 | 5 | r2SCAN | 1.008 → 0.239 | 4.21x | 1.019 → 0.275 | 3.71x |
| alanine dipeptide, 22 | 0 | TPSS | 4.216 → 0.415 | 10.16x | 3.941 → 0.417 | 9.46x |
| Pt(II) amine, 23, ECP | 2 | TPSS | 12.17 → 0.984 | 12.38x | 11.48 → 0.869 | 13.22x |
| Ni(II) dithiolene⁻, 21 | 1 | TPSS | 12.53 → 1.211 | 10.34x | 11.91 → 0.941 | 12.66x |
| penicillin V, 42 | 0 | TPSS | 15.65 → 1.465 | **10.68x** | 14.06 → 0.863 | **16.30x** |
| penicillin V | 0 | r2SCAN | 15.89 → 1.465 | 10.84x | 14.73 → 0.858 | 17.16x |
| penicillin V | 0 | M06-L | 15.50 → 1.465 | 10.58x | 14.80 → 0.930 | 15.91x |
| penicillin V | 0 | M06-2X | 15.84 → 1.465 | 10.81x | 14.43 → 0.886 | 16.28x |
| penicillin V⁺ | 1 | TPSS | 38.38 → 2.848 | **13.48x** | 36.88 → 1.693 | **21.78x** |
| Cu(II) macrocycle²⁺ quartet, 60 | 3 | TPSS | 75.16 → 7.494 | 10.03x | 67.58 → 2.812 | **24.03x** |
| penicillin V, def2-TZVPP | 0 | TPSS | 15.69 → 2.717 | 5.77x | 14.47 → 1.592 | 9.08x |
| taxol, 113 | 0 | TPSS | 183.6 → 21.04 | 8.73x | 103.7 → 3.913 | **26.5x** |

For comparison, B5 on B3LYP was 5.07x and 12.94x at 42 atoms and 3.16x and
22.78x at 113.  The meta-GGA margin is larger because upstream's meta-GGA
atom loop is twice its GGA one (15.65 s against 7.67 s on penicillin V) and
this module's meta-GGA call is only 1.9x its GGA one.  The ablation columns
of the same runs: the atom screen is worth 1.02x–1.18x on `_get_vxc_deriv1`
up to 60 atoms and 1.40x at 113, as for GGAs (it grows with size), and
nothing on `_get_vxc_deriv2`, whose atom dimension is already a contraction.

Where `_get_vxc_deriv1` still goes, penicillin V/TPSS
(`mgga_xchess_probe.py --fast-stages` and `xchess_gemm_probe.py`): the two
GEMMs 0.63 s at 16 TF/s, `aow`/`mow` 0.25 s, `eval_xc_eff` 0.23 s (now 14 %
where it was 1.2 %), `_dR_rho1` 0.13 s, `_wv2` 0.06 s, `_rho` 0.04 s.  On
taxol the GEMMs are **15.4 s of 21.1 s at 17.6 TF/s** (271.6 Tflop), `aow`
and `mow` 3.1 s, so the call is bound by arithmetic it cannot avoid (B5's
§3: the output has a free AO and a free occupied index); it is 21 s where the
GGA one is 9.75 s because it has five GEMM slabs to the GGA's two.  In
def2-TZVPP the margin is smaller because upstream's time does not depend on
the basis (15.65 s at `nao` 430, 15.69 s at 1007 — its per-atom loop is
launch-bound) and this module's does (1.47 → 2.72 s).

---

## 5. Whole Hessians

`benchmarks/df_hess_anatomy.py --patch all --rounds 2`, before
(`FASTXCHESS_MGGA=0`) and after in alternating processes on one GPU, the pair
run twice; the second round of each:

| system | xc | before (pair 1, 2) | after (pair 1, 2) | | traces before / after |
|---|---|---|---|---|---|
| alanine dipeptide, 22 atoms | r2SCAN | 10.71, 11.00 s | **3.39, 3.38 s** | **3.21x** | 20.6898811732 / …732 |
| Ni(II) dithiolene⁻, 21, UKS | TPSS | 32.11, 31.75 | **9.90, 9.91** | **3.22x** | 23.08798041 / …040 |
| Pt(II) amine triplet, 23, UKS | TPSS | 32.15, 32.70 | **10.02, 10.01** | **3.24x** | 19.9589432167 / …155 |
| penicillin V, 42 | TPSS | 41.34, 41.73 | **13.39, 13.37** | **3.10x** | 40.9142598198 / …207 |
| penicillin V, 42, def2-TZVPP | TPSS | 67.71, 67.07 | **41.84, 41.63** | 1.62x | 38.8754196472 / …467 |

The traces agree to the run-to-run spread of the `ip2` kernel's atomics
(4e-9 on the Ni complex before *and* after).  Where the time is now,
penicillin V (after, pair 2):

| stage | before | after |
|---|---|---|
| `_get_vxc_deriv1` | 15.78 s, 37.8 % | 1.46 s, 10.9 % |
| `_get_exc_deriv2` | 15.57 s, 37.3 % | 1.57 s, 11.8 % |
| ↳ of which not `_get_vxc_deriv2`: `_get_vxc_diag` (0.6 s in §1) and the per-atom-pair sums, upstream's | ~1.5 s | ~0.7 s |
| CPHF (`solve_mo1`) | 7.61 s, 18.2 % | 7.61 s, **56.9 %** |
| ↳ `nr_rks_fxc_mo` (W16), 6 calls | 6.61 s | 6.60 s, **49.4 %** |

and in def2-TZVPP the CPHF's `nr_rks_fxc_mo` is 28.5 s, **68 %** of the
Hessian.

---

## 6. Correctness

`tests/test_xchess_mgga.py` (new), relative to the largest element, on one
converged SCF per case:

| case | against GPU4PySCF (W23-repaired) | against PySCF on the CPU |
|---|---|---|
| ethanol TPSS / r2SCAN / M06-L | 2e-14–3e-14 / 1e-15 | 6.6e-13–1.5e-12 / 3e-15 |
| CH₂O def2-TZVPP M06-2X (hybrid, f functions) | 5.7e-15 / 1.5e-15 | 2.4e-12 / 9.0e-15 |
| CH₂O wB97M-V (range separated, VV10) | 3.2e-15 / 8.2e-16 | 3.9e-15 / 9.9e-16 |
| ethanol⁺ TPSS, UKS | 3.7e-15 / 2.6e-16 | 1.7e-12 / 9.7e-16 |
| CH₂O⁺ def2-TZVPP r2SCAN, UKS | 1.0e-14 / 1.3e-15 | 6.3e-13 / 2.9e-15 |
| FeCl₄⁻ sextet TPSS, UKS | 1.2e-13 / 1.0e-15 | 2.4e-12 / 1.4e-15 |

(`deriv1` / `deriv2` in each cell.)  The CPU column is the one that matters:
before W23 the GPU4PySCF reference was itself wrong for meta-GGAs, and
PySCF's is an independent implementation.  A whole DF-TPSS Hessian of
ethanol, patched three times, spreads by 9e-12 Eh/bohr² and is 3.5e-12 from
the unpatched one.  `tests/test_xchess.py` and `tests/test_uxchess.py` now
carry meta-GGA rows (TPSS, r2SCAN, SCAN, M06-L, M06-2X, wB97M-V; restricted
to 22 atoms, unrestricted on CH₃, NO, FeCl₄⁻ and the Ni complex): 18/18 and
15/15, `deriv1` ≤ 1e-13 and `deriv2` ≤ 3.4e-12 absolute.  M06-2X, which those
tests listed as declined, is now covered; only LDA is.
`tests/test_mgga_inplace.py`'s bit-for-bit check now names upstream's
function explicitly (W23's subject), since the patched one uses atomics.

### 6.1 The finite-difference rows

`tests/test_xchess_fd.py` (eps 1e-3 bohr, level-6 grids) has three meta-GGA
rows and an `upstream` column that repeats each with GPU4PySCF's branches:

| row | patched | upstream |
|---|---|---|
| H₂O TPSS | 1.16e-4 | 1.16e-4 |
| NH₃ r2SCAN | **1.14e-2** | **1.14e-2** |
| CH₃ TPSS, UKS | 1.77e-4 | 1.77e-4 |

The r2SCAN row is 1.1e-2 off the finite difference with either
implementation, and 6.6e-3 at level-8 grids: it moves with the grid, so it is
r2SCAN's known grid sensitivity (GPU4PySCF warns "MGGA Hessian is sensitive
to dft grids" below level 5) and not this module — which agrees with
GPU4PySCF and PySCF to 1e-12 on the same function.  The GGA and LDA rows are
unchanged (≤ 8e-5).

---

## 7. What is left

* **The CPHF's meta-GGA response kernel.**  After this round a DF-TPSS
  Hessian is 57 % CPHF at 42 atoms and 72 % in def2-TZVPP, and almost all of
  that is `nr_rks_fxc_mo`: 28.5 s for six calls in def2-TZVPP against 7.2 s
  for B3LYP on the same molecule and basis (3.9x), and 6.6 s against 2.0 s
  in def2-SVP (3.2x).  It already runs `fastxcfxc`'s own blocked code
  (W16).  A meta-GGA's response needs four density products per block where a
  GGA needs one, so some of the 3.9x is arithmetic; whether the rest is the
  same operand-formation cost §3 found here has not been measured.
* **`_get_vxc_diag`** is upstream's for every functional, and for a
  meta-GGA it is 0.6 s (§1) of the 1.57 s `_get_exc_deriv2` row: no atom
  loop, and its tau part is eighteen GEMMs per block.  At 5 % of a 42-atom
  Hessian it is not where the next hour goes.

---

## 8. Tools

```bash
python tests/test_xchess_mgga.py [--no-hessian] [--no-cpu]
python tests/test_xchess.py; python tests/test_uxchess.py
python tests/test_xchess_fd.py [--only tpss] [--level 8]
python benchmarks/mgga_xchess_probe.py --xyz benchmarks/molecules/penicillinV.xyz \
    --basis def2-svp --xc tpss [--fast-stages]            # §1 and §4's stage split
python benchmarks/xchess_gemm_probe.py --xyz ... --xc tpss  # §3
python benchmarks/xchess_bench.py --xyz ... --xc tpss [--spin 1 --charge 1] [--skip-orig]
FASTXCHESS_MGGA=0 python benchmarks/df_hess_anatomy.py --patch all --xc tpss ...  # the baseline
```

Switches: `FASTXCHESS_MGGA=0` (meta-GGAs to GPU4PySCF),
`FASTXCHESS_SCALESUM=0` (the cuTENSOR `aow`/`mow`), and B5's
`FASTXCHESS_OFF`, `FASTXCHESS_ATOMSCREEN`, `FASTXCHESS_MEM`,
`FASTXCHESS_INPLACE`.  Job scripts: `~/vibeqc/_el_jobs/w28_{a,b,c}.sh`.
