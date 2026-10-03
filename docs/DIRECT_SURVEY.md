# The direct SCF path: is there DF-style headroom left?

A survey, asked for after W42–W44 of `docs/DF_WORK_LIST.md`: the density-
fitting rounds kept finding that the wall clock was not the arithmetic but an
allocator, a launch shape, host overhead or a solver nobody had timed.  This
asks the same question of the direct path (`fastk`, `fastj`, `fastjk`,
`fastxc`, and the direct Hessian).  One item from it is implemented (§3);
the rest are leads with bounds.

The survey was first taken on the login node's shared A100-PCIE; §1 is the
compute-node rerun (jobs `el_dscf1`, `el_dscf2`, one A100-SXM4-80GB of a whole
node), which is the one to quote.  The login-node numbers overstated the
host-bound stretches by about 2x (a shared card, a slower host) and are kept
only where the compute node did not measure the item (§2.4).  Scripts are in
`$SCRATCH/w44/dscf/` (`dscf_nvtx.py`, `hess_nvtx.py`: NVTX ranges around
`get_veff`, `get_j`, `get_k`, `nr_rks`, `eig`, `get_fock`, DIIS,
`_VHFOpt.build`, `fastk._prep` and the Hessian's terms; `ana.py` reads the
`nsys` sqlite).  The profiles include §3's change.

## 1. Where the device is idle

`patch_all()`, B3LYP, level-3 grids, `direct_scf_tol` 1e-13, one warm SCF
under `nsys profile -t cuda,nvtx,osrt`:

| SCF | wall | GPU busy | idle | in `nr_rks` | `get_j` | `get_k` | `fastk._prep` |
|---|---|---|---|---|---|---|---|
| taxol def2-SVP, 15 cycles | 16.11 s | 89.9 % | 1.63 s | **1.16 s** (38 % of `nr_rks`) | 0.10 s | 0.09 s | 0.02 s |
| PfPMT 6-31G\*, 17 cycles | 54.32 s | 93.3 % | 3.62 s | **2.74 s** (38 %) | 0.20 s | 0.21 s | 0.03 s |
| PfPMT def2-SVP, 18 cycles | 66.99 s | 94.4 % | 3.77 s | **2.99 s** (34 %) | 0.24 s | 0.13 s | 0.02 s |

and a warm direct Hessian, penicillin V/def2-SVP, 48.2 s: the CPHF
(`solve_mo1`, 32.0 s) is 97 % busy -- its J/K 29.3 s at 99 % -- and the idle
is `jk_ip1` 1.40 s, `_get_exc_deriv2` 0.81 s (61 % of that call), `ejk_ip2`
0.31 s, `_get_vxc_deriv1` 0.27 s.

* **J and K are at the machine**: `get_k` is 52–59 % of the SCF and 99 %
  busy; no GPU4PySCF two-electron kernel ran (every class is this
  repository's).  Nothing inert.
* **What idles the device is the XC build**: on PfPMT `nr_rks` makes 197k
  `cudaLaunchKernel` and 46k `cuLaunchKernel` calls in the SCF (1.6 s of
  launch API), 864 `cudaFree` (0.94 s) and 108 `cudaGetDeviceProperties`
  (0.36 s); 245k GPU-idle gaps, 1.6 s of them between 10 µs and 1 ms.

## 2. Leads, bounded by the measured idle time

1. **`nr_rks` is launch-bound** (`fastxc._nr_rks_grouped` over upstream's
   `_grouped_block_loop`): per call on PfPMT 4,264 `_cart_kernel_deriv1`,
   3,320 `GDFTcontract_rho_kernel`, 830 each of `take_last2d`, `dscale_ao`,
   `add_sparse`.  Bigger grid blocks are not the fix (measured: 4096 →
   32768 points makes the call 4x slower, sparsity is lost -- login node).
   Batching the per-block kernels of a group into one launch is.
2. **`cudaFree` and `cudaGetDeviceProperties` inside `nr_rks`**: 80 frees a
   call (1.1 s over PfPMT's SCF) around libxc's evaluation, and ten 3.3 ms
   property queries a call in upstream's `grouped_gemm` wrapper (0.6 s, in
   device-idle gaps).  The work list's §5.8 in a new place.  Compute node:
   0.94 s of frees and 0.36 s of property queries inside PfPMT's `nr_rks`.
   Items 1 and 2 together are bounded by `nr_rks`'s idle: **1.2 s of taxol's
   16.1 s SCF (7 %), 2.7–3.0 s of PfPMT's (4.5–5 %).**
3. **`fastk._prep` rebuilt its pair lists every call** -- implemented, §3.
4. **K classes with fewer bra pairs than SMs** (`n_blocks = min(432,
   npairs_ij)`, 2–29 blocks on PfPMT's small groups): splitting their ket
   list across streams measured **1.056x** on PfPMT 6-31G\* `get_k`
   (converged density, ΔK 2.6e-15), 1.02x in def2-SVP, nothing on taxol,
   where no class is that small.  It depends on the chemistry (singleton
   shell groups), not the size, so it should be larger on transition-metal
   complexes; not implemented.  **~1.5 s of 60 s.**
5. **The direct Hessian** (penicillin V/def2-SVP, 48.2 s warm): the CPHF's
   J/K is 61 % and 99 % busy; `jk_ip1` (1.40 s idle, 188 gaps over 1 ms),
   `_get_exc_deriv2` (`fastxchess`, 0.81 s of 1.34 s idle, 42k gaps) and
   `_get_vxc_deriv1` -- the launch-bound pattern of `nr_rks` again.
   **~2.8 s (6 %).**  The CPHF solve itself is W44's
   `fastcphf`, which the direct Hessian reaches: at 113 atoms GPU4PySCF's
   Krylov solver was 12.5 % of a DF CPHF; its share in a direct CPHF was
   not measured.

So the direct path has headroom of the DF path's *kind* -- host overhead,
allocator synchronisation, launch shapes -- but not of its size: the device
is 90–94 % busy through a whole SCF, the two-electron builds are at the
machine, and the leads add up to 5–10 % of an SCF and ~6 % of a Hessian.

## 3. Implemented: the pair lists' geometry, once

`electrolites/compat.py:tril_pair_mappings` (reached by `fastk`, `fastjk`
and `fastejk`) built each `(i, j)` shell group's candidate pair list with
boolean-mask indexing -- a device-host synchronisation each -- on every call,
though only the final `q_cond > cutoff` screen depends on the density.  The
unscreened lists are now cached per shell layout (`_tril_geometry`) and
screened in one pass (one `nonzero`, one `bincount`, two synchronisations
per call), producing identical lists: **20 ms → 0.82 ms a call** on
penicillin V/6-31G\* (login node, ten calls round robin), every list equal
to the vendored loop's at three cutoffs; `tests/test_jk.py` passes (K
1.0e-14, fused J 2.1e-14 on a compute node).  On the compute node
`fastk._prep` is now **0.032 s over PfPMT's 18 calls** (the login-node
profile before the change: 0.60 s), about 1 % of that SCF.  `ELECTROLITES_TRIL=pergroup` is the vendored loop.

## 4. Owed

Item 4 (the K split) was measured on the login node only.
