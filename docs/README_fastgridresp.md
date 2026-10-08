# fastgridresp — the grid-response Hessian without storing the weights' second derivative (B9)

Every use of `hessian/rks.py:get_d2weight_dAdB` in GPU4PySCF 1.8.1 — three in
each of the RKS and UKS `_get_exc_deriv2_grid_response`, one in the VV10
`_get_enlc_deriv2` — is

```python
d2w = get_d2weight_dAdB(mol, grids, (g0, g1))   # (natm, natm, 3, 3, ngrids)
d2e += contract('ABdDg,g->ABdD', d2w, eps)
```

`get_d2weight_dAdB` stores `natm² · 9` doubles per grid point (520 MB per
4096 points at 42 atoms), then fills the blocks of each point's own atom by
translation invariance with `18 · natm` small CuPy launches, and asserts the
array is below 2³¹ elements — false from 241 atoms on
(`docs/KERNEL_TARGETS_1.8.1.md` §11.5).

## What changes

`codegen/gen_d2w_fused.py` lifts GPU4PySCF's two second-derivative kernels
(`lib/gdft/gen_grids.cu`) unchanged into `__device__` functions that return
a point's 3×3 block instead of storing it
(`electrolites/kernels/cuda/mygrid/d2w_fused.cu`).  Contracted with `eps`,
the translation-invariance fill becomes two more accumulators:

    out[G,H] = sum_g eps K[G,H]  -  U[G,A]  -  V[A,H]  +  diag(sum_G U[G,A])

with `K` the kernels' blocks (zero where `G` or `H` is the point's atom
`A(g)`), `U[G,A] = sum_{g in A} eps sum_H K[G,H]`, `V[A,H] = sum_{g in A}
eps sum_G K[G,H]`.  Each point's block, times `eps`, goes to all three; the
lanes of a warp that share `(G, H)` sum theirs with shuffles first and one
lane writes, all three at once when the lanes' points share their atom
(`d2w_fused_wrappers.cu`).  Memory is `3 · natm² · 9` doubles whatever the
block, so the int32 wall is gone.

`fastgridresp.d2w_contract` sets up the same inputs as `get_d2weight_dAdB`
(`GDFTbecke_eval_PB`, the distances, `1/ΣP`) and launches the two fused
kernels.  The four callers are GPU4PySCF's source with the two lines
replaced, bound to their modules.  `FASTGRIDRESP_OFF=1` keeps GPU4PySCF's.

## Result

B3LYP/def2-SVP, `grid_response=True`, `--patch all`, against
`FASTGRIDRESP_OFF=1`:

| | `_get_exc_deriv2_grid_response` | Hessian |
|---|---|---|
| alanine dipeptide | 4.64 → 1.36 s (3.4x) | 13.68 → 10.44 s (1.31x) |
| penicillin V | 16.10 → 4.58 s (3.5x) | **56.04 → 44.87 s (1.25x)** |

One 4096-point block: taxol (113 atoms) 0.721 → 0.250 s (2.9x); **PfPMT
(284 atoms) 4.11 s, where GPU4PySCF raises its int32 `AssertionError`.**
The arithmetic is still O(ngrids · natm³) — a 284-atom grid-response
Hessian is now possible but not cheap (~830 blocks × 3 terms).

## Correctness

`tests/test_gridresp.py`: one block against `get_d2weight_dAdB` + `contract`,
both partition schemes, with and without the Treutler radii adjustment, a
block spanning several atoms: ≤ 8.5e-16 relative.  Whole grid-response
Hessians against the replaced functions in the same process: RKS B3LYP
1.3e-12, UKS PBE 2.0e-12, RKS ωB97M-V (the VV10 path) 2.7e-12 Eh/bohr².
