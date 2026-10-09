# fastpcmbatch — PCM's multi-vector `int1e_grids` in one integral pass (F9), and `dqdx` once (F11)

## F9: the response half

`gto/int3c1e.py:int1e_grids` with a stack of densities (or charge vectors)
loops over them, one full `(pair, point)` integral pass each.
`PCM._B_dot_x` — the solvent response inside every PCM CPHF, TDDFT-PCM and
polarizability — calls it with all its right-hand sides at once, twice per
response (the potential of the densities on the cavity; the Fock
contribution of the induced charges).  `electrolites/fastpcmbatch.py`, for
three or more right-hand sides, builds the plain integrals once per chunk of
cavity points (`fast1e`'s kernels, through the new `fast1e._plain_chunk`)
and contracts all of them by GEMM: `v[i,g] = I[g]·dm_i`, `vmat_i = q[i]·I`.
The chunk is sized from free memory; fewer right-hand sides, or a basis
`fast1e` declines, keep the per-vector loop.  Patched wherever the name is
imported (`gto.int3c1e`, `solvent.pcm`, `solvent.hessian.pcm`,
`tdscf.{rhf,uhf}`, `pop.esp`, `properties.eda`).

## F11: `get_dqsym_dx` once

`solvent/hessian/pcm.py:get_dqsym_dx` is computed three times in one PCM
Hessian with the same object, density and atoms; it is kept for one reuse
key (object, atoms, geometry, an element-wise check of the density) and a
copy handed back.  Bitwise.

`FASTPCMBATCH_OFF=1` keeps GPU4PySCF's for both.

## Result

IEF-PCM B3LYP/def2-SVP Hessian, `--patch all`, against `FASTPCMBATCH_OFF=1`:

| | `PCM._B_dot_x` | `get_dqsym_dx` ×3 | PCM Hessian |
|---|---|---|---|
| alanine dipeptide | 1.85 → 0.32 s (5.8x) | 0.47 → 0.19 s | 12.53 → 10.76 s (1.16x) |
| penicillin V | 10.92 → 6.29 s (1.74x) | 1.19 → 0.45 s | **57.99 → 52.27 s (1.11x)** |

Hessian traces agree to 4e-11 / 4e-10.  The `h1` half of F9 (3·natm
single-charge calls in `analytical_grad_vmat`, 1.1 s at 42 atoms) is
unchanged.

## Correctness

`tests/test_pcmbatch.py`: seven densities and seven charge vectors against
the per-vector loop, spherical and cartesian: ≤ 2.2e-15 relative; a whole
ethanol IEF-PCM Hessian against `FASTPCMBATCH_OFF=1` 7.2e-12, a C-PCM TDA
6.5e-15.
