# fastdfmp2 — the DF-MP2 pair-energy loops (catalogue I1)

`docs/KERNEL_TARGETS_1.8.1.md` §5.1 rank 8.  `mp/dfmp2_addons.py`'s
`get_dfmp2_energy_pair_intra` and `get_dfump2_energy_pair_intra` loop over
occupied pairs (`i ≥ j`, or all `i, j` for the opposite-spin term) and per
pair run one `(nvir × naux) @ (naux × nvir)` GEMM, four elementwise kernels
and two host synchronisations (`float(e_bi1)`, `float(e_bi2)`): 73–81 % of the
DF-MP2 step and 92–94 % of DF-UMP2 (§10.10, §12.4).

## The change

`electrolites/fastdfmp2.py` replaces both.  Pairs go in blocks of `B`
occupied orbitals: one `(B·nvir × naux) @ (naux × B·nvir)` GEMM per block
pair (lower block triangle only for the same-spin sums), and one kernel per
block reduces each `(i, j)` slice:

```
bi1 = Σ g_ab² / D_ab     bi2 = Σ g_ab g_ba / D_ab     ss = Σ (g_ab − g_ba)² / D_ab
```

with `D_ab = e_i + e_j − e_a − e_b` — GPU4PySCF's `t = g/D, e = Σ t g` with the
division folded in — and no host synchronisation until the end.  The
amplitudes (`t2`, when `with_t2` asks for them) are written block by block
from the same GEMM output.  Pair sums accumulate in double precision; the
integrals' own precision (`dtype_cderi`, float32 by default) is unchanged.

`B = 16` is a cap, measured: on 81 × 349 and 226 × 897 (occupied × virtual)
integrals `B` = 8 / 16 / 32 / 64 gives 0.16 / **0.07** / 0.08 / 0.09 s and
8.68 / **7.43** / 8.06 / 8.89 s (larger blocks waste more of the diagonal
block pairs, smaller ones leave the GEMM too short); it is lowered further
only if free memory requires.

## Measured

`benchmarks/path_anatomy.py --feature dfmp2 / mp2 --patch all`, with and
without `FASTDFMP2_OFF=1`, def2-SVP, one A100-SXM4-80GB:

| | before | after | |
|---|---|---|---|
| **DF-MP2 step, penicillin V (42)** | 1.29 s (pair loop 0.98 s) | **0.59 s** | **2.2x** |
| **DF-MP2 step, taxol (113)** | 17.06 s (pair loop 13.75 s) | **13.36 s** (10.07 s) | **1.28x** (1.37x) |
| **DF-UMP2 step, penicillin V⁺** | 3.37 s (pair loops 3.01 s) | **0.92 s** | **3.7x** |
| **DF-UMP2 step, taxol⁺** | 55.58 s (pair loops 51.53 s) | **41.88 s** (37.87 s) | **1.33x** (1.36x) |

At 42 atoms the loop was launch- and synchronisation-bound and is now gone
from the profile; at 113 atoms it is the GEMM itself (fp32, ~15 TF, the
`gemm only` probe is 7.75 s of the 7.43 s), so what is left there is
arithmetic.  TF32 would cut it eightfold at 1e-3 relative precision, which is
not on offer.

## Correctness

`tests/test_dfmp2.py`: the three pair-energy modes with and without
amplitudes, on random float64 and float32 integrals of two sizes (one with
`nocc` not a multiple of the block), against GPU4PySCF's functions — ≤ 3.4e-16
relative in float64, ≤ 4.7e-7 in float32 (GPU4PySCF sums in float32, this in
float64) — and whole DF-MP2 / DF-UMP2 correlation energies of glycine /
glycine⁺, equal to 1e-10.  In the timed runs the correlation energies agree
to 3e-14 (RMP2) and 7e-11 Eh (UMP2, float32 integrals).

| | |
|---|---|
| `electrolites/fastdfmp2.py` | both functions |
| `tests/test_dfmp2.py` | the A/B |
| `FASTDFMP2_OFF=1`, `FASTDFMP2_BLOCK` | upstream's loops; the block size |
