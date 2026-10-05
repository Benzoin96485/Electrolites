# fastc6 — C6 coefficients on the device (catalogue C5)

`docs/KERNEL_TARGETS_1.8.1.md` §5.1 rank 9.  `properties/c6.py` computes C6
from the full Casida spectrum: `td.get_ab` builds A and B — the two-electron
part from `mol.intor('int2e_sph')` **on the CPU**, restored to the full
`nao⁴` tensor on the host and copied over — and `_solve_full_spectrum` does the
Cholesky factor of A − B, `Lᵀ(A+B)L`, its `eigh`, a triangular solve and the
back-transformation **in NumPy**.  §11.9: 95 % of a 22-atom run (41 % `get_ab`,
54 % host linear algebra).  And `calc_c6(td, td)`, the C6 of a molecule with
itself, solves the same spectrum twice.

## The change

`electrolites/fastc6.py`, the arithmetic unchanged:

* the AO ERI tensor comes from GPU4PySCF's own GPU fill
  (`scf/int4c2e.get_int4c2e`, which writes one half of each pair index and is
  symmetrised as its own test does) and goes into `get_ab`'s unchanged
  transformation — `get_ab`'s source with only those three lines replaced,
  executed in its module's namespace — when twice `nao⁴` doubles fit on the
  device, otherwise the host path.  The long-range (erf) tensor of an RSH
  functional stays on the host: the GPU fill's `omega` mode does not
  reproduce libcint (C6 of NH3/ωB97X off by 36 % when tried);
* Cholesky, `eigh`, the triangular solve, the back-transformation and the
  normalisation run in CuPy; only the final `(X, Y)` come back;
* `calc_c6(td, td)` solves the spectrum once.

A − B not positive definite falls back to GPU4PySCF's non-Hermitian path.

## Measured

`benchmarks/path_anatomy.py --feature c6 --patch all`, with and without
`FASTC6_OFF=1`, def2-SVP:

| | before | after | |
|---|---|---|---|
| alanine dipeptide (22) / B3LYP | 93.2 s | **14.3 s** | **6.5x** |
| alanine dipeptide (22) / PBE | 92.9 s | **13.9 s** | **6.7x** |
| glycine (10) / B3LYP | 2.92 s | **1.08 s** | **2.7x** |

(The first glycine run on a node with a cold CuPy cache read 6.06 s: the one-
time NVRTC compile of the device linear-algebra kernels.)  C6 values agree to
1.5e-13 relative in the timed runs.  The method is O(nov³) and holds `nao⁴`
on the device; it is for the small molecules C6 is computed for, and beyond
~250 AOs (80 GB) the ERI goes back to the host path.

## Correctness

`tests/test_c6.py`: B3LYP, PBE, HF, ωB97X (RSH), TDA, a molecule with itself
and a pair of different molecules — C6 to ≤ 1.3e-12 relative, excitation
energies to ≤ 1.2e-10 Eh (the GPU fill screens at 1e-13; A and B agree with the
host build to 9e-11).

| | |
|---|---|
| `electrolites/fastc6.py` | the module |
| `tests/test_c6.py` | the A/B |
| `FASTC6_OFF=1` | upstream's functions |
