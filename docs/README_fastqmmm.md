# fastqmmm — QM/MM's QM-nucleus–MM-charge terms (catalogue F6, F13)

`gpu4pyscf.qmmm.itrf` evaluates the point-charge interaction between the QM
nuclei and the MM charges with host NumPy over an `n_mm × natm` array, in
three places:

* `QMMMSCF.energy_nuc`, which `scf/hf.py:energy_tot` calls **every SCF
  cycle** although neither set of positions moves — 2.42 s of an 11.4 s
  patched QM/MM SCF + gradient at 10⁵ charges;
* `QMMMGrad.grad_nuc` and `grad_nuc_mm`, the same pair forces summed once
  over the charges and once over the QM atoms (0.30 s each).

`electrolites/fastqmmm.py` evaluates all three in CuPy, in blocks of 65 536
charges, and caches the energy for the SCF's cycles and the pair forces for
both gradient methods, against the coordinates and charges they came from
(compared on every call, so a moved QM or MM atom recomputes).
`FASTQMMM_OFF=1` keeps GPU4PySCF's methods.

## Result

Penicillin V, B3LYP/def2-SVP, in neutral ±0.4 e point dipoles no closer than
4 Å to a QM atom (`benchmarks/path_anatomy.py --feature qmmm`), `--patch all`:

| | 10⁴ charges | 10⁵ charges |
|---|---|---|
| `energy_nuc`, 14 calls | 0.28 → 0.01 s | 2.42 → 0.05 s |
| `grad_nuc` + `grad_nuc_mm` | 0.06 → 0.01 s | 0.60 → 0.01 s |
| **SCF + gradient (wall)** | 6.80 → 6.60 s | **11.36 → 8.19 s (1.39x)** |

## Correctness

`tests/test_qmmm.py`: against the replaced methods on the same objects, at
10³ and 1.5·10⁵ charges (three device blocks), ≤ 9e-15 relative, including
after the MM charges and the QM atoms move; a whole QM/MM SCF + gradient
against `FASTQMMM_OFF=1`: energy 3.7e-13 Eh, QM forces 6.8e-14, MM forces
1.7e-16.  The device sums in a different order, so the energy is not bitwise
the host's (1.7e-11 Eh at 10⁵ charges, 1e-14 relative).

## What is left

At 10⁵ charges the MM-specific remainder is the two derivative integrals —
`extra_force`'s charge-contracted `int1e_grids_ip1` (0.89 s, 10.9 %) and
`grad_hcore_mm`'s density-contracted `int1e_grids_ip2` (0.55 s, 6.7 %) —
which run GPU4PySCF's `int3c1e_ip` kernels unchanged.  `fast1e`'s far-field
forms (1.45–1.62x on the non-derivative kernels) would carry over to their
`GINT_g1e` and `*_save_u2` paths; on those numbers that is worth about
0.5 s of the 8.2 s, and it needs the `int3c1e_ip` translation unit ported
to NVRTC the way `fast1e`'s was.  Not done in this round.
