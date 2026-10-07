# fastpbcgrad — the periodic gradient's local 1e term and the stress tensor's non-local PP term (J7, J5)

Two entries of `docs/KERNEL_TARGETS_1.8.1.md` §4 J, rank 14 of §5.1, both
measured in §11.6 and both in `electrolites/fastpbcgrad.py`.
`FASTPBCGRAD_OFF=1` keeps GPU4PySCF's code for both.

## J7: one density pass instead of one grid pass per atom

`pbc/grad/{krhf,kuhf}.py:grad_elec` and the Gamma-point
`{rhf,uhf}.Gradients.grad_elec` (FFTDF, i.e. not multigrid) take the local
nuclear / local-pseudopotential derivative atom by atom:
`hcore_generator(ia)` evaluates every AO at every k-point over the whole grid
and builds an `(nk, 3, nao, nao)` matrix, which is then contracted with the
density.  The contraction is linear in the density, so

    sum_k tr(h1(ia)_k dm_k) = sum_r v_ia,x(r) rho(r) - (h1 rows/columns of atom ia)

with `rho(r) = sum_k sum_ij conj(ao_ki) dm_kji ao_kj` evaluated once and
`v_ia,x` the same `ifft(i G SI_ia vloc_ia(G))` GPU4PySCF forms per atom.
`dh1e_local` does that; the four `grad_elec` are GPU4PySCF's source with the
loop replaced, bound to their own modules.

## J5: the non-local PP strain derivative without per-atom host work

`pbc/grad/rks_stress.py:_get_pp_nonloc_strain_derivatives` is a central
difference over 9 strains × 2 cells of the GTH non-local energy, which
GPU4PySCF evaluates per k-point *per atom*: the projector Gaussians on `G+k`
by host `eval_gto` and host `_qli` (identical for every atom of a species),
copied to the device atom by atom, `ft_ao` re-sorting the basis and
allocating `nao × ngrids` per k-point, two GEMMs per atom.
`pp_nonloc_energy` builds the projector table once per species and k-point,
applies the atoms' structure factors on the device, does a species' atoms as
one GEMM and one batched contraction, sorts the basis once per cell, reuses
one `ft_ao` buffer, and applies the basis transformation to the small
projected product rather than to the `ngrids × nao` transform.  Same
displacement, same expressions.  Patched in `rks_stress`, `krks_stress`,
`kuks_stress`, `uks_stress` and `krhf_stress`, which import the name.

## Result

Diamond, PBE/gth-dzvp/gth-pade, `--patch all` against `FASTPBCGRAD_OFF=1`
(`benchmarks/kpt_anatomy.py --grad --stress`):

| | 8-atom 2×2×2 FFTDF | 2-atom 3×3×3 FFTDF | 8-atom 2×2×2 multigrid_v2 |
|---|---|---|---|
| J7 (per-atom `hcore`) | 4.03 s → one pass | 0.17 s → one pass | — (G-space branch) |
| **gradient** | **7.12 → 3.56 s (2.0x)** | 1.64 → 1.52 s | 1.18 → 1.36 s (noise; untouched) |
| J5 (non-local PP strain) | 7.43 → 1.25 s (5.9x) | 5.37 → 1.90 s (2.8x) | 7.84 → 1.49 s (5.3x) |
| **stress** | **17.30 → 10.93 s (1.58x)** | 8.65 → 5.39 s (1.60x) | 13.24 → 8.33 s (1.59x) |

## Correctness

`tests/test_pbcgrad.py`: each patched `grad_elec` against the replaced one
on the same SCF — KRKS / KUKS diamond 2×2×2 and RKS / UKS at Gamma (GTH),
KRKS and RKS LiH all-electron (the Coulomb branch): ≤ 2.1e-14 Eh/bohr.  The
J5 term against the replaced function on the same density (diamond 2×2×2,
SiC 2×2×2 and Gamma, one species and two): ≤ 3e-11 — a central difference
over 2·disp ≈ 6e-5 of energies equal to round-off.
