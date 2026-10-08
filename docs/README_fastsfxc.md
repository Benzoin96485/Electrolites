# fastsfxc — the spin-flip multi-collinear XC kernel without the full tensors (D13)

`tdscf/_uhf_resp_sf.py:eval_xc_eff_sf` serves the spin-flip TDA response
(`cache_xc_kernel_sf`, second order) and its gradient
(`grad/tduks_sf.py:_contract_xc_kernel`, third order).  For each grid point
it evaluates the collinear functional on 20 principal-axis samples through
`NumInt.eval_xc_eff(spin=1)`, which

1. converts libxc's derivatives in `(rho, sigma)` to the full tensors in
   `(rho, ∇rho)` — `xc_deriv.transform_kxc`, a chain of einsums over
   `2·4·2·4·2·4 = 512` doubles per point and sample; measured at **~75 %**
   of the evaluation (14 ms per 57 k points, 14 calls per 40 k-point block);
2. then `ud2ts` turns all of it into the total/spin basis;
3. and `eval_xc_eff_sf` keeps one block of it — `fxc_ts[1,:,1]` and
   `kxc_ts[1,:,1,:,c]` — summed over the samples.

`electrolites/fastsfxc.py` takes libxc's raw derivatives (the functional's
components summed with their weights, as `eval_xc_eff` does) and applies the
chain rule directly in the total/spin basis for just those blocks, in one
CUDA kernel that also sums over the samples:

    ts derivative = Σ f3_ABC ã_A ã_B ã_C + Σ f2_AB (b̃_A ã_B + 2 permutations)

with `A` over `(ρ_u, ρ_d, σ_uu, σ_ud, σ_dd)`, `ã` their first derivatives
with respect to the total/spin variables (linear in the spin gradients) and
`b̃` their constant second derivatives.  Nothing of size `512 · ngrids · 20`
is formed.  LDA and GGA at second and third order; a meta-GGA, or a
functional libxc cannot evaluate on the GPU, keeps GPU4PySCF's path.
`FASTSFXC_OFF=1` keeps GPU4PySCF's.

## Result

SF-TDA (5 roots) from a triplet UKS reference and the gradient of state 1,
B3LYP/def2-SVP, `collinear='mcol'`, `--patch all`, against
`FASTSFXC_OFF=1`:

| | gradient `_contract_xc_kernel` | SF-TDA + gradient |
|---|---|---|
| alanine dipeptide | 2.57 → 0.85 s (3.0x) | 6.68 → 4.85 s (1.38x) |
| penicillin V | 5.27 → 1.61 s (3.3x) | 21.32 → 17.24 s (1.24x) |

## Correctness

`tests/test_sfxc.py`: `fxc_sf` and `kxc_sf` against GPU4PySCF's adapter on
the same densities, LDA, PBE, B3LYP, PBE0, second and third order:
≤ 3e-16 relative.  Water triplet SF-TDA + gradient against
`FASTSFXC_OFF=1`: excitation energies ≤ 1.4e-15, gradients ≤ 4e-15.
