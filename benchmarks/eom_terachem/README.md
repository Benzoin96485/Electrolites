# Singlet EOM-EE-CCSD vs TeraChem RR-EOM-CCSD (TM complexes)

Reference set: TeraChem RR-EOM-CCSD (rank-reduced CCSD, Cholesky 1e-6,
5 singlet roots, transition properties) on Ir/Pt/Rh/Ru closed-shell
complexes with 15–30 atoms, cc-pVDZ (spherical) + LANL2DZ/ECP on the metal,
frozen core = `ccbox_frozen_core` of each input, run on one H200
(`m4787_cfs/data/EOM-RR-CCSD/xyz_15-30atoms_spin1_opt`, 214 cases, 206
finished; `terachem_index.txt` lists case, atoms, nao, frozen core,
finished flag, wall time).

Here: canonical EOM-EE-CCSD (`electrolites.eom_gpu`) on one A100-80GB
(Perlmutter), 9 cases with nao 154–291.  Same basis/ECP/frozen core;
coordinates converted to Bohr with TeraChem's 0.529177249 (E_nuc matches),
SCF energies agree to < 8 µEh.  Method details and the profile are in
`docs/FEATURE_TARGETS_PYSCF_2.13.md` §17.11.

## Files

| file | what |
|---|---|
| `run_eom.py CASE` | SCF + CCSD + right EOM-EE (5 roots) → `results_eom/` |
| `run_props.py CASE` | + Λ, left EOM-EE, transition dipoles, f → `results_props/` |
| `compare.py`, `comparison.txt` | energies vs TeraChem (`results_eom/`) |
| `compare_props.py`, `comparison_props.txt` | energies, f and timings vs TeraChem (`results_props/`) |
| `prof_*.py`, `profile_summary.txt` | profiles: ladder path / per-call times (`prof`, before the MO-vvvv fix; `prof2`, after), per-einsum sigma (`prof3`), left EOM and Λ update (`prof4`), memory (`prof4/mem_`) |
| `launch.sh GPU CASE...`, `launch_props.sh`, `prof.sh` | Perlmutter launchers (cudatoolkit/12.9, env `barw-dev`) |

`results_eom/` is from the first round (the Davidson zero-root and OOM fixes
in; SOXYIN01 re-run after; the others before the chunked MO-vvvv build, so
their EOM times for v ≥ 198 are the AO-ladder ones).  `results_props/` is
the final code throughout.

## Results (9 cases, 45 roots)

* Excitation energies: canonical − RR = +0.025 eV mean, ≤ 0.073 eV.  The
  shift is mostly per molecule (ZEQHOS all roots +0.04–0.05, AFAPOM ±0.006)
  and correlates with D1 (r = +0.61) and inversely with the RR rank/(o·v)
  (r = −0.43): the rank-reduced approximation is the likely main source.
  TeraChem's own last macro-iteration still moves roots by 1–4 meV
  (residual threshold 1e-4), and its transition table uses the previous
  iteration's energies.
* Oscillator strengths: |Δf| ≤ 0.006 except one near-degenerate pair in
  EDOYAZ whose two states swap order (f 0.059/0.018 vs 0.020/0.056, sums
  0.077 vs 0.076); median relative difference ~3 % for f > 0.01.
* Time (whole job incl. Λ, left vectors, properties): 10–15x TeraChem's
  total for v ≤ 182, 4–6.5x for v 198–211, 2x for EDOYAZ (v 226), where
  the MO vvvv does not fit beside the intermediates on 80 GB and the solve
  falls back to the AO-direct ladder.  The largest cases of the set
  (nao ~394) need a memory-scalable sigma before they fit at all.
