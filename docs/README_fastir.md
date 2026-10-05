# fastir — IR intensities without the second CPHF (catalogue C10)

`docs/KERNEL_TARGETS_1.8.1.md` §5.1 rank 2.  GPU4PySCF 1.8.1's
`properties/ir.py:eval_ir_freq_intensity` computes the Hessian
(`hessian_obj.kernel()`, whose `hess_elec` builds `make_h1` and solves the
`3 natm`-right-hand-side CPHF) and then, for the dipole derivative, builds
`make_h1` and solves **the same CPHF a second time** (`ir.py:91-97`, under
upstream's own `# TODO: compact with hessian method, which can save one time
cphf solve`).  §10.2 measured the duplicate at 41.8 % of a patched 42-atom IR
run.

## The change

`electrolites/fastir.py` replaces the driver.  For the duration of
`hessian_obj.kernel()` it wraps the Hessian object's `solve_mo1` (an instance
attribute, removed afterwards) and keeps the `(mo1, mo_e1)` that call
returns; the dipole derivative then uses that `mo1`.  The two solves upstream
does have identical inputs — `mo_energy`, `mo_coeff`, `mo_occ`, `make_h1` for
every atom, `gen_vind` — so this is the same quantity computed once.  The
`mo1` is in the MO basis in both places (`ir.py:93`, "Different from PySCF,
mo1 is all in mo!").  If a Hessian class ever solved without going through
`hessian_obj.solve_mo1`, the driver falls back to the second solve
(`stats()['resolved']`).  Everything else is GPU4PySCF's code.

Bitwise identity against upstream is not available as a test, because
upstream is not bit-reproducible with itself: two stock IR runs on the same
SCF differ by up to 8.5e-9 km/mol (`tests/test_ir.py` prints it).

## Measured

`benchmarks/path_anatomy.py --feature ir --basis def2-svp --patch all`, with
and without `FASTIR_OFF=1`, back to back on one A100-SXM4-80GB; "IR driver"
is `eval_ir_freq_intensity` (the SCF before it, unchanged, is excluded):

| system / functional | IR driver, before | after | | `solve_mo1` calls |
|---|---|---|---|---|
| alanine dipeptide (22) / B3LYP | 13.78 s | **8.30 s** | **1.66x** | 2 → 1 |
| penicillin V (42) / B3LYP | 82.8 s | **45.6 s** | **1.82x** | 2 → 1 |
| penicillin V (42) / wB97M-V | 441.3 s | **237.8 s** | **1.86x** | 2 → 1 |
| taxol (113) / B3LYP | 2159.2 s | **1103.6 s** | **1.96x** | 2 → 1 |

The gain grows with size because the CPHF's share of a Hessian does
(`docs/HESSIAN_1.8.1.md` §4), and it holds for a VV10 functional, where the
CPHF is a different mix of terms (§13.1).  At 113 atoms it is within 2 % of the
2x ceiling.

## Correctness

`tests/test_ir.py`, the patched driver against `ir._eval_ir_freq_intensity_orig`
on the same SCF, each with a fresh Hessian object, five cases (B3LYP, HF,
wB97M-V, PBE with density fitting; water, methanol, glycine): frequencies
within 1.3e-8 cm⁻¹, intensities within 4.2e-9 km/mol — the size of two stock
runs' own difference — and the reuse counter advanced in every case.

## Files and switches

| | |
|---|---|
| `electrolites/fastir.py` | the driver |
| `tests/test_ir.py` | the A/B |
| `FASTIR_OFF=1` | upstream's driver |
