# fastnacms — the force and all NACs through the batched solver (catalogue D12)

`docs/KERNEL_TARGETS_1.8.1.md` §5.1 rank 3.  `td.force_and_nacv` — the call
an FSSH step makes for the active state's force and the NACVs of every state
pair — is a loop in GPU4PySCF 1.8.1 (`tdscf/rhf.py:478-489`): one
`td_nac.kernel(states=(i, j))` per pair, i.e. one Z-vector CPHF and one set of
derivative-integral contractions per pair.  GPU4PySCF also ships a batched
solver, `td.NACGradients()` (`NAC_multistates`, `nac/tdr{h,k}s_grad_nacv.py`,
experimental and singlet-only), that solves the Z-vector equations of every
pair and of the force in one CPHF and contracts the derivative integrals for
all of them in one multi-density call.  §11.1 measured it 1.86–2.1x faster and
in agreement with the serial path.

## The change

`electrolites/fastnacms.py` replaces `TDBase.force_and_nacv` (which the
restricted and unrestricted TDA/TDDFT/TDHF classes all inherit) and takes the
batched path when it is valid:

* a restricted reference and singlet excitations (the batched solver raises
  otherwise; open shell and triplets fall through);
* no solvent model;
* the requested pairs are *all* pairs among the states they name — the
  method's own default and what FSSH asks for.  A sparser list falls through,
  because the batched solver would compute every pair among those states.

The Z-vector settings are taken from the `td_nac` object passed in.  The
return value has the serial method's form.  `stats()` counts batched and
serial calls and why each serial one was declined.

## Measured

`benchmarks/path_anatomy.py --feature forcenac --basis def2-svp --patch all`,
5 roots, the force on state 1 and all 15 NACVs, with and without
`FASTNACMS_OFF=1`, back to back:

| system / functional | serial | batched | |
|---|---|---|---|
| penicillin V (42) / B3LYP | 147.8 s | **71.1 s** | **2.08x** |
| penicillin V (42) / PBE | 127.6 s | **63.4 s** | **2.01x** |
| taxol (113) / B3LYP | 738.5 s | **425.9 s** | **1.73x** |

The Z-vector solve goes 67.8 → 8.0 s (B3LYP, 42 atoms) and 419.7 → 114.7 s
(113 atoms); what is left is D10, the multi-density
`jk_energies_per_atom` (48.9 s and 247.9 s, 58–69 % of the batched run) —
§5.1 rank 6.

## Correctness, and two findings about the reference

`tests/test_nacms.py`, five cases (B3LYP RPA, PBE TDA, HF RPA, glycine, and a
density-fitted B3LYP with the force on the ground state), plus the declines
(a triplet and a sparse pair list take the serial path).  Two properties of
GPU4PySCF's own code set what "agree" can mean:

1. **The Krylov solver stops at a residual of ~3e-7 whatever `tol` says.**
   `lib/cupy_helper.py:krylov` drops new directions whose squared norm is below
   `DSOLVE_LINDEP = 1e-13` and stops when none is left, so `cphf_conv_tol`
   below ~3e-7 has no effect.  With that floor the serial and batched results
   differ by up to 1e-5 relative, and which pairs differ depends on what else
   is in the batch; with the floor lowered (`lindep = 1e-22`, `tol = 1e-10`)
   the excited–excited NACVs agree to 1e-8–1e-11.  The test lowers it.
2. **GPU4PySCF's ground-to-excited NACVs are not reproducible with
   themselves.**  Two serial `nac.kernel(states=(0, 1))` calls on the same
   glycine TDHF states differ by up to 1.1e-4 (largest element 0.25), and the
   batched solver scatters by the same amount.  So the test runs the serial
   path twice and requires the batched result to agree with it to three times
   that scatter (or 1e-6 relative, whichever is larger): it does in every
   case, with the same sign; the force agrees to 1e-8–1e-13.

Neither is introduced here, and both affect the serial path equally; they are
recorded because a tighter test is not possible against this reference.  The
second was traced in rank 6 (`README_fasttdejk.md`): GPU4PySCF's
`_jk_energies_per_atom` kernels are not reproducible, and not correct on
general density pairs; with `fasttdejk` in place the NACVs repeat.

## What is left

Open-shell and triplet NACs (the batched solver is singlet-only; extending it
is upstream's own open item), and D10.

| | |
|---|---|
| `electrolites/fastnacms.py` | the method |
| `tests/test_nacms.py` | the A/B |
| `FASTNACMS_OFF=1` | upstream's serial loop |
