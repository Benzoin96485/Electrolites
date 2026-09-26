# A restricted meta-GGA Hessian that is the same twice: the second in-place contraction

`docs/DF_WORK_LIST.md` **W23**, found by W16.  W16 fixed one in-place cuTENSOR
contraction in `hessian/rks.py:_nr_rks_fxc_mo_task` and left a second source
of run-to-run noise on the meta-GGA Hessian path "not found": an unpatched
TPSS/def2-SVP DF Hessian of glycine gave four traces in four runs spread over
3e-5 Eh/bohr², and a patched one spread just as far.

It is the same bug in a second function.  `hessian/rks.py:_get_vxc_deriv1_task`
writes

```python
wv2 = contract('xyg,sxg->syg', wf, dR_rho1, out=dR_rho1)
```

in both its GGA and its meta-GGA branch — the output is the second operand —
and cuTENSOR gets that wrong at five density components.  The patched Hessian
still spread because `fastxchess`, which replaces `_get_vxc_deriv1`, declines
meta-GGAs and hands them to exactly this line.

---

## 0. Summary

* **Found by bisection, not by grep** (§1): every stage of a meta-GGA Hessian
  evaluated three times on one converged SCF in one process.  Exactly two are
  not reproducible — `_get_vxc_deriv1` (2e-6 relative between calls) and
  `nr_rks_fxc_mo` (1e-4, W16's line) — and every stage downstream of them
  inherits it.  A grep for `contract(..., out=<an operand>)` over the whole of
  GPU4PySCF 1.8.1 afterwards finds these three lines and no others.
* **It is not noise around the right answer; it is a wrong answer that is
  also noisy** (§2).  Against PySCF's CPU `_get_vxc_deriv1` on the same
  orbitals and grid, GPU4PySCF's is off by **1.6e-5 and 4.2e-5** relative in
  two runs; with the fix it agrees to **2.3e-12**.  The Hessian moves by
  3.6e-4 Eh/bohr² in its trace, ten times the run-to-run spread that made the
  bug visible.
* **The fix is one indirection** (§3).  `fastxchess` replaces the `contract`
  that `hessian/rks.py` imported with `_alias_safe_contract`, which is the same
  call unless `out` overlaps an operand, and otherwise contracts into a
  temporary and copies.  That repairs all three lines at once — including
  W16's, so upstream's `nr_rks_fxc_mo` is right again whenever `fastxcfxc` is
  off or declines — and costs nothing measurable (`_get_vxc_deriv1` on a
  42-atom TPSS Hessian: 15.03 s unpatched, 15.18 s and 14.98 s repaired).
* **The success criterion is met** (§4): a 42-atom DF-TPSS Hessian gives
  traces **40.9142598198 and 40.9142598199** in two processes, where the
  unpatched one gives 40.9141875571 and 40.9143467896 (a spread of 1.6e-4);
  glycine gives the same trace to 1e-10 in eight processes; the direct
  (non-DF) Hessian and three repeats inside one process agree to 1e-11.
* **The falsification criterion does not fire**: the SCF is reproducible to
  the printed digit in every run above (E = −1503.6304156812 six times) and
  the spread is gone with the SCF unchanged, so it was never a convergence
  setting.

Everything here is GPU4PySCF 1.8.1, def2-SVP (def2-SVP-jkfit for the DF
runs), level-3 grids, `conv_tol = 1e-9`–`1e-10`, `conv_tol_cpscf = 1e-6`
unless a row says otherwise.  §1–§3 were run on a login-node A100 (they are
correctness measurements, and the rule against the login node in
`DF_WORK_LIST.md` §5.1 is about timings); §4's penicillin rows and the
`_get_vxc_deriv1` timings are one A100-SXM4-80GB of a whole `--qos=debug`
node.

---

## 1. Where the noise is

`benchmarks/mgga_hess_repro.py`: one converged DF-TPSS SCF of glycine, then
each stage called three times with the same arguments, each repeat compared
element-wise with the first (relative to the largest element).  `--load`
reuses one SCF across processes, so every column below is on the same
orbitals.

| stage | GPU4PySCF, unpatched | `--patch fastxchess` | `--patch all` |
|---|---|---|---|
| `cache_xc_kernel` | 0 | 0 | 0 |
| `_get_vxc_diag` | 0 | 0 | 0 |
| `_get_vxc_deriv2` | 0 | 0 | 0 |
| `_get_exc_deriv2` | 0 | 0 | 0 |
| **`_get_vxc_deriv1`** | **2.0e-6**, 0 | 0, 0 | 0, 0 |
| ip2 `_jk_energy_per_atom` | 6e-14, 9e-14 | 8e-14, 1e-13 | 1e-13, 9e-14 |
| h1 `_get_veff` | 1e-17 | 1e-17 | 4e-19 |
| `partial_hess_elec` | 5e-13, 1e-12 | 2e-12, 7e-13 | 1e-12, 7e-13 |
| `make_h1` | **9.1e-7**, 9.1e-7 | 4e-17, 0 | 6e-19 |
| `gen_vind`'s `fx`, 6 right-hand sides | **7.8e-5** | 4e-17 | 4e-17 |
| `nr_rks_fxc` (AO basis) | 0 | 0 | 0 |
| **`nr_rks_fxc_mo`** | **1.2e-4**, 2.7e-5 | 0, 0 | 0, 0 |
| `solve_mo1` | **2.0e-5** | 2e-14 | 1e-14 |
| `hess_elec` | **9.8e-8**, 4.0e-8 | 1.0e-12 | 1.8e-12 |

Two sources, both upstream of everything that differs, and everything that
does not differ is upstream of neither.  The residual 1e-12 in
`partial_hess_elec` and `hess_elec` is the `ejk_int3c2e_ip2` kernel's atomics,
the same 1e-13 on its own row that `docs/README_fastdfhessip.md` §6 records
for the GGA Hessian; the `nr_rks_fxc_mo` column is 0 under `--patch
fastxchess` because the repair of §3 reaches W16's line as well.

A grep afterwards, for every `contract(...)` call in GPU4PySCF 1.8.1 whose
`out=` names one of its own operands, finds three lines, all in
`hessian/rks.py`: 1369 and 1420 (the GGA and meta-GGA branches of
`_get_vxc_deriv1_task`) and 3543 (`_nr_rks_fxc_mo_task`, W16's).
`hessian/uks.py` has none, and an unrestricted DF-TPSS Hessian of the glycine
cation is reproducible unpatched (8.671874422998 and 8.671874422996 in two
processes), which is why the item says *restricted*.

### 1.1 Why only five components

`contract('xyg,sxg->syg', wf, x, out=x)` with random operands, 4096 grid
points, 20 trials (`tests/test_mgga_inplace.py`):

| `ncomp` | worst error of the in-place call, relative |
|---|---|
| 1 (LDA) | 0 |
| 4 (GGA) | 1.3e-16 |
| 5 (meta-GGA) | **0.14 – 0.29** |

It is a race rather than a deterministic miscompile — one call in several
comes out right — which is what makes a Hessian that contains a few hundred of
them differ from run to run.  At four components cuTENSOR evidently reads each
output element's inputs before it writes; at five it does not.  The GGA branch
of `_get_vxc_deriv1_task` carries the same line and is correct only because
`ncomp = 4` there; `fastxchess` has covered the GGA case with its own code
since B5, so no GGA result in this repository ever went through it.

---

## 2. It is a wrong answer, not only a noisy one

The obvious reading of a Hessian that moves by 3e-5 between runs is that the
right answer sits somewhere in the middle.  It does not.  The corrupted
elements are corrupted by a third of their size, and the dilution through the
grid sum is what makes the final damage small, not any averaging.

`_get_vxc_deriv1` for TPSS on glycine, GPU4PySCF against PySCF 2.13's CPU
`hessian/rks.py:_get_vxc_deriv1` on the same orbitals and the same level-3
grid (the CPU result taken to the same MO block):

| | relative to the CPU |
|---|---|
| GPU4PySCF, run 1 | 1.6e-5 |
| GPU4PySCF, run 2 | 4.2e-5 |
| repaired | **2.3e-12** |
| (control) `_get_vxc_diag`, repaired | 3.1e-13 |
| (control) `_get_vxc_deriv2`, repaired | 1.1e-15 |

so the repaired function is the CPU one and upstream's is not, by an amount
that changes each run.  On the whole Hessian (direct, not DF, so that PySCF's CPU Hessian computes
the same quantity; `conv_tol_cpscf = 1e-10` on the GPU and `1e-12` on the
CPU):

| | trace | vs PySCF's trace | largest element vs PySCF |
|---|---|---|---|
| GPU4PySCF, run 1 | 9.18892680 | +2.6e-4 | 1.2e-4 |
| GPU4PySCF, run 2 | 9.18892788 | +2.6e-4 | 1.2e-4 |
| repaired | 9.18868291 | +1.5e-5 | 5.1e-5 |
| PySCF, CPU | 9.18866765 | | |
| *(control)* PBE, GPU4PySCF against PySCF | 9.17324189 / 9.17325179 | −1.0e-5 | 1.7e-5 |

The repaired Hessian is 17 times closer in the trace and 2.3 times in the
largest element, and what is left is the size of the control: a GGA Hessian,
which never went near the in-place line, differs from PySCF's by the same
1e-5.  That residue is GPU4PySCF against PySCF in general, not meta-GGA and
not this bug: GPU4PySCF solves the CPHF's response on its own coarser grid
(`dft/rks.py`'s `cphf_grids`, (50, 194) points with SG-1 pruning), where
PySCF uses the SCF's.

At the loose default `conv_tol_cpscf = 1e-6` the error is larger — GPU4PySCF's
DF trace is 9.18680 against the repaired 9.18716, a shift of **3.6e-4**, more
than ten times the 3e-5 spread that made anyone look.

---

## 3. The repair

`electrolites/fastxchess.py`:

```python
_contract_orig = hess_rks.contract

def _alias_safe_contract(pattern, a, b, alpha=1.0, beta=0.0, out=None):
    if out is None or not (cupy.may_share_memory(out, a)
                           or cupy.may_share_memory(out, b)):
        return _contract_orig(pattern, a, b, alpha, beta, out=out)
    tmp = _contract_orig(pattern, a, b, alpha, 0.0)
    ...                          # out = tmp, or out = beta*out + tmp
    return out

hess_rks.contract = _alias_safe_contract
```

`hessian/rks.py` imported `contract` by name, so replacing the module global
reaches every call in that file and nothing outside it.
`cupy.may_share_memory` compares address ranges — it costs nothing, and it can
only err towards a copy that was not needed, never towards skipping one that
was.  The temporary is the size of the operand it replaces (`3 x 5 x
blksize` doubles in `_get_vxc_deriv1`); `tests/test_mgga_inplace.py`, an
ethanol TPSS Hessian three times over, goes through it 1609 times; and on
the 42-atom DF-TPSS Hessian of §4 `_get_vxc_deriv1` takes 15.03 s unpatched
and 15.18 s and 14.98 s repaired.

Three decisions:

* **In `fastxchess`, and on whenever it is loaded.**  It is the module that
  already owns `hessian/rks.py`'s XC derivatives, and the fall-through it
  makes for meta-GGAs is where the bug was reached.  The repair is not behind
  `FASTXCHESS_OFF`, because turning the module's kernels off is not a reason
  to compute a wrong answer; `FASTXCHESS_INPLACE=1` puts the in-place call
  back for anyone who wants to see it.
* **Not a copy of `_get_vxc_deriv1_task`.**  A patched copy with the one line
  changed would fix one of the three sites and freeze 200 lines of upstream
  meta-GGA code into this repository.  The indirection fixes all three and
  whatever else upstream writes the same way in that file.
* **W16's `fastxcfxc` binding is unchanged.**  It never had the line; what
  changes is that `FASTXCFXC_MO=0`, or a density `fastxcfxc` declines, now
  gets a correct answer from upstream too.  `tests/test_mgga_inplace.py`
  checks upstream's meta-GGA `nr_rks_fxc_mo` against the AO-basis path at
  1e-10 — the comparison `tests/test_dfhessip.py`, which does not load
  `fastxchess`, can still only print as `info`.

---

## 4. The success line

Two runs of the same meta-GGA Hessian agree to 1e-9, as a GGA one does:

| system | configuration | traces |
|---|---|---|
| glycine, DF-TPSS | unpatched, 4 processes | 9.1867978133, 9.1867568088, 9.1867977303, 9.1867966716 |
| | `--patch fastxchess`, 4 processes | 9.1871553665 (all four) |
| | `--patch all`, 4 processes | 9.1871553664, 9.1871553665, 9.1871553665, 9.1871553665 |
| penicillin V (42 atoms), DF-TPSS | unpatched, 2 processes | 40.9141875571, 40.9143467896 |
| | `--patch all`, 2 processes | **40.9142598198, 40.9142598199** |
| | `--patch all`, `FASTXCHESS_INPLACE=1 FASTXCFXC_MO=0` | 40.9142596339, 40.9142948592 |
| glycine, direct TPSS | unpatched, 2 processes | 9.187710852, 9.187716518 |
| | repaired, 2 processes | 9.188073509142, 9.188073509158 |
| | unpatched, `conv_tol_cpscf = 1e-10`, 2 processes | 9.188926800, 9.188927880 |
| ethanol, DF-TPSS, 3 Hessians in one process | repaired | 7.0335441208 ×3, spread 9.6e-12 |

The penicillin run is the 42-atom case the GGA claim in
`docs/README_fastdfhessip.md` §5 was made on.  Its last row is the control:
with only the in-place calls put back, the patched run is not reproducible
either.

---

## 5. What this does not cover, and what it found

* **Unrestricted**: nothing to fix — `hessian/uks.py` has no aliased call,
  and §1 checks the claim rather than asserting it.
* **The meta-GGA XC derivatives are now the largest thing in a meta-GGA
  Hessian.**  On the 42-atom DF-TPSS Hessian, patched, `_get_vxc_deriv1` is
  **15.0 s (37 %)** and `_get_exc_deriv2` 14.7 s of a 40.7 s run — both still
  GPU4PySCF's code, because `fastxchess` (B5) covers GGAs only.  Its atom
  screen and masked GEMMs are not GGA-specific; the meta-GGA branch adds
  the `tau` terms (`for j in range(1, 4)` in `_get_vxc_deriv1_task`).  That
  is `DF_WORK_LIST.md` **W28**, not this item.

---

## 6. Tools

```bash
python benchmarks/mgga_hess_repro.py --xyz benchmarks/molecules/glycine.xyz \
    --basis def2-svp --xc tpss --reps 3 [--patch all] [--direct] \
    [--save scf.npz | --load scf.npz] [--stages _get_vxc_deriv1,...]
python tests/test_mgga_inplace.py [--no-hessian]
python benchmarks/df_hess_anatomy.py --xyz ... --xc tpss --rounds 1 --patch all
FASTXCHESS_INPLACE=1 python ...                  # the in-place calls back
```
