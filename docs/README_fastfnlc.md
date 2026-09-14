# fastfnlc — the VV10 response term, which is 80 % of a wB97M-V polarizability

Not a catalogue entry.  `instructions/06` names the VV10 double sum in the
SCF (**A5**/`fastnlc`'s target) and the VV10 gradient and Hessian (**A6**),
and its C-section entry for the polarizability (**C1**) lists `get_jk` and
`nr_rks_fxc` as the two things a CPKS solve does.  On a functional with a
non-local correlation term there is a third, and it is bigger than both:

| stage of a wB97M-V polarizability | penicillin V, 42 atoms | share |
|---|---|---|
| **total** | **154.67 s** | 100 % |
| **`hessian.rks.nr_rks_fnlc_mo`** | **124.74 s** | **80.6 %** |
| `mf.get_jk` + `mf.get_k` (range-separated) | 19.64 s | 12.7 % |
| `nr_rks_fxc` | 9.95 s | 6.4 % |

19 CPKS iterations, 6.6 s each.  `benchmarks/response_anatomy.py`, def2-SVP,
level-3 grids, one A100-PCIE-40GB.  (A second run of the same input on a
busier node gave 176.99 s with the same shares to a few tenths of a per cent
— 80.1 %, 14.1 %, 5.6 % — which is what §6 means about absolute wall clocks
on this machine.)

`nr_rks_fnlc_mo` is on the path of every **CPKS or CPHF** solve with an NLC
functional — polarizability, NMR shielding, C6 coefficients and the analytic
Hessian (`hessian/rks.py:get_veff_resp_mo`).  On a **restricted** reference it
is *not* on the TDDFT path: `tdscf/rhf.py:408 gen_response` passes
`with_nlc=not self.exclude_nlc` and `exclude_nlc` is on by default, which is
why the same molecule's TDA run (52.4 s) does not show the term at all.

On an **unrestricted** reference it is, and that asymmetry is not documented
anywhere in GPU4PySCF.  `tdscf/uhf.py:752 TDBase.gen_response` — which
`tdscf/uks.py`'s `TDA` and `TDDFT` inherit — is

```python
def gen_response(self, mo_coeff=None, mo_occ=None, hermi=0):
    return self._scf.gen_response(mo_coeff=None, mo_occ=None, hermi=hermi)
```

with **no** `with_nlc` argument and no `exclude_nlc` attribute anywhere in the
class, and `_gen_uhf_response`'s default is `with_nlc=True`.  So an open-shell
TDA or TDDFT run with wB97M-V pays this term on every Davidson iteration where
the closed-shell one pays none of it.  §4c is the unrestricted coverage this
round added, and that is the caller it matters most for — not, as the
restricted side suggests, a polarizability-only path.

---

## 0. Summary

* Two changes, no new arithmetic.  **The ground-state half of the function is
  computed once per solve instead of once per iteration** — it depends only on
  `mo_coeff` and `mo_occ`, which a CPKS solve holds fixed — and **the
  potential contraction is batched over the perturbed densities**.
* **1.44x–1.49x** on the function once the cache is warm, at `nset` = 1, 3
  and 6, and **1.44x–1.45x** on a molecule 2.7x larger — the ratio is
  size-independent, for a reason §6 gives.  The first call of a solve fills
  the cache instead of reading it and gets 1.03x, so over a 19-iteration CPKS
  solve the term comes out at **1.43x** and the **whole polarizability at
  1.39x**, 154.67 s → 111.26 s.
* The cache is **exact**: `tests/test_fnlc.py` asserts bit-identical results
  on the `return_in_mo=True` path, which uses the cache and keeps GPU4PySCF's
  own contraction.  The batched contraction reassociates one grid sum and
  agrees to 3.3e-15.
* What is left is the larger half and needs CUDA this repository does not
  have: `VXC_vv10nlc_hess_eval_f_t`, 64.7 % of a call, is an `O(ngrids²)` fp64
  double sum — 2.7e11 pairs in 4.61 s — and `fastnlc`'s three measured
  techniques apply to it verbatim.  §4.
* The repeated work here is the same *phenomenon* as catalogue entry **D8**
  and the opposite *size*: D8's repeated AO evaluation is 2.2 % of a
  polarizability on 1.8.1, this one's repeated double sum is **22 %**.

---

## 1. Where the time goes

`benchmarks/fnlc_probe.py` wraps the four `libgdft` entry points and the two
grid contractions inside one call with a synchronisation on each.  wB97M-V,
def2-SVP, level-3 grids, penicillin V, `nset = 3`:

| stage | s/call | share | depends on |
|---|---|---|---|
| `eval_rho2` — the ground-state density on the NLC grid (127 blocks) | 0.096 | 1.3 % | mo_coeff |
| `VXC_vv10nlc_hess_eval_omega_derivative` | 0.0001 | 0.0 % | mo_coeff |
| **`VXC_vv10nlc_hess_eval_UWABCE`** — `O(ngrids²)` | **1.869** | **26.2 %** | mo_coeff |
| `eval_rho` — the perturbed densities on the NLC grid (381 calls) | 0.147 | 2.1 % | dm1 |
| **`VXC_vv10nlc_hess_eval_f_t`** — `O(ngrids²)` | **4.613** | **64.7 %** | dm1 |
| the potential contraction back to the AO basis | 0.409 | 5.7 % | dm1 |
| **total** | **7.134** | 100 % | |
| ↳ **the part that only the ground state enters** | **1.965** | **27.5 %** | |

The NLC grid here is 519 680 points — GPU4PySCF puts `nlcgrids` at the same
level as the XC grid, which is the same choice `fastnlc` had to work against
in the SCF.

## 2. The cache

Everything down to `f_gamma_i` — the density and its gradient on the NLC grid,
`omega` and its two derivatives, `kappa`, and `U, W, A, B, C` from the
`UWABCE` double sum — is a function of `mo_coeff` and `mo_occ` alone.
`scf/_response_functions.py`'s `vind` closure captures one `mo_coeff` array
and hands the same object to `nr_rks_fnlc_mo` on every iteration, and
GPU4PySCF recomputes all of it every time.

The cache holds one entry and keys it on `id(mf)`, `id(mo_coeff)`,
`id(mo_occ)` and the grid size, **while keeping a reference to the two
arrays** — which is what makes identity a sound key: as long as the entry is
live those arrays cannot be freed, so their addresses cannot be reused by
something else.  A second `mo_coeff` (a new SCF, a new geometry) misses and
replaces the entry.  What it holds is about twenty `ngrids`-sized fp64
arrays plus two `3 × ngrids` ones — **83 MB** on this molecule, and it grows
with the NLC grid, so about 0.7 GB on a 284-atom cluster whose NLC grid is
3.4e6 points.  That is memory GPU4PySCF allocates transiently on every call
anyway; what changes is that it stays allocated between them.  The cache holds
one entry, so a second SCF or a new geometry releases the first.

Because the cached values are *the same values*, the result is bit-identical.
`FASTFNLC_CACHE=0` is the ablation.

## 3. The batched potential contraction

GPU4PySCF's last block loop issues five kernels per (grid block, density):

```python
V_munu = contract('ig,jg->ij', ao[0], ao[0] * fxc_rho[i_dm])
nabla_dot = contract('dg,dig->ig', fxc_gamma[i_dm], ao[1:4])
V_gamma = contract('ig,jg->ij', ao[0], nabla_dot)
V_munu += V_gamma
V_munu += V_gamma.T
```

so `V = S + G + Gᵀ` with `S = ao₀·(ao₀ f_rho)ᵀ` symmetric and
`G = ao₀·nabla_dotᵀ`.  Since `S = Sᵀ`,

```
M = (ao₀ f_rho + 2 nabla_dot) · ao₀ᵀ  =  S + 2Gᵀ   ⇒   V = (M + Mᵀ)/2
```

and the bracket is exactly `_scale_ao(ao, wv)` with
`wv = (f_rho, 2 fxc_gamma_x, 2 fxc_gamma_y, 2 fxc_gamma_z)` — a four-component
weighted-AO pass.  So the whole batch takes one weighted-AO kernel
(`fxc_scale_gga` from `kernels/fastxcfxc.cu`, the same one the XC response
kernel uses), one GEMM of `(n_dm·nao_sub, ng) × (ng, nao_sub)`, one
`transpose_sum` and one `add_sparse`: **four launches per block instead of
`5·n_dm`**.  `FASTFNLC_BATCH=0` is the ablation.

This is the only place the result is not bit-identical: the GEMM sums the grid
index in a different order.  Measured deviation over `tests/test_fnlc.py` is
3.3e-15 relative.

The `return_in_mo=True` path — what a Hessian's CPHF solve asks for — keeps
GPU4PySCF's loop, because it follows each `V_munu` with two MO
transformations that the batching would have to be rebuilt around.  It still
gets the cache, and it is the column the test uses to prove the cache exact.

## 4. What is left, and why it is not in this round

`VXC_vv10nlc_hess_eval_f_t` is 64.7 % of a call and cannot be cached: it is
the response of the double sum to the perturbed density.  It is the same shape
of kernel `fastnlc` rewrote for the ground state — an `O(ngrids²)` fp64 sum
over pairs whose kernel decays with `|rᵢ − rⱼ|` — and the three things
`docs/README_fastnlc.md` measured there transfer directly:

| `fastnlc`'s change | what it bought on the SCF's double sum |
|---|---|
| one MUFU-seeded reciprocal instead of `div.rn.f64` | 1.22x |
| single precision beyond 10 Bohr | 2.82x (cumulative) |
| an error-budgeted grid trim | 3.92x (cumulative) |

The obstacle is not the technique, it is the source: `fastnlc` could rewrite
`dft/numint.py:_vv10nlc` because its formulas are in Python in the installed
package, while `VXC_vv10nlc_hess_eval_f_t` exists only as compiled code in
`libgdft.so`.  Reproducing it needs GPU4PySCF's CUDA source (or a
re-derivation of equations (20)–(24) of Liang, Feng, Liu and Head-Gordon,
*J. Chem. Phys.* **158**, 204109 (2023), which the function's own docstring
cites), and it needs the `.so` as the oracle either way.  That is a round of
its own; if it lands at `fastnlc`'s 3.28x, a wB97M-V polarizability goes from
the 131 s this module leaves to about 70 s.

One thing that would need care in that round: an error-budgeted grid trim is
*not* exact here in the way it is in the SCF.  In the SCF the trim is inside
the build that defines the energy, so it is a consistent (slightly different)
functional; on the response path the ground-state potential was built on the
untrimmed grid, so trimming the response introduces an inconsistency whose
size has to be measured on the property, not on the double sum.  §4d does
that measurement for the partial trim it adds -- the whole returned response
operator against GPU4PySCF's untrimmed one, which is where that
inconsistency lands -- and gets 4.0e-13 relative at a 1e-12 budget and
1.3e-11 at 1e-10.

## 4b. The other caller: a wB97M-V analytic Hessian

`hessian/rks.py:get_veff_resp_mo` adds the same term to every CPHF
matrix–vector product, with `nset = 3·natm`.  `benchmarks/hessian_anatomy.py
--cphf-only`, alanine dipeptide (22 atoms, `nao = 200`), wB97M-V/def2-SVP,
level-3 grids, stock GPU4PySCF:

| stage of the CPHF solve | s | share |
|---|---|---|
| **total** | **135.56** | 100 % |
| **`nr_rks_fnlc_mo`** | **112.88** | **83.3 %** |
| `jk_mo` (`get_j` + `get_k`, `n_dm` up to 66) | 16.19 | 11.9 % |
| `nr_rks_fxc_mo` | 5.90 | 4.3 % |

(plus `make_h1` ahead of the solve, which is 39.07 s of which
`_get_vnlc_deriv1` is 31.31 s.)

So on an NLC
functional the VV10 terms dominate a Hessian as thoroughly as they dominate a
polarizability — which `docs/HESSIAN_1.8.1.md`, a B3LYP profile, cannot see,
and neither can catalogue §2.3.

**This module helps a Hessian much less than a polarizability, and the reason
is structural: what the cache removes is a per-*call* cost, while what is left
scales with the number of perturbed densities.**  Measured on the same
molecule and the same `return_in_mo=True` path
(`fnlc_bench.py --in-mo`, both implementations in one process):

| `nset` | GPU4PySCF | + fastfnlc | | rel. err |
|---|---|---|---|---|
| 3 | 1.903 s | **1.394 s** | **1.36x** | 0 |
| 15 | 5.197 s | **4.693 s** | **1.11x** | 0 |
| 30 | 6.603 s | **6.089 s** | **1.08x** | 0 |

so at a Hessian's `nset = 3·natm` the cache is a couple of per cent of the
call, and the whole CPHF solve above comes out at 1.03x — inside the
run-to-run band of a 130 s measurement, i.e. no measurable gain.  (The
`rel. err` column is exactly zero at every `nset`: this path keeps
GPU4PySCF's contraction, so the cache is all that is being tested and the
cache is exact.)

**The value of this module is inversely proportional to `nset`,** which is the
opposite of `fastxcfxc`'s scaling: that one grows with `nset` and shrinks with
the molecule, this one is flat in the molecule (§6) and shrinks with `nset`.
Between them a polarizability, an NMR shielding and a C6 coefficient are well
covered — all of them have `nset` of 3 to 9 — and a Hessian is not.

## 4c. The unrestricted reference, and the caller it brings with it

`hessian.uks.nr_uks_fnlc_mo` sums the two perturbed densities and forwards to
the restricted function, because `V_munu` in this function is spin-free —
VV10 is a functional of the total density.  So the unrestricted case needs
two changes and no new arithmetic:

* `_ground_state` sums `eval_rho2` over the two spin channels for the density
  the `O(ngrids²)` `UWABCE` sum needs, and everything downstream of it —
  `omega`, `kappa`, `U`, `W`, `A`, `B`, `C`, `f_gamma` — is unchanged;
* an MO-basis answer (`return_in_mo=True`, the Hessian's CPHF) is projected
  onto two MO bases instead of one and returned as GPU4PySCF's two-element
  tuple.  `return_in_mo=False`, which is what the response function and
  therefore TDDFT asks for, needs no spin index at all, so §3's batched
  contraction covers it unchanged.

Reaching it needs a **second patch point**, and this is where it hid:
`hessian/uks.py:28` is `from gpu4pyscf.hessian.rks import ...,
nr_rks_fnlc_mo`, bound at import, and `nr_uks_fnlc_mo` calls that
module-level name rather than `hessian.rks`'s attribute.  Patching only
`hessian.rks` therefore left **every** open-shell caller on GPU4PySCF's own
code, with the module loaded, the answer right and nothing saying so.
`apply_patch` sets `hessian.uks.nr_rks_fnlc_mo` as well;
`tests/test_patch_aliases.py` is the check that keeps this visible, and
`stats()` — added for the same reason — counts what the dispatch answered
against what it handed back.

Why it matters more than the restricted case: see the note in §0's
neighbourhood above about `tdscf/uhf.py`.  An open-shell TDA or TDDFT run
with wB97M-V pays this term on every Davidson iteration; the closed-shell one
pays none of it.  `docs/VV10_OPENSHELL.md` has the round.

## 4d. The grid trim, and why `FASTFNLC_BUDGET` is 0

`fastnlc`'s error-budgeted grid trim (`docs/README_fastnlc.md` §3) applies to
both `O(ngrids²)` sums here verbatim — every term of both carries a factor
`rho_j w_j`, so the budget bounds what dropping a set of points can do — and
the saving is quadratic in the point count.  It is available as
`FASTFNLC_BUDGET` and it is **off by default**.

Measured (`benchmarks/fnlc_bench.py --budget 0,1e-14,1e-12,1e-10`, one SCF per
molecule, `nset = 3`, cached call): **8 %** on the 21-atom Ni dithiolene at
budget 1e-10 (1.49x → 1.61x, 1.3e-11 relative) and **6 %** on penicillin V at
1e-12 (1.44x → 1.53x, 4.0e-13 relative).  That is a tenth of what the same
criterion is worth in `fastnlc`, and the `budget = 0` row says why: **11–14 %
of the grid is already gone** before any budget is applied, because this
function filters every point on `rho >= 1e-10 and |w| > 1e-14`.  §3 of the
`fastnlc` document measured its 14 % against an SCF path where GPU4PySCF's own
cut removes 0.8 % of the points, so there the budget had 14 % to take and here
it has 2–3 %.

Off by default because what it perturbs is a *response operator* rather than
an energy, because the guarantee in §5 below is **bit-identity** against
GPU4PySCF on the `return_in_mo` path and a trim ends that, and because 6–8 %
is not what that is worth trading for on two systems' evidence.
`docs/VV10_OPENSHELL.md` §4d has the full table.

## 5. Correctness

`tests/test_fnlc.py`, both implementations in one process on the same
converged SCF, eight cases: three NLC functionals (`wb97m-v`, `wb97x-v`,
`b97m-v`), three basis sets, grid levels 1 to 3, a third-row element, `nset`
from 1 to 6, and both `return_in_mo` values.  Every case calls the patched
function **twice** and checks the second call as well, because the second is
the one that reads the cache.

| column | what it checks | result |
|---|---|---|
| `mo` | `return_in_mo=True`: the cache with GPU4PySCF's own contraction | **bit-identical** in all cases |
| `ao` | `return_in_mo=False`: cache plus batched contraction | ≤ 3.3e-15 relative |
| `2nd call` | the same, on the cached call | same |
| `polariz.` | the polarizability tensor on the same SCF, patched against unpatched | ≤ 1.3e-15 relative |

`tests/test_ufnlc.py` is the unrestricted suite, 7 cases entering through
`hessian.uks.nr_uks_fnlc_mo` so that the binding of §4c is on the tested path:
O2, CH3, NO, CuCl2 and the 23-atom Pt complex (an ECP), over the same three
functionals, at def2-SVP, def2-TZVPP (which reaches g) and 6-31G\*.  `mo` is
**bit-identical in all 7, on both calls**, and comes back as GPU4PySCF's
two-element tuple; `ao` is ≤ **4.7e-15** relative.  It then runs a whole
**UKS TDA** — the caller §4c says the restricted side does not have — and the
three roots of O2/def2-SVP/wB97M-V agree to **1.4e-13 Eh**.

## 6. What it is worth

`benchmarks/fnlc_bench.py`, penicillin V/def2-SVP/wB97M-V, level-3 grids,
`min` of two, both implementations in one process:

| `nset` | GPU4PySCF | + fastfnlc | |
|---|---|---|---|
| 1 | 6.330 s | **4.326 s** | **1.46x** |
| 3 | 7.208 s | **4.847 s** | **1.49x** |
| 6 | 8.346 s | **5.795 s** | **1.44x** |

The ratio is flat in `nset` because both halves of the change scale with it:
the cached part is per call, and the batched contraction is per (block,
density).

The **first** call of a solve is a separate number and has to be given
separately, because it fills the cache rather than reading it.  At
`nset = 3`, with the cache cleared immediately before:

| | s/call | |
|---|---|---|
| GPU4PySCF | 6.456 | 1.00x |
| fastfnlc, cold cache | 6.242 | **1.03x** |
| fastfnlc, warm cache | 4.466 | **1.45x** |

So the first iteration of a CPKS solve gets only the batched contraction, and
the other eighteen get both: `6.24 + 18 × 4.47 = 86.7 s` against
`19 × 6.46 = 122.7 s`, i.e. **1.42x** over the solve.

**The unrestricted reference gives the same ratio**, which is the useful
statement about §4c: the change is in how often the ground-state half runs and
how the contraction is associated, and neither knows about spin.
`benchmarks/fnlc_bench.py --spin`, one A100-SXM4-80GB (so not comparable in
seconds with the tables above and below, which are 40 GB parts — the
penicillin V row is repeated on the same hardware for that reason):

| system | reference | `nset` | GPU4PySCF | warm | |
|---|---|---|---|---|---|
| CuCl2, 3 atoms, 67 AO | UKS doublet | 1 | 0.1539 s | **0.0955 s** | **1.61x** |
| | | 3 | 0.1771 s | **0.1088 s** | **1.63x** |
| Ni dithiolene, 21 atoms, 327 AO | UKS doublet | 1 | 2.4274 s | **1.6475 s** | **1.47x** |
| | | 3 | 2.6434 s | **1.7765 s** | **1.49x** |
| penicillin V, 42 atoms, 430 AO | RKS | 1 | 5.6587 s | **3.9564 s** | **1.43x** |
| | | 3 | 6.0844 s | **4.2361 s** | **1.44x** |

The unrestricted rows are on a path that was **1.00x by construction** until
`docs/VV10_OPENSHELL.md`, and the restricted row is the regression check: it
is the same 1.43x–1.44x this section already recorded on other hardware.

**The second size gives the same ratio, and that is worth remarking on.**
Taxol (113 atoms, `nao = 1123`, an NLC grid of 1 374 976 points — 2.6x
penicillin V's, so 7x the pairs):

| `nset` | GPU4PySCF | cold | warm | cold | warm |
|---|---|---|---|---|---|
| 1 | 36.929 s | 36.754 s | **25.460 s** | 1.00x | **1.45x** |
| 3 | 39.677 s | 38.919 s | **27.578 s** | 1.02x | **1.44x** |

against 1.46x and 1.49x on penicillin V.  Catalogue §3.4 asks for a second,
differently-sized system and for the result to count only if the ratios agree;
here they agree to two per cent, and they should, because what the cache
removes is a fixed fraction of the call — the ratio of the two `O(ngrids²)`
double sums to each other — and that ratio is a property of the algorithm, not
of the molecule.  This is the opposite of `fastxcfxc`, whose gain halves
between the same two molecules (`docs/README_fastxcfxc.md` §6) because what it
removes is launch overhead against arithmetic that grows.

(The absolute numbers are a warning of their own: **37 s per call** at 113
atoms, so a taxol wB97M-V polarizability is about twenty minutes of VV10
response alone.  That is what §4's remaining kernel is worth attacking.)

End to end, `benchmarks/response_anatomy.py`, penicillin V/def2-SVP/wB97M-V,
level-3 grids, 19 CPKS iterations:

| | GPU4PySCF 1.8.1 | + Electrolites | |
|---|---|---|---|
| **wB97M-V polarizability, whole run** | **154.67 s** | **111.26 s** | **1.39x** |
| ↳ `nr_rks_fnlc_mo` (this module) | 124.74 s (80.6 %) | **87.21 s** (78.4 %) | **1.43x** |
| ↳ `nr_rks_fxc_st` (`fastxcfxc`) | 9.95 s (6.4 %) | **3.67 s** (3.3 %) | **2.72x** |
| ↳ `mf.get_jk` + the long-range `get_k` | 19.64 s (12.7 %) | 20.07 s (18.0 %) | untouched |

Stock and patched are run back-to-back in one script, because the node is
shared and a whole-run wall clock moves with its load — see
`docs/RESPONSE_1.8.1.md` §4b, which also explains why the J and K row is the
best available check that nothing else changed.

The 1.43x matches the 1.42x the per-call table predicts for a 19-iteration
solve, which is the arithmetic working out: one cold call and eighteen warm
ones.  The SCF ahead of the solve went 32.29 s → 18.78 s in the same pair,
which is `fastnlc` and the other SCF modules rather than this round.

## Files

| file | what |
|---|---|
| `electrolites/fastfnlc.py` | the module: the cache, the batched contraction, the fall-through conditions |
| `benchmarks/fnlc_probe.py` | the stage split of §1 |
| `benchmarks/fnlc_bench.py` | the A/B of §6, cold and cached |
| `tests/test_fnlc.py` | 8 cases against GPU4PySCF, plus the polarizability tensor |
| `tests/test_ufnlc.py` | 7 unrestricted cases through `nr_uks_fnlc_mo`, plus a UKS TDA run |

## Environment switches

| variable | effect |
|---|---|
| `FASTFNLC_CACHE` | 0 recomputes the ground-state half every call (the ablation for §2) |
| `FASTFNLC_BATCH` | 0 puts GPU4PySCF's per-density potential loop back (the ablation for §3) |
| `FASTFNLC_BUDGET` | `fastnlc`'s error-budgeted grid trim, applied to this term's two `O(ngrids²)` sums. **0 (off) by default**; §4d says what it is worth and why the default is off |
| `FASTFNLC_WX`, `FASTFNLC_TSY`, `FASTFNLC_UPB` | thread-block shape of the weighted-AO kernel |
