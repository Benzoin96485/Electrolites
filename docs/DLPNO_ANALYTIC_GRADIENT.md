# Analytic derivatives of the implemented DLPNO energy

This document derives the derivative of `electrolites/dlpno/ref_rhf.py`
and the equivalent closed-shell GPU energy driver.  The baseline is commit
`5e7fe7f` on `dlpno-gpu`.  The initial response implementation is a direct
coupled-response reference, with analytic integral derivatives.  It remains
an accuracy oracle.  The subsequent coordinate-independent adjoint and
local integral implementation is derived in
[DLPNO_LOCAL_ADJOINT.md](DLPNO_LOCAL_ADJOINT.md), including its scaling
conditions and the global costs that remain.

## Sources and what can be taken from each

The six supplied papers were read from `papers/DLPNO-CCSD(T)-gradient`.

1. Pinski and Neese, JCP **150**, 164102 (2019),
   [10.1063/1.5086544](https://doi.org/10.1063/1.5086544): the complete
   PAO-based DLPNO-MP2 Lagrangian, prescreening and PNO corrections,
   localization response, and treatment of continuous localization
   degeneracies.  Equations 21--24, 58--64 and 80--85 are particularly
   relevant.  This is the closest orbital/subspace construction to this code.
2. Rauhut and Werner, PCCP **3**, 4853 (2001),
   [10.1039/B105126C](https://doi.org/10.1039/B105126C): local CC gradients,
   PAO metric and domain response, and coupled strong-CC/weak-MP2 pairs.
3. Ledermüller and Schütz, JCP **140**, 164113 (2014),
   [10.1063/1.4872169](https://doi.org/10.1063/1.4872169): local CC2
   ground/excited-state response.  The Laplace CC2 equations are not the
   CCSD equations of the present implementation.
4. Hayashi, Saitow and Yanai, JCP **163**, 234114 (2025),
   [10.1063/5.0302950](https://doi.org/10.1063/5.0302950): PNO-MP2 gradients
   with localized virtual MOs.  Its LVMO localization constraints do not
   replace the nonorthogonal PAO metric response needed here.
5. Hayashi et al., CC2 gradient preprint (2026),
   [10.26434/chemrxiv.15005833/v1](https://doi.org/10.26434/chemrxiv.15005833/v1):
   the extension of the PNO-MP2 Lagrangian to CC2, including amplitude
   multipliers and retained/discarded PNO mixing.  It provides an organizing
   principle, not a ready-made CCSD(T) nuclear-gradient implementation.
6. Toth, Jiang and Schaefer, JCTC **22**, 7667 (2026),
   [10.1021/acs.jctc.6c00758](https://doi.org/10.1021/acs.jctc.6c00758):
   DLPNO-CCSD lambda equations in the same T1-transformed formalism used
   by the energy code.  Its asymmetric `(T)_Lambda` energy is different
   from the implemented T0/iterative-T energies.  Solving its fixed-PNO
   lambda equations alone does not supply the nuclear orbital and PNO
   responses.

The extension below is a derivation for this repository's energy, rather
than a claim that one of these papers already supplies all of its terms.

## 1. Differentiate the energy that is actually computed

For fixed screening decisions the code computes

\[
 E=E_{\rm HF}+E_{\rm CC,strong}+E_{\rm weak}
   +\Delta E_{\rm crude}+\Delta E_{\rm dip}
   +\Delta E_{\rm PNO}^{(1)}+\Delta E_{\rm PNO}^{(2)}+E_{(T)}.
\]

Both PNO corrections are differences of semicanonical/projected MP2
energies.  The first uses initial semicanonical amplitudes; the second
uses the converged coupled LMP2 amplitudes.  The weak-pair energy includes
projected singles products after CCSD.  It is not simply a constant
MP2 correction to a canonical CCSD gradient.

Discrete AO domains, pair classes, Cholesky pivots, PNO/TNO ranks and
screening masks are piecewise constant.  Derivatives are evaluated on
one such branch.  A finite difference crossing a selection boundary is
not an accuracy test of this local derivative; both displaced selections
must be recorded.  Holding a mask fixed does **not** hold the orbitals,
PNO projectors, overlaps or continuous correction energies fixed.

## 2. Complete constrained derivative and its direct-response form

Collect continuous variables in dependency order:

\[
 y=(C_{\rm occ},U_{\rm loc},P,T_{\rm sc},U_{\rm PNO1},T_{\rm MP2},
        U_{\rm PNO2},t_1,t_2,U_{\rm TNO},t_3).
\]

SCF stationarity, orthonormality, localization, semicanonical equations,
PNO density block diagonality, MP2/CCSD residuals and the triples
equations define constraints \(F(y,R)=0\).  Algebraically defined
variables can be eliminated by substitution.  The augmented Lagrangian is

\[
 \mathcal L=E(y,R)+z^T F(y,R),\qquad
 F_y^T z=-E_y^T,\qquad E^x=E_x+z^T F_x.
\]

Equivalently, solve the direct coupled responses

\[
 F_y y^x=-F_x,\qquad E^x=E_x+E_y y^x.
\]

These are the same derivative: substitution eliminates \(y^x\) and
recovers the Lagrangian expression.  The direct-response oracle implements
this second form, separately for each nuclear direction.  This avoids writing
an independent copy of the long T1-dressed residual equations.  The
nonlinear amplitudes are first converged, then their **linear response**
is solved; no derivative through DIIS history is taken.

At fixed orbitals and PNOs, the CC lambda equation is

\[
 J_{\rm CC}^T\lambda=-\left(\partial E/\partial t\right)^T,
 \quad J_{\rm CC}=\partial R_{\rm CC}/\partial t.
\]

For CCSD(T), the source on the right includes the triples contribution,
including the CC-amplitude dependence of TNOs.  Reusing a CCSD-only
lambda source would omit amplitude relaxation in the triples gradient.

## 3. SCF, frozen core, Boys and PAO responses

Write \(C^x=C U^x\), \(\bar S^x=C^T S^x C\).  Differentiating
\(FC=SC\epsilon\) gives, for occupied \(i\) and virtual \(a\),

\[
 (\epsilon_a-\epsilon_i)U^x_{ai}
 +\{C^T G[D^x]C\}_{ai}
 =-\{C^T F^{[x]}C-\bar S^x\epsilon\}_{ai},
 \qquad U^x_{ij}=-\tfrac12\bar S^x_{ij}
\]

in the symmetric occupied gauge.  Here \(F^{[x]}\) is the explicit
Fock derivative at fixed AO density; the reference's own RHF J/K map
supplies \(G[D^x]\).  For DF-SCF its auxiliary-basis response is included.
After solving CPHF, restore the canonical occupied off-diagonal response

\[
 U^x_{pi}=\frac{(C^T F^x C)_{pi}-\epsilon_i\bar S^x_{pi}}
                    {\epsilon_i-\epsilon_p}\quad(p\ne i),
\]

where required to distinguish frozen core from active occupied orbitals.
This core/active rotation cannot be discarded as an occupied gauge freedom.

The Boys stationarity condition is
\(g_{ij}=\mathbf r_{ij}\cdot(\mathbf r_{ii}-\mathbf r_{jj})=0\).
For transported active orbitals, compute its explicit derivative and solve
\(H_{\rm loc}\kappa^x=-g^{[x]}\); add \(C_{\rm LMO}\kappa^x\)
to the transported derivative.  Exact localization zero modes use
parallel transport, with nullity and the projected-out source reported.
An arbitrary singular-mode result must not be called a unique gradient
of an orbital-dependent energy.

The actual normalized PAOs are

\[
 P_0=I-C_o C_o^T S,\quad N_{uu}=[P_0^TSP_0]_{uu}^{-1/2},\quad P=P_0N.
\]

Their derivative includes \(C_o^x\), \(S^x\) and \(N^x\).
For a pair, \(S_{pq}=X_p^T S_{\rm PAO}[D_p,D_q]X_q\), so its
derivative has both coefficient terms and the PAO metric term.  These
overlap derivatives enter every inter-pair MP2/CCSD/triples projection.

## 4. MP2 and both PNO generations

With \(K^{ij}_{ab}=(ia|jb)\) and \(\widetilde T=2T-T^T\),

\[
 E_{\rm MP2}=\sum_{ij}\langle K^{ij},\widetilde T^{ij}\rangle,
\]
\[
 R^{ij}=K^{ij}+(\epsilon_a+\epsilon_b-f_{ii}-f_{jj})T^{ij}
 -\sum_{k\ne i}f_{ik}S_{ij,kj}T^{kj}S_{kj,ij}
 -\sum_{k\ne j}f_{kj}S_{ij,ik}T^{ik}S_{ik,ij}=0.
\]

The sums use exactly the pair map and Fock screening of the energy code.
Differentiate this residual and solve \(J_{\rm MP2}T^x=-R^{[x]}\).
Both ordered pair multiplicities are retained in the energy; the solver
uses independent \(i\leq j\) amplitudes and symmetric diagonal pairs.

Each PNO construction uses

\[
 D^{ij}=\frac{\widetilde T T^T+\widetilde T^T T}{1+\delta_{ij}},
 \quad D U=U n.
\]

The four product-rule terms in \(D^x\) are included.  Away from
degeneracies,

\[
 (U^T U^x)_{ba}=\frac{(U^T D^x U)_{ba}}{n_a-n_b}.
\]

In particular, \(a\) retained and \(b\) discarded rotations are
essential.  Rotations within a fully retained degenerate block are gauge
choices; a retained/discarded degeneracy requires a subspace treatment or
an explicit failure.  Semicanonicalization is differentiated as a second
eigenproblem.  The same construction applies to TNOs, whose density here
is made from the final CC doubles, as implemented by `tno_transform`.

## 5. CCSD response

The repository's T1-dressed residuals are `_T_n`, `_t1_ints`,
`_t1_fock`, `_R1`, `_R2` and their intermediates in `ref_rhf.py`.
For converged strong-pair amplitudes,

\[
 \begin{pmatrix}R_1\\R_2\end{pmatrix}^{[x]}
 +J_{\rm CC}\begin{pmatrix}t_1^x\\t_{2,s}^x\end{pmatrix}=0.
\]

Weak doubles remain projected MP2 amplitudes and carry their upstream
response.  They are not zero-residual independent CC unknowns.  The
energy, including weak pairs, is

\[
 E_{\rm pair}=\sum_{ij}\langle
 T^{ij}+(S_{ij,ii}t_1^i)(S_{ij,jj}t_1^j)^T,\ 2K^{ij}-(K^{ij})^T\rangle.
\]

`dual.py` evaluates the product rule for these same residual expressions;
GMRES solves their analytic Jacobian-vector system.  Independent primal
and tangent residuals are recorded.  This is analytic forward
differentiation of tensor equations, not a small energy displacement.

## 6. T0 and iterative T

For each retained triplet the code forms \(W\), \(V=W+V_{\rm singles}\),
the diagonal denominator \(\Delta\), and a fixed spin/permutation
contraction \(\mathcal C\).  Its T0 equations are

\[
 t_3=-W/\Delta,\quad E_{T0}=\mathcal C[V,t_3],\quad
 t_3^x=-(W^x+\Delta^x t_3)/\Delta.
\]

Thus \(E_{T0}^x=\mathcal C[V^x,t_3]+\mathcal C[V,t_3^x]\),
with derivatives of all projected integrals, singles, doubles and TNOs.

For iterative T, let \(A_3\) contain denominators and all screened
off-diagonal occupied-Fock/inter-triplet overlap couplings:

\[
 A_3t_3+W=0,\qquad A_3t_3^x=-(W^x+A_3^x t_3).
\]

The implemented final correction is specifically

\[
 E_{(T)}=E_{T0,\rm fine}+\Delta E_{T,\rm screened}
              +E_{T,\rm iter,scaled}-E_{T0,\rm scaled}.
\]

All four derivatives are required.  Differentiating only the final
iterative triples term would compute a different energy derivative.
Iteration skipping is a convergence acceleration, not a physical
constraint; the response reference solves the converged triples equation.

## 7. AO and auxiliary integral derivatives

For raw three-center integrals \(B\) and the auxiliary metric \(M\),

\[
 G=B^TM^{-1}B,\qquad
 G^x=(B^x)^TM^{-1}B+B^TM^{-1}B^x
                 -B^TM^{-1}M^xM^{-1}B.
\]

Orbital coefficients in the transformed B also respond.  Both AO
centers and the auxiliary center of `(mu nu|P)`, and both centers of
`(P|Q)`, are differentiated.  The metric inverse square root uses the
continuous divided-difference Frechet derivative, including repeated
eigenvalues.  Omitting the metric or auxiliary-center terms fails a
translation test.

In a production adjoint implementation the same result can be collected
as \(E_{NN}^x+D:h^x+W:S^x+Q:r^x+\Gamma:B^x+Z_M:M^x\), with
the sign of W determined by the chosen Lagrangian convention.  The
present direct implementation avoids an unverified relaxed-density
construction while establishing the complete derivative.

## Interface, validation and present scope

```python
g = dlpno_object.nuc_grad_method().kernel()       # Eh / bohr
g_gpu = dlpno_object.nuc_grad_method(backend='cupy').kernel()
g_mp2 = dlpno_object.nuc_grad_method(level='mp2').kernel()
g_ccsd = dlpno_object.nuc_grad_method(level='ccsd').kernel()
```

The energy reference must first be converged for derivatives.  For
example, after constructing a closed-shell PySCF `mf`:

```python
from electrolites.dlpno.reference_cpu import orbitals
from electrolites.dlpno.setup import count_frozen_core
from electrolites.dlpno.rhf import DLPNO

mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-10, 150
mf.kernel()
nocc = mf.mol.nelectron // 2
orb = orbitals(mf.mol, mf.get_ovlp(), mf.get_fock(), mf.mo_coeff,
               nocc, count_frozen_core(mf.mol))
dlpno_object = DLPNO(mf, triples='t1', c_lmo=orb.C_lmo,
                     E_CONVERGENCE=1e-11, R_CONVERGENCE=1e-10,
                     MAXITER=150, T_CUT_ITER=0.)
dlpno_object.kernel()
gradient = dlpno_object.nuc_grad_method(backend='cupy').kernel()
```

`orbitals` uses the usual Boys localization followed by a small stationary
Newton polish.  The actual final Boys gradient is checked: CIAH's stopping
record can refer to a preceding keyframe.  For a GPU SCF reference, use
`mf.to_cpu()` for this preparation; the DLPNO energy still uses its GPU
driver.  Use `ncore=0` and `frozen_core=False` together for all-electron
correlation.  `CPUDLPNO` in `reference_cpu.py` supplies the same energy
algebra without requiring a GPU.  The gradient driver checks the primal
residuals and energy replay; it does not silently change or reconverge the
energy supplied by the caller.

The analytic calculation has no nuclear finite-difference step parameter.
`benchmarks/dlpno_gradient_check.py` independently recomputes SCF,
localization, domains, amplitudes and energies at displaced geometries.
It records each component, step, selection stability, response residual
and the difference between the response's primal energy and the energy
driver.  Full-coordinate tests additionally report force/torque sums.
Newly optimized displaced LMOs are aligned only by permutation and sign;
their continuous response is not frozen.  Triplet-priority labels that
change without changing any retained TNO rank are reported separately
from changes of the effective energy branch.

`backend='cupy'` moves the dual tensor contractions/eigenproblems to one
CUDA device.  Analytic AO integrals, CPHF/Boys response, the replayed primal
amplitude iterations and GMRES control still run on the CPU.  This path
has synchronization and launch overhead and is a correctness reference;
no speedup or large-system GPU-memory claim is made.  In particular it
does not inherit the production energy driver's batching, multiple-GPU
partitioning or bounded integral storage.

Initial support is all-electron-basis, nonrelativistic closed-shell RHF,
Boys localization, with or without frozen occupied cores and with exact
or density-fitted SCF.  ROHF/UHF, PM localization, ECP, relativistic and
solvent responses are not covered by this first reference.  Dense AO
three-center storage and one response solve per coordinate limit system
size.  Open-shell extension and production GPU/adjoint scaling require
further work and must not be inferred from closed-shell energy support.

## Verified numerical results (2026-10-16)

The geometries, thresholds, finite-difference energies, retained spaces,
solver residuals and source hashes are in the JSON reports under
`/pscratch/sd/w/weiliang/dlpno-gradient-20261016`.
The water stress test retains as few as two of eight virtual orbitals in
some final pair spaces and uses truncated TNOs; it is not a full-space
canonical-limit-only test.  The water-dimer case has nonzero dipole,
eliminated-pair, PNO and screened-triples corrections.  A separate water
case has a nonzero weak-pair contribution.

| Case | Coordinates | Central step (bohr) | Largest error across energy components (Eh/bohr) |
|---|---:|---:|---:|
| H2O / 6-31G, truncated T1, CPU | all 9 | 1e-4 | 1.41e-9 |
| H2O / 6-31G, truncated T1, GPU energy + CuPy response | all 9 | 2e-4 | 8.53e-8 |
| NH3 / 6-31G, all-electron CCSD | one | 1e-4 | 7.98e-10 |
| NH3 / 6-31G, DF-SCF CCSD | one | 1e-4 | 2.98e-10 |
| NH3 / cc-pVDZ, truncated T0 | one | 1e-4 | 5.97e-10 |
| H2O / 6-31G, weak-pair T1 | one | 1e-4 | 2.73e-10 |
| (H2O)2 / 6-31G, screening T1 | one | 1e-4 | 9.60e-8 |

Each case was checked at multiple steps.  The largest component error
over *all* tested water GPU steps (1e-3, 5e-4, 2e-4 bohr) is 1.54e-7
Eh/bohr.  CPU/CuPy analytic water gradients agree below 1e-12 Eh/bohr;
both have translation and torque residuals below 1.5e-11 in their
respective atomic units.  The GPU energy replay differences observed in
these runs are at most 5.3e-11 Eh, which can be amplified by taking small
finite differences.
These are numerical consistency tests for the implemented approximation,
not bounds on the physical DLPNO truncation error relative to canonical CC.

Example reproduction from the checkout, with PySCF/SciPy and (for GPU)
CuPy/GPU4PySCF available:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
python benchmarks/dlpno_gradient_check.py --case water --basis 6-31g \
  --stress --triples t1 --direction all \
  --output /tmp/water-dlpno-gradient.json

DLPNO_NGPU=1 PYTHONPATH=. \
python benchmarks/dlpno_gradient_check.py --backend gpu --case water \
  --basis 6-31g --stress --direction all --steps 0.001 0.0005 0.0002 \
  --output /tmp/water-dlpno-gradient-gpu.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
python -m pytest -q tests/test_dlpno_gradient.py
```

The CUDA-dependent regression cases skip when no device is present.
On an A100 GPU, the initial gradient/closed-shell energy regression run
passed 11 tests.  Two additional CPU/GPU tests passed after correcting
the energy drivers' empty-triplet case (for example H2), whose iterative
triples correction and derivative are exactly zero.  The audit bundle
contains both test logs and the software versions.
The finite-difference program exits unsuccessfully if effective selections
change, a reported component exceeds 1e-6 Eh/bohr, an energy replay
exceeds 1e-8 Eh, or a full gradient violates translation/rotation tests.
