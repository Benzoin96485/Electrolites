# Local DLPNO-CCSD(T) adjoints and scaling requirements

The objective of this extension is a nuclear gradient with the same
dominant local-correlation scaling as the single-point energy.  The
previous coordinate-response implementation remains an accuracy oracle.
Removing its factor of `3*natm` is necessary but does not alone establish
linear scaling.  This document records the derivation and the remaining
implementation/validation requirements; it is not a completion claim.

The supplied Pinski--Neese DLPNO-MP2 gradient paper (J. Chem. Phys. 150,
164102, 2019, DOI 10.1063/1.5086544), printed page 12, distinguishes
several costs. Bounded pair spaces and a linear number of screened pairs
give O(N) pair-gradient loops and RI integral derivatives. Its complete
prescreen correction and integrals with an unrestricted MO index are
formally O(N^2); Boys localization/response and some global matrix
products remain O(N^3). CPHF and the relaxed Fock derivative inherit the
reference Fock construction's scaling. Thus the paper does not establish
O(N) for the entire gradient. We separately report these stages rather
than infer total scaling from the local pair loop. The user-confirmed
acceptance scope follows this separation: local correlation work must be
O(N), while Boys, CPHF and global orbital backtransforms retain the
scaling of the corresponding reference-state steps. Making every global
response stage O(N) is not required. Empirical finite-size slopes and the
conditional asymptotic operation bounds are reported separately.

## Current hundred-atom validation (2026-10-26)

The [GPU validation report](/pscratch/sd/w/weiliang/dlpno-gpu-gradient-20261018/GPU_GRADIENT_REVIEW.html)
links the accepted reviews, errors, timings and exact source versions.
Source170 is now installed in the gradient worktree; all 53 numerical
modules match the fully validated snapshot. Dated entries below preserve
earlier campaign states; this section describes the current installation.

The fixed asymmetric C34H70/def2-SVP case has 104 atoms and 826 AOs, using
closed-shell RHF/Boys, NormalPNO, `S_CUT=1e-8` and LMO/PAO coefficient
cutoffs of `1e-6`.

| Method | Complete gradient | Gradient seconds | Largest independent FD error (Eh/bohr) |
| --- | --- | ---: | ---: |
| CCSD, source120 | 312 components; accepted | 5928.49 | 7.9467e-8 |
| CCSD(T0), source124 | 312 components; accepted | 12089.81 | 5.8009e-8 |
| CCSD(T1), source165 | 312 components; accepted reference | 19036.96 | 3.3503e-8 |
| CCSD(T1), source170 | 312 components; accepted and installed | 19399.03 | 3.0203e-8 |

FD checks sample two Cartesian components at two step sizes, using eight
independently optimized displaced SCF/Boys solutions and four central
checks. They do not check all 312 components by finite differences.
Gradient times exclude SCF/Boys reference preparation and the separately
recorded correlated-energy step (698.60 seconds for source170). The full
source170 run reused only its verified same-source energy checkpoint and
rebuilt the complete derivative graph. Response-only restart is outside
this acceptance and was not used.

The accepted reviews are `ccsd-gpu-union-full-validation-v138.json`,
`keyed-topology-full104-review-v150.json`,
`full104-t1-validation-v179.json`, and
`full104-ri-identity-comparison-v182.json`, under
`/pscratch/sd/w/weiliang/dlpno-gpu-gradient-20261018/`.
Source170 job 60291702 completed successfully on October 26; its dependent
scientific audit, job 60292192, completed successfully afterward. All 312
source170 components agree with source165 within 1.8876e-9 Eh/bohr; all
energy components agree within 9.7145e-16 Eh. Displaced local selections
were checked by physical triplet identity, without a full-virtual-span
exception. The current case contains 2,837 pairs and 6,146 triplets.

The 353,233,905-variable T1 adjoint keeps its Krylov vectors on the GPU and
uses dedicated persistent allocation pools. In the accepted source170
run, this solve took 37.14 seconds with maximum true residual 1.9983e-13.
The CCSD transpose-operator solve took 83.26 seconds. Those times are
included in the complete gradient's 14,298.57-second amplitude reverse
stage; building the energy derivative graph took another 4,918.11 seconds.
The host derivative graph and reverse sweep remain the dominant costs.

The complete source170 gradient used one A100 40 GB, with observed host
peak 209.09 GiB and sampled GPU peak 33.88 GiB. GPU sampling can miss a
short-lived peak. The batch reserved a full four-GPU node for host memory;
numerical kernels used one selected GPU. Completion of this particular
104-atom case does not establish the same resource bound for every
100-atom molecule, a larger basis, or tighter thresholds.

The FD records retain both original GPU-SCF energies and independently
reevaluated HF energies at the saved converged densities. Source170 has
maximum FD error 3.5714e-6 Eh/bohr with the former and 3.0203e-8 Eh/bohr
with the latter; the source165 values were 3.5351e-6 and 3.3503e-8.
Orbital, local-space and both energy records remain available in
`full104-t1-fd-comparison-v173.json` and its source165 counterpart v168.
The source170 small-system validation passed 137 tests and 12 independent
nuclear FD rows. The installed files are byte-identical to that validated
source; integration is recorded in
`source170-workspace-promotion-result-v185.json`.

Source170 additionally bypasses exact full-envelope RI gathers and their
reverse scatters. Its same-GPU 14-atom ABBA comparison reduced mean T1
gradient time from 264.111 to 235.040 seconds (11.01%), with maximum
gradient change 2.0722e-9 Eh/bohr. The full104 diagnostics did not record
activation of this exact-envelope shortcut. Source170's 323.32-minute
full104 run and source165's 317.28-minute run used different nodes and
are descriptive timings, not a controlled RI speedup measurement.
The source170 first attempt's five-hour timeout remains preserved;
the successful complete rerun had an eight-hour allocation limit.

Local-correlation O(N) remains conditional on bounded domains, neighbor
counts and iterations. Global Boys/CPHF and orbital transformations keep
their reference-state scaling. The finite104 result establishes practical
feasibility for the recorded setup, not an asymptotic proof for the whole
gradient. In this case some AO envelopes span all 826 AOs, so the measured
run alone also does not demonstrate bounded local integral envelopes.

## Dependency-ordered Lagrangian

Let `m` be the converged first-space coupled MP2 amplitudes, `t` the
strong-pair CC singles/doubles, and `s` the iterative triples amplitudes.
The normalized PAOs, both PNO transformations, weak amplitudes, TNOs,
local integral transformations and all continuous energy corrections
are explicit functions of these variables and the reference orbitals.
Eliminate these explicit variables by substitution, without freezing
their derivatives.  The remaining correlation Lagrangian is

\[
 \mathcal L_c=E_c(m,t,s;q)
       +\lambda_m^T R_m(m;q)
       +\lambda_t^T R_t(t;m,q)
       +\lambda_s^T R_s(s;t,m,q),
\]

where `q` denotes orbital and primitive-integral inputs.  Solve backwards
through the dependencies:

\[
 R_{s,s}^T\lambda_s=-E_{c,s},
\]
\[
 R_{t,t}^T\lambda_t=-E_{c,t}-R_{s,t}^T\lambda_s,
\]
\[
 R_{m,m}^T\lambda_m=-E_{c,m}-R_{t,m}^T\lambda_t
                              -R_{s,m}^T\lambda_s.
\]

The derivatives in these expressions include all eliminated PNO/TNO
maps.  For T0 there is no iterative triples solve: reverse the explicit
`s=-W/Delta` expression.  For iterative T the scalar objective is the
existing fine-T0 plus screened correction plus iterative-scaled-T minus
scaled-T0 energy.  A CCSD-only lambda source is insufficient for this
objective.

`reverse.py` supplies exact vector-Jacobian products of the same local
tensor contractions used by the energy reference.  It retains immutable
versions of updated intermediates.  `adjoint.py` represents each implicit
amplitude solution by a custom node whose reverse action solves
`J.T lambda = -incoming_bar`.  A restricted residual graph supplies
`J.T @ vector`; it excludes all paths upstream of the amplitude inputs.
Thus a lambda iteration neither reconstructs the geometry response nor
recursively solves earlier-stage lambda equations.  The Jacobian is never
assembled, and the nonlinear iteration/DIIS history is never recorded.

Restricted VJPs also pass the active-parent mask into matrix products,
Einstein contractions, elementwise products/divisions and solves. During
an amplitude-only Krylov action, these primitives avoid computing
orbital/integral cotangents that would immediately be discarded. The
final energy pullback requests those cotangents normally. Diagnostics
count skipped parent pullbacks, and tests compare restricted actions
with the same components of an unrestricted derivative.

Local slices of the packed global amplitude vector return indexed
cotangents.  The VJP allocates each parent accumulator once and scatters
into it in place.  Allocating a full zero amplitude vector for every
local slice would introduce hidden quadratic work.  Solver diagnostics
include the number of contributed/scattered elements per VJP; a structural
scaling test checks this cost as the packed vector grows.

The initial `DenseInputs` adapter is an unsparsified algebraic test
adapter.  Its AO three-center tensor is cubic storage.  It must not be
presented as the final local integral or linear-scaling implementation.

## Projector, metric and orbital adjoints

For a symmetric eigensystem `D U = U n`, the reverse eigenvector term is

\[
 \bar D=U\{\operatorname{diag}(\bar n)
       +\operatorname{sym}[(U^T\bar U)\circ G]\}U^T,
 \qquad G_{ba}=1/(n_a-n_b).
\]

Retained/discarded blocks are included.  Rotations inside a fully retained
degenerate block are gauge freedoms, while a cut through a degenerate
block is rejected.  Metric matrix powers use the continuous Frechet
divided difference, which remains finite at repeated eigenvalues.

Numerical eigenvalue degeneracy is judged relative to the complete
spectrum: the gap tolerance is `32 n eps max(abs(eigenvalues))`, with the
same tolerance in tangent transport, reverse transport and subspace
selection. An absolute occupation-number tolerance can reject a weak
pair with well-resolved distinct eigenvalues, and would change the
projector response merely by rescaling its density. Tests verify both
the scale-invariant response and rejection of cuts through exact
degeneracies on NumPy and CuPy. This does not change any PNO/TNO selection
threshold or the energy model.

After reverse contraction of the correlation Lagrangian, the independent
orbital sources include bars for `C_occ`, `C_lmo`, `F`, and `S`.
`orbital_adjoint.py` eliminates their coordinate response with one Boys
adjoint and one RHF CPHF adjoint, including core/active mixing.  The
default Boys Hessian and the CPHF operator are matrix-free; compatible
null modes are parallel transported. The optional selected-pair Boys
operator described below stores a sparse Hessian. The final contraction has
the form

\[
 E^x=E_{\rm HF}^x+\bar S:S^x+\bar r:r^x+
       \bar F:F^{[x]}+\Gamma:B^x+Z_M:M^x.
\]

Here the bars already include localization and SCF relaxation.  There
must be no coordinate loop that solves CPHF or any correlated response
equation.  Integral-center derivative contractions accumulate forces
directly on their associated atoms.

For the bilinear derivative of the RHF potential, the implementation uses
the polarization identity for `Q(D)=Tr[D G(D)]/2`:
`Q^x(D0+Fbar)-Q^x(D0)-Q^x(Fbar)`.  This also includes the density-fitted
SCF auxiliary-center response, whose gradient contribution is quadratic
in the density.  Treating it as a linear extra force would be incorrect.

The DF exchange-gradient factorization must retain **every** component
of a signed response density.  The occupied-density helpers in the
installed PySCF version discard singular/occupation values below `1e-8`;
applying that approximation to the polarization terms breaks quadratic
homogeneity.  The response contraction locally binds an exact
factorization `D = I @ D` instead.  It does not change PySCF globals or
the parent SCF energy.  A regression checks homogeneity while a density
eigenvalue crosses the library's truncation threshold.

A small Boys gradient alone does not certify localization.  The local
driver additionally checks the lowest eigenmode of the Hessian after a
positive diagonal congruence scaling (which preserves its inertia).
Stationary saddles are displaced along a negative mode and repaired by
exact two-orbital Jacobi minimizations, followed by stationary polishing
and another curvature check.  This was necessary for a connected C16
chain; an unverified stationary saddle must not be used as evidence for
bounded localized domains.  User-supplied LMOs retain their chosen
branch, with their stationarity checked by the gradient.

## Shared local integral producer

`LocalRIBlocks` evaluates a requested transformed block as

\[
 q^Q_{ij}=\sum_{\mu\nu\in\mathcal A}
       B^Q_{\mu\nu}\widetilde C_{\mu i}\widetilde C_{\nu j}.
\]

The atom-complete coefficient supports are fixed discrete selections.
`lmo_cut=pao_cut=0` reproduces the unscreened reference; nonzero cutoffs
define the same additional approximation in the parent energy and its
gradient.  All PAO columns belonging to one atom share a support so that
rotations of an AO shell commute with truncation.  Independent masks for
the components of a p or d shell create an unphysical torque.

The local energy also defaults to `pao_screen='atom'`: a PAO/PAO integral
block is retained if any DOI within its two AO atom blocks exceeds the
original threshold.  The whole block is retained, so screening commutes
with rotations within that fixed selection branch.  This is a more
conservative screen than the legacy component-wise mask.  Use
`pao_screen='component', lmo_cut=0, pao_cut=0` to reproduce the original
reference energy; its small orientation dependence from component-wise
screening is then retained rather than silently changed.

The gradient checks that its actual coefficient supports equal those of
the parent energy before constructing the response graph.  Agreement of
the scalar energy alone cannot establish this: two masks can give nearby
energies and different derivatives.  A different cutoff is accepted only
when it selects identical supports; otherwise the energy must be rerun.
Changing the cache budget does not change this energy definition.

For a local primitive bar `qbar`, the AO density and coefficient source
are, for example,

\[
 \Gamma^Q_{\mu\nu}=\sum_{ij}\bar q^Q_{ij}
                 \widetilde C_{\mu i}\widetilde C_{\nu j},
 \qquad
 \bar C_{\mu i}=\sum_{Q\nu j}\bar q^Q_{ij}B^Q_{\mu\nu}
                 \widetilde C_{\nu j}+\text{right-index term}.
\]

Apply the same support mask to each coefficient bar.  Compact libcint
molecules contain only the required AO/auxiliary atoms, including compact
atom/environment tables.  Spatial derivatives `int3c2e_ip1`,
`int3c2e_ip2` and `int2c2e_ip1` are contracted in that block and scattered
directly to the owning atoms.  A bounded LRU cache holds raw local blocks.
There is no global `Gamma[nao,nao,naux]` or nuclear derivative axis.

For a local RI contraction `K = A.T @ M^{-1} @ B`, set
`W = M^{-1} @ A` and `V = M^{-1} @ B`.  An incoming `Kbar` gives

\[
 \bar A=V\bar K^T,\qquad \bar B=W\bar K,\qquad
 \bar M=-\operatorname{sym}(W\bar K V^T).
\]

When the two integral factors share coefficients, both sources must be
accumulated.  This metric term is the derivative of the RI inverse; it is
distinct from the PAO overlap response and from the auxiliary response of
the reference DF-SCF potential.  All three enter the total gradient.
The implementation reverses the local solve itself, so no explicit
inverse or coordinate-resolved metric derivative is stored.

An inverse map groups **all** primitive bars by auxiliary atom before
backtransformation.  For each orbital kind it first accumulates
`H[Q,nu,i] = sum_j C[nu,j] qbar[Q,i,j] + right-index contributions`.
One contraction with `B[mu,nu,Q]` supplies the coefficient bars; one
half-contraction with the coefficients supplies the symmetric AO density.
Each auxiliary atom therefore needs its integral derivatives only once.
The metric bars are similarly accumulated by their first auxiliary atom.
This avoids recomputing a derivative block separately for every pair and
prevents pair-order-dependent cache thrashing.

`LocalDLPNO` supplies the matching CPU energy; `LocalGPUDLPNO` adapts the
same producer to the existing batched GPU MP2/CCSD/triples kernels.
NumPy/CuPy adjoints share the same equations. Three-center integrals and
their derivatives, RHF response actions, CCSD/T1 primal solves, and the
T1 amplitude transpose have independently selectable GPU backends,
described below. Boys response, Krylov control, and two-center RI metric
derivatives remain on the CPU. Timings must identify the selected
backends and include transfers, replay, and global response work.

Normalized PAO diagonals are tied to floating-point precision.  The
Cholesky selector resolves such ties deterministically in both host and
device routines.  Finite-difference reports record the selected Cholesky
subspaces in addition to pair, domain and PNO/TNO ranks; fixed ranks alone
do not establish that the same differentiable energy branch was used.

The legacy pivot-selected PAO space has a further limitation.  At finite
`S_CUT`, discarded directions need not be exact null vectors.  Selecting
individual PAO columns can then give a retained approximate space that
depends on the orientation of the AO shells, even while the selected
column indices remain unchanged.  Its energy derivative includes this
orientation dependence.  For the connected C8/STO-3G LoosePNO test, the
z torque is `-5.0646e-7 Eh`.  An independent rigid-rotation energy
difference on the same branch gives `-5.0903e-7 Eh/rad`; a Cartesian
finite difference differs from the analytic derivative by `3.3e-9
Eh/bohr`.  A larger rotation changes the selected branch and is excluded
from derivative validation.  The strict rotation gate remains visibly
failed in that timing report.

For diagnosis, `dlpno_gradient_check.py --freeze-pao-pivots` evaluates
the continuous extension of the reference PAO-column branch.  It binds a
separate local CPU energy class, reuses only discrete column choices,
and independently recomputes SCF, Boys, all metric transformations,
PNO/TNO vectors and amplitudes.  The report identifies this protocol;
ordinary finite differences remain the default.

```python
from electrolites.dlpno.local import LocalDLPNO

cc = LocalDLPNO(mf, triples='t1', lmo_cut=1e-4, pao_cut=1e-4,
               E_CONVERGENCE=1e-11, R_CONVERGENCE=1e-10,
               MAXITER=150, T_CUT_ITER=0.)
cc.kernel()
driver = cc.nuc_grad_method(backend='numpy')
gradient = driver.kernel()                # dE/dR in Eh/bohr
force = -gradient
print(driver.timings, driver.diagnostics)
```

Existing dense energy objects expose `nuc_grad_method(method='adjoint')`;
their default full-support producer preserves the dense energy definition
but cannot acquire locality that is absent from its integral definition.
For the GPU feature adapter, `integral_backend='local'` selects the local
energy and its matching adjoint.  The direct coordinate-response oracle
remains explicitly available with `method='direct'` on dense objects.

### Independent GPU stages and bounded triplet graph storage

The local API separates primal, tensor, integral and reference-response backends:

```python
from electrolites.dlpno.local_gpu import LocalGPUDLPNO

cc = LocalGPUDLPNO(mf, pno='normal', triples='t1', ri_backend='gpu',
                  lmo_cut=1e-6, pao_cut=1e-6, S_CUT=1e-8,
                  E_CONVERGENCE=1e-11, R_CONVERGENCE=1e-10,
                  MAXITER=150, T_CUT_ITER=0.)
cc.kernel()
grad = cc.nuc_grad_method(backend='numpy', response_backend='gpu',
                         triples_backend='cupy', checkpoint_triples=True,
                         ccsd_adjoint_backend='cupy',
                         ccsd_primal_backend='gpu', triples_primal_backend='gpu',
                         checkpoint_ccsd=True, checkpoint_pnos=True,
                         checkpoint_integrals=True,
                         krylov_memory_mib=8192)
de = grad.kernel()
```

Run this example with `DLPNO_NGPU=1`. Here `backend='numpy'` places the
outer differentiated intermediates in host memory. The separate switches
select GPU primal amplitude kernels, CCSD/triples transpose kernels,
RI contractions and reference response. In particular,
`ccsd_adjoint_backend='cupy'` enables the batched CCSD lambda operator;
selecting the GPU energy driver alone does not enable that operator.
The current hundred-atom acceptance boundaries are listed at the start
of this document; source165 has passed complete T1 gradient and independent
finite-difference validation for the recorded hundred-atom case.

`LocalGPUDLPNO` uses the existing batched GPU energy equations.
`ri_backend='gpu'` selects GPU4PySCF Rys three-center kernels and CuPy
coefficient transformations for compact AO/auxiliary domains. The same
provider generates the two analytic integral-center derivatives and
contracts their auxiliary-atom densities on the GPU. Two-center metrics,
their derivatives and the bounded host caches remain on the CPU. This
path never introduces a full-molecule three-center storage requirement.

AO and auxiliary domains can have different atom tables and shell order.
The Rys backend preserves their independent coordinate tables, transforms
all derivative AO axes back from its sorted Cartesian basis, and restores
auxiliary contractions. The derivative kernels require a launch schedule
consistent with their 27-element per-thread output width; omitting that
bound can leave high-angular-momentum output unwritten. Tests compare
both derivatives and primal integrals directly with libcint for distinct
and reordered auxiliary subsets.

The CPU energy oracle and analytic replay use the GPU energy's PAO
ordering: normalize the overlap, sort by increasing off-diagonal absolute
overlap sum, then select Cholesky columns. At finite `S_CUT`, a different
ordering can retain a different approximate subspace even at the same
rank. Validation compares the resulting projectors, and the single-GPU
finite-difference runner records the actual selected columns. Discrete
selection changes are reported as branch changes, not gradient errors.
Both raw and effective topology are saved. If a PAO domain retains an
independent set whose dimension equals the global virtual dimension,
it spans the complete virtual space; changing those columns is an
equivalent basis choice. This exception requires the actual basis rank
after GPU orthogonalization. Partial PAO spaces and PNO/TNO rank changes
remain distinct branches. A finite-difference comparison requires the
same effective subspaces at the center and both displacements. Decreasing
the step does not guarantee this: a nearly singular PAO domain can switch
columns under orbital roundoff even at the smaller step. Tight orbital
convergence and explicit selection checks remain necessary. If a geometry
lies near a selection boundary, retain the failed comparison and identify
any additional smooth-branch test geometry separately.

The GPU PAO orthogonalizer also enforces zero coefficients on discarded
rows after the padded eigensolve. An eigenvector of a small positive
metric eigenvalue can acquire roundoff components in the padded null
space; division by its square root can magnify those components enough
to change the physical span. Masking the rows reproduces the compact
CPU eigensolve's exact support. A nearly singular metric regression
checks both this support and metric orthogonality. The batched cuSOLVER
wrapper checks each matrix's convergence status as required by the
[cuSOLVER API](https://docs.nvidia.com/cuda/archive/12.8.1/cusolver/index.html#cusolverdnxsyevbatched).

`response_backend='gpu'` replaces only the linear RHF Coulomb-exchange
action and the quadratic nuclear contraction described above, plus the
HF reference gradient. The exact signed factorization is tagged on the
GPU density and passed to `get_jk` directly. Calling an intermediate
routine that strips the factorization tags would silently restore the
occupied-density eigenvalue cutoff. Normalizing a response density before
the action, then restoring its linear or quadratic scale, also preserves
very small response terms under integral screening. The fitting basis
is inherited from the converged reference. Boys and Krylov control,
one-electron integrals and global orbital-source transformations remain
CPU work. None of these backend choices changes the staged scaling
criterion stated at the start of this document.

The full-rank DF response still stores an auxiliary-by-rank-by-rank
tensor. Applying its RI metric now overwrites independent occupied-pair
column blocks after their final read, avoiding full-tensor input and
output copies. The nominal block workspace target is 256 MiB; actual
process/device peaks must still be measured. This is exact multiplication,
with no eigenvalue truncation or low-rank approximation. Private copies of
the GPU4PySCF function namespaces bind this operation without changing
installed modules. `GPUReferenceBackend(..., metric_block_bytes=None)`
retains the original contraction for parity checks. The reference
calculation retains its global scaling and full tensor storage.

For each triplet, write its explicit outputs as
\(y_{ijk}=f_{ijk}(x_{ijk})\). For T0 the retained output can be the scalar
energy. Iterative T additionally retains \(W,V,T^{(0)}\), while its coupled
amplitude response still uses the same implicit adjoint solve. During the
reverse pass the local graph is rebuilt once and contracted with its
incoming cotangent:

\[
 \bar x_{ijk} = f'_{ijk}(x_{ijk})^T\bar y_{ijk}.
\]

The calculation retains the boundary inputs and outputs instead of all
intermediate cubic TNO tensors for every triplet simultaneously. There
is no finite difference or omitted derivative. The saved boundary includes
PNO/TNO transformations, pair amplitudes, orbital matrices and newly read
primitive integrals. The recomputation uses the captured stage's inputs,
including its TNO scaling and selected domains. The largest inner graph
is one triplet; retained outputs, CCSD/MP2 graphs and primitive cotangents
still require storage proportional to the retained local problem size.
The full amplitude graph is released before RI and global response
workspaces are created.

For iterative triples, the amplitude equations are linear once the pair
amplitudes and TNOs are fixed. With `P_tu` the occupied-index permutation,
their exact transpose action is

\[
 (J^T\lambda)_u=\Delta_u\lambda_u-
 \sum_t f_{tu}P_{tu}^{-1}
       [(S_{tu}^T)^{\otimes 3}\lambda_t].
\]

`TriplesTranspose` evaluates this action directly on NumPy or CuPy
tensors. It stores local overlap matrices, without recording cubic
coupling intermediates for every Krylov iteration. With
`checkpoint_triples=True`, each residual block's full graph is also
recomputed locally for the final orbital/integral/CC pullback. All
derivatives of the denominator, overlaps, TNOs and triples source remain
in that pullback; the explicit transpose replaces only the amplitude
Jacobian action. Tests compare it with the independent automatic
transpose for unequal triplet ranks and occupied-index permutations.

`triples_backend='cupy'` can run this amplitude transpose on the GPU while
keeping the main tensor graph on NumPy. Its overlap coefficients are
formed on the tensor backend, so the hybrid solve uses the same numerical
operator. Couplings are grouped by padded local ranks and permutation;
three batched matrix products apply each tensor projection, followed by
atomic accumulation into the compact amplitude vector. Each batch has a
bounded temporary workspace. This transfers only the Krylov vector and
its image per action; it does not put the full derivative graph on the
device or omit the final upstream residual pullback.

A C34H70/def2-SVP T1 run (104 atoms, source-v36) completed this hybrid
transpose solve for 338,778,362 amplitude variables and 189,040 couplings.
Eleven Krylov iterations with restart length two gave an explicit
adjoint residual of `1.27e-12` in 82.1 seconds between the adjoint start
and completion callbacks. That interval includes CPU Krylov control and
vector transfers; it excludes operator construction and the surrounding
graph replay. The events and source provenance are retained in
`/pscratch/sd/w/weiliang/dlpno-gpu-gradient-20261018/c34-t1-gpu-triples-adjoint-v61.json`.
The transpose implementation is unchanged in source-v70. This is evidence
for the large-system amplitude solve only: the complete 104-atom T1
nuclear gradient and its independent finite differences remain pending.

An isolated source-v82 candidate reduces temporary memory in the GPU T1
energy iteration. Each Jacobi sweep still forms all coupled residuals
from the previous amplitudes before updating any amplitude. It evaluates
the original residual norms in chunks, then overwrites each consumed
residual slice with its denominator quotient and subtracts that slice
from the amplitudes in place. The squared-norm workspace target is
128 MiB, with one whole triplet as the minimum chunk. Consumed residuals
and duplicate overlap storage are released before the next sweep. This
changes storage and evaluation order, not the equations, thresholds or
convergence tolerances. The candidate remains separate from the current
source-v70 worktree until its full gradient validation finishes.

On the same asymmetric 104-atom reference, this candidate completed the
full T1 energy on an A100 40 GB: all energy components agreed with the
80 GB source-v39 baseline within `6.37e-12 Eh`. The actual local spaces
contain 6,146 triplets and 353,233,905 amplitude variables. The complete
CPU/GPU transpose actions agreed to `4.16e-17` maximum absolute error
and `1.06e-16` relative L2 error. The GPU action took 2.72 s including
vector transfers, versus 138.12 s for the CPU action with one BLAS thread;
operator construction took 41.04 s and 40.67 s, respectively. These are
operator timings, not a whole-gradient speedup. The diagnostic's sampled
GPU peak was 34.36 GiB and host high-water memory was 28.82 GiB. The failed
source-v70 40 GB attempt and its allocation error are preserved alongside
the successful candidate, rather than counted as a completed baseline.

Four independent CPU/GPU full-gradient regression cases passed. On
14-atom C4H10/def2-SVP, all 42 T1 gradient components differed from
source-v70 by at most `1.42e-9 Eh/bohr`. The diagnostic audit is
`/pscratch/sd/w/weiliang/dlpno-gpu-gradient-20261018/c34-t1-gpu40-feasibility-audit-v82.json`;
the small full-gradient comparison is `c4-triples-memory-comparison-v82.json`
in the same directory. Neither result constitutes a complete 104-atom
nuclear gradient or its independent finite-difference validation.

The separate source-v39 C34H70 CCSD run converged its CPU matrix-free
adjoint for 524,059 variables in 17 iterations, with an explicitly checked
infinity-norm residual of `1.57e-11`. The adjoint callbacks span 8284.3 s;
the final upstream density pullback and nuclear response lie outside
that interval. This identifies the CPU CCSD transpose/replay as a remaining
cost even when the numerical CCSD and triples solves use the GPU. The
captured events are in
`/pscratch/sd/w/weiliang/dlpno-gpu-gradient-20261018/c34-ccsd-adjoint-convergence-v67.json`.
The same run subsequently completed all 312 gradient components in
13157.84 s, with host high-water memory of 122.62 GiB and a sampled GPU
peak of 54.48 GiB. Translation and rotation residuals were respectively
`1.22e-12 Eh/bohr` and `1.07e-8 Eh`. These timings exclude the earlier
SCF/Boys preparation. The complete report and resource hashes are in
`/pscratch/sd/w/weiliang/dlpno-gpu-gradient-20261018/c34-ccsd-complete-gradient-v72.json`.
Two Cartesian components, each checked at two step sizes, subsequently
passed independent unfrozen finite differences. Each displaced energy
recomputed SCF, Boys localization and domain selections. The raw energy
points and full effective topologies were independently rechecked, with
the same effective selections in all four comparisons. Using zero-based
atom indices, the signed errors (analytic minus finite difference) are:

| Component | Step (bohr) | Error (Eh/bohr) |
| --- | ---: | ---: |
| Atom 0, z | `1e-4` | `-1.26284e-8` |
| Atom 0, z | `5e-5` | `-2.39663e-9` |
| Atom 103, y | `1e-4` | `2.67736e-8` |
| Atom 103, y | `5e-5` | `3.70054e-8` |

The maximum error is `3.70054e-8 Eh/bohr`. The numerical checks cover two
of the 312 analytically evaluated components. The full
acceptance record and CSV are in
`/pscratch/sd/w/weiliang/dlpno-gpu-gradient-20261018/c34-ccsd-full-validation-v75.json`
and `c34-asymmetric-ccsd-fd-summary-v75.csv` in the same directory.

The execution audit records 520 GPU integral calls, 312 GPU transforms,
104 atom backtransforms, 22 reference Fock actions and three reference
quadratic-force calls. The reference density factorization retained full
rank. The GPU CCSD primal replay took 18.59 s, while its adjoint remained
on the CPU. These counters and their internal timers overlap the broad
gradient stages and do not define an exclusive GPU fraction or a
controlled speedup. The complete audit is
`/pscratch/sd/w/weiliang/dlpno-gpu-gradient-20261018/completed-gpu-stages-v74.json`.

`ccsd_primal_backend='gpu'` also uses the energy driver's batched CCSD
solver to converge the numerical amplitudes during the gradient replay.
It rebuilds the solver from the replay's current PNO basis and integrals;
it does not assume identical orbital gauges to the original energy run.
The numerical state owns copies of its integral lists, so the GPU solver
can release those copies without modifying the derivative graph. The
same analytic residual graph checks the resulting amplitudes and forms
the CCSD adjoint and all upstream sources. This option changes only the
primal solve; the main tensor backend still controls the CCSD adjoint.
The benchmark switch is `--ccsd-primal-backend gpu`, with CPU as the
default. Diagnostics report the selected backend and primal solve time.

For T1, `triples_primal_backend='gpu'` (benchmark
`--triples-primal-backend gpu`) likewise reuses the energy driver's batched
triples iteration, retaining its converged amplitudes for the adjoint.
The complete analytic residual checks these amplitudes in the replay's
current TNO spaces. This option does not change `triples_backend`, which
independently selects the amplitude transpose backend. Its default is CPU.

`checkpoint_ccsd=True` (benchmark `--checkpoint-ccsd`) recomputes each
strong pair's doubles-residual contribution during the reverse sweep.
The shared pair function includes the complete occupied-domain sums and
all overlap, integral, singles and doubles dependencies. Its amplitude
inputs are captured before the implicit amplitudes replace the state's
lists. Local overlap caches are cleared between replays. This trades
repeated local work for smaller stored residual graphs; it does not drop
couplings. A restricted checkpoint VJP visits only the requested parent
paths during a lambda action, while the final density sweep requests all
upstream sources. All checkpoint boundary variables stop the local graph
traversal, including those whose cotangents were not requested; their
ancestors are handled by the outer sweep. The default remains the stored
CCSD residual graph.

`checkpoint_pnos=True` (benchmark `--checkpoint-pnos`) additionally
recomputes each semicanonical MP2 screening energy and each initial MP2
PNO construction. Only the final pair coefficients, energies, integrals,
amplitudes and truncation correction are stored, together with primitive
input leaves. The orthogonalization and pair-density eigensystem graphs
are local temporaries. Their full derivatives are replayed once in the
final density sweep; the implicit MP2 amplitude solve keeps the PNOs
fixed and does not replay these constructions during its lambda actions.
Each checkpoint retains the pair/domain lists of its own screening stage.

The same option checkpoints each TNO subspace construction. It retains
the final TNO coefficients and semicanonical energies, then replays the
PAO metric orthogonalization, pair-density projection, TNO eigensystem,
and canonicalization for their full derivatives. Prescreening, final T0,
and iterative-triples stages retain their own triplet/domain lists and
importance scales. Pair amplitudes are shared immutable inputs. This
avoids retaining every triplet's metric and density intermediates while
preserving responses of both retained and discarded subspaces. The
`tno_subspace_checkpoints` diagnostics report discarded intermediate bytes
and retained output bytes; these counts are distinct from measured RSS.

The final local-density reverse sweep releases intermediate values,
primitive input values, and pullback closures after their last use. The
independent primitive cotangents and selection maps survive for the RI
contraction; input block statistics are captured before the sweep.
Amplitude Krylov actions retain reusable graphs throughout their solves;
only the subsequent density sweep consumes the graph. This reduces overlap
between stored primal intermediates and the growing primitive cotangents.
The dense adjoint oracle retains its graph unless
`densities(release_graph=True)` is requested explicitly. Discarding its
primitive values also requires `release_inputs=True`; the default consuming
sweep preserves those values for callers that still need them.

Before the local gradient builds its own tracked primitives, it releases
the parent energy's regenerable integral caches. Support maps, extended
domains, orbital data, and amplitudes are retained. The released host-cache
bytes and the cache-release time are reported separately; subsequent
energy or gradient requests can regenerate the same integral blocks.

The amplitude GMRES basis has a separate `krylov_memory_mib` target
(8192 MiB by default). Its restart length is at most 40 and is reduced for
large amplitude vectors; at least two directions are retained when the
problem has that many variables. `krylov_basis_bytes` includes the extra
Arnoldi vector. The target covers the basis only, excludes other operator
and solver storage, and can be exceeded by the minimum two-direction basis.

With `triples_backend='cupy'`, source165 keeps T1 Arnoldi vectors, residuals
and the solution on the GPU. Only the small Hessenberg least-squares
problem stays on the host; the accepted solution returns to the host when
the outer graph uses NumPy. CPU fallback and the CCSD SciPy solve retain
host Krylov bases. GPU convergence requires both the true unpreconditioned
L2 residual tolerance and the 1e-8 maximum-component bound. Nuclear tangent
regressions also exercise restart length two for MP2, CCSD and T1.

The same exact recomputation is available for each pair's PNO integral
transformation (`checkpoint_integrals=True`). Only its final integral
outputs are retained; local RI gathers, metric solves and extended-domain
contractions are replayed in the reverse pass. The coupled CCSD residual
and its implicit adjoint remain unchanged. The CPU/tangent reference
uses the same extracted pure pair function without checkpointing.

The RI inverse map stores Cartesian index lists and views of primitive
densities. It does not expand three integer tensors of the same size as
each three-center density. Repeated and unsorted indices retain scatter
addition semantics. This reduces temporary storage without modifying the
coefficient or PAO-pair masks.

Overlapping PAO/PAO, LMO/PAO and LMO/LMO primitive requests share
auxiliary-atom leaves using the parent energy's observed local column
envelopes. Triplet slices are
then internal gather operations that the checkpoint can discard and
rebuild. Their cotangents accumulate on the shared leaves before the RI
force contraction. Requests outside an observed envelope keep the exact
independent-block fallback; no new truncation is introduced.

PAO overlap and Fock primitive requests also share fixed tiles of at most
128 by 128 entries. Checkpoints discard and replay the selected local
gathers; their cotangents accumulate on these common tile leaves. Each
checkpoint's input cotangent is therefore bounded by the fixed tile size,
instead of allocating a full global matrix for every local replay. All
matrix entries and repeated-index scatter contributions are retained.
General index selections keep the exact independent-block fallback.
Each local selection is recorded as one gather operation. Its pullback
returns indexed cotangents only for the required tiles, including repeated
and unsorted indices. Recording separate gathers, concatenations and
permutations retained redundant intermediate arrays outside checkpoints;
the single operation avoids those arrays without changing the derivative.
The existing dense two-index reference matrices are still reported as
global orbital source storage. This change does not establish overall
linear scaling or a lower process memory peak on every small molecule.

For C4H10/def2-SVP NormalPNO T1 with `S_CUT=1e-10`, the shared-tile
implementation in source-v70 differs from source-v39 by at most
`1.91e-9 Eh/bohr` across all 42 gradient components. Twenty-one local
integral/gradient tests and four complete NumPy/CuPy T0/T1 gradient checks
also pass. The retained-array regression fails on the previous
multi-operation gather implementation and passes with the single gather.
The full comparison, source hashes, and resource measurements
are recorded in
`/pscratch/sd/w/weiliang/dlpno-gpu-gradient-20261018/c4-shared-orbital-comparison-v70.json`.
These small-system checks do not constitute a complete 104-atom T1
gradient validation.

The source-v70 C34H70 MP2 run also completed all 312 components, with a
maximum difference of `1.85e-9 Eh/bohr` from source-v39 and identical
reported energies. Host high-water memory was 42.87 GiB, compared with
43.64 GiB for source-v39; both sampled GPU peaks were 49.59 GiB. The
gradient took 895.32 s versus 873.71 s. The nodes and concurrent workloads
differed, so these timings do not establish a speedup. Source-v70 uses
47 tile leaves with 667,428 elements for each of the PAO overlap and Fock
sources. Process memory also includes the local graph, integral data,
response work and allocator storage. The comparison is retained in
`/pscratch/sd/w/weiliang/dlpno-gpu-gradient-20261018/c34-mp2-shared-orbital-comparison-v71.json`.
Its independent FD reference comes from source-v39; the source-v70 run
repeated the full gradient, not the displaced energies.

The intervening source-v57 large MP2 attempt remains incomplete in the
record. Its observed host high-water memory exceeded 65 GiB, and its
optional process controller failed while inspecting an exiting child;
the child's exit cause was not established. The incomplete run and the
controller failure are recorded in
`/pscratch/sd/w/weiliang/dlpno-gpu-gradient-20261018/c34-mp2-v69-incomplete-run-audit.json`.

The GPU energy plans the same auxiliary-atom envelopes from its existing
extended domains before entering the pair loop; the gradient inherits
the parent energy's observed envelopes. This avoids repeated integral
optimizer builds while domains grow incrementally. Planning evaluates no
integrals, changes no coefficient masks, and retains the bounded cache
budget. A request beyond a planned envelope still expands it exactly.

`benchmarks/dlpno_adjoint_scaling.py` exposes independent energy, RI,
reference-response and tensor backends. Its GPU memory observations are
sampled CuPy pool values at phase boundaries, not a measured process-wide
device-memory peak. `benchmarks/dlpno_checkpoint_fd.py` validates a saved
gradient against independently converged displaced energies at multiple
steps, including localization-branch matching and discrete-selection
checks. It verifies the source and trusted-local checkpoint hashes before
using either artifact. Performance and accuracy claims must name the
actual molecule, basis, thresholds and source snapshot.

For a precision audit of very small finite-difference steps,
`--hf-energy-backend cpu` independently evaluates the same converged RHF
density with PySCF/libcint. `--hf-energy-backend legacy-gpu` streams the
older GPU4PySCF three-center engine into occupied blocks and uses libcint
one- and two-center integrals. Neither option reoptimizes orbitals or
changes correlation thresholds. These options select the reference energy
used in the numerical derivative; the report also preserves the original
SCF energies and their finite-difference errors. Corrections larger than
`1e-8` Eh are rejected as more than a numerical precision audit. The
normal default remains `scf`. This check matters because geometry-dependent
energy noise of order `1e-10` Eh can dominate a difference with a step of
`1e-5` bohr. It does not waive the discrete-topology or convergence checks.
The independent reference evaluators are global validation oracles and
are not part of the conditional O(N) local correlation claim.

The checkpoint validator exposes separate SCF energy, SCF orbital-gradient,
and Boys gradient tolerances. It records the actual displaced SCF orbital
residuals, including a failed convergence check. This distinguishes an
energy stopping tolerance below the floating-point noise floor from a
nonstationary orbital solution. `--save-orbitals --save-points` preserves
each orbital solution and each completed energy point before starting the
next displacement; source, input and validator hashes accompany the data.
With `--compact-points --save-points`, the main report stores each energy,
error and convergence check together with the point-file path and SHA-256.
Both complete topology records remain in that point file. Topology equality
is checked on the full in-memory records before compacting the report;
this avoids duplicating large selection histories in every progress write.
These diagnostics do not bypass convergence or topology acceptance gates.

`benchmarks/dlpno_resource_monitor.py` additionally samples the calculation
process's device memory through `nvidia-smi`, and its host high-water mark
through `/proc`. The reported GPU maximum is explicitly a sampled lower
bound, and includes reserved allocator memory.

The source-v39 validation on 2026-10-18 completed all 312 MP2 gradient
components for C34H70 (104 atoms, 826 AOs, def2-SVP, NormalPNO,
`S_CUT=1e-8`, coefficient cutoffs `1e-6`) on one A100 80 GiB. The energy
and gradient stages took 132.04 s and 873.71 s; host high-water memory was
43.64 GiB and the sampled device peak was 49.59 GiB. These are timings of
the converged numerical-reference replay and exclude its earlier SCF/Boys
preparation. Two Cartesian components, each at `1e-4` and `5e-5` bohr,
passed independently rebuilt, unfrozen finite differences; the largest
absolute error was `3.39194e-8` Eh/bohr. All four comparisons retained the
same effective selections, with raw energy-point hashes and full topology
records independently rechecked. The geometry used a predetermined
zero-mean perturbation (seed 20261018, uniform half-width 0.01 bohr) of
the original chain; the original geometry's selection-switching failures
remain in the audit record.

MP2 and CCSD now each have a complete 104-atom gradient with the
independent FD checks reported above. T0 and T1 still require their own
finished gradients, resource records and independent differences.
The MP2 evidence is in
`/pscratch/sd/w/weiliang/dlpno-gpu-gradient-20261018/`:
`c34-asymmetric-mp2-gradient-summary-v41.json` and
`c34-asymmetric-mp2-fd-summary-v41.json`. Finite-size performance evidence
does not remove the bounded-domain assumptions or the global response
costs described below. The source-specific evidence index is
`validation-index-v86.json`; `completed-independent-fd-v86.csv` preserves
the signed completed comparisons. The matching figure is
`completed-independent-fd-v87.pdf` (also PNG/SVG), with pending T0/T1
results explicitly omitted. All these generated artifacts remain in the
same scratch directory.

## Local data structures and cost

At fixed thresholds, let the retained pair/triplet count and the total
number of their coupling edges grow proportionally to `N`.  Bound the
PAO, PNO, TNO and fitting-domain sizes independently of `N`.  Each primal
local contraction and its reverse contraction then have bounded cost,
and a full residual/VJP sweep costs `O(N)`.  This claim additionally
requires a size-independent number of primal and adjoint iterations;
iteration counts must therefore be measured in the scaling benchmark.

More explicitly, if a pair/triplet block contains at most `b` variables
and has at most `z` retained neighbors, a residual or transposed-residual
sweep costs `O(N z C(b))`.  A reverse contraction has a constant multiple
of its parent contraction's arithmetic cost; it does not acquire a
nuclear-coordinate index.  With `k` Krylov iterations, the amplitude
response costs `O(k N z C(b))`, and its recorded graph occupies
`O(N z M(b))`.  The auxiliary-atom backtransformation has the same bound
when the union of all supports incident on one auxiliary atom is bounded.
These are conditional bounds: small gaps, extended orbitals, growing
domains or growing Krylov counts invalidate the claimed linear regime.

The measured sizes must therefore include the **extended union** used
by the derivative integral producer, as well as the individual PAO and
fitting domains.  Saturation of an individual LMO domain alone is not
sufficient evidence for saturation of the derivative calculation.

An auxiliary-atom block owns the orbital/PAO domains that occur in the
pairs and triplets using that atom.  Their inverse maps determine the AO
supports needed for both the parent integral and its derivative.  Local
three-center and metric bars are accumulated into these blocks, then
contracted with derivative integrals while the block is resident.  A
global AO three-center density or a `3N` derivative tensor would defeat
this construction.

The raw-integral cache now maintains an expanding AO envelope for each
auxiliary atom.  Requests for overlapping pair supports reuse the same
block and extract only their requested rows/columns; the coefficient
screens and energy are unchanged.  Envelopes survive LRU eviction as
index maps, and their maximum and total sizes are reported.  This avoids
regenerating integrals for many nearly identical AO subsets.  Cache
bytes remain bounded independently of molecule size.

Transformed `qia`, `qij` and `qab` blocks use auxiliary-atom column
envelopes as well.  Repeated pair requests therefore reuse transformed
local blocks.  The cache reserves at most one quarter of its budget for
raw three-center blocks and one eighth for metric blocks, protecting
the frequently reused transformed blocks.  An oversized streamed block
leaves the small-block cache intact.  Integral and transformation call
counts are included in benchmark reports, alongside the fixed cache
budget.

Long benchmark runs can save their converged primal state with
`--energy-checkpoint` and continue with `--resume-energy`.  A SHA256
manifest binds the checkpoint to every correlation source file and the
energy settings.  PySCF regenerates transient node-local DF files before
the resumed gradient timer starts.  The checkpoint contains a completed
energy state, not a completed gradient; partial reports retain
`complete=false`.

The provider removes global `qia`/`qij` storage. Pair lookup and
occupied-Fock updates are sparse in the local CPU/adjoint path; MP2,
CCSD and triples substitutions use adjacency lists. Enable both
`dipole_tail_options={}` and `local_screening_options={}` to use the
complete hierarchical tail and the spatial DOI producer. The default
exact-pair/dense-DOI paths remain useful accuracy oracles and carry their
original higher costs. Global two-index orbital preparation and response
are retained in either mode, as allowed by the confirmed scope.

| Stage | Work under bounded local domains and bounded iterations |
| --- | --- |
| Local MP2/CCSD/(T) energy contractions | `O(N)` |
| Residual graph construction and one VJP sweep | `O(N)` |
| Amplitude adjoints with fixed GMRES restart and iteration count | `O(N)` |
| Local RI density and integral-center derivative contraction | `O(N)` if the extended auxiliary-atom supports are bounded |
| Complete hierarchical distant-pair correction and its scalar adjoint | `O(N)` at fixed quadrature/order and bounded spatial interaction lists |
| Spatial DOI and populations after coefficient supports are available | `O(N)` with bounded supports, neighborhoods and spatial density |
| Exact distant-pair oracle / dense DOI oracle | `O(N^2)` / `O(N^3)`; optional validation paths |
| Dense support discovery and PAO/orbital source backtransforms | `O(N^2)` / `O(N^3)` reference-orbital work |
| Full Boys response | One matrix-free solve; `O(N^3)` per action, as in localization |
| RHF CPHF and relaxed Fock integral derivative | One CPHF solve plus a fixed number of reference J/K derivative contractions; same primitive scaling as the selected reference backend |

The dense DOI oracle cost follows from the `grid x orbital` products
and Gram matrices in `compute_doi`; a quadratic table size does not imply
quadratic work. These reference/oracle stages are included in their
reported wall time and are never charged as bounded local work.
The local adjoint removes the coordinate-count factor from the response,
but does not remove the global operations inherited from the energy.

Pinski and Neese, [JCP 150, 164102 (2019)](https://doi.org/10.1063/1.5086544),
pp. 11--12, provides the extended-map construction for local RI
derivatives.  Its scaling discussion explicitly distinguishes the linear
local pair/integral loops from quadratic prescreening corrections and
nonlinear orbital-response overhead.  The complete timing audit here
must likewise expose these costs.  In particular, neither a linear
lambda loop nor the phrase "Z-vector" proves that the complete
correlation-gradient calculation scales linearly.

## Completion evidence required

1. Correct total MP2, CCSD, T0 and iterative-T gradients, including
   truncation, weak-pair, dipole and screened-triples corrections, compared
   with the existing direct oracle and independent finite differences.
2. One coordinate-independent adjoint solve per implicit amplitude
   stage, plus coordinate-independent localization/SCF relaxation.
3. Local integral-density/derivative processing without global cubic
   correlation tensors, and adjacency-based pair/triplet couplings.
4. CPU/GPU parity and integration with the existing energy driver.
5. Increasing connected-system sizes at fixed basis and thresholds:
   phase timings, peak memory, numbers and sizes of domains/couplings,
   primal/adjoint iteration counts, and fitted scaling compared with
   the matching single-point phases. A repeated disconnected-monomer
   benchmark alone is insufficient. Finite-size slopes need not yet be
   one when extended supports are still growing; this limitation must
   remain explicit rather than be hidden by a fit or by excluding costs.

Until these checks have passed, the linear-scaling goal remains active.

The optional selected-pair Boys solver below is an experiment beyond the
confirmed acceptance scope. The production default retains the full
response equations. Its failure to pass a bounded-radius residual test
does not justify dropping localization response from the gradient.

## Complete long-range correction: separated kernel and mutual adjoint

The optional `dipole_tail_options` backend replaces the **sum** of distant
pairs and its derivative, as well as the all-pairs dipole prescreen. Sparse
DOI tables and pair masks are selected automatically. The additional
`local_screening_options` replaces dense DOI production and the global
transition-dipole contraction. Global reference-orbital operations remain. The
default exact pair backend remains the reference. No pair is deleted by
an interaction radius. This distinction is necessary: a faster tail sum
alone does not complete the linear-scaling correlation implementation.

For local virtual excitation gaps `g_ia = epsilon_ia - F_ii > 0`, Cartesian
transition dipoles `d_ia`, and `T(R) = (I-3 u u^T)/R^3`, the unordered-pair
energy in the existing ordered-pair convention is

\[
 E_{ij}=-8\sum_{ab}\frac{(d_{ia}^{T}T(R_{ij})d_{jb})^2}{g_{ia}+g_{jb}}.
\]

Use `1/x = integral_0^infinity exp(-t*x) dt`. A positive quadrature with
fixed nodes and weights gives symmetric Cartesian tensor sources

\[
 P_i^{q}=\sqrt{w_q}\sum_a d_{ia}d_{ia}^{T}e^{-t_qg_{ia}},\qquad
 E_{ij}\simeq-8\sum_q\operatorname{Tr}(P_i^qT P_j^qT).
\]

`dipole_laplace.py` uses geometric panels and Gauss-Legendre quadrature.
The input gap interval is fixed for the numerical energy definition; an
out-of-interval gap raises rather than changing nodes during a derivative.
Analytic quadrature remainder/tail bounds cover both `1/x` and `1/x^2`,
the latter being needed for excitation-gap response. The bounds exclude
floating-point roundoff. Quadrature positivity allows the scalar energy
error to be bounded relative to the unexpanded sum of squared couplings;
it does not imply a relative bound on potentially cancelling forces.

Represent `P` in the six orthonormal symmetric matrices `B_a`, with
off-diagonal components normalized by `1/sqrt(2)`. The resulting kernel is

\[
 K_{ab}(R)=\frac{\delta_{ab}}{R^6}
 -6\frac{(B_aR)\cdot(B_bR)}{R^8}
 +9\frac{(R^TB_aR)(R^TB_bR)}{R^{10}}.
\]

For cell centroid `c_A`, normalized Cartesian moments are

\[
 M_{A,\alpha}^{qa}=\sum_{i\in A}P_i^{qa}
       \frac{(r_i-c_A)^\alpha}{\alpha!},\quad |\alpha|\le p.
\]

For well-separated cells the scalar interaction, **truncated by total
degree**, is

\[
 E_{AB}^{(p)}=-8\sum_{q,a,b}\sum_{|\alpha|+|\beta|\le p}
 (-1)^{|\beta|} M_{A,\alpha}^{qa}
 D^{\alpha+\beta}K_{ab}(c_A-c_B)M_{B,\beta}^{qb}.
\]

The total-degree choice preserves rotational covariance of each cell
interaction on a fixed tree branch. `dipole_tree.py` builds kernel Taylor
coefficients analytically by polynomial/binomial expansion, not finite
differences. Upward moment translation is

\[
 M_{A,\alpha}=\sum_{C\subset A}\sum_{\beta\le\alpha}
 M_{C,\beta}\frac{(c_C-c_A)^{\alpha-\beta}}{(\alpha-\beta)!}.
\]

Reverse differentiation of the mutual scalar interactions supplies moment
bars and centroid bars; reversing the upward translations is the downward
field pass. Leaf bars give `dE/dP_i` and `dE/dr_i`. Cell centroids are
unweighted averages of their sources and **their response is included**;
freezing those centers would differentiate a different function. Local
polarizability pullbacks supply both transition-dipole and gap bars, then
the existing PAO, Boys and RHF adjoints supply nuclear derivatives. Near
leaf pairs use the original virtual-index denominator exactly.

The mutual-interaction construction follows the Cartesian strategy in
[Dehnen, JCP 179, 27–42 (2002)](https://doi.org/10.1006/jcph.2002.7026).
The applicability of hierarchy methods beyond the Coulomb kernel is also
discussed by [Fong and Darve, JCP 228, 8712–8725 (2009)](https://doi.org/10.1016/j.jcp.2009.08.031).
Our tensor kernel, total-degree scalar energy, local gap separation and
all centroid/moment pullbacks are derived above; gravitational force
formulae are not substituted for the dipole energy derivative.

An orientation-independent energy remainder bound follows from
`|C_n^s(cos(theta))| <= binomial(n+2s-1,n)` for the Gegenbauer expansion
of inverse radial powers. With `eta=(radius_A+radius_B)/R < 1`, define

\[
 U_s(m,\eta)=\sum_{n\ge\max(m,0)}{n+2s-1\choose n}\eta^n.
\]

Then the spectral norm of the omitted kernel is bounded by

\[
 R^{-6}\left[U_3(p+1,\eta)
 +6\sum_{k=0}^{2}{2\choose k}\eta^k U_4(p-k+1,\eta)
 +9\sum_{k=0}^{4}{4\choose k}\eta^k U_5(p-k+1,\eta)\right].
\]

Multiplying by `8 sum_q (sum_i ||P_i^q||_F)(sum_j ||P_j^q||_F)` bounds
the cell interaction's absolute energy error. Summing these bounds and
the Laplace remainder gives the reported energy bound. It is conservative
and excludes roundoff; a force-error certificate is **not** claimed.
Forces are checked against the exact pair derivative, independent finite
differences, and refinement of order/opening criteria.

Hashed fixed-width leaves and bottom-up parent cells avoid sorting all
particles or storing every ancestor's particle list. A symmetric traversal
partitions unordered pairs into direct leaf pairs and accepted cell pairs;
an accounting check covers all `N(N-1)/2` pairs. Retained correlated pairs
are exclusions. Far interactions require their minimum separation to
exceed the longest excluded separation, so an exclusion is always resolved
exactly in a direct leaf pair. This radius selects an evaluation method,
not a physical cutoff. Fixed order/quadrature count/opening/leaf width,
bounded spatial density and bounded exclusion range give linear hierarchy
work and storage. Pathological density, an increasing exclusion range or
accuracy settings that grow with `N` invalidate that statement.

Opt in with, for example,

```python
cc = LocalDLPNO(mf, triples='t1',
    dipole_tail_options=dict(order=7, theta=0.3, gap_bounds=(0.01, 100.0)))
cc.kernel()
gradient = cc.nuc_grad_method().kernel()
print(cc.dipole_tail_diagnostics)
```

The scalar tail primitive is shared by the CPU/GPU parent energy and local
adjoint adapter. The hierarchy currently runs on the CPU. The direct
coordinate-response and dense-input adjoint oracles reject this optional
numerical energy; they must not silently freeze its tail derivative.


### Conservative sparse dipole preselection

The original dipole preselection quantity is

\[
 B_{ij}=-\frac{16}{R_{ij}^6}\sum_{ab}
       \frac{(d_{ia}\cdot d_{jb})^2}{g_{ia}+g_{jb}}.
\]

Cauchy-Schwarz and `x+y >= 2 sqrt(x*y)` give

\[
 a_i=\sum_a\frac{\lVert d_{ia}\rVert^2}{\sqrt{g_{ia}}},\qquad
 |B_{ij}|\le\frac{8a_i a_j}{R_{ij}^6}.
\]

Therefore `B_ij` can exceed its positive threshold `tau` only within
`R_ij <= (8 a_i a_j/tau)^(1/6)`. `dipole_prescreen.py` uses the largest
such radius as a spatial hash-cell width, examines neighboring cells,
and evaluates the original virtual-index formula only for candidate
pairs. The analytical upper bound and a floating-point radius guard
prevent discarding pairs that could exceed the threshold. Coincident
centers retain the original mandatory-pair rule. Positive excitation
gaps are checked explicitly; a zero screening threshold is not a
linear-scaling regime and is rejected by this optional backend.

The result is unioned with the DOI screen and the diagonal. The optional
backend no longer stores dense dipole energy/bound arrays or their
cotangents; a single scalar tail leaf enters the amplitude Lagrangian.
`SparseValues` retains values above the lowest threshold any consumer
requires and rejects lower-threshold queries. `PairMask` stores selected
neighbors and preserves deterministic pair order. Atom-complete PAO-pair
masks store atom adjacency. Diagnostic dense conversions are explicit;
domain construction and pair refinement use sparse row/column queries.
No contribution to the distant-pair energy is omitted by the preselection
radius.

### Spatial DOI and centered local transition dipoles

`local_screening_options={}` enables an explicit numerical approximation
to domain preparation. Its defaults are coefficient cutoff `1e-8`, AO
envelope tolerance `1e-14`, atomic radial grid radius `20` bohr and Becke
partition radius `40` bohr. The local producer uses the same atomic
Treutler/Lebedev grids, radii adjustment and Becke partition formula as
the dense oracle. A Gaussian shell envelope bounds each omitted AO by

\[
 |\chi_{lm}(r)|\le A_l r^l\sum_p\max_c|c_{pc}|e^{-\alpha_p r^2}.
\]

The radius is solved on the decreasing side of this envelope, beyond
`sqrt(l/(2 min(alpha)))`. Neighbor radius is at least the grid radius plus
the maximum AO envelope radius. A spatial hash constructs neighborhoods;
an atom-to-orbital inverted index identifies active coefficient columns.
Each atomic grid contributes bounded local Gram blocks to sparse DOI
accumulators. Mulliken populations use the same algebra on explicit local
coefficient supports. The AO envelope does **not** certify a total DOI
error: radial-grid, partition and coefficient truncations must be refined,
and resulting domains and gradients compared with the dense oracle.

With the same fixed coefficient maps, let `l_i` and `p_u` be screened
occupied/PAO columns. Screening generally makes `l_i^T S p_u` nonzero.
Simply truncating `l_i^T r p_u` would introduce an origin dependence. Use

\[
 n_i=l_i^T S l_i,\quad c_i^x=\frac{l_i^T r^x l_i}{n_i},\quad
 L_{iu}^x=l_i^T(r^x-c_i^x S)p_u,\quad d_i^x=L_i^x X_i.
\]

Here `X_i` canonicalizes the original local PAO overlap/Fock blocks.
Under `r^x -> r^x + a^x S`, `c_i^x -> c_i^x+a^x` and `L_i^x` is
unchanged. Full supports recover the original orthogonal transition
dipoles. A fixed-support differential is

\[
 dc_i^x=\frac{2\,dl_i^T(r^x-c_i^x S)l_i
        +l_i^T(dr^x-c_i^x dS)l_i}{n_i},
\]
\[
 dL_{iu}^x=dl_i^T(r^x-c_i^x S)p_u
 +l_i^T(r^x-c_i^x S)dp_u
 +l_i^T(dr^x-c_i^x dS-dc_i^x S)p_u.
\]

The reverse implementation includes every term, canonicalization/gap
response and the subsequent Boys/SCF relaxation. Its contractions use
only each LMO support and its domain PAO supports, with work counters for
the maximum AO/PAO dimensions and total AO-square-times-PAO work. Both
the parent and derivative use these centered local dipoles. This is a
numerical energy change, not merely a storage optimization.

For metric canonicalization, the metric eigenvectors are an intermediate
gauge. Differentiating them individually fails at a repeated eigenvalue.
The adjoint instead differentiates the symmetric inverse square root with
the divided difference

\[
 f^{[1]}(s_a,s_b)=\frac{s_a^{-1/2}-s_b^{-1/2}}{s_a-s_b},\qquad
 f^{[1]}(s,s)=-\tfrac12s^{-3/2}.
\]

A fixed orthogonal right frame keeps the primal convention; the final
Fock canonicalization removes its gauge. Actual retained/discarded
PNO/TNO degeneracies remain subject to the existing subspace checks.

At fixed radii, cutoffs, orbital supports and bounded spatial density,
spatial DOI, populations and local transition-dipole preparation have
linear work/storage. Discovering supports still scans dense reference
matrices and is timed separately. Building global PAO matrices, Boys/CPHF
response and relaxed nuclear contractions have not been made linear by
this option. Report these phases and measured support growth; do not
interpret conditional local bounds or isolated kernel fits as a completed
linear-scaling gradient validation.

```python
cc = LocalDLPNO(mf, triples='t1', dipole_tail_options={},
               local_screening_options={})
cc.kernel()
gradient = cc.nuc_grad_method().kernel()
print(cc.screening_diagnostics, cc.dipole_preparation_diagnostics)
```

## Selected-pair Boys adjoint and its full-equation check

Let the occupied transition dipoles be the real symmetric matrices
\(D^x=C_L^T r^x C_L\), and let \(d_i^x=D^x_{ii}\). For a lower-triangle
rotation variable \(X_{ij}\), \(i>j\), the localization constraint is

\[
 g_{ij}=2(d_i-d_j)\mathbin{\cdot}D_{ij},\qquad
 s_{ij}=(C_L^T\bar C_L-\bar C_L^T C_L)_{ij},\qquad H z=-s.
\]

The Hessian has no coupling between two rotation pairs with four distinct
endpoints. This sparsity follows by differentiating the diagonal centers
and transition dipoles under an occupied rotation; it does not require
discarding small dipole values. For a selected pair graph, define
\(\sigma_{ab}=+1\) when \(a>b\) and \(-1\) otherwise. The nonzero entries
of the exact projected PySCF Boys Hessian are

\[
 H_{ij,ij}=2|d_i-d_j|^2-8|D_{ij}|^2,
\]
\[
 H_{ij,\{i,k\}}=\sigma_{ik}
 \{D_{kj}\mathbin{\cdot}(d_j+d_k-2d_i)-4D_{ij}\mathbin{\cdot}D_{ik}\},
\]
\[
 H_{ij,\{j,k\}}=-\sigma_{jk}
 \{D_{ik}\mathbin{\cdot}(d_i+d_k-2d_j)-4D_{ij}\mathbin{\cdot}D_{jk}\}.
\]

The braces in a column label denote the corresponding lower-triangle
pair. These formulas include the nonstationary terms and are tested
column by column against the independent PySCF operator. A fixed-radius
spatial hash selects pairs, and incident-edge lists assemble CSR entries.
At bounded occupied density and fixed radius, maximum degree \(d\) is
bounded: variables are \(O(Nd)\), assembly/storage are \(O(Nd^2)\), and
MINRES costs \(O(kNd^2)\) for \(k\) iterations. An iteration bound is an
additional condition, not a consequence of sparse storage alone.

Restricting the response variables is an approximation even though the
projected Hessian is exact. The optional `boys_pair_radius` must pass both
the projected equation and the **original full equation**
\(\|H z+s\|_\infty\le10^{-8}\); omitted variables are expanded as zeros.
Failure raises `LocalBoysResidualError` and does not return a gradient.
A small projected residual alone is insufficient. The full residual is
an equation check, not a universal nuclear force-error certificate:
physical-source benchmarks additionally compare full nuclear gradients.

The localization Lagrangian has a sparse dipole density. For each retained
pair and multiplier, accumulate

\[
 B^x_{ii}\mathrel{+}=2z_{ij}D^x_{ij},\quad
 B^x_{jj}\mathrel{-}=2z_{ij}D^x_{ij},\quad
 B^x_{ij}=B^x_{ji}\mathrel{+}=z_{ij}(d_i^x-d_j^x).
\]

Thus \(d\mathcal L_{\rm Boys}=\sum_x\langle B^x,dD^x\rangle\),
\(\bar C_L^{\rm Boys}=2\sum_x r^x C_L B^x\), and
\(\bar r^x=C_L B^x C_L^T\). Zero trace of each \(B^x\) ensures that a
common change of dipole origin contributes zero. The selected-pair path
uses these analytic densities instead of a dense differential graph.

Current limitations are explicit: occupied dipoles and the source are
still produced by dense adapters, the full residual still uses the
original global operator, and the AO backtransforms, CPHF and orbital
nuclear contractions remain dense. This optional solver is a verified
building block, not a completed linear-scaling gradient implementation.
`benchmarks/dlpno_boys_locality.py` saves physical MP2/CCSD/(T) sources,
failed and accepted radii, force errors and timings. Its numeric NPZ
reference can be re-used by hash-verified manifest without rerunning the
correlation amplitude adjoints; original and current source hashes are
both recorded.

```python
gradient = cc.nuc_grad_method(boys_pair_radius=24.)  # bohr
forces = gradient.kernel()  # raises if an omitted equation fails
print(gradient.diagnostics['orbitals']['boys'])
```

The mutual tree also caches geometry-only Taylor coefficients across
quadrature batches within a single evaluation. Its default 64 MiB budget
counts numeric payload; Python container overhead is additional. Four
expanded sparse interaction matrices are rebuilt from compact coefficients
when needed, so they are not retained for every cell pair. Setting
`dipole_tail_options={'geometry_cache_bytes': 0}` disables the cache.
Displaced evaluations always create a fresh cache. Cache-on/off energy,
source derivatives, center derivatives and remainder bounds must be
identical at fixed batching; the optimization changes a constant factor,
not the asymptotic claim or force-accuracy requirements.

## Compact GPU overlap indexing

The local GPU triples driver also evaluates each requested pair/triplet
projection directly as `X_p.T @ S_pao[d_p,d_t] @ X_t`, gathering only
the two compact domains. It does not form `S_pao[:,d_t] @ X_t` over
the full AO universe for every triplet batch. Work and temporary storage
per coupling depend only on the local PAO/PNO/TNO sizes; a fixed byte
budget bounds the gathered temporary batches. The shared energy driver
retains its original full-AO projection as an independent algebraic
oracle. Tests compare masked irregular domains at both small and large
batch budgets and exercise complete T0/T1 energy and adjoint paths.

A retained overlap is `S_pq = X_p.T S_pao[d_p,d_q] X_q`. The batched
implementation embeds only the involved pair coefficients into their
PAO union. `overlap_plan.py` builds this union without an AO-universe
mask, maps involved pairs by local sorted lookup, and stores the flattened
PAO-to-union rows once per plan. The CUDA scatter reads these rows
directly. Neither planning a batch nor evaluating it allocates an index
array of length `nbf` or total pair count. The global PAO metric and the
initial ordering of registered couples remain separate costs. Keeping
pair sizes once per producer also avoids scanning all pairs again for
each small requested batch.

The replacement retains the energy code's vectorized workspace estimate
and pair ordering, including extra identity-metric rows used by open-shell
energy callers. Tests compare irregular batched products with independent
dense products and reuse the workspace in a different order. Testing these
extra rows does not extend the closed-shell nuclear-gradient scope to UHF.


### Post-adjoint recovery and batched contraction primitives (2026-10-20)

`LocalGradients(..., response_checkpoint='/path/response.npz')` saves the
relaxed S/F/dipole densities, local RI force and numerical RHF reference
before the final force contractions. Resume with
`LocalGradients.from_response_checkpoint(path).kernel()`; an optional
`response_backend='cpu'` or `'gpu'` selects the final contraction backend.
The checkpoint contains numerical NPZ/JSON data, with exclusive
publication, data hashes, physical settings and matching project Python
source. Existing checkpoints are preserved. Earlier runs without this
checkpoint cannot recover their lost post-adjoint state.

The recovery code and `ragged_adjoint.py` variable-size matrix-chain VJPs
passed 3 CPU tests and a 38-test A100 80 GB suite, including small complete
T0/T1 gradient parity and CPU/GPU restart. The raw evidence is in
`/pscratch/sd/w/weiliang/dlpno-gpu-gradient-20261018/response-and-batched-primitives-audit-v90.json`.
The chain primitives include repeated-block scatter, transposed and
selected-row operands, scalar-weight derivatives and regeneration of
uncached constant overlaps. The opt-in batched CCSD transpose described below
now uses these primitives. The reference-force memory change has additionally
passed the 104-atom test below. These component checks do not certify complete
104-atom T0/T1 gradients or a speedup.


The bounded auxiliary-density GEMM is now active in `GPUReferenceBackend`.
On 104 atoms / 826 AOs, three full-rank signed-density contractions
completed on A100 40 GB, with sampled single-GPU memory of 33.416 GiB.
The maximum difference from the captured unbounded oracle was
6.124532e-10 Eh/bohr. Changing the application pair-buffer target from
256 MiB to 64 MiB changed the asymmetric-system result by only
6.332712e-13 Eh/bohr. The three contraction calls took 7.40, 6.77 and
6.49 seconds, excluding preparation; this is a component timing, not a
full-gradient speedup. No density eigenvalues were removed. The full
104-atom T0/T1 gradient gates remain outstanding. Evidence:
`/pscratch/sd/w/weiliang/dlpno-gpu-gradient-20261018/reference-force-memory-audit-v88.json`.


### Batched GPU CCSD amplitude adjoint (2026-10-20)

Select `base.nuc_grad_method(ccsd_adjoint_backend='cupy', ...)` and set
`DLPNO_NGPU=1` to reuse the energy solver's batched matrix-chain plans
in each CCSD lambda iteration. The default remains the existing reverse
operator. The new operator includes both operands of every product,
T1-dressed intermediates, independent-pair symmetry, and fixed weak MP2
pairs. Only coefficients inside the amplitude linear solve are held fixed;
the complete outer residual pullback still supplies PNO, integral and
geometry derivatives. It does not freeze the orbitals or local domains
in a nuclear gradient.

DF buckets are replayed in bounded GPU batches, and uncached overlap
blocks are regenerated on demand. The operator is built when its lambda
solve begins, so its GPU storage does not overlap the earlier triples
primal. Dense occupied two-index intermediates still mirror the energy
implementation; this change does not prove an O(N) bound for all stages.

A100 40 GB job 60025888 passed all eight tests in
`tests/test_dlpno_ccsd_adjoint_gpu.py`: four random transpose/directional
checks with stored or regenerated overlaps and with strong or test-only
mixed weak pairs, plus four complete water/6-31g gradient checks (CCSD,
T0, T1, and mixed weak-pair CCSD). The largest random transpose difference
from the CPU reverse operator was 7.105428e-15, and the largest amplitude
finite-difference dot-identity error was 1.160938e-11. Complete nine-component
gradients differed by at most 2.248202e-15 Eh/bohr. Independent SCF/Boys/domain
recomputations at nuclear steps 1e-4 and 5e-5 bohr gave a maximum error of
5.557051e-9 Eh/bohr; those nuclear FD checks sample one component per case.
The largest CCSD lambda residual was 1.431373e-11. Response-checkpoint
restarts also passed. Evidence and original failure/retry provenance:
`/pscratch/sd/w/weiliang/dlpno-gpu-gradient-20261018/ccsd-adjoint-validation-v92.json`.

The test timings are small-system regression measurements with one CPU
thread; they are not an optimized CPU-node speedup comparison. Complete
104-atom T0/T1 validation remains a separate gate. Keep the producer's
source snapshot when restarting a response checkpoint: the loader requires
all project numerical-source hashes to match.


### Combined GPU CCSD response and T1 workspace lifecycle (2026-10-20)

The T1 Jacobi iteration now discards gathered overlap originals before
allocating residuals, computes squared residual norms in at most 128 MiB
chunks, and reuses each residual slice for the amplitude update. All
residuals are formed from the previous Jacobi state before any amplitude
is updated. Neither the equations nor selection/convergence thresholds
change. This exact memory change had already passed the 104-atom T1
energy and amplitude-transpose component checks in source-v82.

The combined source-v94 passed eight complete GPU gradient/restart tests
covering T0/T1, NumPy/CuPy outer reverse graphs, and both CCSD adjoint
backends. A complete 14-atom / 106-AO def2-SVP NormalPNO T1 calculation
then passed all 42 gradient components against the source-v82 reference:
maximum difference 2.653473e-9 Eh/bohr and energy difference 2.171486e-12 Eh.
The saved energy state confirms unequal PNO sizes and actual weak pairs,
and the post-adjoint numerical checkpoint passes its source, shape,
geometry, energy and binding checks.

For these two runs the CCSD lambda interval was 312.04 and 6.03 seconds;
the complete gradient took 756.98 and 443.91 seconds. Both used A100
40 GB nodes and one CPU thread, on different nodes. These are stage
measurements for this implementation, not an optimized CPU-node speedup
benchmark. The complete 104-atom T0/T1 gradient and independent FD gates
remain pending. The new large-system workflows retain unchanged physical
settings, stage hash-verified source on node-local disk, and save a response
checkpoint before the final force contractions. Evidence:
`/pscratch/sd/w/weiliang/dlpno-gpu-gradient-20261018/combined-c4-validation-v94.json`.


### Compact triples source replay (2026-10-20)

The default triples checkpoint now evaluates `triples_adjoint.triplet_source`.
It caches each pair-to-triplet overlap and projected amplitude only within a
single forward evaluation or replay. The local occupied-partner sum uses
matrix multiplication, sharing the projections across all six permutations,
as the upstream GPU energy implementation does. The uncheckpointed
`RefDLPNO._triplet_t0` equations remain unchanged as an independent oracle.

For the positive definite local RI metric $A$, the whitening products obey
$$(A^{-1/2}Q)^T(A^{-1/2}R)=Q^TA^{-1}R.$$
All unique occupied right-hand sides are packed into one solve $AU=Q$.
Its pullback is $\bar Q=A^{-T}\bar U$ and
$\bar A=-\bar Q U^T$ (with the symmetric part contributing for a symmetric
metric variation). Thus removing the explicit inverse square root retains
the metric derivative. Projection reuse likewise accumulates every use
into the original PNO, overlap and amplitude variables. No selection,
rank, convergence threshold or geometry derivative is frozen or omitted.
Repeated occupied labels and an empty partner domain are supported.

Job 60028373 passed 18 tests, including NumPy/CuPy source pullbacks,
repeated metric eigenvalues and occupied labels, original-equation
finite differences, complete water T0/T1 gradients, and eight GPU
integral/response/restart cases. The maximum local pullback difference
was 2.498002e-16; the maximum independently recomputed nuclear FD error
was 4.722767e-09 Eh/bohr at steps 1e-4 and 5e-5 bohr (one nuclear component
per method).

On the same A100 40 GB in sequential separate processes, the complete
14-atom / 106-AO NormalPNO def2-SVP T1 gradient took 485.47 seconds
with source-v94 and 306.75 seconds with source-v99. All 42 components
agreed within 5.285221e-10 Eh/bohr. Both processes used OMP=1 and
OpenBLAS=1; this is a single-trial comparison between hybrid gradient
implementations, not an optimized CPU-node speedup benchmark.

Evidence is retained in
`/pscratch/sd/w/weiliang/dlpno-gpu-gradient-20261018/triples-source-validation-v99.json`,
including original logs, source hashes, phase timings and independent FD
errors. The latest 104-atom T0/T1 workflows use immutable source-v99 and
require the separate source-v92 CCSD gradient/FD gate described below.
Complete 104-atom T0/T1 gradients plus independent unfrozen FD remain
outstanding. The global reference/Boys/CPHF costs retain their separate
scaling scope.

### 104-atom GPU CCSD lambda milestone (2026-10-20, 17:04 UTC)

Job 60039673 completed the GPU CCSD amplitude-adjoint solve on one A100
40 GB using immutable source-v92. The asymmetric 104-atom / 826-AO
def2-SVP NormalPNO system has 2,837 pairs and 524,059 independent CCSD
amplitudes. Its geometry, pair counts, domain sizes and physical thresholds
match the earlier complete CCSD reference.

| Quantity | Measured result |
| --- | ---: |
| CCSD lambda interval, including operator construction | 34.091970 s |
| GPU transpose construction | 9.360757 s |
| GMRES iterations | 17 |
| Final amplitude-adjoint residual | 1.568729e-11 |
| GPU residual replay difference from the outer residual | 8.199040e-14 |

The earlier source-v39 run used 8,284.263558 seconds for the CCSD lambda
interval. These are diagnostic stage timings on different allocated
nodes (A100 80 GB and 40 GB), both with OMP=1 and OpenBLAS=1. The new run
also includes short external nonblocking profiler observations. This
comparison does not establish a complete-gradient speedup or an optimized
CPU-node baseline. The CPU outer derivative graph is still required;
its forward construction in the new run took 1,944.865936 seconds.

The immutable milestone evidence is
`/pscratch/sd/w/weiliang/dlpno-gpu-gradient-20261018/ccsd-gpu-lambda-milestone-v104.json`.
Here `v104` labels the audit, not a new numerical source. It records the
source-v92 manifest hash, runtime, exact progress events, partial resource
observations and raw profiler hashes. Short Python-stack samples locate
PNO gather/transform work and later residual graph-record creation; they
do not measure whole-gradient time fractions or establish a native cause
such as garbage collection.

At this milestone, the complete nuclear gradient and independent FD
checks are still unfinished. The full CCSD gate additionally requires all
312 gradient components to agree with the prior complete gradient, plus
independent SCF/Boys/PNO recomputations at two nuclear coordinates and two
step sizes (1e-4 and 5e-5 bohr). Its final audit is
`c34-asymmetric-ccsd-validation-v95.json` in the same scratch directory.
Source-v99 T0/T1 jobs 60040358 and 60040368 depend on that gate. Successful
CCSD lambda convergence alone does not satisfy the complete 104-atom
CCSD, T0 or T1 nuclear-gradient acceptance requirements.


### GPU transpose lifetime during the density sweep (2026-10-20)

A consuming density sweep now releases the amplitude-only CCSD and triples transpose operators immediately after their successful lambda solve. The independent outer residual graph retains every coefficient, PNO, integral and orbital derivative. Reusable VJPs retain their operators; the density-sweep policy is restored even if a sweep raises. The mathematical residuals, thresholds and convergence criteria are unchanged.

Job 60045848 passed 18 tests. Direct weak-reference checks verified release of owned GPU arrays in the consuming case and retention through repeated VJPs in the reusable case. Complete small-molecule CCSD/T0/T1 gradients were independently checked at two nuclear steps; the maximum error across 12 sampled component/step combinations was 5.557050e-09 Eh/bohr.

On the same A100 40 GB, all 42 components of the complete 14-atom def2-SVP NormalPNO T1 gradient matched source-v99 within 2.147312e-09 Eh/bohr. Gradient times were 307.48 and 307.31 seconds, respectively (separate sequential processes, one trial, OMP=1/OpenBLAS=1, energy preparation excluded).

Live allocations reported by the default CuPy pool immediately after each solve were:

| Callback | Source-v99 | Source-v111 |
| --- | ---: | ---: |
| adjoint_done:triples | 150.478 MiB | 31.296 MiB |
| adjoint_done:ccsd | 393.725 MiB | 31.624 MiB |

These counters describe live default-pool allocations at stage boundaries, not total device memory or a full-run physical peak. Full 104-atom acceptance remains separate. Global Boys/CPHF and the existing dense occupied intermediates retain their documented scaling limitations. Evidence: `/pscratch/sd/w/weiliang/dlpno-gpu-gradient-20261018/adjoint-lifetime-validation-v111.json`.


### Complete 104-atom CCSD gradient on A100 40 GB (2026-10-20, 18:49 UTC)

The source-v92 analytic gradient has now finished all stages, including the
reference-force contraction and a durable post-adjoint checkpoint. On the
asymmetric 104-atom / 826-AO def2-SVP NormalPNO system, all 312 gradient
components match the prior complete source-v39 calculation within
1.962654e-09 Eh/bohr. The maximum energy difference is 7.105427e-15 Eh.
Translation and rotation residuals are 1.058044e-12 and 1.121300e-08,
respectively, under the existing audit definitions.

The full gradient took 7,712.012 seconds (128.53 minutes), versus
13,157.843 seconds (219.30 minutes) for the old reference. The two runs
used different A100 40 GB and 80 GB nodes, OMP=1/OpenBLAS=1, with external
profiling observations in the newer run. These are descriptive elapsed
timings, not a controlled speedup or an optimized CPU-node comparison.
Host high-water memory was 165.208 GiB; the sampled GPU peak was
39.166 GiB and is a sampled lower bound. Forward derivative-graph
construction still took 1,944.866 seconds, and the amplitude-adjoint plus
outer coefficient pullback stage took 5,603.639 seconds despite the
34.092-second GPU CCSD lambda solve.

The gradient process exited successfully. The following four-rank FD step
failed before launch because the enclosing Slurm step's CPU binding was
incompatible with the inner allocation; thus enclosing job 60039673 is
marked FAILED. Recovery job 60046723 runs only the independent FD checks,
using the unchanged source-v92 snapshot, validator, gradient checkpoint,
physical thresholds, two coordinates and two step sizes. The original
failure and incomplete audit are preserved. T0/T1 jobs 60040358 and
60040368 now depend on successful recovery and retain their full
scientific preflight checks. New-source CCSD FD acceptance and complete
104-atom T0/T1 acceptance are still pending at this timestamp.

The component differences, raw report hashes, source identity, resource
records and scheduler provenance are in
`/pscratch/sd/w/weiliang/dlpno-gpu-gradient-20261018/ccsd-complete-gradient-milestone-v112.json`.
The mutable final CCSD/FD audit remains
`c34-asymmetric-ccsd-validation-v95.json` in that directory.


### 104-atom CCSD independent FD acceptance (2026-10-20, 19:22 UTC)

The source-v92 A100 40 GB calculation now passes the complete CCSD acceptance gate. FD recovery job 60046723 completed all four independent calculations and the unchanged topology-rechecking auditor. Each displaced SCF, Boys localization, PAO/PNO space and correlated amplitude solution was recomputed. The four comparisons sample two nuclear components, not all 312 analytic components.

| Atom index | Cartesian index | Step (bohr) | Analytic minus FD (Eh/bohr) |
| --- | --- | ---: | ---: |
| 0 | 2 | 1.0e-04 | -1.718615086e-08 |
| 0 | 2 | 5.0e-05 | 2.715171585e-08 |
| 103 | 1 | 1.0e-04 | -1.069738149e-08 |
| 103 | 1 | 5.0e-05 | 9.766249304e-09 |

Indices are zero-based. All four points retain the same effective PAO/PNO selection topology as the central point. The maximum FD error is 2.715171585e-08 Eh/bohr; all 312 analytic components still agree with the prior full gradient within 1.962654e-09 Eh/bohr. The gradient timing and resource qualifications in the preceding section continue to apply.

The complete source/provenance audit is `/pscratch/sd/w/weiliang/dlpno-gpu-gradient-20261018/ccsd-40gb-full-validation-v115.json`. It includes a fresh checksum and topology audit of the saved FD points, exact FD errors, source identity, the response checkpoint, and scheduler records. The batch and all four scientific FD processes exited 0. Slurm retained a nonzero derived exit code solely for a separate failed CUDA-free metadata import; that inventory was subsequently obtained from saved topology JSON. The original failure is preserved.

Source-v111 T0/T1 jobs 60047027 and 60047034 are now eligible and retain full-gradient plus independent FD gates. Complete 104-atom T0/T1 acceptance remains outstanding.

### Shared CCSD PNO overlaps: small-system validation and large-system trial (2026-10-20)

The unpromoted source-v114 candidate shares exact ordered overlaps across CCSD pair replay checkpoints while retaining the PNO-coefficient and PAO-metric pullbacks. It passed 23 tests, including comparison to the uncheckpointed CCSD/T0/T1 equations and 12 independent nuclear FD combinations (maximum error 5.557050e-09 Eh/bohr).

On the same A100 40 GB allocation, separate source-v111 and source-v114 processes gave full 14-atom T1 gradient times of 308.0828 and 294.0362 seconds. The maximum change across all 42 components was 7.283707e-10 Eh/bohr. This is a 4.56% reduction in a single trial; host high-water memory increased from 9,466,432 to 9,545,488 KiB. It does not establish a general speedup or a whole-gradient memory reduction. Evidence: `/pscratch/sd/w/weiliang/dlpno-gpu-gradient-20261018/ccsd-shared-overlap-validation-v114.json`, SHA256 `297cca8585c2a1ec12ed34722bd344ecccfe35bdabeafcd3c63781786fc5ea83`.

Full 104-atom CCSD trial job 60048491 started on A100 40 GB node nid001001 using immutable source-v114 and pipeline-v116. It will compare all 312 components to the accepted source-v92 gradient, then recompute two nuclear coordinates at two FD step sizes with independent SCF/Boys/PAO/PNO/amplitude solutions. It records stage GPU allocation observations and sampled process resources; the comparison is across allocations and is not a controlled overlap-only timing benchmark. At launch the numerical workspace remains source-v111, and the source-v111 T0/T1 jobs remain queued. The candidate is not adopted before large-system assessment.

### Bounded GPU overlap source pullbacks: experimental source-v117 (2026-10-20)

The source-v117 candidate computes batches of `Y = L.T @ M @ R` on the GPU, including both PNO coefficient responses and the full PAO metric response. For output cotangent `B`, the pullbacks are `bar_L = M @ R @ B.T`, `bar_M = L @ B @ R.T`, and `bar_R = M.T @ L @ B`. The metric is not assumed symmetric. Shared source records and repeated PAO indices retain additive cotangents. Local metric blocks are reconstructed within each checkpoint, so they are not retained for every overlap between forward and reverse sweeps.

Forty tests passed on A100 40 GB, including every selected source group, repeated indices and shared inputs through consuming checkpoints, the uncheckpointed CCSD/T0/T1 equations, transpose lifetimes, and 12 independent nuclear FD combinations. Maximum primitive-source directional FD error was 1.544760e-10; maximum nuclear FD error was 5.557051e-09 Eh/bohr. The complete 14-atom T1 comparison passed all 42 components with maximum difference 9.746580e-10 Eh/bohr. Full audit: `/pscratch/sd/w/weiliang/dlpno-gpu-gradient-20261018/ccsd-gpu-overlap-validation-v117.json`, SHA256 `961f3b8e69ebc4496300ead57510f3a9a9ed4d5d2e49f7f249e9262a2dec793f`.

The first full-gradient wall times were 366.7952 seconds for source-v114 and 418.4678 seconds for source-v117. This does not demonstrate acceleration. The sums of the graph and amplitude-adjoint stage timers were 346.2554 and 346.2653 seconds, while checkpoint and report overhead differed substantially; a 175 MB energy checkpoint took 67.2 seconds to save in that allocation. Job 60049479 completed a node-local ABBA comparison: reference times were 293.5806 and 295.9948 seconds; candidate times were 293.2975 and 296.7134 seconds. The candidate mean was 0.074% slower, providing no demonstrated acceleration. All 42 gradient components agreed with the first reference within 2.785360e-9 Eh/bohr. The four runs used the same allocated GPU and CPU affinity, and each rebuilt its energy. All timed output stayed local to the node and was published only after its process exited. The job and every monitored calculation exited successfully; the 20 published artifacts were rehashed against the completed audit. Source-v117 remains experimental and is not promoted.

A metadata-only plan using the saved 104-atom topology gives 25,295 GPU batches, a maximum declared array-workspace envelope of 201 MiB, and 157.5 GiB of padded forward input arrays across the sweep, assuming an initially empty overlap cache. The enforced per-batch array budget is 256 MiB. These are planning quantities, not measured transfers, a physical peak-memory bound, or a scaling proof. The candidate remains isolated and unpromoted; the working numerical implementation remains source-v111. Full 104-atom T0/T1 gradients and independent FD acceptance are still outstanding.

### Shared PAO-union GPU overlap pullbacks: experimental source-v120 (2026-10-20)

The next candidate borrows the upstream energy `OverlapGen` organization: embed
involved pair coefficients in one bounded PAO union and transfer one common
metric block per batch. With `Y[p,q] = C[p].T @ M @ C[q]` and output sources
`B[p,q]`, accumulate `D[p] = sum_q C[q] @ B[p,q].T` and
`A[q] = sum_p C[p] @ B[p,q]`. The exact sources are
`bar_C = M @ D + M.T @ A` and `bar_M = sum_q A[q] @ C[q].T`.
The metric need not be symmetric. Additive embedding and gathered coefficient
sources preserve repeated PAO rows, shared coefficients and zero-rank spaces.
The complete overlap boundary is created before the first local CCSD residual
checkpoint, retaining all metric and coefficient dependencies during replay.

Source-v120 remains an immutable candidate. Job 60051199 completed all 73
tests, including 33 new shared-union source, checkpoint and workspace cases.
The largest primitive VJP error was 1.776357e-15 and the largest source
directional finite-difference error was 2.299946e-10. Complete water CCSD,
T0 and T1 gradients agreed with the uncheckpointed equations within
2.393919e-15 Eh/bohr. Twelve independent unfrozen nuclear FD checks had a
maximum error of 5.557049e-9 Eh/bohr.

On the same A100 80 GB and inherited CPU affinity, complete 14-atom T1
gradients took 295.4577 seconds for source-v114 and 289.2364 seconds for
source-v120, a 2.11% reduction in this single trial. All 42 components agreed
within 1.316478e-9 Eh/bohr. All timed output stayed on the node until each
process exited. Resource samples confirm the same physical GPU UUID; the
published files were rehashed after completion. Actual GPU overlap batches
fell from 216 in source-v117 to 27; this is not an overall scaling claim.
The small host-memory difference is not treated as an established saving.

Full 104-atom source-v120 CCSD validation is running as job 60052589 on an
A100 40 GB with pipeline-v122. It rebuilds the correlated state from the
predetermined SCF/Boys reference, computes all 312 analytic components, and
then runs four fresh unfrozen FD combinations using the unchanged validator.
The working numerical source remains source-v111 while this large test runs.
The source-v114 large CCSD trial and 104-atom T0/T1 queues continue separately;
no complete 104-atom T0/T1 gradient or FD acceptance is claimed.

A separate metadata-only plan from the saved 104-atom topology groups 525,211
ordered overlaps into 2,788 batches, versus 25,295 for source-v117. The largest
declared array-workspace envelope is 255.995 MiB. Padded forward input arrays
sum to 71.76 GiB, versus 157.47 GiB for the previous plan. These are array counts
from topology, not measured transfers, peak physical memory, elapsed speedup
or a scaling proof; replay, BLAS workspaces and allocator reserve are excluded.
The plan assumes an initially empty overlap cache. Evidence and submission
state are in `optimization-progress-v120.json`, `104-gpu-union-plan-v120.json`
and `ccsd-gpu-union-submission-v120.json` under the existing scratch run root.

### GPU local triplet source contractions: experimental source-v124 (2026-10-20)

With `triples_backend='cupy'`, source-v124 also evaluates local triples
integral transformations, the shared RI-metric solve, projected amplitudes,
and rank-three contractions on the GPU within each existing checkpoint.
Explicit transfer records return cotangents to the original backend; metric,
PNO/TNO, orbital-energy and amplitude responses remain differentiable. The
checkpoint outputs retain the outer backend, so the usual NumPy outer graph
does not keep one GPU workspace per triplet. CPU selection and triplet/pair
overlap construction remain, and no full-gradient linear-scaling claim follows.

Job 60053461 passed 92 tests, including 19 new source, transfer and lifetime
cases. The local GPU VJPs agreed with the original equations within
4.718448e-16, and independent source directional FD differed by at most
7.949718e-12. Twelve independent unfrozen nuclear FD checks on the small
molecular cases had a maximum error of 5.557050e-9 Eh/bohr. Keeping and replaying
24 host-output checkpoints left the measured live CuPy pool at zero bytes
after both stages; this checks retention, not physical peak GPU memory.

On the same A100 80 GB, the complete 14-atom T1 gradient took 289.6383 seconds
with source-v120 and 265.2646 seconds with source-v124, an 8.42% reduction in
one sequential comparison. All 42 components agreed within 2.538015e-10
Eh/bohr. Energy-graph time changed from 98.1017 to 90.6920 seconds and the
amplitude-adjoint phase from 187.7702 to 170.7579 seconds. Sampled device peaks
were 9,086 and 9,246 MiB; no memory saving is claimed. Timed outputs stayed
node-local, and the published artifacts and common physical GPU were checked.

A separate source-v120 profile, job 60053463 / pipeline-v125, passed the
full-gradient comparison. It shows remaining CPU tensor contractions,
indexed accumulation and repeated orbital selection work. Its cumulative
call times overlap and include synchronization; its profiled wall time is
not speedup evidence. An earlier profile launch (60053094 / v123) failed its
output-preservation guard before quantum chemistry because that guard matched
its own scheduler log. The corrected guard was checked and the replacement
completed successfully.

Source-v124 remains unpromoted pending large-system validation. The working
numerical source is source-v111; full 104-atom source-v114 and source-v120 CCSD
trials continue separately, and full 104-atom T0/T1 gradients plus independent
FD remain outstanding. Evidence is retained in
`gpu-triples-source-validation-v124.json`,
`gpu-triples-source-small-validation-v125.json`,
`c4-profile-audit-v125.json`, and `optimization-progress-v125.json`
under the existing scratch run root.

### Full104 follow-up for the validated GPU triples candidate (2026-10-20)

The source-v114 full 104-atom CCSD gradient (job 60048491, pipeline-v116)
finished in 6,830.9604 seconds (113.85 minutes). All 312 components agree with
the accepted source-v92 calculation within 2.546937e-9 Eh/bohr. Its four fresh
unfrozen FD combinations are still running. The local correlated energy took
308.9199 seconds; it excludes the predetermined SCF/Boys input. These separate
A100 40 GB runs are descriptive timings, not a controlled speedup measurement.
Host high-water memory was 178,067,716 KiB and the sampled device peak was
39,890 MiB. Full gradient plus FD acceptance remains a separate final gate.
Evidence: `ccsd-shared-overlap-complete-gradient-v126.json` in the scratch root.

Immutable source-v124 is now connected to full104 T0/T1 pipeline-v126, with
60 hashed control files and successful static checks for both methods.
Regular 16-hour jobs 60054959 (T0) and 60054983 (T1) depend on successful
completion of source-v120 CCSD job 60052589. Their execution preflight also
checks the actual complete 312-component parent gradient, four independent FD
combinations, topology, source, resource and response-checkpoint evidence.
The physical thresholds and FD validator are unchanged. The final auditors
require actual GPU overlap and local-triplet-source diagnostics in addition
to full gradients, invariance, durable response checkpoints and fresh FD.

Older regular queue positions are preserved until a newer allocation is ready
and all scientific prerequisites have passed; an already active older run is
preserved. These are submitted validation jobs, not completed T0/T1 results.
The numerical workspace remains source-v111 and source-v124 is unpromoted.
Submission provenance and remaining gates are in `optimization-progress-v126.json`.

### Adopted shared-overlap optimization after full104 CCSD acceptance (2026-10-20)

Job 60048491 completed with batch and derived exit codes both zero. The
source-v114 shared-overlap candidate now passes its full 104-atom CCSD gate:
all 312 analytic components agree with accepted source-v92 within
2.546937e-9 Eh/bohr, and four independently rebuilt nuclear FD combinations
(two coordinates, two step sizes) have maximum error 4.476719e-8 Eh/bohr.
The signed FD errors are +5.579043e-9, +6.715911e-9, +3.481360e-8 and
-4.476719e-8 Eh/bohr in rank order. The unchanged validator uses independently
evaluated legacy-GPU HF energies at each reoptimized displaced density; the
original SCF-energy FD errors are also preserved in the raw records.

The source snapshot, full-gradient/response/energy checkpoints, all twelve
saved FD point records, displaced orbital files, selection topologies and
resource reports were rehashed. This large-system evidence accompanies the
23 passed tests and full14 T1 comparison. The exact two-file source-v114
change has been adopted in the gradient worktree after verifying all 49
previous numerical files against source-v111. No approximation or threshold
has changed. The full104 gradient timing remains 113.85 minutes; its comparison
to the earlier 128.53-minute run spans separate nodes and is not a controlled
speedup measurement.

Evidence: `ccsd-shared-overlap-full-validation-v128.json` and
`ccsd-shared-overlap-promotion-v128.json` under the scratch run root.
The newer source-v120 GPU-union and source-v124 GPU-triplet candidates remain
unpromoted. Complete full104 T0/T1 gradients and their independent nuclear FD
acceptance are still outstanding.


### Actual104 GPU triplet-source checks and complete-gradient follow-up (2026-10-20)

Job 60056634 completed with both batch and derived exit codes zero. On the
same asymmetric 104-atom/826-AO input, source-v124 independently rebuilt T0
and T1 energies, then checked six actual local triplet checkpoints for each
method, including the largest spaces. T0's maximum TNO rank was 71 and T1's
was 57; both use 6,146 triplets. CPU/GPU forward, checkpoint replay and all
source pullbacks agree within 1.154746e-13 (T0) and 4.913900e-14 (T1).
These are local source derivatives, not nuclear-gradient FD errors.

The earlier diagnostic job 60055682 is preserved as failed. Its T0 diagnostic
incorrectly expected production T0 to retain TNOs; the corrected diagnostic
observes the actual final-cut producer without changing its return values.
Its T1 diagnostic compared unrestricted metric cotangents even though the RI
metric varies symmetrically. A separate CPU/GPU symmetric-metric directional
FD check passed with maximum error 1.084177e-11. The revised comparison tests
the symmetric metric cotangent and every other source directly, retains raw
antisymmetric discrepancies, and leaves production source-v124 unchanged.

For five larger selected T0 blocks, GPU time falls by 19.3-29.8%; for five
larger selected T1 blocks, it falls by 6.5-20.9%. The smallest selected block
is slower on GPU by 18.0% (T0) and 26.6% (T1). These are CPU/GPU ABBA local
checkpoint timings on the same A100 40 GB node with one BLAS thread. They
include replay and VJP but exclude RI production, upstream/downstream orbital
response and nuclear-force contraction. They do not establish full-gradient
speedup, optimized CPU-node performance or peak gradient memory.

Pipeline-v129 reuses the validated, hash-bound source-v124 energy checkpoints
and rebuilds the entire gradient graph; all displaced FD states are freshly
reoptimized. Static checks for both methods passed, and the gradient call and
FD physical settings match pipeline-v126. Ordinary four-hour interactive T0
job 60057562 is pending on source-v120 full104 CCSD/FD job 60052589. The latter
is still running at this update. The corresponding interactive T1 script is
prepared but not submitted because both interactive submission slots are in
use. Regular 16-hour T0/T1 jobs 60054959 and 60054983 remain queued. Whether
the complete T0/T1 gradient and FD fit four hours is not established.

Evidence: `local-triplet-probe-validation-v130.json`,
`full-gradient-preparation-v129.json`,
`triples-interactive-controller-t0-v129.json`, and
`optimization-progress-v131.json` under the scratch run root. The working
numerical source remains the accepted source-v114. Source-v120/v124 remain
candidates; full104 T0/T1 nuclear-gradient and FD acceptance is outstanding.


### Complete GPU-union CCSD gradient; final FD and triples handoff (2026-10-20)

Source-v120 job 60052589 completed the full104 analytic gradient in
5,928.4891 seconds (98.81 minutes), with maximum component differences
2.561259e-9 Eh/bohr against accepted source-v92 and 2.288998e-9 against
accepted source-v114. Translation and rotation residuals are 1.026195e-12
and 1.173183e-8 Eh/bohr, respectively. The host high-water mark is
176,108,952 KiB and the sampled GPU peak is 40,088 MiB. The late reference
force stage raises device usage above the earlier partial-stage readings;
no full-gradient GPU-memory reduction is claimed. The 113.85- and
98.81-minute runs used separate A100 40 GB nodes and are descriptive timings.

Four independent unfrozen FD jobs are now running within the same allocation.
The complete-gradient milestone is recorded in
`ccsd-gpu-union-complete-gradient-v135.json`; final acceptance and numerical
promotion remain pending. The supplemental auditor
`audit-ccsd-gpu-union-full-v132.py` requires terminal scheduler success,
rehashes the complete source and FD point/orbital evidence, and compares all
312 components against both accepted full104 references.

Interactive T0 request 60057562 never started: while its parent dependency was
unmet, salloc reported a connection timeout after six minutes and revoked the
allocation. Both the terminal scheduler record and the original controller
log are preserved. No gradient output was created. The revised T0 allocator
v134 uses the unchanged source-v124/pipeline-v129 computation and requests
resources only after parent CCSD/FD acceptance. Durable handoffs v133 (T1)
and v134 (T0) are live; T0 also waits for the T1 handoff before checking the
remaining interactive slot. Both preserve any already active older matching
run and the regular fallback queues. They submit at most one new allocation
per handoff and retain failures for inspection.

The worktree numerical source remains accepted source-v114. Full104 T0/T1
complete gradients and independent nuclear FD remain unverified. Evidence:
`triples-t0-retry-preparation-v134.json`, the v133/v134 handoff controller
records, and `optimization-progress-v135.json` under the scratch run root.


### Adopted GPU-union overlap response after full104 acceptance (2026-10-20)

Source-v120 is now adopted in the gradient worktree. The complete 104-atom,
826-AO CCSD calculation and all four independent nuclear FD combinations
passed; job 60052589 finished with batch and derived exit codes zero. Maximum
FD error is 7.946652e-8 Eh/bohr. All 312 analytic components agree with accepted
source-v114 within 2.288998e-9 and source-v92 within 2.561259e-9 Eh/bohr.
The signed FD errors, original SCF-energy FD errors and stable selection
topologies are preserved in the final audit.

The supplemental audit rehashed all 581 source files, energy/response
checkpoints and twelve FD point records, verified the eight displaced SCF
solutions and their exact requested coordinates, and checked the central
orbitals against the original converged portable reference. The maximum
reported displaced SCF gradient norm is 8.795686e-12. The initial supplemental
auditor incorrectly required a fresh SCF residual for the reused central
reference; its failure is retained. The corrected auditor explicitly verifies
the original reference residual and identical central MOs, while retaining
the displaced-SCF and FD tolerances. The production FD validator is unchanged.

The gradient takes 5,928.4891 seconds (98.81 minutes), excluding the correlated
energy and original SCF/Boys preparation. Actual GPU overlap execution uses
1,655 batches for 296,541 missing overlaps; the declared maximum array
workspace is 268,425,216 bytes. This bound excludes the rest of the gradient,
BLAS workspace and allocator reserve. Measured host high-water memory remains
176,108,952 KiB and the sampled GPU peak 40,088 MiB. Timings against previous
A100 nodes remain descriptive, not controlled speedup ratios.

The adopted five-file change is exactly the tested source-v120 snapshot:
73 tests, small independent FD checks, full14 T1 parity, and this full104 CCSD
acceptance. All 49 prior numerical files were checked before adoption and all
51 resulting numerical files match the candidate manifest. Evidence:
`ccsd-gpu-union-full-validation-v138.json`,
`ccsd-gpu-union-promotion-v138.json`, and the prepared five-file patch in the
scratch run root.

### Concurrent full104 T0/T1 validation on separate nodes (2026-10-20)

The ordinary interactive QoS permits four nodes per job and two submitted
jobs. Source-v124 T0/T1 validation now runs as one two-node allocation,
job 60059052, starting 2026-10-20 23:04:54 UTC with a four-hour limit. T0 runs
on nid003061 and T1 on nid003064. Before starting either gradient, the pipeline
ran four short GPU placement checks per method and verified four distinct
physical GPUs on the assigned node. All later FD steps are explicitly pinned
to that method's node. Each full gradient uses one GPU and each subsequent
four-direction/step FD group uses four. The numerical gradient runner,
physical settings and FD validator remain pipeline-v129's unchanged files.

Both methods loaded the validated same-source energy checkpoints successfully
(8.73 and 9.74 seconds) and are rebuilding complete gradient graphs. The
continuous response sources are active and every displaced FD state will be
reoptimized. Older regular queue positions remain available. The two earlier
single-method waiting controllers were stopped before they submitted any new
allocation; their replacement is recorded separately. Whether the full
T0/T1 gradient and FD stages fit four hours remains unproven.

Upstream commit 8b6c8d84bb036abac2c26da2ceea3da4d1681770 was inspected without
modification. Its new atom-major PNO-integral path requires the direct integral
provider and is not activated by the current LocalDFInts interface. The
current RI pullback already groups sources by auxiliary atom and takes
112.92 seconds in the full104 CCSD gradient; the preceding amplitude/source
pullback takes 4,291.47 seconds. Adapting the upstream scheduling to new local
PNO adjoints requires complete coefficient and metric response validation.
The source-v124 GPU-triplet candidate remains unpromoted while its full104
T0/T1 gradients and independent FD are running.

Evidence: `paired-triples-controls-v137.json`, the allocation and placement
records, `upstream-pno-atom-review-v136.json`, and
`optimization-progress-v138.json` under the scratch run root.


### Full104 T0 accepted after physical-key topology replay (2026-10-21)

Source-v124 completed the 104-atom, 826-AO T0 gradient in 12,089.8059 seconds
(201.50 minutes), excluding the reused energy and reference preparation.
Translation and rotation residuals are 1.68360e-12 and 9.68426e-9 Eh/bohr.
The host high-water mark is 184,050,176 KiB. The original four independent
FD combinations all had errors below 5.800802e-8 Eh/bohr, but their ordered
orthogonalization traces differed. Those failed reports remain unchanged.

The failure arose from adaptive GPU triplet chunk scheduling. Observer-v150
records each orthogonalization against its occupied triplet and PAO domain,
checks its selected columns and returned PAO/TNO ranks, and restores a
canonical triplet order for comparison. It preserves the original execution
order separately. Seven tests passed, including actual GPU energy runs with
different chunk sizes; their maximum energy difference is 1.22125e-15 Eh.
A negative test confirms that exchanging different selections between two
physical triplets is rejected even when the multiset of records is unchanged.

Job 60072671 completed twelve correlation replays with exit codes zero,
using the exact hashed SCF/Boys arrays from the original independently
optimized FD points: eight displaced states and four central states. Every
replay rebuilt all correlation domains and amplitudes. All 12,292 identified
triplet selections per point agree across the complete set. The largest
replayed energy difference is 5.32908e-15 Eh, and the original and replayed
FD derivatives give the same maximum error, 5.800802e-8 Eh/bohr. This accepts
the full 312-component analytic T0 gradient with independent FD sampling of
two components at two step sizes; it does not constitute 312-component FD.

The composite audit explicitly retains job 60059052's failed status: T1
exceeded its host-memory guard, while T0's legacy FD gate failed on record
ordering. T0 acceptance uses its completed gradient, the original orbital
and FD evidence, and the successful physical-key replay. The independent
audit rehashed all 582 source files and the corresponding checkpoints and
point records. Evidence under the scratch run root:
`keyed-topology-full104-review-v150.json` (SHA256
`51786bd4a0c7fc939fd14195db9b14e646f362d259eea1362aa8ff0deb780982`),
`keyed-topology-test-review-v150.json`, and
`review-keyed-topology-full104-v150.py`.

### GPU Krylov vectors and upstream host-memory reclamation (2026-10-21)

The source-v124 full104 T1 process stopped at the 218 GiB host RSS guard
while starting the triples adjoint. Its 353,233,905 amplitude variables
required an 8.48 GB restart-two Krylov basis that SciPy still allocated on
the host, although the transpose products ran on the GPU. Source-v147 moves
the GMRES vectors, preconditioning and orthogonalization to CuPy, checks the
true residual, and returns only the accepted solution to the host graph.
All physical thresholds and the complete coefficient response are retained.

Source-v147 passed 108 tests and twelve independent small-system nuclear FD
checks (maximum error 5.55705e-9 Eh/bohr). A same-GPU full14 T1 comparison
agreed within 4.27923e-10 Eh/bohr, with host peak memory reduced 24.59% and
runtime increased 4.25% in one trial. A 353,233,905-variable synthetic
capacity test with an additional 8 GiB resident GPU buffer also passed.
These results support the implementation and capacity; full104 T1 gradient
and nuclear FD acceptance remain pending. Job 60072791 uses source-v147
with a twelve-hour limit. The scheduler reports maintenance from
2026-10-21 13:00 UTC to 2026-10-28 06:00 UTC (queried with TZ=UTC). Diagnostic job 60074668 tests
the actual full104 T1 transpose with a constructed known-solution right-hand
side and saves a same-source converged energy checkpoint for the long run.

Upstream commit 54dbb5d16049216c81444c25cbfadc10771f4948 provided another
memory-management improvement: clear both CuPy and GPU4PySCF free pinned
blocks and invoke glibc malloc_trim at phase boundaries. Candidate-v152
applies this to coarse gradient stages and GPU adjoint solve boundaries.
It passed 112 tests and twelve independent nuclear FD checks. A separate
same-GPU full14 comparison against source-v147 agreed within 2.35360e-9
Eh/bohr; host high-water memory changed from 7,198,972 to 6,997,448 KiB
(2.80% reduction), while runtime changed from 261.1579 to 263.9960 seconds
(1.09% increase). This is a modest single-trial memory benefit. Source-v152
remains a candidate. The adopted worktree numerical files remain source-v120
until the larger triples optimization is ready for promotion.

Evidence: `full104-t1-host-guard-failure-20261021.json`,
`gpu-krylov-review-v147.json`, `large-krylov-capacity-review-v148.json`,
`host-trim-review-v152.json`, and the source-v151/v153 submission records.


### Full104 physical GPU Krylov capacity and stricter convergence (2026-10-21)

The accepted 104-atom CCSD/source120 and T0/source124 results above remain
unchanged. Full104 T1 is still unaccepted. A capacity test now uses the actual
6146 selected triplets, 353233905 amplitude variables, 206174 couplings and
279076892 overlap elements, with a constructed known-solution right-hand side.
This tests the physical transpose and solver, not the energy-adjoint RHS or
complete nuclear gradient.

Fresh-energy diagnostic153 (job 60074668) reproduced the accepted T1 energy
to 8.89e-16 Eh, built the GPU transpose in 42.25 s, then ran out of device
memory while allocating the initial GMRES residual. It retained 8 GiB of
additional GPU storage for capacity testing. Immediately before the solver,
the default pool held 16.13 GB with only 2.94 GB in use. The installed
GPU4PySCF allocator bypasses that pool for allocations above 100 MiB; its
large CUDA allocations do not automatically reclaim unused pool blocks.
The failure and original artifacts are preserved in
`physical-triplet-krylov-failure-review-v153.json` under the scratch campaign.

Diagnostic154 (job 60075403) loaded the strictly hash-matched source147 energy
checkpoint and flushed inactive default-pool blocks before solving. It ran
17 matrix actions in 34.80 s, with sampled device memory 30674 MiB, but
**failed** the existing maximum-residual acceptance check: the L2 residual
was 4.89e-6 (below the relative tolerance 1.21e-5), while its largest component
was 1.27e-7 (above 1e-8). Only 199865344 bytes were reclaimed in this cold
checkpoint run. Its different allocator history means this result does not
establish that flushing alone fixes the fresh-energy OOM. The separate
failure review is `physical-triplet-krylov-failure-review-v154.json`.

Candidate156, isolated in scratch with source147 as its parent, reclaims
inactive default-pool memory before the triples RHS/workspace allocations.
It also adds an optional absolute maximum residual bound to GPU GMRES; the
triples adjoint requires **both** its original relative L2 condition and the
existing 1e-8 component bound before returning a solution. Scientific
thresholds, operator coefficients and the outer acceptance bound are unchanged.
Seven new independent solve and failure cases exercise refinement after norm
convergence, the early-zero case, exhausted cycles and invalid bounds.

Job 60075720 runs 115 regression cases, 12 independent nuclear FD comparisons
and a complete same-GPU 14-atom T1 comparison. Job 60075754 rebuilds the
104-atom energy from scratch and tests the physical transpose with the same
8 GiB extra residency and both convergence conditions (diagnostic157).
These validations are pending; neither candidate156 nor T1 is accepted.
The workspace numerical files remain source120. Source155 is an untested
cache-only intermediate snapshot and is not submitted or adopted. The
previous source147 full-gradient job 60072791 remains pending; it has not
been relabeled as the newer candidate.


### Separate lifetime of persistent GPU coefficients (2026-10-21)

Source156 passed all 115 regression cases, 12 independent nuclear FD comparisons
(maximum error 5.5571e-9 Eh/bohr), and the complete 14-atom T1 comparison.
The maximum change over 42 gradient components was 1.4071e-9 Eh/bohr;
same-GPU gradient times were 260.605 s and 261.120 s, a single trial with no
observed speedup. The independent review is `gpu-pool-review-v156.json`.

Fresh-energy diagnostic157 (job 60075754) then reproduced GPU OOM before any
GMRES matrix action. Calling `free_all_blocks()` released zero bytes: the
default pool still held 17226612224 bytes with 2940310016 bytes in use.
The unused capacity was about 13.3 GiB. Its complete failure review is
`physical-triplet-krylov-failure-review-v157.json`; the failed run and valid
energy checkpoint are preserved. This is consistent with CuPy's documented
restriction that split blocks cannot be returned until their pieces merge
([MemoryPool documentation](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.MemoryPool.html)).

Source160 keeps the source156 convergence fix and places persistent triples
transpose coefficients in a dedicated CuPy memory pool. Its release method
also releases that owner. The operator formulas, numerical thresholds and
batching are unchanged. This separates the coefficients' lifetime from the
large temporary allocations left by the energy calculation; the full fresh
capacity test remains necessary to establish the benefit.

The 117-case regression passed, including two new allocation/ownership cases
with NumPy and CuPy coefficients. In each, only the denominator consumed the
caller's pool, the actions agreed with the independent NumPy transpose to
8.89e-16, live caller data survived, and all owned coefficient storage was
released. The 12 nuclear FD comparisons passed with maximum error 5.5571e-9
Eh/bohr. Job 60076712 completed successfully: the complete 14-atom T1
comparison changed the 42 gradient components by at most 1.0845e-9 Eh/bohr.
The same-GPU gradient times were 263.359 s and 263.148 s; this single trial
does not establish a speedup. The independent review is
`gpu-isolated-pool-review-v160.json`.

Fresh-energy full104 diagnostic161 (job 60076742) failed with GPU OOM before
its first GMRES matrix action. The dedicated coefficient pool correctly
owned 2939774464 bytes, but the default pool still reserved 19654533632 bytes
with only 545280 bytes live. Explicit pool release returned zero bytes.
The remaining small allocations' owners have not yet been identified;
coefficient pool isolation alone did not fix capacity. The preserved failure
review is `physical-triplet-krylov-failure-review-v161.json`. This diagnostic
uses the actual full104 operator with a known-solution right-hand side and
8 GiB extra residency; it is not a complete nuclear gradient. Source160 is
not adopted, and full104 T1 remains unaccepted.

Unsubmitted source156 pipelines158/159 are superseded preparations; their
controls were never frozen because diagnostic157 failed. Successor full
source160 gradient162 and raw FD point163 drivers are prepared but are also
unsubmitted because diagnostic161 failed its capacity gate. FD163 splits the same two
coordinates and two step sizes into 12 raw points on eight GPU workers,
with at most two energy evaluations per worker. Each displaced SCF/Boys
solution and correlation calculation is independent. Raw-point completion
explicitly leaves topology comparison and gradient acceptance incomplete;
those checks require the saved central/plus/minus records and the eventual
complete analytic gradient. No finite-difference acceptance is inferred
from successful energy jobs alone.


### Persistent batch-address cache identified (2026-10-21)

Controlled full104 diagnostic164 (job 60077725) identified the remaining
allocator owner through CuPy allocation/free hooks. After energy teardown,
1,067 arrays in `vbatch._BASE` occupied 546304 rounded bytes. Together with
one 1024-byte integral-scheme array, these accounted for every live byte in
the default pool. Clearing only `_BASE` reduced the pool reservation from
19111775744 to 81100800 bytes, releasing about 17.72 GiB. The allocation
stacks identify `vbatch._base` as the source of the cached pointer arrays.

The unchanged actual full104 T1 transpose then completed the GPU solve with
353233905 variables and the same 8 GiB artificial extra residency. The
independently recomputed maximum residual was 8.2206e-10 and the maximum
known-solution error was 3.0176e-10. The independent provenance and ownership
review is `physical-triplet-krylov-review-v164.json`. This establishes the
cache-retention mechanism and actual-operator capacity after controlled
cleanup; the diagnostic does not contain the complete nuclear-gradient
coefficient graph or its physical energy-adjoint right-hand side.

Source165 changes only the address-cache allocator relative to source160:
its pointer tables use a dedicated CuPy pool while cache keys, eviction,
contractions, and numerical thresholds remain unchanged. Three new tests
verify caller-pool release, allocator restoration, cache reuse, replacement
of operand storage, and cache eviction against independent NumPy matrix
products. All 120 regression cases and 12 independent nuclear FD comparisons
passed (maximum error 5.5571e-9 Eh/bohr). The complete 14-atom T1 comparison
changed the 42 components by at most 1.0715e-9 Eh/bohr. Same-GPU gradient times
were 263.287 s and 264.056 s; this does not establish a speedup. The review is
`gpu-address-pool-review-v165.json`.

Fresh-energy source165 diagnostic166 (job 60077923) completed successfully
without manually clearing the address cache. Its 1,048 cached pointer arrays
occupied 536576 bytes in their dedicated pool; the default pool retained
81100800 bytes after energy teardown. With the actual full104 operator and
8 GiB extra residency, the 353233905-variable GPU solve took 39.966 s. The
independently recomputed maximum residual was 8.2206e-10 and the maximum
known-solution error was 3.0176e-10. The sampled process GPU peak was 36100 MiB
on an A100 40 GB. The independent review is
`physical-triplet-krylov-review-v166.json`. This validates the production
allocator fix and actual-operator capacity. The right-hand side is constructed
from a known solution; the complete nuclear-gradient graph and physical
energy-adjoint right-hand side still require the full calculation.

The raw FD integration smoke169 (job 60078427) completed all six small-system
energy points, but its strict keyed topology comparison failed. Both FD
errors were below 1.65e-8 Eh/bohr. Every differing selection retained exactly
89 independent PAOs, the complete small-system virtual dimension; only the
chosen spanning columns differed. The existing effective topology remained
stable. This failure is preserved in `raw-fd-smoke-failure-analysis-v169.json`,
without relabeling it as passed or relaxing the full104 gate. The full104
system has 689 global virtual dimensions and reference triplet PAO domains
of at most 288, so this complete-span basis-choice case does not apply there.

Complete source165 T1 gradient pipeline167 (job 60078922) started at
2026-10-21 06:51:47 UTC on nid003776, an A100 40 GB node, with a six-hour
allocation ending at 12:51:47 UTC. Both the full14 comparison and diagnostic166
passed before submission. The controller verified and reused the source165
energy checkpoint from diagnostic166 and rebuilds the entire gradient graph;
response-checkpoint restart is not used. The superseded pending source147
job 60072791 was canceled after replacement submission. Scheduling eligibility
was broadened to A100 40/80 GB before allocation; the actual assigned GPU is
40 GB. A contemplated shorter minimum time was not applied because the job
had already started. Full104 T1 gradient acceptance remains pending.

Independent full104 raw FD pipeline168 (job 60078994) completed all 12 points
on eight GPUs across two nodes, from 06:17:22 to 06:47:07 UTC. The independent
`full104-t1-fd-points-review-v168.json` audit passed source/control/checkpoint
hashes, saved orbitals, SCF/Boys convergence, all point resources, and strict
physical-triplet topology. All 12 effective and keyed topologies agree;
native execution order has three variants. No complete-virtual-span exception
was used. There are four central correlation repeats and eight displaced
points with independently optimized SCF/Boys orbitals and rebuilt PNO/TNO
spaces and amplitudes.

The two FD steps, 1e-4 and 5e-5 bohr, give respectively -0.003063385065616 and
-0.003063394160563 Eh/bohr for component (0,2), and -0.000965964090938 and
-0.000965940216702 Eh/bohr for (103,1). As in the earlier validated protocol,
these slopes use the independent HF energy evaluation plus the unchanged
DLPNO correlation energy. Original GPU-SCF total energies and their FD slopes
are preserved; their largest difference from the independently evaluated HF
slopes is about 3.51e-6 Eh/bohr. The final gradient comparison reports both.
Raw energy/topology acceptance alone does not establish gradient agreement.
Source165 is not yet promoted; the numerical workspace remains source120.


### Exact full-envelope RI reads (2026-10-21)

Source170 is a separate candidate based on source165. The measured source124
profile146 attributed 25.79 s of the small-system reverse scatter time to
three-index RI gathers. When every requested auxiliary row and both orbital
column lists exactly match an existing shared source, source170 now omits
that advanced-index gather. It preserves the existing concatenation and
reshape, which create a fresh returned handle. Reordered, repeated, partial,
and scalar selections retain their original behavior. No integral, domain,
PAO/PNO/TNO, amplitude or response threshold changes.

All 137 regression cases passed. Twelve new cases compare independent
block-based derivatives against shared-source derivatives for qia/qij/qab,
NumPy/CuPy, and ordinary/checkpointed consuming sweeps. They include reordered
auxiliary atoms, repeated subset indices, scalar indices and shared leaves
revisited by later checkpoints. Maximum RI force and orbital-source errors
were 3.34e-16 and 4.84e-13 respectively. The existing 12 independent nuclear
FD comparisons passed with maximum error 5.5571e-9 Eh/bohr.

Job 60079470 performed four complete 14-atom T1 gradients in ABBA order,
separate processes on the same allocated A100 GPU, with node-local timed IO
and OMP/OpenBLAS threads both set to one:

| Run | Source | Gradient seconds |
| --- | --- | ---: |
| Reference 0 | 165 | 263.457 |
| Candidate 0 | 170 | 234.688 |
| Candidate 1 | 170 | 235.392 |
| Reference 1 | 165 | 264.766 |

The mean time decreased from 264.111 to 235.040 s, or 11.01%. Both paired
comparisons improved. The maximum gradient difference over all 42 components
and four runs was 2.0722e-9 Eh/bohr; the maximum energy difference was
3.011e-12 Eh. These timings exclude independently rebuilt energies and apply
to the measured 14-atom configuration, not to the full104 workload.

The original job exited with a postprocessing assertion after all 137 tests
and all four gradient subprocesses succeeded. The performance auditor had
incorrectly required identity-path hits for all three RI kinds. This full14
workload has 4,896 qij reads but no full-source qij selections, so the unchanged
fallback is correct; qia and qab do hit the fast path. All four independent
qij identity derivative tests passed. Audit/review171 corrects only this
activation expectation, retaining the complete-gradient, source/control,
physical-GPU, residual and FD gates. The original auditor, failed Slurm status,
logs and outputs are preserved. `ri-identity-gate-diagnosis-v171.json` records
the diagnosis and `ri-identity-review-v171.json` independently verifies the
numerical and repeated timing evidence. No numerical source was changed by
this audit correction.

Source170 is not promoted or validated on full104. The running full104 T1
job remains immutable source165, and the numerical workspace remains
source120. Full104 T1 acceptance still requires the complete analytic
gradient and comparison with the audited source165 FD points.


### Full104 RI candidate capacity and gradient submission (2026-10-21)

Source170 diagnostic174 (job 60080531) completed on an A100 40 GB GPU at
07:26:58 UTC. Independent review174 rehashed all 586 source files, frozen
controls and the fresh energy checkpoint. The actual 353,233,905-variable
T1 transpose operator, with 8 GiB additional GPU residency, solved a
known-solution right-hand side in 40.129 s. The maximum solution error was
3.018e-10 and recomputed residual maximum was 8.221e-10. Sampled GPU usage
peaked at 36,372 MiB. Fresh source170 energy took 698.599 s and agreed with
the source124 reference within 4.441e-15 Eh. This checks the actual operator's
capacity and arithmetic; it does not establish a complete nuclear gradient
or its full memory footprint. The review is
`physical-triplet-krylov-review-v174.json`, SHA256
`b9cd10db6f96d0438a2d99ac80d1193789f57d01b13469d0feffdc3bc16abc1a`.

After that review passed, pipeline172 (job 60081200) was submitted for the
complete source170 T1 gradient, reusing only the same-source energy checkpoint
and rebuilding the complete derivative graph. Independent FD pipeline173
(job 60081206) assigns its 12 points to 12 GPUs across three nodes; all eight
displaced points reoptimize SCF/Boys orbitals and rebuild correlated spaces.
The FD numerical validator is byte-identical to validated pipeline168.
Controls172/173 freeze both submissions and the future complete-gradient
and FD comparison. At 07:33 UTC, FD173 was running, gradient172 was pending
with a maintenance reservation reason, and reference source165 gradient167
(job 60078922) remained running. Pipeline172 requests six hours with a
four-hour backfill minimum; start and completion before maintenance are
not guaranteed. The reference job was neither changed nor canceled.

Acceptance still requires the complete 312-component gradient, residual
checks and independent FD agreement. When both gradients finish,
review175 additionally compares source165 and source170. The 11.01% timing
reduction above is still restricted to the controlled 14-atom comparison;
no full104 RI speedup is established. The workspace's 52 numerical Python
files were reverified against source120, with no candidate promotion.

The pending pipeline172 request was adjusted at 07:36 UTC from six to five
hours, retaining the four-hour minimum, so its maximum fits before the
13:00 UTC maintenance reservation. Slurm accepted the change and cleared
the maintenance-conflict reason; allocation is still required. The immutable
submission and control files retain the original request, and
`full104-walltime-adjustment-v172.json` records both scheduler states.

`source170-workspace-promotion-plan.json` and its corresponding patch collect
all ten source/test changes relative to source120. Every affected workspace
file matched its original hash, and `git apply --check` passed. The patch
has not been applied; its plan requires complete full104 T1 and independent
FD acceptance followed by another workspace hash check before integration.


Pipeline172 started at 07:44:41 UTC on nid002416, an A100 40 GB node,
with the five-hour allocation ending at 12:44:41 UTC. It loaded only the
verified source170 energy checkpoint and began rebuilding the derivative
graph. Source165 pipeline167 remains a separate running reference.

Independent source170 FD173 completed at 07:46:12 UTC (13 min 17 s).
Review173 passed all 12 raw-point source/orbital/resource checks and strict
physical-triplet topology comparisons, including the eight independently
reoptimized displaced SCF/Boys references. No full-virtual-span exception
was used. The review SHA256 is
`7ad9cd6f9abc184c88fca5f004997e9bba950bc6a5d6d0a1621a01b627cd9f39`.
The reviewed finite-difference slopes are:

| Component (zero based) | Step (bohr) | Source170 FD (Eh/bohr) |
| --- | ---: | ---: |
| (0,2) | 1e-4 | -0.003063398708036 |
| (0,2) | 5e-5 | -0.003063448730245 |
| (103,1) | 1e-4 | -0.000965962954069 |
| (103,1) | 5e-5 | -0.000965976596490 |

Cross-source review `full104-t1-fd-source165-source170-comparison.json`
confirms matching geometry, configuration and all 12 physical topologies.
The independently evaluated total energies differ by at most 5.457e-12 Eh;
the FD slopes differ by at most 5.457e-8 Eh/bohr. Original GPU-SCF slopes
remain recorded; for source170 they differ from the independent-HF slopes
by up to 3.602e-6 Eh/bohr. These are comparisons of two numerical FD
references, not errors against an analytic gradient. Both complete T1
gradients were still running at 07:51 UTC; nuclear-gradient acceptance
and candidate integration remain outstanding.


Completion observer176 started at 08:08 UTC for the existing source165 and
source170 jobs. On successful terminal completion it invokes each frozen
FD comparison (168/173), followed by the complete 312-component cross-source
review175 when both pass. It uses a single-owner lock, preserves existing
audit outputs, follows the latest Slurm restart attempt, and treats transient
observation failures or its observation deadline as neither calculation
failure nor permission to restart. It performs metadata audits only;
candidate installation remains a separate acceptance step. The launch
record is `full104-t1-validation-observer-launch-v176.json`.

At 08:09 UTC, source165 had completed the GPU triples primal solve in
27.713 seconds and was still constructing the derivative graph. Both
complete gradients remained running without a resource-guard event.
This forward solve is not the triples adjoint or a complete nuclear gradient.

Read-only upstream review at commit `8f4c6ae` identified a new live-memory
check before retaining regenerated CCSD overlap chunks: driver-free plus
inactive-default-pool memory must leave more than one quarter of total GPU
memory after the candidate chunk. Source170 still uses its plan-time budget,
and its uncached overlap allocator predates upstream's reusable workspace.
A future port therefore requires both residual and repeated-adjoint checks;
the new commit's own reported validation has not been independently rerun.
The current frozen hundred-atom campaign does not incorporate that change.
The analysis is preserved in
`upstream-live-overlap-headroom-review-20261021.json`.
