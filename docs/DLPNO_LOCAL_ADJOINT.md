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
                  lmo_cut=1e-6, pao_cut=1e-6,
                  E_CONVERGENCE=1e-11, R_CONVERGENCE=1e-10,
                  MAXITER=150, T_CUT_ITER=0.)
cc.kernel()
grad = cc.nuc_grad_method(backend='numpy', response_backend='gpu',
                         triples_backend='cupy', checkpoint_triples=True,
                         ccsd_primal_backend='gpu', triples_primal_backend='gpu',
                         checkpoint_ccsd=True, checkpoint_pnos=True,
                         krylov_memory_mib=8192)
de = grad.kernel()
```

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
remain distinct branches. Smaller finite-difference steps are required
when a displacement crosses one of those selection thresholds.

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

The host GMRES basis has a separate `krylov_memory_mib` target (8192 MiB
by default). Its restart length is at most 40 and is reduced for large
amplitude vectors; at least two directions are retained when the problem
has that many variables. The reported `krylov_basis_bytes` includes the
extra Arnoldi vector. This target covers the basis, not the whole process,
and can be exceeded by the minimum two-direction basis. The residual
tolerances and final independent residual check are unchanged. A nuclear
tangent regression also forces restart length two through all three
MP2, CCSD and T1 adjoint solves.

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

Overlapping PAO/PAO primitive requests share auxiliary-atom leaves using
the parent energy's observed local column envelopes. Triplet slices are
then internal gather operations that the checkpoint can discard and
rebuild. Their cotangents accumulate on the shared leaves before the RI
force contraction. Requests outside an observed envelope keep the exact
independent-block fallback; no new truncation is introduced.

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
These diagnostics do not bypass convergence or topology acceptance gates.

`benchmarks/dlpno_resource_monitor.py` additionally samples the calculation
process's device memory through `nvidia-smi`, and its host high-water mark
through `/proc`. The reported GPU maximum is explicitly a sampled lower
bound, and includes reserved allocator memory.

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


### Candidate GPU CCSD amplitude transpose and response recovery

The isolated v91 candidate adds `ccsd_adjoint_backend='cupy'` to
`LocalGradients` and the scaling benchmark. The default retains the
existing tensor reverse sweep. This candidate requires one visible GPU
(`DLPNO_NGPU=1`). Its correctness and hundred-atom performance are pending
validation; no measured speedup is asserted here.

The amplitude Jacobian uses the energy driver's variable-size batched
GEMM plans. For `Y += w A B C`, the transpose includes `A.T @ Ybar @ C.T`
for the middle operand and the Frobenius contraction for `wbar`.
Repeated block indices scatter-add; the independent amplitude expansion
includes both `ij/ji` transposition and the symmetric diagonal-pair map.
All T1-dressed intermediates participate. Only the amplitude operator has
fixed coefficients: the original outer residual still supplies all
integral, weak-pair MP2, PNO and orbital sources. DF bucket graphs are
recomputed in bounded groups, and uncached overlaps are regenerated for
the transpose rather than retained by a Python closure for every term.

`response_checkpoint='/path/response.npz'` saves the fully relaxed
`s/f/r` densities, local RI contribution, numerical RHF reference and
physical settings after the orbital adjoint, before final force
contractions. Data and metadata are non-pickle NPZ/JSON with exclusive
publication, SHA256 verification and matching project Python source.
`LocalGradients.from_response_checkpoint(path).kernel()` restores only
those final contractions. `response_backend='cpu'` or `'gpu'` may be
selected on restart. Existing checkpoints are preserved. This mechanism
cannot recover densities from earlier runs which did not save them.

Required candidate evidence includes independent NumPy chain derivatives,
random-vector CCSD transpose parity, amplitude-direction identities,
small-system CCSD/T0/T1 nuclear gradients and two unfrozen nuclear finite
differences. A test-only diagonal-CC/off-diagonal-MP2 case checks weak-pair
source propagation. The production thresholds are unchanged. The v88
reference-force candidate additionally requires a full-rank 104-atom
signed-density contraction on A100 40 GB against the captured unbounded
oracle. These component tests do not replace the complete 104-atom
T0/T1 gradient and independent finite-difference acceptance gates.
