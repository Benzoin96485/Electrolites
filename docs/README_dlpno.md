# DLPNO-CCSD(T) on the GPU (`electrolites/dlpno`)

GPU4PySCF has no local correlation method.  This package adds DLPNO-CCSD
with semicanonical (T0) and iterative (T) triples, for closed shells and for
high-spin open shells, at the Loose / Normal / Tight (and VeryTight) PNO
presets:

```python
import electrolites
electrolites.enable('dlpno')
mf = gpu4pyscf.scf.RHF(mol).run()            # or ROHF; UHF -> QRO orbitals
dl = mf.DLPNO_CCSD_T(pno='normal', triples='t1')   # 't0', 't1' or None
e_tot = dl.kernel(); dl.e                      # components (lmp2, ccsd, t0, t, ...)
```

Direct drivers: `electrolites.dlpno.rhf.DLPNO` (closed shell),
`electrolites.dlpno.uhf.UDLPNO` (open shell); any threshold can be passed by
keyword (`T_CUT_PNO=1e-8`, ...).

## Algorithm

* **Closed shell** -- Psi4 1.11's DLPNO-CCSD(T) (Jiang, Sylvetsky,
  DePrince, Valeev et al., JCP 161, 082502 (2024)): T1-dressed CCSD with the
  PNO projections placed as in that paper, trace / energy PNO completeness
  criteria, semicanonical (T0) and the iterative (T) of Guo et al. (JCP 148,
  011101 (2018)), sparse maps of Riplinger 2016.  `ref_rhf.RefDLPNO` is a
  host transliteration of Psi4; the GPU driver reproduces it to round-off.
* **Open shell** -- ROHF reference (or the quasi-restricted orbitals of a
  UHF, as ORCA): one set of PNOs per spatial pair from the spin-averaged
  pair density, alpha virtuals = PNOs, beta virtuals = PNOs + SOMOs,
  spin-orbital T1-dressed CCSD solved per spin block, with the
  closed-shell projection placement carried over (`ref_uhf2`).  The diagonal
  pair of a SOMO carries only its singles: its space is the natural-orbital
  space of the averaged density of the SOMO's pairs.  (T0) / iterative (T)
  in spin-orbital form over the triplet natural orbitals, one aligned
  virtual block per spin triple.  In the canonical limit (all thresholds
  off) the code reproduces PySCF UCCSD(T) on semicanonical ROHF orbitals to
  1e-10 Eh, and for a closed-shell molecule it reproduces the closed-shell
  code (`tests/test_dlpno.py`).

## GPU implementation

| step | how |
|---|---|
| LMOs | PySCF Boys/CIAH with device dipole tensors (`electrolites.lo_gpu`) |
| DOI, dipole pair estimates | device; the dipole domains orthocanonicalised in one batch |
| 3-index integrals | GPU4PySCF `int3c2e_bdiv` per aux-atom group; (Q|uv) only on extended per-atom PAO domains |
| SC-MP2, LMP2 PNOs, PNO spaces | pairs batched by padded PAO-domain size: one Cholesky, pivoted-Cholesky PAO selection (CUDA kernel), `cusolverDnXsyevBatched` for every eigenproblem |
| LMP2 iterations | variable-size batched GEMM chains (`vbatch.VChainPlan`, one CUDA launch per chain step) |
| CC integrals | batched over pairs; open shell: blocks per spin, no dense spin-orbital DF tensor |
| LCCSD residual | every projected term is a static list of GEMM chains replayed per iteration; DF dressing of strong pairs batched in coarse size buckets; 3-external contractions as batched GEMMs; persistent work buffers so no host<->device synchronisation inside an iteration |
| (T0) / (T) | triplets batched by padded TNO size; 3-external triples tensors accumulated per auxiliary atom without forming the fitted (Q|ab); fused CUDA kernel for the six-permutation energy; iterative (T) as batched three-mode transformations |

Memory (what grows with system size, and what is done about it):

* PNO overlaps S(p, q) -- millions of couples.  `ovgen.OverlapGen` keeps
  only each pair's compact PAO-domain coefficients on the device and, for a
  chunk of couples, scatters them into the union of their PAO domains and
  evaluates the chunk with one variable-size GEMM.  The (kl, ij) overlaps of
  the F'_vv / oooo-ladder terms are always regenerated per chunk; all other
  overlaps are stored when they fit in a quarter of the free device memory
  and otherwise regenerated chunk by chunk inside every plan
  (`DLPNO_ONFLY_S=auto|0|1`).
* DF tensors of the strong pairs and the 3-external tensors: pinned host
  memory, streamed per size bucket, when they exceed a third of the free
  device memory (`DLPNO_OFFLOAD=auto|0|1`).
* iterative (T): W and W+V at their true sizes in pinned host memory, D
  rebuilt from the orbital energies; the triples' 3-external tensors are
  accumulated per auxiliary atom without the fitted (Q|ab).
* every variable-size GEMM chain runs in memory-bounded term chunks; the CC
  stores are released before the triples; the CuPy pool allocator is used
  inside the kernels (GPU4PySCF routes >100 MB allocations to cudaMalloc).
* per-atom (Q|uv) blocks (the full 3-index tensor once the molecule is
  smaller than the PAO-domain radius: 41 GB for taxol / cc-pVDZ): pinned
  host memory behind a device LRU cache (`DFInts.qab_get`) when they would
  take more than 35% of the free device memory (`DLPNO_QAB_HOST=auto|0|1`,
  cache size `DLPNO_QAB_CACHE_GB`, default a quarter of the device).  All
  consumers (PNO integrals, pair integrals, triples) contract them one aux
  atom at a time; the cache is dropped before the CC iterations.
* closed-shell PNO integrals are built in pair batches (`pnoints.py`): the
  strong pairs' (k i|a e) over the extended domain are accumulated atom by
  atom from (Q|uv), never as a (Q|a e) tensor; per-pair lists are released
  as soon as the CC stores are assembled.
* (Q|i u) is stored block-sparse (`dfints.BlockQ`) when the dense
  (naux, nLMO, nPAO) tensor would exceed a quarter of the free device
  memory: per auxiliary atom over that atom's extended LMO and PAO domains,
  indexed like the dense array by one gather kernel (`DLPNO_QIA_SPARSE`);
  when even that exceeds a quarter of the device it lives in pinned host
  memory and the gather kernel reads it directly over PCIe (zero copy,
  `DLPNO_QIA_HOST`).
* (Q|uv) direct (`DLPNO_QAB_DIRECT=auto|0|1`; auto when the stored per-atom
  blocks would not fit the host either).  Every consumer (PNO integrals,
  pair / ring integrals, (T0)) asks `DFInts.qab_block(atom, PAOs)`.  The
  batches of a stage are put in spatial (Morton) order and grouped into
  *regions* whose blocks over the union of their PAOs fit a quarter of a
  device; per region each aux atom's AO integrals (mu nu|Q_a) are evaluated
  once and transformed to the region's PAOs; regions are distributed over
  the GPUs (each GPU builds and consumes its own).  Exact (same energies as
  the stored blocks to the last printed digit); for small molecules a
  single region, so no extra cost.  Measured AO-support of the PAOs is
  nearly global (projection tails), so restricting the AO integrals to the
  PAO support would save little.
* closed shell: the (Q|uv) device cache is dropped after the PNO
  integrals; the DF offload decision subtracts that cache; the overlap
  store decision counts only half of the (fragmented) pooled free memory.
* DIIS keeps a host Gram matrix and extrapolates vector by vector (no
  stacked copies of the history; pinned host history when it is large).

## Several GPUs (`mgpu.py`)

All visible GPUs of a node are used (one Python thread per device;
`DLPNO_NGPU` limits the count).  Peer access over NVLink is enabled.

| stage | distribution |
|---|---|
| SC-MP2 / LMP2 PNOs, PNO integrals, open-shell pair integrals, (T0) | work queue of pair / triplet chunks; per device a view of the driver with replicas of the small device tables (qia, qij, metric, S_pao, pair tables); (Q|uv) through the per-device cache |
| LCCSD, closed shell | DF groups of the strong pairs live on (and are dressed by) their device; every plan whose operands include PNO overlaps is chunked, its chunks spread over the devices, each device keeping the overlaps of its chunks (up to `DLPNO_S_KEEP` of its free memory, rest regenerated); dynamic operands replicated per call, static integrals once; partial outputs added on the home device |
| LCCSD, open shell | DF groups and the F'_vv / oooo-ladder chunks (with their overlap generators and workspaces) per device, same reduction |
| open-shell (T0) passes | one spawned process per GPU (`mgpu.run_queue_mp`): the triplet chunks are launch / Python bound, so threads sharing one interpreter lock left the GPUs at ~20 % utilisation; the workers get a small picklable context (large arrays shared through memory-mapped node-local files) and rebuild their device tables; one persistent pool per run (`mgpu.MPPool`), started when the LCCSD starts, so interpreter start, imports and CUDA contexts overlap the CC iterations and both (T0) passes reuse the workers; `DLPNO_MP=0` falls back to threads |

Convergence: the CC iterations stop when |dE| < `E_CONVERGENCE` (1e-6) and
the residual criterion of `R_NORM` holds: `'rms'` -- Psi4's (largest
per-pair residual RMS < `R_CONVERGENCE`, 1e-6), `'max'` -- ORCA's (largest
residual element < `R_CONVERGENCE_MAX`, 1e-5, ORCA's "max. residuum"), or
`'both'` (default) -- the tighter of the two, both must hold.  Comparisons
use the other program's own criterion: `R_NORM='rms'` against Psi4,
`R_NORM='max'` against ORCA.

So the device memory that grows fastest (overlaps, DF tensors) is spread
over the GPUs, and with 4 GPUs overlaps that would have to be regenerated
on one device are kept.

## One GPU: how large (A100-80GB, Normal PNO, (T0))

| system | atoms / basis functions | result |
|---|---|---|
| Cu macrocycle (open), def2-SVP | 60 / ~510 | runs (2513 s, older code) |
| taxol (closed), cc-pVDZ | 113 / 1123 | runs: 1841 s post-SCF, peak 78.4 GB (LCCSD 1535 s: the 34 GB of PNO overlaps are regenerated per chunk) |
| (Gly)20 (closed), def2-SVP | 143 / 1444 | runs: 1172 s post-SCF, peak 78.5 GB (23.5 GB of overlaps regenerated per chunk) |
| (Gly)40, def2-SVP | 283 / 2864 | stored mode: out of memory building the per-atom (Q|uv) blocks; direct + regions: through LMP2 (995 s) and into the PNO integrals |
| Trp-cage 1L2Y (closed, +1), def2-SVP | 304 / 2906 | stored mode: out of memory; direct + regions: DF integrals 126 s at 20.6 GB, LMP2 ~50 min (overlaps regenerated every iteration on one device) |

What the stored tensors would need (closed shell, Normal PNO):

| system | ext. PAOs per aux atom (mean / all) | stored (Q\|uv) | (Q\|iu) block-sparse | (Q\|ij) |
|---|---|---|---|---|
| taxol | 1096 / 1123 | 40.7 GB | 5.5 GB | 0.9 GB |
| (Gly)20 | 953 / 1444 | 37.4 GB | 4.8 GB | 1.9 GB |
| (Gly)40 | 2117 / 2864 | 368 GB | 42.5 GB | 14.9 GB |
| Trp-cage | 2328 / 2906 | 445 GB | 49 GB | 13.7 GB |

A pair needs only ~150-300 PAOs; the extended domain of an aux atom is the
union over every LMO that pairs through it, i.e. nearly the whole molecule
for a compact protein.  Hence the direct (Q|uv) with regions and the host
(Q|iu): the device then holds ~20 GB of integrals for Trp-cage.

Several GPUs, direct mode: (Gly)20 (143 atoms) 346 s total on 2 GPUs (1555 s
on one GPU with stored blocks), identical energies.

**Trp-cage on one node (4 x A100-80GB)**, DLPNO-CCSD(T0)/def2-SVP, Normal
PNO, 304 atoms, 2906 basis functions, 36533 pairs (6977 strong), 36791
triplets: 2301 s post-SCF (SCF 139 s on one GPU).  DF integrals 102 s,
LMP2 138 s, PNO integrals 1093 s (394 regions, AO integrals dominate),
LCCSD 497 s (the 83.5 GB of PNO overlaps stored over the four GPUs),
(T0) 154 + 269 s.  Peak host memory ~200 GB ((Q|iu) 49 GB pinned, DF
tensors of the strong pairs offloaded).  On one GPU the same run gets
through the DF integrals and LMP2 but the overlaps (83.5 GB) must be
regenerated every CC iteration.

## Known issue

Open-shell (T0): per-triplet energies depend on which triplets are batched
together (Ni dithiolene, identical orbitals: -0.1145582 / -0.1145911 /
-0.1145528 / -0.1145534 Eh for default, 2-way, 8-way and one-per-chunk
batching; ~500 of 6193 triplets differ by up to 5 %).  Padding sizes, the
L-domain bucketing, the projection cache and the batched linear algebra are
ruled out; tools: `DLPNO_DBG_T0_SPLIT`, `DLPNO_DBG_PADMAX`,
`DLPNO_DBG_NOBUCKET`, `DLPNO_DBG_QORD`.

## Validation

* closed shell vs Psi4 1.11 (same LMOs): alanine dipeptide / cc-pVDZ, all
  six preset x triples combinations within 1e-7 Eh (CCSD) and 3e-8 Eh (T)
  at Normal/Tight; the Loose and C12 residual differences (1e-6 .. 8e-5 Eh)
  are entirely the DOI quadrature: with Psi4's DOI grid
  (`DLPNO_DOI_GRID`) they drop to 2e-9 / 5e-9 Eh.
* open shell: GPU = host reference to 1e-10 Eh (ethyl, triplet CH2O);
  canonical limit = PySCF UCCSD(T) to 1e-10 Eh (CH3); closed-shell limit =
  closed-shell code.
* ORCA 6.1.1 (ROHF) on transition-metal complexes: see the report in the
  PR.

## Timings vs ORCA (Perlmutter, DLPNO-CCSD(T0), Normal PNO, post-SCF wall s)

GPU: A100-80GB, ORCA 6.1.1 RIJCOSX/DLPNO on one 2x64-core CPU node; CC
convergence ORCA's (`R_NORM='max'`, max. residuum 1e-5).

| system | basis | GPU x1 | GPU x4 | ORCA 64 cores | ORCA 128 cores |
|---|---|---|---|---|---|
| taxol (RHF) | cc-pVDZ | -- | 359 (Psi4 criterion) | 640 | -- |
| Ni dithiolene (ROHF, doublet) | def2-SVP | 345 | 296 | 607 | 691 |
| Cu macrocycle (ROHF, doublet) | def2-SVP | -- | 843 (3 GPUs: 947) | 986 | 1158 |

The two open-shell SCF solutions are not the same state as ORCA's (Ni:
ours 9 mEh lower; Cu: ORCA's 49 mEh lower), so only the timings compare.

## Files

`options.py` presets; `setup.py` orbitals, DOI, domains; `dfints.py`
3-index integrals; `linalg.py` batched eigh / pivoted Cholesky; `ragged.py`,
`vbatch.py` ragged batches and variable-size GEMM chains; `lmp2_gpu.py`,
`ccsd_gpu.py`, `triples_gpu.py`, `rhf.py` closed shell; `uhf.py`,
`uccsd_gpu.py` (term lists, reference residual), `uccsd_vec.py` (vectorised
residual), `utriples_gpu.py` open shell; `ref_rhf.py`, `ref_uhf.py`,
`ref_uhf2.py` host references; `devmem.py` memory policy;
`electrolites/dlpno_ccsd_t.py` the feature entry point.
