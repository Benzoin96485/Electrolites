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

## Validation

* closed shell vs Psi4 1.11 (same LMOs): alanine dipeptide / cc-pVDZ, all
  six preset x triples combinations within 1e-7 Eh (CCSD) and 3e-8 Eh (T)
  at Normal/Tight; the Loose and C12 residual differences (1e-6 .. 8e-5 Eh)
  are entirely the DOI quadrature: with Psi4's DOI grid
  (`DLPNO_DOI_GRID`) they drop to 2e-9 / 5e-9 Eh.
* open shell: GPU = host reference to 1e-10 Eh (ethyl, triplet CH2O);
  canonical limit = PySCF UCCSD(T) to 1e-10 Eh (CH3); closed-shell limit =
  closed-shell code.
* ORCA 6.1.1 (ROHF, same SCF state) on transition-metal complexes: see the
  report in the PR.

## Files

`options.py` presets; `setup.py` orbitals, DOI, domains; `dfints.py`
3-index integrals; `linalg.py` batched eigh / pivoted Cholesky; `ragged.py`,
`vbatch.py` ragged batches and variable-size GEMM chains; `lmp2_gpu.py`,
`ccsd_gpu.py`, `triples_gpu.py`, `rhf.py` closed shell; `uhf.py`,
`uccsd_gpu.py` (term lists, reference residual), `uccsd_vec.py` (vectorised
residual), `utriples_gpu.py` open shell; `ref_rhf.py`, `ref_uhf.py`,
`ref_uhf2.py` host references; `devmem.py` memory policy;
`electrolites/dlpno_ccsd_t.py` the feature entry point.
