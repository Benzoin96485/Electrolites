# fasttdejk — the TDDFT gradient / NAC two-electron derivative (catalogue D10)

`docs/KERNEL_TARGETS_1.8.1.md` §5.1 rank 6.  `grad/tdrhf.py:_jk_energies_per_atom`
computes, for a list of density-matrix pairs `(D1_t, D2_t)` with factors
`j_t, k_t`, the per-atom derivative of `j_t J[D1_t, D2_t] − k_t K[D1_t, D2_t]`,
one output per pair.  It is the two-electron term of every restricted TDDFT /
TDA gradient and NAC, and after `fastnacms` (rank 3) it was **58–69 %** of the
batched force + all-NACs run.

## What GPU4PySCF does, measured

Its kernel (`rys_contract_jk_ip1.cu`, `rys_ejk_ip1_multidm_kernel`) evaluates
each integral quartet once per six pairs (`DM_BLOCK`) and keeps the density
products in a global-memory pool that it re-reads per Rys root per primitive
quartet.  On penicillin V's 38 pairs: one pair 3.0 s, three 4.6 s, seven
10.5 s, twelve 15.1 s, 38 48.9 s — so a pass over six pairs costs 2.5x a pass
over one: the contraction, not the integrals, is most of the time.

Two further findings, both recorded as facts about the reference:

* **its kernels are not correct on a general (non-symmetric) pair.**  Against
  central finite differences of `get_jk` energies at displaced geometries
  (D1, D2 held fixed in the AO basis), on glycine/def2-SVP GPU4PySCF misses by
  **2–6e-2 relative**, and two calls on the same input differ by up to 2e-3 —
  single- and multi-density kernels alike.  On water and formaldehyde, and on
  every physical density the TDDFT code builds that the tests compare, it
  agrees.  This is where its run-to-run NACV scatter (`README_fastnacms.md`)
  comes from.
* **the multi-density kernel faults (illegal address) at exactly six
  pairs**, its `DM_BLOCK`; seven and twelve run.

## The change

`codegen/gen_ejkpair.py` writes a pair family from `gen_ejk.py`'s two styles
(`fastejk`'s division-free Rys scaffolding): `EJK_NT` outputs per launch, each
output's density products in registers, the twelve force terms of every
cartesian function formed once and contracted with every output's product,
the per-output bra and ket reductions.  The density factor is GPU4PySCF's term
for term (eight K products, both orderings of the pair; the symmetrised J
product).  All 63 classes through f are covered (18 one-thread-per-quartet,
45 shared-memory), so nothing is handed back for an spdf basis; a bra pair of
more than 36 primitives or a short-range operator goes to GPU4PySCF's kernel.
`EJK_NT = 2` is the default (1: 22.7 s, 2: 17.2 s, 4: 17.3 s on penicillin V's
38 pairs).

## Measured

`benchmarks/path_anatomy.py --patch all`, with and without `FASTTDEJK_OFF=1`,
same node:

| | D10 before | after | | run before | after | |
|---|---|---|---|---|---|---|
| force + 15 NACVs (batched), penicillin V / B3LYP | 48.8 s | **17.9 s** | **2.73x** | 70.8 s | **40.0 s** | **1.77x** |
| same, penicillin V / PBE | 46.1 s | **16.8 s** | **2.74x** | 64.0 s | **34.1 s** | **1.88x** |
| same, taxol (113) / B3LYP | 247.8 s | **106.2 s** | **2.33x** | 425.0 s | **283.1 s** | **1.50x** |
| one TDDFT gradient + one NAC, penicillin V / B3LYP | 6.0 s | 4.1 s | 1.46x | 20.3 s | 17.8 s | 1.14x |
| one TDDFT gradient + one NAC, taxol / PBE | 18.8 s | 10.4 s | 1.81x | 72.9 s | 64.9 s | 1.12x |

A closed-shell transition-metal complex (Ni dithiolene²⁻, 21 atoms, def2-SVP,
B3LYP; the Ni s shells form bra pairs of more than 36 primitives, so 6 of
3298 class groups go to GPU4PySCF's kernel): TDDFT gradient + one NAC
50.0 → 41.7 s (1.20x), gradient and NAC equal to 5e-11.

With rank 3, a penicillin V B3LYP force + all-NACs step goes 147.8 → 40.0 s
(3.7x).  What is left of the batched run at 113 atoms: the Z-vector CPHF
(115 s), D10 (106 s) and the batched XC kernel (30 s, D9).

## Correctness

`tests/test_tdejk.py`: (1) against finite differences on water/6-31G*,
formaldehyde/def2-SVP and def2-TZVP (f shells) and glycine/def2-SVP — a
general pair, its transpose and a symmetric pair in one call — to ≤ 1.1e-8
relative (the difference step), repeating itself to 4e-15; GPU4PySCF printed
beside it; (2) end to end on glycine (B3LYP and PBE): a TDDFT gradient, a NAC,
the batched force and NACs, to ≤ 5e-12 against GPU4PySCF (whose physical
densities it gets right).  Force norms in the timed runs agree to 1e-11.

| | |
|---|---|
| `electrolites/codegen/gen_ejkpair.py` | the kernel family |
| `electrolites/fasttdejk.py` | the driver |
| `tests/test_tdejk.py` | finite differences and end to end |
| `FASTTDEJK_OFF=1`, `FASTTDEJK_NT` | upstream's function; outputs per launch |
