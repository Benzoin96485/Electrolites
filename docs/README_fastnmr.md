# fastnmr — the NMR shielding's GIAO XC term on the GPU (catalogue C7)

`docs/KERNEL_TARGETS_1.8.1.md` §5.1 rank 5.  `properties/shielding.py:nr_rks`
(1.8.1) walks the grid blocks on the GPU and in every block copies the grid
coordinates to the host, evaluates the GIAO AO factors `GTOval_ig` (3
components) and, for a GGA, `GTOval_ipig` (9 components) with libcint on the
CPU **over all `nao` AOs**, copies them back and keeps the block's `nsub`
columns.  §10.4 measured it at 72–78 % of a patched NMR run.

## The GIAO factors are products of things the block already has

Checked against libcint on random points (`benchmarks`-side check, max
deviation 3e-17 and 1e-16):

```
GTOval_ig[d]      = 1/2 (R_mu x r)_d phi_mu
GTOval_ipig[x, d] = 1/2 [ (R_mu x r)_d  d_x phi_mu  +  eps_{dax} R_mu,a phi_mu ]
```

`R_mu` is the AO's centre and `r` the grid point; `ipig`'s first index is the
derivative direction.  So both are formed on the GPU from the block's `ao`
for its `nsub` AOs only.  Upstream's two contractions per field direction are
regrouped by the AO factor they end on, so that a block is **one GEMM** for
all three directions:

```
V_d  = [L0_d, L1_d, L2_d, L3_d] . [phi, d_x phi, d_y phi, d_z phi]^T
L0_d = 1/2 [ C_d (w0 phi + sum_x w_x d_x phi) + (R x w)_d phi ]
Lx_d = 1/2  C_d w_x phi,             C_d = (R_mu x r)_d
```

## Measured

`benchmarks/path_anatomy.py --feature nmr --basis def2-svp --patch all`, with
and without `FASTNMR_OFF=1`, one A100-SXM4-80GB:

| system / functional | NMR, before | after | | the GIAO XC term | what is left |
|---|---|---|---|---|---|
| alanine dipeptide (22) / B3LYP | 36.1 s | **8.6 s** | **4.2x** | 30.3 s → < 0.1 s | C6 `int2e_ig1` 5.9 s |
| penicillin V (42) / B3LYP | 201.2 s | **44.3 s** | **4.5x** | 131.7 s → < 1 s | C6 38.7 s |
| penicillin V (42) / PBE | 200.3 s | **66.4 s** | **3.0x** | 134.3 s → < 1 s | C6 61.2 s |
| taxol (113) / PBE | 1954.2 s | **1207.5 s** | **1.62x** | 916.9 s → 1.6 s | C6 1175.0 s |

The GIAO XC term itself goes 30–570x faster and is gone from every profile;
the whole-run ratio is set by what it leaves, C6.  The PBE and B3LYP rows at
42 atoms have the same C7 and C6 work; C6 (libcint on the host) moved
38.7 ↔ 61.2 s, and at taxol 1007 ↔ 1175 s, with the CPU load of the other
jobs on the node, and the ratio with it.

**What is left is C6**, the GIAO two-electron term `int2e_ig1`, on the CPU
through PySCF's `_vhf.direct_mapdm` (`shielding.py:139-150`, upstream's own
`# TODO: this can be modified into jk`): 87–97 % of the remaining run (97 % at 113 atoms).  It is
§5.3's second clean-benchmark entry and not a §5.1 row; a GPU `(ig1)` Rys
kernel is greenfield work.

## Correctness

`tests/test_nmr.py`: LDA (SVWN), GGA (PBE), hybrid GGA (B3LYP, PBE0); water,
HCl (a third-row atom), glycine; def2-TZVPP (reaches f); grid level 1.  The
GIAO XC matrix to ≤ 5.4e-15 relative, the shielding tensors to ≤ 2.8e-12 ppm
against GPU4PySCF.  Meta-GGA and NLC raise in upstream and fall through to it
here unchanged.

| | |
|---|---|
| `electrolites/fastnmr.py` | the function |
| `tests/test_nmr.py` | the A/B |
| `FASTNMR_OFF=1` | upstream's function |
