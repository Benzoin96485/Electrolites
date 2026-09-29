# The ip2 call's `(nao, nao)` products: not tile-shaped, but half of them are a triangle

`docs/DF_WORK_LIST.md` **W39**, found by W37.  After W37 every contraction
stage of the restricted ip2 call (`_jk_energy_per_atom`) on taxol runs at
15–18 TF/s, and three of them multiply zeros: D3's `M C`, stage A's `C^T J`
and the pseudo-density's `C X C^T` are dense `(nao, nao)` products on blocks
filled only at the stored AO pairs, 50 % of the block at 113 atoms.  The
entry asked for the AO index to be tiled and each product formed only on
the tile pairs that hold stored pairs, measured at 113 and 284 atoms in
both spin cases; success was the three stages in proportion to the fill and
the restricted call past W37's 1.3x at 113 atoms; the falsification, that
the tile pairs holding any stored pair cover most of the block at a tile
size the GEMMs still run well at.

**The falsification fires at 113 atoms, by count.**  Counted on the
Hessian's own Cartesian pair list (`benchmarks/df_ip2_tile_probe.py`, §1),
the tiles that hold a stored pair cover **94–100 %** of taxol's and the
Co(II) sextet's blocks at every tile from 16 to 128 AOs — the pair fill is
50 %, but it is the product of an atom-pair factor a tile can address and a
shell-level one it cannot, which is W5's finding on the SCF's cderi, and
the Hessian's pairs are the same pairs.  At 284 atoms the tiles cover 61–93
%: a tiled product could remove at most 29 % there at 32 AOs.

**But two of the three products are symmetric, and a triangle is half of any
block.**  The pseudo-density `C X C^T` is symmetric and is read only at the
stored pairs, each of which can be read in the lower triangle; stage A's
`C^T J C` has a symmetric `J`.  So:

* **item 29**: the pseudo-density formed on the lower triangle's row tiles
  (53 % of the dense product): taxol 0.50 → **0.30 s**;
* **item 30**: stage A as `G + G^T`, `G = C^T L C`, `L` the lower triangle
  of `J` with half its diagonal: the fill and the products 0.38 → **0.29 s**
  (inside a stage whose integrals vary by ±0.1 s from call to call);

and W37's two small leads with them:

* **item 31**: the h_aux block's nine `j2c_ip2` GEMMs as three `dsyrk` and
  three GEMMs, 1.97 → **1.85 s**;
* **item 32**: `_SphAux`'s pattern analysed on the device instead of the
  host, stage B 0.47 → **0.30 s**.

D3's `M` holds different values in its two triangles (`-di` above, `dk + di`
below) and has neither structure; it stays dense.

| 113 atoms, ip2 call | W37 before | W37 | this round | |
|---|---|---|---|---|
| taxol RKS, call | 21.21 s | 16.66 s | **16.13 s** | **1.315x** of W37's before |
| taxol, inside a whole Hessian | 20.81 | 16.65 | **16.05** | 1.297x |
| Co(II) sextet UKS, call | 42.31 | 25.71 | **25.36** | 1.668x |

**The success bar is met in the call (1.315x against 1.3x) and missed by
0.3 % inside a whole Hessian (1.297x)**; the call is 1.03x faster than
W37's, and the Co(II) sextet's 1.01x.  Whole-Hessian correctness: 2.3e-11
and 2.7e-11 Eh/bohr² from W37's two final Hessians, which are 3.7e-11
apart from each other.

---

## 1. The count

`benchmarks/df_ip2_tile_probe.py` builds the Hessian's `Int3c2eOpt` as
`df/df.py:DF.build` does and reads `pair_and_diag_indices(cart=True)`; no
integrals, no timing.  `tilefill T` is the fraction of the square covered by
`T x T` tiles holding a stored pair (or its mirror), `lower T` what one GEMM
per row tile of `T` AOs over the columns up to the tile's last pair computes,
as a fraction of the dense product:

| def2-SVP | atoms | nao (Cart.) | pairfill | tilefill 16 | 32 | 64 | 128 | lower 64 |
|---|---|---|---|---|---|---|---|---|
| penicillin V | 42 | 454 | 64.6 % | 98.0 % | 99.8 % | 100 % | 100 % | 57.0 % |
| taxol | 113 | 1185 | 49.5 % | **94.1 %** | 96.9 % | 99.4 % | 100 % | **52.7 %** |
| Co(II) sextet | 113 | 1116 | 51.9 % | 97.7 % | 100 % | 100 % | 100 % | 52.8 % |
| PfPMT | 284 | 2838 | 21.6 % | 61.2 % | 71.1 % | 81.3 % | 92.7 % | 51.1 % |

The two cheap tile sizes a GEMM runs well at (32 and more) cover 97–100 % of
a 113-atom block: the falsification's condition, in the entry's words.  At
284 atoms the tile fill (71 % at 32) is W5's 69–72 % for the SCF's cderi,
and a tiled product would remove up to 29 % of the three stages; PfPMT's ip2
call was not timed — its `j3c_oo` alone is 38 GB (`naux` 15,761, `nocc`
547), so it runs with the occupied index batched and the eigenbasis chunked,
item 8 off, for hours.

The count also found that the stored pairs are in **both** triangles (taxol:
55.3 % with `i > j`, 44.4 % with `i < j`, 0.3 % diagonal; a diagonal shell
block holds both orders), which is why items 29 and 30 read every pair at
`(max(i,j), min(i,j))` rather than at its own address.

---

## 2. What changed

All in `electrolites/fastdfhessip.py` (`_LowTri`), used by both spin cases,
`omega = 0` only.

**Item 29: the pseudo-density on the lower triangle** (`_ejk_ip2`,
`FASTDFHESSIP_PDLOW=0`).  Stage C forms, per auxiliary block,
`dm_tensor[p,q,r] = -k/2 (C X_r C^T)[p,q] + j D[p,q] auxvec[r]` and reads it
at the stored pairs.  It is symmetric, so every pair is read at `(max, min)`
and only the lower triangle is formed: one GEMM per row tile of 64 AOs
over the columns `[0, q1)`, `q1` one past the last column any pair of the
tile reaches, 52.7 % of the dense product on taxol.  The Coulomb part
`j D[p,q] auxvec[r]` is added to the pairs after the read (an elementwise
kernel on `nao_pair x dk`) instead of being written into the whole block.
Unrestricted, both spins are one GEMM of depth `2 nocc`.  Not on the
long-range operator: reading one triangle of a block that is symmetric to
rounding moved water's long-range ip2 term by **1.1e-10** against
GPU4PySCF (2.2e-14 without the item) — item 22's observation, on the
pseudo-density.

**Item 30: stage A on the lower triangle** (`FASTDFHESSIP_ALOW=0`).
`j3c_oo[r] = C^T J_r C` with `J_r` symmetric is `G + G^T` with `G = C^T L C`
and `L` the lower triangle of `J_r` with half its diagonal.  A scatter kernel
writes `L` (the upper triangle of the buffer is never written, so it stays
zero), and the first product `C^T L` runs by column tiles, tile `Q` over the
rows `p >= q0` only: 53 % of the flops.  Measured in isolation
(`benchmarks/df_ip2_stagea_bench.py`, one taxol block of 408):

| per block of 408 | fill | `C^T J` | `(.) C` | `G + G^T` | total |
|---|---|---|---|---|---|
| dense | 2.52 ms | 15.85 ms (16.3 TF/s) | 3.22 | — | 21.59 ms |
| lower, T = 32 | 1.43 | 9.38 (14.2 TF/s, 51 % of the flops) | 3.22 | 0.29 | 14.32 |
| **lower, T = 64** | 1.43 | 9.07 (15.0 TF/s, 53 %) | 3.22 | 0.28 | **14.00** |
| lower, T = 128 | 1.45 | 9.23 (15.5, 55 %) | 3.22 | 0.28 | 14.18 |
| lower, T = 256 | 1.45 | 9.90 (15.8, 60 %) | 3.22 | 0.28 | 14.85 |

In the call it is 0.09 s of stage A (`FASTDFHESSIP_TIME=3` splits the stage:
the fill 46 → 25 ms, the products 330 → 263 ms, job `el_w39c`), and the
stage's integral evaluation varies by more than that from one call to the
next (590–800 ms across twelve calls), which is why the whole-stage rows in
§3 do not show it.  Not on the long-range operator (the twin's stage A note:
its rounding moves a triplet's long-range term by 1e-10).

**Item 31: the h_aux block's `j2c_ip2` sum** (`FASTDFHESSIP_IP2SYM=0`).
Upstream adds `(j2c_10v[x] W^-1 j2c_10v[y]^T) o dm_aux` to every `(x, y)`
block of `h_aux` — nine `naux x naux x nsph` GEMMs — and symmetrises `h_aux`
at the end of the block.  With `dm_aux` symmetric the `(y, x)` term is the
`(x, y)` one transposed, so the three diagonal blocks are `dsyrk` on
`j2c_10v[x] / sqrt(w)` and the three with `x < y` are GEMMs added twice:
4.5 GEMMs' worth instead of nine.

**Item 32: `_SphAux` on the device** (`FASTDFHESSIP_SPHHOST=1`).  The
first- and last-column `argmax` passes over the `ncart x nsph` pattern of
the spherical transformation ran on the host (0.44 s of numpy `argmax` on
the login node, 0.17 s of stage B on a compute node); on the device they
are milliseconds, and only the permutation's small arrays come back.

---

## 3. Measurements

`df_hess_terms_probe.py --terms ip2 --reps 2 --variants ours --env ...`, all
variants round robin in one process, minimum of two after a warm call; one
A100-SXM4-80GB, def2-SVP, B3LYP, `patch_all()` (job `el_w39a`).

| taxol RKS, ip2 call | min | C pseudo-dm | A j3c_oo | D2 h_aux | B metric |
|---|---|---|---|---|---|
| **all four (default)** | **16.13 s** | 0.30 | 1.00–1.02 | 1.85–1.87 | 0.30–0.31 |
| none (W37's code) | 16.66 | 0.50 | 1.01–1.04 | 1.97 | 0.46–0.47 |
| item 29 off | 16.29 | 0.50 | | | |
| item 30 off | 16.08 | | 0.96–1.02 | | |
| item 31 off | 16.15 | | | 1.96 | |
| item 32 off | 16.20 | | | | 0.48–0.50 |
| row tiles of 128 | 16.05 | 0.30–0.31 | | | |
| row tiles of 32 | 16.10 | 0.39 | | | |

(the call's total varies by ±0.05 s between two runs of the same code,
`again` in job `el_w39c`: 16.13–16.19 s.)

| Co(II) sextet UKS, ip2 call | min | C (kernel + pseudo-density) | B | D2 h_aux |
|---|---|---|---|---|
| all four | **25.36 s** | 1.51 | 0.41 | 2.46 |
| none | 25.71 | 1.74 | 0.54 | 2.56 |

**Inside a whole Hessian** (`df_hess_anatomy.py --rounds 1`, job
`el_w39b`): taxol 673.58 s (W37: 675.42 s), ip2 **16.05 s** (W37: 16.65 s;
before W37: 20.81 s), trace 110.8437373795; the Hessian is 2.3e-11 and
2.7e-11 Eh/bohr² from W37's two final Hessians (`$SCRATCH/w37`), which
differ from each other by 3.7e-11 — the ip2 kernel's atomics make two runs
of the same code differ by 4e-12 relative in this call (`again`, §3).

What is left of the call (taxol, inside the Hessian): D5 3.40 s, D8 3.45 s
(the two triangular multiplies), D4 2.13 s, D2 h_aux 1.86 s, D3 `j3c_100`
1.38 s, D3 integrals 1.01 s, the ip2 kernel 0.97 s, stage A 0.82 s, the
pseudo-density 0.37 s, stage B 0.33 s, and `D3 alloc` 0.23 s — the second
pass's 36 GB `cudaMalloc`, right after stage C freed its buffers.

---

## 4. Correctness

`tests/test_dfhessip.py` and `tests/test_dfuhessip.py` (`--no-hessian`) pass
with six new columns on every term and system: each item off
(`PDLOW=0`, `ALOW=0`, `IP2SYM=0`, `SPHHOST=1`) and the triangle's tiles at
16 and at 7 AOs (tiles that cut shells).  Worst against GPU4PySCF: 6.6e-13
restricted, 5.9e-13 unrestricted (W37's file read 5.5e-13 and 4.2e-13 on
the same systems in another SCF).  The long-range rows run items 31 and 32
only, and agree to 6.5e-14 (restricted) and 1.4e-13 (the triplet) after
items 29 and 30 were kept off there (§2).

---

## 5. Tools

```bash
python benchmarks/df_ip2_tile_probe.py --cases taxol,tmc_co_sextet:5:0,PfPMT:0:-1
python benchmarks/df_ip2_stagea_bench.py --xyz benchmarks/molecules/taxol.xyz [--blk 122]
FASTDFHESSIP_TIME=1 python benchmarks/df_hess_terms_probe.py --patch all \
    --terms ip2 --variants ours --reps 2 --xyz benchmarks/molecules/taxol.xyz \
    --basis def2-svp --xc b3lyp \
    --env 'w38:FASTDFHESSIP_PDLOW=0;FASTDFHESSIP_ALOW=0;FASTDFHESSIP_IP2SYM=0;FASTDFHESSIP_SPHHOST=1'
FASTDFHESSIP_TIME=3 ...      # stage A as integrals, fill and products
python tests/test_dfhessip.py; python tests/test_dfuhessip.py
```

Switches: `FASTDFHESSIP_PDLOW=0`, `FASTDFHESSIP_ALOW=0`,
`FASTDFHESSIP_IP2SYM=0`, `FASTDFHESSIP_SPHHOST=1` (items 29–32 off),
`FASTDFHESSIP_LOW_TILE=<n>` (the triangle's tile, 64 AOs),
`FASTDFHESSIP_TIME=3` (stage A split).
