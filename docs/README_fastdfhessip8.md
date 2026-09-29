# The ip2 term where item 8's whole index does not fit: the Cholesky factor in chunks of rows

`docs/DF_WORK_LIST.md` **W40**, found by W38.  With W38's h1 term fitting, a
40 GB UKS Hessian at 113 atoms reaches the unrestricted ip2 term
(`_dfhessip_u._jk_energy_per_atom`), and taxol⁺'s ran in 195 s against 58.6
s on the full card: on 44.6 GB even one spin of W37's item 25 (item 8, the
metric as its Cholesky factor, on the spin index) does not fit, so the
second pass falls back to the eigenbasis-style arithmetic chunked over the
metric dimension, each chunk a whole walk over the derivative integrals.
The entry asked for the stage table at 40 GB and a count of the chunks; if
it is the walks, for the integrals' contraction per chunk to take the whole
metric dimension or the `(10|0)` block to go to the host.  Success: the ip2
call within about 1.5x of its 80 GB time at 113 atoms, restricted and
unrestricted, and a whole 40 GB UKS Hessian of taxol⁺ measured.
Falsification: the extra time being arithmetic the smaller card forces.

**It was the walks, and the arithmetic the fallback chose — not arithmetic
the card forces.**  The table (§1) says taxol⁺ at 40 GB ran **32 chunks** of
176 metric rows, and 212 s of its 227 s were per-chunk: the integrals
walked 32 times (35 s), each chunk's `(10|0) C` product and its GEMM
against the metric's columns (71 s), and the eigenbasis D5, whose per-atom
contractions run over the whole Cartesian auxiliary index for every chunk
(107 s).  The same fallback was also what taxol⁺ ran on the **80 GB** card
(3 chunks, 51 s), which item 25 does not reach either.

So this round does not keep item 8's whole spherical index at all.  It keeps
the Cholesky factor and runs it **in chunks of rows of `Y = L^-1 K`** (item
33): a chunk of rows needs only the Cartesian auxiliary rows whose shell
starts below its last row, so it walks only their integral batches; D4 is
the Gram form on the chunk as it stands; and D5 is rewritten so that nothing
needs `Z = L^-T Y` by rows.  Item 34 then makes room for bigger chunks: the
first pass's `(1|00)` and `dm_oo` are held as their `i <= j` pairs from the
first pass on, which on the 40 GB card is also what lets the h_aux block run
beside them at all.

| ip2 call, 113 atoms | memory | before (W37's code) | after | 80 GB card, after | vs 80 GB |
|---|---|---|---|---|---|
| taxol⁺ doublet, UKS | 40 GB card | 227.1 s (32 chunks) | **44.9 s** (7 chunks) | **34.2 s** (was 51.3) | **1.31x** |
| taxol, RKS | 40 GB card | 31.7 s (3 chunks) | **22.8 s** | 16.6 s (item 8 whole) | **1.37x** |
| Co(II) sextet, UKS | 27.6 GB (a 40 GB card's) | 167.3 s (28 chunks) | **34.6 s** (6 chunks) | 25.9 s (item 25 whole) | **1.33x** |
| Co(II) sextet, UKS | 44.6 GB | 48.1 s (4 chunks) | **29.1 s** | | 1.12x |

**The call is within the entry's 1.5x at 113 atoms in both spin cases**:
taxol⁺ 1.31x of its full-card time (which this round also takes from 51.3
to 34.2 s), taxol 1.37x, the Co(II) sextet 1.33x at a 40 GB card's memory.
The falsification does not fire (§1).  WHOLE_HESSIAN_TBD

---

## 1. Where the time went

`FASTDFHESSIP_TIME=1`, one call after a warm one, W37's code (`main`), jobs
`el_w40a/b/c` and `el_w40h`; s:

| stage | taxol⁺ 80 GB | taxol⁺ 40 GB | taxol 40 GB | Co 44.6 GB | Co 27.6 GB |
|---|---|---|---|---|---|
| chunks × rows | 3 × 1872 (fused) | 32 × 176 | 3 × 1872 | 4 × 1328 | 28 × 192 |
| A, B, C | 3.97 | 4.25 | 3.21 | 3.52 | 4.02 |
| D1 ipaux pass | (fused) | 2.69 | 2.08 | 2.10 | 2.16 |
| D2 h_aux | 3.23 | 3.45 | 1.94 | 2.54 | 2.58 |
| D3 integrals | 3.12 | **34.66** | 3.52 | 3.94 | **31.61** |
| D3 j3c_100 | 13.03 | **70.55** | 7.84 | 11.60 | **45.45** |
| D4 oo_atm | 4.13 | 4.18 | 2.09 | 3.12 | 3.20 |
| D5 per-atom | 23.51 | **107.04** | 10.80 | 20.86 | **78.01** |
| **total** | **51.33** | **227.12** | **31.72** | **48.09** | **167.33** |

Per chunk, three things are repeated: the walk over the `ip1`/`ipaux`
integrals (1.1 s on taxol⁺), the product of every block with the occupied
orbitals (`(10|0) C`, for every auxiliary row, 1.6 s with both spins), and
in D5 the eigenbasis's per-atom contractions against `V[:, v0:v1] / w`,
whose Cartesian index is the whole auxiliary set in every chunk.  None of
it is arithmetic a 40 GB card forces: the Cholesky form item 25 runs on the
full card does the same terms with one walk and a `nsph x nsph` D5.  The
falsification does not fire.

The chunk count is set by what the second pass is left: at 40 GB, taxol⁺'s
call holds `j3c_oo1` (`(3, naux, 2, nocc, nocc)`, 16.2 GB), `dm_oo` (5.4
GB) and the SCF's cderi (14.3 GB) beside it, and 7.4 GB remain for
`(3, rows, 2, nao, nocc)` chunks of 12.9 MB a row.

---

## 2. What changed

`electrolites/fastdfhessip.py` (`_CholRows`, `_SymRows`,
`_h_aux_head_pin`) and the same in `_dfhessip_u.py`; taken wherever item 8
(or item 25) would be taken but its whole spherical index does not fit.

### 2.1 Item 33: the factor in chunks of rows of `Y`

Item 8's second pass forms the spherical `K = C^T (10|0) C_o` whole, then
`Y = L^-1 K` (D8), D4 on `Y` by rows of the auxiliary index, `Z = L^-T Y`
(D8), and D5 on `Z` by atom columns.  In chunks, the row/column mismatch
between D4 and D5 is what forced the fallback.  But:

* **`Y` by rows is one walk per chunk over a prefix.**  The chunk `[v0,
  v1)` of `Y` is `V[:, v0:v1]^T` applied to the Cartesian `(10|0) C`, with
  `V = C L^-T` the factor item 8 already builds; `V[r, t]` is zero unless
  row `r`'s shell has a spherical column `<= t`, so the chunk needs only
  the Cartesian rows whose shell starts below `v1` — a prefix of the
  integral batches.  The chunks run last first, so the first one is the
  whole walk (the one the fused first pass and the h_aux block ride on) and
  the others stop early: `(c + 1) / 2` walks for `c` chunks instead of `c`.
* **D4 is unchanged**: the Gram form sums over the auxiliary index and runs
  on a chunk of rows with unit weights, as the chunked fallback already did.
* **D5's first half** is `sum_{s in B} <Z[s], T1_A[s]>` with the output on
  the spherical row `s`; `Z` restricted to one chunk of `Y` is
  `L^-T[:v1, v0:v1] Y_c`, nonzero on the rows `s < v1` only, so it is one
  GEMM per atom group and block of rows, reduced against `T1` by W37's
  `d5_h1` kernel.
* **D5's second half** is `sum_t j10s[y,s,t] (P Z)[s,x,t]`: `Z` enters summed
  over `t`, so it is `sum_{t in chunk} (j10s L^-T)[y,s,t] (P Y_c)[s,x,t]`,
  the same per-atom product on the chunk, and `j10s L^-T` is the spherical
  rows of the h_aux block's `j2c_10v` (`C^T j2c_10v`), formed once
  (`d5_h2r`, a variant of `d5_h2` for a column block).
* **The Coulomb terms** are upstream's in the factor's basis (the chunked
  fallback's own `_oo_gram` Coulomb half), exact for any `V` with `V V^T =
  C J^-1 C^T`.

Nothing forms `Z` by rows, and the two triangular multiplies (D8) are gone:
their work is in the D3 GEMM (the factor's columns, a prefix of rows) and
the first half's GEMM.  `FASTDFHESSIP_CROWS=0` is the fallback.

### 2.2 Item 34: the memory

* **The plan without upstream's per-atom D5**: the chunked fallback held
  `6 naux nao_A nocc` words (12 unrestricted) back for its D5 temporaries;
  item 33's D5 sizes its groups and row blocks to the second pass's `work`
  buffer, so the plan holds that buffer (2 GB) and D4's `(3, 3, nao, nao)`
  temporaries back instead, and the spherical `j2c_10v` item 33 keeps.
* **Unfused, `dm_oo` and `(1|00)` as their weighted `i <= j` pairs from the
  first pass on** (`_SymRows`, item 22's `_Pairs` weights): the first pass
  writes `(1|00)` packed block by block, `dm_oo` is packed once after stage
  C, and the h_aux block's head — the four products item 22 already ran on
  packed chunks — runs on them in place, with no chunk buffers
  (`_h_aux_head_pin`); `j3c_oo1p` stays packed, goes spherical in place
  (the auxiliary transformation acts on rows), and D5 unpacks a block of rows
  of it and of `dm_oo` at a time.  Half of 21.6 GB on taxol⁺; on the 40 GB
  card the h_aux block's own `15 naux^2` temporaries do not fit beside the
  unpacked pair (`cudaErrorMemoryAllocation` in the block, first version of
  this item), and the second pass is planned on what is left after it.
* Where the pre-packing plan has no row at all it is not taken, and the pass
  is planned after the packing.

`FASTDFHESSIP_PACKROWS=0` is item 33 alone; `FASTDFHESSIP_MEMLOG=1`
prints the second pass's memory (it found the two OOMs of the first
versions); `FASTDFHESSIP_D5_ROWS=<n>` caps D5's row blocks (test-only).

---

## 3. Measurements

`df_hess_terms_probe.py --terms ip2 --variants ours`, one warm call then one
or two timed, `FASTDFHESSIP_TIME=1`; the 40 GB card is `--ballast 40` for
taxol and taxol⁺ (44.6 GB free before the SCF), and `--free-after` for the
Co(II) sextet, whose SCF does not fit 40 GB: 27.6 GB left after the SCF is
what taxol⁺'s calls get on the 40 GB card, 44.6 GB what W37 §3.4 used.  One
A100-SXM4-80GB of a whole node, def2-SVP, B3LYP, `patch_all()`.

| ip2 call | before | items 33 + 34 | item 33 alone (`PACKROWS=0`) | chunks before → after |
|---|---|---|---|---|
| taxol⁺ UKS, 40 GB | 227.1 s | **44.9** | out of memory in the h_aux block | 32 → 7 |
| taxol⁺ UKS, 80 GB | 51.3 | **34.2** | 34.3 (fused: 34 has nothing to pack) | 3 → 3 |
| taxol RKS, 40 GB | 31.7 | **22.8** | 23.8 | 3 → 3 |
| Co(II) sextet UKS, 27.6 GB | 167.3 | **34.6** | 134.4 | 28 → 6 |
| Co(II) sextet UKS, 44.6 GB | 48.1 | **29.1** | 30.7 | 4 → 3 |
| Co(II) sextet UKS, 80 GB (item 25 whole) | 25.8 | 25.9 | — | 1 |
| taxol RKS, 80 GB (item 8 whole) | 16.6 | 16.6 | — | 1 |

("Before" is `main`, W37's code, in the same kind of job; the Co 27.6 GB
column's "before" is `FASTDFHESSIP_CROWS=0` in the same process, which is
that code.  Item 33 alone at taxol⁺'s 40 GB: the first versions of this
round ran out of memory there, in the h_aux block's own temporaries, which
is what item 34's packing from the first pass on is for.)

Stage tables of the "after" column, s:

| stage | taxol⁺ 80 GB | taxol⁺ 40 GB | taxol 40 GB | Co 27.6 GB | Co 44.6 GB |
|---|---|---|---|---|---|
| chunks × rows | 3 × 1872 (fused) | 7 × 800 | 3 × 1872 | 6 × 880 | 3 × 1760 |
| A, B, C | 3.90 | 4.31 | 3.07 | 4.04 | 3.53 |
| D1 ipaux pass | (fused) | 2.70 | 2.09 | 2.17 | 2.10 |
| D2 h_aux | 3.46 | 3.54 | 1.96 | 2.49 | 2.56 |
| D3 integrals | 2.03 | 4.06 | 2.32 | 3.84 | 1.94 |
| D3 j3c_100 | 8.53 | 12.39 | 5.03 | 8.60 | 6.54 |
| D4 oo_atm | 4.14 | 4.12 | 2.08 | 3.14 | 3.11 |
| D5 per-atom | 11.84 | 13.53 | 6.11 | 10.01 | 9.18 |
| **total** | **34.24** | **44.90** | **22.84** | **34.55** | **29.14** |

Against the whole Cholesky form on the full card, what is left is one
thing: the walks.  On taxol⁺'s 80 GB card the fused first pass rides on the
first chunk, three chunks walk `(1 + 2 + 3)/3 = 2` times, and D3 is 8.5 s
against the Co sextet's 2.1 + 5.2 s of its whole form with two spin passes;
D5's first half is a GEMM of the chunk against `L^-T`'s rows, 0.57–0.67 of
a triangular multiply over all rows, where item 25 runs two.  At 40 GB the
first pass is a walk of its own (`D1 ipaux pass`, 2.1–2.7 s) and the
chunks are smaller.

---

## 4. Correctness

`tests/test_dfhessip.py` and `tests/test_dfuhessip.py` (`--no-hessian`)
pass, with the multi-chunk second pass forced (`FASTDFHESSIP_VCHUNK` 160,
64 and 40: 3, 6 and 7 chunks on ethanol and ethanol⁺), fused and unfused,
in five columns each: items 33 and 34 (the default), `CROWS=0` (the
fallback), D5 in atom groups of 8 AOs, D5 in row blocks of 7, and item 33
alone; the memory ladder reaches item 33 in both files (`'chol rows'`).
Worst against GPU4PySCF: 5.0e-13 restricted, 3.4e-13 unrestricted — what
every other column of those files reads.

At 113 atoms, against the whole Cholesky form on the same SCF (the call made
before the memory is withheld, `--prefree-env ''`): taxol at 27.6 GB
**5.5e-12**, the Co(II) sextet **4.5e-11** — the Co sextet's metric (condition
3.3e10) moves its Cholesky form by 4–6e-11 for a one-ulp perturbation of the
metric (W37 §5.1), and item 33 applies the same factor in another order.
Against `CROWS=0` (the same factor through the fallback's order) the Co
sextet agrees to **2.0e-12**; two runs of the same code differ by 4e-12 on
taxol (the ip2 kernel's atomics, W39).

---

## 5. Tools

```bash
FASTDFHESSIP_TIME=1 python benchmarks/df_hess_terms_probe.py --patch all \
    --terms ip2 --variants ours --reps 1 --ballast 40 \
    --xyz benchmarks/molecules/taxol.xyz --charge 1 --spin 1 \
    --basis def2-svp --xc b3lyp \
    --env 'i33:FASTDFHESSIP_PACKROWS=0,c0:FASTDFHESSIP_CROWS=0'
FASTDFHESSIP_TIME=1 python benchmarks/df_hess_terms_probe.py --patch all \
    --terms ip2 --variants ours --free-after 27.6 \
    --xyz benchmarks/molecules/tmc_co_sextet.xyz --spin 5 \
    --basis def2-svp --xc b3lyp --env 'c0:FASTDFHESSIP_CROWS=0'
FASTDFHESSIP_TIME=1 python benchmarks/df_hess_anatomy.py --patch all \
    --xyz benchmarks/molecules/taxol.xyz --charge 1 --spin 1 \
    --basis def2-svp --xc b3lyp --rounds 1 --ballast 40
python tests/test_dfhessip.py; python tests/test_dfuhessip.py
```
