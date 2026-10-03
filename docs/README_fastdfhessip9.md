# The ip2 term on a short card, again, and with its occupied index batched

`docs/DF_WORK_LIST.md` **W42** and **W43**, found by W40 and W39.

* **W42** asked for the rest of taxol⁺'s 40 GB unrestricted ip2 call (44.9 s
  against 34.2 s on the full card) to go: fuse the first pass into the first
  chunk under item 34, cut item 34's packed `(1|00)` to its spherical rows,
  and measure whether keeping the first walk's `(10|0) C` in page-locked host
  memory beats walking the integrals again.  Success: within 1.15x of the
  full card.  Falsified if the fused chunk costs a chunk and the host copies
  are not hidden.
* **W43** asked for PfPMT's restricted ip2 call (284 atoms, the occupied
  index batched: `j3c_oo` alone is 38 GB) to be timed, and item 8 carried to
  a batched occupied index.

**W42: a measurement first.  The 44.9 s was partly the probe's.**
`benchmarks/df_hess_terms_probe.py` kept the SCF's cderi on the device while
it timed the term -- 14.3 GB at 113 atoms -- and a Hessian does not: upstream's
`partial_hess_elec` and `make_h1` call `with_df.reset()` first
(`df/hessian/uhf.py:751,772`, "Release GPU memory").  With the tensor
released, as a Hessian runs it, the call was already **37.9 s** on the
40 GB card against 31.2 s on the full card (1.21x; W40's Hessian table had
it at 39.4 s inside the Hessian, which is where the probe and the Hessian
disagreed).  The probe now releases it (`--keep-cderi` is W40's setting).

Then items 36 and 37 (`electrolites/fastdfhessip.py`, both spin cases):

| ip2 call, 40 GB card | before (W40) | after | full card | after / full |
|---|---|---|---|---|
| taxol⁺ UKS, cderi released (as a Hessian) | 37.8 s (4 chunks, unfused) | 37.9 s (the same plan) | 31.2 s | 1.21x |
| taxol⁺ UKS, cderi kept (W40's probe) | 44.3 s (7 chunks) | **42.7 s** (6 chunks) | 31.2 s | 1.37x |
| taxol RKS, cderi released | 21.4 s (unfused) | **19.5 s** (fused) | 16.6 s | 1.18x |
| Co(II) sextet UKS, 27.6 GB | 34.0 s | **32.6 s** | 25.9 s | 1.26x |

**Not a success on taxol⁺**: inside a Hessian its call is where it was,
1.21x of the full card, and the falsification's first half fires -- planned
on what the packing leaves, the fused pass holds 1120 rows a chunk against
the unfused 1408, five chunks against four, so item 36 (correctly) does not
fuse it; five chunks walk `(5+1)/2 = 3` integral passes where four walk 2.5
plus the first pass's own, and at 4.7 s a walk that is a wash.  Where fusing
costs no chunk it is taken and pays: taxol RKS 21.4 → 19.5 s (1.10x; W40's
1.37x of the full card is now 1.18x).  Item 37's spherical rows pay where
they change the chunk count (the kept-cderi taxol⁺ row, 7 → 6 chunks, and
the Co sextet).  The host copy of `(10|0) C` is priced and not built: the
Cartesian product is `3 naux 2 nao nocc` words, 85 GB at taxol⁺, and the
three later chunks would read 1.5 walks' worth of it back, 127 GB at
26 GB/s = 4.9 s, against the 1.5 walks' recomputation it replaces (7 s at
4.7 s a walk) -- a saving of at most 2 s for 85 GB of page-locked host
memory beside the SCF's, not taken.  What is left at 40 GB is the chunk
count, which is memory: see W47.

**W43**: §3 (item 38).

---

## 1. W42: what changed

**Item 36: fused or not, planned after item 34's packing** (`_plan36`).
Item 34 chose between the fused and unfused second pass on plans made
*before* it packed `dm_oo` and `(1|00)`, against their unpacked sizes, and the
fused plan also held item 3's `10 naux^2` back for the h_aux block -- so on
a 40 GB card the pass never fused.  Now the two are packed first; both plans
are made on what is then free, holding what outlives the h_aux block (its
`j2c_10v` and item 33's spherical copy, `6 naux nsph` words), and the fused
one also what the block allocates beside the pass's buffers (`j2c_10` and its
sorted copy, `6 naux^2`, where item 11 carves the rest from the integral
buffers).  The pass fuses iff that costs no chunk, and then item 24's first
pass from the second pass's `M C` writes into the packed `(1|00)` (a
`(dk, ns, nocc, nocc)` block and `_SymRows.put`); without item 24
(`FASTDFHESSIP_D1D3=0`) the fused first pass writes packed as the unfused
one does.  `FASTDFHESSIP_PACKFUSE=0` is item 34's decision.

**Item 37: item 34's packed `(1|00)` cut to its spherical rows**
(`_sph_rows`): after the h_aux block each direction's first `nsph` rows are
spherical and the rest are never read, but they are `naux - nsph` rows a
direction of the buffer the second pass is planned beside -- 1.2 GB at
taxol⁺.  A copy into a `3 nsph`-row `_SymRows`, made before the unfused
second pass allocates anything.  `FASTDFHESSIP_SPHROWS=0`.

## 2. W42: measurements

`df_hess_terms_probe.py --terms ip2 --variants ours --reps 1`, the ablations
round robin in the same process (`--env pf0:FASTDFHESSIP_PACKFUSE=0,
sr0:FASTDFHESSIP_SPHROWS=0,old:...=0;...=0`), one A100-SXM4-80GB,
`--ballast 40` or `--free-after 27.6`, jobs `el_w44f` (W40's code),
`el_w44h2`, `el_w44i2`; s:

| stage | taxol⁺ 40 GB released | taxol⁺ 40 GB kept, after | kept, before | taxol⁺ full card |
|---|---|---|---|---|
| chunks × rows | 4 × 1408 | 6 × 944 | 7 × 800 | 1 (item 25 whole) |
| D1 ipaux pass | 2.72 | 2.71 | 2.70 | (fused) 2.12 |
| D3 integrals | 2.89 | 3.66 | 4.07 | 2.03 |
| D3 j3c_100 | 8.95 | 11.80 | 12.42 | 2.42 (+ D8 3.32) |
| D5 per-atom | 11.93 | 12.72 | 13.55 | 6.65 (+ D8 3.52) |
| **total** | **37.9** | **42.7** | **44.3** | **31.2** |

Inside a whole Hessian (W44's runs, `docs/README_fastcphf.md` §3.1) the
40 GB call is 37.9 s and the full card's 31.3 s -- the probe with the cderi
released reproduces the Hessian to 0.1 s.  The full card is unchanged by
this round (31.19 against 31.21 s, `old` column).

Correctness: `tests/test_dfhessip.py` and `tests/test_dfuhessip.py` with the
multi-chunk pass forced, fused and unfused, in three more columns each
(`PACKFUSE=0`, `D1D3=0`, `SPHROWS=0`), and the memory ladder now reaches the
packed fused pass (`'packed fused'`): worst 5.5e-13 against GPU4PySCF.  At
113 atoms against item 34's code on the same SCF: 4.3e-14–1.2e-12 relative.

## 3. W43: item 38, item 33 per occupied batch

Item 8 and everything built on it required the occupied index in one batch:
`_SphAux` was built only when `num_occ_batches == 1`, and a batched call fell
back to upstream's eigenbasis, chunked over the metric dimension, a whole
walk per chunk, per occupied batch, with upstream's per-atom D5 over the
whole Cartesian auxiliary index.  But every term of the second pass except
D4 is linear in the batch's orbitals, and item 33 (W40) already runs the
Cholesky factor in chunks of rows:

38. **Item 33 per occupied batch** (`FASTDFHESSIP_OCCROWS=0` is the
    eigenbasis).  With the occupied index batched the factor is item 8's
    (`V = C L^-T`, unit weights) in every line of the call, and each batch
    runs item 33's chunks of rows -- the last rows first, each chunk walking
    only the integral batches its rows reach -- and item 33's D5 on the
    batch's columns: `nocc` there is the batch's `j` and every `i` is
    contracted (`_batched_occ`'s new `nk`); the partners `(1|00)` and
    `dm_oo` are the batch's rows, unpacked (a batch's blocks are not
    symmetric, so items 34 and 36 are off).  Batch 0 holds every orbital,
    as upstream's does, for D4's Gram form (whose `Q` is quadratic in the
    orbitals), and after D4 moves its batch's columns to the front of the
    chunk's buffer in place (`_cols_in_place`) for D5.  `h_sph` is summed
    over the batches.  Taken only where its memory estimate fits, and a call
    that runs out of memory in it is run again without it (below).

Correct to 1.4e-13–3.7e-13 against GPU4PySCF on ethanol with the occupied
index forced into 2 and 3 batches, in one chunk and in several, fused and
unfused, and with D5 in blocks of 7 rows (`tests/test_dfhessip.py`).

**At 113 atoms, with the batching forced** (taxol RKS, full card, the
occupied index in 2 and 4 batches by `FASTDFHESSIP_OCC_BATCH`, round robin
in one process against the unbatched call; job `el_w44o`), s:

| taxol RKS ip2 | unbatched (item 8 whole) | 2 batches, eigen | **2 batches, item 38** | 4 batches, eigen | **4 batches, item 38** |
|---|---|---|---|---|---|
| total | 15.67 | 32.86 | **28.49** (1.15x) | 39.44 | **33.53** (1.18x) |
| D3 integrals | 1.01 | 2.18 | 2.18 | 4.24 | 4.24 |
| D3 j3c_100 | 1.39 (+ D8 3.43) | 6.94 | 6.90 | 8.32 | 8.29 |
| D5 per-atom | 3.39 | 10.18 | **6.07** | 12.13 | **6.42** |
| against the unbatched call | | 8.8e-12 | 6.1e-12 | 8.8e-12 | 8.1e-12 |

On one chunk (the full card holds the whole metric dimension) the walks are
the same in both, so this is item 38's D5 alone -- 1.7x–1.9x on that stage;
where the chunks are several, as they will be at 284 atoms, item 33's prefix
walks add their `(c + 1) / 2` against `c`.  The batched call is still 1.8x–2.1x
the unbatched one at 113 atoms: per batch D3 is the chunked form (`j3c_100`
through `V` rather than item 8's block-diagonal apply), and stage B and the
plan move `dm_oo` through the host.

**Where the batching is not forced but memory makes it** -- taxol RKS with
the card's memory withheld after the SCF (`--free-after`), so that the
occupied index goes into two batches and the second pass into several
chunks, as it would at 284 atoms; the same process against item 38 off, one
A100-SXM4-80GB in an interactive allocation, s:

| taxol RKS ip2, free after the SCF | eigenbasis (before) | **item 38** | | against each other |
|---|---|---|---|---|
| 24 GB (2 batches; 3 chunks with item 38, 2 before) | 47.15 | **33.91** | **1.39x** | 9.6e-12 |
| 20 GB (3 chunks) | 54.62 | **38.64** | **1.41x** | 1.1e-11 |
| 18 GB (3 chunks) | 56.70 | **44.11** | **1.29x** | 8.9e-12 |
| 16 GB | 67.29 | 67.24 (declined) | 1.00x | 4.8e-14 |

Here the walks count as well as D5: at 24 GB D3's integrals are 5.35 s
against 7.36 and `j3c_100` 7.98 against 12.52 (the prefix walks), D5 6.43
against 12.86.  And other classes, forced into batches: the Ni
dithiolene complex (NiC₈F₆N₂S₄, a 3d metal whose def2-SVP reaches f) as a
neutral singlet (4 batches, one chunk) 2.15 →
2.02 s (1.06x, 2.9e-11 → 1.7e-11 against the unbatched call); penicillin V
in 2 batches and 5 chunks 3.25 → 2.37 s (1.37x, 1.7e-12).

**Item 38 holds more than the eigenbasis**: the factor's coefficients (`ms`,
its pseudo-inverse blocks and `L^-1`, `2 naux nsph + nsph^2` words) beside
upstream's arrays.  At 16 GB taxol's first version ran out of memory in the
unfused h_aux block where the eigenbasis runs.  So it is taken only when a
batch's `(1|00)` and `dm_oo`, those coefficients and the larger of the
h_aux block's peak and the second pass's floor fit, with 2 GB spare; and a
call that runs out of memory inside it anyway is run again without it
(`occ_rows_retried` in `stats()`; the retry is made outside the `except`,
whose traceback would otherwise keep the failed attempt's arrays alive --
the first version of the retry ran out of memory in the eigenbasis for that
reason).  At 16 GB it declines (67.2 s, the eigenbasis's time); at 18 GB it
is taken and pays.

**PfPMT itself is not measured.**  Its call before and after needs a
multi-hour job each (`j3c_oo` alone is 38 GB, and W39 priced the call in
hours); two were queued for about a day without starting and were cancelled
in favour of the measurements above, which reach the same code paths --
batched occupied index, several chunks, the memory estimate -- at 113 atoms
in minutes.  Whether the 284-atom call runs at the 113-atom rate per flop,
W43's own criterion, is still open.

## 4. Tools

```bash
FASTDFHESSIP_TIME=1 python benchmarks/df_hess_terms_probe.py --patch all \
    --terms ip2 --variants ours --reps 1 --ballast 40 \
    --xyz benchmarks/molecules/taxol.xyz --charge 1 --spin 1 \
    --basis def2-svp --xc b3lyp [--keep-cderi] \
    --env 'old:FASTDFHESSIP_PACKFUSE=0;FASTDFHESSIP_SPHROWS=0'
FASTDFHESSIP_TIME=1 python benchmarks/df_hess_terms_probe.py --patch all \
    --terms ip2 --variants ours --reps 1 --no-check \
    --xyz benchmarks/molecules/PfPMT.xyz --charge -1 --basis def2-svp \
    --xc b3lyp [--env 'eig:FASTDFHESSIP_OCCROWS=0']        # W43
python tests/test_dfhessip.py --no-hessian; python tests/test_dfuhessip.py --no-hessian
```
