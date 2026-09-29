# Stage A of the batched unrestricted h1 term, once

`docs/DF_WORK_LIST.md` **W41**, found by W38.  W38's item 23 runs the
unrestricted `h1` term (`_dfhessip_u._get_veff`) on a 40 GB card at 113 atoms
by batching the occupied index in stages A–C and keeping `dm_3c` on the host;
of the 3.4 s that cost against the full card on taxol⁺, 2.6 s was stage A's
`(00|0)` integrals evaluated once per occupied batch (1.5 → 4.1 s at four
batches).  The entry asked for one integral pass writing `j3c_00` for every
orbital to the host, stages B and C per batch from there; success was stage A
at its full-card time and the 40 GB call within 1.05x of the full card; the
falsification, B having to read `j3c_00` back at PCIe rate faster than it is
hidden (27 GB each way, 1 s at 26 GB/s).

**Done: stage A runs at its full-card time (1.50 s against 1.50 s), and the
40 GB call is 1.04x of the full card on taxol⁺ (31.3 s against 30.2 s) and
1.05x on the Co(II) sextet (24.9 s against 23.8 s).**  The read-back is
hidden behind stage B except its first chunk per batch (50 ms), and B is
0.26 s slower beside it; the falsification does not fire.  Part of the gain
is not stage A: measured stage by stage, 1.0 s of W38's call was four
`cudaMalloc`s of the batch's 7 GB `dm_3c`, which GPU4PySCF's pool sends to
the driver right after the previous batch's `cudaFree` of the same size
(§5.8 and W34 of the work list), and those buffers are now allocated once.

| 113 atoms, h1, 27.6 GB for the call (a 40 GB card) | W38 | this round | full card |
|---|---|---|---|
| taxol⁺ doublet (4 occupied batches) | 33.53 s (1.11x) | **31.31 s (1.04x)** | 30.16 s |
| Co(II) sextet (3 batches) | 26.36 s (1.11x–1.14x) | **24.91 s (1.05x)** | 23.79 s |

---

## 1. What changed

`electrolites/_dfhessip_u.py:_get_veff`, host mode only (item 23's default
when the occupied index is batched); nothing on a card where one batch fits.

**Item 35: stage A once** (`FASTDFHESSIP_H1UA1=0` is W38's stage A per
batch).  Before the occupied loop, one pass over the integral batches
contracts each block of `(00|0)` with every occupied orbital of both spins,
`j3c_00[r,n,i,q] = sum_p (pq|r) C[p,n,i]`, and copies the block's rows into
the page-locked host array `dm3_host (naux, 2, nocc, nao)` — the array item
23 already allocates for `dm_3c`.  The copies are on the copy stream, from
two device buffers, each block's copy behind the next block's integrals
(`H1UHOST_ASYNC=0`: synchronous, one buffer).  The Coulomb trace that
stage A formed on the first batch (`sum_pq (pq|r) D_pq`) is formed in the
same pass.

Each occupied batch then takes its columns `nocc0:nocc1` of `dm3_host`
back into its device `j3c_00` eight orbitals at a time — stage B's own
chunk (`FASTDFHESSIP_H1UB_CHUNK`, test-only, sets it) — with chunk `c+1`'s
`cudaMemcpy2DAsync` issued before stage B computes on chunk `c`, and the
metric applied as before.  Stage C's copy of the batch's `dm_3c` goes back
into the same columns, which the batch has read by then; the columns of the
later batches still hold `j3c_00`.  So the host array is written twice and
read twice per column, and no second array is needed.

**The batch's device arrays in buffers kept across the batches.**  `j3c_00`
(becoming `dm_3c` in place), `dm_oo` and stage B's chunk buffer are views of
three buffers allocated once before the occupied loop and released before
stage D.  Allocated per batch, each is above GPU4PySCF's 100 MB threshold
(`set_conditional_mempool_malloc`), so it was a `cudaFree` at the end of one
batch and a `cudaMalloc` of the same size at the start of the next, and the
`cudaMalloc` waits for the driver: 0.25 s a batch on taxol⁺, which W38's
stage table charged to "A j3c_00" and this round's first version to
"A j3c_00 in" (1.0 s, §2).  It applies to W38's stage A per batch too
(`H1UA1=0`), so that column below is W38's arithmetic without the waits.

---

## 2. Measurements

`benchmarks/df_hess_terms_probe.py --terms h1 --reps 2`, variants round robin
in one process, minimum of two after a warm call, `FASTDFHESSIP_TIME=1`;
taxol⁺ at `--ballast 40` (27.6 GB free when the call starts), the Co(II)
sextet, whose SCF does not fit 40 GB, at `--free-after 27.6`; the full card
in a separate process.  One A100-SXM4-80GB of a whole Perlmutter node,
def2-SVP, B3LYP, `patch_all()`.

| | taxol⁺, 40 GB | Co(II) sextet, 27.6 GB |
|---|---|---|
| item 35 (default) | **31.31 s** | **24.91 s** |
| `H1UA1=0` (stage A per batch, buffers kept) | 32.79 | 25.70 |
| W38's code (stage A per batch, arrays per batch; job `el_w41a/b`) | 34.05 | 26.40 |
| item 35 with arrays per batch (first version) | 32.77 | 25.29 |
| item 35, synchronous copies | 36.14 | 28.40 |
| full card (one batch; W35's path) | 30.16 | 23.79 |
| the first call of the process (page-locks the host array) | 36.40 | 27.33 |

(W38 measured the same "W38's code" cell at 33.53 s and 26.36 s in its own
jobs; the full card was 30.09 s and 23.12–23.76 s there.)

Stage tables, second call, s:

| stage | taxol⁺ W38 | taxol⁺ item 35 | full card | Co W38 | Co item 35 | full card |
|---|---|---|---|---|---|---|
| A `j3c_00` (per batch / once) | 4.13 | **1.50** | 1.54 | 2.97 | **1.30** | 1.36 |
| A `j3c_00` in (first chunk per batch) | — | 0.04 | — | — | 0.05 | — |
| B metric | 4.37 | 4.61 | 4.34 | 3.31 | 3.47 | 3.32 |
| C (gemm + K + J + int2c2e_ip1 + copy out) | 9.42 | 9.59 | 9.07 | 7.44 | 7.46 | 6.95 |
| D `dm_3c` in | 0.34 | 0.35 | — | 0.35 | 0.36 | — |
| D integrals + D1–D3 | 10.42 | 10.01 | 10.09 | 8.06 | 8.10 | 8.00 |
| E fill + per-aux-atom | 4.27 | 4.16 | 4.00 | 3.22 | 3.22 | 3.04 |
| setup, F | 1.08 | 0.99 | 1.09 | 1.04 | 0.91 | 1.12 |
| **total** | **34.03** | **31.25** | **30.14** | **26.39** | **24.87** | **23.79** |

What is left of the difference to the full card on taxol⁺ (1.1 s): stage B
0.27 s slower while its fetches run beside it, stage C's per-batch GEMMs
0.5 s, E's smaller block (40 functions against 88) 0.16 s, and the first
fetch of stage D, which nothing precedes (0.35 s); the D loop itself is at
its full-card time.

**The falsification.**  Every chunk after the first of each batch is fetched
while stage B works on the one before it: 1 GB against 0.16 s of B, so the
PCIe read is hidden and the visible cost is the first chunk (40-50 ms a batch)
and B's 6 % slowdown beside the copy — 0.3 s in all against the 2.6 s of
integrals it replaces.  It does not fire.

**The first call** pays for page-locking the 27 GB host array (3.3 s,
"A host buffer"), as W38's did, and a Hessian calls `h1` once: 36.40 s, 1.21x
of the full card's first call (the full card has nothing to page-lock).

---

## 3. Correctness

`tests/test_dfuhessip.py --no-hessian` passes ("all ok"): every W38 item-23
row — occupied batches of 8 and 5 orbitals, host async / host synchronous /
`occ`, every h1 branch, B3LYP / HF / `k = 0` / the long-range operator,
ethanol⁺ and the triplet ethanol²⁺ — now runs item 35 in the host modes, and
four new columns run it with stage B's fetch in chunks of 3 orbitals
(`H1UB_CHUNK=3`, several fetches per batch) and W38's stage A
(`H1UA1=0`), asynchronous and synchronous: worst 4.7e-15 against
GPU4PySCF.  At 113 atoms, against the unbatched call on the same SCF made
before the memory is withheld (`--prefree-env FASTDFHESSIP_H1UBATCH=0`): the
Co(II) sextet 8.3e-14 (item 35) and 1.1e-13 (`H1UA1=0`); taxol⁺'s variants
agree with each other to 1.0e-13.

---

## 4. Tools

```bash
python tests/test_dfuhessip.py --no-hessian
FASTDFHESSIP_TIME=1 python benchmarks/df_hess_terms_probe.py --patch all \
    --terms h1 --variants ours --reps 2 --ballast 40 \
    --xyz benchmarks/molecules/taxol.xyz --charge 1 --spin 1 \
    --basis def2-svp --xc b3lyp \
    --env 'A0:FASTDFHESSIP_H1UA1=0,sync:FASTDFHESSIP_H1UHOST_ASYNC=0'
FASTDFHESSIP_TIME=1 python benchmarks/df_hess_terms_probe.py --patch all \
    --terms h1 --variants ours --reps 2 --free-after 27.6 \
    --prefree-env 'FASTDFHESSIP_H1UBATCH=0' \
    --xyz benchmarks/molecules/tmc_co_sextet.xyz --spin 5 \
    --basis def2-svp --xc b3lyp --env 'A0:FASTDFHESSIP_H1UA1=0'
```

Switches: `FASTDFHESSIP_H1UA1=0` (item 35 off: stage A per occupied batch),
`FASTDFHESSIP_H1UB_CHUNK=<n>` (test-only: stage B's orbitals per chunk,
8).
