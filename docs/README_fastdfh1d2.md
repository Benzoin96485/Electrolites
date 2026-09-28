# The h1 term's stage D2 on a smaller card: it was launches, not arithmetic

`docs/DF_WORK_LIST.md` **W35**, found by W32.  With a 40 GB ballast on an
80 GB A100 — the memory of the 40 GB nodes — the restricted B3LYP `h1` call
on taxol was **28.7 s against 17.4 s**, and 10.6 s of the 11.3 s difference
was stage D2, W30's occupied form of the per-atom exchange term, which grew
from 2.5 to 13.1 s.  D2 launches `2 natm` contractions per auxiliary block
and direction, so its launch count grows as the block shrinks while its
arithmetic does not.  The entry asked to count D2's launches and its rate
at both block sizes, and then to batch the atoms into one launch per block
or give D2 a minimum block of its own; it would be falsified if D2's rate
were the same at both sizes and the extra time were recomputation
elsewhere.

D2's rate is not the same at both sizes: it is 9.9 TF/s at the 80 GB block
and **1.9 TF/s** at the 40 GB one, and the difference is exactly the host
cost of the launches.  Batched, the 40 GB call is 17.7 s, and it is now
within 4 % of the 80 GB one.

---

## 0. Summary

* **What D2 was** (§1): at 40 GB the block is 32 auxiliary functions
  instead of 392, about 225 blocks instead of 22, and D2 makes 226
  `contract` calls per block and direction — about 150,000 calls at
  **87 µs each**, which is the host cost of one cuTENSOR `contract` from
  Python (`benchmarks/df_h1d2_probe.py`: 19.6 ms of host time for one
  block's 226 calls, at every block size up to 128).  At 80 GB the same
  loop is 15,000 calls, and there the GEMMs are long enough to hide most of
  it.
* **Item 19** (§2): the D loop holds its block of `j3c` transposed and D1's
  product with `C_o` as `(i, p, r)`, so that `(p, r)` is one contiguous
  index of every D2 operand.  Each atom's two products are then plain
  row-major GEMMs, and the atoms with the same number of AOs are one
  `cublasDgemmBatched` — **4 launches per block and direction** on taxol
  instead of 226, up to 8 on the metal complexes.  Both spins go in the same
  batch.  D1 and D3 read the transposed block with the two AO indices
  swapped; nothing else changes.
* **The success bar is met, restricted** (§3): taxol B3LYP at 40 GB,
  D2 **13.2 → 2.0 s**, the call **28.8 → 17.7 s (1.63x)**, against
  17.0 s at 80 GB — **1.04x** of it, where it was 1.62x.  The Co(III)
  complex (113 atoms): 17.6 → 13.5 s at 40 GB, against 13.7 s at 80.  At
  80 GB the call moves too, 1.04x–1.05x at 113 atoms (D2 2.5 → 1.9 s).
* **Unrestricted, the 40 GB case does not run** — neither before nor
  after: `_get_veff`'s own budget (`2 naux nocc nao` of density plus the
  metric) does not fit a 44.6 GB card at 113 atoms, and GPU4PySCF's twin
  runs out of memory even on the full card (W25).  With a **30 GB**
  ballast (54.6 GB free) the twin is in its memory-short branch, whose D2
  is the same two products per atom with their intermediates made per atom:
  47.7 s of a **76.5 s** call.  Item 19 takes that branch too, and item 20
  (below) gives it a usable block: **76.5 → 34.6 s**, against 30.5 s on
  the full card.  At 80 GB the twin's call is 1.06x (32.4 → 30.5 s) and
  the Co(II) sextet's 1.10x (26.1 → 23.9 s).
* **Item 20, a minimum block** (§4): the integral batch is sized first, at
  80 % of the budget, and D's block from the rest, so on the shortest cards
  the block fell to 8–24 functions beside batches of 74.  Below 32 the
  batch now gives the block that many.
* **W30's threshold goes** (§5): with batched products, the occupied form
  wins or ties at every restricted size W30 measured — including the
  def2-SVP cases below its 4e6 score where W30's loop lost by up to 24 % — so it is
  now taken wherever it saves arithmetic.  Restricted B3LYP h1 at 80 GB:
  1.02x–1.29x of the previous default on nine of eleven systems from 10 to
  60 atoms (ties on the other two, both def2-TZVPP) and 1.04x–1.05x at 113;
  unrestricted 1.00x–1.12x.  Against
  GPU4PySCF the restricted h1 is now **1.6x–2.4x** and the unrestricted
  **1.3x–2.4x**.
* **Correctness** (§6): element-wise against GPU4PySCF's function on
  every system, 7e-16 – 2.7e-14 relative, the same as the per-atom column
  beside it; every new branch (short last block, occupied batching,
  memory-short, rebalanced block) is in `tests/test_dfhessip.py` and
  `tests/test_dfuhessip.py`.

Everything is one A100-SXM4-80GB of a whole Perlmutter node, GPU4PySCF 1.8.1,
level-3 grids, `patch_all()` on both sides.  "40 GB" is a 40 GB cupy array
allocated before the SCF (`benchmarks/df_hess_terms_probe.py --ballast 40`,
44.6 GB free after it), which is what W32's §3.1 measured with.  Function
tables are `benchmarks/df_hess_terms_probe.py`, variants round robin in one
process, minimum of 1–5.

---

## 1. Measured first: D2's launches and its rate at both block sizes

The restricted taxol call, `FASTDFHESSIP_TIME=1` (synchronised stage
timers), W30's loop (`FASTDFHESSIP_H1D2=atom`), second and third calls:

| | 80 GB | 40 GB |
|---|---|---|
| integral batch / D block (aux. functions) | 2010 / 392 | 328 / **32** |
| auxiliary batches, D blocks | 11, ≈22 | 25, ≈225 |
| D2 `contract` calls (`2 natm · 3 · blocks`) | ≈15,000 | ≈152,000 |
| D2 | 2.54 s | **13.24 s** |
| D2 rate (`12 nao² nocc naux` = 2.51e13 flops) | 9.9 TF/s | **1.9 TF/s** |
| 13.24 s / 152,000 calls | | 87 µs per call |

The falsification asked whether the rate is the same at both sizes: it is
not, by 5x.  Nor is the extra time elsewhere: D1, D3 and E together grow by
0.8 s, C not at all.

The probe (`benchmarks/df_h1d2_probe.py`) builds taxol's shapes with random
operands and times one block's D2 at a given `dk`, with the time the host
spends issuing it (nothing synchronised) beside the wall time:

| `dk` | per-atom `contract` (W30) | host | per-atom cuBLAS | host | **batched** | host |
|---|---|---|---|---|---|---|
| 16 | 19.59 ms, 1.0 TF/s | 19.59 | 5.17 ms, 3.9 TF/s | 5.15 | **2.07 ms, 9.8 TF/s** | 0.06 |
| 32 | 19.72, 2.1 | 19.71 | 5.70, 7.1 | 5.07 | **3.47, 11.7** | 0.06 |
| 64 | 19.49, 4.2 | 19.47 | 9.32, 8.7 | 5.16 | **6.22, 13.1** | 0.06 |
| 128 | 19.78, 8.2 | 19.74 | 15.44, 10.5 | 5.17 | **11.09, 14.7** | 0.06 |
| 256 | 28.69, 11.3 | 20.30 | 25.77, 12.6 | 5.93 | **21.83, 14.9** | 0.06 |
| 512 | 57.62, 11.3 | 20.17 | 56.55, 11.5 | 5.13 | **48.00, 13.5** | 0.06 |

Up to `dk = 128` the per-atom loop's wall time **is** its host time —
226 calls at 87 µs, whatever the GEMMs cost — and it becomes arithmetic only
past 256.  The same products as bare cuBLAS calls cost 23 µs each on the
host, a quarter, and are then bound by their own short `K = nao_A dk`; one
batched call per atom size is both.  The same pattern on the unrestricted
shapes (taxol, both spins: 20.1 → 4.0 ms at `dk = 16`, 19.7 → 6.3 at 32)
and on two basis/class changes:

| | `dk` = 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|
| penicillin V / def2-TZVPP (3 atom sizes), W30 → batched, ms | 7.41 → 0.72 | 7.38 → 1.27 | 7.44 → 2.23 | 7.78 → 4.29 | 10.99 → 8.50 | 20.94 → 19.78 |
| Pt(II) amine / def2-TZVPP (4 sizes, ECP) | 4.24 → 0.28 | 4.13 → 0.46 | 4.27 → 0.82 | 4.25 → 1.55 | 4.34 → 3.21 | 5.69 → 6.39 |

The one row where batching loses is the Pt complex at `dk = 512`, 12 %:
four atom sizes of 1–11 atoms each make four small batches, and at that
length the per-atom GEMMs are already arithmetic.  No production block is
that long on a system that small (the Pt complex's block is 486 on the full
card, and its h1 call is 1.04x–1.11x with item 19, §5).

That is also why this is not "give D2 a minimum block": at 80 GB the block
is already 392 and D2 still runs at 9.9 TF/s, because the rate is capped
by 226 launches per block, not by the block.

---

## 2. What changed

`electrolites/_h1d2.py` (new, patches nothing), and the D loop of
`fastdfhessip._get_veff` and `_dfhessip_u._get_veff`.

For one block of `dk` auxiliary functions and one direction, W30's
occupied form adds to every atom `A` (`u` its AOs, `n` the spin)

    vhf[A][n,j,q] += a sum_{u in A, r} j3c[u,q,r] s[n,u,r,j]
                   + a sum_{u in A, r} pri[n,u,r,j] t[n,u,r,q]

with `s = C_o dm_oo` and `t = C_o dm_3c` hoisted per block (items 4 and
12) and `pri = j3c C_o` D1's.  In `j3c[u,q,r]`, `(u, r)` is not one index,
so each product is a two-mode contraction and cuTENSOR is the natural call.

**Item 19.**  The block is scattered transposed, `jT[q,u,r]` (the scatter
with its two address arrays swapped), and D1 forms `pri` as `priT[n,j,u,r]`
— `C_o^T jT`, a plain GEMM.  Then atom `A`'s AOs are a contiguous run
`[p0 dk, p1 dk)` of the `(u, r)` index of `jT`, `s`, `priT` and `t` alike,
and both products are row-major GEMMs `nocc x nao x (nao_A dk)`:

    C = s_A^T jT_A^T        (lda nocc,       ldb nao dk)
    C = priT_A t_A          (lda nao dk,     ldb nao)

Atoms with the same `nao_A` give GEMMs of one shape at different
addresses, so each product is one `cublasDgemmBatched` per distinct atom
size, with both spins in the batch (`_h1d2.AtomGroups`; the pointer arrays
are built once per call and block size and cached).  D1's second product
and D3's two read the transposed block with the index strings swapped
(`'ipr,riq->pq'`, `'qpr,r->pq'`, `'qpr,pq->pr'`), which are the same sums.

A batch's last block is shorter than the buffer, and a view
`j3c_full[:, :, :dk]` then has `blksize` as its row stride, so `(u, r)` is
not contiguous in it.  That block is laid out afresh in the buffer's
leading `nao² dk` elements, zeroed first (the scatter writes the stored
pairs only), and the buffer is zeroed again before the `(00|1)` half of
the batch reads it with the full layout (`_h1d2.contiguous_block`,
`restore`) — the tests' `FASTDFHESSIP_H1BLK=24/40` rows are there to
reach it.

The memory-short branch (the per-atom AO matrices do not fit, so stages C
and E are on the occupied side too) forms D2 as the same two products per
atom, with its three intermediates made per atom and per direction — the
loop the unrestricted twin takes at 113 atoms on anything short of a full
80 GB card.  Item 19 takes it too, with item 12's hoisted `s` and `t`,
whose `2 nao (nao + nocc)` words per auxiliary function come out of the
block.

`FASTDFHESSIP_H1D2=atom` is W30's loop (and the memory-short branch's own);
item 5 off (`FASTDFHESSIP_H1J=upstream`) takes it too, since item 19 forms
D3 in item 5's form.

**Item 20.**  `_get_veff` sizes the integral batch first, at 80 % of what
is left, and D's block from the remaining 15 %.  On a full card the block
is hundreds of functions and this does not matter; on the shortest cards
the batch holds dozens and the block a handful (74 and 8 for the
unrestricted twin with item 19's arrays at 54.6 GB).  Below
`FASTDFHESSIP_H1BLK_MIN` (32) functions the batch is shrunk to give the
block that many — if the batch can still hold the largest auxiliary shell
— and the integral evaluator is rebuilt for it.  `=0` is off.

**Item 12's threshold.**  W30 took the occupied form only above a shape
score of 4e6, because below it the form's `2 natm` launches per block lost
to the AO form's `natm`.  With item 19 it wins or ties everywhere (§5), so
the default threshold is 0 when item 19 is on and W30's 4e6 when it is off.

---

## 3. The 40 GB card

h1, B3LYP, `FASTDFHESSIP_TIME=1`, W30's loop against item 19 in one
process, the minimum of 2–3 calls:

| system | card | block | D2: W30 → **item 19** | call: W30 → **item 19** | 40 / 80 GB |
|---|---|---|---|---|---|
| taxol, 113 atoms | 80 GB | 392 | 2.54 → **1.86 s** (9.9 → 13.5 TF/s) | 17.79 → **17.04 s** (1.04x) | |
| taxol | 40 GB | 32 | 13.24 → **2.04 s** (1.9 → 12.3 TF/s) | 28.82 → **17.67 s** (**1.63x**) | 1.62 → **1.04** |
| Co(III) complex⁺, closed shell, 113 | 80 GB | 800 | 2.00 → **1.48** | 14.47 → **13.75** (1.05x) | |
| Co(III) complex⁺ | 40 GB | 80 | 5.47 → **1.43** | 17.64 → **13.50** (**1.31x**) | 1.22 → **0.98** |

(The item-19 process sized its first call's block at 416 and 40, because
W30's first call ran after it and the pool had moved; §5.9 of the work
list.  The rows are the calls at the block both columns share.)

The rest of the 40 GB call is what the smaller block costs elsewhere, and
it is small: D3 0.43 → 0.80 s and E's fill 1.35 → 1.64 s on taxol (both
launch per block and direction too, but a handful of launches, not
`2 natm`).  Stage C and the integrals do not move.

**Success: met, restricted** — the 40 GB call is 1.04x the 80 GB one at
113 atoms (the entry's bar is 1.2x), 0.98x on the Co complex, and the
80 GB call is not slower anywhere (§5).

The unrestricted twin:

| system | free | block | path | D2 | call |
|---|---|---|---|---|---|
| taxol⁺ doublet, 113 | 80 GB | 88 | occupied form, W30 → item 19 | 5.27 → **3.70 s** (9.6 → 13.6 TF/s) | 32.39 → **30.55 s** (1.06x) |
| Co(II) sextet, 113 | 80 GB | 136 | occupied form, W30 → item 19 | 4.54 → **2.83** | 26.14 → **23.87** (1.10x) |
| taxol⁺ | 54.6 GB | 24 → 8 | memory-short, W30 → item 19 | 47.7 → **6.74** | 76.50 → **39.90** (1.92x) |
| taxol⁺ | 54.6 GB | 24 → 32 | memory-short, item 19 + item 20 | 47.0 → **4.66** | 75.58 → **34.62** (**2.18x**) |
| taxol⁺ / Co(II) sextet | 44.6 GB | — | `Insufficient GPU memory` in `_get_veff`'s own budget, both columns | | |

(The Co(II) sextet at 54.6 GB runs out of memory in its UKS SCF, before
any Hessian term.)  The twin's budget before its block is sized is
`2 naux nocc (nao + nocc)` of density factors plus the metric — 29 GB at
taxol⁺ — so at 113 atoms it needs more than a 40 GB card however D2 is
formed.  That is upstream's layout and not this round's; §7 lists it.

---

## 4. Item 20: the block the batch leaves

The unrestricted twin's memory-short branch on taxol⁺ at 54.6 GB free, the
only 113-atom case measured where the block was below the minimum
(`FASTDFHESSIP_TIME=1`, variants round robin, second call):

| | integral batch / block | auxiliary batches | D2 | D1 | D3 | E fill | call |
|---|---|---|---|---|---|---|---|
| main (W30's loop, no minimum) | 74 / 24 | 99 | 46.97 s | 3.65 | 1.25 | 2.54 | 75.58 s |
| item 19 alone (`H1BLK_MIN=0`) | 74 / **8** | 99 | 6.73 | 4.95 | 3.37 | 3.54 | 40.14 |
| item 19 + a minimum of 16 | 63 / 16 | 111 | 4.96 | 4.57 | 1.75 | 2.80 | 35.38 |
| **item 19 + item 20 (32)** | 37 / **32** | 196 | **4.66** | 4.35 | 1.49 | 2.77 | **34.62** |

Item 19's hoisted `s` and `t` (`2 nao (nao + nocc)` words per auxiliary
function, twice the block of `j3c` itself) took the memory-short branch's
block from 24 to 8, and at 8 D1, D3 and E's fill — a handful of launches
per block and direction each — pay for three times as many blocks: 5.1 s
of the 40.1.  Taking the block back to 32 from the integral batch costs
the integrals little (D integrals 1.34 → 1.56 s at twice the batches, A
2.1 → 2.0 s) and returns 5.5 s.  32 rather than 16 because 32 is 0.8 s
faster and its batch (37) still holds the largest auxiliary shell.

The unrestricted call on this card is then **34.6 s against 30.5 s on the
full card (1.13x)**, where W30's loop was 75.6 s (2.48x).  The restricted
function does not fall below the minimum at 113 atoms on a 40 GB card (its
block is 32–40 there, §3), so item 20 moves no restricted row here; the
restricted memory-cap ladder of the tests does reach it.

---

## 5. On the full card: W30's threshold

W30 measured the occupied form against the AO form (item 4) and put the
crossover at a score between 1.7e6 and 6.1e6.  Re-measured with item 19
(`H1ATOM=batched` is the AO form, `=occ` the occupied form forced, and
`occ_w30` the occupied form with W30's loop), restricted B3LYP, 80 GB:

| system | basis | score | AO form | occupied, W30 | **occupied, item 19** |
|---|---|---|---|---|---|
| glycine, 10 atoms | SVP | 6.0e4 | 174.9 ms | 204.6 | **150.5** |
| alanine dipeptide, 22 | SVP | 2.6e5 | 270.7 | 335.4 | **205.6** |
| Pt(II) amine, 23, ECP | SVP | 4.5e5 | 459.5 | 575.0 | **409.3** |
| Ni(II) dithiolene²⁻, 21 | SVP | 8.4e5 | 757.3 | 783.8 | **729.2** |
| penicillin V, 42 | SVP | 1.3e6 | 1039.8 | 1071.1 | **926.1** |
| Cu macrocycle⁺, closed shell, 60 | SVP | 1.7e6 | 1697.6 | 1706.6 | **1522.0** |
| glycine | TZVPP | 1.4e6 | 180.8 | 215.0 | **159.5** |
| alanine dipeptide | TZVPP | 6.1e6 | 527.2 | 490.7 | **477.0** |
| Pt(II) amine | TZVPP | 1.0e7 | 1060.2 | 943.4 | **912.8** |
| Ni(II) dithiolene²⁻ | TZVPP | 2.2e7 | 2379.5 | **2046.6** | 2080.6 |
| penicillin V | TZVPP | 3.0e7 | 4250.1 | 3282.1 | 3286.6 |

(One process per system, round robin, minimum of 2–5.)  With item 19 the
occupied form is the fastest or within 2 % of it on every row; below the
old threshold it is 1.04x–1.32x the AO form, which W30's loop lost to by up
to 24 %.  Where it ties W30's loop — the def2-TZVPP rows at 21–42 atoms —
the atoms are large (36–48 AOs) and few, so W30's loop was 42–84 calls per
block and direction, already long GEMMs.  So the threshold is 0: the form
is taken wherever it saves arithmetic (`nao > 2 nocc`, which every row
here has), and `FASTDFHESSIP_H1OCC_SCORE` still overrides it.

The new default against the old one (`w30` = `FASTDFHESSIP_H1D2=atom`,
which restores W30's loop *and* its threshold), and against GPU4PySCF:

| system | basis | GPU4PySCF | W30 (main) | **this round** |
|---|---|---|---|---|
| glycine | SVP | 290.7 ms | 181.3 | 184.7 (tie) |
| alanine dipeptide | SVP | 510.6 | 284.6 | **220.0 (1.29x)** |
| Pt(II) amine | SVP | 908.9 | 486.7 | **438.2 (1.11x)** |
| Ni(II) dithiolene²⁻ | SVP | 1225.8 | 769.6 | 750.9 (1.02x) |
| penicillin V | SVP | 2053.8 | 1071.6 | **966.0 (1.11x)** |
| Cu macrocycle⁺ | SVP | 3770.2 | 1714.5 | **1546.4 (1.11x)** |
| glycine | TZVPP | 292.1 | 186.9 | **166.3 (1.12x)** |
| alanine dipeptide | TZVPP | 958.9 | 520.8 | **463.8 (1.12x)** |
| Pt(II) amine | TZVPP | 1853.4 | 981.5 | 939.8 (1.04x) |
| Ni(II) dithiolene²⁻ | TZVPP | 4042.7 | 2078.6 | 2105.0 (tie) |
| penicillin V | TZVPP | 7287.5 | 3302.8 | 3353.7 (tie) |
| Co(III) complex⁺, 113 | SVP | — | 14467 | **13748 (1.05x)** |
| taxol, 113 | SVP | — | 17793 | **17042 (1.04x)** |

and the unrestricted twin (its threshold was the same 4e6):

| system | basis | GPU4PySCF | W30 (main) | **this round** |
|---|---|---|---|---|
| FeCl₄⁻ sextet, 5 atoms | SVP | 245.8 ms | 193.5 | 189.2 (1.02x) |
| Ni(II) dithiolene⁻ doublet, 21 | SVP | 1551.0 | 981.3 | 980.4 (tie) |
| Pt(II) amine triplet, 23 | SVP | 948.7 | 539.4 | **492.4 (1.10x)** |
| penicillin V⁺ doublet, 42 | SVP | 2774.7 | 1422.8 | **1266.7 (1.12x)** |
| Cu(II) macrocycle²⁺ quartet, 60 | SVP | 4831.6 | 2364.4 | **2162.4 (1.09x)** |
| penicillin V⁺ | TZVPP | 11027 | 4747.3 | 4644.0 (1.02x) |
| taxol⁺ doublet, 113 | SVP | — | 32392 | **30549 (1.06x)** |
| Co(II) sextet, 113 | SVP | — | 26138 | **23867 (1.10x)** |

A pure functional's h1 has no exchange term and no D2 (W30's item 14), and
is unchanged: DF-PBE penicillin V 639.9 → 645.7 ms, penicillin V⁺ 721.4 →
756.1 ms, inside the spread of a 0.7 s call on this node.

What depends on size here, and what does not:

* **The number of launches** is `2 x` the number of distinct atom sizes per
  block and direction — 2 (organic def2-SVP: H and heavy atoms) to 4
  (a metal, a halogen or sulfur, and the rest) — and does not grow with
  the molecule.  The batch count per launch does.
* **The GEMM shape** is `nocc x nao x (nao_A dk)`: `dk` is the block,
  `nao_A` the element's shell set.  The probe's `dk = 16` rows are where it
  runs worst, 6–10 TF/s — still 5–20x W30's loop there.
* **The transposed block** costs nothing: it is the same buffer, scattered
  with the address arrays swapped; the short last block of a batch costs a
  zeroing of `nao² dk` twice per batch.
* **Item 20's minimum** is a count of auxiliary functions (32), not a
  size: it only moves memory from the integral batch to the block when the
  block is below it, and never below the largest auxiliary shell.

---

## 6. Correctness

`tests/test_dfhessip.py` and `tests/test_dfuhessip.py`, element-wise
against GPU4PySCF's own function objects on one converged SCF, relative to
the largest element:

| | worst |
|---|---|
| restricted h1, ethanol, B3LYP / HF / `k = 0`: every switch of the list, now including `H1ATOM=occ` (item 19), `H1ATOM=occ H1D2=atom`, `H1ATOM=occ H1J=upstream` and `H1D2=atom` | 1.9e-15 |
| restricted h1, the occupied form at blocks of 40 and 24 (a short last block in every batch), item 19 and W30's loop, and with 5 occupied orbitals per batch (`pri_full`) | 1.6e-15 |
| restricted h1, the memory-short branch (`H1KAO=0`) with item 19, W30's loop, blocks of 24, and blocks of 24 with 5 occupied per batch | 1.5e-15 |
| restricted h1, the memory-cap ladder: item 20 reached at 17.7–28.9 MB | 3.4e-15 |
| unrestricted h1, ethanol⁺, the same switches, blocks of 40 and 24, and the memory-short branch at blocks of 24 | 1.5e-15 |
| unrestricted h1, the memory-cap ladder: item 20 reached at 24.6–38.4 MB | 1.7e-15 |
| whole ethanol B3LYP Hessian, patched against unpatched: RKS / UKS (ethanol⁺) | 1.2e-11 / 2.4e-12 Eh/bohr² |

and on every system of §5 against GPU4PySCF's function: 7e-16 – 2.7e-14
relative, the occupied form's rows up to 3x W30's AO-form rows on the
same system (the Ni(II) dithiolene 8.7e-15 against 3.1e-15, the Cu
macrocycle 1.3e-14 against 6.5e-15): the order of the sums changes, and
W30's own occupied-form column read the same (1.9e-14 on the Cu row).
The 113-atom ballast runs have no GPU4PySCF column (upstream's twin does
not fit, and the restricted one takes 34 s a call); the §4 run was checked
against its own first column in the same process — W30's loop 3.0e-15
from item 19 + 20, the other two block minimums 4.4e-15 and 4.7e-15 —
and the §3 restricted runs were timing-only (`--no-check`).

---

## 7. What is left, and what was not done

* **The unrestricted twin does not fit a 40 GB card at 113 atoms**, in its
  own budget: `dm_3c` and `dm_oo` for both spins, `2 naux nocc (nao +
  nocc)` words, are 29 GB at taxol⁺ before a block is sized.  The
  restricted function batches the occupied index when that does not fit
  (`num_occ_batches`); the twin has no such loop.  That is the next thing
  on a 40 GB node.
* **Whole Hessians were not re-run.**  The h1 call is 3–15 % of a Hessian
  on the full card (W30), so 1.04x–1.29x of it is 1–4 % of the whole; on a
  40 GB card the ip2 term and the CPHF have their own block sizes, which
  nobody has measured there.
* **The per-launch host cost of `contract` is 87 µs** on a compute node
  (130 µs on the login node's PCIe card) against 23 µs for a bare cuBLAS
  call.  Any loop in this repository that issues a `contract` per atom per
  block pays it, and D2 was the largest one; stage E's per-auxiliary-atom
  loop (one set of contractions per auxiliary atom and batch) is the next
  the stage tables show, 1.0 s of 17 at 113 atoms and not growing with the
  card.

---

## 8. Tools

```bash
python tests/test_dfhessip.py --no-hessian
python tests/test_dfuhessip.py --no-hessian
python benchmarks/df_h1d2_probe.py --xyz benchmarks/molecules/taxol.xyz \
    --basis def2-svp [--dk 16,32,64,128,256,512] [--nspin 2]
FASTDFHESSIP_TIME=1 python benchmarks/df_hess_terms_probe.py --patch all \
    --terms h1 --variants ours --reps 2 --no-check --ballast 40 \
    --xyz benchmarks/molecules/taxol.xyz --basis def2-svp --xc b3lyp \
    --env 'atom:FASTDFHESSIP_H1D2=atom'
python benchmarks/df_hess_terms_probe.py --patch all --terms h1 \
    --variants stock,ours --xyz ... --basis ... --xc b3lyp \
    --env 'ao:FASTDFHESSIP_H1ATOM=batched,occ:FASTDFHESSIP_H1ATOM=occ,occ_w30:FASTDFHESSIP_H1ATOM=occ;FASTDFHESSIP_H1D2=atom'
```

Switches: `FASTDFHESSIP_H1D2=atom` (item 19 off, and W30's threshold
back), `FASTDFHESSIP_H1BLK_MIN=<n>` (item 20's minimum, 32; 0 is off),
`FASTDFHESSIP_H1BLK=<n>` (test-only: cap the D loop's block).
`df_hess_terms_probe.py --ballast <GB>` holds that much device memory for
the whole run; without a `stock` variant its first variant is the reference
the others are checked against.
