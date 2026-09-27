# The restricted h1's stage C was already at the GEMM rate — and for a pure functional it did not need its GEMM

`docs/DF_WORK_LIST.md` **W32**, found by W30.  After W30's occupied form the
largest stage left in the restricted `h1` term at 113 atoms was C — for each
*auxiliary* atom `A`, `j3c_1 = j2c_10[:, A] · dm_3c` and its products with
`A`'s rows of `dm_3c` (the exchange matrix) and with `auxvec` (the Coulomb
term) — at 5.6–6.8 s of a 19.6 s call on taxol.  The entry priced it at
6.1e13 flops, read the stage as running at about 10 TF/s against the 15
TF/s W8 measured for the DF Hessian's large GEMMs, blamed one launch per
auxiliary atom (`naux_A` ≈ 50 rows), and asked for the atoms to be batched
into launches of a few thousand rows.  Its falsification: *the per-atom
launches are already at 15 TF/s — then C is arithmetic*.

**For a hybrid the falsification fires**, because the price was wrong: the
auxiliary index of this function is Cartesian, 6598 functions on taxol and
not 5598, and C is **8.3e13** flops.  One launch per atom ran its GEMM at
**15.7 TF/s**.  Grouping consecutive atoms into GEMMs of up to 3072 rows
takes it to **18.9 TF/s** — the A100's FP64 tensor-core rate is 19.5 — and
the exchange contraction beside it from 12.7 to 17.3 TF/s as cuBLAS calls;
that is stage C **5.6–5.8 → 4.56 s (1.23x–1.28x)** at 113 atoms, and there
is nothing left in it but arithmetic.  **For a pure functional C did not
need its GEMM at all**: after W30's item 14 a `k_factor = 0` call formed
`j3c_1` only to contract it with `auxvec`, and that contraction can be done
first — `sum_{r in A} auxvec[r] j2c_10[x, r]` is one row per atom — so the
stage is one `(3 natm) x naux` matrix and one GEMM with `naux / natm` ≈ 58
times less arithmetic: **4.60 s → 0.15 s**, and DF-PBE's h1 call on taxol is
**1.53x** (13.12 → 8.58 s).

---

## 0. Summary

* **Item 18: stage C grouped** (`fastdfhessip._h1_stage_c`, §1), used by
  both spin cases.  Consecutive auxiliary atoms form a group of up to
  `FASTDFHESSIP_H1C_ROWS / 3` (1024) functions, bounded by 80 % of what is
  free when the stage starts; the group's `j3c_1` is one cuBLAS GEMM per
  spin, each atom's exchange contraction `3 ns` cuBLAS GEMMs on the
  group's buffer, and the Coulomb term never touches `j3c_1`.
  `FASTDFHESSIP_H1C=atom` is W30's per-atom loop.
* **Where the time was** (§2): on taxol's shapes the per-atom GEMM runs at
  15.7 TF/s and the exchange contraction at 12.7; the `take` of `dm_3c` rows
  is 19 ms of 5.5 s and the Coulomb pass 49 ms.  Grouped, the GEMM is 18.5–19.0
  TF/s from 2400 rows up (`benchmarks/df_h1c_probe.py`).
* **Restricted h1, B3LYP** (§3): stage C 1.23x–1.28x at 113 atoms and the
  call **1.075x** (18.71 → 17.40 s); 1.05x on the 113-atom Co complex; ties
  within ±3 % from 10 to 60 atoms and in def2-TZVPP, where C is a small
  share of the call.
* **DF-PBE h1** (§3): **1.53x** on taxol, **1.48x** on the Co complex,
  1.00x–1.23x at 10–60 atoms in def2-SVP, 1.04x–1.18x in def2-TZVPP; the
  call is now **2.1x–3.4x** GPU4PySCF's.
* **The unrestricted twin** (§3): DF-PBE 1.02x–1.32x at 5–60 atoms and
  **1.75x on taxol⁺** (21.5 → 12.3 s, stage C 9.04 → 0.18 s); UKS B3LYP
  ties at 5–60 atoms and in def2-TZVPP, 1.025x on taxol⁺.
* **A 40 GB card** (§3.1): 13 groups instead of 7, C 1.18x; the call there
  is dominated by W30's D2 at a small block (13.1 s), a finding.
* **Whole Hessians** (§4) move by what h1 was: DF-PBE on taxol **145.3 →
  140.4 s** (h1 11.7 → 7.3 s); at 42 atoms the h1 call's 0.15–0.2 s is
  inside a run's spread.
* **Correctness** (§5): element-wise against GPU4PySCF's own function,
  every new path and switch, both spin cases: **1.1e-15** worst.

Everything is one A100-SXM4-80GB of a Perlmutter node (an interactive
allocation, one process at a time on its GPU), GPU4PySCF 1.8.1, def2-SVP
with its jkfit set unless a row says otherwise, level-3 grids, `patch_all()`
on both sides.  Function tables are `benchmarks/df_hess_terms_probe.py`,
variants round robin in one process against upstream's function object,
minimum of 1–5.

---

## 1. What changed

`fastdfhessip._get_veff` stage C, for each auxiliary atom `A` (rows `r` of
the metric derivative `j2c_10`, sorted by atom):

    j3c_1[x,r,i,q] = sum_s j2c_10[x,r,s] dm_3c[s,i,q]                 (GEMM)
    X_A[x,p,q]    += sum_{r in A, i} j3c_1[x,r,i,p] dm_3c[r,i,q]       (exchange)
    vhf_atm[A,x]  += sum_{r in A} auxvec[r] j3c_1[x,r]
                   + sum_{r in A} (j2c_10[x,r] . auxvec) dm_3c[r]      (Coulomb)

Three changes (`_h1_stage_c`, shared by the restricted function and
`_dfhessip_u`'s twin through a spin index `ns` of 1 or 2):

1. **The GEMM grouped.**  Consecutive auxiliary atoms are taken together up
   to 1024 functions (3072 GEMM rows), and the group's rows of `j2c_10` are
   gathered `(x, r)`-major so that one cuBLAS call per spin gives `j3c_1`
   for the whole group in the `(x, r, i, q)` layout the per-atom code had.
   The group's operands — the gathered rows, `j3c_1`, the group's rows of
   `dm_3c` (one `take`), and in the memory-short branch the same for
   `dm_oo` — are one buffer of at most 80 % of the free memory, allocated
   after the per-atom AO matrices and released before stage D allocates its
   own (§1.1).  Never less than the largest atom, which is what W30's loop
   held.
2. **The exchange contraction as cuBLAS.**  `'xrip,riq->xpq'` per atom is
   three GEMMs `A_x^T B` with `K = naux_A nocc`, both operands contiguous
   runs of the group's buffers; `_gemm_rm` passes leading dimensions to
   `dgemm` so the unrestricted twin's spin-strided operands go in without a
   copy.  The memory-short branch (the per-atom AO matrices do not fit) is
   the same two GEMMs per direction and spin on the occupied side.
3. **The Coulomb term without `j3c_1`.**  Both halves are linear in the
   rows of `j2c_10`: a `(natm, 3, naux)` matrix `W` with
   `W[A,x,s] = sum_{r in A} auxvec[r] j2c_10[x,r,s]` (a GEMM against the
   atom-of-function indicator) plus `(j2c_10[x,r] . auxvec)` scattered to
   `(A(r), aux_idx[r])`, and then one contraction `W · dm_3c` into every
   atom's rows at once.  `2·3·natm·naux·nocc·nao` flops against the
   per-atom form's `2·3·naux²·nocc·nao`.  With `k_factor = 0` (a pure
   functional, W30's item 14) there is no exchange term, so this is all of
   stage C.

### 1.1 Memory

The first version carved the group from stage D's buffer, allocated early
for the purpose.  That buffer is sized as the whole of the function's
remaining budget, and the budget does not count `j2c_10` (3 naux² words,
6 naux² while it is re-ordered), which upstream frees before D — so the
113-atom Co complex in B3LYP ran out of memory in stage C.  The group is now
sized from `get_avail_mem()` at the start of the stage and released before D
allocates; on a card with less memory the group is smaller, not refused
(§3.1).

## 2. Where the time was

`benchmarks/df_h1c_probe.py` builds a molecule's real shapes — `nao` and
`naux` Cartesian, the auxiliary functions grouped by atom — with random
operands and times the stage's pieces.  Taxol, def2-SVP (`nao` 1185, `nocc`
226, `naux` 6598, 20–93 per atom):

| piece, one launch per atom | time | rate |
|---|---|---|
| all of it (take + GEMM + exchange + Coulomb) | 5505 ms | |
| `j3c_1` GEMM (cuTENSOR `'xrs,siq->xriq'`) | 4460 ms | **15.7 TF/s** |
| the same as one cuBLAS 2-D GEMM, `M = 3 naux_A` | 4593 ms | 15.2 TF/s |
| exchange `'xrip,riq->xpq'` (cuTENSOR) | 990 ms | 12.7 TF/s |
| exchange as 3 cuBLAS GEMMs per atom | **725 ms** | **17.3 TF/s** |
| Coulomb terms | 49 ms | |
| `take` of the atom's `dm_3c` rows | 19 ms | |

| grouped GEMM, rows per launch | launches | buffer | GEMM | rate |
|---|---|---|---|---|
| ≤ 300 | 73 | 0.64 GB | 4360 ms | 16.0 TF/s |
| ≤ 600 | 37 | 1.29 GB | 4335 ms | 16.1 TF/s |
| ≤ 1200 | 19 | 2.57 GB | 3959 ms | 17.7 TF/s |
| ≤ 2400 | 9 | 5.05 GB | 3773 ms | 18.5 TF/s |
| ≤ 4800 | 5 | 10.17 GB | 3686 ms | 19.0 TF/s |

So the per-atom launches were not slow launches: at `N = nocc·nao` = 268k
columns and `K = naux` a 150-row GEMM already fills the card, and what
grouping buys is the last 20 % to the tensor cores' rate.  The entry's "10
TF/s" was the stage's time over a flop count with the spherical auxiliary
size in it.  In the real call (`FASTDFHESSIP_TIME=1`, second call of two,
taxol B3LYP), the grouped stage is GEMM **3.70–3.72 s** (18.8–18.9 TF/s),
exchange 0.72 s and Coulomb plus `int2c2e_ip1` 0.12 s — 4.56 s against
5.61–5.84 s per atom; 1536 rows reads 3.89–3.90 s and 6144 rows 3.70 s, so
the default is 3072.

## 3. The functions

h1 against GPU4PySCF's function and W30's per-atom stage C
(`--env 'atom:FASTDFHESSIP_H1C=atom'`), restricted:

| system | basis | B3LYP: GPU4PySCF | W30 | **W32** | DF-PBE: GPU4PySCF | W30 | **W32** | groups |
|---|---|---|---|---|---|---|---|---|
| glycine, 10 atoms | SVP | 291.5 ms | 177.8 | 177.6 | 292.9 | 139.7 | **131.3** (1.06x) | 1 |
| alanine dipeptide, 22 | SVP | 507.7 | 280.6 | 265.0 | 502.9 | 186.0 | 186.0 | 2 |
| Pt(II) amine, 23, ECP | SVP | 883.5 | 469.8 | 452.9 | 891.1 | 385.5 | 369.5 | 2 |
| Ni(II) dithiolene²⁻, 21 | SVP | 1224.9 | 772.1 | 771.1 | 1117.3 | 643.2 | **524.8** (1.23x) | 3 |
| penicillin V, 42 | SVP | 2060.5 | 1093.2 | 1053.4 | 1911.3 | 790.2 | **677.5** (1.17x) | 3 |
| Cu macrocycle⁺, 60 | SVP | 3832.5 | 1736.9 | 1744.3 | 3441.4 | 1215.7 | **1007.4** (1.21x) | 4 |
| alanine dipeptide | TZVPP | 956.7 | 490.3 | 486.0 | 889.8 | 389.8 | 376.3 | 2 |
| penicillin V | TZVPP | 8195.3 | 3414.8 | 3482.4 | 6126.6 | 2454.3 | **2189.8** (1.12x) | 3 |
| Ni(II) dithiolene²⁻ | TZVPP | 4279.1 | 2129.4 | 2099.1 | 3421.4 | 1574.8 | **1329.8** (1.18x) | 3 |
| Co complex⁺, 113 | SVP | 27861 | 14681 | **14005** (1.05x) | 22952 | 10391 | **7015** (1.48x) | 7 |
| taxol, 113 | SVP | — | 18709 | **17400** (1.075x) | — | 13119 | **8582** (1.53x) | 7 |

(Minimum of 5 at 10 atoms down to 1 at 113; taxol from the stage-table run
of §2, four variants round robin, minimum of 2.)  Against GPU4PySCF the
restricted h1 is now 1.6x–2.4x for B3LYP and **2.1x–3.4x for DF-PBE**.
The B3LYP rows below 113 atoms move by less than their own spread (±3 %):
C is `naux²`, a few per cent of the call there.

The unrestricted twin:

| system | basis | UKS B3LYP: GPU4PySCF | W30 | **W32** | UKS DF-PBE: GPU4PySCF | W30 | **W32** |
|---|---|---|---|---|---|---|---|
| FeCl₄⁻ sextet, 5 atoms | SVP | 247.4 ms | 191.4 | 197.0 | 244.9 | 169.2 | 165.8 |
| Ni(II) dithiolene⁻ doublet, 21 | SVP | 1555.1 | 1005.2 | 1038.0 | 1377.7 | 755.2 | **579.0** (1.30x) |
| Pt(II) amine triplet, 23 | SVP | 964.3 | 546.3 | 542.9 | 928.2 | 402.3 | **365.7** (1.10x) |
| penicillin V⁺ doublet, 42 | SVP | 2761.4 | 1459.8 | 1446.8 | 2354.6 | 949.2 | **724.7** (1.31x) |
| Cu(II) macrocycle²⁺ quartet, 60 | SVP | 4824.9 | 2371.9 | 2372.2 | 4228.1 | 1555.8 | **1178.6** (1.32x) |
| penicillin V⁺ | TZVPP | 10050 | 4699.9 | 4753.9 | 6512.2 | 2860.6 | **2237.3** (1.28x) |
| taxol⁺ doublet, 113 | SVP | — | 33776 | **32939** (1.025x) | — | 21532 | **12338** (1.75x) |

(Taxol⁺ without GPU4PySCF's column: upstream's unrestricted function runs out
of memory at 113 atoms, W25.  Stage C there: UKS B3LYP 10.54–10.67 s → 9.7–10.1 s,
its GEMM at 17.5 TF/s on `N = 2 nocc nao`; UKS DF-PBE **9.04 s → 0.18 s**.)

(A first pass of the unrestricted rows on a GPU shared with another job
read FeCl₄⁻'s B3LYP call at 0.82x of W30; re-measured alone it is the row
above, and its stage table puts C at 44 → 45 ms of a 270 ms call.  The
Ni(II) doublet's B3LYP row is 3 % the wrong way in both passes; its stage
table has C at 199 → 180 ms and the call at 1096 → 1073 ms, so the 3 % is
the call's spread, not the stage.)

### 3.1 A smaller card

`FASTDFHESSIP_MEM` caps what the sizing reads, not what the card has left
by stage C, so it does not emulate a 40 GB card here (the group count stays
at 7).  A 40 GB cupy array allocated before the run does: 44.6 GB free,
taxol B3LYP, second call of two:

| | per atom (W30) | grouped |
|---|---|---|
| groups | 113 | **13** (≈ 510 functions, 1530 rows) |
| stage C | 5.50–5.56 s | **4.66–4.73 s** (GEMM 3.83–3.86 s) |
| h1 call | 30.73 s | **28.68 s** |

The group halves and C keeps 1.18x of its 1.23x–1.28x.  The call itself is
11 s slower than on the full card, and that is **D2**: 2.5 s with 80 GB,
**13.1 s** with 40 — W30's occupied form at the block size the smaller
budget leaves (§6).

## 4. Whole Hessians

`benchmarks/df_hess_anatomy.py --patch all --rounds 2`, `main` (W30/W28)
and this round in alternating processes on one GPU, the pair run twice,
round 1 of each:

| system | basis | xc | before | W32 | h1, before → W32 | traces |
|---|---|---|---|---|---|---|
| penicillin V, 42 atoms | SVP | PBE | 6.35 / 6.34 s | 6.20 / 6.27 s | 0.80 / 0.76 → **0.65 / 0.63 s** | 40.76471284(56/51) / …(52/52) |
| penicillin V | TZVPP | B3LYP | 40.24 / 40.95 | 40.42 / 40.03 | 3.31 / 3.30 → 3.32 / 3.25 | 39.777317038(4/4) / …(4/0) |
| penicillin V⁺, UKS | SVP | PBE | 13.83 / 13.93 | 14.49 / 13.63 | 0.94 / 0.94 → **0.73 / 0.73** | see below |
| taxol, 113 (one pair, round 0) | SVP | PBE | 145.25 | **140.39** | 11.66 → **7.28** | 107.912835764(4) / …(7) |

h1 is 5–13 % of a whole Hessian at 42 atoms, so a 20 % gain in it is inside
the whole run's spread there; at 113 atoms DF-PBE's h1 was 8 % of the
Hessian and 4.4 s of it goes (the ip2 row moved by 0.8 s too, which is not
this round's code — the two processes differ in nothing else — and is read
as spread).  The CPHF (112 s, of which the fxc grid half is 40 s) is the rest.  The UKS DF-PBE traces spread by 2e-5 on both
sides (12.4236543605 / 12.4236416053 before, 12.4236341895 / 12.4236499694
after): that SCF does not land on the same density twice, in GPU4PySCF
itself (`docs/README_fastdfh1occ.md` §4, W25's §4.1).

## 5. Correctness

`tests/test_dfhessip.py` and `tests/test_dfuhessip.py`, element-wise against
GPU4PySCF's own function objects on one converged SCF, relative to the
largest element:

| | worst |
|---|---|
| restricted h1, ethanol, B3LYP / HF / `k = 0`: the grouped default, `H1C=atom`, one atom a group (`H1C_ROWS=3`), groups of a few atoms (`=240`) | 6.2e-16 |
| restricted h1, water, the long-range operator, the same three | 8.4e-16 |
| restricted h1, the memory-short exchange branch (forced, `FASTDFHESSIP_H1KAO=0`), grouped, per atom, `=240`, and with 5 occupied orbitals per batch | 7.5e-16 |
| unrestricted h1, ethanol⁺, the same switches, `k = 0` | 1.1e-15 |
| unrestricted h1 without the per-atom AO matrices, by memory cap and forced | 3.0e-15 |

and on every system of §3 against GPU4PySCF's function: 2e-16 – 3e-14
relative, the same as W30's column beside it.  `FASTDFHESSIP_H1KAO=0` is new
and test-only: the restricted memory-short branch was not reachable by any
cap ethanol's sizing survives, so it had never been run.

## 6. What is left, and what was not done

* **Stage C is arithmetic now** — 18.9 TF/s on a 19.5 TF/s card.  The
  entry's fallback, `j2c_10`'s sparsity, is weak: the derivative of `(r|s)`
  on `r`'s centre decays as `1/R²` between two charge-carrying functions, so
  at taxol's 20 Å nothing drops below a useful threshold.  What would
  remove work is a far-field (multipole) form of `j2c_10 · dm_3c`, a
  project of its own.
* **Antisymmetry does not halve it.**  `j2c_10[x,r,s] = -j2c_10[x,s,r]`,
  but each auxiliary function needs its own full sum over `s`, so no
  product is shared between the two orders.
* **D2 on a smaller card.**  With 40 GB (§3.1) the restricted call on taxol
  is 28.7 s against 17.4 s, and 10.6 of the 11.3 s are D2, W30's per-atom
  exchange term on the occupied side: its auxiliary block shrinks with the
  budget, and it launches `2 natm` contractions per block and direction.
  Not this round's stage; it is the first thing to look at on an A100-40GB.
* **Where the restricted h1 is now** at 113 atoms (B3LYP, 17.4 s): D1 2.2 s,
  D2 2.5 s, B 2.2 s, A 0.9–1.8 s, D and E's integral fills 2.4 s, C 4.6 s.
  A's spread (0.9–2.3 s for the same call) is the three-centre fill and is
  not this round's.

## 7. Tools

```bash
python tests/test_dfhessip.py --no-hessian
python tests/test_dfuhessip.py --no-hessian
python benchmarks/df_h1c_probe.py --xyz benchmarks/molecules/taxol.xyz \
    --basis def2-svp [--rows 300,600,1200,2400,4800] [--nspin 2]
python benchmarks/df_hess_terms_probe.py --xyz ... --basis ... --xc b3lyp|pbe \
    --patch all --terms h1 --variants stock,ours --env 'atom:FASTDFHESSIP_H1C=atom'
FASTDFHESSIP_TIME=1 python benchmarks/df_hess_terms_probe.py ...   # C gemm, C K, C J
```
