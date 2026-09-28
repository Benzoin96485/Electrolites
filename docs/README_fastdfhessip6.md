# The ip2 call's D stages at 113 atoms: at rate once measured, and what was not arithmetic

`docs/DF_WORK_LIST.md` **W37**, found by W34.  With W31's kernel at 1.0 s
and W34's allocation at 0.2 s of a 21 s `_jk_energy_per_atom` call on
taxol, 13.7 s was the auxiliary-index contractions — D5 per atom 4.7 s, D2
`h_aux` 3.4 s, the two `L^-1` triangular multiplies 3.5 s, D4 2.1 s — and
the Co(II) sextet's unrestricted call was worse: D5 16.6 s and D3
`j3c_100` 9.5 s of 42.4 s.  The entry asked for each stage's rate against
its flop count at 113 atoms, restricted and unrestricted, and for whatever
ran well below the GEMM rate to be batched or fused; success was the D
stages at the card's rate and the ip2 call 1.3x or more at 113 atoms in both
spin cases, measured inside a whole Hessian; the falsification, that they
already ran at 15 TF/s or more — in which case the arithmetic is the cost,
and the next thing is the spherical auxiliary index in D5's unrestricted
form, which item 8 did not reach.

**Both halves of the falsification were true, and both were acted on.**
Measured with Nsight Systems kernel by kernel (§1), most of the restricted
call was already at 15–17 TF/s — D2 at 16.3, the triangular multiplies at
15.1, D4 at 12.8 with its memory passes — and the stages that were not were
two: D5, whose two per-atom `sxy` reductions ran through cuTENSOR at
**0.12 TF/s** and whose per-atom products re-read their whole operand for a
14-column output, and the fused first pass (D1), which repeats a
`2 nao² nocc` contraction the second pass has already done.  So the
restricted call got less arithmetic where the arithmetic was the cost
(**item 22**: the occupied pair index packed where it is symmetric, halving
D2 and stage B) and the two stages that were not at rate were fixed
(**items 24, 26, 28**).  The unrestricted call got what the falsification
named: **item 25**, W24's Cholesky form on the spin index, which turns
D3's and D5's eigenbasis GEMMs — 26 of its 42 s — into a pass over memory
and two triangular multiplies.

| 113 atoms, ip2 call | before (W34) | after | |
|---|---|---|---|
| taxol, RKS B3LYP (§3.1) | 21.21 s | **16.59 s** | **1.28x** |
| Co(II) sextet, UKS B3LYP (§3.1) | 42.31 s | **25.81 s** | **1.64x** |
| taxol, inside a whole Hessian (§3.3) | 20.81 s | **16.65 s** | **1.25x** |
| Co(II) sextet, inside a whole Hessian (§3.3) | 42.25 s | **25.97 s** | **1.63x** |

**The success bar is met unrestricted and missed restricted**: 1.25x
inside a Hessian (1.28x in the call) against 1.3x.  What is left of the restricted call is at 15–18 TF/s stage
by stage (§4), so the rest of the bar is arithmetic, and the one measured
lead that would remove some — the AO-pair screening that leaves half of
every `(nao, nao)` block zero in D3, stage A and the pseudo-density — is
W39.

---

## 0. Summary

* **Where the time was** (§1): `benchmarks/df_ip2_stage_profile.py` reads
  an nsys report with an NVTX mark at the end of every stage
  (`FASTDFHESSIP_TIME=1 FASTDFHESSIP_NVTX=1`) and prints each stage's
  kernels.  Restricted, taxol: D2 3.43 s at 16.3 TF/s, D8 3.4 s at
  15.1/14.3 TF/s, D4 2.09 s at 12.8 TF/s — at rate; D5 4.76 s at 11 TF/s,
  of which **3.6 s** were 3,017 cuTENSOR launches; D1 1.46 s, a
  contraction D3 repeats.
* **Item 22** (§2.1): `dm_oo`, `j3c_oo`, `j3c_oo1` and `j3c_oo1p` are
  symmetric in `(i, j)`, and stage B and the `h_aux` block's four big
  products either sum over `(i, j)` or keep it free; on the `i <= j` pairs
  weighted `√2` off the diagonal they are the same products at
  `(nocc+1)/2nocc` of the arithmetic.  D2 3.43 → **1.98 s**, stage B
  0.66 → 0.48 s.  Not on the long-range operator (§5.2).
* **Item 24** (§2.2): the fused first pass's `(1|00)` block is
  `-(C^T M C + h.c.)` from the second pass's own `M C`, because the
  second pass's scatter makes `M + M^T` the `ipaux` block exactly.  D1's
  1.46 s becomes 0.15 s inside D3.
* **Item 25** (§2.3): item 8 on the spin index.  Co(II) sextet: D3
  `j3c_100` 9.47 → 2.07 s, D5 16.47 → 5.27 s (with item 26), plus the two
  triangular multiplies (2.6 + 2.5 s) the eigenbasis did not need.  Where
  both spins' spherical `(10|0)` do not fit in one chunk (the Co sextet on
  80 GB: 2 × 29 GB) the second pass runs once per spin — as many integral
  walks as the eigen form's two chunks.
* **Item 26** (§2.4): D5 by atom groups — two strided-batched GEMMs per
  group and two reduction kernels at memory speed around the per-atom
  `nsph × 3nsph` product, which was already at 18.6 TF/s.  Taxol D5 4.76 →
  **3.38 s**.
* **Item 28** (§2.5): the second pass's scatter in one kernel: 0.15 s
  (taxol), 0.35 s (Co).
* **Item 27, measured and off** (§2.6): D8's `dtrmm` blocked into GEMMs,
  15.2–15.4 against 15.1–15.2 TF/s.
* **Every size and class** (§3.2): the call is faster or unchanged from
  glycine to taxol, in def2-TZVPP, and on five open-shell metal complexes;
  on a 40 GB card (§3.4), where item 8 does not fit in either spin case, 1.06x–1.07x from item 22 alone.
* **Correctness** (§5): every new branch within 6e-13 of GPU4PySCF on the
  test molecules; the Cholesky twin differs from the eigenbasis by what a
  one-ulp metric perturbation moves it (W24's rule, §5.1); whole Hessians
  7e-10 (taxol) and 3.6e-9 Eh/bohr² (Co sextet) apart.

Everything is one A100-SXM4-80GB, GPU4PySCF 1.8.1, def2-SVP and its jkfit
set unless a row says otherwise, `patch_all()` on both sides, a whole
`--qos=debug` node with one GPU visible.  "Before" is this tree with the
new items off (`FASTDFHESSIP_PACK=0 FASTDFHESSIP_D1D3=0 FASTDFHESSIP_D5=atom
FASTDFHESSIP_D3SCATTER=0 FASTDFHESSIP_UCHOL=0`), which is W34's code.

---

## 1. The stages, kernel by kernel

`FASTDFHESSIP_NVTX=1` (with `FASTDFHESSIP_TIME=1`) puts an NVTX mark at
the end of every stage of the stage table, after its synchronise, and
`benchmarks/df_ip2_stage_profile.py` bins every kernel of the report
between two marks into the second one.  Taxol, W34's code, the timed call
of `df_hess_terms_probe.py --nvtx` (21.17 s; 20.33 s of it GPU-busy):

| stage | wall | flops (model) | rate | what ran |
|---|---|---|---|---|
| D5 per-atom | 4.76 s | 5.2e13 | 11 TF/s | 3,017 cuTENSOR launches (`Kernel2`, the product among them) 3.64 s, cutlass 0.93 s |
| D2 h_aux | 3.43 s | 5.6e13 | 16.3 | 19 large `dsyrk`/GEMM launches, 3.24 s |
| D4 oo_atm | 2.09 s | 2.7e13 | 12.8 | `dsyrk` + GEMM 1.69 s, `hadamard_t` 0.31 s |
| D8 L^-T Y | 1.76 s | 2.5e13 | 14.3 | `dtrmm` |
| D8 L^-1 K | 1.67 s | 2.5e13 | 15.1 | `dtrmm` |
| D1 ipaux (fused) | 1.46 s | 1.5e13 | — | cutlass 0.90 s, the Coulomb trace and the symmetric fill 0.43 s |
| D3 j3c_100 | 1.39 s | 1.3e13 | 15.7 (the GEMM) | GEMM 0.80 s, the chol block pass 0.23 s, two indexed scatters 0.19 s, `dk += di`, `di *= -1` 0.17 s |
| D3 integrals | 1.10 s | | | `int3c2e_ip1`, `int3c2e_ipaux` |
| C ip2 kernel | 1.01 s | | | W31's kernel |
| A j3c_oo | 1.02 s | | | `int3c2e` 0.49 s, GEMM 0.30 s |
| B metric | 0.71 s | 7.5e12 | 18 (busy part) | 0.41 s of GEMM, 0.23 s idle (`_SphAux` on the host) |
| C pseudo-dm | 0.51 s | | | |

(`nao` 1185 Cartesian, `nocc` 226, `naux` 6598, `nsph` 5598; flops are
the stage's contractions counted from their shapes, `dsyrk` at `n² k`.)
The reference rate is what the libraries reach on these shapes on this
card: 18.3–18.7 TF/s for D5's product, 15.1–15.4 for the triangular
multiply (§2.6).

So the entry's two cases split the stages between them.  D2, D8 and D4 run
at the card's rate for what they compute, and making them faster means
computing less (§2.1).  D5 does not: its product runs at rate, and around
it are four small contractions per atom that cost almost half as much
again — `benchmarks/df_d5_bench.py --old` at taxol's sizes, one 14-AO
atom:

| D5 contraction (per atom) | time | rate |
|---|---|---|
| `t1 = contract('ysji,ui->ysuj')` | 5.95 ms | 4.0 TF/s — reads all 6.9 GB of `t1s` for 14 columns |
| `contract('xsuj,ysuj->sxy')` | 2.0 ms | — |
| `pa = contract('sji,ui->suj')` | 2.35 ms | 4.0 TF/s — reads all of `dm_oo` |
| `m = contract('suj,xtuj->xst')` | 31.9 ms | 18.6 TF/s |
| `contract('xst,yst->sxy')` | 3.45 ms | 1.5 GB in 3.45 ms, 440 GB/s |

13.8 ms per atom around a 31.9 ms product.  And D1 is not arithmetic that has to be done at all (§2.2).

The unrestricted call has no item 8, so its D3 and D5 are the eigenbasis
GEMMs of W25: `j3c_100 = V^T K` in D3 (1.1e14 flops, 9.5 s) and two
`naux × nsph` contractions per atom in D5 (2.2e14, 16.6 s) — GEMMs at
11–13 TF/s, and three times the arithmetic item 8 needs for the same terms.
That is the falsification's "next thing", and it is item 25.

---

## 2. What changed

### 2.1 Item 22: the occupied pairs, packed

`j3c_oo[r] = C^T (pq|r) C` is symmetric in `(i, j)`; so is `dm_oo`, which
is it through the metric; so is `j3c_oo1` (the derivative is on the
auxiliary function, `(pq|∂r)` is symmetric in `p, q`), and so is
`j3c_oo1p = j3c_oo1 - j2c_10 dm_oo`.  Stage B and the head of the `h_aux`
block form five products from them:

* stage B: `dm_oo = V W^-1 V^T j3c_oo` — `(i, j)` a free index;
* `j3c_oo1p = j3c_oo1 - j2c_10 dm_oo` — free;
* `h_aux = -k/2 j3c_oo1p j3c_oo1p^T`, `dm_aux = -k/2 dm_oo dm_oo^T`,
  `dm_aux1 = -k/2 j3c_oo1p dm_oo^T` — summed over `(i, j)`.

On the `nocc(nocc+1)/2` pairs `i <= j`, with the off-diagonal entries
weighted `√2`, a dot product of two packed rows is the dot product of the
two full ones, and a linear map applied to packed rows is the same map on
the full ones.  So each product is the same product on half the columns
(`_Pairs`, `_metric_packed`, `_h_aux_head_packed`), a chunk of pairs at a
time, a few hundred MB, and `j3c_oo1p` and `dm_oo` are written back
unpacked — both triangles from the one — because D5 and the ip2 kernel
read them whole.  Taxol D2 **3.42 → 1.98 s**, stage B 0.66 → 0.48 s; the
Co(II) sextet's D2 4.89 → 2.57 s.  Where the occupied index is batched
(the restricted function on a short card) the blocks are not square and
the old path runs.

It is not taken on the long-range operator (`omega != 0`).  There one
triangle standing for both — a change in the last bit of each element of
an array that is symmetric only to rounding — moved a triplet's ip2 term
by **6.4e-10** relative against GPU4PySCF (`tests/test_dfuhessip.py`,
ethanol²⁺), where every `omega = 0` case agrees to 1e-13; W25 had seen the
same operator move by 1.1e-10 for reassociating stage A.

### 2.2 Item 24: the first pass from the second

Item 3 fused the first pass (`(1|00)`, the `ipaux` integrals filled
symmetrically and contracted with `C` twice) into the second pass's first
chunk.  The second pass scatters each batch of the same integrals as

    M[j,i] = dk + di,   M[i,j] = -di      (every stored pair i, j)

— so `M + M^T = A`, the `ipaux` block, exactly, on every off-diagonal pair
(on the diagonal the second write wins, and `-2 di = dk` is translational
invariance; the tests agree to 1e-13) — and then forms `T = M C` for its
own contraction.  The first pass's `-C^T A C` is `-(E + E^T)` with
`E = C^T T`: `2 nao nocc²` flops per auxiliary function and direction
instead of `2 nao² nocc` and a symmetric fill (`_d1_from_d3`).  Taxol:
D1 1.46 s → nothing, D3 `j3c_100` 1.39 → 1.60 s (the `E` product), net
**1.3 s**.  Unrestricted, only with both spins in one pass (the second
spin's pass would come after the `h_aux` block needs the result); on the
Co sextet, which runs a pass per spin, D1 stays.

### 2.3 Item 25: item 8 on the spin index

W24's item 8 holds the `(10|0)` block with its auxiliary index spherical
(`K = (10|0) C`, a pass over memory per shell) and carries the metric as
its Cholesky factor `J = L L^T` in every line of the call: D4 on
`Y = L^-1 K`, D5 on `Z = L^-T Y`, `V = C L^-T` with unit weights wherever
upstream's arithmetic takes `V W^-1 V^T`.  W25 did not carry it to the
unrestricted twin; `_chol_d4_d5_u` does, with the spin index on every
occupied quantity: D4 is the restricted Gram form once per spin
(`P_n o R_n + Q_n`), D5 sums both spins in each product, and the Coulomb
half of D4, which squares the trace summed over both spins, accumulates
the trace and squares it after the last spin.

Both spins' spherical `K` for the Co sextet is 2 × 29 GB, which with the
rest of the call does not fit 80 GB in one chunk — and item 8 needs the
whole spherical index in one.  So the second pass runs **once per spin**
where both do not fit (`_plan_v`'s new `nsp`), the fused first pass and
the `h_aux` block riding on the first.  That walks the integrals twice,
which the eigen form already did: its plan split the metric index into two
chunks (`vchunk = 2640` of 5256).  Co sextet: D3 `j3c_100` **9.47 →
2.07 s**, D5 **16.47 → 5.27 s** (with item 26), for 2.60 + 2.54 s of
triangular multiplies the eigenbasis did not need.  `FASTDFHESSIP_USPIN=1`
forces a pass per spin on molecules where both would fit, for the tests.

### 2.4 Item 26: D5 by atom groups

For every atom `A`, D5 adds

    f1 Σ_{u∈A,j} z[x,s,u,j] (t1s[y,s] C_A^T)[u,j]
  + f2 Σ_t j10s[y,s,t] Σ_{u∈A,j} P[s,u,j] z[x,t,u,j]

and the old loop formed `t1s C_A^T` and `P = dm_oo C_A^T` per atom —
batched products that read all of `t1s` (6.9 GB) or `dm_oo` for an
`nA`-column output — and reduced both through cuTENSOR's `sxy`
contractions.  `_d5_chol` forms the two products for a group of atoms at a
time (as many AOs as the second pass's `work` buffer holds, 370 on taxol)
as cuBLAS strided-batched GEMMs (13.3 TF/s, 47.8 and 18.8 ms for a group
of 26 atoms against 26 × 8.3 ms), reduces the first half with a kernel that
reads `z` and the product once per group (`d5_h1`, 12.4 ms for 26 atoms
against 26 × 2.0), and reduces each atom's `nsph × 3nsph` product against
`j10s` with a second kernel (`d5_h2`, 0.85 ms against 3.45).  The product
itself stays a cuTENSOR contraction on the strided slices of `z`, which
runs at 18.6 TF/s; a cuBLAS GEMM on the same strided operands ran at 13.6
and needed them copied contiguous to reach 18.3, and the first version of
this item did that (3.80 s where this one is 3.38).  Taxol D5 **4.76 →
3.38 s**; Co sextet (on top of item 25) 7.5 → 5.3 s.

### 2.5 Item 28: one scatter kernel

The second pass built each direction's `(nao, nao, dk)` block with a
batch-wide `dk += di` and `di *= -1` and then two indexed assignments;
`_scatter_d3` writes both triangles from the two integral batches as they
come.  Taxol 0.15 s, Co sextet 0.35 s.

### 2.6 Item 27, measured and off: the triangular multiplies as GEMMs

D8 runs at 15.1 TF/s against the 18.3–18.7 TF/s D5's product reaches, so the entry's "`dtrsm` ladder" was tried: `L^-1 x` by row blocks
of `b`, bottom up, a small `dtrmm` on each diagonal block and one GEMM
against the rows already done (top down for `L^-T`).  On taxol's shape
(`5598 × 60000`, `benchmarks/df_trmm_bench.py`):

| | `L^-1 x` | `L^-T x` |
|---|---|---|
| `dtrmm` | 124.6 ms, 15.09 TF/s | 123.8 ms, 15.19 |
| blocked, b = 512 | 127.8, 14.72 | 126.2, 14.89 |
| blocked, b = 1024 | 123.4, 15.23 | 122.5, 15.35 |
| blocked, b = 2048 | 124.1, 15.15 | 122.4, 15.37 |

and in the call 16.99 s against 17.02 (§3.1).  cuBLAS's `dtrmm` is already a
blocked GEMM; `FASTDFHESSIP_TRMM=blocked` keeps the variant runnable.

---

## 3. Measurements

### 3.1 The ip2 call at 113 atoms, and each item off

`df_hess_terms_probe.py --terms ip2 --variants ours --env ...`, one
process per molecule, the variants round robin (§5.2 of the work list),
one warm call each, then one timed.  The first two rows and the
`D5=atom` row are the final code (job `el_w37m`); the other ablations were
taken one job earlier (`el_w37e`), with the first version of item 26
(§2.4), whose default column read 17.02 s and 25.80 s:

| variant | taxol, RKS | Co(II) sextet, UKS |
|---|---|---|
| **all items (default)** | **16.59 s** | **25.81 s** |
| before (items 22, 24–26, 28 off) | 21.21 | 42.31 |
| item 26 off (`D5=atom`) | 17.90 (+1.31) | 27.97 (+2.16) |
| item 22 off (`PACK=0`) | 18.73 (+1.71) | 28.51 (+2.71) |
| item 24 off (`D1D3=0`) | 18.33 (+1.32) | 25.82 (—, a pass per spin) |
| item 28 off (`D3SCATTER=0`) | 17.16 (+0.15) | 26.15 (+0.35) |
| item 27 on (`TRMM=blocked`) | 16.99 | |
| item 25 off (`UCHOL=0`) | | 37.77 (+11.97) |

Stage tables (`FASTDFHESSIP_TIME=1`), same calls, s:

| stage | taxol before | after | Co before | after |
|---|---|---|---|---|
| D1 ipaux (fused) | 1.46 | — | 1.62 | 1.61 |
| D3 j3c_100 | 1.38 | 1.39 | 9.47 | 2.07 |
| D2 h_aux | 3.42 | 1.98 | 4.89 | 2.57 |
| D8 L^-1 K | 1.67 | 1.66 | — | 2.60 |
| D4 oo_atm | 2.08 | 2.08 | 3.24 | 3.23 |
| D8 L^-T Y | 1.76 | 1.76 | — | 2.54 |
| D5 per-atom | 4.76 | 3.38 | 16.54 | 5.27 |
| D3 integrals | 1.10 | 1.10 | 1.96 | 1.95 |
| total | 21.20 | 16.59 | 42.31 | 25.80 |

The Co sextet's D3 integrals are the same in both columns because both
walk the integrals twice: the eigen form in two metric chunks, item 25 in
two spin passes.

### 3.2 Other sizes and classes

`--variants stock,ours --env 'w34:...'`, round robin, min of two, full
card (with the first version of item 26, which differs from the final one
only in D5's product; the two 113-atom rows are the final code):

| system | atoms | GPU4PySCF | before | after | after/before |
|---|---|---|---|---|---|
| glycine | 10 | 210.9 ms | 155.0 | 143.6 | 1.08x |
| alanine dipeptide | 22 | 681.3 | 241.2 | 220.5 | 1.09x |
| penicillin V | 42 | 4432.8 | 1189.8 | 1074.2 | 1.11x |
| penicillin V, def2-TZVPP | 42 | 12394.4 | 3465.6 | 2956.1 | 1.17x |
| Ni(II) dithiolene⁻, doublet, UKS | 21 | 3632.7 | 1560.0 | 1281.0 | 1.22x |
| Pt(II) amine, triplet, UKS | 23 | 1702.1 | 776.1 | 683.0 | 1.14x |
| FeCl₄⁻, sextet, UKS | 5 | 540.4 | 352.5 | 343.0 | 1.03x |
| Cu(II) macrocycle²⁺, quartet, UKS | 60 | 10941.8 | 3156.6 | 2488.1 | 1.27x |
| taxol | 113 | | 21206.9 | 16594.3 | 1.28x |
| Co(II) sextet, UKS | 113 | | 42312.4 | 25806.2 | 1.64x |

Nothing is slower.  Every item's size dependence is by construction and
checked on these: item 22 is `(nocc+1)/2nocc` of a stage whose share
grows with `nocc`; item 24 removes a stage that is `nao/nocc` times its
replacement (5.2 on taxol, 4.5 at penicillin V/def2-TZVPP); item 26's
groups are sized from the buffer the second pass already holds and fall
back to one atom; item 25 changes the unrestricted call's arithmetic by
the factor item 8 did the restricted one's, `~1.5 naux/nsph`, at every
size, and chooses one or two spin passes from free memory.  The relative
differences against GPU4PySCF are the ones the tree had before, to within
the round-off of each system's metric (the Cu macrocycle's 1.1e-10 is
unchanged from W34's code, and is §5.1's effect in the eigenbasis).

### 3.3 Inside a whole Hessian

`benchmarks/df_hess_anatomy.py --patch all --rounds 1 --save ...` with
`FASTDFHESSIP_TIME=1`, before and after in separate jobs (the Co row's
"after" with the first version of item 26, whose D5 on the Co sextet is
the final one's to 0.3 %):

| system | before | after | ip2 before → after | trace | Hessians apart |
|---|---|---|---|---|---|
| taxol, RKS | 677.45 s | 675.42 s | 20.81 → **16.65 s** (1.25x) | 110.84373737(96, 97) | 6.9e-10 Eh/bohr² (8e-10 rel.) |
| Co(II) sextet, UKS | 1163.19 | 1147.30 | 42.25 → **25.97 s** (1.63x) | 105.61473986(95), …85(23) | 3.6e-9 (4.5e-9 rel.) |

Inside a Hessian the call is what it is back to back (§5.17 of the work
list does not bite here: nothing in these items allocates differently).
The Hessians are 92–93 % CPHF, whose `_get_jk` is W8's and 70–75 % of the
whole, so a 4–16 s gain in ip2 is 0.5–1.4 % of them.  The Co sextet's
Hessians are further apart than taxol's because its ip2 term moves from
the eigenbasis to the Cholesky factor of a metric of condition 3.3e10
(§5.1).

### 3.4 On a 40 GB card

`--ballast 40` (44.6 GB free), one warm call and one timed:

| system | before | after | |
|---|---|---|---|
| taxol | 34.11 s | 31.96 s | 1.07x |
| Co(II) sextet | 83.08 | 78.55 | 1.06x |

On 44.6 GB neither call can hold item 8's whole spherical index — taxol's
is 36 GB, one spin of the Co sextet's 29 GB beside `j3c_oo1` — so both
take the eigenbasis in chunks (taxol 3, the Co sextet 10, each chunk a walk
over the integrals), items 24–26 do not apply, and what moves is item 22.
The Co sextet's 78.6 s is dominated by those ten walks and the eigenbasis
D5 (40 s); a Cholesky form that does not need the whole spherical index
in one chunk is not in this round.

---

## 4. What is left

After this round the taxol call is, by stage: D5 3.38 s (the product at 18.6
TF/s), D8 3.42 s (15.1 TF/s), D4 2.08 s, D2 1.98 s (16 TF/s on
the packed products), D3 `j3c_100` 1.39 s (the GEMM at 15.5 TF/s), D3
integrals 1.10 s, stage A 0.87 s, the kernel 0.99 s, stage B 0.48 s, the
pseudo-density 0.43 s.  Every contraction stage is at 15–18 TF/s for what
it computes, so the restricted half of the entry's falsification holds
after the fixes: what is left is arithmetic.

Three stages compute more than they need to, all for the same reason:
D3's `M C`, stage A's `C^T J` and the pseudo-density's `C X C^T` are dense
`(nao, nao)` products on blocks that are filled only at the stored pairs.
At 113 atoms `2 nao_pair / nao²` is **50 %** (taxol 349,934 pairs at
`nao` 1185; the Co sextet 324,914 at 1116), so half their arithmetic
multiplies zeros — 1.7 s of the call — and at 284 atoms (22 % fill,
`docs/DF_WORK_LIST.md` §3.2) most of it.  W5's block-sparse gather is the
precedent.  That is **W39**.

Two smaller things were seen and not done: the `h_aux` block's nine
`j2c_ip2` GEMMs are one `dsyrk` (0.13 s), and `_SphAux`'s construction is
0.2 s of host work in stage B.

---

## 5. Correctness

### 5.1 The Cholesky twin and the metric's round-off

W24 established that two forms of this term that differ by 1e-10 do not
say which is off, and measured it by perturbing the metric by one ulp and
seeing how far each moves.  `benchmarks/df_hess_metric_sens.py` now takes
open shells (`--spin`), where its `eigen` column is W25's eigenbasis:

| system | metric condition | form | vs GPU4PySCF | moved by a 1-ulp metric (seeds) |
|---|---|---|---|---|
| Ni(II) dithiolene⁻, doublet | 1.6e9 | eigen | 1.7e-11 | 8.7e-12, 3.6e-12, 3.1e-12 |
| | | **chol (item 25)** | 3.0e-11 | 3.4e-11, 5.6e-11, 5.3e-11 |
| | | chol, a pass per spin | 3.0e-11 | 3.4e-11, 5.6e-11, 5.3e-11 |
| Pt(II) amine, triplet | 2.6e9 | eigen | 1.5e-12 | 1.6e-12, 7.6e-13, 8.5e-13 |
| | | chol | 2.0e-12 | 1.6e-12, 1.6e-12, 1.4e-12 |
| FeCl₄⁻, sextet | 4.6e8 | eigen | 1.2e-11 | 6.2e-12, 7.8e-12, 3.7e-12 |
| | | chol | 9.5e-12 | 1.1e-11, 3.2e-12, 9.2e-12 |
| Co(II) sextet (113 atoms) | 3.3e10 | chol vs eigen | 1.2e-10 | chol 5.6e-11, 4.2e-11; eigen 9.8e-12, 1.2e-11 |

(GPU4PySCF's own unrestricted function does not run at 113 atoms, so the
last row is against the eigenbasis.)  Each Cholesky row differs from its
reference by what a one-ulp change of the metric moves it — the restricted
result of W24, on the spin index — and the Co sextet's 1.2e-10 is two such
moves of the worst-conditioned metric in this repository.

### 5.2 The long-range operator

Item 22 is off for `omega != 0` (§2.1): on the ethanol dication triplet's
long-range term it read 6.4e-10 against GPU4PySCF, with `PACK=0` 7.2e-14.

### 5.3 The tests

`tests/test_dfhessip.py` and `tests/test_dfuhessip.py` pass ("all ok")
with new columns for every switch — `PACK=0`, `PACK_CHUNK=7` (many chunks,
with and without the carve), `D1D3=0`, `D5=atom`, `D5_GROUP=1` (a group
per atom), `D3SCATTER=0`, `TRMM=blocked TRMM_BLK=16`, and in the twin
`UCHOL=0`, `METRIC=eigen`, `USPIN=1` with and without the fusion and with
the other D4.  The memory-cap ladder of the twin now reaches item 25 with
both spins in one pass and with a pass per spin.  The twin's cuBLAS-route
section runs with items 22 and 25 off, since neither calls two of the four
refused contractions.

| | worst (relative to the largest element) |
|---|---|
| restricted ip2, ethanol and water, B3LYP / HF / `k = 0` / `omega = 0.3`, every switch | 5.5e-13 |
| unrestricted ip2, ethanol⁺ doublet and ethanol²⁺ triplet, every switch | 4.2e-13 |
| whole Hessians patched against unpatched | 9.0e-12 (RKS), 5.4e-12 Eh/bohr² (UKS) |

---

## 6. Tools

```bash
FASTDFHESSIP_TIME=1 FASTDFHESSIP_NVTX=1 nsys profile -t cuda,nvtx \
    -c cudaProfilerApi -o $SCRATCH/p python benchmarks/df_hess_terms_probe.py \
    --xyz ... --patch all --terms ip2 --variants ours --reps 1 --nvtx
nsys export --type sqlite -o $SCRATCH/p.sqlite $SCRATCH/p.nsys-rep
python benchmarks/df_ip2_stage_profile.py $SCRATCH/p.sqlite --range ip2:ours:0
                                                     # §1, every stage's kernels
python benchmarks/df_d5_bench.py [--old]             # §1, §2.4: D5's pieces
python benchmarks/df_trmm_bench.py                   # §2.6
FASTDFHESSIP_TIME=1 python benchmarks/df_hess_terms_probe.py --xyz ... \
    --patch all --terms ip2 --variants ours \
    --env 'w34:FASTDFHESSIP_PACK=0;FASTDFHESSIP_D1D3=0;FASTDFHESSIP_D5=atom;FASTDFHESSIP_D3SCATTER=0;FASTDFHESSIP_UCHOL=0'
                                                     # §3.1-3.2 (--ballast 40: §3.4)
python benchmarks/df_hess_metric_sens.py --xyz ... --spin 1 [--no-upstream \
    --variants eigen,chol --seeds 2]                 # §5.1
FASTDFHESSIP_TIME=1 python benchmarks/df_hess_anatomy.py --patch all ... # §3.3
python tests/test_dfhessip.py; python tests/test_dfuhessip.py
```
