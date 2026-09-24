# The density-fitted Hessian's three uncovered terms: a binding, a launch constant, and an array nobody needed

`docs/DF_WORK_LIST.md` **W16**, **W17** and **W18** — the three functions W8's
anatomy of a density-fitted Hessian (`docs/README_fastdfhess.md` §5) found
carrying the time once `patch_all()` had taken the XC derivatives out, and
that nothing in this repository was on:

| item | function | share of a patched 42-atom DF-B3LYP/def2-SVP Hessian, before |
|---|---|---|
| **W16** | `hessian/rks.py:nr_rks_fxc_mo` — the CPHF's grid half | 25 % |
| **W17** | `df/hessian/rhf.py:_jk_energy_per_atom` — the `ip2` term of `partial_hess_elec` | 21 % |
| **W18** | `df/hessian/rhf.py:_get_veff` — the `h1` term of `make_h1` | 10 % |

W16 is a binding in `electrolites/fastxcfxc.py`; W17 and W18 are one new
module, `electrolites/fastdfhessip.py`, as the work list said they would be.
All three are GPU4PySCF's arithmetic in a different arrangement and agree with
it element-wise to round-off (§6).

---

## 0. Summary

* **W16 is a binding job, as the entry guessed, and one that also fixes a
  wrong answer.**  The perturbed density a CPHF cycle hands
  `nr_rks_fxc_mo` is a density matrix as well as a pair of orbital factors
  (`hessian/rhf.py:gen_vind` builds both), so the function is
  `fastxcfxc`'s blocked `nr_rks_fxc` on the plain matrix followed by
  upstream's own `_ao2mo` — plus, where a grid block's AO count is more than
  four times the occupied count, a second route to the same density from
  the orbital factors.  **2.66x** on the function at 42 atoms, **2.23x** at
  113, **4.00x** on a Pt(II) complex and **3.42x** on a Ni(II) one, **1.48x**
  in def2-TZVPP; after it the call is two GEMMs at their rate (§2).  On
  the way the round found that upstream's own function is **wrong for
  meta-GGAs**: its `fxc` contraction is an in-place cuTENSOR call,
  `contract('axg,xyg->ayg', rho1, fxc_w, out=rho1)`, which is exact at four
  density components and corrupts the output at five (§2.2).
* **W17 is 1.46x at 42 atoms and 1.89x at 113, and its largest single win
  is an array that did not need to exist.**  Three things carry it: every
  Rys kernel the function launches is aggregated at `nsp_per_block*4`, and
  `*1` is `fastdfgrad`'s finding again, on four more kernels (1.17x of the
  function by itself at 42 atoms); the first pass over the auxiliary
  derivative integrals is folded into the second (0.38 s at 42 atoms); and
  the per-atom occupied block `X[A,x,r,i,j]` — `3 natm naux nocc²` doubles,
  **18.7 GB at 42 atoms, 775 GB at 113** — is replaced by two `nao²` Gram
  matrices from which the same contraction follows exactly.  That stage goes
  **590 → 130 ms** at 42 atoms and **17.6 → 4.2 s** at 113 (§3).
* **W18 is 1.29x at 42 atoms and 1.57x at 113.**  The launch constant
  again, and the two per-atom loops inside its block loop — `2 natm` launches
  per block per Cartesian direction for the Coulomb half and a
  recomputed-per-atom intermediate for the exchange half — become three
  launches and one hoisted contraction (§4).
* **What a whole Hessian gets** (§5): a patched 42-atom DF-B3LYP/def2-SVP
  Hessian goes **21.25 s → 15.53 s (1.37x)**, the same molecule in DF-PBE
  **12.32 s → 8.37 s (1.47x)** and in def2-TZVPP **58.37 s → 46.62 s
  (1.25x)**; the Pt(II) and Ni(II) complexes are 1.29x and 1.23x, and a
  113-atom one **789 s → 701 s (1.13x)**, where W8's `_get_jk` is now 72 %
  of the run.
* **Three things the round measured and did not ship**, all still runnable:
  the explicit metric inverse in W18 (1e-8 relative error on the long-range
  operator, where the metric is ill-conditioned); the "X + Xᵀ as two
  accumulating contractions" version of W17's per-atom block, which the Gram
  form beats at every size; and a `gout_width` sweep for the `ip2` kernel,
  which is ±3 % and not worth a constant.

Everything here is one A100-SXM4-80GB of a whole node (an interactive
allocation, `CUDA_VISIBLE_DEVICES` set to one device), GPU4PySCF 1.8.1,
def2-SVP with its own jkfit set unless a row says otherwise, level-3 grids,
`conv_tol_cpscf = 1e-6`, and `patch_all()` on both sides of every comparison
— the baseline is `patch_all()` with `FASTDFHESSIP_OFF=1 FASTXCFXC_MO=0`,
which is exactly the configuration W8's anatomy measured.  The function-level
tables are `benchmarks/df_hess_terms_probe.py`, variants round robin in one
process against GPU4PySCF's own function object (`DF_WORK_LIST.md` §5.2);
the rows marked † were taken while other GPUs of the same node ran other
jobs of this round.

---

## 1. Where the time was

`benchmarks/df_hess_anatomy.py` (a lean version of `df_hess_probe.py
--whole`, which also runs W8's traced clone in front and costs minutes at 113
atoms), penicillin V, DF-B3LYP/def2-SVP, round 1 of 2 in one process,
second of two processes (`DF_WORK_LIST.md` §5.3).  Nested rows are
sub-intervals of their parent.

| stage | before | | after | | |
|---|---|---|---|---|---|
| **whole Hessian** | **21.25 s** | | **15.53 s** | | **1.37x** |
| `partial_hess_elec` | 5.68 s | 27 % | 4.33 s | 28 % | |
| ↳ `_jk_energy_per_atom` (W17) | 4.41 s | 21 % | 3.06 s | 20 % | 1.44x |
| `make_h1` | 3.91 s | 18 % | 3.18 s | 20 % | |
| ↳ `_get_veff` (W18) | 2.06 s | 10 % | 1.51 s | 10 % | 1.36x |
| `solve_mo1` (CPHF) | 11.61 s | 55 % | 7.97 s | 51 % | |
| ↳ `nr_rks_fxc_mo` (W16), 6 calls | 5.62 s | 26 % | 2.14 s | 14 % | 2.63x |
| ↳ `_get_jk` (W8), 6 calls | 5.13 s | 24 % | 5.08 s | 33 % | |

After the round the largest single stage of a density-fitted Hessian is
W8's own `_get_jk`, which `docs/README_fastdfhess.md` §2 measured at 78 % of
the card's fp64 peak.

---

## 2. W16: `nr_rks_fxc_mo` is `nr_rks_fxc` on a matrix it already has

`hessian/rks.py:_nr_rks_fxc_mo_task` is `nr_rks_fxc`'s `mo1` branch with the
MO transformation folded onto the end.  Per grid block it evaluates the
perturbed density from the orbitals — `eval_rho4`, one `contract` per
right-hand side — contracts `fxc`, and builds the potential with one
`ao · aowᵀ` per right-hand side: `2 nset` small cuBLAS calls and `2 nset`
elementwise kernels per block.  `nsys` on one call at 42 atoms (126
right-hand sides, 44 grid blocks):

| | upstream | this module |
|---|---|---|
| wall | 1396.5 ms | 413.0 ms |
| GPU busy | 984.5 ms (70 %) | 404.1 ms (98 %) |
| GEMM kernels | 674.3 ms in **11 133** launches | 282.8 ms in **86** |
| `contract_rho_gga` / `fxc_rho1` | 133.5 ms, 5 544 launches | 32.7 ms, 44 |
| `scale_ao` / `fxc_scale` | 115.9 ms, 5 544 | 37.4 ms, 44 |
| `add_sparse` | 30.3 ms | 21.9 ms |

The CPHF's density is a density matrix as well as a pair of factors:
`gen_vind` forms `dm1 = mo1 (2 C_o)ᵀ + h.c.` and tags it with `mo1` and
`occ_coeff`, and `dm1` is symmetric by construction.  So the replacement is
`fastxcfxc`'s blocked path on the untagged matrix at `hermi = 1` — one GEMM
per block for the density, one for the potential, three custom kernels in
between (`docs/README_fastxcfxc.md`) — and then upstream's `_ao2mo`.  No new
kernel.

**The falsification criterion does not fire, and it was the right one to
write.**  The entry said W16 is falsified if the `_mo` variant's time is in
evaluating the density from `mo1`.  It is not: the density GEMM is
`nset · nsub² · ng` on the matrix path against `4 nset · nocc · nsub · ng`
on upstream's orbital path, and at def2-SVP `nsub` averages 214 against
`4 nocc = 368`, so the matrix path does *less* arithmetic.

**But not in every block, and not in every basis.**  With `D = M Oᵀ + O Mᵀ`
(`M` the perturbed orbitals, `O = 2 C_o`),

    D · ao₀ = M (Oᵀ ao₀) + O (Mᵀ ao₀)

is `3 nocc · nsub` per grid point and right-hand side against the matrix
route's `nsub²`: three cuBLAS calls, two of them strided batches over the
right-hand sides with the shared operand at stride 0.  Two of the three have
an inner dimension of `nocc` and run below the matrix route's rate, so the
flop break-even (`nsub = 3 nocc`) is not the time one; swept on a quiet node,
with the route taken in every block where `nsub > r · nocc`:

| `r` | penicillin V, def2-SVP, 126 RHS | def2-TZVPP, 48 RHS | Pt complex, 69 RHS |
|---|---|---|---|
| never (matrix route only) | 409.7 ms | 640.2 ms | 87.8 ms |
| 3 | 415.0 | 552.5 | 92.8 |
| **4** (shipped) | **408.3** | **548.9** | **88.0** |
| 5 | 410.4 | 554.5 | 93.5 |
| 6 | 410.2 | 584.0 | 88.5 |

It is worth 1.17x on the function at def2-TZVPP, where `nsub` averages about
490 against `4 nocc = 368`, and nothing at def2-SVP, where only 28 blocks of
several hundred take it.  LDA and GGA only: a meta-GGA density needs
`D · ao_n` for every gradient component and the route would be four of each.
`FASTXCFXC_FACTORS=0` and `FASTXCFXC_FACTOR_RATIO` are the switches.

### 2.1 What it is worth, and where it is now

| system | RHS | upstream | this module | |
|---|---|---|---|---|
| penicillin V (42 atoms) | 126 | 1084.9 ms | **408.3 ms** | **2.66x** |
| taxol (113 atoms) † | 48 | 2626.4 ms | **1177.5 ms** | **2.23x** |
| penicillin V, def2-TZVPP | 48 | 814.5 ms | **548.9 ms** | **1.48x** |
| Pt(II) amine complex, 23 atoms, ECP | 69 | 351.6 ms | **88.0 ms** | **4.00x** |
| Ni(II) dithiolene dianion, 21 atoms † | 63 | 285.4 ms | **83.3 ms** | **3.42x** |

After the change the def2-SVP call is 70 % two GEMMs, each at 14–15 TF/s —
a `(26838 × 213) · (213 × 4091)` GEMM alone on the same card is 15.3 TF/s —
so the function is at the machine for the arithmetic it does.  Neither
taxol (`4 nocc` = 904) nor the Ni complex (420, against 327 AOs in all)
has a block that takes the factor route.

### 2.2 Upstream's function is wrong for meta-GGAs

`_nr_rks_fxc_mo_task` contracts `fxc` as

```python
wv = contract('axg,xyg->ayg', rho1, fxc_w, out=rho1)
```

— the output is the first operand.  cuTENSOR does not promise that an
in-place contraction is correct, and it is not, at five components:

| `ncomp` | right-hand sides | max error of the in-place call, 20 trials |
|---|---|---|
| 4 (GGA) | 3, 6, 9 | 4.4e-16 |
| 5 (meta-GGA) | 3 | **0.54** |
| 5 | 6 | **0.94** |
| 5 | 9 | **0.27** |

(random operands, 4096 grid points, relative to the largest element).  In the
function itself the damage is diluted — one or two right-hand sides of a call
come out wrong by 1e-7 to 2e-4 relative, a different one on different runs —
and `tests/test_dfhessip.py` shows it: against GPU4PySCF's *AO-basis*
`nr_rks_fxc` on the same density matrix, taken to the MO basis with
upstream's `_ao2mo`, this module agrees to **2–8e-16** for LDA, GGA and TPSS,
and upstream's `nr_rks_fxc_mo` agrees for LDA and GGA and is off by up to
**2.2e-7** for TPSS.  The same line is in the direct path's Hessian, which
calls the same function.

A whole meta-GGA Hessian does not show it cleanly, because it is not
reproducible anyway: an **unpatched** GPU4PySCF TPSS/def2-SVP DF Hessian of
glycine gives traces of 9.18679802, 9.18682691, 9.18680002 and 9.18679915 in
four runs, and the patched one moves by the same 3e-5 Eh/bohr² — so there is
a second source of run-to-run noise on the meta-GGA Hessian path, not in any
of this round's functions and not found.  It is `DF_WORK_LIST.md` **W23**.

---

## 3. W17: the `ip2` term

`_jk_energy_per_atom` is upstream's longest function in `df/hessian/rhf.py`,
and the module carries a copy of its single-device path with a synchronised
stage timer (`FASTDFHESSIP_TIME=1`).  With every change switched off —
`SPMULT=4 OOSUM=transpose FUSE=0` — the copy agrees with upstream's to
1.8e-13 and takes the same time, so the left column of the table below *is*
upstream's function:

| stage | what | 42 atoms, upstream | this module | 113 atoms, upstream | this module |
|---|---|---|---|---|---|
| A | `(pq|P)` → `j3c_oo`, plain integrals | 214.3 ms | 170.2 | 2240.8 | 1166.3 |
| B | the metric, `dm_oo` | 86.0 | 88.3 | 869.4 | 878.0 |
| C | pseudo-density | 22.8 | 32.9 | 482.2 | 485.9 |
| C | **the `ejk_int3c2e_ip2` kernel** | **1706.9** | **1336.5** | **12817.2** | **4107.1** |
| D1 | first `ipaux` pass → `(1|00)` | 446.8 | 67.0 (fused) | 4564.5 | 2395.4 (not fused) |
| D2 | `h_aux`, the auxiliary-auxiliary block | 183.1 | 181.5 | 4678.8 | 4686.3 |
| D3 | `ip1` + `ipaux` integrals | 726.6 | 591.0 | 6440.8 | 2241.9 |
| D3 | `(10|0)` → `j3c_100` | 156.0 | 156.1 | 4886.4 | 4902.8 |
| **D4** | **the per-atom occupied block** | **590.1** | **130.2** | **17640.1** | **4167.5** |
| D5 | the per-atom cross terms | 268.8 | 268.8 | 9258.3 | 8584.2 |
| | **whole call** | **4416 ms** | **3035** | **63.9 s** | **33.7 s** |

(penicillin V and taxol, def2-SVP; the 113-atom columns were taken with other
GPUs of the node busy, and the untimed calls are 63.9 and 34.2 s.)  The
launch constant is worth more the larger the system, as it was for the
gradient: the `ip2` kernel is 1.28x at 42 atoms and **3.12x** at 113, and the
derivative fills 1.23x and 2.87x.  At 113 atoms the fusion is declined —
its memory margin would have split the second pass into two chunks, and a
chunk costs more than the pass it saves (§3.3).

### 3.1 Item 1: the launch constant, on four kernels

Every Rys kernel both W17 and W18 launch — `ejk_int3c2e_ip2`,
`fill_int3c2e_ip1`, `fill_int3c2e_ipaux` and the plain `int3c2e` under
`int3c2e_evaluator` — is handed shell pairs aggregated at
`nsp_per_block[0]*4`, so each thread block loops four times.
`docs/README_fastdfgrad.md` §4.2 found `*1` faster for the gradient's ip1
kernel by a margin that grows with the system, and it is the same here.
Swept on upstream's own function by intercepting the aggregation
(`df_hess_terms_probe.py --variants stock,stock_sp1,stock_sp2,stock_sp8`),
42 atoms:

| `sp_mult` | 1 | 2 | **4** (upstream) | 8 |
|---|---|---|---|---|
| `_jk_energy_per_atom` | **3800 ms** | 4169 | 4447 | 4419 |
| `_get_veff` | **1863 ms** | 1976 | 2043 | 2040 |

The answers agree across the row to 5e-13 and 3e-17 respectively, which is
what makes it a launch configuration and not an approximation.  The module
takes `1` for all four kernels (`FASTDFHESSIP_SPMULT`); the plain evaluator
is reached through `_dfaux._spmult`, the same interception `fastdfcderi`
uses, which asserts that upstream still aggregates at `*4`.  The `ip2`
kernel's `gout_width` was swept too — 18 / 27 / 54 / 108 against upstream's
`None` is 0.97x–1.02x of the function and it stays upstream's.

### 3.2 Item 2: the per-atom occupied block as a Gram matrix

Upstream forms, for every atom `A` and every auxiliary function `r` of a
block,

    X[A,x,r,i,j] = sum_{u in A} J[x,r,u,i] C[u,j],     S = X + Xᵀ (over i,j)

and contracts `sum_{r,i,j} S[A,x,r,i,j] S[B,y,r,i,j] / w_r`.  `X` is
`3 natm nocc²` doubles per auxiliary function — 8.5 MB at 42 atoms and
**138 MB at 113**, so 18.7 GB and 775 GB over the auxiliary set — and
`transpose_sum` on it runs at ~80 GB/s because one side of a transpose is
uncoalesced: **462 ms** of a 4.4 s call, measured alone on the same shape.

Two steps remove it.  `Xᵀ` is the same contraction with the output indices
the other way round, so `S` is two accumulating contractions and the
transpose is gone (`OOSUM=gemm`, 184 ms at 42 atoms).  And then `S` itself
is not needed: with `S = X + Xᵀ`,

    sum_{ij} S_A S_B = 2 (T1[A,B] + T2[A,B])
    T1[A,B] = sum_{u in A, v in B} P[u,v] R[u,v],   R[u,v] = sum_{r,i} J[r,u,i] J[r,v,i] / w_r,   P = C Cᵀ
    T2[A,B] = sum_{u in A, v in B} Q[u,v],          Q[u,v] = sum_r M[r,u,v] M[r,v,u] / w_r,    M_r = J_r Cᵀ

— two `nao × nao` accumulators per Cartesian pair, and the atom sums done at
the very end as `Ind · (R∘P + Q) · Indᵀ` with `Ind` the atom-of-AO indicator.
The flop count goes from `9 natm² naux nocc²` to `12 nao² naux nocc`, which is
**fewer by a factor that grows linearly with the system** (0.66 at 42 atoms,
where the gain is all memory traffic, and 1.72 at 113), and no array has
`natm` in its size any more.  `1/w_r` is split as `1/√w_r` on each factor.
Three details decide whether it is fast rather than merely correct:

* `M` is one cuBLAS GEMM on the flattened `(x,r,u)` row index; `R` is the one
  cuTENSOR contraction left.
* `Q` needs `M_x ∘ M_yᵀ` summed over `r`.  cuTENSOR's
  `'xruv,yrvu->xyuv'` does it and was the slowest step; computing
  `N = Mᵀ` as a strided batch of the same GEMM and summing `M_x ∘ N_y` over
  `r` with a nine-accumulator kernel (`_HADAMARD_R`) reads both operands
  once, in the same layout.  That took the stage from 211 to 130 ms at 42
  atoms.
* The chunk over `r` is bounded by the buffer upstream sized for `X` and by
  1.2e8 elements, because `R`'s contraction hit the cuTENSOR plan wall
  (`DF_WORK_LIST.md` W21) at 42 atoms when the chunk was the whole auxiliary
  set.  `_accumulate_r` halves along `r` if a plan is refused anyway.

Measured on the stage, the Gram form is **590 → 130 ms** at 42 atoms and
**17.64 → 4.17 s** at 113 against upstream, and 16.05 → 4.02 s at 113
against `OOSUM=gemm`; it agrees with upstream to 1e-13 relative on every
test case.

### 3.3 Item 3: one walk over the auxiliary derivative integrals fewer

Upstream walks every auxiliary batch through `fill_int3c2e_ipaux` to build
`(1|00)`, runs the `h_aux` block, and then walks every batch again —
`fill_int3c2e_ipaux` and `fill_int3c2e_ip1` — once per chunk of the metric
dimension to build `(10|0)`.  Nothing between the two passes needs anything
the second produces, so the first pass is folded into the second pass's first
chunk, and the `h_aux` block runs after that chunk's integrals.  Doing that
holds the `h_aux` temporaries (about `24 naux²` words at their peak) while
the second pass's buffers are allocated, so the chunk is planned against the
memory that leaves, and the fusion is taken only if it does not cost the
second pass a chunk — a chunk is a whole walk over the integrals and the
fusion saves exactly one.  At 42 atoms it removes 380 ms.

### 3.4 What is left in it

At 113 atoms `D5` and the `j3c_100` build are 13.4 s of 33.3, and they are
the same `2·3·naux²·nao·nocc`-flop contraction three times over: into the
metric's eigenbasis in D3, back out of it in D5, and once more against
`j2c_10v` in D5.  In the auxiliary AO basis, with the Cholesky factor of the
metric in place of its eigenvectors, one of the three goes away — but that
needs the whole `(10|0)` block resident, 34 GB at 113 atoms, instead of one
metric chunk of it.  And the `ip2` kernel itself is 1.34 s of 3.04 at 42
atoms and 50 % of the call on the two transition-metal complexes, where it
is why W17 is only 1.15x–1.20x (§5).  Both are `DF_WORK_LIST.md` **W24**.

---

## 4. W18: the `h1` term

Same method, same copy-with-timers.  Penicillin V and taxol (the timers
synchronise inside the block loop, so the columns sum to more than the call;
the 113-atom columns were taken with other GPUs of the node busy):

| stage | what | 42 atoms, upstream | this module | 113 atoms, upstream | this module |
|---|---|---|---|---|---|
| A | `(pq|P)` → `j3c_00` | 219.8 ms | 176.7 | 2689.8 | 1814.6 |
| B | the metric → `dm_3c` | 60.0 | 59.7 | 2147.5 | 2165.9 |
| C | `j2c_10`, per auxiliary atom | 165.8 | 173.7 | 5516.8 | 5506.9 |
| D | `ip1` + `ipaux` integrals | 730.7 | 582.9 | 6346.3 | 2138.5 |
| D1 | `vhf1` | 79.1 | 107.3 | 1861.0 | 2174.2 |
| **D2** | **exchange, per atom** | **382.4** | **203.1** | **7991.5** | **4676.9** |
| **D3** | **Coulomb, per atom** | **422.4** | **34.3** | **4043.8** | **411.7** |
| E | `(00|1)` fill and per-auxiliary-atom terms | 191.0 | 218.5 | 2309.1 | 2300.5 |
| | whole call, untimed | 2043 ms | 1578 | 33.5 s | 21.5 s |

* **Item 1**, the launch constant, as in §3.1.
* **Item 4, the exchange half's per-atom intermediate hoisted.**  For each
  block, each Cartesian direction and each atom, upstream forms
  `sum_i C[u,i] dm_3c[r,i,q]` for the atom's AOs and contracts it with the
  atom's slice of the derivative integrals.  The intermediate depends on
  neither the direction nor the atom's slice beyond its rows, so it is
  formed once per block for every AO (`(nao, blk, nao)`, carved from the tail
  of the block loop's own buffer — the block is sized with it in the
  denominator) and each atom's contraction reads its rows.
  `FASTDFHESSIP_H1ATOM`.
* **Item 5, the Coulomb half without the per-atom loop.**  Upstream runs
  `einsum('pqr,pq->r', j3c[p0:p1], dm[p0:p1])` and a contraction for every
  atom — `2 natm` launches per block per direction, 1 618 reduction kernels
  in one call at 42 atoms.  `sum_{p in A, q} j3c[p,q,r] D[p,q]` for every
  atom is one row-wise contraction and one GEMM against the atom indicator,
  and the per-atom update is one contraction into the strided slice.
  `FASTDFHESSIP_H1J`.

D1 and E are *slower* at 42 atoms, by 28 and 27 ms, and D1 by 0.31 s at
113: item 4's extra `nao²` per block halves the auxiliary block, so there
are twice as many of the per-block steps that item 4 does not touch.  Net
1.29x at 42 atoms, 1.57x at 113 (21.3 s against 33.4 in one process), 1.49x
in def2-TZVPP.  What is left at 113 atoms is mostly GEMMs: C, D2 and B
carry `2·3·naux²·nocc·nao`, `2·3·nao³·naux` and `2·2·naux²·nocc·nao` flops,
which is 3.2, 3.2 and 2.1 s at 15 TF/s against 5.5, 4.7 and 2.2 s measured —
C's per-auxiliary-atom GEMMs and D2's per-atom ones are where the remaining
factor is, and it is below 2x.

**Item 6 was measured and is off.**  `B` applies the metric as
`V (W⁻¹ (Vᵀ j3c_00))` — two GEMMs of `naux² · nocc · nao` — and forming
`j2c⁻¹ = V W⁻¹ Vᵀ` once makes it one, which at 113 atoms would be 1 s.  It
costs **1.5e-10** relative on ethanol with HF exchange and **1.4e-8** on the
long-range operator, whose metric is the ill-conditioned one; the two-step
product is what upstream wrote and it stays.
`FASTDFHESSIP_H1METRIC=inverse` runs it.

The Coulomb-only `ip2` term — what a *pure* functional's
`partial_hess_elec` reaches, upstream's `_j_energy_per_atom` — is covered
too: item 1, and **item 7**, the per-atom derivative densities as two GEMMs
against the stored pair list instead of a dense scatter plus `2 natm`
contractions per block.  The one subtlety is the diagonal shell blocks, where
both orderings of an AO pair are stored and upstream's second scatter
overwrites the first, so the `dj` factor is masked there.  1.27x on the
function at 42 atoms (1422 → 1126 ms), and h1 with `k = 0` is 1.43x
(1848 → 1294 ms).

---

## 5. Whole Hessians

`benchmarks/df_hess_anatomy.py --rounds 2`, baseline and patched in
alternating processes on one GPU, the pair run twice; the table is the
second pair's second round (two such chains ran at once, on GPUs 0 and 1).
The traces agree to the run-to-run noise of the unpatched code, which is
the ip2 kernel's atomics.

| system | basis | xc | before | after | | traces, before / after |
|---|---|---|---|---|---|---|
| penicillin V, 42 atoms | def2-SVP | B3LYP | 21.25 s | **15.53 s** | **1.37x** | 41.7858224574 / …573 |
| penicillin V | def2-SVP | PBE | 12.32 s | **8.37 s** | **1.47x** | 40.7647128452 / …452 |
| penicillin V | def2-TZVPP | B3LYP | 58.37 s | **46.62 s** | **1.25x** | 39.7773170383 / …384 |
| Pt(II) amine complex, 23 atoms | def2-SVP + ECP | B3LYP | 8.58 s | **6.64 s** | **1.29x** | 20.8027830596 / …578 |
| Ni(II) dithiolene dianion, 21 atoms | def2-SVP | B3LYP | 10.65 s | **8.65 s** | **1.23x** | 21.9590103866 / …867 |
| taxol, 113 atoms † | def2-SVP | B3LYP | 789.2 s | **701.3 s** | **1.13x** | 110.8437373892 / …822 |

Against GPU4PySCF with nothing patched, the same two 42-atom Hessians are
34.70 s and 25.63 s (second of two processes, second round), so everything
in this repository now takes a DF-B3LYP one 2.23x and a DF-PBE one 3.06x.

The taxol row is one pair of one round each.  At that size the three terms
go 86.4 → 41.4 s (W16), 62.9 → 33.3 s (W17) and 33.7 → 21.1 s (W18) — 94 s
saved — and W8's `_get_jk`, six calls of 339 right-hand sides, is 504.8 s
of the 701: the round has moved the density-fitted Hessian to the point
where nearly all of it is the one function `docs/README_fastdfhess.md`
measured at 78 % of peak.

The function-level ratios for the two transition-metal complexes, which are
the chemically different cases:

| | `nr_rks_fxc_mo` | ip2 | h1 |
|---|---|---|---|
| Pt(II) amine complex, 23 atoms, def2 ECP | 4.00x | 1.20x † | 1.60x † |
| Ni(II) dithiolene dianion, 21 atoms † | 3.42x | 1.15x | 1.15x |

W16 does *better* on both than on the organic cases (few atoms, many grid
blocks with a small `nsub`, so upstream's per-set launches are an even
larger fraction), and W17 does worse: the `ip2` kernel is half the call once
the metal's f shells are in the auxiliary set, and nothing here changes what
that kernel does per shell pair.

---

## 6. Correctness

`tests/test_dfhessip.py`, element-wise against GPU4PySCF's own function
objects in one process on one converged SCF, relative to the largest element:

| | worst |
|---|---|
| ip2 and h1, ethanol, B3LYP factors / HF factors / `k = 0`, every switch alone | **2.3e-13** (ip2), **4.2e-16** (h1) |
| the long-range operator, `omega = 0.3`, water | 2.9e-15 / 8.4e-16 |
| the memory-short paths: second pass in 2–48 chunks, with and without the fusion; 2 and 3 occupied batches with `dm_oo` on the host | 3.9e-13 |
| `nr_rks_fxc_mo`, LDA / GGA / TPSS, 1, 3 and 6 right-hand sides, against the AO-basis path | 7.5e-16 |
| the same, against upstream's `nr_rks_fxc_mo`, LDA and GGA | 8.1e-16 |
| a whole ethanol DF-B3LYP Hessian, patched against unpatched | **5.5e-12 Eh/bohr²** |
| whole Hessians, §5's traces | agree to the unpatched code's own run-to-run spread |

and def2-TZVPP passes the same suite.  Most of the ip2 rows' 1e-13 is the
kernel's own: upstream's function run three times on ethanol differs from
its first run by 3.3–6.6e-14, because `ejk_int3c2e_ip2` accumulates with
atomics, and the reordered sums here add a factor of a few to that.  The memory-short paths
are reached by capping the memory the module sees (`FASTDFHESSIP_MEM`), the
chunk (`FASTDFHESSIP_VCHUNK`) and the occupied batch
(`FASTDFHESSIP_OCC_BATCH`) — three test-only switches, because upstream takes
those paths only when memory is short on a molecule too big for a test.

`benchmarks/coverage_probe.py --hessian --density-fit` on glycine shows
`fastxcfxc fxc_mo` 6/6, `fastdfhessip ip2` 1/1 and `h1` 1/1, and
`tests/test_fxc.py`, `tests/test_dfhess.py` and
`tests/test_patch_aliases.py` still pass.

---

## 7. Coverage and what falls through

* **Restricted only.**  The unrestricted twins are separate functions in
  `df/hessian/uhf.py` and are not covered; an open-shell DF Hessian's CPHF
  grid half already comes through `fastxcfxc`'s `nr_uks_fxc` `mo1` branch.
  `DF_WORK_LIST.md` **W25**.
* More than one device falls through to GPU4PySCF, as everywhere in this
  repository.
* `nr_rks_fxc_mo` hands back anything `fastxcfxc` would decline (HF, NLC,
  an untagged density, `FASTXCFXC_MODE=grouped`), and `FASTXCFXC_MO=0`
  hands back everything.

---

## 8. Tools

```bash
python tests/test_dfhessip.py [--basis def2-tzvpp] [--no-hessian]
python benchmarks/df_hess_terms_probe.py --xyz benchmarks/molecules/penicillinV.xyz \
    --basis def2-svp --xc b3lyp --patch all --terms fxc,ip2,h1 --variants stock,ours \
    [--env 'label:FASTDFHESSIP_OOSUM=transpose;FASTDFHESSIP_FUSE=0,...'] \
    [--variants stock,stock_sp1,stock_sp2]     # §3.1, upstream's own function
FASTDFHESSIP_TIME=1 python benchmarks/df_hess_terms_probe.py ...   # §3, §4 stage tables
nsys profile -t cuda,nvtx -c cudaProfilerApi -o $SCRATCH/p \
    python benchmarks/df_hess_terms_probe.py ... --nvtx            # §2's kernel table
python benchmarks/df_hess_anatomy.py --xyz ... --patch all --rounds 2   # §1, §5
FASTDFHESSIP_OFF=1 FASTXCFXC_MO=0 python benchmarks/df_hess_anatomy.py ...  # the baseline
python benchmarks/coverage_probe.py --xyz ... --hessian --density-fit --patch all
```

`fastdfhessip`'s switches are listed with their meanings at the top of
`electrolites/fastdfhessip.py`: `FASTDFHESSIP_OFF`, `_SPMULT`, `_OOSUM`
(`gram`, `gemm`, `transpose`), `_FUSE`, `_H1ATOM`, `_H1J`, `_H1METRIC`,
`_JPAIR`, `_IP2_GOUT`, `_TIME`, and the three test-only ones `_MEM`,
`_VCHUNK` and `_OCC_BATCH`.
