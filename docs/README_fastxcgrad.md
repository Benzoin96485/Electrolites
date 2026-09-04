# `fastxcgrad` — the XC part of an RKS nuclear gradient

Replaces `gpu4pyscf.grad.rks.get_exc` and `get_nlc_exc`.  LDA, GGA **and
meta-GGA**; the grid-response path, unrestricted densities and multi-GPU fall
through to GPU4PySCF unchanged.

```python
import sys; sys.path.insert(0, '.../side_test/HF/src')
import fastxcgrad                       # patches on import
```

This round is the XC grid path of the gradient — item **B2** of
`instructions/06.gpu4pyscf_kernel_targets.md`, whose §5 ranks it first by
measured Amdahl share, together with the infrastructure item **G1** that §5
pairs it with.  Three things changed, and one thing G1 asks for turned out not
to be worth doing.

`get_exc` at a fixed density, one A100-SXM4-40GB:

| | B3LYP | lrc-wPBEh | ωB97X | ωB97M-V | M06-2X |
|---|---|---|---|---|---|
| `PfPMT`, 284 atoms, 6-31G\* | 3.75x | 3.65x | 3.43x | **3.98x** | **4.46x** |
| `HcgC`, 545 atoms, 6-31G\* | 4.04x | 3.89x | 3.81x | **4.04x** | **4.32x** |
| 70-atom model, def2-TZVPD | 2.27x | 2.23x | 2.22x | **2.69x** | **2.88x** |

The two meta-GGA columns are the ones that did not exist before: ωB97M-V and
M06-2X fell through to GPU4PySCF entirely, and the def2-TZVPD ωB97M-V row is
the block `README_fastejk.md` recorded at 0.86x for exactly that reason.  The
previous round's density-first rewrite is the `1 only` column of the ablation
table below; against *it* this round is a further **1.98x** on B3LYP and
**2.59x** on M06-2X.

## What the gradient's XC block has to compute

Per grid block,

    exc1[i,n] = sum_j (nabla_n phi_i | v_xc | phi_j) D_ij

GPU4PySCF forms the three `nao_sub x nao_sub` matrices in the middle
(`_gga_grad_sum_`, and for meta-GGA `_tau_grad_dot_` as well) and contracts
them with the density afterwards.

### 1. The density contraction comes first

    sum_j (sum_g a_n[i,g] b[j,g]) D_ij  =  sum_g a_n[i,g] (D b)[i,g]

so those matrices are never formed.  One `nao_sub^2 x ngrids` GEMM and a
row-wise reduction over the grid replace three `nao_sub^2 x ngrids` GEMMs,
twice over — and the second of the two needs `D phi`, which the density
evaluation has already produced.

Per grid block that is **2 GEMMs where GPU4PySCF issues 10** for GGA, and
**4 where it issues 19** for meta-GGA.  The meta-GGA count is where the
formulation pays most: `_tau_grad_dot_` alone is nine `nao_sub^2 x ngrids`
GEMMs, and every one of them builds a matrix this route never forms.

The density itself goes through the density matrix rather than the occupied
orbitals, for the reason `README_fastxc.md` gives for the Fock build: with the
orbitals rho costs four GEMMs of `nocc*nao_sub*ngrids`, with the density matrix
one of `nao_sub^2*ngrids`, and AO screening keeps `nao_sub` far below `4*nocc`
on these clusters.  For meta-GGA that choice is what makes change 1 free: tau
already needs `D . nabla_m phi` for `m = x,y,z`, and those are exactly the
three products the tau part of the *gradient* contracts against.

The same identity removes the potential-side GEMM for meta-GGA outright:
`D . sum_m ao[m] wv[m] = sum_m wv[m] (D . ao[m])` because the weights are
diagonal in the grid point, so what is a GEMM for GGA is an elementwise scale
here — and `xcg_exc1` below folds even that in.

### 2. libxc is called once per group of grid blocks, not once per block

This is the largest single item, and it is not arithmetic.  GPU4PySCF calls
`eval_xc_eff` inside the block loop, so a PfPMT/6-31G\* gradient makes 830
calls on 4096 points each.  On that grid (`work/t_libxc.py`):

| functional | all 3 396 608 points, 1 call | the same points, 830 calls |
|---|---|---|
| B3LYP (GGA) | **3.2 ms** | **388.0 ms** |
| M06-2X (meta-GGA) | **4.5 ms** | **555.4 ms** |
| LDA (SVWN) | **1.1 ms** | **239.2 ms** |

Essentially all of it is per-call overhead.  As a share of the XC gradient it
is **48 %** for B3LYP and **51 %** for M06-2X once change 3 is in (42 % and
35 % before it, when the reductions still cost more) — bigger than the GEMMs,
bigger than `eval_ao`, in every configuration.  So grid blocks are accumulated
into a group,
one libxc call covers the group, and the potential contraction runs over the
same blocks while their AO values are still resident.

**The grouping keeps its own buffer, and that is not a detail.**  GPU4PySCF
already has a `_grouped_block_loop` (`dft/numint.py:2113`) — the one G1 asks to
extend — but it lets `eval_ao` allocate every block separately, and because
`nao_sub` differs from block to block (mean 324, max 635 on PfPMT/6-31G\*) the
memory pool cannot reuse them.  Measured on PfPMT/6-31G\*
(`work/t_grouploop.py`):

| grid walk, nothing else | deriv=1 | deriv=2 |
|---|---|---|
| `ni.block_loop` (one reused buffer) | 58.6 ms | **130.8 ms** |
| `NI._grouped_block_loop` (per-block allocation) | 60.7 ms | **1600.2 ms** |

1.60 s against 0.13 s, for the same 830 AO evaluations.  Using it as-is made
the XC gradient *slower* than the per-block version (1520 ms against 1004 ms
for B3LYP on PfPMT).  `_group_block_loop` here is `ni.block_loop` with `nslot`
fixed-size slots instead of one; `nslot` follows free memory and the caller's
scratch, never a property of the molecule.

### 3. One kernel for the whole per-block contribution — `fastxcgrad.cu`

What is left after changes 1 and 2 is a reduction over grid points that keeps
the AO index, and that is the shape cuTENSOR handles worst here.  Measured on
one block (`work/t_reduce.py`, nao_sub 180 and 320, 4096 points):

| call | nao_sub=180 | nao_sub=320 |
|---|---|---|
| `contract('nig,ig->ni', ao[1:4], t)` | 0.080 ms | 0.101 ms |
| `contract('ig,ig->i', ao[5], t)` | 0.070 ms | 0.072 ms |

Nearly independent of how much data it touches — it is cuTENSOR plan creation
(`create_contraction`, `estimateWorkspaceSize`, `create_plan` and a workspace
allocation, on **every** call) plus cupy dispatch, about 50–70 µs a call.  With
830 grid blocks and up to six such call sites per block that was most of the
accumulate step when this was measured: 23 % of a GGA XC gradient and 30 % of a
meta-GGA one.  It is 12 % and 7 % now.

`xcg_exc1` does the whole contribution in one launch:

    rT[n][i] = sum_g [ nabla_n phi_i * u[i][g] + sum_b H[n][b][i][g] * V[b][i][g] ]
    u[i][g]    = t[i][g] + wv[0][g] c[0][i][g]
    V[b][i][g] = wv[1+b][g] c[0][i][g]  ( + wv[4][g] c[1+b][i][g] for meta-GGA )

`H` is the AO Hessian and `c[m] = D . ao[m]`.  This is the density-contracted
form of `_gga_grad_sum_` plus, for meta-GGA, `_tau_grad_dot_`.  Fusing them
removes the `(3, nao_sub, ngrids)` intermediate that `_make_dR_dao_w` writes
and the reduction immediately reads back, removes the `_scale_ao` that forms
`aow`, and for meta-GGA removes the potential-side product as well: the AO
block is read once and nothing of grid size is written.

The tau term is also the reason a kernel is needed rather than more `contract`
calls.  Its multiplier `wv[4] * (D . nabla_m phi)` depends on the AO index as
well as the grid point, so GPU4PySCF's `_make_dR_dao_w` — which is a single
fused kernel, but takes grid-only weights — cannot be reused; and the packed
Hessian order `XX XY XZ YY YZ ZZ` has no three contiguous components forming a
row of `H`, so the Python version needs eight calls where the kernel needs
one.  `FASTXCGRAD_FUSE=0` keeps that eight-call version (through
`xcg_rowdot3` and `xcg_hessdot`, the same reductions as separate kernels) so
the step can be measured on its own.

### What G1 asks for that is *not* worth doing

G1's stated mechanism is P8: batch the per-block GEMMs, because "one block's
potential GEMM is 36 CUTLASS tiles on 108 SMs".  That premise does not survive
change 1.  Once the density is contracted first there is one GEMM per block
instead of ten, `nao_sub x nao_sub` by 4096, and it is flop-bound: on one group
of 68 blocks, `grouped_gemm` takes **6.12 ms** against **5.80 ms** for the same
68 `cupy.dot` calls.  The `FASTXCGRAD_MODE=groupedgemm` ablation keeps the
CUTLASS grouped path so the comparison can be re-run; it is 4–8 % *slower* than
plain cuBLAS on every case measured.

So of the two things G1 bundles — batch the grid blocks, batch the GEMMs — the
first is worth **1.73× on B3LYP and 1.84× on M06-2X** (the `1+3` -> `1+2+3`
step of the ablation table) and the second is worth nothing here.  What made
the difference was not the GEMM batching but what riding along with a resident
group of blocks allows: one libxc call instead of 830.

## Results

`bench/check_xcgrad2.py`, one A100-SXM4-40GB, **one run at a time per GPU**,
`get_exc` at a fixed density, minimum of three timings.  Changes 2 and 3 are
independent, so each column below turns one of them off: **1+3, per block** is
the fused kernel without the grouping, **1+2, no kernel** is the grouping with
cuTENSOR reductions, and **1+2+3** is the default.

| system | functional | type | GPU4PySCF | 1+3, per block | 1+2, no kernel | **1+2+3** | speedup | max diff |
|---|---|---|---|---|---|---|---|---|
| `PfPMT`, 284 atoms, 2268 AOs, 6-31G\*, 830 blocks | `b3lyp` | GGA | 1901.0 ms | 880.9 ms | 612.9 ms | 506.9 ms | **3.75x** | 3.65e-15 |
| `PfPMT`, 284 atoms, 2268 AOs, 6-31G\*, 830 blocks | `lrc-wpbeh` | GGA | 1846.0 ms | 804.7 ms | 603.9 ms | 506.3 ms | **3.65x** | 5.33e-15 |
| `PfPMT`, 284 atoms, 2268 AOs, 6-31G\*, 830 blocks | `wb97x` | GGA | 1752.2 ms | 730.8 ms | 601.4 ms | 510.4 ms | **3.43x** | 8.76e-15 |
| `PfPMT`, 284 atoms, 2268 AOs, 6-31G\*, 830 blocks | `wb97m-v` | MGGA | 2675.1 ms | 948.6 ms | 1093.3 ms | 672.6 ms | **3.98x** | 4.70e-15 |
| `PfPMT`, 284 atoms, 2268 AOs, 6-31G\*, 830 blocks | `m06-2x` | MGGA | 2971.0 ms | 1236.3 ms | 1111.4 ms | 666.2 ms | **4.46x** | 3.96e-15 |
| `HcgC`, 545 atoms, 4170 AOs, 6-31G\*, 1575 blocks | `b3lyp` | GGA | 4988.5 ms | 1874.7 ms | 1424.7 ms | 1233.7 ms | **4.04x** | 1.05e-14 |
| `HcgC`, 545 atoms, 4170 AOs, 6-31G\*, 1575 blocks | `lrc-wpbeh` | GGA | 4778.1 ms | 1726.6 ms | 1431.5 ms | 1227.9 ms | **3.89x** | 4.18e-15 |
| `HcgC`, 545 atoms, 4170 AOs, 6-31G\*, 1575 blocks | `wb97x` | GGA | 4684.1 ms | 1587.3 ms | 1428.7 ms | 1229.1 ms | **3.81x** | 8.66e-15 |
| `HcgC`, 545 atoms, 4170 AOs, 6-31G\*, 1575 blocks | `wb97m-v` | MGGA | 6949.8 ms | 2201.7 ms | 2643.2 ms | 1703.6 ms | **4.08x** | 8.21e-15 |
| `HcgC`, 545 atoms, 4170 AOs, 6-31G\*, 1575 blocks | `m06-2x` | MGGA | 7494.7 ms | 2743.3 ms | 2606.6 ms | 1734.0 ms | **4.32x** | 6.68e-15 |
| 70-atom model, 1746 AOs, def2-TZVPD, 209 blocks | `b3lyp` | GGA | 788.4 ms | 397.9 ms | 413.1 ms | 346.6 ms | **2.27x** | 1.61e-15 |
| 70-atom model, 1746 AOs, def2-TZVPD, 209 blocks | `lrc-wpbeh` | GGA | 764.7 ms | 382.8 ms | 416.4 ms | 342.9 ms | **2.23x** | 1.80e-15 |
| 70-atom model, 1746 AOs, def2-TZVPD, 209 blocks | `wb97x` | GGA | 747.6 ms | 359.7 ms | 399.5 ms | 337.0 ms | **2.22x** | 2.72e-15 |
| 70-atom model, 1746 AOs, def2-TZVPD, 209 blocks | `wb97m-v` | MGGA | 1605.5 ms | 651.3 ms | 968.1 ms | 596.6 ms | **2.69x** | 2.91e-15 |
| 70-atom model, 1746 AOs, def2-TZVPD, 209 blocks | `m06-2x` | MGGA | 1715.2 ms | 802.4 ms | 869.7 ms | 596.1 ms | **2.88x** | 2.29e-15 |

The five functionals land in a **3.4–4.5×** band at 6-31G\*, and the two
meta-GGAs are at the *top* of it rather than being handed back as they were
before this round.  That is not an accident of the grid: GPU4PySCF's meta-GGA
XC gradient issues 19 GEMMs per block against a GGA's 10 and calls libxc on a
five-component rho, so it has more of both things changes 1 and 2 remove.

The two changes are worth different amounts to the two functional types, which
the 2 x 2 makes visible (`logs/g2_xcg_pfpmt_631_full.log`):

| system | functional | type | GPU4PySCF | 1 only | 1+3 | 1+2 | **1+2+3** | speedup | max diff |
|---|---|---|---|---|---|---|---|---|---|
| `PfPMT`, 284 atoms, 2268 AOs, 6-31G\*, 830 blocks | `b3lyp` | GGA | 1906.9 ms | 1002.9 ms | 878.0 ms | 601.8 ms | 507.3 ms | **3.76x** | 5.01e-15 |
| `PfPMT`, 284 atoms, 2268 AOs, 6-31G\*, 830 blocks | `m06-2x` | MGGA | 2971.0 ms | 1730.5 ms | 1232.9 ms | 1118.2 ms | 669.1 ms | **4.44x** | 3.61e-15 |

Two things in those columns are worth stating plainly rather than hiding in a
ratio.  First, **the grouping on its own is not always a gain**: at def2-TZVPD
`nao_sub` is 771 and there are only 209 grid blocks, so the libxc overhead it
removes is ~12 % of the baseline while the resident group costs several
gigabytes of buffer, and *1+2 without the kernel* comes out slightly behind
*1+3 without the grouping*.  Second, the default (all three) is the fastest
configuration on **every** case measured, at both bases and all five
functionals, which is why it is the default rather than something chosen per
system.

## In a whole gradient

`bench/prof_grad.py` on `PfPMT`/6-31G\*, all nine modules against unpatched
GPU4PySCF 1.7.0, at the same fixed density.

| component | B3LYP, GPU4PySCF | B3LYP, nine modules | x | M06-2X, GPU4PySCF | M06-2X, nine modules | x |
|---|---|---|---|---|---|---|
| 2e derivative (`RYS_per_atom_jk_ip1` / `fastejk`) | 16.80 s | 10.31 s | 1.63 | 17.15 s | 10.42 s | 1.65 |
| **XC (`get_exc` / `fastxcgrad`)** | **1.92 s** | **0.50 s** | **3.84** | **2.97 s** | **0.67 s** | **4.43** |
| one-electron (`get_hcore` / `fastgradh`) | 1.79 s | 0.24 s | 7.5 | 1.78 s | 0.22 s | 8.1 |
| `int3c2e.get_dh1e`, overlap, driver | 0.54 s | 0.50 s | — | 0.52 s | 0.52 s | — |
| **whole gradient** | **21.05 s** | **11.55 s** | **1.82** | **22.42 s** | **11.84 s** | **1.89** |

Both codes see the same density (`prof_grad.py`'s one Fock build from the minao
guess, cached), and the two gradients agree to the eighth decimal on every
component printed.  These are the numbers from the **second** gradient in the
process: the first pays one-time costs on both sides (NVRTC module load for
`fastxcgrad.cu`, the first `cudaMalloc` of the AO group buffer, cuBLAS and
cuTENSOR handle setup), which on a single gradient put the XC block at 1.17 s
and 1.21 s instead of 0.50 s and 0.67 s.  A geometry optimization or an MD
trajectory pays them once; a one-shot gradient pays them every time, and
`logs/g2_pg_pfpmt_*_{ref,new}.log` is that case.

**M06-2X is the row to look at.** Before this round its XC gradient block ran
GPU4PySCF's own code, so the whole 2.97 s was untouched; it is now 0.67 s, and
an M06-2X gradient on this cluster is 1.89x rather than 1.75x.


And on the 70-atom def2-TZVPD model, ωB97M-V — the case where the whole XC and
NLC grid path used to fall through:

| component | GPU4PySCF | nine modules | x |
|---|---|---|---|
| 2e derivative, two builds (full + long range) | 134.40 s | 55.09 s | 2.44 |
| **XC (meta-GGA)** | **1.61 s** | **0.57 s** | **2.82** |
| **VV10 non-local correlation** | **4.80 s** | **1.83 s** | **2.62** |
| one-electron, `get_dh1e`, overlap, driver | 0.78 s | 0.55 s | — |
| **whole gradient** | **141.59 s** | **58.04 s** | **2.44** |

### A note on where B2 actually sits

The XC block is a **tail** of these gradients — 9-13 % at 6-31G\* and **1.1 %**
at def2-TZVPD — because the two-electron derivative is 75-95 % of them.  This
is where the catalogue's own ranking has to be read carefully: §5 puts B2 first
on the strength of a 42-atom def2-SVP profile where the XC gradient was 74.9 %
of the gradient, and §7's Q9 already warns that the AO-count difference between
the two measurement regimes "is enough to invert the priority order".  It does
invert here: on 284-545 atom clusters at 6-31G\*/def2-TZVPD, B1 (`ejk_ip1`,
which `fastejk` did in the previous round) is the whole gradient and B2 is a
few per cent of it.

Two things still make this round worth doing.  M06-2X and ωB97M-V had **no
coverage at all** in this block before it, so their gradients carried
GPU4PySCF's own XC cost in full; and the two changes that turned out to matter
— one libxc call per group of blocks, and one kernel for the per-block
contraction — are not specific to the gradient.  `nr_rks_fxc` (item B4d) and
`hessian.rks._get_vxc_deriv1/2` (item B5) call `eval_xc_eff` inside the same
`block_loop`, and `fastxc` still hands meta-GGA back in the SCF (item A4).
Those are where the same 388-ms-per-830-calls measurement should be pointed
next.

## Accuracy

Three things differ from GPU4PySCF's arithmetic — rho through the density
matrix instead of the occupied orbitals, one libxc call per group instead of
one per block, and a different summation order in the grid reduction — so the
agreement below is not a tautology.

| test | what it compares | worst difference |
|---|---|---|
| `bench/check_xcgrad2.py` | `get_exc` against GPU4PySCF's own, same fixed density, 5 functionals x 3 systems x 3 ablations | **1.05e-14** on a per-atom gradient component |
| `bench/split_nlcgrad.py` | `get_nlc_exc`'s first grid pass against `eval_rho2` over the same blocks | **8.7e-16 relative** (1.2e-10 absolute against a max \|rho\| of 7.4e4) -- fp64 round-off between the density-matrix and orbital routes |
| `bench/test_grad.py` | the whole gradient, patched against unpatched, same converged density, **23 cases** (six bases to cc-pVQZ, twelve functionals including TPSS, M06-2X and ωB97M-V, third row, a cation, an open-shell radical) | **3.31e-11** on a gradient component -- and that case (SiCl4/def2-TZVP/B3LYP) is `fastejk`'s long-range/short-range screening difference, not this module: the eight meta-GGA cases are at 1.6e-13..7.9e-12 |
| `bench/test_full.py` + `bench/diff_full.py` | SCF **and** gradient, whole stack on or off in **separate processes**, 15 cases | energies **9.09e-13 Eh**, gradient components **3.31e-11** (the same SiCl4 case); the four meta-GGA cases are at 1.3e-12, 4.9e-13, 3.8e-12 and 8.0e-12, and all 15 converge in both codes |

## What is left

`bench/split_xcgrad.py` on PfPMT/6-31G\*, by incremental accumulation, for the
per-block path (so the libxc row is exactly what change 2 removes):

| block | B3LYP | | M06-2X | |
|---|---|---|---|---|
| `eval_ao` + `block_loop` | 130.7 ms | 14.8 % | 131.0 ms | 10.6 % |
| the density/potential GEMMs | 177.2 ms | 20.1 % | 271.8 ms | 22.0 % |
| the rho contractions | 45.8 ms | 5.2 % | 114.7 ms | 9.3 % |
| **libxc, once per block** | **422.6 ms** | **47.9 %** | **627.9 ms** | **50.8 %** |
| `xcg_exc1` | 106.4 ms | 12.1 % | 91.4 ms | 7.4 % |
| total | 882.6 ms | | 1236.8 ms | |

Change 2 turns the libxc row into ~5 ms, which is the 882.6 -> 507.3 and
1236.8 -> 669.1 of the tables above (the residual 40-55 ms is the grouping's
own copies and its buffer).  What is left of the default, then, is:

| block | share of B3LYP's 507 ms | why it is still there |
|---|---|---|
| the density/potential GEMMs | **35 %** | 1.43 TFLOP in 177 ms = **8.1 TFLOPS**, i.e. 84 % of the A100's fp64 rate without tensor cores but only 41 % of its DMMA rate. One experiment's worth: whether cuBLAS takes the DMMA path for a `nao_sub x nao_sub` by 4096 dgemm at all, and what alignment it wants if not. Nothing here is a Python-side cost any more |
| `eval_ao` at deriv=2 | **26 %** | catalogue item **A5**, not B2. All ten components are needed for a GGA gradient (the AO Hessian goes into the `phi_i . nabla_n phi_j` term), so there is nothing to cut from the *component count*; the kernel itself is the target |
| `xcg_exc1` | **21 %** | reads 12 rows of `nao_sub x ngrids` per block, 106 GB over the call, in 106 ms = **1.0 TB/s**, about two thirds of the card's bandwidth. `FASTXCGRAD_NT` sweeps the block size; past that it needs a different decomposition than one CUDA block per AO |
| the rho contractions | 9 % | `GDFTcontract_rho` is GPU4PySCF's own lean kernel |
| the grouping's copies, libxc | 9 % | — |

For M06-2X the GEMMs are 41 % and `xcg_exc1` 14 %, because meta-GGA needs four
density products where GGA needs one.  Both functional types now have the same
two things in front of them, and neither is what B2 or G1 named.

The NLC gradient is a separate matter.  Neither of its two grid passes calls
libxc, so change 2 does nothing there; what the two passes gain is change 1 and
the first-derivative AOs in pass one.  On PfPMT/6-31G\*/wB97M-V
(`bench/split_nlcgrad.py`):

| | GPU4PySCF | this module |
|---|---|---|
| pass 1, rho over the whole grid | 908.1 ms | **194.4 ms**, 4.67x |
| pass 2, the potential contraction | (not separable in GPU4PySCF) | 426.6 ms |
| both passes together | 1639.4 ms | 621.0 ms |
| the VV10 double sum between them | 58 954 ms | `fastnlc`'s business, 3.2x |
| whole `get_nlc_exc` | 60 594 ms | — |

so this is a 1.0 s saving on a 60 s call, and it is listed for completeness
rather than as a result.  Grouping pass 2 is in fact 20 ms *slower* than not
grouping it, exactly as the no-libxc argument predicts.

## Is any of this size-dependent?

| ingredient | depends on | handling |
|---|---|---|
| group size `nslot` | free GPU memory and the caller's per-block scratch | `MEM_FRACTION` (0.3) of `get_avail_mem()` divided by the bytes one block needs; GPU4PySCF's own rule is 0.2 for the AO values alone. Not a tuned constant, and nothing about the molecule enters. |
| kernel block size | the hardware | 256 threads over a grid block of `MIN_BLK_SIZE = 4096` points — GPU4PySCF's constant, the same for every molecule, so the per-thread loop is 16 iterations regardless of system size. The kernel's *grid* is `nao_sub`, so occupancy grows with the system and never shrinks. |
| grid, block size, AO screening | GPU4PySCF's own | untouched. |
| functional type | what the caller asked for | LDA, GGA and meta-GGA are handled; NLC has its own entry point; grid response, unrestricted and multi-GPU fall through. |

The *speedup* does depend on the regime, and the three sizes show which way:
at 6-31G\* there are many small blocks and the per-call overheads changes 2 and
3 remove are a large share, while at def2-TZVPD `nao_sub` is 771 and there are
only 209 blocks, so the GEMMs dominate and the same code gains less.  That is
the honest shape of the result and it is why the def2-TZVPD row is reported
next to the 6-31G\* ones rather than instead of them.

## Files

| file | what |
|---|---|
| `fastxcgrad.py` | the replacement `get_exc` and `get_nlc_exc`; `FASTXCGRAD_MODE=grouped\|perblock\|groupedgemm`, `FASTXCGRAD_KERNEL=0`, `FASTXCGRAD_FUSE=0`, `FASTXCGRAD_MGGA=0` and `FASTXCGRAD_MEM` select the ablations |
| `fastxcgrad.cu` | `xcg_exc1` (the fused per-block contribution) and `xcg_rowdot3` / `xcg_hessdot` (the same arithmetic as separate reductions, for `FASTXCGRAD_FUSE=0`) |
| `bench/check_xcgrad2.py` | `get_exc` across functionals and ablations, against GPU4PySCF at a fixed density |
| `bench/split_xcgrad.py` | where the time goes, by incremental accumulation |
| `bench/split_nlcgrad.py` | the two grid passes of the NLC gradient, without the VV10 double sum |
| `bench/test_grad.py` | the whole-gradient correctness suite, extended with M06-2X and lrc-wPBEh |
| `work/t_libxc.py`, `work/t_grouploop.py`, `work/t_reduce.py` | the three micro-benchmarks quoted above: libxc batched against per block, the two grid walks, the reduction shapes |
| `work/t_naosub.py` | the per-block `nao_sub` distribution (mean/max), which is why the per-block allocation in GPU4PySCF's `_grouped_block_loop` cannot be reused |
| `work/t_xcg_small.py` | the fast LDA/GGA/meta-GGA/NLC agreement check on a 7-atom molecule, for iterating |
| `work/serial_g2.sh`, `work/serial_g2b.sh`, `work/serial_g2c.sh` | the benchmark runs that produced `logs/g2_*`, one at a time on one GPU |

The previous round is reproducible from this file:
`FASTXCGRAD_MODE=perblock FASTXCGRAD_KERNEL=0 FASTXCGRAD_MGGA=0` is exactly
what `README_fastejk.md` describes.
