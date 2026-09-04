# fastejk / fastgrad / fastxcgrad / fastgradh — faster RKS nuclear gradients

Four drop-in replacements for GPU4PySCF 1.7.0's analytical gradient, measured
on one A100-SXM4-40GB.  The earlier rounds (`fastk`, `fastj`, `fastxc`,
`fastrsh`, `fastnlc`) only touch the SCF: a gradient calls
`RYS_per_atom_jk_ip1`, `grad.rks.get_exc` and `grad.rhf.get_hcore`, none of
which any of them patches, so before this round a gradient ran entirely on
GPU4PySCF's own code.

```python
import sys; sys.path.insert(0, '.../side_test/HF/src')
import fastk, fastj, fastxc, fastrsh, fastnlc          # the SCF
import fastejk, fastgrad, fastxcgrad, fastgradh        # the gradient
from gpu4pyscf import dft
mf = dft.RKS(mol, xc='b3lyp'); mf.kernel()
g = mf.nuc_grad_method().kernel()
```

| module | replaces | what it covers |
|---|---|---|
| `fastejk` | `grad.rhf._jk_energy_per_atom` | the per-atom derivative of `j*J - k*K`, every `(li lj\|lk ll)` class an spdf basis reaches, for the full-range and the long-range (erf) operator |
| `fastgrad` | `grad.rks.Gradients.energy_ee` | range-separated hybrids: the long-range operator in place of GPU4PySCF's short-range one |
| `fastxcgrad` | `grad.rks.get_exc`, `grad.rks.get_nlc_exc` | the XC (LDA/GGA) and VV10 grid contractions |
| `fastgradh` | `grad.rhf.GradientsBase.get_hcore` | the nuclear-attraction part of the one-electron derivative, on the GPU |

All four are spin-restricted closed-shell only; a UKS gradient reaches
`_jk_energy_per_atom` with two density matrices and `fastejk` hands it back.
Anything else outside their scope also falls through, per angular-momentum
group, so the gradient is the same either way.

## Where a gradient's time is

One gradient at a fixed density, GPU4PySCF 1.7.0, `direct_scf_tol=1e-13`,
default level-3 grids, one A100 (`bench/prof_grad.py`):

| block | `PfPMT`, B3LYP/6-31G\* | `HcgC`, B3LYP/6-31G\* | `PfPMT`, wB97M-V/def2-TZVPD |
|---|---|---|---|
| 2e derivative (`RYS_per_atom_jk_ip1`) | 16.79 s (75 %) | 58.80 s (78 %) | 2310.3 s (96 %) |
| XC | 3.18 s (14 %) | 5.37 s (7 %) | 19.2 s (0.8 %) |
| one-electron (`int1e_ipkin`+`int1e_ipnuc`, CPU) | 1.89 s (8 %) | 9.45 s (13 %) | 12.0 s (0.5 %) |
| VV10 non-local correlation | — | — | 71.0 s (2.9 %) |
| `int3c2e.get_dh1e`, overlap, driver | 0.57 s | 1.42 s | 3.6 s |
| **total** | **22.43 s** | **75.04 s** | **2416.1 s** |

so the two-electron derivative is the round, and everything else is a tail.

## Results

`conv_tol=1e-9`, `direct_scf_tol=1e-13`, default level-3 grids, one
A100-SXM4-40GB, **one run at a time per GPU** -- two benchmarks sharing a GPU
move a class by 2x, which is enough to invert a sweep.  The SCF rows are the
earlier five modules; the gradient rows are this round's four.

### B3LYP-D3(BJ)/6-31G\*, energy and gradient

| | GPU4PySCF 1.7.0 | with all nine modules | speedup |
|---|---|---|---|
| **`PfPMT_cluster`**, 284 atoms, 2268 AOs | | | |
| SCF (single-point energy) | 112.6 s | 56.2 s | 2.00x |
| gradient | 22.49 s | **13.04 s** | **1.72x** |
| energy + gradient | 135.1 s | **69.2 s** | **1.95x** |
| **`HcgC_cluster`**, 545 atoms, 4170 AOs | | | |
| SCF | 544.0 s | 269.4 s | 2.02x |
| gradient | 76.69 s | **41.48 s** | **1.85x** |
| energy + gradient | 620.7 s | **310.9 s** | **2.00x** |

`PfPMT` converges to `-7652.560086450` in both codes and its gradient agrees to
all eight digits printed per component.  `HcgC` converges to
`-13864.692237968` and `-13864.692243465`: that cluster's SCF is chaotic --
GPU4PySCF against unpatched GPU4PySCF is already 2.9e-6 apart at the first
cycle (`logs/wb97x_hcgc_scf_stability.md`) -- so its correctness evidence is
the fixed-density comparison below, not the converged number.

Per gradient component (`bench/prof_grad.py`, at the converged density):

| component | `PfPMT` | | | `HcgC` | | |
|---|---|---|---|---|---|---|
| | GPU4PySCF | this round | x | GPU4PySCF | this round | x |
| 2e derivative | 17.52 s | 10.70 s | 1.64 | 60.29 s | 36.33 s | 1.66 |
| XC | 2.25 s | 1.37 s | 1.64 | 5.31 s | 2.56 s | 2.07 |
| one-electron (`fastgradh`) | 2.16 s | 0.40 s | 5.4 | 9.69 s | 1.17 s | 8.3 |
| `int3c2e.get_dh1e` | 0.42 s | 0.42 s | 1.00 | 0.96 s | 1.00 s | 0.96 |
| overlap, driver | 0.15 s | 0.15 s | — | 0.44 s | 0.42 s | — |

The two systems differ by 1.8x in AO count and land within 1 % of each other on
the two-electron derivative (1.64x against 1.66x) and within 7 % on the whole
gradient, which is the empirical form of "nothing here is tuned to a problem
size".

### The two-electron gradient build alone, at a fixed density

`bench/check_ejk.py`, `j_factor = k_factor = 1`:

| | GPU4PySCF | `fastejk` | speedup | agreement |
|---|---|---|---|---|
| `PfPMT`/6-31G\*, 2268 AOs | 17.33 s | **10.50 s** | 1.65x | 2.2e-13 relative |
| `HcgC`/6-31G\*, 4170 AOs | 63.27 s | **38.50 s** | 1.64x | 3.9e-13 relative |
| 70-atom model/def2-TZVPD, 1746 AOs | 69.60 s | **32.36 s** | 2.15x | 6.7e-13 relative |

Those are the whole `_jk_energy_per_atom` call, screening setup included, at a
three-cycle B3LYP density; the converged-density rows in the table above
(1.64x and 1.66x) are the same build inside a real gradient.

Per angular-momentum class (`bench/perclass_ejk.py`, `PfPMT`/6-31G\*, the
build re-timed three times and the minimum taken): **18.0 s -> 10.6 s, 1.70x**,
with every one of the 25 classes faster -- 1.19x to 2.83x -- and each class
agreeing to 1e-15..3e-13 on the per-atom derivative.

### def2-TZVPD

An spdf basis reaches 65 angular-momentum classes and GPU4PySCF unrolls
**18** of them, so 47 land on its general `rys_ejk_ip1_kernel`.  That is why
the two-electron derivative is 96 % of a wB97M-V/def2-TZVPD gradient where it
is 75 % of a B3LYP/6-31G\* one, and it is where this round has the most to
take.

One `PfPMT` gradient at a fixed density (`bench/prof_grad.py --sys PfPMT
--basis def2-TZVPD --xc wb97m-v`):

| block | GPU4PySCF 1.7.0 | all nine modules | speedup |
|---|---|---|---|
| 2e derivative -- two builds, full-range + short-range against full-range + long-range | 2310.3 s | **1305.9 s** | 1.77x |
| VV10 non-local correlation | 71.0 s | **22.1 s** | 3.22x |
| XC (meta-GGA, which `fastxcgrad` declined at the time) | 19.2 s | 22.2 s | 0.86x |
| one-electron | 12.0 s | 3.2 s | 3.8x |
| `int3c2e.get_dh1e`, overlap, driver | 3.6 s | 3.8 s | — |
| **gradient** | **2416.1 s** | **1357.1 s** | **1.78x** |

Forty minutes to twenty-three.  The two gradients agree to 1.3e-8 on `|de|`
(1.1975062487 against 1.1975062613) and to the eighth decimal on every
component printed -- that is the long-range/short-range screening difference
under `fastgrad` below, accumulated over 284 atoms and 6609 AOs, not round-off.
Importing `fastejk` *without* `fastgrad` keeps GPU4PySCF's short-range route
(which `fastejk` declines, so it falls through bit-for-bit) and brings the
agreement back to 1e-13, at the cost of most of the range-separated speedup.

The XC row is a meta-GGA, which `fastxcgrad` handed back when this was
measured; the 0.86x is run-to-run noise on a 19-second block, not a regression
it introduced.  Meta-GGA is covered in the round `README_fastxcgrad.md`
describes, so this row is stale on the low side.

A converged wB97M-V/def2-TZVPD gradient on `HcgC` is ~2.7 h in GPU4PySCF, so
the second size at that basis is a 70-atom model cut from `PfPMT`
(`work/pfpmt70.xyz`, 1746 AOs against `PfPMT`'s 6609, a 3.8x span).  Per class
on it (`bench/perclass_ejk.py --basis def2-TZVPD`): **69.0 s -> 32.3 s,
2.13x**, all 65 classes faster.


## What `fastejk` changes

The integral arithmetic is the same Rys quadrature and the same root/weight
tables.  `gen_ejk.py` writes the kernels from the angular-momentum class
alone — nothing is lifted from GPU4PySCF's source, so the 47 classes an spdf
basis reaches that GPU4PySCF has no unrolled gradient kernel for are covered
the same way as the 18 it does unroll.  What changes around the arithmetic is
what `fastk` changed for the exchange matrix, plus two things specific to the
gradient.

1. **No double-precision division in the primitive loops.**  GPU4PySCF
   evaluates `cicj*ckcl/(aij*akl*sqrt(aij+akl))` once per *primitive quartet*
   and `rt/(aij+akl)`, `.5/aij`, `.5/akl` once per Rys root.  Here `1/aij` is
   block-uniform and cached with the rest of the bra data, `1/akl` is hoisted
   out of the bra loop, and one `rsqrt(aij+akl)` turns the rest into
   multiplies.
2. **Block-uniform bra data computed once.**  For a given block the bra shell
   pair is fixed, so `aij`, `1/aij`, `aj/aij`, `2ai`, `2aj`, the product centre
   and `cicj` depend only on the primitive-pair index; GPU4PySCF recomputes
   them per thread per primitive quartet.
3. **The Rys roots in registers** for the one-thread-per-quartet classes,
   where GPU4PySCF stages them through shared memory.
4. **The density products `dd` in registers.**  GPU4PySCF's *general* kernel —
   the one the other 47 classes land on — writes `nf` of them per shell quartet
   to a global-memory pool (`dd_pool`, up to 101250 doubles per SM) and reads
   them back once per Rys root per primitive quartet.
5. **The gout tile is exactly `nf` wide.**  The general kernel pads every shell
   to 3x3x3x3 whatever the class, and its g-array addresses come from shared
   index tables, so no two of the fifteen loads per gout element can be reused.
   Unrolled, every address is a compile-time constant.
6. **Range separation is one `fma`**: `rsqrt(aij+akl)` becomes
   `rsqrt(aij+akl + aij*akl/omega^2)`, which is the long-range (erf) operator,
   and `1/omega^2 = 0` gives the full-range kernel back bit for bit.  This is
   `fastk`'s identity, and it is what makes `fastgrad` possible.

### The gout odometer, and why it is not GPU4PySCF's

The wide classes cannot hold their density products in registers on one
thread, so a block splits them over `threadIdx.y`: a lane owns all `nfi`
cartesian functions of the bra-i shell for `PER` of the `(j,k,l)`
combinations.  That is `gen_khigh.py`'s arrangement for the exchange matrix,
and there the only thing a lane needs from `(j,k,l)` is an *address*, which
folds into a base pointer.

A derivative needs more: `fj = 2aj*g(j+1) - j*g(j-1)` and
`fk = 2ak*g(k+1) - k*g(k-1)` use the cartesian *powers* of j and k as
multipliers.  So the odometer is ordered `o = l + nfl*(k + nfk*j)` — l fastest
of the three, where GPU4PySCF orders it i,j,k,l — and `PER <= nfl` gives a lane
a single `(j,k)` pair, so those multipliers are two lane-uniform integers
instead of one per gout element.  The l power is never needed: the fourth
derivative comes from translational invariance, `fl = -fi-fj-fk`.

A lowering term whose power is zero still evaluates its address, which can run
off the front of the 2D-integral array; the array therefore starts `stride_k`
doubles into the shared buffer, and that pad is zeroed once per block.  The
alternative — branching on a lane-uniform integer — costs a predicate per gout
element.

### Two emission styles, chosen by measurement

`reg` puts the whole 2D-integral array in a local array at compile-time
indices, so nvcc keeps the live part in registers, deletes the rest, and there
is no `__syncthreads()` anywhere in the primitive loops; `shm` puts it in
shared memory over `NSQ` quartets and `GSTRIDE` lanes.  `--reg-max` is the
largest `ngout` emitted as `reg`, and 0 / 48 / 64 / 128 measure **1.22x /
1.56x / 1.70x / 1.66x** over the whole `PfPMT`/6-31G\* build
(`bench/perclass_ejk.py`).  64 is the default: it puts `(dp|ps)` and `(ds|pp)`
— 17 % of the build between them — on the register style, where they run at
2.83x and 2.18x instead of 1.33x and 1.80x, and leaves `(dp|dp)` and
`(dd|ps)` on the shared one, where the register style loses.

## `fastgrad`: the long-range operator instead of the short-range one

A range-separated hybrid's exchange is `hyb*K(1/r) + (alpha-hyb)*K(erf(wr)/r)`.
Because `K(erfc) = K(1/r) - K(erf)` that is the same matrix as
`alpha*K(1/r) + (hyb-alpha)*K(erfc(wr)/r)`, and GPU4PySCF's gradient takes the
second form — its own comment says "Prefer computing the SR part".  The
short-range operator's Rys quadrature is the difference of two Boys functions
and needs **twice the roots**; the long-range one is the single `fma` above.
`fastgrad` swaps the two builds round, so both run on the generated kernels at
the same Rys order.  When `hyb == 0` the first exchange build disappears
entirely.

The exchange half of one wB97M-V gradient at a fixed density
(`bench/check_ejk_rsh.py`: `alpha = 1.0`, `hyb = 0.15`, `omega = 0.3`; the two
builds summed, so the comparison is of the same matrix computed two ways):

| | GPU4PySCF, full + SR | `fastejk`/`fastgrad`, full + LR | speedup |
|---|---|---|---|
| `PfPMT`/6-31G\* | 17.56 s | **9.83 s** | 1.79x |
| 70-atom model/def2-TZVPD | 130.87 s | **52.61 s** | 2.49x |

The two routes agree to 1.1e-10 and 3.8e-10 on a per-atom component
(4.2e-10 and 1.2e-9 relative), which is larger than everything else in this
file and is not round-off: `K(1/r) - K(erfc)` and `K(erf)` are the same
operator but not the same *screening*, so the two paths drop different quartets
at `direct_scf_tol=1e-13`.  It is the same effect `README.md` records for the
fused range-separated exchange *matrix* (8.8e-13 there), six orders below the
1e-4..1e-6 Eh/Bohr at which a gradient is used.

## `fastxcgrad`: contract with the density first

Per grid block the XC gradient wants
`exc1[i,n] = sum_j (grad_n phi_i | v_xc | phi_j) D_ij`, and GPU4PySCF forms the
three `nao_sub x nao_sub` matrices in the middle before contracting them.
Doing the density contraction first makes them unnecessary:

    sum_j (sum_g a_n[i,g] b[j,g]) D_ij  =  sum_g a_n[i,g] (D b)[i,g]

so one `nao_sub^2 x ngrids` GEMM and a row-wise reduction replace three, twice
over — and the second of the two needs `D phi`, which the density evaluation
has already produced.  The density itself goes through the density matrix
rather than the occupied orbitals, for the reason `README_fastxc.md` gives.
Two GEMMs per block where GPU4PySCF issues ten.

The same change applies to the two grid passes of `get_nlc_exc`; the VV10
double sum between them is `numint._vv10nlc`, which is `fastnlc`'s kernel when
that module is imported.

**A later round rewrote this module**: meta-GGA is covered (M06-2X, ωB97M-V,
TPSS), the grid blocks are grouped so libxc is called once per group instead of
once per block, and the whole per-block contribution is one CUDA kernel.
`README_fastxcgrad.md` has that round — the numbers in the tables above are the
per-block version described here, which is now the `FASTXCGRAD_MODE=perblock`
ablation.

## `fastgradh`: the one-electron derivative on the GPU

`get_hcore` builds `(grad i | h | j)` with two PySCF CPU integrals.
`int1e_ipkin` is O(nbas^2); `int1e_ipnuc` carries a sum over every nucleus, so
it is O(natm*nbas^2) and it is the whole of the cost — 13 % of a B3LYP/6-31G\*
gradient on the 545-atom `HcgC`, single-threaded on the host while the GPU
idles.  GPU4PySCF already has the integral: `gto.int3c1e_ip.int1e_grids_ip1`
with the nuclei as the charge points and `-Z` as the charges reproduces
`int1e_ipnuc` to 3e-12.  It is just not what the gradient calls.

## Accuracy

`bench/test_grad.py` runs 18 cases through both codes at the same converged
density and compares the gradient: 6-31G\*, def2-SVP, def2-TZVP, def2-TZVPD,
cc-pVTZ and cc-pVQZ (g functions, where the general kernel runs its multi-tile
loop); B3LYP, B3LYP-D3(BJ), PBE, PBE0, TPSS (meta-GGA, where `fastxcgrad`
falls through), wB97X, wB97X-V, wB97M-V, CAM-B3LYP and HSE06 (`omega < 0`,
where `fastejk` falls through); third-row elements; a cation; and an open-shell
radical through UKS.  The worst absolute difference over all 18 is
**4.8e-12** on a gradient component.

`bench/test_full.py` runs 12 of them again as separate processes, with the
whole stack on or off, so the SCF modules are compared too rather than left
half-patched: energies agree to **9.1e-13 Eh** and gradient components to
**3.4e-11**, and every case converges in both.

The two clusters are the size test: they differ by 1.9x in AO count and 1.9x in
atom count and land within 1 % of each other on the two-electron derivative
(1.64x against 1.66x) and 8 % on the whole gradient (1.72x against 1.85x, the
gap being the one-electron block, which is a larger share of the bigger
system).  That is the empirical form of "nothing here is tuned to a problem
size"; the third size is the 70-atom def2-TZVPD model, 3.8x smaller than
`PfPMT` in AOs.

## What is left

Shares are of the 13.04 s `PfPMT` and 41.48 s `HcgC` B3LYP-D3/6-31G\*
gradients.

| block | share (`PfPMT` / `HcgC`) | why it is still there |
|---|---|---|
| the 2e derivative | 82 % / 88 % | 1.64x / 1.66x. Within it the mid-size register-style classes are the weak ones -- `(ps\|ps)`, `(ps\|pp)`, `(pp\|ss)`, `(pp\|ps)`, `(ds\|ss)`, `(ds\|ps)`, `(ds\|ds)`, `(dp\|ss)`, `(dd\|ss)` run at 1.19-1.49x, against 1.6-1.7x for `(ps\|ss)`, `(ss\|ss)` and `(pp\|pp)` and 1.8-2.8x for everything wider. They are exactly the ones that spill: 24 to 832 bytes of local memory each at 255 registers, because the twelve force accumulators and `nf` density products sit on top of a 2D-integral array that is already 3*g_size doubles wide. Emitting the contraction interleaved with the recurrences, the way GPU4PySCF's generated code does, would shorten every value's live range; the flat "all recurrences, then all contractions" order this generator emits is what forces them all to coexist |
| range separation, unfused | — | `energy_ee` runs the full-range and long-range builds as two passes. `fastrsh` fusing the same two passes in the *SCF* is worth 1.12-1.16x, and the same fusion applies here |
| XC | 10 % / 6 % | 1.6-2.2x done |
| `int3c2e.get_dh1e` | 3.2 % / 2.4 % | untouched. It is GPU4PySCF's int3c2e path and it is already on the GPU |
| one-electron (`fastgradh`) | 3.1 % / 2.8 % | `int1e_ipkin` is what is left, and it is O(nbas^2) on the host |
| overlap, driver | 1.2 % / 1.0 % | `int1e_ipovlp` on the host, 0.07 s and 0.20 s |

## Measured and rejected

| idea | measurement | verdict |
|---|---|---|
| More occupancy: `__launch_bounds__(256, N)` for N = 2, 3, so the kernels run at 128 and 80 registers instead of 255 | whole `PfPMT`/6-31G\* build 1.70x -> 1.09x -> 0.42x | rejected. `fastk` found per-class occupancy tuning worth 1.3-1.5x on the *exchange matrix*; a gradient kernel holds twelve force accumulators and the density products on top of the 2D integrals, and at 128 registers every class spills |
| `#pragma unroll` on the Rys-root loop (GPU4PySCF unrolls it) | spills fall from 808 to 448 bytes on `(ds\|ds)` when it is removed | removed. Unrolling keeps NROOTS generations of the 2D-integral array live at once |
| The shared-memory style for every class (`--reg-max 0`) | 1.22x against 1.70x | rejected |
| The register style for every class it fits (`--reg-max 128`) | 1.66x against 1.70x | rejected: `(dd\|pp)` and `(ds\|dp)` are 2.1x and 2.2x on the shared style and 1.4x and 1.5x on the register one |
| Spreading the two horizontal transfers over every lane instead of the three the vertical recurrences use (`--split`, `gen_khigh.py`'s `--par-hrr`) | 2.25x against 2.13x on def2-TZVPD, but per class it is up on some and down on others by the same margin, and the classes it should help most (`(fp\|dp)`, `(fd\|dp)`) move by 3 % | rejected as inside the run-to-run spread |
| A wider gout tile per lane (`--gout-max 96`) | 2.19x against 2.13x on def2-TZVPD | same, inside the spread |

## Reproducing

```bash
source env.sh
gpurun python bench/prof_grad.py --sys PfPMT --basis 6-31G* --xc b3lyp --d3 \
    --patch fastejk,fastgrad,fastxcgrad,fastgradh
gpurun python bench/check_ejk.py    --sys PfPMT --basis 6-31G*      # the 2e build
gpurun python bench/check_xcgrad.py --sys PfPMT                     # the XC gradient
gpurun python bench/perclass_ejk.py --sys PfPMT --basis 6-31G*      # per class
gpurun python bench/test_grad.py                                    # the 18 cases
gpurun python bench/test_full.py --patch ''  > a.json                # whole stack, A
gpurun python bench/test_full.py --patch all > b.json                # whole stack, B
```

Regenerating the kernels (deterministic — the generator takes nothing from
GPU4PySCF, only the angular-momentum classes):

```bash
src/build_ejk.sh                    # every class an spdf basis reaches
src/build_ejk.sh 3333 --style shm   # one class, one style
```

The def2-TZVPD rows use a 70-atom model cut from `PfPMT_cluster`:

```bash
python - <<'PY'
lines = open('PfPMT_cluster/cluster.xyz').read().splitlines()
sub = lines[2:2+int(lines[0])][:70]
open('work/pfpmt70.xyz', 'w').write(
    f'{len(sub)}\nfirst 70 atoms of PfPMT_cluster\n' + '\n'.join(sub) + '\n')
PY
```
