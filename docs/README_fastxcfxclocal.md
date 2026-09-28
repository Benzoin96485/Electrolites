# A local `nocc` for the CPHF's occupied-side route: the count says no, and the fusion does not pay either

`docs/DF_WORK_LIST.md` **W36**, found by W33.  W33's occupied-side route
(`electrolites/fastxcfxc.py`, `docs/README_fastxcfxcmgga.md`) contracts every
grid block over all `nocc` occupied orbitals, so its margin over the AO-matrix
route is `nsub/nocc`: 5.2x at a ratio of 5.4 (penicillin V, def2-TZVPP) and
1.7x–1.9x at 1.8 (taxol, def2-SVP).  In a fixed basis that margin falls as the
molecule grows, because `nocc` is global and `nsub` is local to the block.  The
entry asked for two things.  The first was to rotate `C_o` to localised
orbitals once per CPHF and keep, for each grid block, only the orbitals with
support on it.  The second was to fuse the density kernel into the `Y` GEMM.
Its falsification criterion was: *"the number of localised orbitals with
support on a grid block's AOs at 113 atoms is not well below `nocc`"*.  It
also said to count this before writing any code.

**The count was made first, and the falsification fires.**  It fires at 113
atoms, and also at 284 atoms and on open-shell metal complexes.  No
localisation was implemented in `electrolites/`.  Measuring the second half
found a ceiling of 10 % of the call, and a hand-written fp64 tensor-core GEMM
that would carry the fusion runs at 81–91 % of cuBLAS on the route's shapes.
The fusion therefore does not pay either.  **Nothing in `electrolites/`
changes.**  This round ships a probe for each measurement, plus one option on
W33's probe.

---

## 0. Summary

* **Support count** (§1).  For each block of the grid the CPHF runs on, the
  probe counts the occupied orbitals `k` with `max_{n,g} |X_k[n][g]| > tau`,
  where `X_k = C_o[:, k]^T ao[n]` on the block's AO subset.  An orbital below
  the screen drops out of both halves of the route.  The count covers three
  sets of orbitals: canonical, Cholesky-localised and Pipek–Mezey-localised.
  It covers 42 and 113 atoms, def2-SVP and def2-TZVPP, and three kinds of
  system: closed-shell organic molecules, the UKS Cu(II) quartet (both
  spins) and the 113-atom UKS Co sextet (both spins).  At the tolerance the
  route needs (§2), `tau` ≤ 1e-8, **every block sees every orbital**.  That
  is 226.0 of 226 at 113 atoms and 92.0 of 92 at 42 atoms in def2-TZVPP,
  for all three sets of orbitals.  The count first drops below `nocc` at
  `tau` = 1e-4: to 161 of 226 for Pipek–Mezey orbitals on taxol.  At that
  threshold the dropped potential columns are 1.4e-4 of the largest one.
* **Why** (§2).  The route's grid is SG-1, 4096 points to a block, and a
  block's AO subset averages **400 of 1123** functions at 113 atoms.  The
  AO screen (`AO_THRESHOLD` = 1e-10) keeps a basis function wherever its
  value exceeds 1e-10.  A localised orbital's tail is above 1e-8 across a
  molecule the size of taxol.  Smaller blocks do not change this: at 256
  points the count at 1e-8 is still 225.9 of 226.  The 284-atom PfPMT⁻ is
  the first case where the count drops below `nocc` at a usable threshold:
  503 of 547 at 1e-8.  By then `nsub/nloc` is 0.83, so W33's ratio test
  already keeps most of those blocks on the matrix route.
* **The fusion's ceiling** (§3).  Nsight Systems on the occupied route
  (A100-SXM4-80GB, TPSS) shows the call's kernels are **87 % GEMMs at 113
  atoms and 89 % in def2-TZVPP**.  `fxc_w_occ_mgga` and `fxc_rho_occ_mgga`
  are 5.8 % + 4.8 % and 4.9 % + 3.9 % respectively, and both run at close
  to the card's bandwidth.  A fused kernel removes only those two kernels,
  so it wins only if the GEMM that carries it stays within 10.6 % (113
  atoms) or 8.8 % (def2-TZVPP) of cuBLAS.
* **The GEMM that would carry the fusion** (§3).  I wrote a plain fp64
  tensor-core GEMM with no epilogue (`benchmarks/fxc_dmma_probe.cu`:
  `mma.m8n8k4`, `cp.async`, two or three stages).  Its best configuration
  reaches **14.51 TF/s against cuBLAS's 15.92 (91 %)** on taxol's `Y` shape.
  On a def2-TZVPP-sized shape it reaches **14.29 against 17.61 (81 %)**.
  Even before the epilogue is added, the fused version would lose in
  def2-TZVPP and gain at most 2 % at 113 atoms.  It was not written.
* **Correctness**: nothing numerical changed.  `tests/test_fxc_occ.py`
  still passes on this branch (every case ≤ 3.8e-15, route taken).

Unless stated otherwise, every number was measured on one A100-SXM4-80GB of
a Perlmutter debug node, with GPU4PySCF 1.8.1.  The counts are not timings,
so they were taken on the login node's A100.  Two probe runs there are
marked as such.

---

## 1. The count

`benchmarks/fxc_local_occ_probe.py` (new) converges a DF SCF and caches its
orbitals.  For the 284-atom case it reads a cached density and uses its
pivoted Cholesky factor.  It then walks the CPHF grid (`mf.cphf_grids`,
SG-1 pruned (50, 194)) with `fastxcfxc`'s own `block_loop` and AO screen.
For each block and set of orbitals it counts:

* **grid**: orbitals with `max_{n=0..3, g} |X_k[n][g]| > tau`, where
  `X_k[n] = sum_{u in block} C_uk ao[n][u][g]`.  This is the screen an
  implementation could apply, because the route computes `X` for every block
  anyway, at `1/nset` of the cost of `Y`.
* **coef**: orbitals with a coefficient above `tau` on one of the block's
  AOs.  This is the coarser screen that needs no grid.  It is always worse.

The three sets of orbitals are the canonical ones; a pivoted Cholesky of
`C_o C_o^T`, which is W15's localisation (`U = C_o^T S L` is checked to be
orthogonal to 1e-8); and Pipek–Mezey (`pyscf.lo.PM`, on the host).  PM took
67 s at 113 atoms and 582 s at 284.  The tables below give the
grid-weighted mean of the local count, with the block maximum after the
slash.

**Taxol, 113 atoms, def2-SVP, `nocc` = 226, 118 blocks, `nsub` mean 400 (`nsub/nocc` 1.77):**

| `tau` | canonical | Cholesky (SCF grid) | **Pipek–Mezey** | PM, coef screen | PM `nsub/nloc` |
|---|---|---|---|---|---|
| 1e-3 | 176.1 / 208 | 118.6 / 208 | **102.3 / 177** | 222.7 | 3.91 |
| 1e-4 | 199.2 / 225 | 180.4 / 226 | **160.8 / 217** | 225.9 | 2.49 |
| 1e-6 | 224.4 / 226 | 224.4 / 226 | **222.4 / 226** | 226.0 | 1.80 |
| 1e-8 | 226.0 / 226 | 226.0 / 226 | **226.0 / 226** | 226.0 | 1.77 |
| 1e-10 | 226.0 / 226 | 226.0 / 226 | **226.0 / 226** | 226.0 | 1.77 |

(The Cholesky column was taken on the level-3 SCF grid: 336 blocks, a third
of the points per unit volume in each one.  That makes the count more
favourable, not less, and even there it does not move at 1e-8.)

**The same count on other sizes, bases and classes.**  All are TPSS
orbitals with Pipek–Mezey localisation; each entry is the grid-weighted
mean local count:

| system | basis | `nocc` | `nsub` mean | 1e-3 | 1e-4 | 1e-6 | **1e-8** | 1e-10 |
|---|---|---|---|---|---|---|---|---|
| penicillin V, 42 | SVP | 92 | 226 | 58.7 | 76.9 | 91.6 | **92.0** | 92.0 |
| penicillin V, 42 | TZVPP | 92 | 500 | 55.7 | 74.1 | 91.8 | **92.0** | 92.0 |
| Cu(II) macrocycle²⁺ quartet, 60, UKS α | SVP | 100 | 315 | 71.7 | 92.4 | 100.0 | **100.0** | 100.0 |
| — β | | 97 | | 70.9 | 89.3 | 97.0 | **97.0** | 97.0 |
| Co sextet, 113, UKS α | SVP | 207 | 391 | 94.7 | 151.0 | 205.6 | **207.0** | 207.0 |
| — β | | 202 | | 96.9 | 149.8 | 200.5 | **202.0** | 202.0 |
| taxol, 113 | SVP | 226 | 400 | 102.3 | 160.8 | 222.4 | **226.0** | 226.0 |
| PfPMT⁻, 284 (Cholesky) | SVP | 547 | 420 | 92.9 | 174.7 | 375.8 | **503.3** | 542.4 |
| PfPMT⁻, 284 (PM, SCF grid) | SVP | 547 | 379 | 66.8 | 131.2 | 331.8 | **485.5** | 538.9 |

    python benchmarks/fxc_local_occ_probe.py --xyz benchmarks/molecules/taxol.xyz \
        --basis def2-svp --xc tpss --loc pm,canon --err --cache $SCRATCH/w36/mo
    python benchmarks/fxc_local_occ_probe.py --xyz benchmarks/molecules/PfPMT.xyz \
        --charge -1 --basis def2-svp --xc tpss --loc chol --err \
        --dm $SCRATCH/dfloc_dm.PfPMT.npy

(The Co sextet's SCF was stopped at 50 cycles unconverged, 75 s on the login
node.  Its orbitals are those of a nearly converged density, which is all a
support count needs.)

The open-shell metal complexes behave like the organic molecules of the same
size.  Chemical class enters the count only through the molecule's size and
the block's AO count.  The Co complex has 113 atoms, like taxol, and gives
the same answer as taxol.

**Block size.**  The 4096-point block is GPU4PySCF's `MIN_BLK_SIZE`.  The
probe's `--blk` option changes it (taxol, def2-SVP, PM, SCF grid):

| points per block | blocks | `nsub` mean | nloc at 1e-4 | 1e-6 | 1e-8 |
|---|---|---|---|---|---|
| 4096 | 336 | 369 | 146.2 | 220.7 | 226.0 |
| 1024 | 1343 | 336 | 131.4 | 218.0 | 226.0 |
| 256 | 5371 | 313 | 122.3 | 215.3 | 225.9 |

A block 16x smaller spatially has barely fewer AOs above the screen, and
exactly as many orbitals above 1e-8.  The AO subset, not the grid points, is
what makes the support non-local.

## 2. What tolerance the route needs, and why the answer is "all of them"

Dropping orbital `k` from a block changes two things.  It removes
`4 Y_k X_k` from that block's density, where `Y_k = M_k^T ao` is not local
because `M = C mo1` runs over the virtual orbitals too.  It also removes
the potential column `(V C_o)[:, k] = sum_n ao[n] W_k[n]`, which is linear
in `X_k`.  The probe's `--err` mode prices each `tau` in those two terms.
It uses three random right-hand sides (`mo1 ~ 1e-2 N(0,1)`, as
`fxc_occ_probe.py` makes them) and reports the largest error on any block:

| `tau` | taxol: density error / max | taxol: dropped column / max column | PfPMT⁻: density | PfPMT⁻: column |
|---|---|---|---|---|
| 1e-3 | 3.5e-5 | 1.3e-3 | 5.0e-5 | 2.2e-3 |
| 1e-4 | 2.8e-6 | 1.4e-4 | 1.7e-5 | 2.3e-4 |
| 1e-6 | 1.0e-8 | 1.1e-6 | 6.4e-8 | 2.6e-6 |
| 1e-8 | 0 | 0 | 2.0e-10 | 2.1e-8 |
| 1e-10 | 0 | 0 | 1.4e-12 | 1.7e-10 |

The potential error is `tau` to within a factor of two, as the linearity
predicts.  The reference points are the CPHF's own convergence tolerance
(`conv_tol_cpscf` = 1e-9 per right-hand side, `hessian/rhf.py:721`), the
AO screen the route already applies (`AO_THRESHOLD` = 1e-10), and W33's
agreement with GPU4PySCF (≤ 3.7e-15).  A usable screen therefore sits at
`tau` ≤ 1e-8, and more safely at 1e-10.  At 1e-8 the local count is `nocc`
on every system up to 113 atoms, in both bases and for both spins.  At 284
atoms it is 0.92 `nocc` (Cholesky) and 0.89 `nocc` (PM).  The entry's
falsification fires as written.

Its warning about W15 applies literally.  W15's orbitals were local on the
index it measured (266 AOs of 2697 at 284 atoms).  W36's orbitals are local
in the same sense: the PM count at 1e-3 is 45 % of `nocc` at 113 atoms.
Neither was local on the index the arithmetic runs over.  Here that index
is "has a value above 1e-8 somewhere on a block whose AO subset is a third
of the molecule".  The only remedy would be to give each block a smaller AO
subset, and that is set by GPU4PySCF's AO screen, not by this route.

## 3. The second half: fusing the round trip

With no localisation to build on, the entry's second step was priced on its
own.  At 113 atoms, W33 attributed "about 20 % of the call" to the `Y`/`W`
round trip.  That figure came from the potential stage's effective rate,
11.8 TF/s against 16.5.  Nsight Systems splits it, for the occupied route
only, TPSS, one call:

    nsys profile -t cuda,nvtx -c cudaProfilerApi -o $SCRATCH/w36/prof \
        python benchmarks/fxc_occ_probe.py --xyz benchmarks/molecules/taxol.xyz \
        --basis def2-svp --xc tpss --reps 2 --patch all --routes occ

| kernel | taxol, def2-SVP | penicillin V, def2-TZVPP |
|---|---|---|
| cuBLAS DGEMMs (every `d884gemm`) | 13.69 s (**87.0 %**) | 1.00 s (**88.8 %**) |
| `fxc_w_occ_mgga` | 0.91 s (5.8 %) | 0.056 s (4.9 %) |
| `fxc_rho_occ_mgga` | 0.75 s (4.8 %) | 0.044 s (3.9 %) |
| `take`, `scatter_update`, `add` (the gathers and the `V C` scatter) | 0.23 s (1.5 %) | 0.024 s (2.1 %) |
| everything else | 0.15 s | 0.007 s |
| **kernel sum** | **15.73 s** | **1.13 s** |

So the round trip is 10.6 % of the call, not 20 %.  The rest of W33's gap in
effective rate came from the `W` kernel and the gathers being charged to a
stage whose flops are only the GEMMs'.  Both kernels already run at the
card's bandwidth.  At 113 atoms `Y` is `4·nset·nocc·ng` = 10.0 GB per block
(`nset` = 339, `nocc` = 226), and the density kernel reads it for 104
blocks in 0.75 s, which is 1.4 TB/s.  So neither kernel can get faster; it
can only disappear.  It disappears only inside the GEMM that produces its
input (the density: an epilogue reducing `Y·X` over `k`) or consumes its
output (the potential: a prologue generating `W` from `wv` and `X` as it
loads the tile).  cuBLAS offers neither.

So the question is whether a GEMM that allows fusion can run near cuBLAS.
`benchmarks/fxc_dmma_probe.cu` is the plain version, with no epilogue, which
bounds any fused version from above.  It uses `mma.sync.m8n8k4.f64`, 16-byte
`cp.async` with the leading dimension padded even (as a fused kernel would
allocate it), a two- or three-stage pipeline, and templated block and warp
tiles.  It agrees with cuBLAS bit for bit (the same m8n8k4 accumulation
order).

| shape `M × N × K` | cuBLAS | best DMMA (`BM,BN,WM,WN,BK,NST`) | ratio |
|---|---|---|---|
| 76614 × 4096 × 401 (taxol's `Y`: `nset·nocc` × block × `nsub`) | 15.92 TF/s | 14.51 (128,64,64,32,16,2) | **91.1 %** |
| 8464 × 4096 × 499 (def2-TZVPP scale) | 17.61 | 14.29 (128,64,64,32,16,2) | **81.2 %** |

    python benchmarks/fxc_dmma_probe.py --shape 76614,4096,401

The other configurations measured were 64×128 (2 and 3 stages) and 128×128
(12.4–13.9 TF/s).  On the login node's PCIe card the best was 87.6 % of
cuBLAS on the taxol shape.

**Break-even.**  Suppose both halves were fused at the rate of the plain
kernel, with zero epilogue cost.  At 113 atoms the call's GEMM time would go
from 13.69 to 15.03 s (at 91.1 % of cuBLAS) to save 1.66 s: a net gain of
0.32 s of 15.7 s, or 2 %.  In def2-TZVPP it would go from 1.00 to 1.23 s to
save 0.10 s, a net loss.  The fused epilogue, a reduction over `k` with two
to four products per element plus atomics across `k` tiles, only adds to
this.  A fused kernel breaks even only above about 90 % of cuBLAS in
def2-TZVPP and 88 % at 113 atoms, **after** its epilogue.  That means
matching a library whose DGEMMs here are at 82–90 % of the A100's fp64
peak.  The fusion is therefore not pursued, and the prototype is kept
runnable so that the next person can check this bar against it.

## 4. What this leaves

* The occupied-side route is at the machine: 87–89 % of its kernel time is
  cuBLAS DGEMMs at 15.9–17.6 TF/s, and the remaining two kernels are at
  bandwidth.  Its margin is `nsub/nocc`, and after this round that is a
  statement about the system rather than something left to tune: `nsub` is
  set by GPU4PySCF's AO screen and grid blocking, and `nocc` cannot be
  reduced locally at any tolerance the CPHF can use, up to 284 atoms.
* So the remaining gap between a 113-atom def2-SVP margin (1.9x) and a
  42-atom def2-TZVPP one (5.3x) is the price of the problem, not of the
  code.  W33 §6 already stated both of its dependences: basis size and
  molecule size.
* A local route would first become reachable in the `nsub` direction, with
  a tighter per-block AO subset.  That is W5's and W15's lesson again: the
  sparsity that exists is not the sparsity a blocked contraction can
  address.  Whether it can be made to exist on the XC grid is a question
  about GPU4PySCF's grid blocking, and this round does not open it.

## 5. Files

* `benchmarks/fxc_local_occ_probe.py` (new): the count and its error
  pricing.  Options: `--loc canon,chol,pm,boys`, `--taus`, `--err`, `--blk`,
  `--scf-grids`, `--dm`, `--cache`, `--hist`.
* `benchmarks/fxc_dmma_probe.py`, `benchmarks/fxc_dmma_probe.cu` (new): the
  fp64 tensor-core GEMM that a fusion would be built on, timed against
  cuBLAS.
* `benchmarks/fxc_occ_probe.py`: new `--routes mat,occ`, so that one route
  can be timed or profiled alone.  The timed calls now sit between
  `cudaProfilerStart`/`Stop`, so `nsys -c cudaProfilerApi` captures the
  calls and not the SCF.
* `electrolites/`: unchanged.  `tests/test_fxc_occ.py` passes on this
  branch.
