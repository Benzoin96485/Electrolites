# fastnlc — a faster VV10 non-local-correlation build for GPU4PySCF

ωB97M-V, ωB97X-V, B97M-V and VV10 add a non-local correlation term whose
potential is a double sum over the quadrature grid,

```
F_i = -1.5 Σ_j  RpW_j / (g_i · gp_j · gt)      g_i  = R²·W0_i  + K_i
U_i =      Σ_j  T_ij · (1/g_i + 1/gt)          gp_j = R²·W0p_j + Kp_j
W_i =      Σ_j  T_ij · (1/g_i + 1/gt) · R²     gt   = g_i + gp_j
```

with `R² = |r_i − r_j|²`.  GPU4PySCF puts the NLC grid at the same level as
the XC grid, so on the 284-atom `PfPMT` cluster that is 3.40e6 points and
**1.13e13 pairs**, and the double sum is **64.4 s of a 134 s wB97M-V/def2-TZVPD
Fock build — 48 % of it**, more than the exchange.  On the 545-atom `HcgC`
cluster it is 240 s of 637 s.

Importing the module replaces `gpu4pyscf.dft.numint._vv10nlc`:

```python
import sys; sys.path.insert(0, '.../side_test/HF/src')
import fastnlc
```

Everything around the double sum — the grid, the density on it, the VV10
parameters, the 1e-10 zero-density threshold, the potential contraction, the
`nr_nlc_vxc` bookkeeping — is GPU4PySCF's own code.  Spin-unrestricted
calculations are covered too: `nr_nlc_vxc` sums the two density matrices before
it reaches the double sum.

## Results (A100-SXM4-40GB, wB97M-V/def2-TZVPD, level-3 grids)

`bench/check_nlc_vmat.py`, one whole `nr_nlc_vxc` at a representative density:

| system | atoms | AOs | grid points | pairs | GPU4PySCF | fastnlc | speedup |
|---|---|---|---|---|---|---|---|
| `PfPMT_cluster` | 284 | 6609 | 3.40e6 | 1.13e13 | 65.09 s | **19.87 s** | **3.28×** |
| `HcgC_cluster` | 545 | 12301 | 6.45e6 | 4.09e13 | see below | | |

and the double sum on its own (`bench/check_nlc.py`), which is what the
kernels touch: 60.78 s → 22.69 s without the grid trimming below, i.e. the
kernel change alone is 2.68×.

## Where the time goes, and what each change buys

`bench/nlc_micro.py` keeps every 8th grid point — the same spatial extent, so
the same near/far pair mix, at 1/64 of the pair count — which turns a 61 s
reference pass into 1.07 s and makes the variants comparable in one run.

| variant | time | speedup | ΔE_nlc |
|---|---|---|---|
| GPU4PySCF `vv10_kernel` | 1067 ms | 1.00× | — |
| fp64, one MUFU-seeded reciprocal (`FASTNLC_RCUT=inf`) | 875 ms | 1.22× | 0 (1.8e-16 per point) |
| + fp32 beyond 10 Bohr (`FASTNLC_RCUT=10`) | 379 ms | 2.82× | 9.4e-13 Eh |
| + the grid trimming (`FASTNLC_BUDGET=1e-12`) | 272 ms | **3.92×** | 1.1e-12 Eh |

### 1. One reciprocal, and it is not a division — 1.22×

GPU4PySCF's kernel is already down to a single fp64 division per pair (it
forms `T = RpW/(gp·ggt²)` and recovers `F`, `U` and `W` from it), and at
1.13e13 pairs × ~26 effective fp64 operations it runs at the A100's fp64 peak.
So the only way through is fewer operations.  A `div.rn.f64` is about fifteen
of them.  Ours is

```
inv = rcp.approx.ftz.f64(x);  inv = inv*(2-x*inv);  inv = inv*(2-x*inv);
```

— one MUFU.RCP64H and two Newton steps, about five.  Two steps take the
2^-23 seed to below fp64 round-off, and the result agrees with GPU4PySCF's
per-point `exc` to 1.8e-16 relative.  The seed cannot underflow: the caller's
own 1e-10 density cut puts `K = Kvv·ρ^(1/6) > 0.34`, so `g·gp·gt > 0.08`; where
it overflows, the true reciprocal is below 3e-39.

`FASTNLC_RCP=0` puts the IEEE division back, which is the ablation: it
measures 1.00× against GPU4PySCF, i.e. all of this 1.22× is the reciprocal.
A `1.0f/(float)x` seed is *not* a shortcut — without fast math that is a full
IEEE fp32 division, and it measured 1.01×.

### 2. The distant pairs in single precision — 2.82×

The pair kernel falls off as R⁻⁶, but not fast enough to cut off: on `PfPMT`
only 3.2 % of pairs are within 6 Bohr and 41 % are beyond 24 Bohr, and
dropping everything past 25 Bohr moves E_nlc by about 5 mEh.  What the decay
does buy is *precision*.  A pair at 10 Bohr contributes ~1e-5 of what a pair at
1 Bohr does, so evaluating it in fp32 costs 1e-7 of 1e-5 of the answer while
running at twice the fp64 rate on an A100 (19.5 against 9.7 TFLOP/s) and
turning the reciprocal into a single `rcp.approx.ftz.f32`.

The choice is made per (outer block, inner tile) from their bounding boxes, so
it is uniform across the block — no divergence, and the fp64 path is untouched
where it is used.  For a box test to screen anything the points have to be
spatially local, and the grid GPU4PySCF hands us is ordered by nearest atom
and then by radial shell, which puts a whole Lebedev sphere in consecutive
slots; `fastnlc` Morton-orders the points first, which reorders the same sum.
Sorting on its own is free (`RCUT=inf` measures 880 ms sorted against 875 ms
unsorted).

fp32 positions are stored relative to the outer block's box centre, so a
distant point carries ~3e-6 Bohr of coordinate noise — 6e-7 of R at 10 Bohr,
and 4e-6 of a pair term that is itself 1e-5 of the sum.

`FASTNLC_RCUT` is the knob and both ends are measurable: `inf` is the exact
fp64 kernel above, `0` puts *every* pair in fp32 and still lands within
1.2e-10 Eh, which is the empirical statement that the error budget here is
not tight.  10 Bohr is where the curve flattens (2.82× against 2.85× at 6 Bohr
and 2.76× at 0) with the smallest error, so it is the default.

### 3. The grid points that carry no weight — 3.92×

Every term of every sum is `ρ_j·w_j / (g·gp·gt)` with the denominator bounded
below by 0.08, so a set of points left out of the inner sum changes any `F_i`
by at most `1.5·Σ|ρ_j w_j| / 0.08`.  On a level-3 grid **14 % of the points
carry a total |ρw| of 1e-14** — the tails of the atomic grids, where
GPU4PySCF's 1e-10 density cut removes only 0.8 % of the points.  `fastnlc`
drops the smallest-|ρw| points until their total reaches `FASTNLC_BUDGET`,
which states the same idea as an error rather than as a density.

The same set comes out of the outer sum, where a point's share of E_nlc is
`ρ_i w_i (Beta + F_i/2)` and of the potential matrix is
`w_i vxc_i ⟨ao_i|ao_j⟩` — both bounded by the same `|ρ_i w_i|`.

Measured on `PfPMT` (`bench/nlc_micro.py --stride 4 --budget ...`):

| budget | points dropped | speedup | ΔE_nlc | bound on ΔE_nlc |
|---|---|---|---|---|
| 0 | 0 % | 2.76× | 4.3e-12 Eh | — |
| 1e-14 | 13.9 % | 3.73× | 5.1e-12 Eh | 1e-10 Eh |
| **1e-12** | **15.3 %** | **3.86×** | **5.3e-12 Eh** | 1e-8 Eh |
| 1e-10 | 17.0 % | 4.05× | 5.1e-12 Eh | 1e-6 Eh |
| 1e-8 | 19.4 % | 4.31× | −3.0e-11 Eh | 1e-4 Eh |
| 1e-6 | 22.3 % | 4.68× | −3.7e-9 Eh | 1e-2 Eh |

The observed error is about 4e-3 of the budget, i.e. the bound is loose by two
and a half orders of magnitude, because it assumes every dropped point sits at
the minimum of `g·gp·gt`.  The default 1e-12 is the last budget whose *bound*
is smaller than an SCF convergence threshold; `FASTNLC_BUDGET=0` turns the
trimming off.

The trimming makes the per-point `exc` and `vxc` differ by 100 % at the dropped
points, so the only meaningful comparisons are the integrated ones.  On
`PfPMT` at a fixed density: the electron count agrees to all 13 digits
`nr_nlc_vxc` reports, E_nlc to 1.2e-10 Eh, and the potential matrix to
**3.3e-10 relative in Frobenius norm** (5.2e-10 in max norm).

## Accuracy

`bench/test_nlc.py` runs nine cases through both codes and compares the
converged energy at `conv_tol=1e-10`: `wb97m-v`, `wb97x-v`, `b97m-v` and plain
`vv10`; `6-31G*`, `def2-SVP`, `def2-TZVP`, `def2-TZVPD` and `cc-pVTZ`; a
third-row element; and an open-shell radical through UKS.  All nine converge in
both codes and agree to **1.7e-13 Eh or better**, except the UKS radical at
3.0e-8 Eh.

## Is any of this size-dependent?

| ingredient | depends on | handling |
|---|---|---|
| tile and block width | nothing | 128 points, one outer point per thread; `FASTNLC_NGO` sweeps the register blocking and 1 measured best on both systems and at three grid densities |
| Morton box size | the extent of the molecule | the axis range divided by 2^16, so the code is scale-free |
| `RCUT` | the decay of the pair kernel | a property of VV10, not of the system; the same 10 Bohr is at the flat part of the curve for both clusters |
| `BUDGET` | the error the caller will accept | an absolute error budget in Eh-like units; the *number* of points it drops follows from the grid |
| near/far split | tile bounding boxes | computed per tile pair on the device |
| padding | the point count | points are padded to a multiple of the tile with `RpW = 0`, so no bounds tests in the inner loop |

The two clusters differ by 1.9× in grid points and 3.6× in pair count and land
within 8 % of each other on the whole-`nr_nlc_vxc` ratio, which is the
empirical form of "nothing here is tuned to a problem size".

## Files

| file | role |
|---|---|
| `fastnlc.py` | the replacement `_vv10nlc`: thresholding, trimming, Morton order, tile boxes, launch |
| `fastvv10.cu` | the two kernels (all-fp64, and mixed with the box test), and a note on the symmetric variant that was tried and rejected |
| `../bench/check_nlc.py` | the double sum against GPU4PySCF, timing and per-point agreement |
| `../bench/check_nlc_vmat.py` | the whole `nr_nlc_vxc`: nelec, E_nlc and the potential matrix |
| `../bench/nlc_micro.py` | the variant sweep on a strided subset, and the pair-distance and |ρw| distributions |
| `../bench/test_nlc.py` | the nine converged-energy cases |
