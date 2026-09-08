# fast1e — faster three-centre one-electron integrals for GPU4PySCF

`int1e_grids` computes

```
(mu nu | 1/|r - C|)      for every basis pair (mu,nu) and every point C
```

optionally with a Gaussian charge of exponent `q` smeared over each point.  One
kernel family serves **PCM, QM/MM, ESP, CHELPG and the Hessian's
`hess_nuc_elec`**, and an IEF-PCM SCF calls two contractions of it every cycle:
`get_int3c1e_density_contracted` for the potential of the density on the cavity,
and `get_int3c1e_charge_contracted` for the Fock-matrix contribution of the
surface charges.  Together they are **236 ms per SCF cycle of a 35.4 s Taxol
SCF** at GPU4PySCF's default cavity on one A100, and 508 ms per cycle on
Valinomycin.

This is entry **F2** of `instructions/06.gpu4pyscf_kernel_targets.md`, rank 6 in
its §5, where it is described as *"`int3c1e`: no screening whatsoever"* with
P10/P11 — far-field precision and an error budget — as the treatment.  Neither
turns out to be the mechanism: "What each change buys" below says what is, and
"What screening is worth" says what the entry's own suggestion is worth once
measured.

Importing the module replaces all three drivers in `gpu4pyscf.gto.int3c1e`:

```python
import electrolites
electrolites.patch('fast1e')          # or patch_all()
```

Everything around them — the cavity, the `VHFOpt` shell-pair screening, the
Hermite pair ordering, the `cart2sph` transforms, the PCM solver — is
GPU4PySCF's own code, and anything the kernels decline (a short-range operator,
angular momentum past g, a `BasisProdCache` whose layout does not match) falls
through to GPU4PySCF's driver.

## Results (A100-SXM4-80GB, IEF-PCM/B3LYP/def2-SVP, `lebedev_order = 29`)

`benchmarks/pcm_bench.py`, one GPU, patched and unpatched in separate
processes.  The per-call columns are the two contractions on a fixed cavity,
fixed density and fixed charges; the SCF column is the whole calculation.

| system | atoms | AOs | cavity pts | density-contracted | charge-contracted | **per SCF cycle** | SCF wall |
|---|---|---|---|---|---|---|---|
| Vitamin C | 20 | 208 | 2 428 | 5.90 → 3.76 ms (1.57×) | 4.67 → 2.98 ms (1.57×) | **1.57×** | 6.90 → 6.82 s (1.012) |
| Penicillin V | 42 | 430 | 4 721 | 17.81 → 9.85 ms (1.81×) | 14.35 → 9.91 ms (1.45×) | **1.63×** | 15.26 → 15.00 s (1.017) |
| Taxol | 113 | 1 123 | 10 963 | 123.1 → 62.2 ms (1.98×) | 112.9 → 75.8 ms (1.49×) | **1.71×** | 35.43 → 34.08 s (1.040) |
| Valinomycin | 168 | 1 542 | 17 142 | 254.2 → 124.3 ms (**2.04×**) | 253.4 → 162.4 ms (1.56×) | **1.77×** | 54.48 → 51.19 s (1.064) |

Coefficients of variation over 12 repetitions are 0.001–0.073 on the kernels
(the Penicillin V charge-contracted row is the noisiest at 0.11) and
0.0007–0.031 on the SCF.

On the same node with all four GPUs visible — GPU4PySCF splits J and K over
them but the PCM path stays on one — the kernel ratios are unchanged and the
SCF ratios rise to 1.084 (Taxol) and 1.138 (Valinomycin), because the part this
module does not touch got four times as much hardware.  **An SCF ratio here is a
statement about a configuration, not about the kernels.**

Basis dependence, Penicillin V at fixed geometry: def2-SVP 1.63× per cycle,
def2-TZVPP **1.41×**.  Higher angular momenta spend proportionally more time in
the recursion and the contraction, which this module does not change.

**The speedup rises with system size and falls with basis size, which is the
right sign for a far-field treatment**: a bigger cavity means a larger fraction
of (pair, point) tasks are far apart, while a richer basis means more work per
task that is not the Boys function.

### Correctness

| check | result |
|---|---|
| `tests/test_fast1e.py`, 32 cases against PySCF's CPU `intor` | worst error **8.36e-13** relative to the largest element, against **8.37e-13** for unpatched GPU4PySCF on the same cases.  The largest change in the error between the two arms is **2.4e-15** |
| GPU4PySCF's own suites, patched (`tests/run_upstream_int1e_suites.py`) | **78 passed**, same as unpatched: `gto/tests/test_int1e_grids{,_ip,_ipip}.py`, `solvent/tests/test_pcm.py`, `test_pcm_grad.py` |
| IEF-PCM SCF energy, Taxol | ΔE = **+9.1e-12 Eh** against a run-to-run spread of 4.5e-11 Eh |
| IEF-PCM SCF energy, Valinomycin | ΔE = **+1.1e-11 Eh**, spread 3.6e-11 Eh |
| Vitamin C, Penicillin V | ΔE = 0 and +9.1e-13 Eh |

The test grid deliberately contains the nuclei themselves — `r2 = 0`, the one
place the far-field branch must *not* fire — as well as points several Bohr
out, and runs def2-QZVPP (s through g) in both cart and sph, with and without
Gaussian charge exponents.

## What each change buys

`benchmarks/pcm_bench.py` with `FAST1E_NO_LAUNCH_TUNE=1`, per SCF cycle, one GPU:

| | Taxol | Penicillin V |
|---|---|---|
| GPU4PySCF 1.8.1 | 236.1 ms | 32.16 ms |
| + far-field kernels | 155.7 ms (**1.52×**) | 19.91 ms (**1.62×**) |
| + launch shape | 138.0 ms (**1.71×**) | 19.76 ms (1.63×) |

`FAST1E_OFF=1` returns 237.0 ms on Taxol against the unpatched 236.1 ms, which
is the fallback path measuring itself.

### 1. The far field, which is most of it — 1.52–1.62×

For one primitive pair with product exponent `p = aij` at `P`, prefactor `eij`,
and a point `C` with charge exponent `q`, the Boys argument is

```
x = a0 * r2,   a0 = p*q/(p+q) * theta,   r2 = |P - C|^2
```

and above `x = 40` both transcendentals are constant to double precision:

```
erfc(sqrt(40)) = 3.7e-19   -> relative error of F_0 is 3.7e-19   (0.002 ulp)
exp(-40)       = 4.2e-18   -> enters as exp(-x)/F_0(x) = 3.0e-17 (0.13 ulp)
```

That much is only worth skipping two function calls.  The useful part is that
with `erf -> 1` the **whole charge-smearing and range-separation algebra
cancels**.  With `f = q/(p+q)`:

```
prefactor * F_0 = (2*pi*eij/p) * sqrt(theta*f) * sqrt(pi/4) / sqrt(p*f*theta*r2)
                = 2*pi*sqrt(pi/4) * eij * rsqrt(p^3 * r2)
```

— no `q`, no `theta`, which is just the statement that far from a smeared charge
only its total charge is visible.  One `rsqrt` and three multiplies replace
`erf`, `exp`, two square roots and four fp64 divisions.  The order-1 term is
`R_1 = -R_0/r2`, again free of `q`.  The test itself needs no division:
`a0*r2 > X` is `p*q*r2 > X*(p+q)`.

For two or more Rys roots the same substitution collapses the root algebra.
`GINTrys_root<NROOTS>` already branches to `r_k = R_k/x`, `w_k = W_k*sqrt(pi/4x)`
above `x = 35 + 5*NROOTS`; feeding those exact forms into what `GINT_g1e` does
with them gives

```
qt2_over_p_plus_q = R_k / (p*r2)                          exactly
prefactor * w_k   = 2*pi*sqrt(pi/4)*eij*W_k*rsqrt(p^3*r2) exactly
```

so `GINT_g1e_roots` takes the branch **at GPU4PySCF's own threshold** and
computes the same numbers with 2 fp64 divisions instead of 4 + 3·NROOTS
divisions and square roots, and with no table lookup.  `prefactor` moves from
`gy` to `gz`; the three axis recursions are separately linear, so a scalar can
move between them without changing any `gx*gy*gz` product.

**How often the branch fires** (`benchmarks/int3c1e_farfield_probe.py`,
Penicillin V / def2-SVP / `lebedev_order = 29`, 1.84e8 primitive-pair–point
tasks):

| threshold | share of all tasks | share of the NROOTS = 1 tasks |
|---|---|---|
| x > 20 | 96.4 % | 96.2 % |
| x > 40 | **91.9 %** | **92.0 %** |
| x > 60 | 87.9 % | 88.5 % |

and 76.8 % of all tasks are `NROOTS == 1`.  That matters because
**`GINTrys_root<1>` had no large-x branch at all**, and neither did the six
hand-written (ss| and (ps| kernels in `g1e_root_1.cu`, which inline `erf` and
`exp` unconditionally.  Those six kernels are half the time, and they are the
ones that gain — per kernel, `nsys` on Penicillin V / def2-SVP, one GPU, kernel
time only:

| kernel | upstream | fast1e | | share of upstream |
|---|---|---|---|---|
| (ss\| charge-contracted | 20.1 ms | 9.0 ms | **2.24×** | 7.9 % |
| (ps\| charge-contracted | 39.3 ms | 22.1 ms | **1.78×** | 15.4 % |
| (pp\| charge-contracted | 32.2 ms | 27.6 ms | 1.17× | 12.6 % |
| (ds\|, (dp\|, (dd\| charge-contracted | 39.1 ms | 35.8 ms | 1.06–1.11× | 15.3 % |
| L=0 density-contracted | 26.4 ms | 8.7 ms | **3.04×** | 10.3 % |
| L=1 density-contracted | 43.4 ms | 19.8 ms | **2.20×** | 17.0 % |
| L=2 density-contracted | 42.7 ms | 17.6 ms | **2.42×** | 16.7 % |
| L=3, L=4 density-contracted | 11.8 ms | 7.8 ms | 1.39–1.55× | 4.6 % |
| total | 255.0 ms | 148.3 ms | **1.72×** | |

(The density-contracted rows carry the launch-shape change of the next section as well; the
charge-contracted ones do not, because upstream's `10` is already the optimum at
this cavity size.)  The pattern is the point: the more of a kernel's work is the
Boys function, the more it gains, and the `(d*|` charge-contracted kernels — the
ones with 250+ registers and spills — barely move.

### 2. The launch shape — a further 1.13× on Taxol, nothing on Penicillin V

Both upstream drivers hard-code the work per thread:

```python
n_pair_sum_per_thread   = nao_cart      # density-contracted
n_charge_sum_per_thread = 10            # charge-contracted
```

The kernels make about `(ntasks_ij / n) * ngrids` threads and end with one
`atomicAdd` each, so `n` trades parallelism against atomic traffic.  `nao_cart`
means *one thread walks every shell pair of its angular type*, which pins the
launch at roughly `8*ngrids` threads however many pairs there are.  Sweeping it
(density-contracted, patched kernels, ms):

| ngrids | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | upstream |
|---|---|---|---|---|---|---|---|
| 2 428 | 1.61 | **1.24** | 1.26 | 1.41 | 1.67 | 1.83 | 2.01 |
| 4 721 | 12.1 | 6.27 | 4.99 | **4.92** | 5.16 | 5.59 | 6.71 |
| 10 963 | 228.7 | 96.9 | 57.2 | 48.6 | **48.4** | 49.9 | 61.6 |
| 17 142 | 670.7 | 275.5 | 143.1 | 108.4 | **103.2** | 104.5 | 122.6 |

The optimum tracks `ngrids/700` and saturates, so:

```python
n_pair_sum_per_thread   = min(16, max(2, ngrids // 700))
n_charge_sum_per_thread = min(40, max(10, ngrids // 500))
```

Upstream's `10` for the charge-contracted kernel is already right below about
10 000 cavity points; above that the optimum drifts to 20–40, worth 3–5 %.

**This is a hardware tuning and it is fitted.**  The same optima came out on an
A100-PCIE-40GB and an A100-SXM4-80GB, and the curves are flat to within ~5 %
over a factor of two in `n` on both, so the risk is a few per cent rather than a
cliff — but it has not been checked on another GPU generation.
`FAST1E_NO_LAUNCH_TUNE=1` restores upstream's values.

An earlier version of this rule was **wrong in a way worth recording**: it asked
for "enough threads to fill the device" per angular-momentum type,
`n = ntasks_ij*ngrids/(SMs*2048*oversub)`, which gives `n = 1` to every small
type.  On Taxol that made the density-contracted kernel *slower than upstream's
`nao_cart`*, because the small types then paid the full atomicAdd contention.
The fix was to stop deriving `n` from occupancy and read it off the sweep.

## What screening is worth — the part of the catalogue entry that does not pay

The far-field amplitude is also a cheap rigorous bound on a task's
contribution, so "how much can be dropped inside an error budget" (catalogue
P11) has a data answer.  Same probe, Penicillin V, over the same 1.84e8 tasks
whose total `|v|` is 9.85e5:

| drop tasks with \|v\| < | tasks dropped | total dropped magnitude |
|---|---|---|
| 1e-16 | 21.1 % | 1.4e-10 |
| 1e-14 | 26.3 % | 2.4e-8 |
| 1e-12 | 33.0 % | 2.7e-6 |

A fifth to a quarter of the tasks, and they come off the **cheap** branch — after
this patch the far field is one `rsqrt` and a few multiplies.  So screening is
the smaller half of entry F2, not the larger one, and this module does not
implement it.  The catalogue's F2 text should be read with that correction.

## What is left

**Register pressure**, in the charge-contracted kernels.  `ptxas -v` on the
unmodified 1.8.1 kernels at `sm_80`:

| kernel | registers | stack frame | spill st/ld |
|---|---|---|---|
| `charge_contracted_kernel_expanded<2,2>` | **255** | 888 B | 244 / 248 B |
| `charge_contracted_kernel_expanded<2,1>` | 252 | 288 B | 0 |
| `charge_contracted_kernel_expanded<1,1>` | 166 | 192 B | 0 |
| `density_contracted_kernel_general<8>` | **255** | 1432 B | 1472 / 1464 B |

255 registers at 64 threads per block is 256 resident threads per SM, 12.5 %
occupancy.  The cause is `double g[3*NROOTS*(LI+1)*(LJ+1)]` held live across the
whole contraction — 81 doubles for `(dd|`.  The three axis recursions and the
contraction are **independent per Rys root**, so building `g` one root at a time
would divide that array by `NROOTS` at no arithmetic cost.  The per-kernel table
above is the evidence that this is now the limiter: the four `expanded<LI,LJ>`
charge-contracted kernels move 1.06–1.17× where the density-contracted kernels
of the same angular momentum move 2.2–3.0×, and after the patch they are 43 % of
what is left.  Not attempted.

**No `ncu` numbers.** Nsight Compute cannot collect counters on this machine —
"a driver resource was unavailable … no other tool (like DCGM) is concurrently
collecting profiling data" — on both login and compute nodes.  The register and
spill figures above come from `ptxas -v` at compile time; everything else here
is wall clock.  So "occupancy is the limiter" is inferred, not measured.

**The gradient and Hessian paths are only half done.**  `g1e_ip_root_1.cu`'s
five (ss|/(ps| kernels get the `erf`/`exp` skip, which is bitwise identical to
upstream there, but the `*_save_u2` variants that the higher-angular-momentum
`int3c1e_ip`/`_ipip` kernels use keep the full root algebra: `u2` in the far
field still needs one division by `a0`, so the cancellation is smaller and it
was not worth the risk here.  `fast1e` patches only the three `int3c1e`
drivers, so `int3c1e_ip` and `int3c1e_ipip` run GPU4PySCF's kernels unchanged.

**F3 of the catalogue entry has nothing to recover.** It names
`_pcm_left_multiply_S_offdiagonal`'s untiled loop, but in 1.8.1
`solvent/pcm.py:802` takes the matrix-free branch only when
`lowmem_intermediate_storage` is set; the default builds K once and calls
`lu_solve`, and `_pcm_d_s` runs **once per geometry** — 0.026 s of a 141 s Taxol
SCF.

## Compile time

The whole translation unit is one NVRTC compile of about 12 s cold; with the
Python import and a PySCF setup around it, a first call costs 6.7 s for 6-31G\*
and 8.6 s for def2-TZVPP, and CuPy caches it on the source hash afterwards.
Only the angular-momentum classes the basis reaches are compiled (`-DE1E_LMAX`),
and `_function` falls back to the full module if a driver ever asks for a class
the subset left out.

## Files

| | |
|---|---|
| `electrolites/fast1e.py` | the three drivers, the dispatch, the launch rule, the `BasisProdCache` check |
| `electrolites/kernels/cuda/my1e/gint1e_boys.cuh` | the order-0/1 far-field forms |
| `electrolites/kernels/cuda/my1e/gint1e_g1e_roots.cuh` | the same collapse for two or more roots |
| `electrolites/kernels/cuda/my1e/int3c1e_prologue.cuh` | the declarations NVRTC needs, `BasisProdCache` verbatim |
| `electrolites/kernels/cuda/my1e/int3c1e_wrappers.cu` | one `extern "C"` entry point per angular-momentum class |
| `electrolites/kernels/cuda/my1e/{g1e,g1e_root_1,g3c1e,rys_roots,constant}.cu` | GPU4PySCF 1.8.1, patched; see NOTICE |
| `tests/test_fast1e.py` | against PySCF's CPU `intor` |
| `tests/run_upstream_int1e_suites.py` | GPU4PySCF's own suites, patched |
| `benchmarks/pcm_bench.py` | the timing tables above |
| `benchmarks/int3c1e_farfield_probe.py` | the far-field share and the screening headroom |
