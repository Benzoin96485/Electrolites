# Range-separated hybrids: `fastk`'s `omega` path and `fastrsh`

A range-separated hybrid such as ωB97X needs

```
K = hyb * K(1/r)  +  (alpha - hyb) * K(erf(omega r)/r)
```

with, for ωB97X, `omega = 0.3`, `alpha = 1`, `hyb = 0.157706`.  GPU4PySCF's
`dft.rks.get_veff` builds those two matrices with two independent calls to
`get_k`, so an ωB97X SCF does **two** exchange builds per iteration where
B3LYP does one.  On `PfPMT_cluster` that is 76 % of the run.

Before this round `fastk` declined every `omega != 0` call, so half of that
76 % ran on GPU4PySCF's kernels.  Two changes fix it:

* `fastk` now builds the long-range operator itself — one extra `fma` per
  primitive quartet, see below;
* `fastrsh` replaces `RKS.get_veff` with a copy of GPU4PySCF's that asks for
  the *combination* in one pass, so the two builds share their screening,
  their primitive data and their `atomicAdd`s.

```python
import sys; sys.path.insert(0, '.../side_test/HF/src')
import fastk, fastj, fastxc, fastrsh      # fastrsh needs fastk
from gpu4pyscf import dft
mf = dft.RKS(mol, xc='wb97x'); mf.kernel()
```

## Range separation is one `fma`

For the long-range (erf-attenuated) operator GPU4PySCF scales the Rys
quadrature (`gvhf-rys/rys_roots_for_k.cu`): with `theta = aij*akl/(aij+akl)`
and `theta_fac = w^2/(w^2+theta)` it evaluates the roots at
`theta_fac*theta*rr` instead of `theta*rr`, then multiplies every root by
`theta_fac` and every weight by `sqrt(theta_fac)` — a division, a `sqrt` and a
pass over the shared root buffer, per primitive quartet.

In our kernels `inv_s = rsqrt(aij+akl)` is the *only* place those three
quantities come from:

| quantity | how the kernel forms it |
|---|---|
| Rys argument | `x = aij*akl * rr * inv_s^2` |
| `rt_aa` (the root, scaled) | `rt * inv_s^2` |
| geometric prefactor | `cicj*ckcl * (1/aij) * (1/akl) * inv_s` |

Writing `s = aij+akl` and `t = aij*akl`, the long-range operator wants
`x = w^2 t rr/(w^2 s + t)`, `rt_aa = rt w^2/(w^2 s + t)` and a prefactor
carrying `w/sqrt(w^2 s + t)`.  All three follow from

```
inv_s = rsqrt(s + t/w^2)
```

with nothing else changed, because `1/(s + t/w^2) = w^2/(w^2 s + t)`.  So the
whole of range separation is

```c
double inv_s = rsqrt(fma(t, inv_om2, s));      // inv_om2 = 1/omega^2, or 0
```

and `inv_om2 = 0` reproduces the full-range kernel *exactly* — `fma(t,0,s)` is
`s` with no rounding.  One kernel family serves both operators, and the
long-range one costs one extra `fma` per primitive quartet instead of a
division, a `sqrt` and a scaling pass.

Measured on the `PfPMT` long-range build at a fixed density
(`bench/check_klr.py`): GPU4PySCF 6.56 s, `fastk` 3.07 s, **2.13×** (2.43× as
of the high-`l` round, 2.70 s), with the
K matrices agreeing to 4.3e-15 relative.

The short-range (`omega < 0`, `erfc`) operator is *not* covered: it needs twice
the Rys roots and GPU4PySCF's `s_estimator` screening, and `_usable` still
hands it back.  HSE06 therefore falls through, and is in the test suite for
that reason.

## Fusing the two builds

`k_rs_<class>` is the same kernel with a range loop around the innermost
primitive block:

```c
for (int h = 0; h < NRANGE; ++h) {
    double inv_s = rsqrt(fma(t, h == 0 ? 0. : inv_om2, s));
    ...
    fac *= (h == 0 ? coef0 : coef1);
    rys_roots_reg<NROOTS>(t * rr * inv_s2, rw, tab);
    for (int irys = 0; irys < NROOTS; ++irys) { /* into the same gout */ }
}
```

`NRANGE` is a template parameter, so `k_<class>` (`NRANGE == 1`) compiles to
exactly what it compiled to before.  With `NRANGE == 2` the two operators
accumulate into the *same* `gout` registers, so everything outside the range
loop is paid once instead of twice: the task screening, the ket primitive data
(a division and an `exp` per ket primitive pair), the per-quartet geometry, and
the density contraction with its `atomicAdd`s.  Only the Rys roots and the 2D
recurrences — the part that genuinely differs — run twice.

Measured at a fixed density (`bench/check_krsh.py`), against two separate
`fastk` passes: 5.96 s → **5.41 s** on `PfPMT` (1.102×) and 24.78 s →
**21.83 s** on `HcgC` (1.135×).  That ratio is also the honest measurement of
how much of an exchange build is *not* Rys roots and 2D recurrences — about
18 %; it is not a 2× saving, and it grows slightly with system size rather than
shrinking, which is what one would expect if the shared part is the screening
and the density contraction.

Against GPU4PySCF's own two passes the whole fused build was **2.32×**
(`PfPMT`, 12.57 s → 5.41 s) and **2.23×** (`HcgC`, 48.58 s → 21.83 s).

**Since the high-`l` round** (`logs/high_l_kernels.md`) the same measurements
are 12.65 s → 5.24 s two-pass → **4.69 s** fused on `PfPMT` (2.70×, of which
1.116× is the fusion) and 48.59 s → 21.82 s → **18.89 s** on `HcgC` (2.57×, of
which 1.155×).  At def2-TZVP (5241 AO) it is 226.9 s → 101.2 s → **90.0 s**,
**2.520×**, of which 1.124× is the fusion.  The fusion ratio is the one number
that barely moved, which is the expected outcome: it saves the part of the
build that is *not* Rys roots and recurrences, and the new kernels made the
roots-and-recurrences part cheaper, not the shared part.  Note that only the one-thread-per-quartet
classes are fused: `get_k_rsh` runs `gen_khigh.py`'s kernels once per operator,
with the coefficient riding on the geometric prefactor, so roughly a further
9 % of their time is still on the table.

Screening for the fused classes uses the **full-range** `q_cond`, which is the
larger of the two, so nothing the separate long-range build would have kept is
dropped.  The fused result therefore includes a few quartets whose long-range
contribution the two-pass build screened away; at `direct_scf_tol = 1e-13` that
shows up as an 8.7e-13 (`PfPMT`) / 1.2e-12 (`HcgC`) relative difference from
GPU4PySCF's two-pass answer, and it is on the accurate side of it.  Restoring
the exact long-range test per task is possible — carry `q_cond` for the
long-range operator into the kernel and set the range-loop trip count to 1
where it fails — but it was not done: the residual is already three to ten
orders of magnitude below what an ill-conditioned SCF such as `HcgC`'s
reproduces between two runs of GPU4PySCF itself
(`logs/wb97x_hcgc_scf_stability.md`).

The four classes GPU4PySCF splits across `threadIdx.y` and the seven it does
not unroll at all still take one pass per operator; they carry the coefficient
on the geometric prefactor instead, so they accumulate into the same matrix.

## What `fastrsh` replaces

`RKS.get_veff`, with a copy of GPU4PySCF's in which the
"SR and LR exchange with different ratios" branch calls `fastk.get_k_rsh`.
Every other branch, and everything else in the function — grids, NLC, the
Coulomb build, the energy bookkeeping — is GPU4PySCF's code unchanged, and the
fused path returns `None` (falling back to GPU4PySCF's two calls) for
unrestricted calculations, non-hermitian densities, density fitting, more than
one device, and basis sets with `l > LMAX` shells.
