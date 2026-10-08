# fastjkip1 — the direct Hessian's h1 two-electron term (B15)

`hessian/rhf.py:_get_jk_ip1` builds `((∇i) j|kl)` contracted to J and K for
`make_h1`, one `RYS_build_jk_ip1` launch per angular-momentum class (1372 for
penicillin V, 5292 for the Ni dithiolene anion).  Catalogue B15 lists three
steps; all three are in `electrolites/fastjkip1.py` and `libmyjkip1`, which
`_nvcc` builds from `kernels/cuda/myjkip1/` (written by
`codegen/gen_jk_ip1.py` from GPU4PySCF 1.8.1's sources).

## (a) the device query, once

GPU4PySCF's driver calls `cudaGetDeviceProperties(&prop, 0)` on every launch
(on device 0 whatever the current device).  Measured with events around each
launch: **7.4–7.7 s of host time inside the C entry** for 5292 launches,
**0.11 s** with the multiprocessor count of the current device cached.
`_get_jk_ip1` reads the function off `libvhf_rys` at call time, so ours is
bound over that attribute.

But on an idle node the device was the bottleneck all along: the kernels are
10.3–11.3 s of device time there, so (a) alone is 1.06–1.13x on the term
(more under host contention, where §10.7 saw it double).

## (b) both spins from one integral pass

The open-shell `make_h1` (`hessian/uhf.py`, `uks.py`, and the long-range
pass of `uks.py`) called `_get_jk_ip1` once per spin, and J alone costs as
much as J+K (9.76 against 10.57 s on Ni) — the integrals are the cost, and
they were computed twice.  Every `rys_vjk_ip1` kernel (37 unrolled, one
general) now loops K over several densities: with `n_dm > 1`, J from
density 0 and K from densities `1..n_dm-1` into
`vk[(ia*3+x)*(n_dm-1)+d]`; `n_dm == 1` is GPU4PySCF's contract.
`get_jk_ip1_spins` is GPU4PySCF's `_get_jk_ip1` source with the
single-density assert and the `vk` shape changed, called on
`[dma+dmb, dma, dmb]`; the two `make_h1` are GPU4PySCF's source calling it.
`FASTJKIP1_NO_SPINS=1` keeps (a) and P1 only.

## P1

As for B3 (`codegen/gen_ejk_ip2.py:p1`, same anchors): no fp64 division in
the primitive and root loops of the unrolled and general kernels.  Measured
at ~2 % here.

## Result

B3LYP/def2-SVP, `--patch all`:

| | `FASTJKIP1_OFF=1` | (a) + P1 | (a) + P1 + (b) |
|---|---|---|---|
| Ni(II) dithiolene⁻ UKS Hessian | 78.34 s (`jk_ip1` 22.80 s) | 76.66 s (21.10 s) | **67.26 s (1.16x)** |
| penicillin V RKS Hessian | 34.38 s (`jk_ip1` 3.79 s) | 33.95 s (3.44 s) | — |

Hessian traces agree to 2e-11 (Ni) and 6e-10 (penicillin V).

## Correctness

`tests/test_jkip1.py`: `_get_jk_ip1` with ours against GPU4PySCF's,
element-wise — J+K, J, K, an atom slice, the long-range operator, an
open-shell density, def2-TZVP with f functions on Cu: ≤ 3.2e-13 relative.
The spin path against two GPU4PySCF calls: ≤ 1.5e-10 — one call screens on
all three densities, so it keeps a superset of the quartets the two
per-spin calls keep.  Whole UKS Hessians (B3LYP, ωB97X) against GPU4PySCF's
`make_h1`: ≤ 4.4e-12 Eh/bohr².

## Left

The kernels' own device time (~10 s on Ni for both spins now) is the term;
the catalogue's (c) — emitting the ip1 classes from `gen_ejk`/`gen_khigh`
with the P2–P5 treatment — is the remaining lever and was not built.
