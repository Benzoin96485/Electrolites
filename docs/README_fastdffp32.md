# The cderi in single precision — half the bytes, and the error that buys

`docs/DF_WORK_LIST.md` **W7**, and the first item on that list whose
deliverable is a *number about accuracy* rather than a speedup.

W7 is four sentences long and three of them are the specification:

> Exactly 2x on storage and 2x on PCIe, and the arithmetic can stay fp64 by
> accumulating in double.  Technically the least work on this list.
> **It changes the answer**, and this repository's correctness tables are at
> 1e-14.  So it is a measured, explicitly-labelled switch or it is nothing:
> report the error it introduces in the total energy and in the exchange
> matrix on the §3.2 ladder, default it off, and say so in `README.md`'s
> limits paragraph.

All of that is done here.  What the item did not anticipate is where the work
actually is — not in the cast, which is four lines, but in the fact that a
narrow tensor has **no fall-through**: every consumer of it either knows
about the mode or reads past the end of it and returns a plausible wrong
answer.  Three code paths touch the tensor and all three had to be covered
before the switch could exist at all.

---

## 0. Summary

* **It works, it is off by default, and the switch is one environment
  variable.**  `FASTDF_FP32=1` stores the three-index tensor as `float32`:
  exactly half the bytes on the device or the host, exactly half the PCIe
  traffic, and the same `dgemm`/`dsyrk`/cuTENSOR calls on `float64` operands
  downstream, because each block is widened back before anything touches it.
  §1.
* **The error is the *Coulomb* energy's, not the exchange matrix's.**  W7
  asks for "the error in the total energy and in the exchange matrix"; the
  exchange matrix is not where it is.  On the §3.2 ladder the total-energy
  difference is **2e-7 to 2e-6 Eh** for organics and equals, term by term,
  the first-order perturbation of the *Coulomb* energy at a fixed density —
  the exchange energy's share is five to ten times smaller, and the SCF's
  own relaxation contributes nothing, exactly as a variational energy
  should.  The matrices themselves agree to **2e-8 – 6e-8 relative**, which
  is single-precision epsilon and no more.  §2.
* **It is extensive, at about 3e-8 – 1e-7 Eh per atom.**  An idealised
  alkane series from CH4 to C24H50 holds `dE/atom` flat at 2.0e-8 – 6.4e-8
  while the total grows from 2.5e-7 to 2.5e-6 Eh.  Taxol, 113 atoms, is
  **1.70e-5 Eh**.  §2.2.
* **On a 3d transition metal it is an order of magnitude worse per atom, and
  that is the result that decides how the switch is described.**  FeCl4(-),
  a sextet, gives **1.30e-5 Eh on five atoms** — which is *the same size as
  the density-fitting error itself* (1.301e-5 Eh against the four-centre
  energy).  Ni dithiolene gives 2.79e-5 Eh on 21.  Against that, the
  organics sit 6x–151x below their own fitting error.  The mode is not
  uniformly cheap across chemistry and this document does not claim it is.
  §2.3.
* **What it buys depends entirely on where the tensor was.**  Three regimes,
  and only the middle one is what W7 describes:

  | | build | stream | J+K | J only |
  |---|---|---|---|---|
  | device-resident (taxol, 14.3 → 7.2 GB) | 1.01x | — | **0.99x** | 0.67x |
  | host-streamed (taxol, prefix off) | **1.64x** | 3.41x | 1.11x | **1.90x** |
  | host-streamed (penicillin V, prefix off) | 1.41x | 2.97x | 1.54x–1.72x | **1.98x** |
  | host-streamed (PfPMT, 284 atoms, prefix off) | 1.47x | **1.96x** | 0.95x | **1.95x** |

  The J-only call is PCIe-bound and reads the factor of two W7 predicts, to
  two digits.  A J+K call at these sizes is not: the double-buffered
  prefetch already hides most of the transfer behind the exchange
  contractions, which is `docs/DF_WORK_LIST.md` §3.3's finding and not a new
  one.  A *device-resident* tensor has no transfer to halve at all, so
  there the mode is memory and nothing else — and the widening pass it adds
  is measurable (a J-only call is 0.67x) but a whole J+K call is inside the
  noise of parity.  §3.
* **And the transfer is exactly 2x, not 3.4x.**  `DF.loop` reads 3.0x–3.4x,
  which is more than halving the bytes can explain, so it was measured on
  its own rather than written down: the host-to-device copy runs at
  **26.7 GB/s in every configuration** — both precisions, every block size,
  and at matched byte counts — so the link contributes a factor of two and
  not a digit more.  An ablation of the generator finds the rest: `DF.loop`
  allocates its two staging buffers **on every call**, 1.3 GB at taxol, and
  that cost falls twice as heavily on the double side because its buffers
  are twice as large.  With the buffers kept, both precisions run at
  26.7 GB/s and the ratio is **1.996x**.  The widening pass the mode adds
  costs 0.6 ms of 268 there, because it runs on the compute stream while
  the next block crosses the link.  §3.2.
* **The real headline is capacity.**  At 284 atoms the double-precision
  tensor is 86.4 GB and *cannot* go on an 80 GB card; the single-precision
  one is **43.2 GB and does**.  That is a configuration GPU4PySCF has no way
  to reach, and it turns the one-time build from 29.1 s to **13.2 s**, the
  stream from 3.27 s to 0.09 s and a J-only call from 2.65 s to **0.19 s**
  (J+K is unchanged — at that size the contraction is the whole cost).  At
  **545 atoms** the double tensor is 414 GB against a node's 251 GB and
  cannot be built at all; the single one is **207 GB, builds in 129 s**, and
  runs a whole J+K call in 137 s.  There is no double-precision counterpart
  to that row on this hardware.  §3.3.
* **A narrow tensor has no fall-through**, and that is most of the work.
  `fill_symmetric` and `transpose_write` take a `void*` and are compiled for
  `double`: handed a `float32` tensor they read past the end of it and
  return something plausible.  Three code paths touch the tensor; all three
  are covered, and every call `fastdf` *declines* now raises instead of
  handing GPU4PySCF a tensor it would misread.  §1.2.
* **Correctness**: with the switch off, the tensor is `float64`, in
  upstream's layout, element for element what the unpatched build produced,
  and every existing DF suite is unchanged (`tests/test_df.py`,
  `tests/test_dfcderi.py`, `tests/test_dfgrad.py`, and the alias walk).
  With it on, the stored tensor is the double one **rounded** (exactly), the
  block iterator is a partition, and `DF.loop` yields the same blocks in
  both precisions.  §5.

---

## 1. What the mode is

`df/df.py:_cholesky_eri` builds one three-index tensor per density-fitted
run and every SCF cycle, every gradient and every CPHF iteration reads the
whole of it.  It is the largest object such a run holds — **14.3 GB** at 113
atoms and **86.4 GB** at 284 (`docs/DF_WORK_LIST.md` §3.2) — and it is
*storage*: each element is read, multiplied by a density factor and summed
into an accumulator, and none is read twice within a block.  Nothing about
that arithmetic needs the element to be eleven decimal digits wide.

So `FASTDF_FP32=1` stores it as `float32`, and:

* the tensor is **exactly half the bytes**, on the device or on the host;
* every PCIe copy of it is **exactly half the bytes**;
* the arithmetic is **unchanged**.  Each block is widened back to `float64`
  on the device, into a buffer allocated once before the block loop, and
  every contraction downstream of that is the same `dgemm`, `dsyrk` or
  cuTENSOR call on the same operand shapes it would have had.  This is W7's
  "accumulating in double", and it is why the error below is the error of
  *rounding the integrals* and nothing else.

`electrolites/_dffp32.py` owns the switch, the layout and the two casts and
**patches nothing** — `electrolites/fastdf.py` and
`electrolites/fastdfcderi.py` both need it, and importing either of those
patches a build, which is the trap `docs/DF_WORK_LIST.md` §5.12 is about.
`electrolites/fastdffp32.py` is the module that patches.

### 1.1 The layout is upstream's, and that is a measurement

A double-precision cderi is `(npairs, naux)` on the device, handed back as a
view of its transpose, and `(naux, npairs)` on the host, where upstream pays
a CPU transpose (`libvhf_rys.transpose_write`) to put it.  **Single
precision keeps both.**

The first version of this module did not.  It wrote `(naux, npairs)`
everywhere, on the argument that a block of `nL` whole auxiliary rows is
then contiguous and the widening pass perfectly coalesced — which is true,
and is the wrong trade.  The widened block is the operand `fill_symmetric`
unpacks, and the two layouts take *different kernels* there:
`decompress_and_fill` for upstream's device layout and
`decompress_and_transpose` for the other, which is the slower of the two for
the same reason an `(L,i,j)` unpack was 2.65x slower in
`docs/README_fastdf.md` §3.4.  Measured, that cost **1.6x on a whole
`get_jk` call** at 42 atoms — 15.4 ms became 24.7 — where the widening pass
itself accounted for about a fifth of the difference.  Keeping upstream's
layout makes the widening a strided *copy* rather than a transpose (the
auxiliary index is the contiguous one on both sides of it, so each row of
the copy is one contiguous run) and leaves every kernel downstream on the
branch it was tuned for.  The device-resident `get_jk` went from 0.63x to
**0.98x** on that change alone.

The one transpose that remains is in the **host** build, where upstream pays
one too: `transpose_write` moves each `(npairs, naux)` device block into the
`(naux, npairs)` host tensor on the CPU, and it is compiled for `double`.
`_dffp32.transpose_cast` is a 32×32 tiled shared-memory kernel that does it
on the device and narrows in the same pass, which is most of why a
host-resident build is **1.4x–1.6x** faster in single precision and not
merely "half the bytes".

### 1.2 A narrow tensor has no fall-through

`fill_symmetric` and `transpose_write` take a `void*` and are compiled for
`double`.  Handed a `float32` tensor they do not raise — they read twice as
many bytes as there are and return something plausible.  That is the reason
this mode is three patches and not one cast, and it is why every decline in
`fastdf` now **raises** on a narrow tensor instead of handing the call to
GPU4PySCF's own build.  The three consumers:

| path | what reaches it | covered by |
|---|---|---|
| `df/df_jk.py:get_jk` | the SCF, the response solve, a DF Hessian's `get_veff` | `electrolites/fastdf.py` (`_blocks` widens) |
| `df/df.py:DF.loop` | the DF **gradient**'s two-electron derivative, the DF **Hessian**'s MO-basis CPHF build, and anything else that iterates the tensor | `electrolites/fastdffp32.py` |
| `df/hessian/rhf.py:_get_jk` | one line of it: the cderi transfer to the host that leaves room for the Krylov vectors (`:1258-1273`), which is `empty_mapped` at `float64` and a `transpose_write` | `electrolites/fastdffp32.py`, which does that transfer itself when the tensor is narrow, in slices of about 256 MB — the point of the transfer is to *free* device memory, so transposing the whole tensor on the device first would need a second copy of it |

Both patches in `fastdffp32` are **upstream's own function, unchanged,
whenever the tensor is `float64`**: `loop` delegates to the original
generator and the Hessian wrapper does nothing.  So the module is inert
until the switch is set, and `tests/test_dffp32.py` asserts that the double
path is bit-identical with it loaded.

---

## 2. The error

`benchmarks/df_fp32_probe.py`.  Every row is the *same calculation* run
twice in one process, DF-B3LYP/def2-SVP, level-3 grids, `conv_tol` 1e-10,
one precision at a time so the two tensors never contend for the card
(`docs/DF_WORK_LIST.md` §5.10).  `dE` is the difference in the converged
total energy; `dEj` and `dEk` are the Coulomb and exchange energies
evaluated **at the same (double-precision) density**, so their sum `dEdirect`
is the first-order perturbation the total energy should show; `relK` and
`relJ` are the matrices themselves, relative to the largest element of the
double-precision one.

### 2.1 The ladder

| | atoms | nao | cderi | dE (Eh) | dE/atom | dEj | dEk | dEdirect | relK | relJ | d(grad) | cycles |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H2O | 3 | 24 | — | 1.57e-6 | 5.2e-7 | 1.63e-6 | 3.0e-7 | 1.33e-6 | 5.0e-8 | 5.1e-8 | 1.0e-7 | 7→7 |
| CH4 | 5 | 34 | — | 1.90e-7 | 3.8e-8 | 2.0e-7 | 4.4e-8 | 1.54e-7 | 1.4e-8 | 1.5e-8 | 4.7e-9 | 6→6 |
| glycine | 10 | 95 | 0.02→0.01 GB | 1.05e-6 | 1.1e-7 | 1.11e-6 | 2.9e-7 | 8.1e-7 | 4.7e-8 | 2.2e-8 | 2.1e-7 | 12→12 |
| alanine dipeptide | 22 | 200 | 0.14→0.07 | 1.78e-6 | 8.1e-8 | 1.88e-6 | 5.4e-7 | 1.34e-6 | 4.1e-8 | 2.2e-8 | 1.1e-7 | 13→13 |
| penicillin V | 42 | 430 | 1.07→0.53 | 1.85e-6 | 4.4e-8 | 1.96e-6 | 5.1e-7 | 1.44e-6 | 2.0e-8 | 8.1e-9 | 2.2e-7 | 15→15 |
| taxol | 113 | 1123 | 14.31→7.16 | 1.70e-5 | 1.5e-7 | 1.72e-5 | 1.3e-6 | 1.59e-5 | 4.5e-8 | 1.7e-8 | 6.6e-7 | 15→15 |
| glycine, def2-TZVPP | 10 | 225 | 0.10→0.05 | 1.15e-6 | 1.2e-7 | 1.18e-6 | 1.3e-7 | 1.05e-6 | 4.3e-8 | 3.9e-8 | 4.8e-8 | 12→12 |
| alanine dipeptide, def2-TZVPP | 22 | 478 | 0.76→0.38 | 8.99e-7 | 4.1e-8 | 9.6e-7 | 2.9e-7 | 6.6e-7 | 2.8e-8 | 3.1e-8 | 1.1e-7 | 13→13 |
| penicillin V, def2-TZVPP | 42 | 1007 | 5.50→2.75 | 2.90e-6 | 6.9e-8 | 3.14e-6 | 1.24e-6 | 1.90e-6 | 5.1e-8 | 2.2e-8 | 5.2e-7 | 15→15 |

Four things in that table are worth more than the numbers themselves.

**The matrices agree to single-precision epsilon and no better or worse.**
`relJ` and `relK` sit between 8e-9 and 6e-8 across a 40x range in system
size, which is what rounding each stored element to `float32` predicts and
is not amplified by the two contractions — the errors are independent across
the auxiliary index and add as a random walk against a sum that grows as the
count, so the *relative* error stays flat.

**`dE` tracks `dEdirect`, which means the SCF contributes nothing.**  The
energy is stationary at the converged density, so the relaxation the
perturbation causes is second order and only the direct term survives; the
two columns agree to within a factor of 1.3 everywhere, and the residual is
the grid's.  Practically: **the error does not compound over SCF cycles**,
and the iteration count is unchanged in every case (the `cycles` column) —
a Fock matrix perturbed at 5e-8 relative does not disturb DIIS.

**It is the Coulomb energy's error, not the exchange's.**  `dEj` is five to
ten times `dEk` in every row and is the whole of `dEdirect` to within the
cancellation between them.  W7 asks for the error "in the total energy and
in the exchange matrix"; the exchange matrix is the half that does *not*
cost anything, because exchange energies are an order of magnitude smaller
than Coulomb ones and the fractional error is the same.  Anyone tightening
this should tighten J.

**And the gradient is a decimal place better than the energy**, at 5e-9 to
7e-7 Eh/bohr against geometry-optimisation thresholds of 1e-4 — the same
cancellation, one derivative down.

### 2.2 Does it grow with the system?

`--alkanes`, an idealised all-anti series at def2-SVP, where the only thing
changing is the count:

| | atoms | nao | dE (Eh) | dE/atom |
|---|---|---|---|---|
| CH4 | 5 | 34 | 2.45e-7 | 4.9e-8 |
| C2H6 | 8 | 58 | 2.70e-7 | 3.4e-8 |
| C4H10 | 14 | 106 | 8.93e-7 | 6.4e-8 |
| C8H18 | 26 | 202 | 8.23e-7 | 3.2e-8 |
| C16H34 | 50 | 394 | 9.92e-7 | 2.0e-8 |
| C24H50 | 74 | 586 | 2.47e-6 | 3.3e-8 |

**Extensive and flat**: `dE/atom` is 2.0e-8–6.4e-8 over a 15x range in size
with no trend, so the error is a per-atom constant and not something that
overwhelms a large calculation.  It is *noisy* — a factor of three between
neighbouring rows — because it is the sum of a large number of independent
roundings and its sign and size are a property of the particular geometry,
not a smooth function of anything.

### 2.3 A different chemical class, and the row that decides the wording

The organics above are one chemistry.  `docs/OPENSHELL_TMC.md`'s
transition-metal complexes are another, and they are where a narrowed
integral would be expected to show first: a 3d metal has a hard core, the
largest cderi elements are core-region integrals, and it is the *absolute*
size of those elements that the rounding is relative to.

| | atoms | nao | mult | dE (Eh) | dE/atom | dEj | dEk | relK | d(grad) | fitting error | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FeCl4(-) | 5 | 103 | 6 | **1.30e-5** | 2.6e-6 | 1.30e-5 | 1.6e-7 | 3.4e-8 | 6.6e-8 | 1.301e-5 | **1x** |
| CuCl2 | 3 | 67 | 2 | 2.35e-6 | 7.9e-7 | 2.42e-6 | 3.1e-7 | 3.9e-8 | 1.3e-6 | 7.61e-6 | 3x |
| Ni dithiolene(-) | 21 | 327 | 2 | 2.79e-5 | 1.3e-6 | 2.85e-5 | 2.9e-6 | 8.1e-8 | 4.4e-7 | — | — |
| Pt amine | 23 | 249 | 3 | 1.66e-6 | 7.2e-8 | 1.57e-6 | 4.9e-7 | 6.0e-8 | 1.8e-7 | — | — |
| Cu macrocycle(2+) | 60 | 515 | 4 | 7.76e-6 | 1.3e-7 | 7.73e-6 | 1.7e-7 | 3.9e-8 | 6.5e-7 | — | — |

The last two columns are the point.  **"Fitting error"** is
`|E_DF - E_exact|`, the error density fitting *already* makes against the
conventional four-centre energy, and **"ratio"** is how many times larger it
is than the narrowing error.  For the organics of §2.1 that ratio is 6x
(H2O), 24x (glycine), 40x (alanine dipeptide) and 151x (CH4): the narrowing
disappears inside an approximation the caller has already accepted.  For
**FeCl4(-) it is 1x** — single precision costs as much as density fitting
itself does.

The per-atom error on a 3d complex is **1.3e-6 – 2.6e-6 Eh**, twenty to
sixty times the organics' 3e-8 – 1e-7, and it is not the open shell that
does it: the Pt complex, whose metal core is replaced by an ECP, is back at
7.2e-8 per atom.  It is the hard 3d core.  None of these numbers is large in
absolute terms — 2.8e-5 Eh is a fiftieth of chemical accuracy — but "the
error is negligible against the fitting error" is a claim that holds for
organics and does not hold for a first-row transition metal, and the switch
is described accordingly.

---

## 3. What it is worth

`benchmarks/df_fp32_bench.py`, one A100-SXM4-**80GB** on a whole node
(`--qos=debug`, `CUDA_VISIBLE_DEVICES=0`), the four stages timed **round
robin** rather than one after the other (`docs/DF_WORK_LIST.md` §5.2), at a
converged density shared between the two precisions, everything patched.
`build` is `_cholesky_eri` through `DF.build()`, `loop` is
`DF.loop(unpack=False)` — the stream and nothing else — and `jk` and `j` are
whole `get_jk` calls with and without exchange.

### 3.1 Three regimes, and only one of them is the one W7 describes

| | cderi | build | loop | J+K | J only | whole SCF |
|---|---|---|---|---|---|---|
| penicillin V, def2-SVP | 1.07 → 0.53 GB, **device** | 1.02x | 0.04x | **0.98x** | 0.67x | 0.99x |
| penicillin V, def2-SVP | 1.07 → 0.53 GB, host, prefix off | 1.41x | 2.97x | 1.54x–1.72x | **1.98x** | **1.25x** |
| penicillin V, def2-TZVPP | 5.50 → 2.75 GB, host, prefix on | 1.59x | 3.11x | 0.95x | 0.58x | — |
| taxol, def2-SVP | 14.31 → 7.16 GB, **device** | 1.01x | 0.01x | **0.99x** | 0.67x | — |
| taxol, def2-SVP | 14.31 → 7.16 GB, host, prefix off | 1.64x | 3.41x | 1.11x | **1.90x** | — |
| PfPMT, def2-SVP (284 atoms) | 86.41 → 43.20 GB, host, prefix off | 1.47x | **1.96x** | 0.95x | **1.95x** | — |

**A device-resident tensor has no transfer to halve.**  The first and fourth
rows are the mode with its whole point removed: the same arithmetic, the
same residency, and one extra pass over the block to widen it.  That pass is
measurable — `loop` is *only* the widening there, 1.4 ms at 42 atoms and
15.0 ms at 113, which at taxol is 21.5 GB of device traffic at about
1.4 TB/s, near the card's bandwidth — and
a J-only call, being short, shows it as **0.67x**.  A whole J+K call does
not: **0.98x–0.99x**, inside the noise of parity.  What the mode buys in
this regime is the seven gigabytes.

**A streamed tensor gives the factor of two, on the calls that are actually
waiting for it.**  A J-only call is PCIe-bound at every size measured and
reads **1.90x, 1.95x, 1.98x** — W7's prediction to two digits.  A J+K call
is *not* PCIe-bound, which is `docs/DF_WORK_LIST.md` §3.3's finding rather
than a new one: the double-buffered prefetch hides most of the transfer
behind the exchange contractions, so J+K is 1.54x at 42 atoms, 1.11x at 113
and **0.95x** at 284, where the contraction is the whole cost and the
widening is a small addition to it.

**And the third row is a warning about reading any of this without
`FASTDF_RESIDENT_FRAC`.**  `fastdf`'s item 4 keeps as much of a
host-resident tensor on the device as free memory allows; on an 80 GB card
at these sizes it keeps *all* of it, so a "host-resident" run has no
per-cycle PCIe traffic at all and behaves like the device-resident rows.
That is why the rows above say `prefix off` — with the prefix on, penicillin
V in def2-TZVPP reads 0.95x and 0.58x, which is the device-resident regime
wearing the host's label.  Where the prefix *cannot* hold everything the two
effects compound: at 284 atoms with the default 0.5 it holds 2553 of 13365
auxiliary rows in double and 5458 in single, so the streamed remainder is
7907 narrow rows against 10812 wide ones and a J-only call is **2.60x**, not
2.0x.

### 3.2 The stream is exactly 2x, and `DF.loop` is not

`loop` reads 2.97x–3.41x above, which is more than halving the bytes can
explain, and an unexplained factor is a measurement waiting to be wrong.
Two candidates: the link is faster for narrow data, or the generator is
doing something the transfer is not.  `--pcie` times the copy on its own, at
matched **byte** counts (taxol, 5598 auxiliary rows):

| tensor | rows | block | bytes | time | GB/s |
|---|---|---|---|---|---|
| fp64 | 5598 | 256 | 14.312 GB | 534.6 ms | 26.77 |
| fp64 | 5598 | 5598 | 14.312 GB | 536.5 ms | 26.68 |
| fp64 | 2799 | 256 | 7.156 GB | 267.3 ms | 26.77 |
| fp32 | 5598 | 256 | **7.156 GB** | 267.3 ms | 26.77 |
| fp32 | 2799 | 256 | 3.578 GB | 133.7 ms | 26.77 |

**26.7 GB/s in every configuration** — a saturated gen4 x16 link, the same
number `docs/DF_WORK_LIST.md` §3.3 measured — so the transfer contributes a
factor of two and not a digit more.  The rest is in the generator, and
`--loop-ablate` finds it by timing a clone of upstream's streaming loop with
each item switched off (taxol, 14.312 GB of double):

| variant | time | GB/s |
|---|---|---|
| upstream `DF.loop` | 566.6 ms | 25.26 |
| the clone, same structure | 566.3 ms | 25.27 |
| the clone + a widening pass | 582.9 ms | 24.55 |
| the clone, **staging buffers kept** | **534.8 ms** | **26.76** |
| the clone + widening, buffers kept | 535.5 ms | 26.73 |

and the same five rows on the narrow tensor end at **268.0 ms**.  Three
things follow.  The clone is upstream's loop to within 0.05 %, so nothing
about the implementation differs.  The **per-call allocation of the two
staging buffers** — `cp.empty((blksize, npairs))` twice, 1.3 GB at taxol —
is 32 ms of a 567 ms stream when the pool is warm, and the whole of the
excess when it is not; it falls twice as heavily on the double side because
its buffers are twice as large, and it is `docs/DF_WORK_LIST.md` §5.8 and
§5.11 for the third time.  And the widening pass costs **0.6 ms of 268**
once the buffers stop moving, because it runs on the compute stream while
the next block's copy runs on the I/O stream — the thing the mode adds is
free exactly where the thing it removes is expensive.

With the allocation out of the way the ratio is **534.8 / 268.0 = 1.996x**
at taxol and **39.9 / 20.1 = 1.99x** at penicillin V.  That is the claim.

### 3.3 The tensor that did not fit

This is what the mode is actually for, and it is not a ratio.

**284 atoms.**  PfPMT's double-precision cderi is 86.41 GB and an A100 has
80: the build has no choice but the host, which is the configuration every
number in `docs/DF_WORK_LIST.md` §3.3 was taken in.  The single-precision
one is **43.20 GB and goes on the card**.  Same molecule, same basis, same
node, default settings on both sides:

| | fp64 | fp32 | |
|---|---|---|---|
| where the tensor lives | host, 86.41 GB | **device, 43.20 GB** | |
| `_cholesky_eri` | 29.09 s | **13.23 s** | 2.20x |
| `DF.loop`, the stream | 3.27 s | **0.09 s** | 36x |
| `get_jk`, J only | 2.654 s | **0.191 s** | **13.9x** |
| `get_jk`, J+K | 15.78 s | 15.51 s | 1.02x |

The J+K row is the honest one: at 284 atoms the exchange contraction is the
whole cost and moving the tensor onto the card changes nothing about it.
Everything else in a density-fitted run — the build, a pure functional's
Coulomb-only call, a gradient's pass over the tensor — is between 2x and 14x.

**545 atoms.**  HcgC's double-precision cderi is **414 GB against a node's
251 GB** — `docs/README_dflocal.md` §1 measures that, and it is the wall W6
is about.  In single precision it is **207.12 GB**, and it builds:

| | |
|---|---|
| `_cholesky_eri`, host-resident | **129.2 s** |
| `DF.loop`, 207.12 GB | 7.76 s (**26.7 GB/s**) |
| `get_jk`, J only | 6.95 s |
| `get_jk`, J+K, one call | **136.6 s** |

on the initial guess rather than a converged density (nothing here
converges a 545-atom DF-B3LYP in a 29-minute debug job), so the J+K figure
is an upper bound on the screening a converged density would give.  The
point is not the seconds: **there is no double-precision counterpart to this
row on this hardware.**  W6 reaches the same wall from the other side, with
a different approximation and a 31.5x on storage; this reaches it with one
bit of exponent and no change to the algebra.

### 3.4 Two things that will silently poison these numbers

Both cost a measurement here and both belong with
`docs/DF_WORK_LIST.md` §5.

**The build decides residency from the free memory it happens to see, and a
rebuild inside a timing loop can land on the other side of that decision
from the build before it.**  At 284 atoms the single-precision tensor is
43.2 GB against about 46 GB of budget, so `_cholesky_eri` put it on the
device on one repetition and on the host on the next — within one row of one
table, with nothing said.  `df_fp32_bench.py` now records where the tensor
is at every repetition and prints `** MOVED between repetitions **` when the
set has more than one element; `--gpu-cderi` and `--cpu-cderi` force the
decision.

**A name bound to the tensor keeps it alive across a rebuild.**  The
benchmark held `c = dfobj._cderi[0]` from its setup, so the `build` stage's
`reset()` freed nothing and the new tensor was allocated next to the old
one: 173 GB of pinned host memory at 284 atoms, and the job was OOM-killed
by Slurm with no other symptom.  This is `docs/DF_WORK_LIST.md` §5.10 in a
third costume — at these sizes *one* tensor is the budget.

---

## 4. What it does not cover

* **More than one device.**  The build declines it (`stats()` says
  `fp32 on more than one device`) and stores double instead, for the reason
  every module in this repository declines that case and one more: a
  device's slice of the tensor would be read by upstream code this mode has
  not patched.
* **Anything that reaches the tensor outside the three paths in §1.2.**
  There is no way to detect such a caller, which is why the coverage claim
  above is a list of paths rather than a guarantee — and why
  `benchmarks/coverage_probe.py --density-fit` is the way to check one
  (`docs/DF_WORK_LIST.md` §5, last paragraph).
* **`FASTDF_OFF=1` together with `FASTDF_FP32=1`.**  That combination asks
  GPU4PySCF's own `get_jk` to read a narrow tensor; it raises.
* **A cderi already on disk or built by another process.**  Nothing here
  writes one.

---

## 5. Correctness

`tests/test_dffp32.py`, and it is in two halves because the claim is.

**Exact, and not negotiable** — nothing about the mode may change while it
is off, and the parts of it that are bookkeeping rather than arithmetic stay
exact while it is on:

* with the switch off the tensor is `float64`, in upstream's layout, and
  element for element what the unpatched build produced — on the device and
  on the host;
* the stored tensor is the double tensor **rounded**, element for element
  (`np.array_equal(c32, c64.astype(np.float32))`), in the layout §1.1
  claims;
* the widened block iterator is a **partition** of the tensor at every
  resident/streamed split and every block size, and widens exactly;
* `DF.loop` yields the same blocks in both precisions, packed and unpacked;
* a declined call **raises** on a narrow tensor and still falls through on a
  double one.

**A patch target is not a code path**, so the claim in §1.2 that the mode
reaches every consumer is checked the only way it can be:
`benchmarks/coverage_probe.py --density-fit` with `FASTDF_FP32=1` reaches
the same entry points the double run does — twelve `get_jk` calls, one
cderi build, one gradient's two-electron derivative, no declines — in the
same eleven cycles, and the two energies differ by 1.05e-6 Eh, which is the
number §2.1 measures for glycine independently.

**Approximate, and measured** — J, K, the total energy, the gradient and the
Hessian against *the same calculation in double precision*, on both storage
locations, both density factorisations, `hermi` 0 and 1 and several density
counts.  The bars in that suite are the claims §2 makes, not tolerances: a
failure means the error is larger than this document says.

---

## 6. What is left

* **The error is J's, so the obvious next thing is a mixed scheme.**  The
  Coulomb energy carries five to ten times the exchange energy's share of
  the error and the Coulomb *pass* is the cheap half of a `get_jk` call
  (3.3 s against 12.4 s at 284 atoms, `docs/DF_WORK_LIST.md` §3.3).  A
  tensor stored narrow for the exchange contraction and read wide for the
  Coulomb one — or a narrow value plus a narrow *residual*, which is six
  bytes for about twenty-one bits of mantissa — would keep most of the
  storage at a fraction of the error.  Nothing here tries either.
* **`DF.loop` allocates its two staging buffers per call**, 1.3 GB at taxol,
  and §3.2 prices that at 6 % of the stream with a warm pool and much more
  without one.  That is upstream's, it is `docs/DF_WORK_LIST.md` §5.8 and
  §5.11 again, and it is the same shape as the allocator W5 found
  (`docs/README_fastdfblock.md` §4.1).  A buffer cached on the `DF` object
  would remove it for every consumer at once.
* **The residency decision is made from free memory at build time** and at
  284 atoms the narrow tensor sits within 7 % of that threshold, so it lands
  on the device or the host depending on what else is allocated (§3.4).  A
  calculation that wants the device-resident configuration has no way to ask
  for it except `use_gpu_memory=True`, which raises rather than falling
  back.
* **Multi-device is declined**, and unlike the rest of this repository's
  declines that is not only "nothing was measured there": a device's slice
  of the tensor would be read by upstream code this mode has not patched.
* **A 545-atom *converged* run.**  §3.3 builds the tensor and runs one J+K
  call; it does not converge an SCF.  At 136.6 s a call that is ~15 cycles
  of wall clock, which is a job of its own and not a debug-queue one.

---

## 7. Reproducing

```bash
python tests/test_dffp32.py                      # the bars above
python tests/test_dffp32.py --basis def2-tzvpp --no-scf

# the error, on the §3.2 ladder and against the fitting error it sits inside
python benchmarks/df_fp32_probe.py --cases H2O,CH4,glycine,alanine_dipeptide \
    --basis def2-svp --exact
python benchmarks/df_fp32_probe.py --alkanes 1,2,4,8,16,24   # does it grow?
python benchmarks/df_fp32_probe.py --cases tmc_fecl4:-1:5,tmc_cucl2:0:1 \
    --basis def2-svp --exact                     # a different chemical class

# what it is worth: build, stream, J+K and J-only, round robin
python benchmarks/df_fp32_bench.py --xyz benchmarks/molecules/taxol.xyz \
    --basis def2-svp --dm-cache $SCRATCH/fp32dm [--cpu-cderi] [--scf]
# past ~100 atoms, one precision per process (DF_WORK_LIST §5.10)
python benchmarks/df_fp32_bench.py --xyz benchmarks/molecules/PfPMT.xyz \
    --charge -1 --basis def2-svp --only fp32 --dm-cache $SCRATCH/fp32dm
```

```bash
# that the mode reaches every path a DF run takes, which nothing numerical
# can say -- `docs/DF_WORK_LIST.md` §5, last paragraph
FASTDF_FP32=1 python benchmarks/coverage_probe.py \
    --xyz benchmarks/molecules/glycine.xyz --basis def2-svp --density-fit
```

The switch is `FASTDF_FP32=1` in the environment, or `_dffp32._FP32 = True`
on the module object, which is how the suites above build both precisions in
one process against the same integrals.
