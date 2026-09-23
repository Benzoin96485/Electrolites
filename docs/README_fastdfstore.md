# The cderi build's other three costs: the store, the buffer, and the optimiser nobody was timing

`docs/DF_WORK_LIST.md` **W11**, **W12** and **W13** — the three items
`docs/README_fastdfcderi.md` §9 left behind when W4 closed. All three are on
the same function, `df/df.py:_cholesky_eri`, and none of them is arithmetic:
they are a host store that overlaps nothing, a page-locked buffer that is
allocated again on every build, and a setup stage that turns out to be a
`cupy` launch-latency loop.

Each of the three names a suspect and each suspect is wrong.

* **W11** says `libvhf_rys.transpose_write` is "a synchronous host-side
  transpose-and-copy on one thread" and asks for the write's GB/s first, on
  the theory that "it is the transpose, not the transfer". It is neither, and
  it is not on the host: disassembled, `transpose_write` calls
  `cudaHostGetDevicePointer` and launches a CUDA kernel on the **default
  stream**, and its 17.4 GB/s does not move by a microsecond when OpenMP is
  given 1, 4, 16 or 32 threads. At 284 atoms it reaches 24.6 GB/s of a link
  that does 26.3 — it is not slow there at all, it is *synchronous*. The
  item's conclusion, a stream and a second buffer, is right — and it is only
  half of what pays, because the *other* half is handing the write to a copy
  engine instead of to the SMs that `fill_int3c2e` is already using. Moving
  upstream's kernel to its own stream and changing nothing else is 1.00x on
  the real function at every size measured.
* **W12** says the page-locking is "paid again on every `DF.build`: a
  geometry optimisation pays it per step, and a range-separated functional
  pays it twice per SCF". The first half is right and the second half is not:
  `DF.range_coulomb` keeps its second `DF` in `_rsh_df`, so the two tensors
  are **alive at the same time** and no pool can hand one the other's buffer.
  What a pool does buy is the repeated build, and it buys all of it — 1.62 s
  down to 0.0 at 113 atoms.
* **W13** asks which of `int2c2e`, `potrf` and the triangular solve the
  20–62 ms is, and whether a CPU factorisation is faster. The answer is
  **none of the three**: 70–80 % of `_decompose_j2c` at the small end is the
  `Int2c2eOpt` object `int2c2e` constructs per call and throws away, and
  inside that, one `O(groups²)` loop of tiny cupy calls building an index
  table. The CPU factorisation is rejected by six to twenty-six times.

---

## 0. Summary

* **W11.** `transpose_write` is a CUDA kernel on stream 0, not a host
  thread; its rate is set by how long a contiguous run it writes into host
  memory, so it gets 17.4 GB/s at 113 atoms (365 kB rows) and **24.6 at 284**
  (888 kB rows) against a link that does 26.3. Put on a second stream with
  two destination blocks, **and handed to a copy engine** instead of to the
  SMs, the host branch's block loop goes **181.4 → 163.8 ms** at 42 atoms,
  **2649.0 → 2385.3** at 113 and **28768 → 23787** at 284, and it is worth
  **1.16x / 1.19x / 1.41x** inside the whole module. Moving upstream's own
  kernel to the same stream and changing nothing else is **1.00x at all
  three sizes** — a store done by the SMs has nothing to overlap with —
  although a *synthetic* block loop says the opposite, which is
  `docs/DF_WORK_LIST.md` §5.7 again. §2.
* **W12.** `cudaHostAlloc` page-locks taxol's 14.32 GB host tensor at
  2.1 GB/s (**6.7–6.9 s**) and `cudaHostRegister` at 8.7 GB/s (1.62 s, which
  is W4's result). A pool across builds makes the **second and later builds
  pay 0.0–0.1 ms**, and a three-step geometry-optimisation loop on taxol
  goes **28.80 s → 7.40 s**. The half of W12 about range-separated
  functionals is not true and the reason is W22. §3.
* **W13.** `_decompose_j2c` is 18.9–42.4 ms at 10–42 atoms, of which
  **13.8–31.6 ms is one loop** that builds `Int2c2eOpt.bas_ij_cache` with
  about eight cupy calls per pair of angular-momentum classes. Built on the
  host and moved in one transfer it is **0.83–1.92 ms**, a 10x–17x, and it
  is the same table to the bit. Keeping the optimiser as well makes a second call at
  the same geometry free: `_decompose_j2c` is **18.77 → 4.39 ms** at 10
  atoms and **41.61 → 9.76** at 23, and `_cholesky_eri` itself, which W4
  left at **0.97x** on glycine, is **1.87x**. The two-centre integrals are
  **exactly** GPU4PySCF's — 0.0, not 1e-14 — at ω = 0, +0.3 and −0.3. §4.
* **It is the same tensor.** Patched against unpatched, element-wise, over
  six molecule/basis pairs, both residencies, all three stores, all three
  metric routes and both aggregations: **6.5e-12 relative**, unchanged from
  `docs/README_fastdfcderi.md` §7 — nothing in this round touches the
  arithmetic. §6.

---

## 1. Where these three sit

`docs/README_fastdfcderi.md` §2 measured `_cholesky_eri` stage by stage and
left this picture at 113 atoms with the cderi on the host: 39.4 % of the
block loop in the three-centre integrals, 36.5 % in the metric contraction,
**24.0 % in the store**, and — outside the loop — 6.8 s of `cudaHostAlloc`
against a 3.4 s loop, plus 20–62 ms of `_decompose_j2c` that matters only at
the small end. W4 took the integrals and the contraction. This round takes
the other three.

They do not overlap in where they pay:

| | what it is | where it pays |
|---|---|---|
| **W11**, the store | 16–24 % of a host-resident block loop | any molecule too large for its cderi to sit on the card |
| **W12**, the buffer | 1.6 s at 113 atoms, 10 s at 284, per build | any calculation that builds more than once: a geometry optimisation, a scanner, a frequency run |
| **W13**, `_decompose_j2c` | 14–32 ms per build, nearly flat in system size | the small end, where the whole build is 30–170 ms |

---

## 2. W11 — the store, and the stream it runs on

### 2.1 What `transpose_write` actually is

W11's first instruction is a falsification: *"Measure the write's GB/s first;
14.31 GB in 821 ms is 17 GB/s, which is well under a PCIe gen4 x16 link and
suggests it is the transpose, not the transfer."*

`benchmarks/df_store_probe.py` measures it on two sets of shapes — taxol's,
a `(45683, 5598)` device block into a `(5598, 319781)` page-locked host
tensor (2.05 GB a block), and PfPMT's, `(111000, 13365)` into
`(13365, 888000)` (11.87 GB a block):

| | taxol, ms | GB/s | PfPMT, ms | GB/s |
|---|---|---|---|---|
| `libvhf_rys.transpose_write` (GPU4PySCF) | 117.8 | 17.37 | 482.3 | **24.61** |
| `cudaMemcpy` of the same bytes, no transpose | 77.6 | **26.35** | 451.6 | **26.28** |
| the tiled device transpose alone, no copy | **2.7** | 744 | **16.1** | 736 |
| device transpose + `cudaMemcpy2DAsync` | **84.0** | **24.36** | 475.4 | 24.96 |
| the same, in 256 MB slices of auxiliary functions | 84.0 | 24.35 | 474.7 | **25.00** |
| upstream's access pattern into mapped host memory | 115.0 | 17.80 | 479.5 | 24.75 |
| the *tiled* transpose into mapped host memory | 116.0 | 17.64 | 1830.9 | 6.48 |

and against `OMP_NUM_THREADS`, at taxol's shapes:

| threads | 1 | 4 | 16 | 32 |
|---|---|---|---|---|
| `transpose_write` | 117.6 ms | 117.6 ms | 117.6 ms | 117.6 ms |

Four digits, unchanged. There is no host loop to give threads to, and
`objdump` says why:

```
transpose_write:
    call   cudaHostGetDevicePointer
    call   __cudaPushCallConfiguration        # grid (naux,1,1) block (512,1,1)
    call   transpose_write_kernel(double*, double*, ulong, ulong, int, int)
```

It is a CUDA kernel, launched on **stream 0**, that writes into the mapped
host buffer — one thread block per auxiliary function, its 512 threads
striding along the pair index. The sixth row of the table is that access
pattern reimplemented here, and it lands within 2 % of upstream at both
sizes, which is what says the reading of the disassembly is right.

**So the item's premise is wrong, and the table says what the right one is.**
The kernel's rate is set by the length of the contiguous run it writes into
host memory, and that run is the block's *pair* count: 45683 pairs is
365 kB a row and gets 17.4 GB/s, 111000 pairs is 888 kB a row and gets
**24.6**, against a link that does 26.3 with no transpose at all. Upstream's
store is not slow because of the transpose and not slow because of a host
thread. At 284 atoms it is not slow at all — it is at 94 % of the link. What
it is, at every size, is **synchronous**.

The last row is the same lesson from the other side: the *tiled* transpose,
which is the right kernel for a device-to-device transpose, writes 32-double
runs and is catastrophic into host memory — 6.48 GB/s at PfPMT's shapes, a
quarter of upstream's. Access pattern is the whole story once the
destination is across PCIe.

### 2.2 The stream, and the second block

`HostStore` owns **two** destination blocks and hands them out in turn, with
an event saying the store two batches back has finished with the one it is
about to hand over, so block `n`'s write runs under block `n+1`'s integrals
on a `non_blocking` stream. The cost is one extra `(batch_pairs, naux)`
buffer — 2.05 GB at 113 atoms, 11.9 GB at 284 — and the object declines the
whole arrangement, with a reason, if `get_avail_mem` says it will not fit.

### 2.3 Who does the write, which the item does not name and which decides it

Two routes share that stream, and they differ only in what performs the PCIe
write.

* **`direct`** is upstream's own kernel, moved. The SMs do the write.
* **`stage`** does the transpose on the device into a staging slice — 2.7 ms
  of 117, 744 GB/s, free — and hands what is left to a **copy engine** as a
  `cudaMemcpy2DAsync`.

On a **synthetic** block loop, with a kernel of a known duration standing in
for `fill_int3c2e` on enough thread blocks to fill the device, `direct` wins:

| | taxol shapes, 7 × 250 ms | PfPMT shapes, 8 × 1200 ms |
|---|---|---|
| the "integrals" alone | 1750 ms | 9600 ms |
| + one store, which cannot be overlapped | 1868 | 10082 |
| GPU4PySCF, on stream 0 | 2594.7 | 13769.0 |
| `direct` — the same kernel, on the store stream | **1886.1** | **10261.6** |
| `stage` — device transpose + copy engine | 1936.7 | 12520.1 |

**On the real function it is the other way round, and by more.**
`benchmarks/df_cderi_bench.py` times the host branch's block loop inside the
module; the `pin` column is GPU4PySCF's store (it changes only the
page-locking, which the loop excludes):

| block loop, cderi on the host | GPU4PySCF's store | `direct` | `stage` |
|---|---|---|---|
| penicillin V, def2-SVP (42 atoms) | 181.4 ms | 182.5 ms (0.99x) | **163.8 ms (1.11x)** |
| taxol, def2-SVP (113 atoms) | 2649.0 ms | 2631.9 ms (1.01x) | **2385.3 ms (1.11x)** |
| PfPMT, def2-SVP (284 atoms) | 28768.4 ms | 28763.7 ms (1.00x) | **23787.1 ms (1.21x)** |

`direct` is 1.00x at all three sizes, to three digits, which is as clean a
null result as this repository has.

Inside the whole module — where the rest of the loop is faster too, so the
store is a larger share of it — `all` against `all -store`:

| | with the store | with upstream's | |
|---|---|---|---|
| penicillin V | **119.7 ms** | 139.2 ms | **1.16x** |
| taxol | **1406.3 ms** | 1675.6 ms | **1.19x** |
| PfPMT | **11589.7 ms** | 16365.5 ms | **1.41x** |

A store done by the SMs has nothing to overlap with, because
`fill_int3c2e` and the metric contraction are what the SMs are already doing.
A spin kernel is not that: it occupies thread blocks without wanting
bandwidth or issue slots, so it leaves room the integrals do not. This is
`docs/DF_WORK_LIST.md` §5.7 — *a kernel benchmark cannot see a real call* —
for the third time in this repository, and it is why `stage` is the default
and `direct` is an ablation rather than the other way about.

**Two other things were measured and are not used.** Splitting `stage`'s
transposes and copies over two streams with two staging slices, so the copy
engine is never behind a transpose, is *slower* — 2177 against 1937 ms at
taxol's shapes, 12880 against 12520 at PfPMT's. And the staging slice has to
be taken along the **auxiliary** index, not the pair index: a slice of pairs
makes every row of the 2-D copy as narrow as the slice (45 kB at a 256 MB
budget) and a copy that narrow does not reach the link, which cost 400 ms of
a 2.6 s synthetic loop. A slice of auxiliary functions keeps every row the
whole block wide, and 84.0 ms sliced is 84.0 ms whole.

### 2.4 The instrument, and why the whole function is the wrong one

The block loop is timed *inside* the module rather than inferred from a
difference of two columns, because the round robin drifts across the columns
of one round on this branch. An `all` column repeated in the last position,
with an identical configuration, read **2218–2259 ms against 1479–1575** in
three separate jobs.

The same three jobs read the **block loop** of those two columns as
1406.3 / 1410.2 ms and 119.7 / 120.0 — four digits apart. So the drift is
entirely outside the loop, in the page-locking and allocation half of the
function, and the loop is the instrument to use for anything that lives
inside it. `docs/DF_WORK_LIST.md` §5.14.

## 3. W12 — the buffer, kept

### 3.1 The rates

`benchmarks/df_store_probe.py --pool`, taxol's 14.32 GB host tensor, built
four times over in one process:

| | 1st | 2nd | 3rd | 4th | GB/s |
|---|---|---|---|---|---|
| `cudaHostAlloc` (GPU4PySCF) | 6709 ms | 6692 ms | 6860 ms | 6841 ms | 2.09–2.14 |
| `cudaHostRegister` (W4) | 1622 ms | 1641 ms | 1645 ms | 1722 ms | 8.31–8.83 |
| `cudaHostRegister` + pool | 1662 ms | **0.1 ms** | **0.0 ms** | **0.0 ms** | — |

W12's success criterion is "the second and later builds of one process pay
nothing", and that is literally what the second row of that table says.

### 3.2 What holds it, and what makes it let go

The pool is a module-level list of page-locked buffers keyed by capacity
(`electrolites/_dfpin.py`). The array handed to the build carries a *lease*
rather than the registration itself, so when the last view of the cderi is
dropped — `DF.reset()` setting `_cderi = None`, which is what a geometry
optimisation step does — the buffer is offered back instead of being
unregistered.

W12 attaches its own falsification: *"the pool holds 86 GB of the node's 251
across a calculation that needs the DRAM for something else. A pool that
releases under `get_avail_mem` pressure is the version to write."* The
pressure that matters is **host** memory, not device memory, so the test is
`MemAvailable` from `/proc/meminfo`: a buffer is retained only while that
stays above a reserve (the larger of 8 GB and a tenth of the node), and the
pool is capped at 60 % of the node's memory, with the least recently released
buffer evicted first. `FASTDFCDERI_POOL_GB` sets the cap explicitly and
`FASTDFCDERI_POOL=0` turns the pool off while keeping the cheap page-locking.

The reuse window is deliberately narrow — a buffer is handed out only for a
request between its capacity and 1.25x smaller than it. A build that asks for
14 GB must not be given the 86 GB buffer of the last one and then hold it for
the rest of the process; what the pool is for is the *same* calculation built
again, and that asks for exactly the same number of bytes.

### 3.3 The half of W12 that is not true

W12 says a range-separated functional "pays it twice per SCF because
`DF.range_coulomb` builds a second object". It does build a second object,
and it does page-lock a second buffer, but it **keeps** the first:
`df/df.py:233` stores the range-separated `DF` in `self._rsh_df[key]` and
both objects live until the SCF ends. Two tensors of the same size are alive
at once, so there is nothing for a pool to hand over, and the pool correctly
does nothing there.

That is worth recording for a second reason: at 284 atoms it means a
range-separated density-fitted SCF holds **two** 86.41 GB host tensors, 173
of the node's 251 GB, and nothing warns. It is a capacity item, not a speed
one, and it belongs on `docs/DF_WORK_LIST.md` rather than here.

What the pool does pay for is every caller that builds the same tensor more
than once at the same size: a geometry optimisation step loop, a scanner, a
finite-difference frequency run, and any benchmark that repeats a build.

---

## 4. W13 — `_decompose_j2c`, which is not any of the three things it is made of

### 4.1 The breakdown

`benchmarks/df_j2c_probe.py`, one A100 of a whole node, minimum of 20
synchronised repeats — this stage is latency, so it needs many more repeats
than an arithmetic one:

| | naux | `_decompose_j2c` | `int2c2e` | of which `Int2c2eOpt()` | the kernel | `cholesky` | `dtrsm` |
|---|---|---|---|---|---|---|---|
| glycine, def2-SVP | 471 | 18.9 ms | 17.1 | **14.1** | 2.9 | 0.70 | 0.40 |
| alanine dipeptide, def2-SVP | 974 | 20.2 ms | 17.4 | **14.3** | 2.9 | 1.67 | 0.92 |
| Pt amine, def2-SVP | 1358 | 42.4 ms | 38.1 | **32.3** | 5.9 | 2.51 | 1.46 |
| Ni dithiolene, def2-SVP | 1992 | 35.4 ms | 29.3 | **23.9** | 5.3 | 3.71 | 2.67 |
| penicillin V, def2-SVP | 2191 | 30.6 ms | 23.2 | **19.5** | 4.1 | 4.42 | 3.16 |
| taxol, def2-SVP | 5598 | 65.3 ms | 25.2 | **15.0** | 10.4 | 14.3 | 26.0 |

W13 names three operations and the answer is none of them. The integral
*kernel* is 2.9–10.4 ms, `potrf` 0.6–13.2 and the solve 0.4–26.0; what the
stage is, everywhere below about a hundred atoms, is the optimiser object
`int2c2e` builds on every call and drops when it returns. It is **nearly flat
in system size** — 14.1 ms at `naux` 471 and 15.0 ms at 5598 — which is the
signature of launch latency rather than of work.

### 4.2 Inside the optimiser

| | l-classes | pairs | `from_cell` | `_scale_sp_ctr_coeff` | `PBCIntEnvVars` | the pair table |
|---|---|---|---|---|---|---|
| glycine | 11 | 66 | 0.00 ms | 0.12 | 0.14 | **13.79** |
| alanine dipeptide | 11 | 66 | 0.00 | 0.12 | 0.14 | **13.84** |
| penicillin V | 13 | 91 | 0.00 | 0.17 | 0.15 | **18.49** |
| Ni dithiolene | 15 | 120 | 0.00 | 0.21 | 0.15 | **23.25** |
| Pt amine | 18 | 171 | 0.00 | 0.17 | 0.14 | **31.57** |
| taxol | 11 | 66 | 0.00 | 0.15 | 1.07 | **14.41** |

It is one loop. `Int2c2eOpt.__init__` builds `bas_ij_cache`, the shell-pair
index table, with about eight individual cupy calls — an `arange` each side,
a broadcast, a `tril_indices`, a ravel, an `asarray` — for **every pair of
angular-momentum classes**, and a def2-SVP jkfit basis on four elements
already has eleven classes and sixty-six pairs. 13.79 ms over 66 pairs is
0.21 ms a pair, which is what eight kernel launches cost. It scales with the
pair count and with nothing else: Pt amine's 171 pairs cost 31.57 ms and
taxol's 66 cost 14.41 at eight times the auxiliary basis.

`SortedGTO.from_cell` costs **0.00 ms**, because the auxiliary basis handed
to `int2c2e` is already a `SortedMole` and `from_mol` returns it unchanged.

None of that table depends on the geometry, on a density, or on anything a
kernel computed. `electrolites/_dfj2c.py` builds the same indices with numpy
and moves the whole table across in **one** transfer, the per-class arrays
being views of it:

| | GPU4PySCF's loop | on the host | |
|---|---|---|---|
| glycine | 13.79 ms | **0.83 ms** | 16.6x |
| alanine dipeptide | 13.84 | **0.98** | 14.1x |
| penicillin V | 18.49 | **1.78** | 10.4x |
| Ni dithiolene | 23.25 | **1.84** | 12.6x |
| Pt amine | 31.57 | **1.92** | 16.4x |
| taxol | 14.41 | **5.26** | 2.7x |

and the probe checks, on every row, that it is the same table.

### 4.3 And then it is kept

The optimiser is also cached, keyed on a fingerprint of `_atm`, `_bas` and
`_env` — the basis structure and the coordinates — so a second call at the
same geometry costs the integral kernel alone: **2.9–10.4 ms** against
17.1–38.1. That reaches the gradient's own `_decompose_j2c`, a Hessian's, and
any repeated build, because `fastdfcderi` replaces `int3c2e_bdiv.int2c2e` at
the *module* level and `df/df.py:602` looks it up there at call time.

One word of `_env` is left out of the fingerprint, and it is the reason a
range-separated functional gets anything out of this at all. `mol.omega` is
not an attribute, it is `_env[PTR_RANGE_OMEGA]`, so `DF.range_coulomb`
setting it changes the array being hashed and a naive fingerprint would
rebuild the optimiser for the second of the two builds every such SCF makes.
It is safe to leave out because the kernel does not read it — `fill_int2c2e`
takes `omega` as an explicit argument — and that is measured rather than
assumed: an optimiser built with ω = 0 in its environment, then asked for
ω = +0.3 and ω = −0.3, reproduces GPU4PySCF's own matrices to **0.0** on a
def2-TZVPP jkfit basis. What *does* read `cell.omega` is `_check_rsh_factors`,
when the call names no omega, so the kept optimiser's cell is put back in step
before every call; that path is checked too, also at 0.0.

### 4.4 The CPU factorisation, rejected

W13's other question is "whether a batched or a CPU factorisation is faster
at those orders". Measured, with both transfers charged to it:

| | naux | cuSOLVER `potrf` | LAPACK `dpotrf` + D2H + H2D | GPU `potrf`+`trsm` | host, both |
|---|---|---|---|---|---|
| glycine | 471 | **0.70 ms** | 4.38 | **1.10** | 4.94 |
| alanine dipeptide | 974 | **1.67** | 7.47 | **2.60** | 9.31 |
| Pt amine | 1358 | **2.51** | 22.14 | **3.98** | 25.68 |
| Ni dithiolene | 1992 | **3.71** | 43.40 | **6.38** | 50.55 |
| penicillin V | 2191 | **4.42** | 55.22 | **7.58** | 62.12 |
| taxol | 5598 | **14.26** | 373.32 | **40.30** | 522.02 |

Six times behind at the smallest size measured and twenty-six times at the
largest, and the gap *grows* with the order, which is the opposite of what a
latency argument would predict — the factorisation is not latency-bound on
the device at any of these sizes, the stage around it was. The lead is
rejected.

The one thing the breakdown does say about `cholesky` is that the copy, the
`tril` and the `dev_info[0]` read around `potrf` are 0.02–0.13 ms of it —
2–8 % — so there is nothing there either.

---

## 5. What it is worth

`benchmarks/df_cderi_bench.py`, every column **round robin** in one process
(`docs/DF_WORK_LIST.md` §5.2), minimum of the repeats, one A100-SXM4-80GB of
a whole node.

**Read the pooled columns as repeats.**  `min` over the repeats of a column
that uses the pool is, by construction, a build that got its buffer for
nothing — which is what a geometry optimisation step or a scanner gets, and
*not* what the single build inside one SCF gets.  `all/cold` empties the pool
before every timed call and is the first-build number; `--steps` is the same
distinction drawn as a sequence.

### 5.1 `_cholesky_eri`, cderi on the device

Nothing in this round is on the host branch's store or buffer here; what
moves these rows is W13, and it moves the small end of the table W4 left at
0.97x.

| | atoms | GPU4PySCF | `sp_mult` | `j2c` | `trmm` | all | |
|---|---|---|---|---|---|---|---|
| glycine, def2-SVP | 10 | 31.1 ms | 1.004x | **1.870x** | 0.982x | **16.6 ms** | **1.87x** |
| Pt amine, def2-SVP | 23 | 91.4 ms | 1.004x | **1.568x** | 1.013x | **56.3 ms** | **1.62x** |
| penicillin V, def2-SVP | 42 | 165.7 ms | 1.129x | 1.122x | 1.138x | **107.0 ms** | **1.55x** |
| penicillin V, def2-TZVPP | 42 | 583.8 ms | 1.302x | 1.012x | 1.241x | **315.7 ms** | **1.85x** |
| taxol, def2-SVP | 113 | 1925.4 ms | 1.229x | 1.009x | 1.460x | **944.1 ms** | **2.04x** |

W4's own table read **0.97x** on glycine and **1.02x** on the Pt amine,
because below `_MIN_WORK` the only thing it had was the aggregation and the
aggregation does nothing there.  W13 is what those two rows were missing.

### 5.2 `_cholesky_eri`, cderi on the host

| | atoms | GPU4PySCF | `pin` (W4) | `pool` (W12) | `all` | | `all/cold` | |
|---|---|---|---|---|---|---|---|---|
| penicillin V, def2-SVP | 42 | 961.6 ms | 2.473x | 3.841x | **169.5 ms** | **5.67x** | 341.5 ms | 2.82x |
| taxol, def2-SVP | 113 | 12035.4 ms | 2.551x | 4.363x | **1574.7 ms** | **7.64x** | 3464.8 ms | 3.47x |
| PfPMT, def2-SVP | 284 | 91278.6 ms | 2.180x | — | **12037.6 ms** | **7.58x** | 24024.3 ms | **3.80x** |

`docs/README_fastdfcderi.md` §6.2 measured the last of those at **2.95x**
with W4 alone (87.1 s → 29.5 s); the first build of a process is now
**24.0 s** and every later one **12.0 s**.

`all` is a repeat and `all/cold` is a first build, and the gap between them
is the page-locking: the pool is the only thing in this round that a single
SCF cannot use.

The `store` column of this table reads 0.99x–1.01x and that is not what it is
worth: it is measured against GPU4PySCF on every *other* axis, which on this
branch means measured against a `cudaHostAlloc` that is two thirds of the
function, and §2.4 says a difference that size cannot be read off this
column at all.  §2.3 has it on the block loop, where it is **1.11x** alone
and **1.16x–1.19x** inside the module.

### 5.3 `_decompose_j2c`, priced on the function rather than inferred

A column difference carries whatever else moved with the switch, so the stage
is timed directly as well:

| | `stock` | `build` | `cache` | |
|---|---|---|---|---|
| glycine, def2-SVP | 18.77 ms | 5.65 | **4.39** | **4.28x** |
| Pt amine, def2-SVP | 41.61 ms | 12.11 | **9.76** | **4.26x** |
| penicillin V, def2-SVP | 31.42 ms | 13.87 | **11.60** | **2.71x** |
| penicillin V, def2-TZVPP | 32.59 ms | 13.75 | **11.54** | **2.82x** |
| taxol, def2-SVP | 66.98 ms | 61.30 | **51.42** | 1.30x |
| PfPMT, def2-SVP | 386.77 ms | 403.75 | **357.12** | 1.08x |

**The last row is where the host-built table stops paying.** `build` is
*slower* than `stock` at 284 atoms — 403.75 against 386.77 — because the
numpy index arithmetic grows with the number of shells while upstream's
launch latency grows with the number of angular-momentum classes, and at that
size the first has overtaken the second. It costs 17 ms of a 24 s first
build, 0.07 %, and the cache more than pays it back on every later one, so
there is no threshold on it; but it is the reason there is a `build` column
at all.

`build` is the host-built pair table on a *first* call and `cache` is a
repeat; the difference between the two columns is what the table costs, and
the difference from `stock` is what the whole optimiser costs.

### 5.4 A geometry-optimisation step loop

`--steps 3`: three whole `DF.build()` calls on geometries displaced by a
thousandth, which is the one caller that pays the page-locking more than
once.  Every column starts cold.

taxol, def2-SVP, cderi on the host:

| | step 1 | step 2 | step 3 | total |
|---|---|---|---|---|
| GPU4PySCF | 9.63 s | 9.59 s | 9.58 s | 28.80 s |
| `cudaHostRegister` (W4) | 4.59 | 4.57 | 4.46 | 13.62 |
| + the pool (W12) | 4.57 | **2.84** | **2.80** | **10.21** |
| the whole module | 3.32 | **1.83** | **2.24** | **7.40 s** |

penicillin V, def2-SVP, cderi on the host:

| | step 1 | step 2 | step 3 | total |
|---|---|---|---|---|
| GPU4PySCF | 0.85 s | 0.85 s | 0.82 s | 2.52 s |
| `cudaHostRegister` (W4) | 0.49 | 0.44 | 0.42 | 1.36 |
| + the pool (W12) | 0.50 | **0.31** | **0.28** | **1.09** |
| W13 alone | 0.88 | 0.79 | 0.80 | 2.48 |
| the whole module | 0.44 | **0.23** | **0.23** | **0.89 s** |

### 5.5 The SCF the build sits in

Both rounds run, the second quoted (`docs/DF_WORK_LIST.md` §5.3).  A single
SCF builds the tensor **once**, so the pool is worth nothing here and this
table is W4's plus W11 and W13:

| | GPU4PySCF | + the module | | build's share |
|---|---|---|---|---|
| taxol, def2-SVP, B3LYP, 15 cycles, cderi on the **host** | 24.00 s | **17.61 s** | **1.36x** | 53 % |
| penicillin V, def2-TZVPP, 14 cycles, cderi on the device | 3.99 s | **3.76 s** | 1.06x | 14 % |

The energies agree to **3.2e-10** and **4.1e-11 Eh**.

`docs/README_fastdfcderi.md` §6.3 measured the first of those at 1.37x with
W4 alone, and that is the honest reading of this table: **inside one SCF
this round is worth almost nothing**, because the two items that are worth
something there — the store, which is 24 % of a block loop that is itself a
fifth of the build once the buffer is cheap, and `_decompose_j2c`, which is
15 ms — are small against a build that is one-time.  What this round is
worth is to everything that builds the tensor *more than once*, which §5.4
is, and to the small end of §5.1, which is the regime W4 left at 0.97x.

---

## 6. Correctness

`python tests/test_dfcderi.py`, extended for this round. The bar is
unchanged — the tensor itself, element-wise against GPU4PySCF's own build,
both implementations in one process — and three things are added to it.

* **All three stores** on every host-resident case, on top of the three
  metric routes and both aggregations that W4's suite already sweeps. None of
  them can change an element, and that is exactly why they need checking: an
  ordering mistake in a two-buffer, two-stream store shows up as a *block* of
  the tensor being the wrong one, which nothing but an element-wise
  comparison would catch. The suite also runs with the staging slice forced
  small enough to be walked many times over, so the slicing itself is
  exercised.
* **The pool**: a second build of the same size must get the first one's
  buffer — checked by address as well as by the counter — and must produce
  the same tensor, because a pool that hands out the wrong buffer computes
  the wrong answer and no timing would say so. `FASTDFCDERI_POOL=0` is
  checked too.
* **The two-centre integrals**, against `int2c2e_orig` at ω = 0, +0.3 and
  −0.3, built and then cached and then cached again, in two basis sets.

And one thing had to be repaired in the suite itself before any of it meant
anything. W13 replaces `int3c2e_bdiv.int2c2e` at the module level, and the
"unpatched" reference — `DFMOD._cholesky_eri` saved before patching — looks
that function up on the module at call time. Without care the reference and
the patched build would share the very thing being checked, and the symptom
would be the two columns agreeing to three digits: `docs/DF_WORK_LIST.md`
§5.13, in a third costume. Both `tests/test_dfcderi.py` and
`benchmarks/df_cderi_bench.py` now force `FASTDFCDERI_J2C=stock` — and every
other switch back to GPU4PySCF's own value — around the reference.

---

## 7. What it does not cover

* **More than one device.** `_dfstore` and `_dfpin` are indifferent to it,
  but nothing here has been measured there, and the cderi build itself
  declines the sparse metric route on more than one device for
  `docs/README_fastdfcderi.md` §8's reason.
* **A device-resident cderi.** W11 and W12 are the host branch only; a
  tensor that fits on the card is neither transposed nor page-locked. W13
  applies to both.
* **A periodic cell.** `_dfj2c` declines anything that is not a molecular
  `Mole`, or whose BvK mesh is not `[1, 1, 1]`, and hands the call back.
* **An auxiliary basis past `L_AUX_MAX`**, likewise.
* **A build too tight for one more block.** `HostStore` asks
  `get_avail_mem` and falls back to upstream's store rather than failing.

`fastdfcderi.stats()` reports each of those, under `store`, `pin` and `j2c`,
because nothing numerical can.

---

## 8. What is left

* **Two call sites that this does not reach.**
  `electrolites/fastdfgrad.py:974` and `electrolites/fastdfj3c.py:644` both
  do `from ... import int2c2e` and so bind the function at import; the
  module-level replacement does not reach them, and each therefore still
  pays 14–32 ms of pair table per call. One line each.
* **A range-separated density-fitted SCF holds two host tensors at once**
  (§3.3) — 173 GB of the node's 251 at 284 atoms, with no warning. A
  capacity item.
* **The pool cannot span processes.** A geometry optimisation that runs one
  process per step pays the page-locking every step no matter what; the
  buffer would have to be `mmap`-backed and named to survive, which is a
  different thing from this.
* **`recontract` allocates inside the block loop.** `_create_pair_recontraction`'s
  closure calls `asarray` on three index arrays per batch, which is
  `docs/DF_WORK_LIST.md` §5.8 exactly; it is upstream's and this round did
  not touch it.
* **The `direct` store is a null result on this card and may not be on
  another.** It is upstream's own access pattern moved to the store stream,
  and it is 1.00x at 42, 113 and 284 atoms because the SMs it writes with
  are the SMs `fill_int3c2e` wants. On a card whose copy engines are
  contended, or against an integral kernel that leaves issue slots free, it
  is the route with no staging buffer at all. It stays runnable for that
  reason and because it is the measurement that separates the stream from
  the copy engine.
* **The host-built pair table stops paying at 284 atoms** (§5.3): `build` is
  17 ms *slower* than `stock` there, because its cost grows with the number
  of shells and upstream's with the number of angular-momentum classes. It
  is 0.07 % of a first build and the cache repays it on every later one, so
  it carries no threshold — but a system large enough for that to matter
  would want one, and the crossover has not been located.
* **The staging slice is a constant.** 256 MB is not tuned; sweeping it at
  284-atom shapes moved a synthetic loop by 5 % and the real build by less
  than the harness resolves. `FASTDFCDERI_STAGE_MB` is there for the
  measurement that would place it.

---

## 9. Reproducing

```bash
# the correctness bar
python tests/test_dfcderi.py [--basis def2-tzvpp]

# W13: `_decompose_j2c` taken apart, and the alternatives
python benchmarks/df_j2c_probe.py \
    --cases glycine,tmc_pt_amine,tmc_ni_dithiolene,alanine_dipeptide,penicillinV \
    --basis def2-svp --reps 20
python benchmarks/df_j2c_probe.py --cases taxol:0:0 --basis def2-svp --reps 10

# W11 and W12: the store's rates, the thread sweep that says there is no
# host loop, the overlap against a synthetic block loop, and the buffer
python benchmarks/df_store_probe.py --naux 5598 --blk 45683 --nblk 7 \
    --threads 1,4,16,32 --compute-ms 250 --pool
python benchmarks/df_store_probe.py --naux 13365 --blk 111000 --nblk 8 \
    --reps 2 --compute-ms 1200 --stage-mbs 256,1024,4096

# what each item is worth on the real function, round robin in one process
python benchmarks/df_cderi_bench.py --xyz benchmarks/molecules/taxol.xyz \
    --basis def2-svp [--cpu-cderi] [--scf]
# W12 end to end: `DF.build()` again and again, as an optimisation step loop
python benchmarks/df_cderi_bench.py --xyz benchmarks/molecules/taxol.xyz \
    --basis def2-svp --cpu-cderi --steps 3
```

Switches, for the ablation columns:

    FASTDFCDERI_STORE=host     upstream's kernel on the default stream
    FASTDFCDERI_STORE=stage    device transpose + `cudaMemcpy2DAsync` on the
                               store stream, so a copy engine does the write
                               (the default)
    FASTDFCDERI_STORE=direct   upstream's own kernel, moved to the store
                               stream and nothing else
    FASTDFCDERI_STAGE_MB=<x>   the staging slice, in MB (default 256)
    FASTDFCDERI_PIN=alloc      GPU4PySCF's `cudaHostAlloc`
    FASTDFCDERI_POOL=0         page-lock the cheap way but keep nothing
    FASTDFCDERI_POOL_GB=<x>    cap the pool at x GB of host memory
    FASTDFCDERI_J2C=stock      GPU4PySCF's `int2c2e`, optimiser per call
    FASTDFCDERI_J2C=build      build the optimiser cheaply but keep none
    FASTDFCDERI_J2C=cache      build it cheaply and keep it (default)
