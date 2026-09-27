# The ip2 kernel was a launch problem first: two Rys kernels launched as their own tiles

`docs/DF_WORK_LIST.md` **W29**, found by W24.  After W24 the
`ejk_int3c2e_ip2` kernel was 1.33 s of a 2.9 s density-fitted ip2 call on
penicillin V, 46–52 % of the call at 21–42 atoms, and the entry asked for
what W19 did to the three-centre Coulomb kernels: profile it class by class,
establish what bounds the high classes, and generate what can be generated.

The profile answered a different question first.  **Before any class is
slow, the launch is**: GPU4PySCF gives each thread block a few hundred
shell pairs and a whole auxiliary batch — every shell of one angular
momentum and contraction pattern — along a grid one block high, and the
kernel holds one 254-register block per SM.  Launched as its own tile
instead — as many pairs as a block runs at once, sixteen auxiliary shells —
the *unchanged* kernel is **2.2x–6.4x** faster, which is the entry's success
bar (W19's 4.4x–6.5x) at 21–60 atoms and short of it at 113.  The two
derivative fills beside it (`fill_int3c2e_ip1`/`_ipaux`) had the same launch
and are **3.8x** by the same change.  Then, with the kernel launched well,
the class-by-class ablation says where it is bound: not by the Rys roots
(1–3 %), but by the contraction of the `g` arrays in shared memory
(53–83 %), and at 113 atoms 18 % by its atomics — which is a generator of
its own, **W31**.

---

## 0. Summary

* **Item 15: `ejk_int3c2e_ip2` launched as its own tile**
  (`electrolites/_ip2launch.py`, §2).  Per auxiliary angular momentum `lk`,
  a block is `nst / min(nst, 16)` pairs — what the kernel's 256 threads run
  side by side at that `lk` — against 16 auxiliary shells along the grid's
  second dimension.  The kernel reads a range's pseudo-density
  component-major *within the range*, so the batch's columns are permuted
  once (§2.2).  Nothing inside the kernel changes; the answers agree to its
  own atomics' spread (§5).
* **Item 16: the two derivative fills sized the same way** (§3).  They walk a
  block's triples as one flat list, so only the pair count needs changing
  and the output layout does not: `ceil(2 nst / nksh)` pairs per block.
  Every caller of `_ip1_evaluator` — the ip2 term's two passes, `h1`, the
  Coulomb-only term — gets it.
* **The kernel alone** (`benchmarks/df_ip2_class_probe.py --whole`, §2.3):
  RESULTS_KERNEL
* **The functions** (§4): RESULTS_FUNCTIONS
* **Whole Hessians** (§4.2): RESULTS_HESSIAN
* **What bounds it now** (§6): an ablation of the kernel's four parts at
  equal launch geometry.  Removing the Rys roots saves 1–3 %; the `gout`
  contraction over the shared-memory `g` arrays is 53 % (taxol), 62 %
  (penicillin V) and 83 % (the Pt(II) complex) of the kernel; the `g`
  recurrences 9–13 %; the per-triple atomics 9 % at 42 atoms and 18 % at
  113.  That is the part W19's method applies to, and it is **W31**.
* **A stage table that lied** (§3.1): half of what `FASTDFHESSIP_TIME`
  reported as "D3 integrals" was the second pass's buffer allocation, which
  now has a row of its own.

Everything is one A100-SXM4-80GB, GPU4PySCF 1.8.1, def2-SVP with its jkfit
set unless a row says otherwise, `patch_all()` on both sides.  "W24" is
`patch_all()` with `FASTDFHESSIP_IP2_TILE=0 FASTDFHESSIP_FILL_TILE=0` — the
previous commit's code paths, bit for bit; "stock" is GPU4PySCF's own
function object.  Measured on a shared interactive node whose other three
GPUs were running other work (§7 says what that does and does not affect).

---

## 1. The launch, and why it was the first problem

`gvhf-rys/ejk_int3c2e_ip2.cu`'s block reads `gout_stride` for its
`(li, lj, lk)` from the scheme (`int3c2e_scheme_ip2`), runs
`nst = 256 / gout_stride` `(pair, auxiliary shell)` triples at a time —
`min(nst, 16)` auxiliary shells times `nst / min(nst, 16)` pairs — and loops
over the rest: pairs outermost, the auxiliary shells of its range inside.

`fastdfhessip._ejk_ip2` (upstream's loop, with item 1's launch constant)
gives it:

* **pairs**: `mol.aggregate_shl_pairs(cache, nsp_per_block[0])`, the
  `lk = 0` scheme's count for each `(li, lj)` — 256 for an `(ss|` pair,
  whatever `lk` the launch is for;
* **auxiliary shells**: the whole batch, `nbatches_ksh = 1` — on penicillin V
  one launch walks 237 `s` shells, another 207 `p` shells, 129 `d` shells.

So a block of `(ss|` pairs against the 237-shell batch runs 16 pairs x 16
shells at a time, 16 x 15 times in series, and a block of `(dd|` pairs
against the `g` batch runs one triple at a time.  The launch is a few hundred
such blocks, on 108 SMs holding one each (254 registers x 256 threads is the
whole register file).  The first probe made this visible by accident: the
kernel restricted to one class of 49 shell pairs against one auxiliary
shell took **0.3 ms**, and against 237 shells **1.5 ms** — a single block's
latency, not throughput.

`benchmarks/df_ip2_class_probe.py --whole` launches the production loop —
every pair class, one auxiliary batch per launch — with the arguments
`_ejk_ip2` builds, on a seeded random pseudo-density (the kernel screens
nothing, so its time does not depend on the values), and takes the launch
geometry as two knobs: `--pchunk` pairs per block (`-1`: the tile per `lk`)
and `--kchunk` auxiliary shells per range.  Penicillin V / Ni(II)
dithiolene⁻ / Pt(II) amine triplet, ms:

| pairs per block | aux shells per range | penicillin V | Ni(II) | Pt(II) |
|---|---|---|---|---|
| upstream (`nsp_per_block[0]`) | whole batch (upstream) | 1319 | 1276 | 622 |
| upstream | 64 | 1008 | 1110 | |
| upstream | 16 | 626 | 883 | |
| 16 | whole batch | 252 | 473 | 362 |
| 4 | whole batch | 366 | 368 | 328 |
| 16 | 16 | 207 | 358 | 304 |
| 8 | 16 | 232 | 324 | 279 |
| 4 | 16 | 368 | 332 | 296 |
| **tile per `lk`** | **16** | **205** | **274** | **272** |
| tile per `lk` | 8 | 309 | 345 | 308 |
| tile per `lk` | 32 | 219 | 292 | 279 |
| tile per `lk` | whole batch | 247 | 359 | 325 |

A fixed pair count cannot be right for every class at once — 16 is best on
penicillin V's `(ss|`- and `(ps|`-heavy set and 8 on the metal complexes,
whose blocks of `(dd|` and `(fd|` pairs against `l = 4–6` shells are the
long ones — and the tile per `lk` is within 1 % of the best fixed count on
each molecule and better than all of them on the Ni complex.  Sixteen
auxiliary shells is the kernel's own `BLOCK_SIZE`; eight doubles the blocks
and loses, 32 halves them and loses less.

---

## 2. Item 15

### 2.1 The tile

`_ip2launch.Tiles` builds one `shl_pair_offsets` per `lk` with
`nst[lk, li, lj] / min(nst, 16)` pairs per block — what
`aggregate_shl_pairs` returns for that per-`(li, lj)` size array; the pair
order, `bas_ij_idx`, is the same for every block size — and cuts each
auxiliary batch into ranges of 16 shells.  The offsets are built on the host
in one transfer and cached per `bas_ij_cache` and block size
(`_ip2launch.pair_offsets`): upstream's builder is one device `arange` per
pair class, and item 16 asks for a new size per auxiliary batch, which at
ten atoms cost a measurable part of a 180 ms call before it was cached.
The kernel takes the ranges as its `ksh_offsets` and the grid's second
dimension as their count; nothing else in its argument list changes.  For
the Coulomb-only term (`_j_energy_per_atom`, which passes an auxiliary
vector instead of a pseudo-density) upstream launches every batch in one
grid; that becomes one launch per `lk`, since the pair tile depends on it.

### 2.2 The layout

With a pseudo-density the kernel reads

    dm[pair, ao_loc[ksh0] - aux_offset + k * nksh + (ksh - ksh0)]

for component `k` of shell `ksh` in the range `[ksh0, ksh1)` it was given,
`nksh = ksh1 - ksh0`: component-major within the *range*.  Over the whole
batch that is GPU4PySCF's own auxiliary order (`reorder_aux`), so upstream
never notices.  Cut into ranges, each range wants its own component-major
block at its own offset, and `Tiles.permutation` is the column order that
produces it from the batch's: for each range, `k * nksh_batch + s` over
`k` then `s`.  The caller gathers the batch's pseudo-density through it into
a second buffer the size of the first — `nao_pair x batch_size` doubles,
which `_jk_energy_per_atom` sizes at 2 % of free memory — once per batch.
On penicillin V that is 13 gathers of 67 355 x ≤ 774 doubles, well under a
millisecond each.  The Coulomb-only path indexes its auxiliary vector by
`ao_loc[ksh]` and needs none.  A batch of at most 16 shells is one range and
is not permuted.

`FASTDFHESSIP_IP2_TILE=0` is item 1's launch; `FASTDFHESSIP_IP2_KCHUNK` and
`FASTDFHESSIP_IP2_PCHUNK` (`tile`, `upstream` or a count) move the tile, and
`tests/test_dfhessip.py` runs a forced three-shell range (every batch
permuted), one range, and five pairs per block against upstream.

### 2.3 The kernel on eleven systems

`benchmarks/df_ip2_class_probe.py --whole --kchunk 0 --pchunk 0` (upstream's
launch) against `--kchunk 16 --pchunk -1`, the same kernel binary:

| system | atoms | basis | upstream launch | tiles | |
|---|---|---|---|---|---|
| glycine | 10 | SVP | 68.7 ms | 27.9 ms | 2.46x |
| alanine dipeptide | 22 | SVP | 270.5 | 42.9 | **6.31x** |
| FeCl₄⁻ sextet | 5 | SVP | 251.6 | 175.0 | 1.44x |
| Ni(II) dithiolene⁻ | 21 | SVP | 1275.6 | 274.0 | 4.66x |
| Pt(II) amine triplet, ECP | 23 | SVP | 621.8 | 272.0 | 2.29x |
| penicillin V | 42 | SVP | 1319.5 | 204.7 | **6.45x** |
| Cu(II) macrocycle²⁺ quartet | 60 | SVP | 2518.1 | 518.3 | 4.86x |
| taxol | 113 | SVP | 3695.1 | 1667.4 | 2.22x |
| Co(II) sextet | 113 | SVP | 4499.4 | 1885.4 | 2.39x |
| penicillin V | 42 | TZVPP | 3055.2 | 714.8 | 4.27x |
| Ni(II) dithiolene⁻ | 21 | TZVPP | 2832.0 | 668.0 | 4.24x |

(The open shells run the kernel with the same arguments as their closed
counterparts here — the probe times the kernel, not the spin — and
`_dfhessip_u._ejk_ip2` launches it through the same `Tiles`.)  The gain is
largest where the upstream launch was fewest blocks per unit of work —
20–60 atoms, organic or not — and smallest in two places: **at 113 atoms**,
where there are enough pairs that upstream's launch was already several
waves deep and what is left is the kernel's own arithmetic (§6), and on the
**small metal complexes** (FeCl₄⁻, the Pt complex), whose time is a few
shells of `l = 4–6` on the metal whose blocks are long whatever the tile.
Both are where W31 is.

---

## 3. Item 16: the derivative fills

The ip2 term's second pass, `h1`'s derivative pass and the Coulomb-only
term evaluate `(10|0)` and `(00|1)` with `fill_int3c2e_ip1` and
`fill_int3c2e_ipaux` through `_ip1_evaluator` — the same launch: item 1's
pair count, the whole batch, a grid one block high.  These kernels are built
differently inside: a block walks its `npairs x nksh` triples as one flat
list, `nst` at a time, so a block's work is balanced within it and only its
*size* is wrong.  The pair count is therefore the only thing changed —
`ceil(FILL_TILE x nst[lk, li, lj] / nksh)` pairs per block, for
`FILL_TILE = 2` passes of the block's threads — and the auxiliary range stays
the batch, so the output's layout, which every caller reads, does not change
and nothing is permuted.

Measured in the ip2 call's stage table on penicillin V, repeated in one
process (`FASTDFHESSIP_TIME=1`):

| | W24 | item 16 |
|---|---|---|
| ip2, second pass: `D3 integrals` | 573.3, 573.3, 573.4 ms | 134.8, 151.7, 151.5 ms |
| h1: `D integrals` | 582.0, 582.7 ms | 149.4, 134.8 ms |

**3.8x–4.3x** on the fills; the kernels themselves, from an `nsys` trace of
three calls, are 56.6 + 57.2 ms per call after and do not vary between
calls.

### 3.1 The row that was half something else

The first stage tables of this round showed `D3 integrals` at 150 ms in one
call and 680–720 ms in the next, with the same launch geometry printed for
both.  The trace said the kernels were 114 ms in every call.  The stage's
first mark was taken after the second pass's buffers — `j3c_buf`, up to
`3 x metric_size x nao x nocc` doubles, 2.2 GB at 42 atoms, and `work` —
were allocated and zeroed, and that allocation is sometimes 10 ms and
sometimes 540 ms after the previous call's `free_all_blocks`.  It now has
its own row, `D3 buffers`.  The same row was inside W24's "D3, the
derivative integrals" figures (585–620 ms at 42 atoms); how much of each is
not recoverable from those tables.

---

## 4. The functions

`benchmarks/df_hess_terms_probe.py --terms ip2,h1 --variants stock,ours
--env 'w24:…,tile:FASTDFHESSIP_FILL_TILE=0'`, one process per system, the
four variants round robin, min of two (taxol: one); "item 15" is the tile
launch of the ip2 kernel with the fills as W24 left them:

| system | ip2: stock | W24 | item 15 | **15 + 16** | vs W24 | h1: W24 | **16** | vs W24 |
|---|---|---|---|---|---|---|---|---|
| glycine, 10 atoms | 246.9 ms | 199.9 | 138.1 | 153.3 | 1.30x | 175.4 | 178.5 | 0.98x |
| FeCl₄⁻ sextet, 5, UKS | 528.5 | 474.7 | 393.8 | 395.0 | 1.20x | 203.0 | 226.1 | 0.90x ‡ |
| alanine dipeptide, 22 | 886.5 | 743.9 | 508.7 | 451.2 | 1.65x | 297.9 | 281.2 | 1.06x |
| Ni(II) dithiolene²⁻, 21 | 3022 | 2467 | 1504 | **1216** | **2.03x** | 1078 | 831 | 1.30x |
| Ni(II) dithiolene⁻, 21, UKS | 3895 | 3096 | 2231 | 1809 | 1.71x | 1308 | 1041 | 1.26x |
| Pt(II) amine, 23, ECP | 1415 | 1140 | 785 | 738 | 1.55x | 539 | 479 | 1.13x |
| penicillin V, 42 | 4404 | 2855 | 2308 | **1829** | 1.56x | 1499 | **1047** | **1.43x** |
| penicillin V, 42, DF-PBE | 1374 | 1126 | 1019 | **567** | **1.99x** | 1271 | 858 | 1.48x |
| Cu(II) macrocycle²⁺, 60, UKS | 10980 | 5949 | 3944 | 3448 | 1.73x | 2953 | 2450 | 1.21x |
| penicillin V, 42, TZVPP | 12502 | 6750 | 4274 | 3811 | 1.77x | 4771 | 4294 | 1.11x |
| taxol, 113 | 63216 | 25170 | 22860 | 21927 | 1.15x | 21917 | 20794 | 1.05x |

(DF-PBE's `ip2` is the Coulomb-only term, whose kernel takes the auxiliary
vector and whose fills are the whole of its integral work, so it gains the
most from item 16.)  Against GPU4PySCF the ip2 term is now **1.3x–3.3x** at
every size — 2.4x at 42 atoms, 3.2x on the Cu complex and 2.9x at 113 atoms
— and h1 1.1x–2.2x.

The two ‡ rows are calls of 150–230 ms at 5–10 atoms, where the same code
measured in two processes moves by more than the difference: the h1
variants "W24" and "item 15" are the *same* code for h1 and differ by 6 %
on FeCl₄⁻, and a round-robin sweep of item 16's pass count in one process
(below) puts `FILL_TILE=0` and the default within 3 % of each other on both
molecules, in either order.  Item 16 neither wins nor loses measurably
there; it is left on everywhere rather than given a threshold nothing can
place.

`FILL_TILE`, round robin in one process, ms (ip2 / h1):

| passes per block | glycine | FeCl₄⁻ | alanine | Pt(II) | penicillin V | penicillin V, PBE |
|---|---|---|---|---|---|---|
| 0 (W24) | 144 / 189 | 411 / 201 | 297 / 289 | 843 / 564 | 2295 / 1510 | 1016 / 1296 |
| 1 | 140 / 183 | 396 / 206 | 237 / 282 | 733 / 495 | 1838 / 1094 | 569 / 874 |
| **2** | 142 / 182 | 399 / 203 | 240 / 277 | 721 / 476 | 1847 / 1072 | 573 / 871 |
| 4 | 148 / 180 | 397 / 206 | 242 / 287 | 721 / 482 | 1879 / 1070 | 570 / 883 |
| 8 | 144 / 185 | 397 / 200 | 254 / 281 | 735 / 484 | 1891 / 1090 | 605 / 921 |

Flat from one to four passes on every molecule; two is the default.  The
alanine row of this table and of the one above differ by 1.9x for the same
code (240 against 451 ms): at these sizes a process's first-touch
allocations (§3.1) can dominate a call, which is why every comparison here
is within one process.

### 4.2 Whole Hessians

RESULTS_HESS_TABLE

---

## 5. Correctness

`tests/test_dfhessip.py` and `tests/test_dfuhessip.py`, element-wise against
GPU4PySCF's own function objects in one process, relative to the largest
element, with new switch columns for items 15 and 16:

| | worst |
|---|---|
| ip2 and h1, ethanol, B3LYP / HF / `k = 0` factors, default | 3.3e-13 / 3.3e-16 |
| the same with `IP2_KCHUNK=3` (every batch cut into three-shell ranges and permuted), `IP2_KCHUNK=0`, `IP2_PCHUNK=5`, `IP2_TILE=0` | 3.7e-13 / 3.3e-16 |
| `FILL_TILE=0`, `FILL_TILE=0.05` (one pair per block) | 5.4e-13 / 4.5e-16 |
| the long-range operator (`omega = 0.3`), water | 5.4e-14 (`KCHUNK=3`), 1.8e-15 otherwise |
| the unrestricted twins, ethanol⁺, every switch | RESULTS_UTEST |

The ip2 term's spread is the kernel's atomics, which are unordered with any
launch: upstream against itself is 3e-14–4e-13 on these molecules
(`docs/README_fastdfhessip2.md` §7), and so is every column here.

---

## 6. What bounds the kernel now

`ncu` still cannot collect counters on these nodes ("a driver resource was
unavailable … DCGM"), so the profile is an ablation: GPU4PySCF's
`ejk_int3c2e_ip2.cu`, built here with `nvcc -O3 -arch=sm_80` (254 registers,
no spills — the same as the shipped library, which it matches to 0.98x and
1e-14), with each of its four parts compiled out in turn, launched as tiles.
The contraction's removal drops the kernel to 96 registers, so each variant
was also run with 90 kB of dynamic shared memory per block to hold it at one
block per SM; the times did not move, so occupancy is not what the ablation
measures.  ms:

| variant | penicillin V | Pt(II) complex | taxol |
|---|---|---|---|
| the kernel | 208.6 | 278.2 | 1694.0 |
| Rys roots replaced by constants | 203.3 | 274.5 | 1630.1 |
| the `gout` contraction removed | 79.2 | 47.9 | 798.5 |
| … and the `g` recurrences | 53.4 | 24.3 | 605.0 |
| … and the roots | 50.2 | 23.7 | 582.0 |
| the atomics removed (alone) | 188.4 | 274.2 | 1377.7 |

So, of the kernel:

| | penicillin V | Pt(II) | taxol |
|---|---|---|---|
| Rys roots (the Boys-function part) | 1–3 % | 1 % | 4 % |
| the `gout` contraction over shared-memory `g` | **62 %** | **83 %** | **53 %** |
| the `g` recurrences (`BUILD_3C_GXYZ`) | 12 % | 9 % | 11 % |
| per-triple atomics into `ejk` | 9 % | 0 % | **18 %** |
| the rest: primitive loops, `exp`, barriers | ~14 % | ~8 % | ~15 % |

The entry's falsification criterion asked whether the high classes are
"already at the card's rate for the arithmetic a Rys second derivative
needs".  They are not near it: the part that dominates is not the
arithmetic of the roots but the contraction's reads of `g` from shared
memory — per Cartesian component and root, eighteen `gx` loads, nine
`_c_cartesian_lexical_xyz` constant loads that diverge whenever
`gout_stride` splits a triple across a warp, a float-reciprocal decode of
the component index and up to twelve data-dependent branches — at eight
warps per SM.  That is the part W19's generator removed from its kernels
(literal indices, no table, no decode), and it is left as **W31** rather
than done here: this kernel's `g` arrays do not fit a thread's registers at
the classes that dominate (`(dd|g)` is 240 doubles a root), so it is a
different generator from W19's, not an extension of it.  The atomics at
113 atoms are the other half of W31: 24 per triple from every `gout` thread,
into an `ejk` of `natm² x 9` doubles.

---

## 7. Tools

```bash
python benchmarks/df_ip2_class_probe.py --cases penicillinV,tmc_ni_dithiolene:-1:1 --whole --no-classes \
    [--kchunk 16 --pchunk -1]                        # §1, §2.3: the launch, upstream's or the tile
python benchmarks/df_ip2_class_probe.py --cases ... --kchunk 16 --pchunk -1 --summary   # per (li, lj, lk)
python benchmarks/df_ip2_class_probe.py --cases ... --whole --ours --lib path/to/lib.so [--shm 90000]
                                                     # §6: another build of the same entry point
python benchmarks/df_hess_terms_probe.py --xyz ... --terms ip2,h1 --variants stock,ours \
    --env 'w24:FASTDFHESSIP_IP2_TILE=0;FASTDFHESSIP_FILL_TILE=0,tile:FASTDFHESSIP_FILL_TILE=0'
FASTDFHESSIP_TIME=1 python benchmarks/df_hess_terms_probe.py ...   # §3's stage rows
python tests/test_dfhessip.py [--no-hessian]; python tests/test_dfuhessip.py [--no-hessian]
```

Switches added by this round (read at call time): `FASTDFHESSIP_IP2_TILE`,
`FASTDFHESSIP_IP2_KCHUNK` (16), `FASTDFHESSIP_IP2_PCHUNK` (`tile`),
`FASTDFHESSIP_FILL_TILE` (2).

**The node.**  Every number here is from one interactive allocation whose
other three GPUs ran other jobs throughout.  The kernel tables are CUDA-event
times of kernels alone and were stable to 1 % run to run; the function and
Hessian tables are wall clock with host work in them and are what §5.2 of
the work list would call second-best — the before and after of each row
were run in the same process or in alternating processes on the same GPU,
so both halves carry the same neighbours.
