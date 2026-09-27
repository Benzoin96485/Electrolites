# The ip2 kernel from the inside: 21 sums, a register path, and a block's atomics

`docs/DF_WORK_LIST.md` **W31**, found by W29.  W29 launched GPU4PySCF's
`ejk_int3c2e_ip2` as its own tile without changing a line of it, and an
ablation at equal launch geometry then put the kernel's time in the
contraction of the shared-memory `g` arrays (53–83 %) and, at 113 atoms, its
per-triple atomics (18 %).  The entry asked for the contraction generated per
`(li, lj, lk)` with literal indices and for the atomics replaced by a
block-level reduction; success was another 2x at 113 atoms and on the Pt/Fe
complexes, and the falsification a literal-index contraction that is not
faster than the ablation's share allows.

**The answer splits along a register budget.**  With `g` in shared memory,
literal indices do not pay: the fully unrolled contraction spills at 255
registers and is at best 1.16x the generic loop.  Where a triple's whole `g`
fits in one thread's registers — the light classes, `(ss|`–`(ps|`–`(pp|`
against low `lk`, which are most of the triples of an organic molecule — the
recurrences and the contraction compiled per class with every index a literal
are **3.7x–4.7x** the tile-launched upstream kernel class by class.  Together
with an algebraic cut of the contraction (30 of upstream's 51 sums are linear
combinations of the other 21) and a block-level reduction of the atomics, the
kernel is **1.39x–1.95x** W29's on eleven systems: **1.95x at 113 atoms**
(taxol), 1.83x on the 113-atom Co(II) complex, 1.78x on penicillin V — and
**1.53x** on the Pt(II), Ni(II) and FeCl₄⁻ complexes, where the success bar
is missed.  What is left there is the heavy classes' shared-memory
contraction (58–76 % of the kernel after this round), which no change in this
round's family moved, and which is bounded by eight warps per SM (§4).

---

## 0. Summary

* **Item 17: `ejk_int3c2e_ip2` rewritten** (`electrolites/kernels/cuda/myip2/`,
  built by `_nvcc.ip2_library`, chosen by `_ip2launch.kernel`): the same entry
  point, arguments and shared-memory layout, so item 15's launch drives it
  unchanged.  Four changes (§2):
  1. the contraction accumulates 21 sums per triple instead of 51 (the kk,
     ik and jk blocks follow from translational invariance, once per triple);
  2. the light classes — `g` at most 24 doubles per direction, at most 36
     components, upstream's `gout_stride` 1 — run a triple per thread with
     `g` in registers, recurrences and contraction per class, all indices
     literals, no barrier inside a triple, as a 256-thread kernel of their own;
  3. the other classes keep the shared-memory `g`, with the 21-sum
     contraction, on **512** threads (two per triple where upstream has one,
     the same triples and shared memory per block, 16 warps per SM at 128
     registers);
  4. a block parks its triples' sums in shared memory and adds them up — pair
     terms over its 16 auxiliary shells, kk terms over its pairs, ik/jk terms
     over runs of pairs on one atom — before one `atomicAdd` per target.
* **The kernel** (§3.1): **1.39x–1.95x** W29's; cumulatively 2.2x–11.5x
  GPU4PySCF's own launch of its own kernel.
* **The ip2 term** (§3.2): 1.01x–1.24x W24+W29's (1.24x in DF-PBE, 1.15x on
  Pt(II), 1.17x on FeCl₄⁻, 1.08x in def2-TZVPP, 1.04x at 113 atoms); against
  GPU4PySCF 1.6x–3.6x.  The kernel is 7 % of the call at 42 atoms after this
  round; the largest stage is now a **buffer allocation**, 557–637 ms of a
  1.84 s call (§3.3).
* **Whole Hessians** (§3.4): 1.00x–1.05x, traces unchanged to the kernel's
  own spread.
* **Rejected** (§5): the shared-memory contraction unrolled with literals
  (spills, ≤1.16x, and 0.76x–0.92x once the block reduction also wanted
  registers); a branch-free runtime contraction (0.93x); auxiliary-fastest
  block order (0.81x at 113 atoms); 1024 threads (0.34x).

Everything is one A100-SXM4-80GB, GPU4PySCF 1.8.1, def2-SVP and its jkfit
set unless a row says otherwise, `patch_all()` on both sides.  "W29" is this
tree with `FASTDFHESSIP_IP2_KERNEL=upstream`: item 15's tile launch of
GPU4PySCF's kernel, the previous commit's code path.  Measured on a shared
interactive node (§7).

---

## 1. Where the time was, measured rather than ablated

`ncu` works on Perlmutter once the allocation asks for `--perf=generic` (it
stops DCGM's sampling); without it every counter fails with "a driver
resource was unavailable", which is why W29 had to ablate.  On penicillin V's
largest launch (`l = 2`, 129 auxiliary shells, 10 431 blocks), GPU4PySCF's
kernel under W29's launch:

| | upstream | 21 sums (first step, §2.1) |
|---|---|---|
| duration | 37.4 ms | 29.2 ms |
| instructions issued | 6.71 G | 5.61 G |
| FP64 pipe busy | 54 % | 40 % |
| IPC | 1.39 | 1.48 |
| warps per SM (theoretical / achieved) | 8 / 7.86 | 8 / 7.82 |
| dominant stall | `wait` 1.82, `math_pipe_throttle` 0.74, `short_scoreboard` 0.62 cycles per issue | `wait` 1.61, `short_scoreboard` 0.80 |

Two limits hold the block to one per SM at once: 254 registers x 256 threads
is the register file, and the scheme sizes every class's triples per block to
fill 160 KB of shared memory (the launch's dynamic allocation is 155 KB).
Eight warps cannot cover the FP64 pipe's dependency latency, and the
dominant stall is exactly that (`wait`: a fixed-latency dependency).  The hot
loop's instruction mix (SASS, per component and root) is 54 DFMA + 40 DMUL
against 52 IMAD, 29 I2F, 26 LEA, 24 IADD3, 13 ISETP and 9 constant loads —
more integer than floating point, the index decode and the `i±1`/`j±1`
address arithmetic W29 named.

So the contraction had three costs to remove: arithmetic it did not need,
integer work that literal indices remove, and latency that only more
independent work per SM hides.

---

## 2. Item 17

### 2.1 Twenty-one sums

Per Cartesian component upstream forms the value `I` and the derivative
factors `fi`, `fj`, `fij`, `fii`, `fjj` in each direction and adds their
products into 51 accumulators: the pair–pair block `v1` (9), the i–i and j–j
blocks (6 + 6), and per triple the k–k (6), i–k (9) and j–k (9) blocks.  The
last three are not independent: the auxiliary centre's derivative is minus
the two on the pair, so per triple

    kk = vi + v1 + v1ᵀ + vj,   ik[d][e] = -(v1[d][e] + vi[d][e]),
    jk[d][e] = -(v1[e][d] + vj[d][e])

(`k_term`).  A triple accumulates 21 sums, 33 floating-point operations per
component instead of about 100, and forms the other 24 once before the
atomics.  Alone, on 256 threads: penicillin V 204.2 → 167.2 ms (1.22x), the
Pt(II) complex 272.8 → 209.0 ms (1.30x), taxol 1668 → 1376 ms (1.21x).

### 2.2 The light classes in registers

W31's entry assumed W19's register-resident design would not carry over,
because `(dd|g)` needs 240 doubles of `g` per root.  That is true of the
heavy classes and false of the ones with the most triples.  A class with
`(li+2)(lj+2)(lk+1) ≤ 24` and at most 36 components whose scheme
`gout_stride` is 1 — `(ss|s–g)`, `(ps|s–d)`, `(pp|s–p)`, `(ds|s–p)`,
`(dp|s)`, `(dd|s)`, `(fs|s–p)`, `(fp|s)`, `(gs|s)`, to which upstream
already gives one thread per triple — runs `triple_reg<li, lj, lk>`:
the Rys roots into the thread's own column of
shared memory, then per root the three directions' recurrences and the
contraction on a register array, every index and offset a literal and every
`if (ix > 0)` resolved at compile time, the triple's pseudo-density read
once into registers (≤ 32 components).  There is no barrier inside a triple.

A single kernel holding both paths is budgeted for the worse one: at 255
registers the heavy path had no room to grow, and at 128 the register path
spilled into its own loop.  The two are therefore **two kernels** on the same
grid (`MODE` 1 and 2), each returning at once from a block of the other
kind; the light kernel is 256 threads, the heavy one 512 (§2.3).  Per class,
the register path is 3.7x–4.7x the tile-launched upstream kernel on the Pt
complex's light classes (`(pp|p)` 4.73x, `(ps|p)` 3.95x, `(pp|s)` 4.05x,
`df_ip2_class_probe.py --ours --summary`), and the kernel over all classes
1.42x what it is without the path (build flag `IP2_REG=0`, §7:
taxol 1207 → 849 ms, penicillin V 144 → 115 ms; the Pt(II) complex, whose
time is in heavy classes, 176 → 178 ms).

The thresholds were swept (`IP2_REG_GMAX` 16/18/24/27/32, `IP2_REG_NFMAX`
32/36/54, `IP2_DMV_MAX` 0/12/32): 16 is 13 % slower on penicillin V, and
from 18 up the three molecules move by ≤ 2 % while spills grow; 24/36/32 is
the default.

### 2.3 The heavy classes on 512 threads

The heavy path keeps upstream's shared-memory `g` and the 21-sum contraction.
Its launch reads upstream's `gout_stride` table and doubles it, so a block
holds the same triples in the same shared memory with twice the threads: two
per triple, sixteen warps per SM at 128 registers (380 bytes of spills,
outside the component loop).  Against 256 threads: Pt(II) 195.1 → 177.8 ms,
FeCl₄⁻ 126.3 → 114.2, penicillin V 118.4 → 114.7, taxol 838 → 848 (the one
row that moves the other way, inside 1.2 %).  1024 threads (64 registers)
spills throughout and is 0.34x.

### 2.4 The atomics

After a triple, its sums go to shared memory (in the `g` area, which the
triple is done with) and the block adds them: the 21 pair terms of each pair
over the block's auxiliary shells, the 6 kk terms of each auxiliary shell
over the block's pairs, and the ik/jk terms over runs of consecutive pairs on
the same atom (`block_flush`).  A 256-triple light block issues about 700
atomics instead of 11 520.  Taxol 1195 → 849 ms (1.41x, build flag
`IP2_REDUCE=0` is per-triple atomics), penicillin V 138 → 115.  What remains
of the reduction plus the atomics, by removing both: 19 % at 113 atoms
(849 → 686 ms), 11 % at 42, 2 % on Pt(II).

### 2.5 The launch and the switch

`_ip2launch.kernel()` returns `libmyip2`'s entry point unless
`FASTDFHESSIP_IP2_KERNEL=upstream` or `nvcc` is missing; `Tiles._run` calls
it, so item 15's launches (the pseudo-density path of both spins and the
Coulomb-only term) get it and item 1's (`FASTDFHESSIP_IP2_TILE=0`) keep
GPU4PySCF's.  The light kernel is launched with at least 88 KB of dynamic
shared memory whatever the scheme asked (its reduction buffer and roots at
256 triples).  The range-separated operator (`omega < 0`) takes the heavy
path for every class.  `ELECTROLITES_LIBMYIP2` points at a prebuilt library;
the build is 35 s and cached by a hash of the sources.

---

## 3. Measurements

### 3.1 The kernel on eleven systems

`benchmarks/df_ip2_class_probe.py --whole --no-classes --kchunk 16 --pchunk -1
--ours --reps 3`, the same tile launch for both columns, CUDA events, min of
three:

| system | atoms | basis | W29 (upstream kernel) | W31 | | rel. err. |
|---|---|---|---|---|---|---|
| glycine | 10 | SVP | 27.7 ms | 19.9 ms | 1.39x | 1.5e-14 |
| alanine dipeptide | 22 | SVP | 42.6 | 27.4 | 1.56x | 2.3e-14 |
| FeCl₄⁻ sextet | 5 | SVP | 174.3 | 114.1 | 1.53x | 8.5e-14 |
| Ni(II) dithiolene⁻ | 21 | SVP | 273.5 | 177.3 | 1.54x | 9.1e-14 |
| Pt(II) amine triplet, ECP | 23 | SVP | 271.2 | 177.2 | 1.53x | 8.0e-14 |
| penicillin V | 42 | SVP | 203.4 | 114.5 | **1.78x** | 1.1e-13 |
| Cu(II) macrocycle²⁺ quartet | 60 | SVP | 516.4 | 314.6 | 1.64x | 2.0e-13 |
| taxol | 113 | SVP | 1670.2 | 857.6 | **1.95x** | 9.2e-14 |
| Co(II) sextet | 113 | SVP | 1889.3 | 1031.4 | **1.83x** | 1.3e-13 |
| penicillin V | 42 | TZVPP | 715.9 | 423.9 | 1.69x | 4.2e-14 |
| Ni(II) dithiolene⁻ | 21 | TZVPP | 665.3 | 434.2 | 1.53x | 2.5e-13 |

With W29's table (`docs/README_fastdfhessip3.md` §2.3, its "upstream launch"
column) this is 2.2x (FeCl₄⁻: 251.6 → 114.1 ms) to 11.5x (penicillin V:
1319.5 → 114.5 ms) against GPU4PySCF's own launch of its own kernel, 4.3x at
113 atoms.

**The size and class audit.**  The gain is set by how much of a molecule's
work is in light classes, which is a property of the basis on each atom, not
of the molecule's size: organic molecules in def2-SVP are mostly `s`/`p`
shells against a jkfit set whose `l ≤ 2` shells dominate, so their triples are
mostly light (1.56x–1.95x, rising with size because the atomics' share rises
with size and §2.4 removes most of it); the transition-metal complexes' time
is in the metal's `d`/`f` shells against `l = 3–6` auxiliary shells, all
heavy, so they get §2.1 and §2.3 only (1.53x on every one of them, 5 to 23
atoms, Fe, Ni, Pt, open and closed shell); def2-TZVPP moves organic work into
heavy classes (1.69x against 1.78x for the same molecule).  The 60- and
113-atom complexes sit between, because most of their atoms are organic.
Nothing in item 17 depends on the system's size except through that mix: the
thresholds are per class, and the block reduction's buffer is checked against
the launch's shared memory at run time.

### 3.2 The ip2 term

`benchmarks/df_hess_terms_probe.py --patch all --terms ip2 --variants
stock,ours --env w29:FASTDFHESSIP_IP2_KERNEL=upstream --reps 2`, one process
per system, round robin, min of two (taxol one):

| system | stock | W29 | **W31** | vs W29 | vs stock |
|---|---|---|---|---|---|
| glycine, 10 atoms | 212.7 ms | 142.5 | **130.9** | 1.09x | 1.63x |
| FeCl₄⁻ sextet, 5, UKS | 528.1 | 386.3 | **328.8** | 1.17x | 1.61x |
| alanine dipeptide, 22 | 683.5 | 238.3 | **227.1** | 1.05x | 3.01x |
| Ni(II) dithiolene²⁻, 21 | 3002 | 1152 | **1075** | 1.07x | 2.79x |
| Ni(II) dithiolene⁻, 21, UKS | 3724 | 1633 | **1605** | 1.02x | 2.32x |
| Pt(II) amine, 23, ECP | 1421 | 705 | **613** | 1.15x | 2.32x |
| penicillin V, 42 | 4429 | 1859 | **1835** | 1.01x | 2.41x |
| penicillin V, 42, DF-PBE | 1374 | 584 | **472** | **1.24x** | 2.91x |
| Cu(II) macrocycle²⁺, 60, UKS | 11019 | 3382 | **3222** | 1.05x | 3.42x |
| penicillin V, 42, TZVPP | 12429 | 3785 | **3502** | 1.08x | **3.55x** |
| taxol, 113 | 63594 | 21988 | **21071** | 1.04x | 3.02x |

Every row agrees with GPU4PySCF's function to the same relative error as
W29's column (3e-13 – 1e-10, the largest on the open-shell Cu complex, and
the same in both columns).  DF-PBE's Coulomb-only term, whose integral work
is all kernel and fills, gains most.

### 3.3 Why the kernel's 1.8x is 1.01x of the call

`FASTDFHESSIP_TIME=1` on penicillin V, two calls each:

| stage | W29 | W31 |
|---|---|---|
| C ip2 kernel | 213.8, 213.7 ms | 133.8, 124.1 ms |
| **D3 buffers** | **557.1, 562.3 ms** | **609.6, 637.4 ms** |
| D3 integrals | 114.6, 114.5 | 114.5, 114.5 |
| D5 per-atom | 177.2, 177.2 | 177.3, 177.3 |
| total | 1838, 1844 | 1809, 1837 |

The kernel stage moves by 80–90 ms, exactly as §3.1 says; the call moves by
7–30 ms, because the stage W29 split out of "D3 integrals" — two `cp.empty`
calls of 2.2 GB and a few GB right after the call's `free_all_blocks`, and
two zero-fills — is **a third of the call** and moves by ±50 ms between calls.
It is not this round's to fix; it is the largest stage of the ip2 term at 42
atoms, and the next item on this path.

### 3.4 Whole Hessians

`benchmarks/df_hess_anatomy.py --patch all --rounds 2`, W29
(`FASTDFHESSIP_IP2_KERNEL=upstream`) and W31 in alternating processes on one
GPU, the pair twice; round 1 of each:

| system | basis | W29 | W31 | | ip2 W29 → W31 | traces |
|---|---|---|---|---|---|---|
| penicillin V, 42 | SVP | 13.10, 13.30 s | **12.60, 12.60 s** | 1.05x | 1.35 → 1.21 s | 41.78582245(6–7) both |
| Pt(II) amine, 23, ECP | SVP | 5.90, 5.86 | 5.85, 5.80 | 1.01x | 0.70–0.77 → 0.63–0.67 | 20.80278306 ± 5e-10 |
| Ni(II) dithiolene²⁻, 21 | SVP | 6.66, 6.81 | 6.61, 6.98 | 1.00x | 1.13–1.15 → 1.04–1.14 | 21.959010(40) ± 2e-9 |
| penicillin V, DF-PBE | SVP | 6.29, 6.37 | 6.18, 6.47 | 1.00x | 0.55–0.57 → 0.47 | 40.764712845(4) both |
| Ni(II) dithiolene⁻, UKS | SVP | 13.72, 13.89 | 13.69, 13.88 | 1.00x | | 22.03359042(4) ± 2e-9 |
| penicillin V, 42 | TZVPP | 39.99, 39.96 | **39.64, 39.56** | 1.01x | 3.77 → 3.42–3.54 | 39.777317038 ± 6e-10 |

The ip2 term is 5–17 % of a Hessian after W29, and the kernel a fraction of
that; the CPHF is what a Hessian is (`docs/README_fastdfhessip3.md` §4.2).
The Ni²⁻ and DF-PBE rows move by less than their pairs disagree.

---

## 4. What bounds it now

The final design with each part removed (build flags, the same probe, ms):

| variant | penicillin V | Pt(II) | taxol |
|---|---|---|---|
| item 17 | 114.7 | 177.8 | 848.5 |
| heavy path's contraction removed | 47.6 | 42.1 | 427.2 |
| reduction and atomics removed | 102.4 | 174.5 | 686.4 |
| per-triple atomics (`IP2_REDUCE=0`) | 137.6 | 181.9 | 1194.8 |
| no register path (`IP2_REG=0`) | 144.2 | 175.9 | 1207.2 |

The heavy classes' shared-memory contraction is **58 % of the kernel on
penicillin V, 76 % on the Pt(II) complex and 50 % at 113 atoms** — the
metals are where it has not moved, and the reason is structural: a heavy
block fills 155 KB of shared memory with `g`, so an SM runs one block, and
the only way this round found to get more than eight warps onto it was more
threads per triple (§2.3).  Precomputing the derivative factors per root
into shared memory — which would cut the per-component work by about 3x — needs
1.5x–3.4x the storage of `g` and so fewer triples per block, which moves the
cost into the recurrences' serial stages; and the literal contraction needs
the registers the heavy path does not have (§5).  A heavy-class kernel that
gets its occupancy from somewhere else (triples split across a warp with
`g` exchanged by shuffles, or the factor arrays in a second pass) is the
work left, and it is a new design rather than a generator.

The reduction plus atomics remain 19 % at 113 atoms, most of it the
block's barrier and the serial sums in `block_flush`, not the atomics.

---

## 5. Measured and rejected

Same probe, penicillin V / Pt(II) / taxol where all three were run, ms:

| variant | pen. V | Pt(II) | taxol | why it lost |
|---|---|---|---|---|
| §2.1 alone, generic loop | 167.2 | 209.0 | 1376 | (the baseline for the next rows) |
| + shared-memory contraction fully unrolled, literal indices, ≤ 60 / ≤ 200 components | 159.7 / 150.6 | 225.2 / 205.3 | | 5 KB / 55 KB of spills at 255 registers |
| + the same with `nst` a literal and a compiler barrier every 2 / 4 components | 148.6 / 144.3 | 206.2 / 202.1 | 1395 (4) | 24 KB / 12 KB spills; ≤ 1.16x |
| + the same with §2.4's reduction also in the kernel | 222.2 | 218.6 | 2201 | the reduction's registers pushed the spills into the loop |
| branch-free runtime factors (read a neighbour at offset 0 and multiply by 0) | 123.8 (from 114.5) | 195.3 (177.2) | 899 (848) | the branches are warp-uniform and skip loads |
| auxiliary ranges fastest in the grid (so concurrent blocks add into different atoms) | 159.6 | 192.6 | 1716 (from 1376) | loses the pair-major reuse of the pseudo-density; contention was not the cost |
| 1024 threads, 64 registers | 597.6 | 393.8 | | spills throughout |
| one kernel for both paths, 256 threads | 115.4 | 183.0 | 878 | the heavy path at 8 warps (§2.3) |

---

## 6. Correctness

`tests/test_dfhessip.py` and `tests/test_dfuhessip.py`, element-wise against
GPU4PySCF's function objects, with two new columns (`IP2_KERNEL=upstream`,
and W31's kernel under a three-shell auxiliary range with five pairs per
block):

| | worst (relative to the largest element) |
|---|---|
| ip2, ethanol, B3LYP / HF / `k = 0` factors, every switch, def2-SVP and def2-TZVPP | 4.0e-13 |
| the long-range operator (`omega = 0.3`), water — the heavy path for every class | 5.6e-14 |
| h1 (not touched) | 1e-15 |
| the unrestricted twin, ethanol⁺, every switch | 2.0e-13 |
| whole Hessians patched against unpatched | **7.8e-12** (RKS), **3.8e-12 Eh/bohr²** (UKS) |

The kernel against GPU4PySCF's, whole launches: ≤ 2.5e-13 on the eleven
systems of §3.1.  Class by class (`df_ip2_class_probe.py --ours`), the worst
is 1.2e-10 relative to that class's own largest element, on a Pt(II) class of
38 triples whose result is a near-cancelling sum; the register path and the
heavy path give the same figure on it (the build without the register path
reads 1.2e-10 too), so it is the order of summation of the derived kk/ik/jk
terms, not a path.

---

## 7. Tools

```bash
python benchmarks/df_ip2_class_probe.py --cases ... --whole --no-classes \
    --kchunk 16 --pchunk -1 --ours [--basis def2-tzvpp]     # §3.1
python benchmarks/df_ip2_class_probe.py --cases ... --kchunk 16 --pchunk -1 \
    --ours --summary                                        # per class, §2.2
python benchmarks/df_ip2_class_probe.py --cases ... --whole --no-classes \
    --kchunk 16 --pchunk -1 --ours --lib path/to/libmyip2.so  # a variant build
python benchmarks/df_hess_terms_probe.py --patch all --xyz ... --terms ip2 \
    --variants stock,ours --env w29:FASTDFHESSIP_IP2_KERNEL=upstream   # §3.2
FASTDFHESSIP_TIME=1 FASTDFHESSIP_IP2_KERNEL=w31|upstream \
    python benchmarks/df_hess_terms_probe.py ...            # §3.3
python tests/test_dfhessip.py [--basis def2-tzvpp]; python tests/test_dfuhessip.py
```

A variant build is `nvcc -O3 -std=c++17 -arch=sm_80 -rdc=true -shared
-I electrolites/kernels/cuda/gvhf-rys181{,/gvhf-rys}
electrolites/kernels/cuda/myip2/ejk_int3c2e_ip2.cu` plus the two Rys table
sources `_nvcc._IP2_SOURCES` lists, with `-D` flags: `IP2_REG=0` (no register
path), `IP2_REDUCE=0` (per-triple atomics), `IP2_SPLIT=0` (one kernel),
`IP2_THREADS=<n>` (the heavy path's block), `IP2_REG_GMAX`, `IP2_REG_NFMAX`,
`IP2_DMV_MAX` (the register path's classes and preload), and the two
ablations `IP2_NOCONTRACT`, `IP2_NOATOMIC` (§4; wrong answers by design).

`ncu` needs `--perf=generic` on the allocation (`sbatch`/`salloc`); `-o`
onto `$SCRATCH`.

**The node.**  §3.1, §4 and §5 are CUDA-event times of kernels alone and
repeat to 1 %; §3.2–§3.4 are wall clock on an interactive node whose other
GPUs ran other work, before and after in one process (§3.2) or in
alternating processes on one GPU (§3.4), as W29's were.
