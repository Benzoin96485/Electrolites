# The bra primitive-pair bound, and the JIT a first run pays

`docs/OPENSHELL_TMC.md` §4 ranks eight things an open-shell transition-metal
complex still runs on GPU4PySCF's own code with every module of this repository
loaded.  Five have been taken.  Of the three left, **items 6 and 8 are the two
that are not about spin at all** -- they are basis-set limits, and both fire on
a closed-shell organic molecule in the same basis:

| §4 | entry | what it is |
|---|---|---|
| 6 | `MAX_PRIM_PAIR = 36`, "6-prim x 6-prim is the worst case in Pople sets" | a def2 set forms bra primitive pairs of 49, 56, 64 and 100, and every class block built on one fell through |
| 8 | first-run JIT | a def2 basis reaches f, so the angular-momentum split does not fire and the first run pays the whole NVRTC cost |

This round takes both.  Item 7 -- g shells in `fastj` and `fastejk` -- is left
where it is.

---

## 1. Four corrections to §4, before the measurements

**Item 6 calls this a transition-metal limit.  It is a def2-TZVP limit.**  The
entry's evidence is a 3d metal's innermost s shell, contracted over 8
primitives, and S or Cl over 7.  Both are right, and the second of them has
nothing to do with metals: **penicillin V in def2-TZVP -- a closed-shell
organic molecule, and one this repository has been benchmarking on since the
first round -- forms a 7 x 7 = 49 bra pair on its sulfur**, and every def2-TZVP
molecule with a second-row heteroatom does.  Going further up the basis is
worse and again not about metals: **def2-QZVP contracts S over 10 primitives**,
so the same molecule forms a pair of 100.  The bound was measured on Pople
sets, and what invalidates it is the basis set, not the element.

**Item 6 says raising the bound costs "six `double[MAX_PRIM_PAIR]` arrays per
block".  Six is the count in one kernel of one of the four families.**
`fastk_prologue.cu`'s `(0000)` kernel declares six; `gen_kernels.py`'s
one-thread-per-quartet family declares seven, `gen_khigh.py` seven, and
`gen_ejk.py` **nine**.  The count matters because it is the whole argument for
*not* raising the bound globally -- at 100 pairs that is 7.2 KB of static
shared memory per block on the gradient's kernels, carried by 65 classes of
which 7 can ever see a long bra pair.

**Item 6 says "0.5 % of the class blocks is not 0.5 % of the *work*" and
leaves it there.  It is 15x that, not a fraction of it.**  On
CuCl2/def2-TZVP the 25 class blocks past the bound are 0.50 % of the 4 968 and
**7.3 %** of the time the exchange build spends inside class blocks
(`benchmarks/prim_pair.py --report`).  They are compact core `(s,s)` and
`(p,s)` pairs, so the arithmetic in them is small; what they are not is few.

**Item 8 says "a shared `ELECTROLITES_GENERATED_DIR` and a warm CuPy cache is
the answer for a dataset run, not a change to the split".  The first half is
right and there was no way to act on it.**  `_gen.ensure_all()` fills the
generated-CUDA cache -- 0.8 s of Python -- but nothing filled the *NVRTC* cache,
which is where essentially all of the cost is; the only way to warm it was to
run a calculation and hope it happened to ask for the same module keys.  What
was missing was not an idea, it was a command.  §4c also puts a number on the
entry's "36 s rather than 98 s": measured here, a Pople first run pays **42 s**
of NVRTC and a def2 transition-metal one **186-198 s**, which is not an
overhead on the first run but **86-89 % of it**.

---

## 2. What was added

### A second module, not a wider array: `_ksplit.mpp_plan`

The bra shell pair's primitive data -- `aij`, `1/aij`, `aj/aij`, the product
centre and `ci*cj*exp(-ai*aj*rr/aij)` -- is block-uniform, so every kernel here
computes it once per bra pair into shared memory instead of once per thread per
primitive quartet.  That is the third of the four changes `docs/README_fastk.md`
lists, and it is why the array exists at all.  The array is fixed size, the
size is `MAX_PRIM_PAIR`, and `_usable` declines a bra pair longer than it.

The obvious fix is to make the number bigger.  The reason not to is that
**whatever is compiled with the wider bound carries the wider array**, whether
it can ever see a long bra pair or not: nine `double[MAX_PRIM_PAIR]` arrays per
block in `gen_ejk.py`, seven in `gen_kernels.py` and `gen_khigh.py`, six in
`fastk_prologue.cu`.  At a bound of 100 that is 7.2 KB of static shared memory
on every one of `fastejk`'s 65 classes, of which **7** can ever be launched
with a long bra pair.

So the bound became a compile-time parameter and the driver compiles a *second*
module for the classes that need one.  `_ksplit.mpp_plan` decides, from the
`(l, nprim)` group table alone:

```python
mpp, lmax = _ksplit.mpp_plan(uniq_l_ctr, MAX_PRIM_PAIR)
```

`mpp` is the longest bra pair this basis can form that the default does not
cover and `lmax` the largest angular momentum of a group that takes part in
one; `(36, -1)` means there is nothing to do, which is every Pople set and
every def2 set over first-row elements alone.  Both come from the basis and
the elements and never from how many pairs survive screening, which is the
property the rest of `_ksplit` is built around: two molecules in the same basis
over the same elements produce the same second module, the same source text and
therefore the same CuPy disk-cache entry.

`lmax` is what keeps the second module small, and it is small because of which
groups a long pair can be made of.  The bra pair of a class block is the
quartet's highest-l group -- `_classes` enumerates `(i, j, k, l)` with `j <= i`,
`k <= i`, `l <= k` over an l-major group order -- so the plan's `lmax` is the
highest l that appears on the left of a long pair.  At def2-TZVP the groups
deep enough to make one are s (8 primitives on a 3d metal, 7 on S/Cl/P, 6 on
C/N/O) and p (6 on a 3d metal), so the plan comes out at **`lmax = 1`**: the
second module is the seven classes with `l <= 1` and their range-separated
twins, **14 kernels, 127 KB of `fastk`'s 388 KB and 0.28 MB of `fastejk`'s
6.8 MB**.  At def2-QZVP it comes out at `lmax = 2`, because a 3d metal's d
shell is contracted over 6 primitives and its innermost s over 11.

Three details that are load-bearing rather than incidental:

* **The bound is a preprocessor definition, for the same reason `EJK_NDM` is
  one** (`docs/OPENSHELL_TMC.md` §3): as a kernel argument the array would have
  to be sized for the worst case always, and the common path would pay for the
  rare one in shared memory it cannot be shown not to pay.
* **Only the files whose arrays follow the macro may go into the second
  module.**  `gen_k2_kernels.py`'s output -- which is tracked, and regenerating
  it needs a GPU4PySCF checkout -- writes `double[36]` as a literal.  Compiled
  with `-DMAX_PRIM_PAIR=64` it would write past the array rather than fall
  through, which is an illegal access and not a slow path; `_sources(wide=True)`
  is the list that may, and it is a correctness restriction with a comment
  saying so.  `gen_ejk.py` and `gen_khigh.py` baked the same literal and now
  emit the macro.
* **A class that does not fit still falls through.**  `fastejk._usable`
  re-does the shared-memory arithmetic before using the second module, because
  `gen_ejk.py` sized the shm-style classes' dynamic allocation against a 36-pair
  cache.  At the default cap nothing reaches that test -- the widest shm-style
  class an over-36 bra pair can reach is 44.5 KB and a 100-pair cache is another
  7.2 KB -- but `ELECTROLITES_MAX_PRIM_PAIR` is a knob, and the arithmetic that
  says it is safe belongs in the code rather than in a comment on the knob.

### `electrolites.warmup`

    python -m electrolites.warmup --basis def2-tzvp --elements Cu,Cl,C,H,N,O \
        --spin 2 --jobs 8 --cupy-cache /shared/cupy \
        --generated-dir /shared/electrolites/cuda

It walks the same `_module` functions the drivers call, with the same arguments
they will ask for -- the same `lmax` from `_ksplit.names_within`, the same
`mpp` from `_ksplit.mpp_plan`, the same `EJK_NDM` -- so every compile it does is
an entry the run finds already there rather than a near-miss.  Getting those
keys right by hand is the part that is easy to get wrong: a warmup that
compiled `fastejk` at `EJK_NDM=1` and then ran an open-shell gradient would
have warmed nothing.

The arguments are a property of the basis and the elements, not of the
geometry, so `--elements Cu,Cl,C,H,N,O` builds one atom of each, far apart,
and reads the group table off GPU4PySCF's own `_VHFOpt` -- the same table the
drivers will sort against.  `--xyz` takes a real geometry when one is at hand.

`--jobs N` runs the compiles in N worker *processes*.  They are independent
translation units and CuPy writes its cache with a rename, so two workers that
happen to compile the same thing race harmlessly.  Processes rather than
threads because the point is wall clock and NVRTC is reached through CuPy.

## 3. Correctness

The second module is a different compile of the same arithmetic, so the thing
to check is that the answer does not depend on which module served a class
block.  `tests/test_prim_pair.py` fixes one density per case and calls the
exchange build, the fused build and the gradient's per-atom derivative three
ways in one process: with the second module allowed, with it forbidden
(`--cap 0`, the behaviour before this round), and with GPU4PySCF's own kernels.
The density is fixed after a few SCF cycles rather than converged -- all three
are handed the same one, so convergence decides nothing here, and CuCl2 in
particular is slow to converge for reasons `docs/OPENSHELL_TMC.md` §3 records.

| case | AOs | longest pair | K, bound on/off | K vs GPU4PySCF | JK vs GPU4PySCF | ejk on/off | ejk vs GPU4PySCF |
|---|---:|---:|---:|---:|---:|---:|---:|
| glycine / 6-31G\* | 80 | 36 | 1.8e-16 | 2.0e-14 | 2.0e-14 | 8.1e-16 | 2.0e-14 |
| glycine / def2-SVP | 95 | 36 | 1.8e-16 | 2.0e-14 | 2.0e-14 | 8.0e-16 | 2.6e-14 |
| CuCl2 / def2-SVP | 67 | 36 | 4.6e-17 | 5.0e-15 | 5.0e-15 | 2.4e-16 | 2.2e-14 |
| glycine / def2-TZVP | 185 | 36 | 4.6e-16 | 4.1e-15 | 1.0e-14 | 6.0e-15 | 3.9e-14 |
| penicillin V / def2-TZVP | 858 | **49** | 5.3e-16 | 1.0e-14 | 3.2e-14 | 2.3e-14 | 2.9e-13 |
| CuCl2 / def2-TZVP | 119 | **64** | 3.6e-16 | 3.9e-15 | 3.9e-15 | 6.1e-16 | 3.4e-14 |
| FeCl4- / def2-TZVP | 193 | **64** | 1.5e-16 | 5.9e-15 | 6.1e-15 | 2.7e-15 | 6.1e-14 |
| Pt amine / def2-TZVP (ECP) | | **49** | \- | \- | \- | \- | \- |
| CuCl2 / def2-TZVPP | 148 | **64** | 3.6e-16 | 3.9e-15 | 4.8e-15 | 7.2e-16 | 3.7e-14 |
| CuCl2 / def2-QZVP | 224 | **100** | 3.6e-16 | 1.1e-14 | 1.1e-14 | 2.9e-15 | 7.4e-14 |

The "on/off" columns are the point: **1.8e-16 to 2.3e-14 relative**, i.e. the
answer does not know which module served a class block.  The "vs GPU4PySCF"
columns are the usual agreement of these kernels with the reference and are
unchanged by this round.

**Two of these rows failed the first time the test ran, and both were the
test's expectation rather than the code.**  They are worth recording because
each is a fact about the basis sets that is easy to get wrong from the outside:

* the def2-QZVP plan is `(100, **2**)`, not `(100, 1)`.  A 3d metal's d shell
  is contracted over 6 primitives there and its innermost s over 11, so a
  `(d, s)` bra pair is 66 -- the plan has to reach `l = 2`, and §5 says what
  still falls through when it does.
* the **Pt complex's** longest pair is **49**, not 64.  Its metal carries the
  def2 ECP, which *replaced* the deep core, so the 8-primitive s shell that
  makes a 3d complex's 64 does not exist; the pair that survives is its
  chlorines' 7 x 7.  §4 item 6 reads the bound as a transition-metal problem,
  and here is a transition-metal complex where the metal is the one atom that
  does not cause it.

The **range-separated driver** gets its own row, because `get_k_rsh` has its
own class loop and its own copy of the bra-pair test -- it builds
`lr*erf + sr*erfc` in one pass over the quartets rather than going through
`get_k` -- and a fix applied to one and not the other would be invisible.
Writing that check turned up two things about the driver worth recording, both
of which cost a failed run first:

* **the operator GPU4PySCF asks for is not the one it passes.**  1.8.x hands
  `get_k` a raw `(omega, lr_factor, sr_factor)` and `_check_rsh_factors`
  resolves it; wB97X passes `(+0.3, 1.0, 0.158)` and the *resolved* omega is
  negative, which is the fused request.  Testing the raw value says "not
  range-separated" about a call that is.
* **the option object cannot be rebuilt.**  For the fused request the driver is
  called on a `_VHFOpt` built inside `mol.with_range_coulomb(|omega|)`, and
  `get_k_rsh` asserts exactly that.  Replaying a recorded call on a freshly
  built full-range option object trips the assertion -- which is the assertion
  doing its job, and the reason the test captures `self` along with the
  arguments.

With both fixed the row reads, on CuCl2/def2-TZVP with wB97X:

    tmc_cucl2/def2-tzvp  119  get_k_rsh (fused)  omega=-0.300
                              on/off 2.1e-15   vs GPU4PySCF 7.0e-13

-- the wider bound changes nothing, and the 7.0e-13 against GPU4PySCF is the
fused decomposition's own round-off (`erf + erfc == 1` turns one request into
two passes with different cancellations), not this round's.

The first half of the test needs no GPU: `_ksplit.mpp_plan` is checked against
hand-written `(l, nprim)` group tables -- a Pople set, a def2-SVP one, def2-TZVP
with and without sulfur, def2-QZVP -- and against a pair past the cap, which
must be left out rather than clipped.  It runs before any compile, so a wrong
plan fails in a second instead of after four minutes of NVRTC.

Two rows of that table are the regression check rather than a comparison.
6-31G\* and def2-SVP produce `(36, -1)`: no second module is planned, no
`-DMAX_PRIM_PAIR` is passed, and the source text and compile line handed to
NVRTC are byte for byte what they were before this round, so the existing CuPy
disk-cache entries are still valid and the default path is not merely
equivalent but identical.

## 4. What it is worth

**Where these were measured, and why the shape of the measurement matters
more than usual here.**  `docs/OPENSHELL_TMC.md` §5 records that a shared
login node at full load is useless for A/B timing -- the same stock build
measured 10.8 s in one process and 77.3 s in the next.  These numbers were
taken on exactly such a node (one A100-PCIE-40GB, `CUDA_VISIBLE_DEVICES=0`,
load average around 38 throughout), and they are usable anyway **because the
two columns never live in different processes**: the two bounds are two entries
of the same module cache, so `benchmarks/prim_pair.py` flips between them on
the same fixed density and the same `_VHFOpt`, alternating and taking the
minimum of several repetitions of each.  A load spike lands on both columns or
on neither.  The absolute seconds in the tables below are therefore worth less
than the ratios, and the *differences* -- which turn out to be nearly constant
across a 7.8x range of system size -- are the part to read.

### 4a. The exchange build and the gradient's two-electron derivative

`benchmarks/prim_pair.py`, one fixed density per case, both bounds alternating
in one process, minimum of two rounds of three repetitions each:

| system | basis | AOs | longest pair | `get_k` at 36 | at the wider bound | | `ejk` at 36 | at the wider bound | |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CuCl2, doublet | def2-TZVP | 119 | 64 | 0.924 s | 0.884 s | **1.046x** | 3.616 s | 3.591 s | 1.007x |
| FeCl4-, sextet | def2-TZVP | 193 | 64 | 1.008 s | 0.967 s | **1.042x** | 3.999 s | 3.973 s | 1.007x |
| Ni dithiolene, doublet | def2-TZVP | 689 | 64 | 2.812 s | 2.765 s | 1.017x | 11.002 s | 10.971 s | 1.003x |
| Cu macrocycle, quartet | def2-TZVP | 924 | 64 | 4.526 s | 4.478 s | 1.011x | 13.452 s | 13.423 s | 1.002x |
| penicillin V, closed shell | def2-TZVP | 858 | 49 | 1.676 s | 1.670 s | 1.003x | 6.218 s | 6.219 s | 1.000x |
| **CuCl2, doublet** | **def2-QZVP** | 224 | **100** | 4.954 s | 3.987 s | **1.242x** | 32.46 s | 31.18 s | **1.041x** |
| glycine, closed shell | 6-31G\* | 80 | -- | 0.030 s | 0.029 s | 1.021x | 0.056 s | 0.056 s | 1.003x |
| FeCl4-, sextet | def2-SVP | 103 | -- | 0.288 s | 0.288 s | 1.000x | 1.139 s | 1.139 s | 1.000x |

The penicillin V row is three repetitions at eight reps each; the first
measurement of it, at three reps, came out at **0.976x on the gradient term**
and that was noise -- the three repeats are 0.9998x, 1.0004x and 0.9995x.  It
is in this document because a 2.4 % regression on the one closed-shell organic
case would have been exactly the shape of a real finding, and the thing that
settled it was repeating the measurement rather than explaining it.

The last two rows are the regression check rather than a comparison: both
bases plan `(36, -1)`, so the two columns are the *same compiled kernels*
reached by the same path.  FeCl4-/def2-SVP reads 1.000x/1.000x on calls of
288 ms and 1.14 s; glycine/6-31G\* reads 1.021x on a 30 ms call, which is what
this harness's noise looks like when the call is short enough for launch
overhead to dominate.  That pair is also the yardstick for everything above:
differences below about 2 % on a sub-100 ms call mean nothing here, and
differences on a call of a second or more mean what they say.

Three things to take from the table rather than from its largest number.

**What the bound costs is nearly constant in system size, so the ratio decays
while the saving does not.** Across def2-TZVP the saving is 41, 41, 47 and
49 ms of exchange build over a **7.8x** range of AO count (119 to 924), and 25
to 31 ms of gradient.  What the bound blocked was a fixed handful of compact
core `(s,s)` and `(p,s)` class blocks -- one per metal atom, a few per second-row
ligand -- and a molecule ten times larger has no more of them.  So 1.046x at
119 AOs and 1.011x at 924 AOs are the same absolute fact seen against different
denominators, and neither extrapolates: the right statement is "about 45 ms per
exchange build", not a ratio.

**The gradient barely moves, and it is not for want of coverage.**  `fastejk`
takes back the same 25 class blocks -- `benchmarks/coverage_probe.py` in §4b
confirms 4968/4968 -- but its per-call cost is three to four times the exchange
build's on the same system, so the same ~28 ms is 0.2-0.7 % rather than 1-5 %.

**def2-QZVP is a different problem, not a bigger version of the same one.**
1.242x on the exchange build and 1.041x on the gradient, on a 3-atom molecule,
because at quadruple zeta the deep contraction is not a property of the metal:
*every* element has an 8- to 11-primitive innermost s shell, the plan's `lmax`
reaches 2 rather than 1, and what fell through was a large share of the class
blocks rather than a handful of core ones.  This is the number that makes the
entry worth taking, and it is the one no def2-TZVP measurement would have
predicted.

### 4b. The bound no longer fires

`benchmarks/coverage_probe.py` counts, inside each call, how many
angular-momentum class blocks our kernels took and how many went to
GPU4PySCF's.  On the CuCl2 doublet, a UKS B3LYP SCF and gradient, at both
bounds:

| basis | bound | `fastk` ours / total | `fastj` fell through | `fastejk` ours / total |
|---|---|---|---|---|
| def2-TZVP | 36 | 97327 / 97852 (**525 back**) | 0 | 4943 / 4968 (**25 back**) |
| def2-TZVP | **64** | **97852 / 97852** | 0 | **4968 / 4968** |
| def2-TZVPP | 36 | 91780 / 92305 (**525 back**) | 672 | 3613 / 4860 |
| def2-TZVPP | **64** | **92305 / 92305** | 672 | 3638 / 4860 |

The first two rows are `docs/OPENSHELL_TMC.md` §4 item 6's table with a second
line: the bound fired on 525 exchange and 25 gradient class blocks, and now
fires on none.

The last two settle something **item 7** said in passing and could not check.
It reads the 525 blocks `fastk` handed back at def2-TZVPP as "the
`MAX_PRIM_PAIR` pairs of item 6 and nothing to do with the angular momentum".
That is exactly right: widening the bound takes `fastk` to 92305 / 92305 at
`lmax = 4`, so it now covers **every class an spdfg basis reaches**.  What is
left going back is `fastj`'s 672 blocks and `fastejk`'s 1222 -- 1247 before,
the 25 being item 6's -- and all of those are g-containing classes, which is
item 7 and is untouched here.

### 4c. First-run JIT, and what a warmup takes off it

First, how large the thing is.  `benchmarks/compile_time.py` compiles each
family on its own, at the subset an `lmax` reaches:

| family | lmax 2 (6-31G\*) | lmax 3 (any def2 on a metal) |
|---|---|---|
| `fastk` | 42/89 kernels, 0.93 MB, **14.4 s** | 89/89, 2.80 MB, **40.1 s** |
| `fastejk` | 25/65, 1.92 MB, **19.2 s** | 65/65, 6.84 MB, **64.1 s** |
| `fastj` | 16/29, 0.79 MB, **9.7 s** | 29/29, 4.71 MB, **75.0 s** |
| total | **43.3 s** | **179.2 s** |

> **A correction to that benchmark, not to this round's numbers.**  Its third
> row per family, the whole-family baseline it prints "vs full" ratios against,
> comes back at 1.6-2.9 s -- *faster* than the subset it is supposed to bound.
> Those rows produce byte-identical source to the `lmax = 3` rows, so they are
> CuPy disk-cache hits: `CUPY_CACHE_IN_MEMORY=1`, which that script sets in its
> child precisely to stop this, does not suppress the disk cache on CuPy 14.
> The per-`lmax` rows above are real compiles; the "vs full" column is not
> measuring what it says and is not quoted here.

Then, what a run actually pays and what the warmup takes off it.
`benchmarks/warm_start.py` runs one SCF cycle and one gradient three times,
each against its own freshly created `CUPY_CACHE_DIR`:

| case | cold job | warm job | NVRTC inside the run | warmup, 1 worker | warmup, 8 | |
|---|---:|---:|---:|---:|---:|---:|
| glycine / 6-31G\*, closed shell | 63.5 s | 21.0 s | **42.4 s** (67 %) | 50.7 s | 27.8 s | 1.83x |
| FeCl4- / def2-SVP, open shell | 223.1 s | 25.0 s | **198.1 s** (89 %) | 248.3 s | 92.4 s | 2.69x |
| CuCl2 / def2-TZVP, open shell | 215.8 s | 29.8 s | **186.0 s** (86 %) | 272.5 s | 89.6 s | 3.04x |

**This is §4 item 8 as a measurement rather than a claim.**  The entry says a
def2 basis "pays the full 98 s" where a 6-31G\* job pays 36 s.  The shape is
right and the size is larger: on this node a Pople first run pays **42 s** of
NVRTC and a def2 transition-metal one pays **186-198 s**, which is **4.4x**,
and it is **86-89 % of the entire first run** rather than an overhead on it.
(The absolute seconds are inflated by the node's load; the ratio between the
two columns is not, because both were measured in the same conditions.)

Three things the table says that the entry could not.

**The warmup removes it from the run completely.**  The warm job is 21-30 s in
every case, and it is the same whether the cache was filled by one worker or
eight -- the cache is the cache.  What is left in those 21-30 s is the
calculation plus about 2 s of imports.

**Eight workers are worth 1.8x-3.0x, not 8x, and the reason is a floor rather
than an inefficiency.**  The compiles are independent processes, so the wall
clock is the *longest* one: `fastj` at `lmax = 3` alone is 75 s of the 90 s.
The gain is largest exactly where it is wanted -- the def2 cases, where there
are more and larger independent units -- and smallest at 6-31G\*, where the
whole thing is 43 s of work in four uneven pieces.

**A serial warmup costs slightly more than the run's own NVRTC** (248 s against
198 s, 272 s against 186 s), and that is not waste to be tuned away: the warmup
compiles `fastejk` at **both** `EJK_NDM` values and, at def2-TZVP, the wide
bra-pair modules of §2 as well.  A single job compiles only the variants it
happens to reach.  A dataset run of mixed closed- and open-shell complexes
reaches both, which is the case the warmup is for -- and the wide modules add
only **24 s** to it (272.5 s at def2-TZVP against 248.3 s at def2-SVP), which
is what keeping the second module to 14 kernels buys.

## 5. What is still not covered

**Item 7, g shells in `fastj` and `fastejk`, is untouched** and is the largest
of the three basis-set entries: on CuCl2/def2-TZVPP `fastejk` answers 3 613 of
4 860 class blocks against 4 943 of 4 968 at def2-TZVP, and `fastj` hands back
672 blocks where it hands back none below g.  Nothing here changes that, and
`docs/OPENSHELL_TMC.md` §4 item 7 still says what would have to be written.

**Classes outside the one-thread-per-quartet family still fall through at a
long bra pair.**  The second module carries `gen_kernels.py`'s classes and not
`gen_k2_kernels.py`'s or `gen_khigh.py`'s, for the reason §2 gives -- the first
of those bakes the bound in as a literal, and the second was sized against it.
At def2-TZVP this costs nothing, because the plan comes out at `lmax = 1` and
every `l <= 1` class is in the one-thread family.  At **def2-QZVP it is
real**: the plan reaches `lmax = 2`, and a class like `(2021)` whose bra is a
d group paired with a 10-primitive s group has no one-thread kernel, so it
falls through as before.  The measured def2-QZVP number below is therefore a
floor rather than the whole of what the entry is worth.

**A bra pair past `ELECTROLITES_MAX_PRIM_PAIR` still falls through.**  100
covers every def2 set; an ANO or a Dunning set with a deeper contraction would
need the cap raised, and raising it is a one-line change whose cost is the
shared memory §2 accounts for, on the second module only.

**The warmup does not make a single job's first run cheaper.**  It moves the
cost out of the timed run and divides it by the number of workers; the total
NVRTC work is the same, and a user who runs one calculation on a cold cache
pays what they always paid.  What changed is that filling the cache is now a
command rather than a hope, and that the command compiles the exact module keys
the run will ask for.

## Files

| file | what |
|---|---|
| `electrolites/_ksplit.py` | `mpp_plan`, `MAX_PRIM_PAIR_CAP`, and `ELECTROLITES_MAX_PRIM_PAIR` |
| `electrolites/fastk.py` | `_sources(wide=)`, `_module(lmax, mpp)`, the `mpp` argument of `_usable`/`_function`/`_run_k1`, and the dispatch that picks a module per class block |
| `electrolites/fastjk.py` | the same, for the fused build |
| `electrolites/fastejk.py` | the same, plus `SHM_PER_PRIM_PAIR`/`SHM_OTHER` and the shared-memory test that keeps an shm-style class from being launched from a module it does not fit in |
| `electrolites/kernels/fastk_prologue.cu`, `fastejk_prologue.cu` | `#ifndef MAX_PRIM_PAIR` |
| `electrolites/codegen/gen_ejk.py`, `gen_khigh.py` | emit `double[MAX_PRIM_PAIR]` rather than `double[36]` |
| `electrolites/warmup.py` | the plan, the job table, the worker pool, and the CLI |
| `tests/test_prim_pair.py` | the plan with no GPU, and the three builds at both bounds against GPU4PySCF |
| `benchmarks/prim_pair.py` | both bounds in one process, and `--report` for which class blocks the bound moves |
| `benchmarks/warm_start.py` | cold job, warmup, warm job, in fresh cache directories |
