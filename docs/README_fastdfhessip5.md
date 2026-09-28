# The ip2 term's "D3 buffers": a 37 GB buffer, and when the driver makes it cost

`docs/DF_WORK_LIST.md` **W34**, found by W31.  After W31 the ip2 kernel was
124–134 ms of a 1.84 s `_jk_energy_per_atom` call on penicillin V and the
largest row of the stage table was "D3 buffers": 557–637 ms for what the
table said were two `cp.empty` calls and two zero-fills.  The entry asked for
those buffers timed with and without the call's `free_all_blocks`, taken from
a buffer held across calls, and with the first write separated from the
allocation; success was the row at the written volume's bandwidth cost (tens
of ms) and the call faster by what it loses, at 21–113 atoms; the
falsification, a first-touch cost at any placement.

**The row was one `cudaMalloc`, and it was the wrong size.**  Of the two
buffers, `j3c_buf` is 2.2 GB and `work` is **37.4 GB** on penicillin V,
because upstream sizes `work` to hold `_oo_atm_block`'s `X` at
`aux_blksize`, and `_plan_v` sets `aux_blksize` to everything the plan
leaves — the whole free card, capped at `aux_batch_size` rows.  The Gram form
of D4 (W17's item 2) and the Cholesky path (W24's item 8), which is what
every patched call takes, never build `X` and touch 4 GB of it.  GPU4PySCF
sends every allocation over 100 MB straight to `cudaMalloc`
(`gpu4pyscf/__init__.py` calls `set_conditional_mempool_malloc`), so CuPy's
pool and the call's `free_all_blocks` have nothing to do with it; what costs
is that a `cudaMalloc` of tens of GB within about a second of the process
having freed tens of GB **waits for the driver** (§2).  Nothing about first
touch: writing all 37 GB is 21 ms.

**Item 19** sizes `work` to what is built in it (`_work_words`): the `X` term
only where `X` is built, the Gram form's chunk from what the other uses
hold.  The row is **0.6–1.7 ms** (from 543–669), the call **1.53x** (1845 →
1204 ms) and its peak **33 GB lower** on penicillin V when calls follow one
another.  **In a whole Hessian it was never paying** — the call there comes
seconds after the previous large free, and the row reads 2–8 ms with
upstream's size — so whole Hessians do not move (§3.3).  At 113 atoms the
row is 200 ms in both, because there the card really is full: the 36 GB
`j3c_buf` is used.

---

## 0. Summary

* **Where the 0.6 s was** (§1): one `cudaMalloc` of 37.4 GB, the second
  pass's `work` buffer, whose size is `6 natm nocc² min(aux_batch_size,
  what is left)` words — upstream's `X`, which the patched D4 does not build.
* **What makes it cost** (§2): not the pool (removing `free_all_blocks`
  changes nothing; the buffer never goes through the pool), not first touch
  (21 ms to write 37 GB), not the allocation as such (3–8 ms as the first
  call of a process, 0.0 ms held across calls); the driver's reclaim of
  memory freed less than a second earlier.  With two seconds' sleep between
  calls the row is 2 ms at upstream's size.
* **Item 19** (§1.2): `work` sized to its uses, in `fastdfhessip` only (the
  unrestricted twin already left `X` out of its Gram form's buffer).
  `FASTDFHESSIP_WORK=upstream`; `FASTDFHESSIP_D4_ROWS=<n>` fixes the Gram
  form's chunk.
* **The ip2 term** (§3.1): 1.53x at 42 atoms in back-to-back calls; nothing
  at 10, 21, 22 and 113 atoms, in def2-TZVPP, or on a 40 GB card, because
  there the buffer was small (≤ 1.6 GB) or genuinely needed.
* **Whole Hessians** (§3.3): unchanged, 1.00x ± spread, traces to 1e-9.
* **The falsification does not fire** — the time is not first touch — and
  the success bar is met on the row but not on the call "at 21–113 atoms":
  the call only lost it where it followed another large free.

Everything is one A100-SXM4-80GB, GPU4PySCF 1.8.1, def2-SVP and its jkfit
set unless a row says otherwise, `patch_all()` on both sides, a whole
`--qos=debug` node with one GPU visible.  "Upstream's size" is this tree
with `FASTDFHESSIP_WORK=upstream`, i.e. W31.

---

## 1. Where the time was

### 1.1 The stage, split

W29 had split "D3 buffers" out of "D3 integrals", and it charged everything
between the end of stage C and the first integral batch.  Three more marks
(`D0 evaluators`, `D0 plan`, `D3 alloc`, all under `FASTDFHESSIP_TIME=1`),
penicillin V, one process, three calls after a warm one:

| row | upstream's size | |
|---|---|---|
| D0 evaluators (`_ip1_evaluator` × 2) | 4.7–4.9 ms | |
| D0 plan (`_avail`, `_plan_v`) | 0.3 ms | |
| **D3 alloc** (`cp.empty` × 2) | **543–669 ms** | first call of the process: 7.9–11.3 ms |
| D3 buffers (the two zero-fills) | 1.9 ms | |

`nsys` with `--cuda-memory-usage=true` put the whole row in one API call:
`cudaMalloc` of **37.386 GB** taking 444–615 ms, next to the 2.196 GB
`j3c_buf` at 0.9 ms.  That size is upstream's

```python
work = cp.empty(max(aux_unit2*aux_blksize, ...))   # aux_unit2 = 6 natm nocc^2
```

with `aux_blksize = word_avail // aux_unit2` in `_plan_v` — whatever the
plan has not spent — capped at `aux_batch_size`: 42 × 6 × 92² × 2191 words.
The term exists for `_oo_atm_block`, upstream's D4, which builds
`X[A,x,r,i,j]` for every atom.  The patched call builds no `X`: its D4 is
the Gram form (`_oo_gram`, W17) or item 8's `_chol_d4_d5` (W24), whose uses
of `work` are the integral buffers, D5's per-atom temporaries, and the Gram
form's chunk, which took `work.size // per_r` rows — i.e. all of them,
because the buffer was the card.

The term grows as `natm nocc² naux` and is capped by free memory, so it is
largest in the middle: 96 MB at glycine, 1.6 GB at alanine dipeptide, 37 GB
at penicillin V, and at taxol not the maximum at all — there `j3c_buf` is
36 GB and what is left for `work` is 12 GB either way.

### 1.2 Item 19

`_work_words` returns the largest of the uses actually made:

* the second pass's integral buffers, `6 nao_pair batch_size +
  (nao + nocc) nao blksize2` (upstream's second term);
* D5's two per-atom forms (upstream's third and fourth);
* item 8's temporaries: `3 ncart nsph`, and `3 nsph nao_A nocc + ncart
  nao_A nocc + 3 ncart nsph` for D5's spherical form (bounded explicitly here;
  upstream relied on its other terms being larger);
* `X` at `aux_blksize` **only** without the Gram form and without item 8
  (`FASTDFHESSIP_OOSUM=transpose|gemm`, upstream's D4);
* the Gram form's chunk: at least one row, `FASTDFHESSIP_D4_ROWS` rows if
  set.  By default it is whatever the other uses hold — 679 of 2191 rows on
  penicillin V, four chunks.  The chunk does not matter (§3.2).

Item 11's carve of the `h_aux` block from `work` was already bounded by the
integral buffers, and every `_allocate` from `work` asserts, so a missed use
would have failed the tests rather than read past the end.

The buffer is smaller than upstream's wherever upstream's term was the
largest, and the same otherwise, so there is no memory size at which item 19
fits where upstream did not.

---

## 2. What makes a `cudaMalloc` cost half a second

The entry's three questions, on the same molecule and the same stage:

| variant | D3 alloc, calls 2–4 | call |
|---|---|---|
| upstream's size | 543–669 ms | 1845 ms |
| … without the call's `free_all_blocks` (`FASTDFHESSIP_D3PROBE=nofree`) | 639–655 | 1838 |
| … with both buffers held across calls (`=hold`) | **0.0** | **1178** |
| … as the first call of a process | 7.9–11.3 | |
| … after sleeping 0.5 s since the previous call | 81–103 | 1342–1356 |
| … after sleeping 1 s | 13.1–13.6 | 1279–1284 |
| … after sleeping 2 s | 2.1 | 1217–1223 |
| item 19 | **0.6–1.7** | **1204** |

(The sleep rows are `benchmarks/df_ip2_alloc_probe.py --xyz`, the others
`df_hess_terms_probe.py` in separate processes.)

* `free_all_blocks` is irrelevant: a buffer over 100 MB is a bare
  `cudaMalloc` under GPU4PySCF's allocator and is freed with `cudaFree` the
  moment its last reference goes (the pool held 0.36 GB at the end of the
  second pass).
* First touch is not the cost: the zero-fills are the 1.9 ms "D3 buffers"
  row, and a `cudaMemset` of all 37 GB is 21 ms.
* Placement is: a buffer that is never freed costs nothing, and neither does
  one allocated a second after the last large free.

The allocation alone, with no chemistry (`df_ip2_alloc_probe.py --malloc`):
allocate, write, free, wait, allocate again, three times each:

| size | no wait | 0.5 s | 1 s | 2 s |
|---|---|---|---|---|
| 4 GB | 1–3 ms | 2–8 | 2–11 | 2–4 |
| 8 GB | 1–4 | 3–14 | 3–6 | 3–6 |
| 16 GB | 1–7, once **804** | 2–12 | 6–19 | 6–13 |
| 37 GB | 3–11, **1927, 1931** | 3–17, **1332, 1928** | 14–19, **1317** | 14–17 |

(pairs "first/second" in the probe's output; the bold entries are the
allocations that waited.)  A 37 GB allocation right after a 37 GB free waits
1.3–1.9 s on its own, where the ip2 call's waited 0.45–0.67 s — the call's
free came 0.5 s before its allocation, which is the 0.5 s column.  Below
about 16 GB a wait is rare.  Why the driver waits is not visible from user
space; what is measured is when it does — a large request, within about a
second of a large free — and that neither the pool nor the first write is
involved.

So the rule the stage obeys is the driver's, not this code's: memory the
process has just written and freed is not available to a large `cudaMalloc`
at once; the call waits until the driver has taken it back, which it does
in the background in about a second.  A benchmark that calls the ip2 term
back to back — `df_hess_terms_probe.py` with one variant, which is how W31's
§3.3 table was taken — frees 40 GB at the end of each call and asks for 40 GB
half a second into the next.  A round robin with GPU4PySCF's 4.4 s function
between the calls does not see it (§3.1), and neither does a Hessian (§3.3).

---

## 3. Measurements

### 3.1 The ip2 term, calls back to back

`df_hess_terms_probe.py --terms ip2 --variants ours --reps 3 --no-check`
with `FASTDFHESSIP_TIME=1`, upstream's size and item 19 in separate
processes (min of three after a warm call; penicillin V run twice):

| system | atoms | upstream's size | item 19 | | D3 alloc |
|---|---|---|---|---|---|
| glycine | 10 | 153.4 ms | 153.1 | 1.00x | 0.1 → 0.1 ms |
| alanine dipeptide | 22 | 239.8 | 239.7 | 1.00x | 0.4 → 0.4 |
| Ni(II) dithiolene²⁻ | 21 | 1072.1 | 1050.8 | 1.02x | 1.4 → 0.5 |
| **penicillin V** | 42 | **1844.9, 1862.7** | **1204.1, 1204.3** | **1.53x** | 543–669 → 0.6–1.7 |
| penicillin V, def2-TZVPP | 42 | 3492.5 | 3469.4 | 1.01x | 2.3 → 1.3 |
| taxol | 113 | 21131 | 21046 | 1.00x | 199–204 → 188–208 |
| taxol, 40 GB taken first | 113 | 33913 | 33985 | 1.00x | 155–170 → 147–165 |

The unrestricted twin, which had no `X` term in its Gram form's buffer, has
the row at 1.0–1.1 ms on the Cu(II) macrocycle²⁺ quartet (60 atoms) and
199–230 ms on the 113-atom Co(II) sextet (FeCl₄⁻, Ni(II)⁻ and Pt(II) calls:
353, 1531 and 779 ms, unchanged).

In def2-TZVPP the same molecule has `nao = 1007`, so the integral buffers
and `j3c_buf` take the card and upstream's `X` term was no longer the
largest; at 113 atoms the row is `j3c_buf` (36 GB) and `work` (12 GB), both
used, allocated 1.8 s after stage A and C freed 8 GB.  With a 40 GB ballast
(`benchmarks/with_ballast.py 40 …`, 44.6 GB free) taxol's call takes the
unfused, one-chunk plan in both columns and the row is the same.

A round robin in one process (`--variants stock,ours --env
upw:FASTDFHESSIP_WORK=upstream,…`, min of three):

| variant | call | vs GPU4PySCF |
|---|---|---|
| GPU4PySCF | 4454.4 ms | |
| item 19 | 1239.1 | 3.59x |
| upstream's size | 1200.0 | 3.71x |
| W29 (`IP2_KERNEL=upstream`) | 1347.3 | 3.31x |

With GPU4PySCF's own 4.4 s call between each pair, upstream's size costs
nothing, and the two columns are within the round robin's ±3 %.  This is
also why W31's §3.2 table, a round robin, read 1835 ms where its §3.3 stage
table read the 0.6 s row: the two were taken differently.

### 3.2 The Gram form's chunk

Item 19 cut the Gram form's auxiliary chunk from all 2191 rows to 679 on
penicillin V.  Round robin, `FASTDFHESSIP_D4_ROWS`:

| chunk | call |
|---|---|
| 679 (default) | 1226.1 ms |
| 256 | 1266.4 |
| 1024 | 1190.2 |
| all 2191 (`=100000`) | 1229.5 |
| upstream's size (all 2191) | 1232.4 |

within the round robin's spread; D4 itself is 76 ms at 679 rows against
88 ms at 2191 in the stage tables of §2.

### 3.3 Whole Hessians

`benchmarks/df_hess_anatomy.py --patch all` with `FASTDFHESSIP_TIME=1`,
upstream's size and item 19 in separate processes, the ip2 row and its
"D3 alloc":

| system | basis | upstream's size | item 19 | ip2 | D3 alloc | trace |
|---|---|---|---|---|---|---|
| penicillin V, 42 | SVP | 13.38, 12.81 s | 13.08, 12.61 s | 1.34, 1.24 → 1.32, 1.21 s | 7.9, 2.1 → 5.5, 0.6 ms | 41.785822457(4–8) both |
| Ni(II) dithiolene²⁻, 21 | SVP | 7.73, 6.91 | 7.28, 6.82 | 1.43, 1.11 → 1.17, 1.12 | 1.4 → 2.6–5.6 | 21.95901039(3–76) |
| penicillin V, 42 | TZVPP | 38.63, 37.95 | 38.63, 37.79 | 3.65, 3.53 → 3.67, 3.42 | 2.3–2.9 → 1.4–4.1 | 39.777317037(4–8) |
| taxol, 113 | SVP | 680.60 | 679.94 | 20.92 → 21.05 | 209 → 211 | 110.84373738(0–1) |
| Pt(II) amine triplet, 23, UKS | SVP | 11.18, 9.08 | 11.30, 9.03 | 0.88 → 0.79 (round 1; the twin's code is unchanged, this is its spread) | 0.4 both | 20.23927778(5–6) |

(Two rounds per process except taxol; round 0 includes compilation.)  In a
Hessian the ip2 call follows `make_h1`'s and the CPHF's work by seconds, the
driver has taken back whatever they freed, and upstream's 37 GB `cudaMalloc`
is 2–8 ms: the half second W31 measured was a property of the benchmark
that measured it.  The Hessian's peak memory is lower by the difference at
42 atoms in def2-SVP, which matters to nothing that runs at the same time.

---

## 4. What is left

* **At 113 atoms the row is 0.2 s of a 21 s call (1 %)**, and it is the same
  mechanism at a size that is used: stage A and C free 8 GB and the second
  pass asks for 48 GB 1.8 s later.  Carving stage C's and stage A's buffers
  out of the second pass's would remove it — which means allocating the
  second pass's buffers before the first, against a plan that is made from
  what the first leaves — and is worth 1 % of the call.  Not done.
* **The kernel.**  Nothing in this round touched it.  After W31 the heavy
  classes' shared-memory contraction is 58 % of `ejk_int3c2e_ip2` on
  penicillin V, 76 % on the Pt(II) complex and 50 % at 113 atoms, capped at
  eight warps per SM by 155 KB of `g` per block
  (`docs/README_fastdfhessip4.md` §4).  But the kernel is 124 ms of the
  1.2 s call at 42 atoms and 1.0 s of 21 s at 113, so the ip2 call's time
  is now the D stages: at 113 atoms D5 per-atom 4.7 s, D2 `h_aux` 3.4 s,
  D4 2.1 s and the two triangular multiplies 3.5 s — 13.7 of 21 s, all
  contractions against the auxiliary index, where the kernel is 5 %.
* **A benchmark rule** (for `docs/DF_WORK_LIST.md` §5): a call timed back to
  back with itself sees allocator effects a Hessian never does, and a round
  robin hides them.  A stage table taken in back-to-back calls needs the
  row checked in a whole Hessian before it is an item.

---

## 5. Correctness

`tests/test_dfhessip.py` (418 checks) and `tests/test_dfuhessip.py` (251)
pass, with four new columns in the first: `WORK=upstream`, `D4_ROWS=7` in
both D4 forms (the Gram form in many chunks), and `D4_ROWS=100000` (one
chunk):

| | worst (relative to the largest element) |
|---|---|
| ip2, ethanol and water, B3LYP / HF / `k = 0` / `omega = 0.3`, every switch | 4.0e-13 (unchanged) |
| the new columns | 3.96e-13 |
| whole Hessians patched against unpatched | **3.5e-12** (RKS), **4.0e-12 Eh/bohr²** (UKS) |

The whole-Hessian traces of §3.3 agree between the two sizes to the
Hessian's own spread (≤ 1e-9).  The terms of §3.1 agree with GPU4PySCF's
function to 3.8e-12 relative in both columns on penicillin V.

---

## 6. Tools

```bash
FASTDFHESSIP_TIME=1 [FASTDFHESSIP_WORK=upstream] \
    python benchmarks/df_hess_terms_probe.py --xyz ... --patch all \
    --terms ip2 --variants ours --reps 3 --no-check         # §1, §3.1
FASTDFHESSIP_D3PROBE=nofree|hold FASTDFHESSIP_WORK=upstream ...   # §2
python benchmarks/df_ip2_alloc_probe.py --malloc 4,8,16,37 \
    --waits 0,0.5,1,2                                       # §2, no chemistry
[FASTDFHESSIP_WORK=upstream] python benchmarks/df_ip2_alloc_probe.py \
    --xyz benchmarks/molecules/penicillinV.xyz --basis def2-svp \
    --xc b3lyp --sleeps 0,0.5,1,2                           # §2, the sleeps
python benchmarks/with_ballast.py 40 benchmarks/df_hess_terms_probe.py ...
                                                            # a 40 GB card
nsys profile -t cuda,nvtx --cuda-memory-usage=true -c cudaProfilerApi \
    -o $SCRATCH/p python benchmarks/df_hess_terms_probe.py ... --nvtx
    # every cudaMalloc/cudaFree with its size, in CUDA_GPU_MEMORY_USAGE_EVENTS
FASTDFHESSIP_TIME=1 python benchmarks/df_hess_anatomy.py --patch all ...  # §3.3
python tests/test_dfhessip.py; python tests/test_dfuhessip.py
```

`FASTDFHESSIP_D3PROBE` is measurement only: `nofree` skips the call's
`free_all_blocks`, `hold` keeps the second pass's two buffers in a module
dictionary across calls (never freed).
