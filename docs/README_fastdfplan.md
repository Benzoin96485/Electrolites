# The cuTENSOR plan wall — what it refuses, and asking instead of retrying

`docs/DF_WORK_LIST.md` **W21**.  Three modules here have hit
`CUTENSOR_STATUS_NOT_SUPPORTED` with tens of GB free: `fastdfhess` on the DF
Hessian's CPHF build (W8), `fastdfgrad` on the DF gradient's ip1 loop (W10)
and `fastdfhessip` on the Gram form of the per-atom occupied block (W17).
Each guarded it differently — W8 with a bisected element cap and a
halve-and-retry, W10 with the retry alone because "a cap cannot describe it",
W17 with a 1.2e8-element chunk bound and a halve-along-`r` retry — and the
entry asks for one statement about what cuTENSOR actually refuses, and one
fewer constant.

It gets both, and the falsification criterion fires: the refusal **is**
reproducible and predictable, so a cheap test beats a retry and all three
modules now carry it.

---

## 0. Summary

*(cuTENSOR 2.6.0, CUDA runtime 12.9, CuPy 14.0.1.  The plan queries give the
same answers on a login node's A100 80 GB PCIe and a compute node's
A100-SXM4-40GB; everything timed is on whole Perlmutter GPU nodes.)*

* **A plan depends on shapes and strides, not on data or free memory, so it
  can be asked for without the data.**  `electrolites/_cuplan.py` builds the
  descriptors on a one-kilobyte buffer and makes exactly the two calls
  `gpu4pyscf/lib/cutensor.py:contraction` makes.  A query is ~0.3 ms and a
  37 GB output costs the same to ask about as a 37 kB one; the pointer's
  alignment does not enter (offsets of 0, 8, 16, 64 and 128 bytes give the
  same accepted set).  `tests/test_cuplan.py` allocates the real arrays and
  checks that the query's answer is the real call's, both ways.
* **The one statement.**  *An output of more than 2²⁹ = 536 870 912 elements
  is refused unless one mode of it can be cut, by factors no larger than 13,
  into pieces of at most 2²⁹ elements.*  The mode is, for every pattern
  measured, the last output mode of the first operand.  The limit is in
  **elements**, not bytes — it is the same for float32, float64 and
  complex128 — and it is on the **output** only: an input of 4.3e9 elements
  plans (§1).
* **That rule is exact** on all 5999 extents of the gradient's ip1
  contraction that were walked, on W8's contraction at two shapes, and on
  3994 of 4000 random shapes over ten patterns; the six exceptions are
  outputs of 30–50× the limit on the five-mode Hessian pattern, where
  cuTENSOR evidently cuts a second mode too.  No algorithm (`DEFAULT`,
  `DEFAULT_PATIENT`, `GETT`, `TGETT`, `TTGT`), workspace preference (`MIN`,
  `RECOMMENDED`, `MAX`) or JIT mode changes a single verdict — 0 of 30
  combinations plan a refused extent (§1.3).
* **W10's "not monotonic in the block" was the remainder.**  Logged at the
  moment of refusal (`benchmarks/cutensor_plan_probe.py --run`), the
  contraction refused with the ip1 block pinned at 1096 on taxol has
  `r = 914`, not 1096: the first auxiliary batch leaves 914 = 2 × 457
  functions after its 1096-wide block, and `1185² × 914` is past 2²⁹ with no
  factor of 13 or less.  1096 = 8 × 137 plans; so does 1240, whose remainder
  is 770 = 2 × 5 × 7 × 11.  W10 was right that neither free memory nor the
  output's size is the variable, and wrong about which extent was refused
  (§2).
* **W8's cap was a real boundary, not a coincidence.**  Its 5e8 sits under
  2²⁹ = 5.37e8, and the four bisected ceilings it could not explain —
  5.4e9 elements at nao 430, 1.1e9 at 478, 5.2e8 at 1007 and 1123 — are
  10, 2, 1 and 1 times 2²⁹: the 13-smooth part of `nao` (430 = 2·5·43,
  478 = 2·239, 1007 = 19·53, 1123 prime), because in `Lpq,snqi->snpiL` the
  cuttable mode is `p`.  The query reproduces all four bisections, 22 and
  530 among them, from nothing but the shape.
* **W17's refusal is not reproducible at the shapes the code now has.**  The
  chunk bound was put there because `'xrui,yrvi->xyuv'` was refused once at
  42 atoms with the whole auxiliary set in one chunk; that contraction's
  output is `(3, 3, nao, nao)` whatever the chunk, and with the bound removed
  the whole-set chunk (2.75e8 input elements) plans and runs.  The bound is
  now a query, and `FASTDFHESSIP_RCHUNK` puts W17's back.
* **All three modules ask instead of retrying**, and every retry is kept
  behind the query as a backstop: `fastdfgrad` chooses every ip1 block —
  the remainder included — as an extent that plans (`_cuplan.prange`, which
  cuts a refused remainder in two near the middle: 914 → 458 + 456);
  `fastdfhess` takes the largest density batch that plans in place of W8's
  constant, and plans the last batch too; `fastdfhessip` bounds its chunk by
  the query.  One constant fewer in the repository (`MAX_PLAN_ELEMENTS` now
  defaults to 0, meaning "ask"), one fewer in `fastdfhessip`, and
  `plan_retries` reads 0 on every case in §3.

---

## 1. What cuTENSOR refuses

### 1.1 A threshold on the output, in elements

`benchmarks/cutensor_plan_probe.py --threshold`: for `iqr,pi->pqr` (the ip1
contraction, `nocc = 226`), the smallest prime extent of `r` that is refused,
as `nao` moves:

| dtype | `nao` | first refused prime `r` | output elements | log₂ |
|---|---|---|---|---|
| float64 | 200 | 13441 | 537 640 000 | 29.0021 |
| float64 | 400 | 3359 | 537 440 000 | 29.0015 |
| float64 | 800 | 839 | 536 960 000 | 29.0002 |
| float64 | 1185 | 383 | 537 818 175 | 29.0025 |
| float64 | 1600 | 211 | 540 160 000 | 29.0088 |
| float64 | 2400 | 97 | 558 720 000 | 29.0576 |
| float32 | 200 | 13441 | 537 640 000 | 29.0021 |
| complex128 | 200 | 13441 | 537 640 000 | 29.0021 |

Every row is the first prime past `2²⁹ / nao²` (at `nao = 2400` the prime
before 97 is 89, which is under it).  `nocc` does not move it at all —
20, 50, 100, 226, 400 and 800 all give 383 at `nao = 1185` — and neither does
the dtype, so it is a count of elements, not of bytes.

And it is the *output* that is counted: `ik,kj->ij` with an input of
1.08e9 or 4.30e9 elements plans; the same GEMM with an output of that size,
both extents prime, is refused.

### 1.2 Past the threshold, a cut

Scanned over every `r` from 1 to 5999 at `(nao 1185, nocc 226)`, **2714 are
refused** — the same 2714 on both cards.  The first are primes (383, 389, 397, …), then products of two
primes above 13 (391 = 17·23, 437 = 19·23, …), then twice a large prime
(766, 778, …, 914).  The accepted set is exactly

> `r` has a divisor `d` with `nao² · d ≤ 2²⁹` and `r / d` 13-smooth,

i.e. cuTENSOR cuts `r` by factors of at most 13 until each piece fits:
403 = 13·31 plans and 391 = 17·23 does not, so the largest usable factor is
between 13 and 16, and a smooth-part rule with 11 in place of 13 mismatches
216 extents where 13 mismatches none.

Which mode is cut is a property of the pattern:

| pattern | cut | not cut |
|---|---|---|
| `iqr,pi->pqr` (ip1) | `r` | `p`, `q` — 1185 = 3·5·79 would give 15 pieces and does not |
| `Lpq,snqi->snpiL` (W8) | `p` | `n`, `i`, `L` — `L = 256` would give 256 |
| `ik,kj->ij` | `i` | `j` |
| `iab,ic->abc` | `b` | `a`, `c` |
| `ia,ibc->abc` | `a` | `b`, `c` |

— in every case the last mode of the first operand that is also an output
mode.  For W8's pattern that makes the refusal a plain threshold in the batch
(`n` cannot be cut), which is why a cap worked there and why it could not in
the gradient, where the extent that moves *is* the one that is cut.

`--rule --trials 4000` draws random shapes over ten patterns with outputs of
2^27.5–2^32 elements for the free modes: the rule predicts 3994.  The six it
misses are all `Lpq,snqi->snpiL` with outputs of 1.6e10–2.6e10 elements, thirty to fifty
times the limit, which cuTENSOR accepts — there it cuts a second mode.  No
shape this repository produces is within an order of magnitude of that, and
nothing here depends on the rule: the modules ask.

### 1.3 Nothing in the plan preference moves it

`--options`: at `r = 383` and `r = 914`, every combination of the five
algorithms, the three workspace preferences and two JIT modes is refused (0
of 30), and at 1096 every one plans.  `estimateWorkspaceSize` itself fails —
Cython prints the exception and returns 0 — and `createPlan` is where it
surfaces.  It is a limit of the kernels cuTENSOR 2.6.0 has, not of how it is
asked.

---

## 2. What W10 measured, read again

`FASTDFGRAD_BLK=1096` on taxol, run under `cutensor_plan_probe.py --run`,
which prints each refused contraction's operands:

```
@@REFUSED {"pattern": "iqr,pi->pqr", "a": [[226, 1185, 914], ...],
           "c": [[1185, 1185, 914], [8664720, 7312, 8]], ...}
```

The ip1 loop cuts each auxiliary batch with `lib.prange(0, nb, blksize)`,
and taxol's first batch is 2010 functions: 1096 and then **914**.  W10's
table — "1096 is refused at 70.3, 62.2, 54.2, 50.2 and 46.2 GB free alike,
1240 is accepted at all five, 1456, 2048 and 3072 are accepted" — is exactly
what the rule says of the blocks those settings produce from a 2010-wide
batch: 1096 + **914** (refused), 1240 + 770, 1456 + 554, and 2010 in one
block for the last two — all of which plan.  And 548, which halving 1096
produced, gives 548 × 3 + 366, which plans too.  The halving that
"cleared every refusal" cleared it by changing the remainder.

---

## 3. What the three modules do now, and what it is worth

| module | before | now |
|---|---|---|
| `fastdfgrad` | `lib.prange(0, nb, blksize)`; on a refusal, halve `blksize` and restart the whole ip1 loop | `_block_range`: every block, the remainder included, an extent both ip1 contractions plan; the retry behind it (`FASTDFGRAD_PLANQ=0` is the ablation) |
| `fastdfhess` | batch capped at `5e8 / (nspin nao nocc blksize)`; retry | the largest batch every batch-sized contraction of the chosen arrangement plans, and the batches themselves (the last one included) cut by `_cuplan.prange`; retry.  `FASTDFHESS_MAX_PLAN=5e8` restores W8 |
| `fastdfhessip` | chunk bounded at `1.2e8 / (3 nao²)`; halve-along-`r` retry | chunk bounded by the query; retry.  `FASTDFHESSIP_RCHUNK=1.2e8` restores W17 |

The restart is not free: a refused block is found only when the loop reaches
it, and everything before it in the loop is done again at half the block.

`benchmarks/df_grad_bench.py`, DF-B3LYP, the two-electron derivative and the
whole gradient, each setting in its own process and the pair twice:

| | card | `plan_retries` (retry / query) | retry alone | **query** | |
|---|---|---|---|---|---|
| taxol, ip1 block pinned at 1096 (W10's case) | 80 GB | 1 per call / 0 | 3347–3367 ms | **2466–2510** | **1.34x** |
| ↳ whole gradient | | | 4249–4253 ms | **3354–3435** | **1.24x** |
| taxol, block from free memory | 40 GB | 0 / 0 | 2479–2522 | 2485–2501 | 1.00x |
| PfPMT, 284 atoms, block from free memory | 40 GB | 1 / 0 | 64 882–65 043 | 65 284 | 1.00x |
| PfPMT, block from free memory | 80 GB | 0 / 0 | 54 683 | 54 776 | 1.00x |

Where the refused block is late in the loop, the restart re-does everything
before it at half the block, and asking first is 1.34x on the term and 1.24x
on the gradient.  Where it comes early (the 284-atom case on a 40 GB card
refuses in its first batch) the restart costs nothing measurable, and where
nothing is refused the query costs nothing either — 0.3 ms a query, a few
dozen queries a call, cached after the first.  So this is not a speed item on
the default path; it is that the three modules no longer *depend* on a
retry, and that the one statement above is now what they are written
against.

---

## 4. Correctness

`tests/test_cuplan.py` allocates what it asks about — outputs of 2.7 and
4.3 GB — and checks that a refused extent is refused for real
(`CuTensorError`), that an accepted one runs and is right against cuBLAS, that
an accepted extent can be *larger* than a refused one (1024 against 839 and
1678 at `nao = 800`), that `_cuplan.prange` returns only extents that plan
over three totals and four steps, and that `fastdfhess`'s query reproduces
W8's two hand bisections (22 at `nao 1007`, 530 at `nao 430`).
`tests/test_dfgrad.py`, `tests/test_dfhess.py` and `tests/test_dfhessip.py`
pass unchanged, including the injected-refusal test of the gradient's retry.

---

## 5. Tools

```bash
python tests/test_cuplan.py
python benchmarks/cutensor_plan_probe.py --scan 'iqr,pi->pqr' \
    --shape 'i=226,q=1185,p=1185' --mode r --range 1:6000
python benchmarks/cutensor_plan_probe.py --threshold --rule --options
python benchmarks/cutensor_plan_probe.py --run benchmarks/df_grad_bench.py \
    --xyz benchmarks/molecules/taxol.xyz --basis def2-svp --xc b3lyp \
    --cycles 1 --reps 1 --variants fastdfgrad      # prints @@REFUSED lines
FASTDFGRAD_PLANQ=0 python benchmarks/df_grad_bench.py ...   # the retry alone
```

Switches: `FASTDFGRAD_PLANQ=0`, `FASTDFHESS_MAX_PLAN=<elements>`,
`FASTDFHESSIP_RCHUNK=<elements>` — each puts back what its module did before.
