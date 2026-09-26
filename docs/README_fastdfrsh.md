# A range-separated DF run's second tensor: how big it is, when it is not needed, and when one does the work of two

`docs/DF_WORK_LIST.md` **W22**.  `DF.range_coulomb` builds a second cderi
for the long-range operator `erf(ωr)/r` and keeps it next to the full-range
one for the rest of the SCF.  The entry priced that at *"two 86.41 GB host
tensors, 173 of the node's 251 GB"* at 284 atoms, asked whether the second
tensor has to be a second tensor, and set the minimum at `DF.build` saying
how much host memory the pair will take.

The price was not measured, and it is wrong twice.  The answer to the
question is *no* for one class of functional, exactly, and *no* for the rest
at a cost that can be measured.  And measuring it found a patched 284-atom
wB97X-D3 run that did not get past its second cycle.

---

## 0. Summary

*(A100-SXM4-80GB on whole Perlmutter GPU nodes, GPU4PySCF 1.8.1, def2-SVP
with def2-universal-jkfit unless another basis is named, one variant per
process — `docs/DF_WORK_LIST.md` §5.)*

* **The second tensor is 26–64 % of the first, not 100 %.**  The long-range
  metric `(P|erf|Q)` is numerically singular in an auxiliary basis built for
  `1/r` — its smallest eigenvalue is *negative* at rounding level on every
  system measured — so GPU4PySCF's Cholesky fails on it every time and its
  eigen route keeps only the modes above `LINEAR_DEP_THR = 1e-7`.  At
  ω = 0.25 (wB97X-D3) that is 30.6 % of the rows at 284 atoms; at ω = 0.4
  (ωB97) 61.9 %.  The modes it drops carry 1e-8–6e-8 Eh of exchange (§1).
* **And at 284 atoms on an 80 GB card the second tensor is not on the host
  at all.**  26.05 GB fits on the device after the 86.41 GB full-range tensor
  has gone to the host, so the host holds **86.41 GB**, not 173.  Forced to
  the host it is 112.5 GB; for ωB97, whose second tensor is 53 GB, the pair
  is **139.4 GB** of host memory — the only case measured where the entry's
  worry is real (§1).
* **A long-range-only functional builds a tensor and multiplies it by zero.**
  ωB97, LC-ωPBE, LRC-ωPBE and LC-VV10 have `hyb = 0`, and
  `_DFHF.get_veff` still builds the full-range tensor, streams it through a
  whole exchange pass every cycle and scales the result by `0.0`; the tensor
  is used only for J, which the three-centre engine computes with no tensor
  at all.  `electrolites/fastdfrsh.py` builds only the long-range tensor —
  the same one, through the same call — and takes J from the engine.  **It is
  the same energy** (≤ 1e-11 Eh at `conv_tol` 1e-11, restricted and
  unrestricted, three functionals), it is on by default, and at 284 atoms it
  is **139.4 → 53.0 GB** of host memory, **2.62x** per SCF cycle
  (18.91 → 7.21 s of two-electron work) and **2.30x** on the whole SCF
  (269.3 → 117.0 s); a whole taxol ωB97 SCF is **1.63x** (§2).
* **For the other range-separated hybrids one tensor fits the functional's
  own operator as well as two fit its parts.**  `hyb/r + (α − hyb)·erf(ωr)/r`
  is positive definite whenever `hyb ≥ 0` and `α ≥ 0`, so it has a robust fit
  of its own: its own integrals, its own metric, one full-rank tensor and one
  exchange pass.  Against the exact four-centre energy the two-tensor fit is
  off by 1.69e-4 / 2.89e-4 / 4.34e-4 Eh (penicillin V, a Cu(II) quartet,
  taxol) and the fused fit by 1.68e-4 / 2.89e-4 / 4.34e-4: they differ from
  each other by **3.2e-7–7.2e-7 Eh, 0.1–0.2 % of the fitting error they
  share**.  It is a different number, so it is opt-in (`FASTDFRSH=fuse`),
  and it is **1.29x** per cycle at 113 atoms (0.78 → 0.60 s), a whole taxol
  SCF 1.16x, and at 284 atoms one 86.41 GB tensor, **11.55 s** per cycle
  against GPU4PySCF's 24.64, and a whole SCF **1.31x** (218.5 → 166.3 s)
  against the patched two-tensor run.  At 284 atoms the two fits differ by
  1.5e-6 Eh (§3).
* **A patched 284-atom wB97X-D3 run died in its second cycle.**  `fastdf`
  sizes its resident prefix of a host tensor (item 4) and its tile buffer
  (W5) from the free memory at the first call and keeps both; the long-range
  tensor is built after that call, on the device, in what is left; the next
  full-range call then has nowhere to put its prefetch buffers.  GPU4PySCF's
  own run survives, because it keeps neither.  A call that runs out of memory
  now gives both back and runs again (§4).
* **`DF.build` now says it.**  Every host-resident tensor prints its size,
  what is already alive and what the node has left before it is allocated;
  one that cannot fit raises `MemoryError` with those numbers instead of
  reaching the OOM killer mid-store; and the page-locked pool evicts what it
  holds for a build that needs the room (§5).

---

## 1. What the pair actually is

`benchmarks/df_rsh_rank_probe.py` builds each operator's tensor in the full
eigenbasis of its metric and reads off how much exchange each mode carries
(`tr(D K[D]) = Σ_q tr(D B_q D B_q)/w_q`), so one build gives the error of
every truncation threshold at once.  `benchmarks/df_rsh_bench.py` then drives
the real calculation and reports what each `DF.build` left behind.

### 1.1 The long-range metric is rank-deficient, and GPU4PySCF already knows

| | atoms | naux | ω | route | rows stored | of naux | w_min |
|---|---|---|---|---|---|---|---|
| glycine | 10 | 471 | 0.25 | `ed` | 211 | 44.8 % | −6.6e-14 |
| Fe(III)Cl₄⁻, sextet, def2-TZVP | 5 | 776 | 0.25 | `ed` | 254 | 32.7 % | −1.3e-14 |
| Pt(II) amine, triplet (ECP) | 23 | 1 358 | 0.25 | `ed` | 500 | 36.8 % | −6.6e-13 |
| penicillin V | 42 | 2 191 | 0.25 | `ed` | 770 | 35.1 % | −1.5e-12 |
| Cu(II) macrocycle, quartet | 60 | 2 533 | 0.25 | `ed` | 814 | 32.1 % | |
| taxol | 113 | 5 598 | 0.25 | `ed` | 1 691 | 30.2 % | −2.1e-11 |
| PfPMT | 284 | 13 365 | 0.25 | `ed` | 4 091 | 30.6 % | −1.9e-11 |
| penicillin V | 42 | 2 191 | 0.40 | `ed` | 1 396 | 63.7 % | |
| Cu(II) macrocycle | 60 | 2 533 | 0.40 | `ed` | 1 560 | 61.6 % | |
| taxol | 113 | 5 598 | 0.40 | `ed` | 3 353 | 59.9 % | |
| PfPMT | 284 | 13 365 | 0.40 | `ed` | 8 279 | 61.9 % | |

The full-range metric takes the Cholesky route (`cd`, every row) on every one
of these.  The long-range one never does: `erf(ωr)/r` is smooth, so an
auxiliary basis with functions steep enough for `1/r` spans far more than the
long-range operator can see, and the metric's smallest eigenvalue is at or
below zero in double precision.  GPU4PySCF's `_decompose_j2c` catches the
failed `potrf` and keeps the eigenvectors with `w > 1e-7`.

**The fraction is set by ω far more than by the size or the chemistry**:
30–37 % at ω = 0.25 on every system from 5 to 284 atoms but glycine (45 %) —
organic, open-shell 3d, a 5d metal behind an ECP and a triple-ζ basis alike —
and 60–64 % at ω = 0.4.  A larger ω is a
shorter-ranged `erf`, which resolves more of the auxiliary space.

What the dropped modes carry, from the per-mode energies (with the
functional's own `α − hyb` on them):

| | dropped at `w/w_max < 1e-11` | `< 1e-10` | GPU4PySCF's own cut |
|---|---|---|---|
| glycine | 7.8e-10 Eh | 5.7e-9 | at `1.5e-10 w_max`: between 5.7e-9 and 5.8e-8 |
| penicillin V | 7.4e-9 | 6.3e-8 | at `5.4e-11 w_max`: between 7.4e-9 and 6.3e-8 |
| Fe(III)Cl₄⁻, def2-TZVP | | 1.3e-8 | |
| Pt(II) amine | | 3.9e-8 | |

so the truncation upstream already does is three to four orders under the
fitting error (§3.1) and there is nothing to be had from cutting harder: at `1e-8
w_max` penicillin V keeps 21.5 % of the rows and drops 5e-6 Eh.

### 1.2 Where the pair lives at 284 atoms

`benchmarks/df_rsh_bench.py --xyz PfPMT.xyz --charge -1 --xc
hyb_gga_xc_wb97x_d3`, GPU4PySCF unpatched, twice:

| | shape | size | where | build |
|---|---|---|---|---|
| full-range | 13 365 × 808 156 | 86.41 GB | host | 71.4 / 71.6 s |
| long-range, ω = 0.25 | 4 091 × 795 868 | 26.05 GB | **device** | 15.1 / 15.3 s |

`_cholesky_eri` puts a tensor on the device whenever it fits in 95 % of the
free memory, and after the full-range tensor went to the host 26 GB does.  So
the node's host memory holds 86.41 GB (peak RSS 87.8 GB), not the 173 GB the
entry priced.  On a card that cannot take the second tensor — forced here with
`use_gpu_memory = False` — the pair is **112.5 GB**.  The case where it is as
bad as the entry feared is a larger ω: ωB97's long-range tensor is 8 279 rows,
53 GB, it does not fit on the card, and the pair is **139.4 GB** on the host.

---

## 2. The functionals that need only one tensor, exactly

`_DFHF.get_veff`, for any hybrid:

```python
vj, vk = self.get_jk(mol, dm, hermi)          # full-range tensor, J and K
vk *= hyb
if omega != 0:
    vklr = self.get_k(mol, dm, hermi, omega=abs(omega))
    vklr *= (alpha - hyb)
    vk += vklr
```

For `hyb = 0` the first line builds the full-range tensor (86.4 GB and
23–72 s at 284 atoms), runs the whole exchange contraction against it every
cycle, and the second multiplies the result by zero.  The only thing the
tensor is used for is J — and `DF.get_jk` hands a J-only request with no
tensor built to `df_jk.get_j`, the three-centre engine, which is the same
fitted Coulomb matrix computed from the integrals directly (`fastdfj3c`,
W19: 0.14 s at 284 atoms).

`electrolites/fastdfrsh.py` wraps `_DFHF.get_veff` and, for the length of
the call, answers `get_jk` with J from the engine and the exchange from
upstream's own long-range call (`with_df.get_jk(..., omega=ω)`, the one
`_DFHF.get_k` makes), already scaled by α; `rsh_and_hybrid_coeff` tells the
body `(0, 1, 1)` so it neither rescales it nor asks for it again.  The grid
pass, the `ecoul`/`exc` bookkeeping and the returned `tag_array` are
upstream's lines.

**It is the same calculation**: `tests/test_dfrsh.py` §1, against the same
process with the module off, SCF converged to 1e-11:

| | ωB97 | LRC-ωPBE | LC-VV10 |
|---|---|---|---|
| glycine, RKS, def2-SVP | 7.6e-12 | 7.7e-12 | 7.8e-12 |
| CH₂OH, UKS, def2-SVP | 2.3e-12 | 2.4e-12 | — |
| glycine, RKS, def2-TZVP | 9.6e-12 | 9.6e-12 | 9.5e-12 |
| CH₂OH, UKS, def2-TZVP | 2.4e-12 | 2.5e-12 | — |

and on the SCF runs below at the default 1e-9: penicillin V 0, the Cu(II)
quartet 2e-10, taxol 1.7e-9 Eh.

What it is worth (`patch_all()` both sides; *off* is `FASTDFRSH=off`, i.e.
upstream's `get_veff` with every other module still on):

| ωB97 | host tensors | per cycle, two-electron | whole SCF |
|---|---|---|---|
| penicillin V, 42 atoms, off → on | — | | 3.06 → 2.72 s (1.13x) |
| Cu(II) macrocycle, 60, quartet | — | | 8.80 → 7.80 s (1.13x) |
| taxol, 113 | 0 (22.8 → 8.5 GB on the device) | 0.946 → 0.378 s (**2.50x**) | 16.59 → 10.15 s (**1.63x**) |
| PfPMT, 284 | **139.4 → 53.0 GB** | 18.91 → 7.21 s (**2.62x**) | 269.3 → 117.0 s (**2.30x**), 15 cycles each |

(per cycle: mean of three `get_veff` calls on the guess density after the
builds, two processes each; the two agree to 0.3 %.)

---

## 3. The functionals that need two, and one that does the work of both

### 3.1 The operator, and what fitting it costs in accuracy

For `hyb > 0` the exchange a range-separated hybrid asks for is one operator,

    g(r) = hyb / r + (α − hyb) · erf(ωr) / r,

`4π/k² · (hyb + (α − hyb) e^{−k²/4ω²})` in reciprocal space — positive for
every `k` exactly when `hyb ≥ 0` and `α ≥ 0` — wB97X-D3, wB97X-V, wB97M-V
and CAM-B3LYP all qualify.  A positive
definite operator has a robust fit of its own: `(ij|g|P)`, `(P|g|Q)`, one
Cholesky factor, one tensor.  GPU4PySCF fits the two parts separately and
adds the exchange matrices.  Both are variational fits of the same
exchange; neither is the other, and the question is how far apart they are
against how far both are from the exact answer.

A 10-atom CPU check first (`pyscf`, four-centre integrals, the converged
density): the exact exchange energy of glycine is −11.19980917 Eh, the
two-tensor fit is off by 6.924e-5, the fused one by 6.945e-5, and the two
differ by 2.1e-7.  Then whole SCFs, against the four-centre SCF
(`df_rsh_bench.py --no-df`, `fastk`):

| wB97X-D3 | exact | two tensors − exact | fused − exact | fused − two |
|---|---|---|---|---|
| penicillin V, 42 atoms | −1503.1205204725 | −1.687e-4 | −1.683e-4 | **+4.5e-7** |
| Cu(II) macrocycle, 60, UKS quartet | −2565.5558479515 | −2.889e-4 | −2.886e-4 | **+3.2e-7** |
| taxol, 113 | −2926.6503130145 | −4.344e-4 | −4.337e-4 | **+7.2e-7** |
| PfPMT, 284 | (not run) | | | **+1.5e-6** |

The difference is 0.1–0.2 % of the fitting error both carry, 230–640 times
smaller, and on all three the fused fit is the (marginally) closer one.  At
284 atoms (no four-centre reference run) the two differ by 1.5e-6 Eh —
5e-9 per atom, as at 113 (6e-9) — so the gap is size-extensive, as a
fitting difference should be.  It
is still a different number, and this repository's rule is that a module that
changes the answer is off unless asked for — `FASTDF_FP32` and `dflocal` are
the precedents.  So: `FASTDFRSH=fuse`.

`tests/test_dfrsh.py` §2 holds the fused tensor to its own definition instead
of to upstream's number: its exchange matrix against `B_gᵀ J_g⁻¹ B_g D` built
from PySCF's CPU integrals is **4.6e-12** (def2-SVP) and **1.4e-12**
(def2-TZVP) relative, on both residencies.

### 3.2 How it is built

`fastdfcderi._cholesky_eri(..., fuse=(hyb, α − hyb, ω))`: the full-range
evaluator is built as usual, a long-range one is cloned from its
`clone_context` — so it walks the same batches over the same pairs, which are
a superset of the long-range pair list because `erf(ωr)/r ≤ 1/r` bounds every
integral — and each block is `a·(ij|P) + b·(ij|P)_lr` in place before the
metric contraction.  The metric is `a·J + b·J_lr`, positive definite, so the
Cholesky route and W4's sparse contraction apply.  Each operator's integral
block gets half of the fifth of the card upstream gives one.  Everything after
the block — the store, the pin, the host note — is the ordinary build.

The fused tensor lives in `DF._rsh_df` under its own key, so `DF.reset` drops
it like the long-range one.  J comes from the three-centre engine as in §2.

### 3.3 What it is worth

| wB97X-D3 | tensors | per cycle, two-electron | whole SCF |
|---|---|---|---|
| penicillin V, off → fuse | 1.07 + 0.37 GB → 1.07 GB | | 2.91 → 2.97 s |
| Cu(II) macrocycle, UKS | 2.17 + 0.69 → 2.17 GB | | 9.81 → 9.69 s |
| taxol | 14.31 + 4.27 → 14.31 GB | 0.780 → 0.604 s (**1.29x**) | 14.55 → 12.56 s (**1.16x**) |
| PfPMT | 86.41 (host) + 26.05 (device) → 86.41 (host) | 15.95 → 11.55 s (**1.38x**); GPU4PySCF's own 24.64 | 218.5 → 166.3 s (**1.31x**), 15 cycles each |

The build is the one place the fused path pays: at 284 atoms it evaluates two
operators' integrals for one store, 27.7 s, against 23.0 + 11.8 s for the two
builds it replaces.

---

## 4. The patched two-tensor run that stopped at its second cycle

Measuring §3.3's baseline, every patched wB97X-D3 run at 284 atoms died in
its second `get_veff`:

```
File ".../electrolites/fastdf.py", line 641, in _blocks
    buf_prefetch = cp.empty_like(buf)
cupy_backends.cuda.api.runtime.CUDARuntimeError: cudaErrorMemoryAllocation: out of memory
```

and GPU4PySCF's own run of the same molecule did not (24.64 s per cycle,
two repeated calls).  The sequence:

1. cycle 1, full-range call: the tensor is on the host, and `fastdf` item 4
   puts a resident prefix of it on the device, sized at half the free memory,
   and the tiled path (W5) a 10.8 GB buffer whose block is fixed there
   (§5.9) — both **kept** for every later call;
2. cycle 1, long-range call: the long-range tensor is built — after the
   prefix exists — and `_cholesky_eri` puts it on the device because 26 GB
   still fits in 95 % of what is free;
3. cycle 2, full-range call: the block buffers are sized from the free memory
   *now*, but the two prefetch buffers of a host-streamed block are not in
   that budget, and there is not a gigabyte left for them.

Nothing in `fastdf`'s own tests could see this: it takes a second tensor
built *between* two calls.  The fix is the one that does not need to predict
what will be built next: `fastdf.get_jk` catches an out-of-memory error,
gives back every resident prefix (for good; it is not rebuilt) and every
tile buffer (to be re-sized from the memory there is now) that any DF object
holds, collects the failed call's frames, frees the pool, and runs the call
again.  Nothing else is retried, and a call that runs out of memory with
nothing to give back raises as before.  `stats()['resident_dropped']` and
`['tiles_dropped']` count it.  A first version gave back only the prefix; the
retry then failed the same way with 15.5 GB sitting in the pool and the tile
buffer still sized for the first call, which is why both go.  With it, the patched two-tensor run
at 284 atoms runs: 15.95 s per cycle (16.57 s mean over three, the first
paying the failed attempt), against GPU4PySCF's 24.64.  It is slower than it
would be with its prefix — the full-range tensor is now streamed whole — which
is one more reason §3's one tensor is the better shape at this size.

One more thing turned up on the way: `DF.range_coulomb` makes the long-range
object with `self.copy()`, which hands it the full-range object's
`_fastdf_resident` entry.  A device-resident tensor never reads that entry, so
the copy would keep the other tensor's prefix alive for the rest of the SCF
even after it had been given back.  `fastdf` now clears the entry on any
device-resident object it is called with.

---

## 5. Saying it

`electrolites/fastdfcderi.py:_host_preflight`, before every host-resident
tensor is allocated, at `NOTE` level:

```
cderi (full-range) 13365 x 808156 fp64 = 86.41 GB on the host of the node's 269.7 GB (MemAvailable 246.0 GB)
cderi (omega=0.25) 4091 x 795868 fp64 = 26.05 GB on the host; with the 86.41 GB already alive in this process, 112.46 GB of the node's 269.7 GB (MemAvailable 158.5 GB)
```

`electrolites/_dfpin.py` now counts the page-locked tensors that are alive
(`live_bytes()`, `stats()['live_gb']`) — which is how the second line knows
about the first.  Two consequences:

* A tensor larger than `MemAvailable` plus what the pool could give back
  raises `MemoryError` with that line and a pointer to `FASTDF_FP32=1`,
  instead of `cudaHostRegister` pinning past the node's memory and the kernel
  killing the process in the middle of the store.  One that fits but leaves
  less than the pool's reserve (8 GB or a tenth of the node) warns.
* Before registering a new buffer, the pool evicts retained ones, oldest
  first, until the new one leaves the reserve free
  (`stats()['evicted_for_room']`).  Before this, a pool holding one
  calculation's buffers would sit on memory a different calculation's build
  then needed.

`tests/test_dfrsh.py` §3 checks all four: the note is recorded for both
tensors, the live count is exactly the two tensors' bytes and returns to zero
when they go, 1 MB of reported host memory raises `MemoryError`, and a pooled
buffer is evicted when the reported memory is under the reserve.

---

## 6. Measured and not done

* **The long-range metric's eigen-decomposition** is 3.6 s of the 11.8 s
  long-range build at 284 atoms (`cp.linalg.eigh` on 13 365²; `syevdx`
  asking only for `w > 1e-7` is 8 % faster, because the tridiagonal reduction
  is the cost).  A randomized range finder on a subspace of half the order,
  one power iteration, is 3x on taxol's metric and keeps the same 1 691 modes,
  but it changes the fitted subspace at the 1e-7 relative level, and the
  whole stage is 1 % of a 284-atom SCF.  Not done.
* **Tighter truncation of the long-range tensor** is not worth it: §1.1's
  curve is smooth, with no knee to sit at.
* **The fused gradient.**  Under `FASTDFRSH=fuse` the gradient and the
  response still use two tensors (the gradient resets the DF object and
  builds its own), so they are the derivative of an energy 1e-7–1e-6 Eh away
  from the one the SCF reported.  The module says so once, at `NOTE` level.
  The long-range-only path (§2) has no such caveat.

---

## 7. Is it general

* **Size**: 42, 60, 113 and 284 atoms, both functionals.  The long-range-only
  win grows with size (1.13x → 1.63x on a whole SCF, 2.50x → 2.62x per
  cycle) because what it removes, a whole exchange pass, is a larger share of
  the cycle.
* **Class**: a UKS Cu(II) quartet in the SCF table, UKS radicals in the
  tests, Fe(III) and Pt(II) (behind an ECP) in the rank table, and def2-TZVP
  in the tests.
* **ω**: the long-range tensor's share is set by ω (§1.1), which is why the
  capacity story differs between wB97X-D3 (ω = 0.25, 26 GB, fits on the card)
  and ωB97 (ω = 0.4, 53 GB, does not).
* **Which path a run takes** is read off `fastdfrsh.stats()` — `lr_only`,
  `fused`, `declined_reasons` — and a declined call is upstream's code.

Reproduce:

```bash
python tests/test_dfrsh.py [--basis def2-tzvp]
python benchmarks/df_rsh_rank_probe.py --xyz .../penicillinV.xyz --basis def2-svp \
    --xc hyb_gga_xc_wb97x_d3                       # §1.1, per-mode energies
python benchmarks/df_rsh_rank_probe.py --xyz .../PfPMT.xyz --charge -1 \
    --basis def2-svp --xc hyb_gga_xc_wb97x_d3 --spectrum-only   # no SCF
FASTDFRSH=off|auto|fuse python benchmarks/df_rsh_bench.py --xyz ... \
    --basis def2-svp --xc hyb_gga_xc_wb97 --patch all --veff-reps 3   # §2, §3.3
FASTDFRSH=... python benchmarks/df_rsh_bench.py --xyz ... --scf [--no-df]
```

`pyscf`'s `wb97x-d3` string is parsed as wB97X plus a D3 term it does not
support; the functional itself is `hyb_gga_xc_wb97x_d3` (ω = 0.25,
`hyb` = 0.195728), and the dispersion term does not touch the fitting.
