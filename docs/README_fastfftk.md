# fastfftk — FFTDF k-point exchange with time reversal (J3)

`pbc/df/fft_jk.py:get_k_kpts`, the exchange of every k-point hybrid on the
default FFTDF back end, is 94–97 % of such an SCF
(`docs/KERNEL_TARGETS_1.8.1.md` §10.11).  It loops over every pair of
k-points `(k1, k2)` and, per pair, over the AOs of `k1` in blocks of 32:
pair densities with the occupied orbitals of `k2`, an FFT, the Coulomb
kernel, an inverse FFT, a contraction — `nao × nocc` FFTs per pair, `nk²`
pairs.

## Time reversal

With real basis functions the Bloch AOs satisfy `phi_{-k} = conj(phi_k)`
(a reciprocal lattice vector leaves a Bloch sum unchanged), so when the
k-point set is closed under `k → -k` and every density satisfies
`dm(-k) = conj(dm(k))`, `K(-k) = conj(K(k))`.  `fastfftk.get_k_kpts` is
GPU4PySCF's function building only one `k1` of each pair and conjugating
for the other; both conditions are checked on every call (the densities to
1e-10 of their largest element) and the full loop runs otherwise, and
always with `kpts_band`.

It saves `(nk - n_TRIM)/2` of the `nk` outer iterations: nothing on a
Gamma-centred 2×2×2 mesh (all eight points are their own partners),
14/27 built on 3×3×3, 36/64 on 4×4×4.

Sizing the AO block from free memory instead of 32 was measured (1.00x —
the per-pair FFTs are already large enough) and dropped.

## Result

`benchmarks/fftk_bench.py`, one `get_k_kpts` on a converged KRKS/PBE
density, diamond gth-dzvp/gth-pade:

| | GPU4PySCF | `fastfftk` | |
|---|---|---|---|
| 2-atom, 3×3×3 | 22.84 s | 10.99 s | 2.08x |
| 2-atom, 4×4×4 | 54.84 s | 30.88 s | 1.78x |
| 8-atom, 3×3×1 | 48.55 s | 26.89 s | 1.81x |
| 2-atom / 8-atom, 2×2×2 | | | 1.00x (all points time-reversal invariant) |

Whole PBE0 SCF, 2-atom 3×3×3, `--patch all` against `FASTFFTK_OFF=1`
(run side by side): **149.47 → 78.84 s (1.90x)**, `get_k_kpts`
142.65 → 72.51 s, the same energy to 7e-15 Eh.

## Correctness

`tests/test_fftk.py`: against `_get_k_kpts_orig` on 3×3×3, 3×3×1, 2×2×2,
a shifted 2×2×2 mesh (no partners — falls back), orbital-tagged and
untagged densities, two spin densities, a density that breaks time reversal
(falls back) and `kpts_band` (falls back): ≤ 2.7e-15.

## What is left

The exchange is FFT-bound: `nao × nocc` FFTs per k-point pair.  Applying K
only to the occupied orbitals and building the adaptively compressed
exchange operator (Lin, *J. Chem. Theory Comput.* **12**, 2242 (2016)) needs
`nocc × nocc` — 6.5x fewer on 8-atom diamond/gth-dzvp — and leaves the
occupied-space action, the energy and the SCF fixed point unchanged, but it
changes the virtual orbital energies and the AO-basis K that callers see,
so it is a different contract rather than a drop-in replacement.  Not built.
