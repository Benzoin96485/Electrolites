# Comparing a GPU feature with PySCF on the CPU: the protocol

**What this is**: the rules every GPU-against-CPU number in this repository
follows, written before any is taken.  `docs/FEATURE_TARGETS_PYSCF_2.13.md`
§0 rule 1 says which features are measured against PySCF on the CPU rather
than against GPU4PySCF.  This file says *how*: the machine, the software, the
threads, the problem definition, the correctness gate, the timing and the way
a ratio is written down.  `benchmarks/cpu_baseline.py` is the harness that
implements it, and `~/vibeqc/_el_jobs/phase0/` holds the first runs' scripts
(§8).

Why it is needed: the LLM-GPU-QM landscape review found that published
GPU-against-CPU speedups (30x over 32 cores, 60x over 100 logical cores, 52x
over 128 cores) cannot be compared with each other.  None of them states its
CPU side in the headline, and several used a CPU code that was not the best
available.  A ratio here that does not carry its CPU side is not a result.

---

## 1. When which baseline

| the GPU feature … | is measured against | example |
|---|---|---|
| has **no GPU path** today (FEATURE_TARGETS status absent / stub / host) | **PySCF 2.13 on a CPU node**, by this protocol | stability analysis, CCSD(T), CASSCF, NEVPT2, MC-PDFT, GW |
| is an **alternative route** to something already on the GPU | **the best existing GPU path under `patch_all()`**, on the same GPU, by KT's rules (separate processes, back to back) | COSX against exact K (`fastk`); DF-CASSCF integrals against direct ones |
| replaces a GPU4PySCF kernel (everything before this phase) | **GPU4PySCF**, as in KT | — |

The three kinds of ratio are **never pooled**, averaged or put in one table
column.  For the second kind the CPU number is still taken (it says whether
the GPU route is needed at all) and is reported as context, not as the
speedup.

## 2. The machine

**CPU side**: one whole Perlmutter **CPU node**, 2 × AMD EPYC 7763 (64 cores
each, 128 cores, 256 hardware threads), 512 GB, allocated exclusively:

    salloc -N 1 -C cpu -q interactive -A m4787 -t 04:00:00 -L cfs \
        srun -n 1 -c 256 --cpu-bind=cores <script>

**`srun` is not optional.** `salloc` given a script runs that script on the
**login node**: the first phase-0 launch did exactly that and its numbers
were discarded (FEATURE_TARGETS §11.0). `phase0/cpu_env.sh` and
`gpu_env.sh` refuse to run on a host called `login*`. The interactive QOS
allows **two** jobs per user, so longer series are chained one allocation
after another (`phase0/lane.sh`). Do not `pkill -f` or `pgrep -f` a pattern
on the login node: it matches its own shell, and twice it revoked an
allocation.

**GPU side**: one A100 of a Perlmutter GPU node, with `patch_all()`, by
`benchmarks/path_anatomy.py` (or the feature's own harness). The A100 is
40 GB unless the row says 80 GB. The other three GPUs either sit idle or run
other measurements of the same kind (KT's practice); which of the two is
stated.

The GPU node's own 64-core CPU is **not** the CPU baseline. If it is measured
too (it is what a user of that node has), it is a separate, labelled row.

## 3. Software, threads, memory

- **Environment**: `barw-dev` (`~/vibeqc/_el_jobs/env.sh`), the same Python
  stack on both sides. PySCF **2.13.0 as the pip wheel ships it**.
- **Two BLAS libraries are involved, and both are recorded.** NumPy uses
  conda's OpenBLAS 0.3.30 (pthreads). PySCF's C extensions (`libcc`,
  `libfci`, `libao2mo`, `libnp_helper`, …) load the **OpenBLAS 0.3.3 bundled
  in the wheel**, together with its own `libgomp`. On the login node
  threadpoolctl reports that bundled copy at 1 thread, by design: PySCF's C
  code threads with OpenMP around single-threaded BLAS calls. So the wheel is
  the baseline every PySCF user gets, but it may not be the fastest PySCF
  this CPU can run.
  - Whenever PySCF's own C BLAS calls dominate the CPU run — the (T) kernel,
    the FCI σ, `ao2mo` — a ratio is labelled **"vs PySCF 2.13 wheel"**.
  - A one-off comparison with a source build against a current BLAS (AOCL or
    OpenBLAS 0.3.30) is owed for each such family before its ratio is quoted
    in a summary.
- **Threads**: `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS` = the family's count
  (below), and **no `OMP_PROC_BIND` / `OMP_PLACES`**: binding OpenMP's master
  thread pins NumPy's OpenBLAS pthreads, which inherit its affinity, to one
  core (the harness then reports 2 visible CPUs instead of 256). Process
  binding is `srun --cpu-bind=cores`'s alone, and the harness's
  `cpus_visible` must read 256.
  - The count is **chosen by measurement once per method family** (SCF/COSX,
    CC, CASSCF): 32, 64 and 128 threads (`phase0/item0_threads_cpu.sh`),
    keeping the fastest.
  - The choice and the scaling are written next to the first ratio of that
    family. A baseline that is slower at 128 threads than at 64 is reported at
    64.
  - **Measured (FEATURE_TARGETS §11.0)**:

    | family | threads | why |
    |---|---|---|
    | CC | fastest per system | 64 for glycine/TZVP and alanine dipeptide/SVP, 128 for ferrocene/SVP; CC rows run all three counts |
    | SCF, COSX, stability | 128 | |
    | DF-CASSCF | 32 | flat to 128; its HDF5 `_ERIS` gets slower with more threads |
- **Memory**: `max_memory = 400 GB` (`PYSCF_MAX_MEMORY`), so PySCF takes its
  in-core paths wherever a user on this node would. PySCF's 4 GB default would
  push the larger runs out of core and slow the baseline artificially. The
  harness records whether the SCF went in-core (`eri_incore`).
  `PYSCF_TMPDIR` points at home/CFS, so the run survives a `/pscratch` outage.

## 4. The same problem on both sides

Both sides take:

- **Geometry**: the same geometry file from `benchmarks/molecules/`, and the
  same charge and spin.
- **Basis and fitting**: the same basis, auxiliary basis, functional and
  grid level. The DFT grids at a given level are identical: the vSAP test
  reproduces PySCF's matrix to 1e-14 on GPU4PySCF's grid.
- **Method settings**: the same `conv_tol` (1e-9 for the SCF unless stated)
  and the same frozen core (all-electron unless stated). For CASSCF, the same
  CAS, number of roots, weights and reference orbitals (ROHF for an open
  shell).
- **Initial guess**: the same.
- **Convergence**: both cycle counts are reported. A ratio between runs that
  took different numbers of cycles is written with both counts beside it,
  and **per-cycle** times are given as well.
- **Precision**: fp64 everywhere. An approximate or reduced-precision GPU
  option is a separate row with its error (FEATURE_TARGETS §0 rule 2).

## 5. Correctness gate, before any timing

The GPU result agrees with the CPU run **of the same row** to the
FEATURE_TARGETS §6 tolerance. Examples:

| result | tolerance |
|---|---|
| SCF energies | 1e-8 Eh (GPU and libcint integrals differ at ~1e-10; measured 6e-10 to 7e-9) |
| independently converged CCSD correlation energies, `conv_tol` 1e-9 and PySCF's default `conv_tol_normt` 1e-5 on **both** sides | **1e-7 Eh**: the amplitude tolerance sets the floor (measured 2.7e-9 to 3.6e-8). A run at PySCF's default `conv_tol` 1e-7 is not comparable (1.7e-6 off on ferrocene) |
| (T) on the same amplitudes | 1e-10 Eh |
| CASSCF energies | 1e-9 Eh |
| gradients | 1e-6 Eh/bohr |
| eigenvalues | 1e-8 |

The difference is printed in the table row. A row that fails the gate is not
timed: it is a bug report.

## 6. Timing

- **What is timed**: the feature's own step (e.g. "(T)", "CASSCF",
  "stability") **and** the whole run (SCF + the step), both in one cold
  process, without the interpreter start or the imports.
  - The GPU side includes its first-call compilation unless the row says
    "warm". `electrolites/warmup.py` and the kernel cache make the warm number
    meaningful.
  - Both are reported, because a short step can be dominated by one-time
    costs.
- **Stages**: `cpu_baseline.py` and `path_anatomy.py` wrap the stages, so a
  ratio can be broken down. `..` rows are sub-intervals of their parent.
- **Repetitions**: the CPU side runs once per row when the timed step exceeds
  60 s, else min of 3. The GPU side follows KT (min of 3, separate
  processes). Variation between repeated CPU runs is recorded once per
  family.
- **Records**: `cpu_baseline.py --json` keeps every run with the full
  environment:
  - the node and its CPU model;
  - the cores visible;
  - the OpenMP, PySCF and both BLAS thread counts and libraries;
  - versions;
  - peak RSS;
  - stage times and results.

  Records are kept with the job logs (`~/vibeqc/_el_jobs/phase0/results/`).
  The tables in `docs/` quote them, giving the file name.

## 7. How a ratio is written

Always as the two sides, then the ratio:

> CCSD(T) step, alanine dipeptide/def2-SVP: GPU 41.2 s (1 × A100-40GB,
> `patch_all()`, cold) against CPU 612 s (PySCF 2.13 wheel, 2 × EPYC 7763,
> 128 threads), **14.9x**; |ΔE| 3e-11 Eh.

**Never**:

- a ratio without its CPU line;
- a GPU-against-CPU ratio in a column of GPU-against-GPU ratios;
- a "speedup" for an alternative route measured against the CPU.

Every acceptance (FEATURE_TARGETS §9) needs a second size and a second class.
The closed-shell transition-metal case for restricted-only methods is
`benchmarks/molecules/tmc_ferrocene.xyz`. Its geometry is idealised (D5h,
built from textbook parameters) because every complex in
`tmc_cases.txt` is open-shell.

## 8. The first runs (FEATURE_TARGETS §8 items 1–4), prepared

Scripts in `~/vibeqc/_el_jobs/phase0/`. Each one fits a 4-hour interactive
allocation, and each writes `logs/` and `results/` there.

| script | allocation | what | FEATURE_TARGETS |
|---|---|---|---|
| `item0_threads_cpu.sh` | CPU node, ~1 h | thread scaling 32/64/128 for CC, COSX, DF-CASSCF | §3 here |
| `item1a_cosx_cpu.sh`, `item1b_cosx_cpu.sh` | CPU node, ~3 h + ~4 h | PySCF SGX (`pjs`, level 2): penicillin V at def2-SVP/TZVP/TZVPD, Ni dithiolene⁻ UKS at SVP/TZVP, taxol at SVP/TZVP; one exact-K CPU SCF | item 1 |
| `item1_cosx_gpu.sh` | GPU node, 4 GPUs | exact-K SCF with `patch_all()` on the same systems (plus taxol/TZVPD); also SGX's error reference | item 1 |
| `item2_ccsdt_cpu.sh` / `item2_ccsd_gpu.sh` | CPU node ~2 h / GPU node | CCSD and (T) split on the CPU; `fastccsd`'s CCSD on the GPU: glycine/TZVP, alanine dipeptide/SVP, ferrocene/SVP | item 2 |
| `item3_casscf_cpu.sh` | CPU node, ~3 h | DF-CASSCF(10,9) and (14,13) on Ni dithiolene⁻ and the Cu macrocycle, and SA-5 on Ni: `_ERIS`, AH J/K, FCI, rest | item 3 |
| `item4_stability_cpu.sh` / `item4_scf_gpu.sh` | CPU node ~3 h / GPU node | UKS internal + external stability on Ni dithiolene⁻ and the Co sextet; the GPU UKS SCF they follow | item 4 |

Launch, e.g.:

    salloc -N 1 -C cpu -q interactive -A m4787   -t 04:00:00 -L cfs srun -n 1 -c 256 --cpu-bind=cores ~/vibeqc/_el_jobs/phase0/item2_ccsdt_cpu.sh
    salloc -N 1 -C gpu -q interactive -A m4787_g -t 01:00:00 -L cfs --gpus 4 srun -n 1 -c 128 --gpus 4 ~/vibeqc/_el_jobs/phase0/gpu_all.sh
    ./lane.sh <job id to wait for, or 0> item1a_cosx_cpu,04:00:00,128 item3a_casscf_cpu,04:00:00,32

`THREADS=` (or the spec's third field) sets the thread count.

**Run (2026-10-13/14); results in FEATURE_TARGETS §11.** Two deviations
from the table:

- `item3_casscf_cpu.sh` became `item3a` (Ni) and `item3b` (Cu). The Cu
  CAS(14,13) run was dropped, because it would not fit one allocation.
- The GPU references were re-run where the first pass was not comparable
  (`gpu_rerun.sh`, `gpu_rerun_tax.sh`):
  - the Co UKS and taxol/TZVPD at `max_cycle` 150;
  - the CCSDs at `conv_tol` 1e-9.
