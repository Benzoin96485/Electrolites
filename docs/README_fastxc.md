# fastxc — a faster exchange–correlation (Vxc) build for GPU4PySCF

On an A100-40GB, B3LYP/6-31G\* on these clusters spends about 19 % of the SCF
inside `numint.nr_rks` once the J and K builds have been sped up.  `fastxc`
replaces `nr_rks` and leaves the grid, the AO screening, the libxc call and the
symmetry factors alone.

```python
import sys; sys.path.insert(0, '.../side_test/HF/src')
import fastxc                    # patches gpu4pyscf.dft.numint.NumInt.nr_rks
```

## Results (A100-SXM4-40GB, B3LYP/6-31G\*, level-3 grids)

`bench/check_xc.py`, one `nr_rks` call at a representative SCF density:

| system | AOs | grid points | GPU4PySCF | fastxc | speedup | ΔV (relative) |
|---|---|---|---|---|---|---|
| `PfPMT_cluster` | 2268 | 3.40e6 | 711.8 ms | **306.7 ms** | **2.32×** | 3.4e-15 |
| `HcgC_cluster`  | 4170 | 6.45e6 | 2399.0 ms | **816.5 ms** | **2.94×** | 2.6e-15 |
| benzene | 96 | 1.44e5 | 23.8 ms | **15.9 ms** | 1.50× | 3.5e-15 |

Over the whole SCF the XC build goes from **12.97 s to 6.25 s** on `PfPMT`
(2.08×; the difference from the single-call figure is the first iteration,
which also builds the AO-screening cache).  The electron count and the XC
energy agree exactly, and the potential matrix to 3e-15 relative.

## Why the XC build is not flop-bound

The AO screening is already very effective: on `PfPMT` a 4096-point grid block
sees 324 of the 2268 AOs on average.  The whole XC build is then about 0.4
TFLOP of GEMM per iteration, which an A100 could do in ~30 ms — against the
711 ms it actually takes.  What costs the time is *shape*: the potential-matrix
GEMM for one block is 324×324 with an inner dimension of 4096, i.e. 36 CUTLASS
64×64 tiles for a 108-SM GPU.  Issued one grid block at a time, two thirds of
the machine is idle, and there are ~4000 such launches per iteration.

Shrinking the grid block does not help, because it makes the problem worse:
measured on `PfPMT`, `nr_rks` takes 1990 / 1081 / 916 / 866 ms at block sizes
1024 / 2048 / 4096 / 8192, even though the arithmetic falls from 0.46 to
0.33 TFLOP as the blocks get smaller and the AO masks tighter.

## What fastxc changes

1. **One pass over the grid instead of two.**  GPU4PySCF walks the grid once to
   build rho, calls libxc on the whole grid, then walks it again to contract
   the potential — so every AO and its three gradients are evaluated twice per
   SCF iteration.  `fastxc` does rho, libxc and the potential contraction in
   one pass.
2. **The GEMMs are batched across grid blocks.**  Grid blocks are accumulated
   until their AO values fill a fixed fraction of free memory (GPU4PySCF's own
   `grouped_block_loop`), and then one `grouped_gemm` builds rho for all of
   them and one `grouped_dot` builds all their potential blocks.  That is what
   fills the machine, and it is the largest of the three effects.
3. **The density is contracted with the density matrix, not the orbitals.**
   GPU4PySCF takes the orbital route whenever the caller passes a density
   matrix tagged with `mo_coeff`, which the SCF always does: rho then costs
   four GEMMs of `nocc*nao_sub*ngrids`.  With the density matrix it is one GEMM
   of `nao_sub^2*ngrids`, which is cheaper whenever `nao_sub < 4*nocc` — on
   these clusters by a factor of 6.7, because screening keeps `nao_sub` near
   320 while `nocc` is 546 and 1013.  The comparison is made from those two
   numbers per block, so the orbital route is still used where it wins (small
   or dense systems); forming the density matrix from the orbitals once costs
   `2*nao^2*nocc`, under a millisecond here.
4. **Nothing is copied back to the host inside the loop.**  The electron count
   and the XC energy accumulate in device scalars instead of forcing a
   synchronisation per grid block.

Items 1, 3 and 4 alone — a fused single pass without batching — are *slower*
than GPU4PySCF (0.54× on benzene), because calling libxc once per 4096-point
block replaces one large call with hundreds of small ones.  The batching in
item 2 is what makes the fusion pay: libxc is called once per group, which is
hundreds of blocks.

## Is any of this size-dependent?

| ingredient | depends on | handling |
|---|---|---|
| group size | free GPU memory | GPU4PySCF's own rule (AO values up to 0.2 of free memory); `fastxc` adds rho and the weighted AOs, which are another half of that. Not a tuned constant. |
| rho route | block AO count and `nocc` | compared per block; neither is the size of the molecule. |
| grid, block size, AO screening | GPU4PySCF's own | untouched: `MIN_BLK_SIZE`, the atomic grid grouping and the 1e-10 AO threshold are all as they were. |
| functional type | what the caller asked for | LDA and GGA are handled; meta-GGA, NLC, non-hermitian densities, multiple densities and multi-GPU fall through to GPU4PySCF. Spin-unrestricted DFT goes through `nr_uks`, which is not patched. |

The speedup *grows* with system size (1.50× / 2.32× / 2.94× at 96 / 2268 / 4170
AOs) because the batching has more blocks to work with and the per-launch
overhead it removes is a larger share of a bigger problem — the opposite of the
usual worry, but worth stating: the mechanism is not a small-system artefact.

## Files

| file | role |
|---|---|
| `fastxc.py` | the replacement `nr_rks`; `FASTXC_MODE=grouped\|perblock` and `FASTXC_RHO=auto\|dm\|mo` select the ablations. `FASTXC_RHO=mo` is for the ablation only: it is the route `auto` rejects on these systems, and forcing it makes the batched group hold four times as much scratch, which runs out of memory on `HcgC`. |
| `../bench/check_xc.py` | correctness and timing against GPU4PySCF, all mode/route combinations |
| `../bench/xc_explore.py` | the block-size and grid-ordering scan quoted above |
