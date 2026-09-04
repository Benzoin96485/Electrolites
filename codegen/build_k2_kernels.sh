#!/bin/bash
# Regenerate fastk2_generated.cu (the classes GPU4PySCF splits across
# threadIdx.y) from GPU4PySCF's unrolled_rys_k.cu.
set -e
SRC=${1:?"usage: $0 <path to the gpu4pyscf source this lifts>"}
cd "$(dirname "$0")"
KERNELS=${KERNELS:-"$(cd "$(dirname "$0")/../electrolites/kernels" && pwd)"}
python gen_k2_kernels.py "$SRC" "${@:2}" > "$KERNELS/fastk2_generated.cu"
python gen_k2_kernels.py "$SRC" --variants "${@:2}" > fastk2_variants.cu
echo "wrote $KERNELS/fastk2_generated.cu ($(wc -l < "$KERNELS/fastk2_generated.cu") lines)"
