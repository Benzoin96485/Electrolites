#!/bin/bash
# Regenerate fastj_generated.cu (and the launch-bound sweep variants) from
# GPU4PySCF's unrolled_md_j.cu.
set -e
SRC=${1:?"usage: $0 <path to the gpu4pyscf source this lifts>"}
cd "$(dirname "$0")"
KERNELS=${KERNELS:-"$(cd "$(dirname "$0")/../electrolites/kernels" && pwd)"}
python gen_j_kernels.py "$SRC" "${@:2}" > "$KERNELS/fastj_generated.cu"
python gen_j_kernels.py "$SRC" --variants "${@:2}" > fastj_variants.cu
echo "wrote $KERNELS/fastj_generated.cu ($(wc -l < "$KERNELS/fastj_generated.cu") lines)"
