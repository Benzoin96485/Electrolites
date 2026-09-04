#!/bin/bash
# Regenerate fastk_generated.cu from GPU4PySCF's unrolled_rys_k.cu.
set -e
SRC=${1:?"usage: $0 <path to the gpu4pyscf source this lifts>"}
CLASSES=${2:-1000,1010,1100,2000,1011,1110,2010,2100,2110,1111,2011,2020,2200,3000,3010,3020,3100,3200}
cd "$(dirname "$0")"
KERNELS=${KERNELS:-"$(cd "$(dirname "$0")/../electrolites/kernels" && pwd)"}
python gen_kernels.py "$SRC" "$CLASSES" "${@:3}" > "$KERNELS/fastk_generated.cu"
echo "wrote $KERNELS/fastk_generated.cu ($(wc -l < "$KERNELS/fastk_generated.cu") lines)"
