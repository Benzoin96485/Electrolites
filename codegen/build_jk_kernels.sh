#!/bin/bash
# Regenerate the *fused* Coulomb-and-exchange kernels (catalogue B4b) from
# GPU4PySCF's unrolled_rys_jk.cu -- the build its response path asks for.
#
# Two of the three families here are lifts and so need a GPU4PySCF checkout:
# the nineteen classes with one thread per shell quartet and the six whose gout
# is spread over threadIdx.y.  Their output is tracked, like fastk's.  The
# third family (the forty-five classes gen_khigh.py writes from the angular
# momenta alone) needs no source and is produced on demand by
# electrolites/_gen.py; `--table` here only refreshes its launch table.
set -e
SRC=${1:?"usage: $0 <path to gvhf-rys/unrolled_rys_jk.cu>"}
HERE="$(cd "$(dirname "$0")" && pwd)"
KERNELS=${KERNELS:-"$(cd "$HERE/../electrolites/kernels" && pwd)"}
GEN=${GEN:-"$(cd "$HERE/../electrolites/codegen" && pwd)"}
# the classes GPU4PySCF unrolls with gout_stride == 1, plus (ss|ss), which the
# fused family lifts rather than hand-writing
CLASSES=${2:-0000,1000,1010,1100,2000,1011,1110,2010,2100,2110,1111,2011,2020,2200,3000,3010,3020,3100,3200}

python "$HERE/gen_kernels.py" --jk "$SRC" "$CLASSES" "${@:3}" \
    > "$KERNELS/fastjk_generated.cu"
mv "$HERE/fastjk_launch.json" "$KERNELS/fastjk_launch.json"

python "$HERE/gen_k2_kernels.py" --jk "$SRC" "${@:3}" \
    > "$KERNELS/fastjk2_generated.cu"
mv "$HERE/fastjk2_launch.json" "$KERNELS/fastjk2_launch.json"

python "$GEN/gen_khigh.py" --jk --table "$KERNELS/fastjkhigh_launch.json" \
    > /dev/null

for f in fastjk_generated.cu fastjk2_generated.cu; do
    echo "wrote $KERNELS/$f ($(wc -l < "$KERNELS/$f") lines)"
done
echo "refreshed $KERNELS/fastjkhigh_launch.json"
