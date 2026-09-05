#!/bin/bash
# Regenerate fastejk_generated.cu: the Rys nuclear-gradient two-electron
# kernels, written from the angular-momentum class alone (nothing is taken
# from GPU4PySCF's source).  Covers every class an spdf basis reaches.
set -e
cd "$(dirname "$0")"
KERNELS=${KERNELS:-"$(cd "$(dirname "$0")/../electrolites/kernels" && pwd)"}
# these two generators moved into the package (electrolites/codegen); their
# output is generated on demand by electrolites/_gen.py and is not tracked
GEN=${GEN:-"$(cd "$(dirname "$0")/../electrolites/codegen" && pwd)"}
CLASSES=${1:-$(python - <<'PY'
print(','.join(f'{i}{j}{k}{l}' for i in range(4) for j in range(i+1)
                for k in range(i+1) for l in range(k+1)))
PY
)}
python "$GEN/gen_ejk.py" "$CLASSES" --table "$KERNELS/fastejk_launch.json" "${@:2}" \
    > "$KERNELS/fastejk_generated.cu"
echo "wrote $KERNELS/fastejk_generated.cu ($(wc -l < "$KERNELS/fastejk_generated.cu") lines)"
