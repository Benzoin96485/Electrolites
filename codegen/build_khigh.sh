#!/bin/bash
# Regenerate fastkhigh_generated.cu: unrolled Rys exchange kernels written from
# the angular-momentum class alone.  Unlike the other two generators this one
# takes nothing from GPU4PySCF's source -- the 2D recurrences and the density
# contraction are emitted from (li,lj,lk,ll), so it covers the 40 classes an f
# basis reaches that GPU4PySCF has no unrolled kernel for, and it also turned
# out to beat the lifted kernels on five of the six classes gen_k2_kernels.py
# handles.
set -e
cd "$(dirname "$0")"
KERNELS=${KERNELS:-"$(cd "$(dirname "$0")/../electrolites/kernels" && pwd)"}
# Everything with l <= 3 except: the 19 classes gen_kernels.py runs with one
# thread per shell quartet (that design keeps the 2D integrals in registers and
# is still faster for them), and 2021, where GPU4PySCF's own unrolled kernel
# beats this one -- 0.87x, measured.
CLASSES=${1:-$(python - <<'PY'
KEEP = {'0000','1000','1010','1011','1100','1110','1111','2000','2010','2011',
        '2020','2100','2110','2200','3000','3010','3020','3100','3200','2021'}
print(','.join(f'{i}{j}{k}{l}' for i in range(4) for j in range(i+1)
                for k in range(i+1) for l in range(k+1)
                if f'{i}{j}{k}{l}' not in KEEP))
PY
)}
python gen_khigh.py "$CLASSES" --table "$KERNELS/fastkhigh_launch.json" "${@:2}" \
    > "$KERNELS/fastkhigh_generated.cu"
echo "wrote $KERNELS/fastkhigh_generated.cu ($(wc -l < "$KERNELS/fastkhigh_generated.cu") lines)"
