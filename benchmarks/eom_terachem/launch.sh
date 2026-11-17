#!/bin/bash
# launch.sh GPU CASE...  (runs cases sequentially on one GPU)
export CUDA_VISIBLE_DEVICES=$1; shift
module unload cudatoolkit 2>/dev/null; module load cudatoolkit/12.9
source /global/common/software/m4787/miniconda/etc/profile.d/conda.sh; conda activate barw-dev
export PYTHONPATH=$(readlink -f "$(dirname "$0")/../.."):$PYTHONPATH
export OMP_NUM_THREADS=16
cd "$(dirname "$(readlink -f "$0")")"; mkdir -p logs
for c in "$@"; do python run_eom.py $c > logs/${c//\//__}.log 2>&1; done
