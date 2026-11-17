#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1; shift
module load cudatoolkit/12.9 2>/dev/null; source /global/common/software/m4787/miniconda/etc/profile.d/conda.sh; conda activate barw-dev
export PYTHONPATH=$(readlink -f "$(dirname "$0")/../..") OMP_NUM_THREADS=16
cd "$(dirname "$(readlink -f "$0")")"; mkdir -p logs/prof2
python prof_eom.py $1 > logs/prof2/${1//\//__}.log 2>&1
