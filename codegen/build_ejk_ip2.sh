#!/bin/bash
# Regenerate the lifted second-derivative ERI kernels (catalogue B3) from a
# GPU4PySCF 1.8.x checkout.  Unlike the other lifts these are compiled by nvcc
# rather than NVRTC -- they are split over translation units and need
# relocatable device code -- so the tracked output is CUDA source that
# electrolites/_nvcc.py builds into libmyejk2.so on first use.
#
# The support headers are copied verbatim because the kernels bind to this
# release's struct layouts (BoundsInfo gained per-group q_cond/s_cond after
# 1.7.0, so the copies fastk uses are not interchangeable) and a wheel install
# carries no CUDA source.
set -e
SRC=${1:?"usage: $0 <path to gpu4pyscf/lib/gvhf-rys of a 1.8.x checkout>"}
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=${OUT:-"$(cd "$HERE/../electrolites/kernels/cuda" && pwd)"}
python "$HERE/gen_ejk_ip2.py" "$SRC" "$OUT" "${@:2}"
