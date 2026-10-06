#!/bin/sh
# Rebuilds the committed PTX and test fixtures from the pinned tandem-cuda and tandem-c commits.
# Needs nvcc 12.8 and clang 19: `pixi run -e nvcc ptx` in cuda/. TANDEM_CUDA and TANDEM_C may
# name local checkouts that contain the commits, else the script clones them into target/.
set -eu
TANDEM_CUDA_COMMIT=2693c6342bab9cfc055b27bcb0aaa31dd935b299
# The normal and exponential fixtures of the reference, tandem-c.
TANDEM_C_COMMIT=121db5902d6136c7e5258970c0160121af3ab1d0

here=$(cd "$(dirname "$0")/.." && pwd)
repo=${TANDEM_CUDA:-$here/target/tandem-cuda}
[ -d "$repo/.git" ] || git clone -q https://github.com/tandem-rng/tandem-cuda "$repo"
git -C "$repo" cat-file -e "$TANDEM_CUDA_COMMIT^{commit}" 2>/dev/null || git -C "$repo" fetch -q origin
src=$here/target/tandem-cuda-$TANDEM_CUDA_COMMIT
rm -rf "$src"
mkdir -p "$src"
git -C "$repo" archive "$TANDEM_CUDA_COMMIT" tandem.cuh include tests/cross_fill_below.h tests/cross_fill_normal.h |
    tar -x -C "$src"
c=${TANDEM_C:-$here/target/tandem-c}
[ -d "$c/.git" ] || git clone -q https://github.com/tandem-rng/tandem-c "$c"
git -C "$c" cat-file -e "$TANDEM_C_COMMIT^{commit}" 2>/dev/null || git -C "$c" fetch -q origin
git -C "$c" archive "$TANDEM_C_COMMIT" tests/cross_normal.h tests/cross_exponential.h | tar -x -C "$src"

# conda activation prepends its own -ccbin, which nvcc would warn about.
export NVCC_PREPEND_FLAGS=
# sm_80 PTX JIT-compiles on every later GPU. Driver 570 (CUDA 12.8) reads PTX ISA 8.7 at most,
# which is what nvcc 12.8 emits.
out=$here/src/main/resources/io/github/tandemrng/cuda
mkdir -p "$out"
nvcc -ptx -arch=sm_80 -std=c++20 -O3 -ccbin clang++ -I"$src" -I"$src/include" \
    -o "$src/all.ptx" "$here/kernels/tandem_fills.cu"
# The host launchers of tandem.cuh instantiate its 42 template kernels, which would make the
# driver's first JIT of the module take 2.5 s instead of 1 s. Our entries are extern "C", so
# every mangled entry goes except the Float64 normal kernels of tandem.cuh, which TandemCuda
# launches by their mangled names.
# nvcc names the static device tables of normal_tables.hpp with hashes that differ between
# machines. Zeros of the same length keep the mangled names valid and the file reproducible.
awk '/^\t\/\/ \.globl\t_Z/ && !/fill_normal64_/ { skip = 1 } !skip { print } skip && /^}$/ { skip = 0 }' \
    "$src/all.ptx" |
    sed -E 's/_INTERNAL_[0-9a-f]{8}_([0-9]+_tandem_fills_cu)_[0-9a-f]{8}/_INTERNAL_00000000_\1_00000000/g' \
    > "$out/tandem_fills_sm80.ptx"
ptxas -arch=sm_80 "$out/tandem_fills_sm80.ptx" -o /dev/null

mkdir -p "$here/src/test/resources/cross"
cp "$src/tests/cross_fill_below.h" "$src/tests/cross_fill_normal.h" "$src/tests/cross_normal.h" \
    "$src/tests/cross_exponential.h" "$here/src/test/resources/cross/"
