#!/bin/sh
# Rebuilds the committed PTX and test fixtures from the pinned tandem-cuda commit. Needs nvcc
# 12.8 and clang 19: `pixi run -e nvcc ptx` in cuda/. TANDEM_CUDA may name a local checkout
# that contains the commit, else the script clones it into target/.
set -eu
TANDEM_CUDA_COMMIT=5806e517c0948b32102cf8c1614f85b7bdbdf757

here=$(cd "$(dirname "$0")/.." && pwd)
repo=${TANDEM_CUDA:-$here/target/tandem-cuda}
[ -d "$repo/.git" ] || git clone -q https://github.com/tandem-rng/tandem-cuda "$repo"
src=$here/target/tandem-cuda-$TANDEM_CUDA_COMMIT
rm -rf "$src"
mkdir -p "$src"
git -C "$repo" archive "$TANDEM_CUDA_COMMIT" tandem.cuh include tests/cross_fill_below.h tests/cross_fill_normal.h |
    tar -x -C "$src"

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
# every mangled entry goes.
awk '/^\t\/\/ \.globl\t_Z/ { skip = 1 } !skip { print } skip && /^}$/ { skip = 0 }' \
    "$src/all.ptx" > "$out/tandem_fills_sm80.ptx"
ptxas -arch=sm_80 "$out/tandem_fills_sm80.ptx" -o /dev/null

mkdir -p "$here/src/test/resources/cross"
cp "$src/tests/cross_fill_below.h" "$src/tests/cross_fill_normal.h" "$here/src/test/resources/cross/"
