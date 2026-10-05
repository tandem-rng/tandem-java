#!/usr/bin/env bash
# Builds libtandem, tandem-c's tandem.c at the pinned commit as a shared library, into
# target/native for the native fills. TANDEM_C_DIR names a local tandem-c checkout to build
# from instead of a clone; it must hold the pinned commit.
set -euo pipefail

TANDEM_C_COMMIT=d9e1e54a95392ad6d3352603ee9ca210de9e9fb8

cd "$(dirname "$0")/.."
out=target/native
mkdir -p "$out"
src=${TANDEM_C_DIR:-$out/tandem-c}
if [ ! -d "$src/.git" ]; then
    git clone -q https://github.com/tandem-rng/tandem-c "$src"
fi
git -C "$src" cat-file -e "$TANDEM_C_COMMIT^{commit}" 2>/dev/null || git -C "$src" fetch -q origin
git -C "$src" show "$TANDEM_C_COMMIT:tandem.c" > "$out/tandem.c"
git -C "$src" show "$TANDEM_C_COMMIT:tandem.h" > "$out/tandem.h"
git -C "$src" show "$TANDEM_C_COMMIT:tandem_normal_tables.h" > "$out/tandem_normal_tables.h"

case "$(uname -s)" in
    Darwin) lib=libtandem.dylib ;;
    *) lib=libtandem.so ;;
esac
# The flags of tandem-c's Makefile, so the library gives the bits its tests check.
${CC:-clang} -std=c23 -O2 -ffp-contract=off -fPIC -shared -o "$out/$lib" "$out/tandem.c" -lm
echo "$out/$lib"
