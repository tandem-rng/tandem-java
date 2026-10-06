# Tests

```sh
pixi run test        # mvn -B verify, JDK 25
pixi run test-native # builds libtandem, then the suite in Java and again through it
cd cuda && pixi run test && pixi run gpu-test    # gpu-test needs a GPU host
```

## Suite

- `ConformanceTest` checks the spec's conformance files and every item of its
  `conformance/CHECKLIST.md`.
- `VectorsTest` checks every spec vector, generated from `vectors.json` by `tools/gen_vectors.py`.
- `StreamsTest` checks stream dumps in `src/test/resources/data`, copied from `tandem-c`.
- `FillsTest` checks fills against scalar draws at cut offsets. `TandemCudaTest` checks GPU fills.
- `NativeFillsTest` checks that every array fill of 512 elements or more, which runs in libtandem
  when it is loaded, equals the same fill in Java pieces in values and end position.

`ConformanceTest` follows `conformance/CHECKLIST.md` at tandem-spec `b31af72` and reads the
copies in `src/test/resources/conformance`. It checks every case of
`fill_below.json`, `normal.json`, `exponential.json` and `choice.json` in values and end
position, bounded fallbacks and normal misses included, and the scalar draws of `below.json` and
of every normal, exponential and choice case. It checks that a start one draw later shifts the
fallbacks by one element, the draw width from the range, the seven empty fills, the odd float
normal fills and the Box-Muller pairs, and that each case cut at elements 1, 7, 20, 21 and
`n - 1` and filled in pieces equals the whole fill. A float normal fill writes whole pairs, so it
is cut at elements 2, 8, 20 and the largest even element below `n`. It builds the alias table of each `vectors.json` choice case and
compares the capacity, cuts and aliases, and rejects invalid weights. It checks the SHA-256 of
the 12 stream dumps of `hashes.json`, the hashes of the long normal and exponential dumps, the
imaginary part of a complex draw across a block boundary, and starts below 2^63 with a draw past
it. The item on a fill that reaches 2^64 does not apply: starts lie below 2^63 and a fill holds
at most 2^31 elements, so a fill cannot reach 2^64. Every value compares bit for bit, the float normals and exponentials too.

`DerivedTest` checks the bounded draws of the `RandomGenerator` interface against `below.json`,
and the float normal formula. `ChoiceTest` checks a chi-square statistic of 1e6 choices
against the weights, and that a power-of-two scale of the weights keeps the table. `FillsTest`
compares every fill with the same count of scalar draws at offsets and lengths that cut rows,
blocks and chunks, and the block cache with random access at every chunk length.
`GaussianTest` checks that scalar normals, pair calls and fills give one sequence, that a double
fill cut at any element, also at a miss, equals the whole fill, that an empty double fill aligns
the position, and the kept float half. `ExponentialTest` checks fills cut at block boundaries
from an unaligned start, empty fills, and the moments to fourth order and the
Kolmogorov-Smirnov distance of 1e7 samples against Exp(1). `TandemTest` covers child keys,
forks, serialization and the `RandomGenerator` contract. `TandemProviderTest` covers the
Commons RNG adaptor.

`TandemCudaTest`, tagged `gpu`, compares the four uniform fills with the CPU fills in values and
final position at chunk lengths 1, 2, 4, 8, 32 and 65536 and at starts and lengths that cut words,
blocks, rows and groups. It compares the bounded fills with the CPU fills at ranges that reject
often, rarely and never, and the normal fills with the CPU fills at every start slot, double
normals bit for bit at lengths that take the fused kernel and the two-kernel path at every start
modulo 4 draws, which covers the octet stores. It compares both exponential fills with the CPU
fills bit for bit. It checks the bounded and float normal fixtures of `tandem-cuda`
(`cross_fill_below.h`, `cross_fill_normal.h`) and the double normal and exponential fixtures of
`tandem-c` (`cross_normal.h`, `cross_exponential.h`) bit for bit with end positions, the stream dumps of the core tests, and the device fills against the
array fills. `NoDriverTest` checks the error without a driver.

## Fixtures

`src/test/resources/conformance` holds byte-identical copies of the JSON files in tandem-spec's
`conformance/` at commit `b31af72`. `Vectors.java` is generated from the spec repository's
`vectors.json` by `tools/gen_vectors.py`. The stream dumps in `src/test/resources/data` are
byte-identical copies of the dumps in `tandem-c`.
`cross_fill_below.h` and `cross_fill_normal.h` of `tandem-cuda`, and `cross_normal.h` and
`cross_exponential.h` of `tandem-c` at commit `121db59`, are copied into the GPU test resources by
`cuda/tools/build_ptx.sh`. `tools/gen_zig_tables.py` generates `ZigTables.java` from the spec's
`tables/normal_f64_zig1024.json`.

## CI

- CI runs JDK 25 and 27 on Linux and macOS.
- Each CI test job builds libtandem from the tandem-c commit pinned in `tools/build_native.sh`
  and runs the suite in Java, again with the library named by `tandem.native`, and
  `NativeFillsTest` with it on `java.library.path`.
- CI fails when `Vectors.java` or `ZigTables.java` differ from the spec's data, when the
  conformance copies differ from tandem-spec `b31af72`, or when the PTX or its fixtures differ
  from what the tandem-cuda commit pinned in `cuda/tools/build_ptx.sh` gives.
