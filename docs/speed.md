# Speed

`pixi run bench` produces the CPU figures, `pixi run bench-native` adds the row with libtandem,
and `pixi run bench` in `cuda/` gives the GPU figures.

## CPU

One thread, Apple M4 Pro, JDK 25.0.2 (Azul Zulu). Minimum of seven runs after three warm-up
runs, in GiB/s of output: 2^25 scalar draws at 4 bytes for `nextInt` and 8 for the others, and
arrays of 2^24 elements. Every row is the median of three such runs on 2026-10-05, in one
session.

| generator | nextInt GiB/s | nextLong GiB/s | nextDouble GiB/s | nextGaussian GiB/s | nextExponential GiB/s | int[] loop GiB/s | double[] loop GiB/s |
|---|---|---|---|---|---|---|---|
| Tandem | 3.30 | 4.93 | 4.50 | 3.03 | 2.78 | 3.26 | 4.41 |
| L64X128MixRandom | 3.25 | 6.71 | 6.61 | 2.19 | 2.11 | 3.28 | 6.54 |
| SplittableRandom | 8.59 | 15.29 | 12.16 | 3.31 | 3.21 | 9.23 | 15.17 |
| Random | 0.93 | 0.92 | 0.93 | 0.54 | 0.81 | 0.09 | 0.15 |

Scalar draws always run in Java. Array fills, in GiB/s, in Java and through `libtandem` (tandem-c
d9e1e54, built by `pixi run native`), both columns from one earlier session the same day:

| fill | Java | libtandem |
|---|---|---|
| `fill(int[])` | 9.37 | 18.64 |
| `fill(long[])` | 8.49 | 18.80 |
| `fill(float[])` | 5.44 | 16.05 |
| `fill(double[])` | 8.58 | 16.33 |
| `fillGaussian(double[])` | 3.97 | 7.58 |
| `fillGaussian(float[])` | 1.85 | 5.46 |
| `fillExponential(double[])` | 3.31 | 6.03 |
| `fillExponential(float[])` | 2.17 | 6.60 |

The libtandem column matches tandem-c's own figures. A fill of 512 elements or more takes the
library, which is faster than Java from that length on. Each call writes straight into the
Java array as a critical call, at most 2^16 elements at a time, so a long fill does not hold off
the garbage collector for its whole length. When the library loads, a fill of 300 longs and 300
normals is compared with the Java fill, and a library that disagrees is not used.

The Java double normal fill reads one table of 2048 pairs, the layer threshold and the width with
the sign of bit 10, by the 11 low bits of the draw. Its loop has no call and only records the
misses, which a second loop finishes. A miss draws its fallback by random access, F on locals,
and two draws share one lane.

Each generator runs in its own JVM. Arrays are filled by a loop of draws or by `Tandem.fill`.

A scalar draw costs in proportion to the bytes it takes, because every row is generated on
demand and generation dominates: a `long` takes twice the stream of an `int`. The cached block is
read through fields of the generator, with one unsigned range check per draw and the refill out of
line. Each lane of a row is independent, so the cache holds a block of 32 rows. Two lanes step
through the block together with their state in registers. Four lanes need 32 state words, more
than C2 keeps in the AArch64 registers, and spilled half the loop to the stack. The seeding
function runs two lanes at a time in the same way. `int[]`, `long[]` and `double[]` fills generate
whole blocks directly into the destination. The row loops end on `b != end`, which C2 does not
count: it unrolls counted loops with many xors twice, and the unrolled `int[]` loop spilled. A
double takes its high word through the floating-point units, `hi 2^-32 + (lo >>> 11) 2^-53`,
which is exact and leaves the integer units to the step. The JIT does not vectorise the row step,
so fills stay near 9 GiB/s on the M4 Pro against about 20 GiB/s in `tandem-c`.

The JIT does not vectorise the polynomial logarithm either: the loop runs at the same speed with
`-XX:-UseSuperWord`. So `fillExponential`
costs a uniform fill plus a scalar transform of about the same cost, while `tandem-c` runs the
transform on NEON or AVX2 lanes. That puts it under half of `tandem-c`.

Normals use no libm: the double ziggurat takes the tables of the spec and the polynomial
logarithm of `tandem.c` in explicit `Math.fma`, and the float Box-Muller its polynomials, so both
are bit identical to tandem-c on every JVM and compiler (tests check the hashes of 1e6 double
and 2e6 float normals at five positions).

The Vector API (`jdk.incubator.vector`, still incubating in JDK 25) is not used. It has no
widening or high-half 32-bit multiply, which the step needs twice, and C2 multiplies long lanes
through the general registers on NEON. With the high half from four 16-bit products, one
four-lane group of rows with its store reached 9.3 GiB/s on the M4 Pro without seeding or double
conversion, bound by the latency of that multiply chain. Two groups need more than the 32 NEON
registers and spill. Two-vector `selectFrom` is not intrinsified on NEON, two-vector `rearrange`
costs two `tbl` and a select, and a loop that also steps scalar lanes passes C2's inlining node
limit, after which the vectors are boxed. The scalar fill is as fast and needs no `--add-modules`.

## GPU

GPU: NVIDIA A100 40 GB (PCIe), driver 570.124, JDK 25, `pixi run bench` in `cuda/`. GiB/s written
by each device fill. Minimum of 21 samples of ten launches. The last column is the
[`tandem-cuda`](https://github.com/tandem-rng/tandem-cuda/blob/main/docs/speed.md) figure for
the same kernel when this table was measured.

| fill | 2^26 GiB/s | 2^28 GiB/s | tandem-cuda 2^28 GiB/s |
|---|---|---|---|
| `fillInts` | 1333 | 1385 | 1383 |
| `fillLongs` | 1367 | 1390 | 1395 |
| `fillFloats` | 1328 | 1381 | 1383 |
| `fillDoubles` | 1365 | 1392 | 1394 |
| `fillBelowU32`, range 1000 | 1288 | 1340 | 1336 |
| `fillBelowU64`, range 1000 | 1326 | 1356 | 1348 |
| `fillGaussianFloats` | 1140 | 1200 | 1290 |
| `fillGaussianDoubles` | 865 | 1016 | 1065 |

Array fills copy over PCIe: `fill(double[])` of 2^26 elements runs at 9.3 GiB/s. Keep data on the
device when a GPU consumes it.

GPU figures: each sample is ten launches and one synchronize, and the figure is the minimum of 21
samples after a half-second warm-up, GPU idle. The last column of the table above is the
`tandem-cuda` figure for the same kernel at 2^28 elements.

The uniform and bounded fills run at the card's memory bandwidth, as in `tandem-cuda`. The
`fillGaussianDoubles` row is the ziggurat, the kernels of `tandem.cuh` itself, measured on
2026-10-05 with the same method. The normal rows vary by about 6% from run to run. In one process, our float normal kernel and `tandem-cuda`'s
measured 1211 to 1281 GiB/s alike. The device fills of the API also allocate their memory on each
call. The array fills add the copy over PCIe into the Java heap: `fill(double[])` of 2^26 elements
runs at 9.3 GiB/s end to end, about 150 times slower than the device fill.
