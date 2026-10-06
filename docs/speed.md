# Speed

`pixi run bench` produces the CPU figures, `pixi run bench-native` adds the row with libtandem,
and `pixi run bench` in `cuda/` gives the GPU figures.

## CPU

One thread, Apple M4 Pro, JDK 25.0.2 (Azul Zulu), `pixi run bench-native`. Minimum of seven runs
after three warm-up runs, in GiB/s of output: 2^25 scalar draws at 4 bytes for `nextInt` and 8
for the others, and arrays of 2^24 elements. Both tables are the median of three such runs in
one session on 2026-10-06, baselines included.

| generator | nextInt GiB/s | nextLong GiB/s | nextDouble GiB/s | nextGaussian GiB/s | nextExponential GiB/s | int[] loop GiB/s | double[] loop GiB/s |
|---|---|---|---|---|---|---|---|
| Tandem | 3.32 | 5.14 | 4.64 | 3.13 | 2.83 | 3.35 | 4.68 |
| L64X128MixRandom | 3.26 | 6.70 | 6.59 | 2.18 | 2.08 | 3.28 | 6.56 |
| SplittableRandom | 8.59 | 15.21 | 12.12 | 3.30 | 3.17 | 9.28 | 14.96 |
| Random | 0.93 | 0.93 | 0.93 | 0.54 | 0.80 | 0.07 | 0.13 |

Scalar draws always run in Java. Array fills, in GiB/s, in Java and through `libtandem` (tandem-c
d9e1e54, built by `pixi run native`). The baseline is a loop of the matching `SplittableRandom`
draw into the array, `(float)` cast for the float normal and exponential. `SplittableRandom` is
the fastest JDK generator in every column of the table above.

| fill | Java | libtandem | `SplittableRandom` loop |
|---|---|---|---|
| `fill(int[])` | 10.04 | 18.75 | 9.18 |
| `fill(long[])` | 9.38 | 18.87 | 16.48 |
| `fill(float[])` | 8.52 | 16.16 | 8.30 |
| `fill(double[])` | 8.95 | 16.36 | 15.17 |
| `fillGaussian(double[])` | 4.48 | 7.63 | 3.20 |
| `fillGaussian(float[])` | 2.11 | 5.48 | 1.64 |
| `fillExponential(double[])` | 3.43 | 6.04 | 3.11 |
| `fillExponential(float[])` | 2.55 | 6.61 | 1.59 |

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
function runs two lanes at a time in the same way. `int[]`, `long[]`, `float[]` and `double[]`
fills generate whole blocks directly into the destination. The row loops end on `b != end`, which
C2 does not count: it unrolls counted loops with many xors twice, and the unrolled loop spilled.
Each iteration steps two rows instead, which halves the range checks and safepoint polls of the
uncounted loop. C2 has no 32 x 32 to 64-bit multiply and zero-extends both operands of each
product, so the two words that feed the next products stay zero-extended in longs. A double takes
its high word through the floating-point units, `hi 2^-32 + (lo >>> 11) 2^-53`, which is exact and
leaves the integer units to the step.

The JIT does not vectorise the row step. The loop issues about 35 instructions per 16 bytes, near
the integer issue rate of the M4 Pro core, so fills stay near 9 to 10 GiB/s against about 19 in
`tandem-c`. A scalar `nextLong` pays that generation plus about 20 instructions of position
update, block check and load. `SplittableRandom` spends fewer than that on its whole draw, so the
pure Java scalar draws cannot reach it.

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
by each device fill. Each sample is ten launches and one synchronize, and the figure is the
minimum of 21 samples after a half-second warm-up. The GPU had no other process, and a second run
agreed within 3 %. The tandem-cuda column is the figure of its
[speed page](https://github.com/tandem-rng/tandem-cuda/blob/main/docs/speed.md) for the same
kernel at 2^28 elements. The cuRAND columns are Philox4x32-10 of cuRAND 10.3.9 in the same run, by
the same method, through its host API in the context the fills use. cuRAND has no 64-bit integer
output for Philox, no bounded integers and no exponentials, so those rows give the nearest call,
marked "nearest": `curandGenerate` into the same bytes, or the uniform the exponential reads.

| fill | 2^26 GiB/s | 2^28 GiB/s | tandem-cuda 2^28 GiB/s | cuRAND 2^26 GiB/s | cuRAND 2^28 GiB/s | cuRAND call |
|---|---|---|---|---|---|---|
| `fillInts` | 1338 | 1380 | 1377 | 1273 | 1300 | `curandGenerate` |
| `fillLongs` | 1368 | 1393 | 1388 | 1285 | 1314 | `curandGenerate`, nearest |
| `fillFloats` | 1331 | 1381 | 1377 | 1262 | 1284 | `curandGenerateUniform` |
| `fillDoubles` | 1364 | 1392 | 1389 | 793 | 799 | `curandGenerateUniformDouble` |
| `fillBelowU32`, range 1000 | 1293 | 1347 | 1334 | 1269 | 1301 | `curandGenerate`, nearest |
| `fillBelowU64`, range 1000 | 1329 | 1358 | 1354 | 1285 | 1312 | `curandGenerate`, nearest |
| `fillGaussianFloats` | 1188 | 1235 | 1204 | 882 | 885 | `curandGenerateNormal` |
| `fillGaussianDoubles` | 867 | 1018 | 1058 | 573 | 582 | `curandGenerateNormalDouble` |
| `fillExponentialFloats` | 1063 | 1090 | 1149 | 1244 | 1273 | `curandGenerateUniform`, nearest |
| `fillExponentialDoubles` | 896 | 913 | 941 | 790 | 791 | `curandGenerateUniformDouble`, nearest |

Array fills copy over PCIe: `fill(double[])` of 2^26 elements runs at 8.8 GiB/s. Keep data on the
device when a GPU consumes it.

The uniform and bounded fills run at the card's memory bandwidth, as in `tandem-cuda`. The
`fillGaussianDoubles` row is the ziggurat, the kernels of `tandem.cuh` itself. The normal and
exponential fills run below the uniforms because the card holds 250 W: their arithmetic lowers its
clock until it bounds them, see tandem-cuda's
[design](https://github.com/tandem-rng/tandem-cuda/blob/main/docs/design.md). The kernels come
from tandem-cuda 2693c63, whose exponentials take the same bits with fewer operations and whose
f32 normal takes its square root without the range check: 1185 to 1235 GiB/s. cuRAND leads the
f32 exponential row, where its nearest call is a uniform without the logarithm. The device fills
of the API also allocate their memory on each call. The array fills add the copy over PCIe into
the Java heap: `fill(double[])` of 2^26 elements runs at 8.8 GiB/s end to end, about 150 times
slower than the device fill.
