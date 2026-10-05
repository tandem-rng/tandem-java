# Speed

`pixi run bench` produces the CPU figures, and `pixi run bench` in `cuda/` the GPU figures.

## CPU

One thread, `pixi run bench`, Apple M4 Pro, JDK 25.0.2 (Azul Zulu). Minimum of seven runs after
three warm-up runs. Nanoseconds per scalar draw, GiB/s for arrays of 2^24 elements. The Tandem
and SplittableRandom rows are the median of three such runs on 2026-10-05, one window.

| generator | nextInt ns | nextLong ns | nextDouble ns | nextGaussian ns | int[] loop GiB/s | double[] loop GiB/s | int[] fill GiB/s | double[] fill GiB/s |
|---|---|---|---|---|---|---|---|---|
| Tandem | 1.14 | 1.53 | 1.69 | 2.49 | 3.21 | 4.43 | 9.35 | 8.59 |
| L64X128MixRandom | 1.13 | 1.10 | 1.12 | 3.35 | 3.32 | 6.60 | - | - |
| SplittableRandom | 0.44 | 0.49 | 0.62 | 2.28 | 9.32 | 15.15 | - | - |
| Random | 3.98 | 7.97 | 7.96 | 13.76 | 0.07 | 0.13 | - | - |

Exponentials, same machine and method:

| generator | nextExponential ns | `fillExponential(double[])` GiB/s | `fillExponential(float[])` GiB/s |
|---|---|---|---|
| Tandem | 2.69 | 3.31 | 2.18 |
| L64X128MixRandom | 3.55 | - | - |

`fillGaussian` runs at 3.98 GiB/s for `double[]` and 1.85 GiB/s for `float[]`, the median of the
same three runs. The double fill reads one table of 2048 pairs, the layer threshold and the
width with the sign of bit 10, by the 11 low bits of the draw. Its loop has no call and only
records the misses, which a second loop finishes. A miss draws its fallback by random access, F
on locals, and two draws share one lane.

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
