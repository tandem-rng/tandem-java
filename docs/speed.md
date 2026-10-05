# Speed

`pixi run bench` produces the CPU figures, and `pixi run bench` in `cuda/` the GPU figures.

## CPU

One thread, `pixi run bench`, Apple M4 Pro, JDK 25.0.2 (Azul Zulu). Minimum of seven runs after
three warm-up runs. Nanoseconds per scalar draw, GiB/s for arrays of 2^24 elements.

| generator | nextInt ns | nextLong ns | nextDouble ns | nextGaussian ns | int[] loop GiB/s | double[] loop GiB/s | int[] fill GiB/s | double[] fill GiB/s |
|---|---|---|---|---|---|---|---|---|
| Tandem | 1.22 | 1.72 | 1.95 | 3.26 | 2.98 | 3.86 | 5.64 | 6.33 |
| L64X128MixRandom | 1.13 | 1.10 | 1.12 | 3.35 | 3.32 | 6.60 | - | - |
| SplittableRandom | 0.43 | 0.48 | 0.61 | 2.23 | 9.38 | 14.74 | - | - |
| Random | 3.98 | 7.97 | 7.96 | 13.76 | 0.07 | 0.13 | - | - |

Exponentials, same machine and method:

| generator | nextExponential ns | `fillExponential(double[])` GiB/s | `fillExponential(float[])` GiB/s |
|---|---|---|---|
| Tandem | 3.00 | 2.89 | 2.02 |
| L64X128MixRandom | 3.55 | - | - |

`fillGaussian` runs at about 2.8 GiB/s for `double[]` and 1.4 GiB/s for `float[]`. The double
figures are the ziggurat, measured on 2026-10-05 with the same method; `tools/Bench.java` now
prints both fills. A miss draws its fallback by random access, one block per draw.

Each generator runs in its own JVM. Arrays are filled by a loop of draws or by `Tandem.fill`.

A scalar draw costs in proportion to the bytes it takes, because every row is generated on
demand and generation dominates: a `long` takes twice the stream of an `int`. The cached block is
read through fields of the generator, with one unsigned range check per draw and the refill out of
line. Each lane of a row is independent, so the cache holds a block of 32 rows. Four lanes step
through the block together with their state in registers and store packed `long` pairs. The
seeding function runs four lanes at a time in the same way. `long[]` and `double[]` fills generate
whole blocks directly into the destination. The JIT does not vectorise the row step, so fills stay
at roughly 6 GiB/s on the M4 Pro against about 20 GiB/s in `tandem-c`.

The JIT does not vectorise the polynomial logarithm either: the loop runs at the same speed with
`-XX:-UseSuperWord`. So `fillExponential`
costs a uniform fill plus a scalar transform of about the same cost, while `tandem-c` runs the
transform on NEON or AVX2 lanes. That puts it under half of `tandem-c`.

Normals use no libm: the double ziggurat takes the tables of the spec and the polynomial
logarithm of `tandem.c` in explicit `Math.fma`, and the float Box-Muller its polynomials, so both
are bit identical to tandem-c on every JVM and compiler (tests check the hashes of 1e6 double
and 2e6 float normals at five positions).

The Vector API (`jdk.incubator.vector`, still incubating in JDK 25) is not used. It has no
widening or high-half 32-bit multiply, which the step needs twice. Emulating it with 16-bit pieces
or with long vectors measured 20 to 22 ns for one four-lane step on the M4 Pro, against 7.2 ns for
four scalar lanes, so a vector fill would be slower than the scalar one.

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
