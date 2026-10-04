# tandem-java

Pure Java implementation of [Tandem8x32](https://github.com/tandem-rng/spec), a noncryptographic
pseudorandom number generator built to be fast on CPUs and GPUs alike. The artifact is
`tandem-rng`. It produces the stream the specification defines, bit for bit.

- Java 25 (the current LTS) or later, built with `--release 25` and tested on JDK 25 and 27. No native code, no runtime dependencies.
- A generator is its transport form (128-bit key, 64-bit bit position, chunk length `K`) plus
  a cache of one block of 32 rows of the stream. The cache never changes a drawn value and is
  not serialized.
- Every type in the specification: `boolean`, `byte`, `short`, `int`, `long`, 128-bit
  integers as `{lo, hi}`, `float`, `double`, binary16 as bit patterns, `char` as a code point,
  complex `float` and `double` as `{re, im}` pairs. Fills for `int[]`, `long[]`, `float[]`,
  `double[]`, `byte[]`, `short[]` and `boolean[]`. Random access without advancing. Split by
  index, fork at the current block, sub by purpose.
- Bounded integers (`nextInt(bound)`, `nextLong(bound)`) and standard normals are not in the
  specification. They follow the shared device core in `tandem-cuda`, so every port returns the
  same values.
- Standard normals come in Box-Muller pairs: two uniform draws give a cosine half and a sine
  half. `nextGaussian2()` and `nextGaussianFloat2()` return the pair. `fillGaussian(double[])` and
  `fillGaussian(float[])` fill elements `2j`, `2j + 1` from uniforms `2j`, `2j + 1` of the plain
  fill, and an odd length uses the cosine half of its last pair and still consumes both
  uniforms. Scalar `nextGaussian()` and `nextGaussianFloat()` return the cosine half and keep the
  sine half for the next call, so repeated calls give exactly the fill sequence. The kept half is
  dropped by `setPosition`, `split`, `fork` and `sub` and is not serialized: after
  deserialization the next call starts a fresh pair. The double normals agree with the other
  ports to a relative 1e-12. The float normals follow the host rule of `tandem-cuda`: float
  radius, angle 2 pi b in double, cosine and sine rounded to float, product in float. They agree
  across ports to a few ulps, not bit for bit.
- Bounded array fills (`nextInts(int[], bound)`, `nextLongs(long[], bound)`, and the unsigned
  `fillBelowU32`, `fillBelowU64`) follow the `tandem-cuda` contract: element i uses draw i of
  the plain fill, the position moves by exactly n draws, and a rejected draw retries on
  `sub(0x424c573332 or 0x424c573634).split(g)` of the fill's key, from position 0, where g is the aligned start position over the width in bits plus i, so chunked fills equal whole fills. Scalar
  `nextInt(bound)` stays the sequential rejection loop, so after a rejection the two differ.
- Implements `java.util.random.RandomGenerator.SplittableGenerator`, so it drives `ints()`,
  `doubles()`, `splits()` and `Collections.shuffle`. The interface's
  default `nextGaussian` is overridden by the Box-Muller transform. Every bounded integer draw
  uses Lemire's method: `nextInt(bound)`, `nextInt(origin, bound)`, `nextLong(bound)`,
  `nextLong(origin, bound)` and the bounded `ints`, `longs` and `doubles` streams, which are
  sequential loops over those scalar draws (so `ints(n, 0, 1000)` equals n calls of
  `nextInt(1000)`, and an origin shifts by addition). The bounded `nextDouble` and `nextFloat`
  scale our uniforms. These replace the JDK default algorithms. Do not run the streams in
  parallel.
- `TandemProvider` adapts a generator to Apache Commons RNG's `UniformRandomProvider`. The
  dependency `commons-rng-client-api` is optional: add it to your build to use the class.
- `Serializable`. The stream survives a round trip through its transport form.

A generator is not thread-safe. Give each thread its own child from `split` or `fork`.

## Use

```java
import io.github.tandemrng.Tandem;

Tandem rng = Tandem.seed(42);                  // 128-bit seed with whitening, default K
double x = rng.nextDouble();
int[] words = new int[1 << 20];
rng.fill(words);                               // the values of repeated nextInt()
double[] z = rng.nextComplexDouble();          // {re, im}, two double draws
int i = rng.nextInt(10);                       // uniform in [0, 10), Lemire
double g = rng.nextGaussian();                 // Box-Muller from two double draws
Tandem worker = rng.split(7);                  // by index, from the key alone
Tandem[] kids = rng.fork(4);                   // from the current block, parent moves on
double y = rng.atDouble(1000);                 // element 1000 of the fill from here, no move
int[] key = rng.key();
long position = rng.position();
int k = rng.chunkLength();
```

Raw keys and other chunk lengths:

```java
Tandem a = new Tandem(new int[] {1, 2, 3, 4}, 0L, 8);   // key words, position, K
Tandem b = Tandem.seed(lo, hi, 64);                     // 128-bit seed as two longs, K = 64
```

With the JDK and Commons RNG:

```java
import java.util.random.RandomGenerator;
import org.apache.commons.rng.UniformRandomProvider;

RandomGenerator r = Tandem.seed(42);
double[] samples = r.doubles(1000).toArray();
Collections.shuffle(list, r);                  // RandomGenerator overload
UniformRandomProvider p = new TandemProvider(Tandem.seed(42));
```

Positions are unsigned 64-bit values held in a `long`. `Tandem.seed(long)` reads its argument
as an unsigned 64-bit seed. `belowU32` and `belowU64` take their bound as an unsigned value, and a bound of 0 returns 0 and consumes one draw (specification Appendix A), so
they reach ranges above `Integer.MAX_VALUE` and `Long.MAX_VALUE`.

Parallel use: element `i` of a fill is draw `i`, so ranks, threads or devices that start at the
position of their first element, or draw from `split(task)`, reproduce a serial run for any
decomposition, as
[Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative)
of the specification shows.

## Tests

```sh
pixi run test        # mvn -B verify, JDK 25
```

`VectorsTest` checks every vector of the specification. `Vectors.java` is generated from the
spec repository's `vectors.json` by `tools/gen_vectors.py`, and CI fails when it is out of date.
`StreamsTest` compares long fills, scalar draws and random access against the reference stream
dumps in `src/test/resources/data`, complex fills included. They are byte-identical copies of
the dumps in `tandem-c`. `DerivedTest` compares bounded integers and normals with values from
the `tandem-cuda` core, bounded fills with rejections, and the cosine and sine halves of float and double normal pairs and fills, generated by `tools/gen_derived.cpp` into `Derived.java`, which CI
regenerates and diffs. `FillsTest` compares every fill with the same count of scalar draws at
offsets and lengths that cut rows, blocks and chunks, and the block cache with random access at
every chunk length. `GaussianTest` checks that scalar normals, pair calls and fills give one sequence, and the kept half. `TandemTest` covers child keys, forks, serialization and the
`RandomGenerator` contract. `TandemProviderTest` covers the Commons RNG adaptor.

CI runs `mvn -B verify` on JDK 25 and 27, on Linux and macOS.

## Speed

One thread, `pixi run bench` (`tools/Bench.java`), Apple M4 Pro, JDK 25.0.2 (Azul Zulu). Each
generator runs in its own JVM. The numbers are the minimum of seven runs after three warm-up
runs: nanoseconds per scalar draw, and GiB/s for arrays of 2^24 elements, filled by a loop of
draws or by `Tandem.fill`.

| generator | nextInt ns | nextLong ns | nextDouble ns | nextGaussian ns | int[] loop GiB/s | double[] loop GiB/s | int[] fill GiB/s | double[] fill GiB/s |
|---|---|---|---|---|---|---|---|---|
| Tandem | 1.24 | 1.77 | 1.99 | 4.27 | 2.94 | 3.87 | 5.61 | 6.26 |
| L64X128MixRandom | 1.16 | 1.13 | 1.15 | 3.42 | 3.24 | 6.52 | - | - |
| SplittableRandom | 0.44 | 0.50 | 0.62 | 2.25 | 9.28 | 14.86 | - | - |
| Random | 4.06 | 7.97 | 8.00 | 14.01 | 0.10 | 0.16 | - | - |
A scalar draw costs in proportion to the bytes it takes, because every row is generated on
demand and generation dominates: a `long` takes twice the stream of an `int`. The cached block is read through fields of the generator, with one unsigned range check per draw and the refill out of line. Each lane of a
row is independent, so the cache holds a block of 32 rows. Four lanes step through the block
together with their state in registers and store packed `long` pairs. The seeding function runs
four lanes at a time in the same way. `long[]` and `double[]` fills generate whole blocks directly into the destination. The JIT does not vectorise the row step, so fills stay at roughly 6 GiB/s on the M4 Pro against about 20 GiB/s in `tandem-c`.

Normals use no `Math` transcendental, only correctly rounded double arithmetic, so they are the
same on every JVM: an integer range reduction and a polynomial for the log, `Math.sqrt`, and the
angle folded into an octant with a short series for the sine and cosine. `fillGaussian` runs at
about 2.1 GiB/s for `double[]` and 1.3 GiB/s for `float[]`.


The Vector API (`jdk.incubator.vector`, still incubating in JDK 25) is not used. It has no
widening or high-half 32-bit multiply, which the step needs twice. Emulating it with 16-bit pieces
or with long vectors measured 20 to 22 ns for one four-lane step on the M4 Pro, against 7.2 ns for
four scalar lanes, so a vector fill would be slower than the scalar one.

## GPU fills

The Maven module in `cuda/` is a second artifact, `tandem-rng-cuda`. It fills on an NVIDIA GPU
through the CUDA driver API, called through the Foreign Function and Memory API. It uses no JNI
and no native library of its own, and needs only the core jar and the driver. The core jar does
not depend on it.

```java
import io.github.tandemrng.Tandem;
import io.github.tandemrng.cuda.TandemCuda;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

try (TandemCuda gpu = TandemCuda.open(); Arena arena = Arena.ofConfined()) {   // device 0
    Tandem rng = Tandem.seed(42);
    double[] x = new double[1 << 24];
    gpu.fill(x, rng);              // the values of rng.fill(x), and rng moves as rng.fill(x) moves it
    gpu.fillGaussian(x, rng);
    MemorySegment d = gpu.fillDoubles(arena, 1L << 28, rng);   // device memory, freed with the arena
    long devicePointer = d.address();                          // for other CUDA code
}
```

- `TandemCuda.open()` and `open(int ordinal)` use the device's primary context. They throw
  `CudaException` when no driver or device is present, and need compute capability 8.0 or later.
- Array fills: `fill(int[] | long[] | float[] | double[], Tandem)`,
  `fillBelowU32(int[], int range, Tandem)`, `fillBelowU64(long[], long range, Tandem)` and
  `fillGaussian(float[] | double[], Tandem)`.
- Device fills: `fillInts`, `fillLongs`, `fillFloats`, `fillDoubles`, `fillGaussianFloats` and
  `fillGaussianDoubles(Arena, long n, Tandem)`, `fillBelowU32(Arena, long n, int range, Tandem)` and
  `fillBelowU64(Arena, long n, long range, Tandem)`. Each returns a segment of length 0 whose
  address is new device memory, so Java code cannot read device memory by accident. The memory is
  freed when the arena closes, also after `close()` of the handle.
- Every fill writes the values of the CPU fill of the same name and moves the generator's position
  as that fill does. Uniform and bounded fills are equal to the CPU fills bit for bit. Double
  normals agree to a relative 1e-12. Float normals agree to 16 ulps plus 3e-6, because the GPU
  takes the angle through `__sincosf`, whose absolute error of up to 2^-21.41 the radius multiplies.
- A fill sets the new position through `setPosition`. So it drops the kept halves of
  `nextGaussian`, and its end position must stay below 2^63.

Run the JVM with `--enable-native-access=ALL-UNNAMED`, or with the name of your module. Without the
flag, JDK 25 prints a warning at the first driver call.

The kernels are `cuda/kernels/tandem_fills.cu`, entry points over `tandem.cuh` and `core.hpp` of
[tandem-cuda](https://github.com/tandem-rng/tandem-cuda) at commit `79a4ad0`. They mirror its fill
kernels: the tile kernel for `K >= 8` with 16-byte stores, the direct kernel for smaller `K`, and the
normal kernels with the fast `__sincosf` float path. `cuda/tools/build_ptx.sh` compiles them with
nvcc 12.8 (clang 19 as host compiler, `pixi run -e nvcc ptx` in `cuda/`) into
`tandem_fills_sm80.ptx`, a resource of the jar. The driver JIT-compiles it for the GPU at hand on
the first `open`, which takes about 1 s once and is cached by the driver after that. The PTX
targets `sm_80` with ISA 8.7, so it needs driver 570 (CUDA 12.8) or newer and runs on every GPU
of compute capability 8.0 or later. There is no `sm_90` PTX, because the kernels use nothing that `sm_90` adds.

```sh
cd cuda
pixi run test        # builds the core jar, then the tests without a GPU
pixi run gpu-test    # on a GPU host: the tests tagged gpu
pixi run bench
```

`TandemCudaTest`, tagged `gpu`, compares the four uniform fills with the CPU fills in values and
final position at chunk lengths 1, 2, 4, 8, 32 and 65536 and at starts and lengths that cut words,
blocks, rows and groups. It compares the bounded fills with the CPU fills at ranges that reject
often, rarely and never, and the normal fills with the CPU fills at every start slot. It checks the
bounded and normal fixtures of `tandem-cuda` (`cross_fill_below.h`, `cross_fill_normal.h`, copied
into the test resources), the stream dumps of the core tests, and the device fills against the
array fills. `NoDriverTest` checks the error without a driver. CI compiles the module and runs
`NoDriverTest` on Linux and macOS, because the runners have no GPU. It also rebuilds the PTX and the
fixtures from the pinned `tandem-cuda` commit and fails on any difference.

NVIDIA A100 40 GB (PCIe), driver 570.124, JDK 25, GPU idle, `pixi run bench`: GiB/s written by
each device fill into one reused device buffer. Each sample is ten launches and one synchronize,
and the figure is the minimum of 21 samples after a half-second warm-up. The last column is the
`tandem-cuda` README's figure for the same kernel at 2^28 elements.

| fill | 2^26 GiB/s | 2^28 GiB/s | tandem-cuda 2^28 GiB/s |
|---|---|---|---|
| `fillInts` | 1333 | 1385 | 1383 |
| `fillLongs` | 1367 | 1390 | 1395 |
| `fillFloats` | 1328 | 1381 | 1383 |
| `fillDoubles` | 1365 | 1392 | 1394 |
| `fillBelowU32`, range 1000 | 1288 | 1340 | 1336 |
| `fillBelowU64`, range 1000 | 1326 | 1356 | 1348 |
| `fillGaussianFloats` | 1140 | 1200 | 1290 |
| `fillGaussianDoubles` | 712 | 720 | 765 |

The uniform and bounded fills run at the card's memory bandwidth, as in `tandem-cuda`. The normal
rows vary by about 6% from run to run. In one process, our float normal kernel and `tandem-cuda`'s
measured 1211 to 1281 GiB/s alike. The device fills of the API also allocate their memory on each
call. The array fills add the copy over PCIe into the Java heap: `fill(double[])` of 2^26 elements
runs at 9.3 GiB/s end to end, about 150 times slower than the device fill. Keep the data on the
device when a GPU consumes it.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
