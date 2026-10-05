# API

## Use

```java
import io.github.tandemrng.Tandem;

Tandem rng = Tandem.seed(42);                  // 128-bit seed with whitening, default K
double x = rng.nextDouble();
int[] words = new int[1 << 20];
rng.fill(words);                               // the values of repeated nextInt()
int i = rng.nextInt(10);                       // uniform in [0, 10), Lemire
double g = rng.nextGaussian();                 // 1024-layer ziggurat of one long draw
double e = rng.nextExponential();              // -ln(1 - u) from one double draw
Tandem worker = rng.split(7);                  // by index, from the key alone
Tandem[] kids = rng.fork(4);                   // from the current block, parent moves on
double y = rng.atDouble(1000);                 // element 1000 of the fill from here, no move
```

## Reference

- `Tandem`: key, position and chunk length `K`, with `Tandem(int[] key, long position, int k)`
  and `Tandem.seed(lo, hi, k)`. It is `Serializable` and not thread-safe.
- Every specification type: `boolean`, `byte`, `short`, `int`, `long`, 128-bit `{lo, hi}`,
  `float`, `double`, binary16 bits, `char`, complex `{re, im}`. Fills for the array types.
- `atDouble` and the other `at` methods read at a position without advancing.
- `split(i)`, `fork(n)`, `sub(purpose)`, `setPosition`, `key()`, `position()`, `chunkLength()`.
- Bounded integers: `nextInt(bound)`, `nextLong(bound)`, `nextInts`, `nextLongs`,
  `belowU32`, `belowU64`, `fillBelowU32`, `fillBelowU64`. They use Lemire's method.
- Normals: `nextGaussian`, `nextGaussianFloat`, `nextGaussianFloat2`,
  `fillGaussian(double[] | float[])`. They are bit identical to `tandem-c`.
- Exponentials: `nextExponential`, `nextExponentialFloat`, `fillExponential`.
  They are bit identical to `tandem-c` and `tandem-cuda`.
- `java.util.random.RandomGenerator.SplittableGenerator`, so it drives `ints()`, `doubles()`,
  `splits()` and `Collections.shuffle`.
- `TandemProvider` for Apache Commons RNG. Add `commons-rng-client-api` to use it.
- GPU, `TandemCuda` in `cuda/` (see [GPU module](gpu.md)): `open()`, `fill`, `fillBelowU32`, `fillBelowU64`, `fillGaussian`,
  `fillExponential` on arrays, and `fillInts`, `fillLongs`, `fillFloats`, `fillDoubles`,
  `fillGaussianFloats`, `fillGaussianDoubles`, `fillExponentialFloats`, `fillExponentialDoubles`
  into device memory. It needs compute capability 8.0 and driver 570.
  Run the JVM with `--enable-native-access=ALL-UNNAMED`.

### Details

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
- Double normals are the 1024-layer ziggurat of Appendix A, one long draw each.
  `nextGaussian()` and element `i` of `fillGaussian(double[])` take that draw. A draw that misses
  the fast path, 0.43 % of them, continues on `sub(0x4e524d3634).split(g)` of the key at
  position 0, where `g` is the draw's global index. So a fill equals the scalar calls, a fill cut
  at any element equals the whole fill, and the values equal `tandem-c`'s bit for bit. An empty
  fill aligns the position to 64 bits. The tables are generated from the spec's JSON into
  `ZigTables.java` by `tools/gen_zig_tables.py`.
- Float normals come in Box-Muller pairs: two float draws give a cosine half and a sine half.
  `nextGaussianFloat2()` returns the pair. `fillGaussian(float[])` fills elements `2j`, `2j + 1`
  from uniforms `2j`, `2j + 1` of the plain fill, and an odd length uses the cosine half of its
  last pair and still consumes both uniforms. Scalar `nextGaussianFloat()` returns the cosine
  half and keeps the sine half for the next call, so repeated calls give exactly the fill
  sequence. The kept half is dropped by `setPosition`, `split`, `fork` and `sub` and is not
  serialized: after deserialization the next call starts a fresh pair. Float normals are
  computed in float and are bit identical to `tandem-c`.
- Standard exponentials follow Appendix A of the specification: `-ln(1 - u)` from one uniform
  each, so a fill is random access. `nextExponential()` (the interface method, overridden) and
  `nextExponentialFloat()` draw one double or float. `fillExponential(double[])` and
  `fillExponential(float[])`, with `(a, off, len)` forms, give element `i` from uniform `i` of the
  plain fill, so a fill equals the scalar sequence and cuts anywhere. The logarithm is the
  polynomial of the normals in explicit `Math.fma`, in double for `double` and in float for
  `float`, so both are bit identical to `tandem-c` and `tandem-cuda` on every JVM.
- Bounded array fills (`nextInts(int[], bound)`, `nextLongs(long[], bound)`, and the unsigned
  `fillBelowU32`, `fillBelowU64`) follow the `tandem-cuda` contract: element i uses draw i of
  the plain fill, the position moves by exactly n draws, and a rejected draw retries on
  `sub(0x424c573332 or 0x424c573634).split(g)` of the fill's key, from position 0, where g is
  the aligned start position over the width in bits plus i, so chunked fills equal whole fills.
  Scalar `nextInt(bound)` stays the sequential rejection loop, so after a rejection the two
  differ.
- Draw width of bounded draws (Appendix A of the specification): an interface typed by result or
  bounds takes the width from the range, 32 bits when the range is at most 2^32 and 64 bits
  above, whatever the result type. So `nextLong(1000)` equals `nextInt(1000)` on the same stream,
  `nextLongs(a, 1000)` consumes 32-bit draws, and `[lo, hi)` draws on `hi - lo` and adds `lo`.
  `belowU32`, `belowU64`, `fillBelowU32` and `fillBelowU64` keep their widths.
- Implements `java.util.random.RandomGenerator.SplittableGenerator`, so it drives `ints()`,
  `doubles()`, `splits()` and `Collections.shuffle`. The interface's default `nextGaussian` is
  overridden by the ziggurat. Every bounded integer draw uses Lemire's method:
  `nextInt(bound)`, `nextInt(origin, bound)`, `nextLong(bound)`, `nextLong(origin, bound)` and
  the bounded `ints`, `longs` and `doubles` streams, which are sequential loops over those scalar
  draws (so `ints(n, 0, 1000)` equals n calls of `nextInt(1000)`, and an origin shifts by
  addition). The bounded `nextDouble` and `nextFloat` scale our uniforms. These replace the JDK
  default algorithms. Do not run the streams in parallel.
- `TandemProvider` adapts a generator to Apache Commons RNG's `UniformRandomProvider`. The
  dependency `commons-rng-client-api` is optional: add it to your build to use the class.
- `Serializable`. The stream survives a round trip through its transport form.

## More use

Raw keys and other chunk lengths:

```java
Tandem a = new Tandem(new int[] {1, 2, 3, 4}, 0L, 8);   // key words, position, K
Tandem b = Tandem.seed(lo, hi, 64);                     // 128-bit seed as two longs, K = 64
```

More draws and accessors:

```java
double[] z = rng.nextComplexDouble();          // {re, im}, two double draws
float[] ex = new float[1 << 20];
rng.fillExponential(ex);                       // the values of repeated nextExponentialFloat()
int[] key = rng.key();
long position = rng.position();
int k = rng.chunkLength();
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
as an unsigned 64-bit seed. `belowU32` and `belowU64` take their bound as an unsigned value, and
a bound of 0 returns 0 and consumes one draw (specification Appendix A), so they reach ranges
above `Integer.MAX_VALUE` and `Long.MAX_VALUE`.

## Parallel use

Element `i` of a fill is draw `i`, so ranks, threads or devices that start at the
position of their first element, or draw from `split(task)`, reproduce a serial run for any
decomposition (Appendix B of the [spec](https://github.com/tandem-rng/spec/blob/main/SPEC.md)).

A generator is not thread-safe. Give each thread its own child from `split` or `fork`.
