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
  `sub(0x424c573332 or 0x424c573634).split(i)` of the fill's key, from position 0. Scalar
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
| Tandem | 1.25 | 1.75 | 1.98 | 23.92 | 2.95 | 3.84 | 5.70 | 6.25 |
| L64X128MixRandom | 1.15 | 1.11 | 1.13 | 3.38 | 3.29 | 6.40 | - | - |
| SplittableRandom | 0.44 | 0.50 | 0.62 | 2.29 | 8.97 | 14.66 | - | - |
| Random | 4.10 | 8.20 | 8.18 | 13.88 | 0.08 | 0.13 | - | - |
A scalar draw costs in proportion to the bytes it takes, because every row is generated on
demand and generation dominates: a `long` takes twice the stream of an `int`. The cached block is read through fields of the generator, with one unsigned range check per draw and the refill out of line. Each lane of a
row is independent, so the cache holds a block of 32 rows. Four lanes step through the block
together with their state in registers and store packed `long` pairs. The seeding function runs
four lanes at a time in the same way. `long[]` and `double[]` fills generate whole blocks directly into the destination. The JIT does not vectorise the row step, so fills stay at roughly 6 GiB/s on the M4 Pro against about 20 GiB/s in `tandem-c`.

`nextGaussian` is slow because it uses `StrictMath` for the same normals on every JVM, and each
pair costs one `log`, one `cos` and one `sin`.

The Vector API (`jdk.incubator.vector`, still incubating in JDK 25) is not used. It has no
widening or high-half 32-bit multiply, which the step needs twice. Emulating it with 16-bit pieces
or with long vectors measured 20 to 22 ns for one four-lane step on the M4 Pro, against 7.2 ns for
four scalar lanes, so a vector fill would be slower than the scalar one.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
