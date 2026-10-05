<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .java"></p>

# tandem-java

[![CI](https://github.com/tandem-rng/tandem-java/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/tandem-rng/tandem-java/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-tandem--rng.github.io-7fb3ee.svg)](https://tandem-rng.github.io/tandem-java/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

Pure Java implementation of [Tandem8x32](https://github.com/tandem-rng/spec), a noncryptographic
pseudorandom number generator. It produces the specified stream bit for bit, as a JDK
`RandomGenerator`, and the optional `tandem-rng-cuda` module fills on NVIDIA GPUs.

The artifact is `tandem-rng`. It needs Java 25 or later and no runtime dependencies.
`pixi run native` builds tandem-c's `libtandem` at a pinned commit, and with its path in
`-Dtandem.native` (plus `--enable-native-access=ALL-UNNAMED`) or the library on
`java.library.path`, array fills of 512 elements or more run in it. Without it they run in
Java and give the same bits.

```sh
pixi run test        # mvn -B verify
```

```java
import io.github.tandemrng.Tandem;

Tandem rng = Tandem.seed(42);                  // 128-bit seed with whitening, default K
double[] xs = new double[1 << 20];
rng.fill(xs);                                  // the values of repeated nextDouble()
Tandem worker = rng.split(7);                  // by index, from the key alone
double g = worker.nextGaussian();              // ziggurat, bit identical to tandem-c
```

See [API](docs/api.md) for every draw, [GPU module](docs/gpu.md) for the CUDA fills, and
[tests](docs/tests.md) and [speed](docs/speed.md) for the rest.

Portions of the code were generated with the assistance of LLMs.

[Documentation](https://tandem-rng.github.io/tandem-java/) · [Apache 2.0 license](LICENSE)
