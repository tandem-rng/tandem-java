# tandem-java

Pure Java implementation of Tandem8x32. It produces the stream of the
[specification](https://github.com/tandem-rng/spec/blob/main/SPEC.md) bit for bit, as a JDK
`RandomGenerator`, and the optional `tandem-rng-cuda` module fills on NVIDIA GPUs.

- [API](api.md): the generator, draws, fills, bounded integers, normals, exponentials and the JDK and Commons RNG interfaces.
- [GPU module](gpu.md): `tandem-rng-cuda` and its kernels.
- [Tests](tests.md): what the suites check, where the fixtures come from, and what CI runs.
- [Speed](speed.md): Apple M4 Pro and A100 figures.

## Install

The artifact is `tandem-rng`. Java 25 or later, no native code, no runtime dependencies.

Java 25 (the current LTS) or later, built with `--release 25` and tested on JDK 25 and 27.

```sh
pixi run test        # mvn -B verify
```

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.
