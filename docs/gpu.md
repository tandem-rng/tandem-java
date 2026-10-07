# GPU module

GPU fill, in the `cuda/` module:

```java
try (TandemCuda gpu = TandemCuda.open(); Arena arena = Arena.ofConfined()) {   // device 0
    double[] a = new double[1 << 24];
    gpu.fill(a, rng);                                           // the values of rng.fill(a)
    MemorySegment d = gpu.fillDoubles(arena, 1L << 28, rng);    // device memory
}
```

The Maven module in `cuda/` is a second artifact, `tandem-rng-cuda`. It fills on an NVIDIA GPU
through the CUDA driver API, called through the Foreign Function and Memory API. It uses no JNI
and no native library of its own, and needs only the core jar and the driver. The core jar does
not depend on it.

- `TandemCuda.open()` and `open(int ordinal)` use the device's primary context. They throw
  `CudaException` when no driver or device is present, and need compute capability 8.0 or later.
- Array fills: `fill(int[] | long[] | float[] | double[], Tandem)`,
  `fillBelowU32(int[], int range, Tandem)`, `fillBelowU64(long[], long range, Tandem)`,
  `fillGaussian(float[] | double[], Tandem)` and `fillExponential(float[] | double[], Tandem)`.
- Device fills: `fillInts`, `fillLongs`, `fillFloats`, `fillDoubles`, `fillGaussianFloats`,
  `fillGaussianDoubles`, `fillExponentialFloats` and `fillExponentialDoubles(Arena, long n, Tandem)`,
  `fillBelowU32(Arena, long n, int range, Tandem)` and `fillBelowU64(Arena, long n, long range, Tandem)`. Each returns a segment of length 0 whose
  address is new device memory, so Java code cannot read device memory by accident. The memory is
  freed when the arena closes, also after `close()` of the handle.
- Every fill writes the values of the CPU fill of the same name and moves the generator's position
  as that fill does. Uniform, bounded, double normal and exponential fills are equal to the CPU
  fills bit for bit. Float normals agree to 16 ulps plus 3e-6, because the GPU
  takes the angle through `__sincosf`, whose absolute error of up to 2^-21.41 the radius
  multiplies.
- A rejected bounded draw retries on the fallback stream the CPU fill uses, keyed by the global
  draw index, as `tandem-cuda` does at the pinned commit. Its fixtures include fills from
  nonzero starts with rejections.
- A fill sets the new position through `setPosition`. So it drops the kept half of
  `nextGaussianFloat`, and its end position must stay below 2^63.

Run the JVM with `--enable-native-access=ALL-UNNAMED`, or with the name of your module. Without the
flag, JDK 25 prints a warning at the first driver call.

The kernels are `cuda/kernels/tandem_fills.cu`, entry points over `tandem.cuh` and `core.hpp` of
[tandem-cuda](https://github.com/tandem-rng/tandem-cuda) at commit `e98daee`. They mirror its fill
kernels: the tile kernel for `K >= 8` with 16-byte stores, the direct kernel for smaller `K`, and
the float normal kernels with the fast `__sincosf` path. The double normals launch the kernels of
`tandem.cuh` itself, by their mangled names, as its `fill_normal_f64_impl` plans them: the fused
kernel below 2^16 elements, else the table pass, which lists its misses in memory from
`cuMemAllocAsync`, and the kernel that continues them. The table pass stores in octets when the
output's 32-byte sectors start 8 or 16 bytes into the stream's rows. `cuda/tools/build_ptx.sh` compiles them
with nvcc 12.8 (clang 19 as host compiler, `pixi run -e nvcc ptx` in `cuda/`) into
`tandem_fills_sm80.ptx`, a resource of the jar. The script zeroes the per-machine hashes that nvcc
puts in the names of static device tables, so CI rebuilds the file byte for byte. The driver
JIT-compiles it for the GPU at hand on
the first `open`, which takes about 1 s once and is cached by the driver after that. The PTX
targets `sm_80` with ISA 8.7, so it needs driver 570 (CUDA 12.8) or newer and runs on every GPU
of compute capability 8.0 or later. There is no `sm_90` PTX, because the kernels use nothing that
`sm_90` adds.

```sh
cd cuda
pixi run test        # builds the core jar, then the tests without a GPU
pixi run gpu-test    # on a GPU host: the tests tagged gpu
pixi run bench
```
