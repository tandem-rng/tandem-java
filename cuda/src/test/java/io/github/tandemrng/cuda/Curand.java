package io.github.tandemrng.cuda;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;

/**
 * cuRAND Philox4x32-10 through its host API, for the bench's comparison rows. It runs in the
 * device's primary context, which TandemCuda's allocations share, on the legacy default stream.
 * libcurand comes from the pixi environment.
 */
final class Curand implements AutoCloseable {
    private static final int PHILOX4_32_10 = 161;
    private static final Linker LINKER = Linker.nativeLinker();

    private final MethodHandle generate, uniform, uniformDouble, normal, normalDouble, destroy;
    private final MemorySegment gen;
    final int version;

    Curand(long seed) {
        String prefix = System.getenv("CONDA_PREFIX");
        SymbolLookup lib = SymbolLookup.libraryLookup(
                prefix == null ? Path.of("libcurand.so.10") : Path.of(prefix, "lib", "libcurand.so.10"),
                Arena.global());
        generate = bind(lib, "curandGenerate", ADDRESS, ADDRESS, JAVA_LONG);
        uniform = bind(lib, "curandGenerateUniform", ADDRESS, ADDRESS, JAVA_LONG);
        uniformDouble = bind(lib, "curandGenerateUniformDouble", ADDRESS, ADDRESS, JAVA_LONG);
        normal = bind(lib, "curandGenerateNormal", ADDRESS, ADDRESS, JAVA_LONG, JAVA_FLOAT, JAVA_FLOAT);
        normalDouble = bind(lib, "curandGenerateNormalDouble", ADDRESS, ADDRESS, JAVA_LONG,
                JAVA_DOUBLE, JAVA_DOUBLE);
        destroy = bind(lib, "curandDestroyGenerator", ADDRESS);
        try (Arena a = Arena.ofConfined()) {
            MemorySegment p = a.allocate(ADDRESS), v = a.allocate(JAVA_INT);
            check((int) bind(lib, "curandCreateGenerator", ADDRESS, JAVA_INT).invokeExact(p, PHILOX4_32_10));
            gen = p.get(ADDRESS, 0);
            check((int) bind(lib, "curandSetPseudoRandomGeneratorSeed", ADDRESS, JAVA_LONG).invokeExact(gen, seed));
            check((int) bind(lib, "curandGetVersion", ADDRESS).invokeExact(v));
            version = v.get(JAVA_INT, 0);
        } catch (Throwable e) {
            throw new CudaException("cuRAND: " + e.getMessage());
        }
    }

    /** n values of the call for kind into device memory at out, as the bench rows name them. */
    void fill(TandemCuda.Kind kind, long address, long n) {
        MemorySegment out = MemorySegment.ofAddress(address);
        try {
            check(switch (kind) {
                case U32, U32_BELOW -> (int) generate.invokeExact(gen, out, n);
                case U64, U64_BELOW -> (int) generate.invokeExact(gen, out, 2 * n);
                case F32, EXP_F32 -> (int) uniform.invokeExact(gen, out, n);
                case F64, EXP_F64 -> (int) uniformDouble.invokeExact(gen, out, n);
                case NORMAL_F32 -> (int) normal.invokeExact(gen, out, n, 0.0f, 1.0f);
                case NORMAL_F64 -> (int) normalDouble.invokeExact(gen, out, n, 0.0, 1.0);
            });
        } catch (Throwable e) {
            throw new CudaException("cuRAND: " + e.getMessage());
        }
    }

    /** The call that stands for kind, and whether cuRAND lacks the kind's own output. */
    static String call(TandemCuda.Kind kind) {
        return switch (kind) {
            case U32 -> "curandGenerate";
            case U64, U32_BELOW, U64_BELOW -> "curandGenerate, nearest";
            case F32 -> "curandGenerateUniform";
            case F64 -> "curandGenerateUniformDouble";
            case EXP_F32 -> "curandGenerateUniform, nearest";
            case EXP_F64 -> "curandGenerateUniformDouble, nearest";
            case NORMAL_F32 -> "curandGenerateNormal";
            case NORMAL_F64 -> "curandGenerateNormalDouble";
        };
    }

    @Override
    public void close() {
        try {
            check((int) destroy.invokeExact(gen));
        } catch (Throwable e) {
            throw new CudaException("cuRAND: " + e.getMessage());
        }
    }

    private static MethodHandle bind(SymbolLookup lib, String name, java.lang.foreign.MemoryLayout... args) {
        return LINKER.downcallHandle(lib.find(name).orElseThrow(), FunctionDescriptor.of(JAVA_INT, args));
    }

    private static void check(int status) {
        if (status != 0) throw new CudaException("cuRAND status " + status);
    }
}
