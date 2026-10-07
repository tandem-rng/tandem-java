package io.github.tandemrng;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * The array fills of tandem-c's {@code libtandem}, bound through the native linker. The library
 * is the file named by the system property {@code tandem.native}, else {@code libtandem} on
 * {@code java.library.path}. Without one, or with one whose output differs from this port's,
 * every fill runs in Java. A library named by the property must load and agree, or class
 * initialization fails.
 */
final class NativeFills {
    private NativeFills() {}

    /** Shorter fills stay in Java, where they cost less than the call and the native state. */
    static final int MIN = 512;

    /**
     * Elements per native call. A critical call writes straight into the Java array and holds
     * off the garbage collector, so a long fill runs as calls of at most 512 KiB.
     */
    private static final int STEP = 1 << 16;

    // tandem_rng is returned by value. Any struct over 16 bytes comes back through memory the
    // caller provides, so a larger layout than tandem.h's 432 bytes is ABI compatible and leaves
    // room should the struct grow.
    private static final MemoryLayout RNG = MemoryLayout.structLayout(MemoryLayout.sequenceLayout(128, JAVA_LONG));

    private static final MethodHandle FROM_KEY, U32, U64, F32, F64, NORMAL_F32, NORMAL_F64, EXP_F32, EXP_F64;

    /** Whether the fills of {@code Tandem} at {@link #MIN} elements or more run in libtandem. */
    static final boolean ENABLED;

    static {
        String named = System.getProperty("tandem.native");
        Path path = named != null ? Path.of(named) : onLibraryPath();
        MethodHandle[] h = path == null ? null : bind(path, named != null);
        ENABLED = h != null;
        MethodHandle[] b = h != null ? h : new MethodHandle[9];
        FROM_KEY = b[0];
        U32 = b[1];
        U64 = b[2];
        F32 = b[3];
        F64 = b[4];
        NORMAL_F32 = b[5];
        NORMAL_F64 = b[6];
        EXP_F32 = b[7];
        EXP_F64 = b[8];
    }

    private static Path onLibraryPath() {
        String file = System.mapLibraryName("tandem");
        for (String dir : System.getProperty("java.library.path", "").split(java.io.File.pathSeparator)) {
            if (dir.isEmpty()) continue;
            Path p = Path.of(dir, file);
            if (Files.isRegularFile(p)) return p;
        }
        return null;
    }

    /** The handles, or null when the library does not load or disagrees and was not named. */
    private static MethodHandle[] bind(Path path, boolean required) {
        try {
            SymbolLookup lib = SymbolLookup.libraryLookup(path, Arena.global());
            Linker linker = Linker.nativeLinker();
            MethodHandle[] h = new MethodHandle[9];
            h[0] = linker.downcallHandle(
                    find(lib, "tandem_from_key"), FunctionDescriptor.of(RNG, ADDRESS, JAVA_LONG, JAVA_INT));
            String[] fills = {"tandem_fill_u32", "tandem_fill_u64", "tandem_fill_f32", "tandem_fill_f64",
                "tandem_fill_normal_f32", "tandem_fill_normal_f64", "tandem_fill_exponential_f32",
                "tandem_fill_exponential_f64"};
            for (int i = 0; i < fills.length; i++)
                h[i + 1] = linker.downcallHandle(find(lib, fills[i]),
                        FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_LONG), Linker.Option.critical(true));
            if (!agrees(h)) throw new IllegalStateException(path + " gives other values than tandem-java");
            return h;
        } catch (RuntimeException e) {
            if (required) throw new IllegalStateException("tandem.native: cannot use " + path, e);
            return null;
        }
    }

    private static MemorySegment find(SymbolLookup lib, String name) {
        return lib.find(name).orElseThrow(() -> new IllegalArgumentException("no symbol " + name));
    }

    /** A known-answer check against the Java fill: the uniform and normal draws of one key. */
    private static boolean agrees(MethodHandle[] h) {
        int[] key = {1, 2, 3, 4};
        long pos = 12345;
        long[] want = new long[300];
        new Tandem(key, pos, 32).fill(want);
        double[] wantNormal = new double[300];
        new Tandem(key, pos, 32).fillGaussian(wantNormal);
        long[] got = new long[want.length];
        double[] gotNormal = new double[wantNormal.length];
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment k = arena.allocateFrom(JAVA_INT, key);
            MemorySegment rng = (MemorySegment) h[0].invokeExact((SegmentAllocator) arena, k, pos, 32);
            h[2].invokeExact(rng, MemorySegment.ofArray(got), (long) got.length);
            rng = (MemorySegment) h[0].invokeExact((SegmentAllocator) arena, k, pos, 32);
            h[6].invokeExact(rng, MemorySegment.ofArray(gotNormal), (long) gotNormal.length);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
        return Arrays.equals(want, got) && Arrays.equals(wantNormal, gotNormal);
    }

    /** Writes n elements of f from the aligned bit position p of g's key and chunk length. */
    private static void fill(MethodHandle f, Tandem g, long p, MemorySegment out, int elementBytes, long n) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment k = arena.allocateFrom(JAVA_INT, g.key());
            MemorySegment rng = (MemorySegment) FROM_KEY.invokeExact((SegmentAllocator) arena, k, p, g.chunkLength());
            for (long i = 0; i < n; i += STEP) {
                long m = Math.min(STEP, n - i);
                f.invokeExact(rng, out.asSlice(i * elementBytes, m * elementBytes), m);
            }
        } catch (Throwable t) {
            throw new IllegalStateException("libtandem fill failed", t);
        }
    }

    static void fillU32(Tandem g, long p, int[] a, int off, int n) {
        fill(U32, g, p, MemorySegment.ofArray(a).asSlice(4L * off), 4, n);
    }

    static void fillU64(Tandem g, long p, long[] a, int off, int n) {
        fill(U64, g, p, MemorySegment.ofArray(a).asSlice(8L * off), 8, n);
    }

    static void fillF32(Tandem g, long p, float[] a, int off, int n) {
        fill(F32, g, p, MemorySegment.ofArray(a).asSlice(4L * off), 4, n);
    }

    static void fillF64(Tandem g, long p, double[] a, int off, int n) {
        fill(F64, g, p, MemorySegment.ofArray(a).asSlice(8L * off), 8, n);
    }

    static void fillNormalF32(Tandem g, long p, float[] a, int off, int n) {
        fill(NORMAL_F32, g, p, MemorySegment.ofArray(a).asSlice(4L * off), 4, n);
    }

    static void fillNormalF64(Tandem g, long p, double[] a, int off, int n) {
        fill(NORMAL_F64, g, p, MemorySegment.ofArray(a).asSlice(8L * off), 8, n);
    }

    static void fillExponentialF32(Tandem g, long p, float[] a, int off, int n) {
        fill(EXP_F32, g, p, MemorySegment.ofArray(a).asSlice(4L * off), 4, n);
    }

    static void fillExponentialF64(Tandem g, long p, double[] a, int off, int n) {
        fill(EXP_F64, g, p, MemorySegment.ofArray(a).asSlice(8L * off), 8, n);
    }
}
