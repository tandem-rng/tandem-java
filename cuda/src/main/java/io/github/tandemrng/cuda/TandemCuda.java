package io.github.tandemrng.cuda;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import io.github.tandemrng.Tandem;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Tandem fills on a CUDA GPU, bound to the CUDA driver through the Foreign Function and Memory
 * API. Every fill takes its key, position and chunk length from a {@link Tandem}, writes the
 * values of the generator's CPU fill of the same name, and moves the generator's position as
 * that fill does. The uniform and bounded integer fills are equal to the CPU fills bit for bit.
 * The double normals agree to a relative 1e-12, the float normals to 16 ulps plus 3e-6, because
 * the GPU computes the transcendentals differently.
 *
 * <p>A non-empty fill moves the position through {@link Tandem#setPosition}, which drops the
 * kept normal halves of {@code nextGaussian} and needs an end position below 2^63.
 *
 * <p>The array fills compute on the device and copy to the array. The device fills return
 * device memory that lives until its arena closes. Their segments have length 0, so Java code
 * cannot touch device memory by accident: pass {@link MemorySegment#address()} to other CUDA
 * code. Device memory stays valid after {@link #close()} until its arena closes.
 *
 * <p>Instances use the device's primary context and may be shared between threads.
 */
public final class TandemCuda implements AutoCloseable {
    private static final int THREADS = 256;
    private static final int TILE_STEPS = 8;
    private static final int TILE_BYTES = 32 * 1024;
    private static final int CC_MAJOR = 75;

    /** One kernel entry of tandem_fills.cu, with the stream width it draws. */
    enum Kind {
        U32("fill_u32", 32, 4), U64("fill_u64", 64, 8), F32("fill_f32", 32, 4), F64("fill_f64", 64, 8),
        U32_BELOW("fill_u32_below", 32, 4), U64_BELOW("fill_u64_below", 64, 8),
        NORMAL_F32("fill_normal_f32", 32, 4), NORMAL_F64("fill_normal_f64", 64, 8);

        final String entry;
        final int bits, bytes;

        Kind(String entry, int bits, int bytes) {
            this.entry = entry;
            this.bits = bits;
            this.bytes = bytes;
        }

        boolean normal() {
            return this == NORMAL_F32 || this == NORMAL_F64;
        }

        /** The CPU fill of no elements, whose effect on the position the GPU fills copy. */
        void emptyFill(Tandem g, long range) {
            switch (this) {
                case U32 -> g.fill(new int[0]);
                case U64 -> g.fill(new long[0]);
                case F32 -> g.fill(new float[0]);
                case F64 -> g.fill(new double[0]);
                case U32_BELOW -> g.fillBelowU32(new int[0], 0, 0, (int) range);
                case U64_BELOW -> g.fillBelowU64(new long[0], 0, 0, range);
                case NORMAL_F32 -> g.fillGaussian(new float[0]);
                case NORMAL_F64 -> g.fillGaussian(new double[0]);
            }
        }
    }

    private final Driver cu;
    private final int device;
    private final MemorySegment context, module;
    private final MemorySegment[] functions = new MemorySegment[Kind.values().length];
    /** The normal entries for a start at an odd draw. */
    private final MemorySegment[] oddFunctions = new MemorySegment[Kind.values().length];
    private final AtomicBoolean closed = new AtomicBoolean();
    /** One hold for the open handle and one per live device allocation. */
    private final AtomicLong holds = new AtomicLong(1);

    private TandemCuda(Driver cu, int ordinal) {
        this.cu = cu;
        cu.init();
        device = cu.device(ordinal);
        int major = cu.attribute(CC_MAJOR, device);
        if (major < 8)
            throw new CudaException("device " + ordinal + " has compute capability " + major + ".x, the kernels need 8.0 or later");
        context = cu.retainPrimaryContext(device);
        MemorySegment m = null;
        try {
            cu.setCurrent(context);
            byte[] ptx = ptx();
            try (Arena a = Arena.ofConfined()) {
                MemorySegment image = a.allocate(ptx.length + 1L);
                MemorySegment.copy(MemorySegment.ofArray(ptx), 0, image, 0, ptx.length);
                m = cu.loadModule(image);
            }
            for (Kind k : Kind.values()) {
                functions[k.ordinal()] = cu.function(m, k.entry);
                if (k.normal()) oddFunctions[k.ordinal()] = cu.function(m, k.entry + "_odd");
            }
        } catch (RuntimeException e) {
            if (m != null) cu.unloadModule(m);
            cu.releasePrimaryContext(device);
            throw e;
        }
        module = m;
    }

    private static byte[] ptx() {
        try (InputStream in = TandemCuda.class.getResourceAsStream("tandem_fills_sm80.ptx")) {
            if (in == null) throw new IllegalStateException("tandem_fills_sm80.ptx is missing from the jar");
            return in.readAllBytes();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** Opens device 0. Throws {@link CudaException} when no CUDA driver or device is present. */
    public static TandemCuda open() {
        return open(0);
    }

    /** Opens the device of this ordinal. Throws {@link CudaException} when no CUDA driver or device is present. */
    public static TandemCuda open(int ordinal) {
        return new TandemCuda(Driver.get(), ordinal);
    }

    // ---- Array fills ------------------------------------------------------------------------

    /** As {@link Tandem#fill(int[])}. */
    public void fill(int[] dst, Tandem rng) {
        toHost(Kind.U32, MemorySegment.ofArray(dst), dst.length, 0, rng);
    }

    /** As {@link Tandem#fill(long[])}. */
    public void fill(long[] dst, Tandem rng) {
        toHost(Kind.U64, MemorySegment.ofArray(dst), dst.length, 0, rng);
    }

    /** As {@link Tandem#fill(float[])}. */
    public void fill(float[] dst, Tandem rng) {
        toHost(Kind.F32, MemorySegment.ofArray(dst), dst.length, 0, rng);
    }

    /** As {@link Tandem#fill(double[])}. */
    public void fill(double[] dst, Tandem rng) {
        toHost(Kind.F64, MemorySegment.ofArray(dst), dst.length, 0, rng);
    }

    /** As {@link Tandem#fillBelowU32} on the whole array, with {@code range} read as unsigned. */
    public void fillBelowU32(int[] dst, int range, Tandem rng) {
        toHost(Kind.U32_BELOW, MemorySegment.ofArray(dst), dst.length, Integer.toUnsignedLong(range), rng);
    }

    /** As {@link Tandem#fillBelowU64} on the whole array, with {@code range} read as unsigned. */
    public void fillBelowU64(long[] dst, long range, Tandem rng) {
        toHost(Kind.U64_BELOW, MemorySegment.ofArray(dst), dst.length, range, rng);
    }

    /** As {@link Tandem#fillGaussian(float[])}, to 16 ulps plus 3e-6. */
    public void fillGaussian(float[] dst, Tandem rng) {
        toHost(Kind.NORMAL_F32, MemorySegment.ofArray(dst), dst.length, 0, rng);
    }

    /** As {@link Tandem#fillGaussian(double[])}, to a relative 1e-12. */
    public void fillGaussian(double[] dst, Tandem rng) {
        toHost(Kind.NORMAL_F64, MemorySegment.ofArray(dst), dst.length, 0, rng);
    }

    // ---- Device fills -----------------------------------------------------------------------

    /** As {@link #fill(int[], Tandem)} into {@code n} ints of new device memory. */
    public MemorySegment fillInts(Arena arena, long n, Tandem rng) {
        return onDevice(Kind.U32, arena, n, 0, rng);
    }

    /** As {@link #fill(long[], Tandem)} into {@code n} longs of new device memory. */
    public MemorySegment fillLongs(Arena arena, long n, Tandem rng) {
        return onDevice(Kind.U64, arena, n, 0, rng);
    }

    /** As {@link #fill(float[], Tandem)} into {@code n} floats of new device memory. */
    public MemorySegment fillFloats(Arena arena, long n, Tandem rng) {
        return onDevice(Kind.F32, arena, n, 0, rng);
    }

    /** As {@link #fill(double[], Tandem)} into {@code n} doubles of new device memory. */
    public MemorySegment fillDoubles(Arena arena, long n, Tandem rng) {
        return onDevice(Kind.F64, arena, n, 0, rng);
    }

    /** As {@link #fillBelowU32(int[], int, Tandem)} into {@code n} ints of new device memory. */
    public MemorySegment fillBelowU32(Arena arena, long n, int range, Tandem rng) {
        return onDevice(Kind.U32_BELOW, arena, n, Integer.toUnsignedLong(range), rng);
    }

    /** As {@link #fillBelowU64(long[], long, Tandem)} into {@code n} longs of new device memory. */
    public MemorySegment fillBelowU64(Arena arena, long n, long range, Tandem rng) {
        return onDevice(Kind.U64_BELOW, arena, n, range, rng);
    }

    /** As {@link #fillGaussian(float[], Tandem)} into {@code n} floats of new device memory. */
    public MemorySegment fillGaussianFloats(Arena arena, long n, Tandem rng) {
        return onDevice(Kind.NORMAL_F32, arena, n, 0, rng);
    }

    /** As {@link #fillGaussian(double[], Tandem)} into {@code n} doubles of new device memory. */
    public MemorySegment fillGaussianDoubles(Arena arena, long n, Tandem rng) {
        return onDevice(Kind.NORMAL_F64, arena, n, 0, rng);
    }

    /** Releases the primary context once the device memory of every fill is freed as well. */
    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) release();
    }


    // ---- Internals --------------------------------------------------------------------------

    private void toHost(Kind kind, MemorySegment host, long n, long range, Tandem rng) {
        if (n == 0) {
            kind.emptyFill(rng, range);
            return;
        }
        long end = end(kind, rng.position(), n);
        try (Arena a = Arena.ofConfined()) {
            MemorySegment d = allocate(a, n * kind.bytes);
            launch(kind, d.address(), n, range, rng);
            copyToHost(host, d, n * kind.bytes);
        }
        rng.setPosition(end);
    }

    private MemorySegment onDevice(Kind kind, Arena arena, long n, long range, Tandem rng) {
        if (n < 0) throw new IllegalArgumentException("n must not be negative");
        if (n == 0) {
            kind.emptyFill(rng, range);
            return MemorySegment.NULL;
        }
        long end = end(kind, rng.position(), n);
        MemorySegment d = allocate(arena, Math.multiplyExact(n, kind.bytes));
        launch(kind, d.address(), n, range, rng);
        synchronize();
        rng.setPosition(end);
        return d;
    }

    private static long start(Kind kind, long pos) {
        return (pos + kind.bits - 1) & -kind.bits;
    }

    /** Draws a fill of n elements consumes: normals come in pairs. */
    private static long draws(Kind kind, long n) {
        return kind.normal() ? n + (n & 1) : n;
    }

    /** The position after a fill of n > 0 elements, refused where the CPU fill or setPosition would refuse it. */
    static long end(Kind kind, long pos, long n) {
        long p0 = start(kind, pos), draws = draws(kind, n);
        if (Long.compareUnsigned(p0, pos) < 0 || Long.compareUnsigned(Long.divideUnsigned(-1L - p0, kind.bits), draws) < 0)
            throw new IllegalStateException("position overflow");
        long end = p0 + kind.bits * draws;
        if (end < 0) throw new IllegalStateException("end position " + Long.toUnsignedString(end) + " is beyond 2^63 - 1");
        return end;
    }

    /** Device memory freed when the arena closes, as a segment of length 0. */
    MemorySegment allocate(Arena arena, long bytes) {
        if (closed.get() || holds.getAndUpdate(h -> h == 0 ? 0 : h + 1) == 0)
            throw new IllegalStateException("closed");
        try {
            cu.setCurrent(context);
            long p = cu.alloc(bytes);
            try {
                return MemorySegment.ofAddress(p).reinterpret(arena, s -> free(p));
            } catch (RuntimeException e) {
                cu.free(p);
                throw e;
            }
        } catch (RuntimeException e) {
            release();
            throw e;
        }
    }

    private void free(long p) {
        try {
            cu.setCurrent(context);
            cu.free(p);
        } finally {
            release();
        }
    }

    private void release() {
        if (holds.decrementAndGet() != 0) return;
        try {
            cu.setCurrent(context);
            cu.unloadModule(module);
        } finally {
            cu.releasePrimaryContext(device);
        }
    }

    /** Launches the fill of n > 0 elements into device memory at {@code out}, without waiting. */
    void launch(Kind kind, long out, long n, long range, Tandem rng) {
        if (closed.get()) throw new IllegalStateException("closed");
        int k = rng.chunkLength(), shift = Integer.numberOfTrailingZeros(k);
        long p0 = start(kind, rng.position()), draws = draws(kind, n);
        // The first and last group of 8 chunks the kernel walks, as the launchers of tandem.cuh plan them.
        long g0, g1;
        boolean odd = kind.normal() && (p0 >>> (kind == Kind.NORMAL_F64 ? 6 : 5) & 1) != 0;
        if (kind == Kind.NORMAL_F64) {
            long ba = (p0 >>> 7) + (odd ? 1 : 0);
            g0 = ba >>> 3 >>> shift;
            g1 = (ba + draws / 2 - 1) >>> 3 >>> shift;
        } else if (kind == Kind.NORMAL_F32) {
            long s0 = p0 >>> 5;
            g0 = s0 >>> 5 >>> shift;
            g1 = (s0 + draws - 1) >>> 5 >>> shift;
        } else {
            g0 = p0 >>> 10 >>> shift;
            g1 = (p0 + kind.bits * draws - 1) >>> 10 >>> shift;
        }
        long blocks = (g1 - g0 + THREADS / 8) / (THREADS / 8);
        if (blocks > Integer.MAX_VALUE) throw new IllegalArgumentException("fill too long for one launch");
        int shared = !kind.normal() && k >= TILE_STEPS ? TILE_BYTES : 0;
        int[] key = rng.key();
        // Every argument sits in an 8-byte slot: key words, position, K, n, the range of the
        // non-normal entries, and the output pointer.
        int count = kind.normal() ? 8 : 9;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment values = a.allocate(8L * count, 8), params = a.allocate(ADDRESS, count);
            for (int i = 0; i < 4; i++) values.set(JAVA_INT, 8L * i, key[i]);
            values.set(JAVA_LONG, 32, rng.position());
            values.set(JAVA_INT, 40, k);
            values.set(JAVA_LONG, 48, n);
            if (!kind.normal()) values.set(JAVA_LONG, 56, range);
            values.set(JAVA_LONG, 8L * (count - 1), out);
            for (int i = 0; i < count; i++) params.setAtIndex(ADDRESS, i, values.asSlice(8L * i));
            cu.setCurrent(context);
            cu.launch((odd ? oddFunctions : functions)[kind.ordinal()], (int) blocks, THREADS, shared, params);
        }
    }

    void synchronize() {
        cu.setCurrent(context);
        cu.synchronize();
    }

    /** Copies device memory into a native or heap segment, after the work launched before. */
    void copyToHost(MemorySegment host, MemorySegment device, long bytes) {
        cu.setCurrent(context);
        cu.copyToHost(host, device.address(), bytes);
    }
}
