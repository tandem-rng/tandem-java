package io.github.tandemrng.cuda;

import static java.lang.foreign.ValueLayout.ADDRESS;
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
 * that fill does. The uniform, bounded integer, double normal and exponential fills are equal to
 * the CPU fills bit for bit. The float normals agree to 16 ulps plus 3e-6, because the GPU computes
 * the transcendentals differently.
 *
 * <p>A non-empty fill moves the position through {@link Tandem#setPosition}, which drops the
 * kept normal half of {@code nextGaussianFloat} and needs an end position below 2^63.
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

    /**
     * One fill, with its kernel entry of tandem_fills.cu and the stream width it draws. The
     * double normals launch the kernels of tandem.cuh instead, see {@link #launchNormal64}.
     */
    enum Kind {
        U32("fill_u32", 32, 4), U64("fill_u64", 64, 8), F32("fill_f32", 32, 4), F64("fill_f64", 64, 8),
        U32_BELOW("fill_u32_below", 32, 4), U64_BELOW("fill_u64_below", 64, 8),
        NORMAL_F32("fill_normal_f32", 32, 4), NORMAL_F64(null, 64, 8),
        EXP_F32("fill_exponential_f32", 32, 4), EXP_F64("fill_exponential_f64", 64, 8);

        final String entry;
        final int bits, bytes;

        Kind(String entry, int bits, int bytes) {
            this.entry = entry;
            this.bits = bits;
            this.bytes = bytes;
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
                case EXP_F32 -> g.fillExponential(new float[0]);
                case EXP_F64 -> g.fillExponential(new double[0]);
            }
        }
    }

    private final Driver cu;
    private final int device;
    private final MemorySegment context, module;
    private final MemorySegment[] functions = new MemorySegment[Kind.values().length];
    /** The float normal entry for a start at an odd draw. */
    private final MemorySegment normal32Odd;
    /**
     * tandem.cuh's double normal kernels: fused, the table pass for even and odd starts and in
     * octets for each start parity, and the misses.
     */
    private final MemorySegment normal64Fused, normal64Even, normal64Odd, normal64Octets0, normal64Octets1, normal64Misses;
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
            for (Kind k : Kind.values())
                if (k.entry != null) functions[k.ordinal()] = cu.function(m, k.entry);
            normal32Odd = cu.function(m, "fill_normal_f32_odd");
            normal64Fused = cu.function(m, "_ZN6tandem6detail19fill_normal64_fusedEjjjjjmmmPd");
            normal64Even = cu.function(m, "_ZN6tandem6detail20fill_normal64_kernelILb0EEEvjjjjjmmmPdPNS0_10NormalMissEPym");
            normal64Odd = cu.function(m, "_ZN6tandem6detail20fill_normal64_kernelILb1EEEvjjjjjmmmPdPNS0_10NormalMissEPym");
            normal64Octets0 = cu.function(m, "_ZN6tandem6detail20fill_normal64_octetsILj0EEEvjjjjjmmmPdPNS0_10NormalMissEPym");
            normal64Octets1 = cu.function(m, "_ZN6tandem6detail20fill_normal64_octetsILj1EEEvjjjjjmmmPdPNS0_10NormalMissEPym");
            normal64Misses = cu.function(m, "_ZN6tandem6detail20fill_normal64_missesEjjjjjmmmmPKNS0_10NormalMissEPKymPd");
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

    /** As {@link Tandem#fillGaussian(double[])}. */
    public void fillGaussian(double[] dst, Tandem rng) {
        toHost(Kind.NORMAL_F64, MemorySegment.ofArray(dst), dst.length, 0, rng);
    }

    /** As {@link Tandem#fillExponential(float[])}. */
    public void fillExponential(float[] dst, Tandem rng) {
        toHost(Kind.EXP_F32, MemorySegment.ofArray(dst), dst.length, 0, rng);
    }

    /** As {@link Tandem#fillExponential(double[])}. */
    public void fillExponential(double[] dst, Tandem rng) {
        toHost(Kind.EXP_F64, MemorySegment.ofArray(dst), dst.length, 0, rng);
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

    /** As {@link #fillExponential(float[], Tandem)} into {@code n} floats of new device memory. */
    public MemorySegment fillExponentialFloats(Arena arena, long n, Tandem rng) {
        return onDevice(Kind.EXP_F32, arena, n, 0, rng);
    }

    /** As {@link #fillExponential(double[], Tandem)} into {@code n} doubles of new device memory. */
    public MemorySegment fillExponentialDoubles(Arena arena, long n, Tandem rng) {
        return onDevice(Kind.EXP_F64, arena, n, 0, rng);
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

    /** Draws a fill of n elements consumes: float normals come in pairs. */
    private static long draws(Kind kind, long n) {
        return kind == Kind.NORMAL_F32 ? n + (n & 1) : n;
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
        int[] key = rng.key();
        long k0 = key[0] & MASK32, k1 = key[1] & MASK32, k2 = key[2] & MASK32, k3 = key[3] & MASK32;
        cu.setCurrent(context);
        if (kind == Kind.NORMAL_F64) {
            launchNormal64(k0, k1, k2, k3, k, shift, p0 >>> 6, n, out);
            return;
        }
        // The first and last group of 8 chunks the kernel walks, as the launchers of tandem.cuh plan them.
        long g0, g1;
        boolean odd = false;
        if (kind == Kind.NORMAL_F32) {
            long s0 = p0 >>> 5;
            odd = (s0 & 1) != 0;
            g0 = s0 >>> 5 >>> shift;
            g1 = (s0 + draws - 1) >>> 5 >>> shift;
        } else {
            g0 = p0 >>> 10 >>> shift;
            g1 = (p0 + kind.bits * draws - 1) >>> 10 >>> shift;
        }
        long blocks = (g1 - g0 + THREADS / 8) / (THREADS / 8);
        int shared = kind != Kind.NORMAL_F32 && k >= TILE_STEPS ? TILE_BYTES : 0;
        MemorySegment f = odd ? normal32Odd : functions[kind.ordinal()];
        if (kind == Kind.NORMAL_F32) run(f, blocks, 0, k0, k1, k2, k3, rng.position(), k, n, out);
        else run(f, blocks, shared, k0, k1, k2, k3, rng.position(), k, n, range, out);
    }

    /** tandem.cuh's fused kernel continues each miss where it finds it, short fills take it. */
    private static final long NORMAL_LIST_MIN = 1L << 16;

    /**
     * The double normal fill of n elements from draw d0, planned as {@code fill_normal_f64_impl}
     * of tandem.cuh: the table pass writes the fast path's values and lists the misses, 0.43 % of
     * the elements, in stream-ordered memory with room for n / 128, and the second kernel
     * continues each listed miss. The table pass stores in octets when the output's 32-byte
     * sectors start 8 or 16 bytes into the stream's rows, else it pairs elements across lanes
     * when the output is 8 bytes off the stream's 16-byte blocks.
     */
    private void launchNormal64(long k0, long k1, long k2, long k3, int k, int shift, long d0, long n, long out) {
        long ba = d0 >>> 1, bb = (d0 + n - 1) >>> 1;
        long g0 = ba >>> 3 >>> shift, g1 = bb >>> 3 >>> shift, chunks = 8 * (g1 - g0 + 1);
        long blocks = (chunks + THREADS - 1) / THREADS, cap = n / 128;
        long scratch = n < NORMAL_LIST_MIN ? 0 : cu.allocAsync(16 + 16 * cap);
        if (scratch == 0) {
            run(normal64Fused, blocks, 0, k0, k1, k2, k3, k, g0, d0, n, out);
            return;
        }
        long count = scratch, list = scratch + 16;
        cu.zeroAsync(count, 2);
        long off = (out - 8 * d0) & 31;
        MemorySegment table = (out & 31) == 0 && (off == 8 || off == 16) ? ((d0 & 1) != 0 ? normal64Octets1 : normal64Octets0)
                : (off & 15) == 0 ? normal64Even : normal64Odd;
        run(table, blocks, 0, k0, k1, k2, k3, k, g0, d0, n, out, list, count, cap);
        run(normal64Misses, (n / 200 + THREADS - 1) / THREADS, 0, k0, k1, k2, k3, k, g0, chunks, d0, n, list, count, cap, out);
        cu.freeAsync(scratch);
    }

    private static final long MASK32 = 0xffffffffL;

    /**
     * Launches {@code f} with each argument in an 8-byte slot. A 32-bit parameter reads the low
     * half of its slot, as the device is little endian.
     */
    private void run(MemorySegment f, long blocks, int shared, long... args) {
        if (blocks > Integer.MAX_VALUE) throw new IllegalArgumentException("fill too long for one launch");
        try (Arena a = Arena.ofConfined()) {
            MemorySegment values = a.allocate(8L * args.length, 8), params = a.allocate(ADDRESS, args.length);
            for (int i = 0; i < args.length; i++) {
                values.set(JAVA_LONG, 8L * i, args[i]);
                params.setAtIndex(ADDRESS, i, values.asSlice(8L * i));
            }
            cu.launch(f, (int) blocks, THREADS, shared, params);
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
