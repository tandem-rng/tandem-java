package io.github.tandemrng.cuda;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.tandemrng.Tandem;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** GPU fills against the CPU fills of the core library, tandem-cuda's fixtures and the stream dumps. */
@Tag("gpu")
class TandemCudaTest {
    private static final int[] KS = {1, 2, 4, 8, 32, 65536};
    private static final int[] LENGTHS = {0, 1, 2, 3, 33, 1000, 70001, 1 << 20};

    private static TandemCuda gpu;

    @BeforeAll
    static void open() {
        gpu = TandemCuda.open();
    }

    @AfterAll
    static void close() {
        gpu.close();
    }

    /** Starts that cut words, blocks of 128 bits, rows of 1024 bits and groups of 1024 K bits. */
    private static long[] starts(int k) {
        long group = 1024L * k;
        return new long[] {0, 1, 31, 32, 33, 64, 96, 127, 129, 1000, group - 32, group + 64, 3 * group - 1, (1L << 40) + 7};
    }

    private static int[] key(Random r) {
        return new int[] {r.nextInt(), r.nextInt(), r.nextInt(), r.nextInt()};
    }

    @Test
    void uniformFillsEqualCpuFills() {
        Random r = new Random(1);
        for (int k : KS)
            for (long start : starts(k))
                for (int n : LENGTHS) {
                    Tandem g = new Tandem(key(r), start, k);
                    String at = "K " + k + ", start " + start + ", n " + n;
                    Tandem cpu = g.copy(), dev = g.copy();
                    int[] wi = new int[n], gi = new int[n];
                    cpu.fill(wi);
                    gpu.fill(gi, dev);
                    assertArrayEquals(wi, gi, "u32 " + at);
                    assertEquals(cpu.position(), dev.position(), "u32 position " + at);
                    long[] wl = new long[n], gl = new long[n];
                    cpu.fill(wl);
                    gpu.fill(gl, dev);
                    assertArrayEquals(wl, gl, "u64 " + at);
                    assertEquals(cpu.position(), dev.position(), "u64 position " + at);
                    float[] wf = new float[n], gf = new float[n];
                    cpu.fill(wf);
                    gpu.fill(gf, dev);
                    assertArrayEquals(wf, gf, "f32 " + at);
                    assertEquals(cpu.position(), dev.position(), "f32 position " + at);
                    double[] wd = new double[n], gd = new double[n];
                    cpu.fill(wd);
                    gpu.fill(gd, dev);
                    assertArrayEquals(wd, gd, "f64 " + at);
                    assertEquals(cpu.position(), dev.position(), "f64 position " + at);
                }
    }

    /** Ranges that never reject (powers of two, 0), rarely reject, and reject often. */
    @Test
    void boundedFillsEqualCpuFills() {
        int[] ranges32 = {0, 1, 3, 1000, 1 << 20, 0x80000001, -1};
        long[] ranges64 = {0, 1, 3, 1000, 1L << 40, 0x8000000000000001L, -1L};
        Random r = new Random(2);
        for (int k : new int[] {4, 32})
            for (long start : starts(k))
                for (int n : new int[] {0, 5, 4097, 70001})
                    for (int i = 0; i < ranges32.length; i++) {
                        Tandem g = new Tandem(key(r), start, k);
                        String at = "K " + k + ", start " + start + ", n " + n;
                        Tandem cpu = g.copy(), dev = g.copy();
                        int[] wi = new int[n], gi = new int[n];
                        cpu.fillBelowU32(wi, 0, n, ranges32[i]);
                        gpu.fillBelowU32(gi, ranges32[i], dev);
                        assertArrayEquals(wi, gi, "u32 below " + Integer.toUnsignedString(ranges32[i]) + ", " + at);
                        assertEquals(cpu.position(), dev.position(), "u32 below position " + at);
                        long[] wl = new long[n], gl = new long[n];
                        cpu.fillBelowU64(wl, 0, n, ranges64[i]);
                        gpu.fillBelowU64(gl, ranges64[i], dev);
                        assertArrayEquals(wl, gl, "u64 below " + Long.toUnsignedString(ranges64[i]) + ", " + at);
                        assertEquals(cpu.position(), dev.position(), "u64 below position " + at);
                    }
    }

    @Test
    void boundedFillsMatchTandemCudaFixtures() throws IOException {
        CrossFixtures f = new CrossFixtures("cross_fill_below.h");
        int[] key = CrossFixtures.key();
        List<CrossFixtures.Row> below32 = f.tables.get("CROSS_BELOW32"), below64 = f.tables.get("CROSS_BELOW64");
        assertTrue(below32.size() > 3 && below64.size() > 3);
        for (CrossFixtures.Row row : below32) {
            int[] got = new int[row.values().length];
            gpu.fillBelowU32(got, (int) row.head(), new Tandem(key, 0, 32));
            for (int i = 0; i < got.length; i++)
                assertEquals((int) CrossFixtures.integer(row.values()[i]), got[i], "range " + row.head() + ", element " + i);
        }
        for (CrossFixtures.Row row : below64) {
            long[] got = new long[row.values().length];
            gpu.fillBelowU64(got, row.head(), new Tandem(key, 0, 32));
            for (int i = 0; i < got.length; i++)
                assertEquals(CrossFixtures.integer(row.values()[i]), got[i], "range " + Long.toUnsignedString(row.head()) + ", element " + i);
        }
    }

    private static void near(double want, double got, String what) {
        assertTrue(Math.abs(got - want) <= 1e-12 * (1 + Math.abs(want)), what + ": want " + want + ", got " + got);
    }

    /**
     * The float step takes its angle through __sincosf, whose absolute error of up to 2^-21.41 the
     * radius multiplies, and float radii reach 5.77. So the bound is 16 ulps plus 3e-6, not the
     * 1e-6 of tandem-cuda's tests: 14 of 1.7e8 values exceed 16 ulps plus 1e-6, by up to 1.67e-6.
     */
    private static void near(float want, float got, String what) {
        double ulp = Math.abs((double) want) * 0x1p-23;
        assertTrue(Math.abs((double) got - want) <= 16 * ulp + 3e-6, what + ": want " + want + ", got " + got);
    }

    @Test
    void normalFillsAgreeWithCpuFills() {
        Random r = new Random(3);
        for (int k : new int[] {1, 8, 32})
            for (long start : starts(k))
                for (int n : LENGTHS) {
                    Tandem g = new Tandem(key(r), start, k);
                    String at = "K " + k + ", start " + start + ", n " + n;
                    Tandem cpu = g.copy(), dev = g.copy();
                    double[] wd = new double[n], gd = new double[n];
                    cpu.fillGaussian(wd);
                    gpu.fillGaussian(gd, dev);
                    for (int i = 0; i < n; i++) near(wd[i], gd[i], "f64 " + at + ", element " + i);
                    assertEquals(cpu.position(), dev.position(), "f64 position " + at);
                    float[] wf = new float[n], gf = new float[n];
                    cpu.fillGaussian(wf);
                    gpu.fillGaussian(gf, dev);
                    for (int i = 0; i < n; i++) near(wf[i], gf[i], "f32 " + at + ", element " + i);
                    assertEquals(cpu.position(), dev.position(), "f32 position " + at);
                }
    }

    @Test
    void normalFillsMatchTandemCudaFixtures() throws IOException {
        CrossFixtures f = new CrossFixtures("cross_fill_normal.h");
        int[] key = CrossFixtures.key();
        List<CrossFixtures.Row> normal64 = f.tables.get("CROSS_NORMAL64"), normal32 = f.tables.get("CROSS_NORMAL32");
        assertTrue(normal64.size() > 2 && normal32.size() > 2);
        for (CrossFixtures.Row row : normal64) {
            double[] got = new double[(int) row.count()];
            gpu.fillGaussian(got, new Tandem(key, row.head(), 32));
            for (int i = 0; i < got.length; i++)
                near(Double.parseDouble(row.values()[i]), got[i], "pos " + row.head() + ", element " + i);
        }
        for (CrossFixtures.Row row : normal32) {
            float[] got = new float[(int) row.count()];
            gpu.fillGaussian(got, new Tandem(key, row.head(), 32));
            for (int i = 0; i < got.length; i++)
                near(Float.parseFloat(row.values()[i]), got[i], "pos " + row.head() + ", element " + i);
        }
    }

    private static ByteBuffer dump(String name) throws IOException {
        Path p = Path.of(System.getProperty("tandem.data"), name);
        return ByteBuffer.wrap(Files.readAllBytes(p)).order(ByteOrder.LITTLE_ENDIAN);
    }

    @Test
    void fillsMatchStreamDumps() throws IOException {
        int[] key = {1, 2, 3, 4};
        for (int k : new int[] {32, 8}) {
            ByteBuffer b = dump("k1234_K" + k + "_u32.bin");
            int[] want = new int[b.remaining() / 4], got = new int[want.length];
            b.asIntBuffer().get(want);
            gpu.fill(got, new Tandem(key, 0, k));
            assertArrayEquals(want, got, "u32 K " + k);
        }
        ByteBuffer b = dump("k1234_K32_u64.bin");
        long[] wl = new long[b.remaining() / 8], gl = new long[wl.length];
        b.asLongBuffer().get(wl);
        gpu.fill(gl, new Tandem(key, 0, 32));
        assertArrayEquals(wl, gl, "u64");
        b = dump("seed42_K32_f32.bin");
        float[] wf = new float[b.remaining() / 4], gf = new float[wf.length];
        b.asFloatBuffer().get(wf);
        gpu.fill(gf, Tandem.seed(42));
        assertArrayEquals(wf, gf, "f32");
        b = dump("seed42_K32_f64.bin");
        double[] wd = new double[b.remaining() / 8], gd = new double[wd.length];
        b.asDoubleBuffer().get(wd);
        gpu.fill(gd, Tandem.seed(42));
        assertArrayEquals(wd, gd, "f64");
    }

    private interface DeviceFill {
        MemorySegment run(TandemCuda c, Arena a, long n, Tandem g);
    }

    private interface ArrayFill {
        MemorySegment run(int n, Tandem g);
    }

    /** Device memory outlives the handle until its arena closes, and holds the array fill's values. */
    @Test
    void deviceFillsEqualArrayFills() {
        record Case(String name, int bytes, DeviceFill device, ArrayFill array) {}
        List<Case> cases = List.of(
                new Case("u32", 4, (c, a, n, g) -> c.fillInts(a, n, g), (n, g) -> { int[] x = new int[n]; gpu.fill(x, g); return MemorySegment.ofArray(x); }),
                new Case("u64", 8, (c, a, n, g) -> c.fillLongs(a, n, g), (n, g) -> { long[] x = new long[n]; gpu.fill(x, g); return MemorySegment.ofArray(x); }),
                new Case("f32", 4, (c, a, n, g) -> c.fillFloats(a, n, g), (n, g) -> { float[] x = new float[n]; gpu.fill(x, g); return MemorySegment.ofArray(x); }),
                new Case("f64", 8, (c, a, n, g) -> c.fillDoubles(a, n, g), (n, g) -> { double[] x = new double[n]; gpu.fill(x, g); return MemorySegment.ofArray(x); }),
                new Case("u32 below", 4, (c, a, n, g) -> c.fillBelowU32(a, n, 0x80000001, g), (n, g) -> { int[] x = new int[n]; gpu.fillBelowU32(x, 0x80000001, g); return MemorySegment.ofArray(x); }),
                new Case("u64 below", 8, (c, a, n, g) -> c.fillBelowU64(a, n, 1000, g), (n, g) -> { long[] x = new long[n]; gpu.fillBelowU64(x, 1000, g); return MemorySegment.ofArray(x); }),
                new Case("f32 normal", 4, (c, a, n, g) -> c.fillGaussianFloats(a, n, g), (n, g) -> { float[] x = new float[n]; gpu.fillGaussian(x, g); return MemorySegment.ofArray(x); }),
                new Case("f64 normal", 8, (c, a, n, g) -> c.fillGaussianDoubles(a, n, g), (n, g) -> { double[] x = new double[n]; gpu.fillGaussian(x, g); return MemorySegment.ofArray(x); }));
        int n = 100003;
        Tandem start = new Tandem(new int[] {5, 6, 7, 8}, 4161, 32);
        for (Case c : cases) {
            Arena arena = Arena.ofConfined();
            TandemCuda own = TandemCuda.open();
            Tandem dev = start.copy(), arr = start.copy();
            MemorySegment d = c.device().run(own, arena, n, dev);
            MemorySegment want = c.array().run(n, arr);
            assertEquals(arr.position(), dev.position(), c.name() + " position");
            own.close();
            assertThrows(IllegalStateException.class, () -> c.device().run(own, arena, 1, dev.copy()), c.name() + " after close");
            MemorySegment got = arena.allocate(want.byteSize(), 8);
            own.copyToHost(got, d, (long) c.bytes() * n);
            assertEquals(-1, got.mismatch(want), c.name());
            arena.close();
        }
    }
}
