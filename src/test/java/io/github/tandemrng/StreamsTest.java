package io.github.tandemrng;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Test;

/**
 * Long streams from the Julia reference, dumped as little-endian arrays. Each is compared with
 * a fill, with scalar draws and, at sampled indices, with random access.
 */
class StreamsTest {
    private static final int[] KEY = {1, 2, 3, 4};

    private static ByteBuffer dump(String name) throws IOException {
        try (InputStream in = StreamsTest.class.getResourceAsStream("/data/" + name)) {
            return ByteBuffer.wrap(in.readAllBytes()).order(ByteOrder.LITTLE_ENDIAN);
        }
    }

    private static Tandem keyed(int k) {
        return new Tandem(KEY, 0L, k);
    }

    private static Tandem seeded() {
        return Tandem.seed(42);
    }

    @Test
    void u32K32() throws IOException {
        int[] want = new int[(int) dump("k1234_K32_u32.bin").remaining() / 4];
        dump("k1234_K32_u32.bin").asIntBuffer().get(want);
        checkInts(keyed(32), want);
    }

    @Test
    void u32K8() throws IOException {
        int[] want = new int[dump("k1234_K8_u32.bin").remaining() / 4];
        dump("k1234_K8_u32.bin").asIntBuffer().get(want);
        checkInts(keyed(8), want);
    }

    private static void checkInts(Tandem g, int[] want) {
        int[] got = new int[want.length];
        Tandem fill = g.copy(), draws = g.copy(), access = g.copy();
        fill.fill(got);
        assertArrayEquals(want, got);
        for (int i = 0; i < want.length; i++) assertEquals(want[i], draws.nextInt(), "draw " + i);
        assertEquals(fill.position(), draws.position());
        for (int i = 0; i < want.length; i += 97) assertEquals(want[i], access.atInt(i), "access " + i);
    }

    @Test
    void u64() throws IOException {
        long[] want = new long[dump("k1234_K32_u64.bin").remaining() / 8];
        dump("k1234_K32_u64.bin").asLongBuffer().get(want);
        long[] got = new long[want.length];
        Tandem fill = keyed(32), draws = keyed(32), access = keyed(32);
        fill.fill(got);
        assertArrayEquals(want, got);
        for (int i = 0; i < want.length; i++) assertEquals(want[i], draws.nextLong(), "draw " + i);
        assertEquals(fill.position(), draws.position());
        for (int i = 0; i < want.length; i += 97) assertEquals(want[i], access.atLong(i), "access " + i);
    }

    @Test
    void f64() throws IOException {
        double[] want = new double[dump("seed42_K32_f64.bin").remaining() / 8];
        dump("seed42_K32_f64.bin").asDoubleBuffer().get(want);
        double[] got = new double[want.length];
        Tandem draws = seeded(), access = seeded();
        seeded().fill(got);
        assertArrayEquals(want, got);
        for (int i = 0; i < want.length; i++) assertEquals(want[i], draws.nextDouble(), "draw " + i);
        for (int i = 0; i < want.length; i += 97) assertEquals(want[i], access.atDouble(i), "access " + i);
    }

    @Test
    void f32() throws IOException {
        float[] want = new float[dump("seed42_K32_f32.bin").remaining() / 4];
        dump("seed42_K32_f32.bin").asFloatBuffer().get(want);
        float[] got = new float[want.length];
        Tandem draws = seeded(), access = seeded();
        seeded().fill(got);
        assertArrayEquals(want, got);
        for (int i = 0; i < want.length; i++) assertEquals(want[i], draws.nextFloat(), "draw " + i);
        for (int i = 0; i < want.length; i += 97) assertEquals(want[i], access.atFloat(i), "access " + i);
    }

    @Test
    void u8() throws IOException {
        byte[] want = dump("seed42_K32_u8.bin").array();
        byte[] got = new byte[want.length];
        Tandem draws = seeded(), access = seeded();
        seeded().fill(got);
        assertArrayEquals(want, got);
        for (int i = 0; i < want.length; i++) assertEquals(want[i], draws.nextByte(), "draw " + i);
        for (int i = 0; i < want.length; i += 97) assertEquals(want[i], access.atByte(i), "access " + i);
    }

    @Test
    void shortsAreTheHalvesOfTheWords() throws IOException {
        int[] words = new int[dump("k1234_K32_u32.bin").remaining() / 4];
        dump("k1234_K32_u32.bin").asIntBuffer().get(words);
        short[] got = new short[2 * words.length];
        Tandem draws = keyed(32), access = keyed(32);
        keyed(32).fill(got);
        for (int i = 0; i < got.length; i++) {
            short want = (short) (words[i / 2] >>> (16 * (i & 1)));
            assertEquals(want, got[i], "fill " + i);
            assertEquals(want, draws.nextShort(), "draw " + i);
            if (i % 101 == 0) assertEquals(want, access.atShort(i), "access " + i);
        }
    }

    @Test
    void f16Bits() throws IOException {
        short[] want = new short[dump("seed42_K32_f16bits.bin").remaining() / 2];
        dump("seed42_K32_f16bits.bin").asShortBuffer().get(want);
        short[] got = new short[want.length];
        Tandem draws = seeded(), access = seeded();
        seeded().fillFloat16Bits(got);
        assertArrayEquals(want, got);
        for (int i = 0; i < want.length; i++) assertEquals(want[i], draws.nextFloat16Bits(), "draw " + i);
        for (int i = 0; i < want.length; i += 97) assertEquals(want[i], access.atFloat16Bits(i), "access " + i);
    }

    @Test
    void codePoints() throws IOException {
        int[] want = new int[dump("seed42_K32_char.bin").remaining() / 4];
        dump("seed42_K32_char.bin").asIntBuffer().get(want);
        int[] got = new int[want.length];
        Tandem draws = seeded(), access = seeded();
        seeded().fillCodePoints(got);
        assertArrayEquals(want, got);
        for (int i = 0; i < want.length; i++) assertEquals(want[i], draws.nextCodePoint(), "draw " + i);
        for (int i = 0; i < want.length; i += 97) assertEquals(want[i], access.atCodePoint(i), "access " + i);
    }

    @Test
    void long128() throws IOException {
        long[] want = new long[dump("seed42_K32_u128.bin").remaining() / 8];
        dump("seed42_K32_u128.bin").asLongBuffer().get(want);
        long[] got = new long[want.length];
        seeded().fillLong128(got);
        assertArrayEquals(want, got);
        Tandem draws = seeded();
        for (int i = 0; i < want.length; i += 2) {
            assertArrayEquals(new long[] {want[i], want[i + 1]}, draws.nextLong128(), "draw " + i / 2);
        }
    }

    @Test
    void complexDouble() throws IOException {
        double[] want = new double[dump("seed42_K32_c64.bin").remaining() / 8];
        dump("seed42_K32_c64.bin").asDoubleBuffer().get(want);
        double[] got = new double[want.length];
        seeded().fill(got);
        assertArrayEquals(want, got);
        Tandem draws = seeded();
        for (int i = 0; i < want.length; i += 2)
            assertArrayEquals(new double[] {want[i], want[i + 1]}, draws.nextComplexDouble(), "draw " + i / 2);
    }

    @Test
    void complexFloat() throws IOException {
        float[] want = new float[dump("seed42_K32_c32.bin").remaining() / 4];
        dump("seed42_K32_c32.bin").asFloatBuffer().get(want);
        float[] got = new float[want.length];
        seeded().fill(got);
        assertArrayEquals(want, got);
        Tandem draws = seeded();
        for (int i = 0; i < want.length; i += 2)
            assertArrayEquals(new float[] {want[i], want[i + 1]}, draws.nextComplexFloat(), "draw " + i / 2);
    }

    @Test
    void bools() throws IOException {
        byte[] want = dump("seed42_K32_bool.bin").array();
        boolean[] got = new boolean[want.length];
        Tandem fill = seeded(), draws = seeded(), access = seeded();
        fill.fill(got);
        for (int i = 0; i < want.length; i++) {
            assertEquals(want[i] != 0, got[i], "fill " + i);
            assertEquals(want[i] != 0, draws.nextBoolean(), "draw " + i);
            if (i % 97 == 0) assertEquals(want[i] != 0, access.atBoolean(i), "access " + i);
        }
        assertEquals(want.length, fill.position());
    }
}
