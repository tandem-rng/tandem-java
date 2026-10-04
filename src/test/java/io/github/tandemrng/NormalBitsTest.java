package io.github.tandemrng;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

/**
 * The normal fills are bit identical to tandem-c on every JVM and compiler: the FNV-1a hash of
 * 1e6 pairs of double and float normals from five positions equals the value that tandem-c's
 * tests/test_normal_bits.c records, over the same little-endian bytes.
 */
class NormalBitsTest {
    private static final long EXPECTED_HASH = 0x9414e1315e2653beL;

    private static long fnv(long h, long bits, int bytes) {
        for (int i = 0; i < bytes; i++) h = (h ^ ((bits >>> (8 * i)) & 0xff)) * 0x100000001b3L;
        return h;
    }

    @Test
    void normalFillsMatchTheCReferenceBitForBit() {
        int pairs = 1_000_000;
        double[] d = new double[2 * pairs - 1];
        float[] f = new float[2 * pairs - 1];
        long h = 0xcbf29ce484222325L;
        for (long start : new long[] {0, 1, 77, 12345, 1L << 30}) {
            Tandem g = Tandem.seed(2026, 7);
            g.setPosition(start);
            g.fillGaussian(d);
            for (double x : d) h = fnv(h, Double.doubleToRawLongBits(x), 8);
            g.fillGaussian(f);
            for (float x : f) h = fnv(h, Float.floatToRawIntBits(x) & 0xffffffffL, 4);
        }
        assertEquals(EXPECTED_HASH, h);
    }

    @Test
    void scalarDrawsEqualTheFill() {
        double[] want = new double[2001];
        Tandem.seed(2026, 7).fillGaussian(want);
        Tandem g = Tandem.seed(2026, 7);
        for (int i = 0; i < want.length; i++) assertEquals(Double.doubleToRawLongBits(want[i]), Double.doubleToRawLongBits(g.nextGaussian()), "element " + i);
    }
}
