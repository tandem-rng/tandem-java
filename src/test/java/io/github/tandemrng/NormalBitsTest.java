package io.github.tandemrng;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

/**
 * The normal fills are bit identical to tandem-c on every JVM: the FNV-1a hashes of the little
 * endian bytes that tandem-c's tests/test_normal_bits.c records.
 */
class NormalBitsTest {
    private static final long[] STARTS = {0, 1, 77, 12345, 1L << 30};

    private static long fnv(long h, long bits, int bytes) {
        for (int i = 0; i < bytes; i++) h = (h ^ ((bits >>> (8 * i)) & 0xff)) * 0x100000001b3L;
        return h;
    }

    /** 10^6 double normals from each of five positions, about 21000 misses and 270 tail values. */
    @Test
    void doubleFillsMatchTheCReference() {
        double[] d = new double[1_000_000];
        long h = 0xcbf29ce484222325L;
        for (long start : STARTS) {
            Tandem g = Tandem.seed(2026, 7);
            g.setPosition(start);
            g.fillGaussian(d);
            for (double x : d) h = fnv(h, Double.doubleToRawLongBits(x), 8);
        }
        assertEquals(0xa61cfa844c85f7c1L, h);
    }

    /** A Python implementation written from the text of Appendix A, with its end positions. */
    @Test
    void doubleFillsMatchThePythonReference() {
        double[] d = new double[200_000];
        long[][] ref = {{0, 0x0c4059ed409d578dL, 12800000}, {2373, 0x30ce40c86b295193L, 12802432}};
        for (long[] r : ref) {
            Tandem g = new Tandem(new int[] {1, 2, 3, 4}, r[0], 32);
            g.fillGaussian(d);
            long h = 0xcbf29ce484222325L;
            for (double x : d) h = fnv(h, Double.doubleToRawLongBits(x), 8);
            assertEquals(r[1], h);
            assertEquals(r[2], g.position());
        }
    }

    /** 2 x 10^6 - 1 float normals from each of the five positions. */
    @Test
    void floatFillsMatchTheCReference() {
        float[] f = new float[2_000_000 - 1];
        long h = 0xcbf29ce484222325L;
        for (long start : STARTS) {
            Tandem g = Tandem.seed(2026, 7);
            g.setPosition(start);
            g.fillGaussian(f);
            for (float x : f) h = fnv(h, Float.floatToRawIntBits(x) & 0xffffffffL, 4);
        }
        assertEquals(0xaa1ea656ce73a4fbL, h);
    }
}
