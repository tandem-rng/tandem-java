package io.github.tandemrng;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Objects;
import java.util.function.BiFunction;
import org.junit.jupiter.api.Test;

/**
 * Every fill equals the same count of scalar draws and leaves the same position, from starts
 * that are unaligned, on row boundaries and inside rows, with lengths that cut rows and chunks.
 */
class FillsTest {
    private static final int[] KEY = {5, 6, 7, 8};
    private static final int[] CHUNKS = {1, 8, 32, 64, 1024};
    private static final long[] STARTS = {0, 1, 7, 8, 13, 33, 64, 127, 128, 1000, 1023, 1024, 1025, 4100, 32773, 1024L * 40 + 5, 1024L * 100 + 3};
    private static final int[] LENGTHS = {0, 1, 5, 31, 63, 64, 65, 129, 257, 1100};

    private record Case(String name, BiFunction<Tandem, Integer, Object> fill, BiFunction<Tandem, Integer, Object> draws) {}

    private static final Case[] CASES = {
        new Case("int", (t, n) -> {
            int[] a = new int[n];
            t.fill(a);
            return a;
        }, (t, n) -> {
            int[] a = new int[n];
            for (int i = 0; i < n; i++) a[i] = t.nextInt();
            return a;
        }),
        new Case("long", (t, n) -> {
            long[] a = new long[n];
            t.fill(a);
            return a;
        }, (t, n) -> {
            long[] a = new long[n];
            for (int i = 0; i < n; i++) a[i] = t.nextLong();
            return a;
        }),
        new Case("float", (t, n) -> {
            float[] a = new float[n];
            t.fill(a);
            return a;
        }, (t, n) -> {
            float[] a = new float[n];
            for (int i = 0; i < n; i++) a[i] = t.nextFloat();
            return a;
        }),
        new Case("double", (t, n) -> {
            double[] a = new double[n];
            t.fill(a);
            return a;
        }, (t, n) -> {
            double[] a = new double[n];
            for (int i = 0; i < n; i++) a[i] = t.nextDouble();
            return a;
        }),
        new Case("byte", (t, n) -> {
            byte[] a = new byte[n];
            t.fill(a);
            return a;
        }, (t, n) -> {
            byte[] a = new byte[n];
            for (int i = 0; i < n; i++) a[i] = t.nextByte();
            return a;
        }),
        new Case("short", (t, n) -> {
            short[] a = new short[n];
            t.fill(a);
            return a;
        }, (t, n) -> {
            short[] a = new short[n];
            for (int i = 0; i < n; i++) a[i] = t.nextShort();
            return a;
        }),
        new Case("boolean", (t, n) -> {
            boolean[] a = new boolean[n];
            t.fill(a);
            return a;
        }, (t, n) -> {
            boolean[] a = new boolean[n];
            for (int i = 0; i < n; i++) a[i] = t.nextBoolean();
            return a;
        }),
        new Case("float16", (t, n) -> {
            short[] a = new short[n];
            t.fillFloat16Bits(a);
            return a;
        }, (t, n) -> {
            short[] a = new short[n];
            for (int i = 0; i < n; i++) a[i] = t.nextFloat16Bits();
            return a;
        }),
        new Case("codepoint", (t, n) -> {
            int[] a = new int[n];
            t.fillCodePoints(a);
            return a;
        }, (t, n) -> {
            int[] a = new int[n];
            for (int i = 0; i < n; i++) a[i] = t.nextCodePoint();
            return a;
        }),
        new Case("long128", (t, n) -> {
            long[] a = new long[2 * n];
            t.fillLong128(a);
            return a;
        }, (t, n) -> {
            long[] a = new long[2 * n];
            for (int i = 0; i < n; i++) {
                long[] v = t.nextLong128();
                a[2 * i] = v[0];
                a[2 * i + 1] = v[1];
            }
            return a;
        }),
    };

    @Test
    void fillsEqualScalarDraws() {
        for (int k : CHUNKS)
            for (long start : STARTS)
                for (Case c : CASES)
                    for (int n : LENGTHS) {
                        Tandem fill = new Tandem(KEY, start, k), draws = fill.copy();
                        String where = c.name + " K=" + k + " start=" + start + " n=" + n;
                        assertTrue(Objects.deepEquals(c.draws.apply(draws, n), c.fill.apply(fill, n)), where);
                        // An empty fill still aligns the position, which no scalar draw did.
                        if (n == 0) continue;
                        assertEquals(draws.position(), fill.position(), where);
                        assertEquals(draws.nextLong(), fill.nextLong(), where + " continuation");
                    }
    }

    /** Random access evaluates the specification's definition directly, with no row or block cache. */
    @Test
    void blocksAgreeWithRandomAccessAtEveryChunkLength() {
        for (int k : new int[] {1, 2, 8, 64, 1024, 65536}) {
            Tandem g = new Tandem(KEY, 0L, k);
            int[] a = new int[32 * 100 + 7];
            g.copy().fill(a);
            for (int i = 0; i < a.length; i++) assertEquals(g.atInt(i), a[i], "K=" + k + " word " + i);
        }
    }

    @Test
    void fillOfASliceWritesOnlyTheSlice() {
        Tandem a = Tandem.seed(9), b = a.copy();
        int[] whole = new int[100];
        a.fill(whole, 10, 80);
        int[] want = new int[100];
        for (int i = 10; i < 90; i++) want[i] = b.nextInt();
        assertArrayEquals(want, whole);
    }

    @Test
    void emptyFillAlignsThePosition() {
        Tandem g = Tandem.seed(9);
        g.nextBoolean();
        g.fill(new double[0]);
        assertEquals(64, g.position());
    }

    /** Moving the position back reseeds the group, so the cache never leaks across a jump. */
    @Test
    void rereadingAfterASeek() {
        Tandem g = Tandem.seed(3, 4, 8);
        long[] first = new long[600];
        g.fill(first);
        g.setPosition(0);
        long[] again = new long[600];
        g.fill(again);
        assertArrayEquals(first, again);
        g.setPosition(64L * 300);
        assertEquals(first[300], g.nextLong());
    }
}
