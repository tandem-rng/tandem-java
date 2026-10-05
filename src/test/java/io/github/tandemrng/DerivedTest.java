package io.github.tandemrng;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.random.RandomGenerator;
import org.junit.jupiter.api.Test;

/**
 * Bounded integers and normals against values from the tandem-cuda core, which every port
 * follows. Each case starts from {@code seed(42)} after one bit draw, which leaves the
 * position unaligned.
 */
class DerivedTest {
    private static Tandem start() {
        Tandem g = Tandem.seed(42);
        g.nextBoolean();
        return g;
    }

    @Test
    void belowU32() {
        for (int c = 0; c < Derived.RANGE32.length; c++) {
            Tandem g = start();
            for (int i = 0; i < Derived.BELOW32[c].length; i++)
                assertEquals(Derived.BELOW32[c][i], g.belowU32(Derived.RANGE32[c]), "range " + c + " draw " + i);
            assertEquals(Derived.BELOW32_END[c], g.position());
        }
    }

    @Test
    void belowU64() {
        for (int c = 0; c < Derived.RANGE64.length; c++) {
            Tandem g = start();
            for (int i = 0; i < Derived.BELOW64[c].length; i++)
                assertEquals(Derived.BELOW64[c][i], g.belowU64(Derived.RANGE64[c]), "range " + c + " draw " + i);
            assertEquals(Derived.BELOW64_END[c], g.position());
        }
    }

    @Test
    void nextIntIsBelowForPositiveBounds() {
        for (int c = 0; c < Derived.RANGE32.length; c++) {
            if (Derived.RANGE32[c] <= 0) continue;
            Tandem g = start();
            for (int i = 0; i < Derived.BELOW32[c].length; i++)
                assertEquals(Derived.BELOW32[c][i], g.nextInt(Derived.RANGE32[c]));
        }
    }

    /** Ranges above 2^32 draw 64 bits, as belowU64 does. */
    @Test
    void nextLongIsBelowU64ForWideBounds() {
        for (int c = 0; c < Derived.RANGE64.length; c++) {
            if (Derived.RANGE64[c] <= 0 || Derived.RANGE64[c] <= 1L << 32) continue;
            Tandem g = start();
            for (int i = 0; i < Derived.BELOW64[c].length; i++)
                assertEquals(Derived.BELOW64[c][i], g.nextLong(Derived.RANGE64[c]));
        }
        long edge = (1L << 32) + 1;
        Tandem g = start(), h = start();
        for (int i = 0; i < 50; i++) assertEquals(h.belowU64(edge), g.nextLong(edge));
        assertEquals(h.position(), g.position());
    }

    /** Appendix A: a range up to 2^32 draws 32 bits whatever the result type. */
    @Test
    void nextLongWithNarrowRangeEqualsNextInt() {
        for (int c = 0; c < Derived.RANGE32.length; c++) {
            long range = Derived.RANGE32[c] & 0xffffffffL;
            Tandem g = start();
            for (int i = 0; i < Derived.BELOW32[c].length; i++)
                assertEquals(Derived.BELOW32[c][i] & 0xffffffffL, g.nextLong(range), "range " + c + " draw " + i);
            assertEquals(Derived.BELOW32_END[c], g.position());
            Tandem h = start();
            for (int i = 0; i < Derived.BELOW32[c].length; i++)
                assertEquals(-5 + (Derived.BELOW32[c][i] & 0xffffffffL), h.nextLong(-5, -5 + range));
        }
        Tandem g = start(), h = start();
        for (int i = 0; i < 50; i++) assertEquals(h.nextInt() & 0xffffffffL, g.nextLong(1L << 32));
        assertEquals(h.position(), g.position());
    }

    /** The span of the widest int range, 2^32 - 1, exceeds Integer.MAX_VALUE. */
    @Test
    void widestRangesUseTheUnsignedSpan() {
        Tandem g = start();
        int[] want = Derived.BELOW32[Derived.BELOW32.length - 1];
        for (int w : want) assertEquals(Integer.MIN_VALUE + w, g.nextInt(Integer.MIN_VALUE, Integer.MAX_VALUE));
        Tandem h = start();
        long[] want64 = Derived.BELOW64[Derived.BELOW64.length - 1];
        for (long w : want64) assertEquals(Long.MIN_VALUE + w, h.nextLong(Long.MIN_VALUE, Long.MAX_VALUE));
    }

    @Test
    void normalsThroughTheInterface() {
        RandomGenerator g = start();
        for (int i = 0; i < Derived.NORMAL.length; i++)
            assertEquals(Double.doubleToRawLongBits(Derived.NORMAL[i]), Double.doubleToRawLongBits(g.nextGaussian()), "normal " + i);
        assertEquals(Derived.NORMAL_END, ((Tandem) g).position());
    }

    @Test
    void boundedFillsFollowTheSharedContract() {
        for (int c = 0; c < Derived.FILL_RANGE32.length; c++) {
            int[] a = new int[Derived.FILL_BELOW32[c].length];
            Tandem g = Tandem.seed(42);
            g.fillBelowU32(a, 0, a.length, Derived.FILL_RANGE32[c]);
            assertArrayEquals(Derived.FILL_BELOW32[c], a, "range32 " + c);
            assertEquals(32L * a.length, g.position());
        }
        for (int c = 0; c < Derived.FILL_RANGE64.length; c++) {
            long[] a = new long[Derived.FILL_BELOW64[c].length];
            Tandem g = Tandem.seed(42);
            g.fillBelowU64(a, 0, a.length, Derived.FILL_RANGE64[c]);
            assertArrayEquals(Derived.FILL_BELOW64[c], a, "range64 " + c);
            assertEquals(64L * a.length, g.position());
        }
    }

    @Test
    void positiveBoundFills() {
        int[] a = new int[64];
        Tandem.seed(42).nextInts(a, 1000);
        assertArrayEquals(Derived.FILL_BELOW32[3], a);
        long[] b = new long[64], wide = new long[64];
        Tandem g = Tandem.seed(42);
        g.nextLongs(wide, 1L << 40);
        assertEquals(64L * 64, g.position());
        Tandem.seed(42).fillBelowU64(b, 0, 64, 1L << 40);
        assertArrayEquals(wide, b);
    }

    /** A bound up to 2^32 fills from 32-bit draws and equals the int fill, widened. */
    @Test
    void nextLongsWithNarrowBoundEqualsNextInts() {
        for (int c = 0; c < Derived.FILL_RANGE32.length; c++) {
            long bound = Derived.FILL_RANGE32[c] & 0xffffffffL;
            long[] a = new long[Derived.FILL_BELOW32[c].length];
            Tandem g = Tandem.seed(42);
            g.nextLongs(a, bound);
            for (int i = 0; i < a.length; i++) assertEquals(Derived.FILL_BELOW32[c][i] & 0xffffffffL, a[i], "range " + c + " element " + i);
            assertEquals(32L * a.length, g.position());
        }
        long[] a = new long[40];
        int[] want = new int[40];
        Tandem.seed(42).nextLongs(a, 1L << 32);
        Tandem.seed(42).fill(want);
        for (int i = 0; i < 40; i++) assertEquals(want[i] & 0xffffffffL, a[i]);

        // Longer than one block of the widening, from an unaligned start, at a rejecting range.
        long[] wide = new long[3000];
        int[] narrow = new int[3000];
        Tandem p = Tandem.seed(5), q = Tandem.seed(5);
        p.nextByte();
        q.nextByte();
        p.nextLongs(wide, 0x80000001L);
        q.fillBelowU32(narrow, 0, narrow.length, 0x80000001);
        for (int i = 0; i < wide.length; i++) assertEquals(narrow[i] & 0xffffffffL, wide[i], "element " + i);
        assertEquals(q.position(), p.position());
    }

    @Test
    void boundedLongStreamUsesTheRangeWidth() {
        long[] got = Tandem.seed(9).longs(100, 10, 1010).toArray();
        Tandem g = Tandem.seed(9);
        for (int i = 0; i < 100; i++) assertEquals(10 + g.nextInt(1000), got[i]);
    }

    @Test
    void floatNormalsAgreeWithTheCoreToFewUlps() {
        Tandem g = start();
        for (int i = 0; i < Derived.NORMALF.length; i++) {
            float z = g.nextGaussianFloat();
            assertTrue(Math.abs(z - Derived.NORMALF[i]) <= 16 * Math.ulp(Derived.NORMALF[i]) + 1e-6, "normal " + i);
        }
        assertEquals(Derived.NORMALF_END, g.position());
    }

    @Test
    void normalFillsFollowTheCore() {
        for (int c = 0; c < Derived.FILLN_POS64.length; c++) {
            double[] want = Derived.FILLN64[c], got = new double[want.length];
            Tandem g = new Tandem(Tandem.seed(42).keyLo(), Tandem.seed(42).keyHi(), Derived.FILLN_POS64[c], 32);
            g.fillGaussian(got);
            assertArrayEquals(want, got, "f64 pos " + c);
        }
        for (int c = 0; c < Derived.FILLN_POS32.length; c++) {
            float[] want = Derived.FILLN32[c], got = new float[want.length];
            Tandem g = new Tandem(Tandem.seed(42).keyLo(), Tandem.seed(42).keyHi(), Derived.FILLN_POS32[c], 32);
            g.fillGaussian(got);
            for (int i = 0; i < want.length; i++)
                assertTrue(Math.abs(got[i] - want[i]) <= 16 * Math.ulp(want[i]) + 1e-6, "f32 pos " + c + " element " + i);
        }
    }

    /** The float normal evaluates the same formula on two float draws, to a few ulps. */
    @Test
    void floatNormalFollowsTheFormula() {
        Tandem g = start(), probe = start();
        for (int i = 0; i < 500; i++) {
            double u = probe.nextFloat(), v = probe.nextFloat();
            double r = Math.sqrt(-2 * Math.log(1 - u));
            double[] want = {r * Math.cos(2 * Math.PI * v), r * Math.sin(2 * Math.PI * v)};
            for (int h = 0; h < 2; h++) {
                float got = g.nextGaussianFloat();
                assertTrue(Math.abs(got - want[h]) <= 8 * Math.ulp((float) want[h]) + 1e-6, "draw " + i + " half " + h);
            }
        }
        assertEquals(probe.position(), g.position());
    }

    @Test
    void rangeZeroReturnsZeroAndConsumesOneDraw() {
        Tandem g = Tandem.seed(17);
        assertEquals(0, g.belowU32(0));
        assertEquals(32, g.position());
        assertEquals(0L, g.belowU64(0L));
        assertEquals(128, g.position());
        int[] a = {5, 5, 5};
        long[] b = {5, 5, 5};
        g.fillBelowU32(a, 0, 3, 0);
        assertEquals(0, a[0] | a[1] | a[2]);
        assertEquals(128 + 96, g.position());
        g.fillBelowU64(b, 0, 3, 0L);
        assertEquals(0L, b[0] | b[1] | b[2]);
        assertEquals(256 + 192, g.position());
    }

    /** Fallback streams are keyed by the global draw index, so a cut fill equals the whole fill. */
    @Test
    void boundedFillsCutAnywhereEqualTheWholeFill() {
        long start = 64L * 3 + 17;
        for (int cut : new int[] {1, 7, 33, 100, 255}) {
            int[] whole = new int[300], parts = new int[300];
            new Tandem(Tandem.seed(42).keyLo(), Tandem.seed(42).keyHi(), start, 32).fillBelowU32(whole, 0, 300, 0x80000001);
            Tandem g = new Tandem(Tandem.seed(42).keyLo(), Tandem.seed(42).keyHi(), start, 32);
            g.fillBelowU32(parts, 0, cut, 0x80000001);
            g.fillBelowU32(parts, cut, 300 - cut, 0x80000001);
            assertArrayEquals(whole, parts, "u32 cut " + cut);
            long[] whole64 = new long[300], parts64 = new long[300];
            new Tandem(Tandem.seed(42).keyLo(), Tandem.seed(42).keyHi(), start, 32).fillBelowU64(whole64, 0, 300, 0x8000000000000001L);
            Tandem h = new Tandem(Tandem.seed(42).keyLo(), Tandem.seed(42).keyHi(), start, 32);
            h.fillBelowU64(parts64, 0, cut, 0x8000000000000001L);
            h.fillBelowU64(parts64, cut, 300 - cut, 0x8000000000000001L);
            assertArrayEquals(whole64, parts64, "u64 cut " + cut);
        }
    }

    @Test
    void boundedFillsFromUnalignedStartsFollowTheCore() {
        Tandem seed = Tandem.seed(42);
        for (int s = 0; s < Derived.FILLS_START.length; s++) {
            for (int c = 0; c < Derived.FILL_RANGE32.length; c++) {
                int[] a = new int[64];
                new Tandem(seed.keyLo(), seed.keyHi(), Derived.FILLS_START[s], 32).fillBelowU32(a, 0, 64, Derived.FILL_RANGE32[c]);
                assertArrayEquals(Derived.FILLS_BELOW32[s][c], a, "u32 start " + s + " range " + c);
            }
            for (int c = 0; c < Derived.FILL_RANGE64.length; c++) {
                long[] a = new long[64];
                new Tandem(seed.keyLo(), seed.keyHi(), Derived.FILLS_START[s], 32).fillBelowU64(a, 0, 64, Derived.FILL_RANGE64[c]);
                assertArrayEquals(Derived.FILLS_BELOW64[s][c], a, "u64 start " + s + " range " + c);
            }
        }
    }
}
