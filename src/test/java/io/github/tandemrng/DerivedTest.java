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

    @Test
    void nextLongIsBelowForPositiveBounds() {
        for (int c = 0; c < Derived.RANGE64.length; c++) {
            if (Derived.RANGE64[c] <= 0) continue;
            Tandem g = start();
            for (int i = 0; i < Derived.BELOW64[c].length; i++)
                assertEquals(Derived.BELOW64[c][i], g.nextLong(Derived.RANGE64[c]));
        }
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
        // Elements alternate the cosine half and the kept sine half.
        for (int i = 0; i < Derived.NORMAL.length; i++) {
            double z = g.nextGaussian();
            assertTrue(Math.abs(z - Derived.NORMAL[i]) <= 1e-12 * Math.abs(Derived.NORMAL[i]), "normal " + i);
        }
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
        long[] b = new long[64];
        Tandem.seed(42).nextLongs(b, 1000);
        assertArrayEquals(Derived.FILL_BELOW64[3], b);
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
            for (int i = 0; i < want.length; i++)
                assertTrue(Math.abs(got[i] - want[i]) <= 1e-12 * Math.abs(want[i]), "f64 pos " + c + " element " + i);
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
}
