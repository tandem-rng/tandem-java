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
        // Each scalar pair is a cosine half, which the fixtures hold, then a kept sine half.
        for (int i = 0; i < Derived.NORMAL.length; i++) {
            double z = g.nextGaussian();
            g.nextGaussian();
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
            g.nextGaussianFloat();
            assertTrue(Math.abs(z - Derived.NORMALF[i]) <= 4e-6 * (1 + Math.abs(Derived.NORMALF[i])), "normal " + i);
        }
        assertEquals(Derived.NORMALF_END, g.position());
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
}
