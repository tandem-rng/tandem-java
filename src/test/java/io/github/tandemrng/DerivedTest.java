package io.github.tandemrng;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.tandemrng.Conformance.Case;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * The bounded draws of the {@link java.util.random.RandomGenerator} interface against the
 * scalar bounded cases of the spec's {@code below.json}, which start from the key of seed 42 at
 * position 1, and the float normal formula.
 */
class DerivedTest {
    private static List<Case> below(String kind) {
        return Conformance.cases("below.json").stream().filter(c -> c.kind().equals(kind)).toList();
    }

    @Test
    void nextIntIsBelowForPositiveBounds() {
        for (Case c : below("below_u32")) {
            int range = (int) c.word("range");
            if (range <= 0) continue;
            Tandem g = c.generator();
            for (long w : c.hex("values")) assertEquals((int) w, g.nextInt(range), c.id());
        }
    }

    /** Ranges above 2^32 draw 64 bits, as belowU64 does. */
    @Test
    void nextLongIsBelowU64ForWideBounds() {
        for (Case c : below("below_u64")) {
            long range = c.word("range");
            if (range <= 1L << 32) continue;
            Tandem g = c.generator();
            for (long w : c.hex("values")) assertEquals(w, g.nextLong(range), c.id());
        }
        long edge = (1L << 32) + 1;
        Tandem g = Tandem.seed(42), h = Tandem.seed(42);
        for (int i = 0; i < 50; i++) assertEquals(h.belowU64(edge), g.nextLong(edge));
        assertEquals(h.position(), g.position());
    }

    /** Appendix A: a range up to 2^32 draws 32 bits whatever the result type. */
    @Test
    void nextLongWithNarrowRangeEqualsNextInt() {
        for (Case c : below("below_u32")) {
            long range = c.word("range");
            Tandem g = c.generator(), h = c.generator();
            for (long w : c.hex("values")) {
                assertEquals(w, g.nextLong(range), c.id());
                assertEquals(-5 + w, h.nextLong(-5, -5 + range), c.id());
            }
            assertEquals(c.num("end"), g.position());
        }
        Tandem g = Tandem.seed(42), h = Tandem.seed(42);
        for (int i = 0; i < 50; i++) assertEquals(h.nextInt() & 0xffffffffL, g.nextLong(1L << 32));
        assertEquals(h.position(), g.position());
    }

    /** The span of the widest int range, 2^32 - 1, exceeds Integer.MAX_VALUE. */
    @Test
    void widestRangesUseTheUnsignedSpan() {
        Case c = Conformance.find("below.json", "CROSS_U32[5]");
        Tandem g = c.generator();
        for (long w : c.hex("values")) assertEquals(Integer.MIN_VALUE + (int) w, g.nextInt(Integer.MIN_VALUE, Integer.MAX_VALUE));
        Case d = Conformance.find("below.json", "CROSS_U64[4]");
        Tandem h = d.generator();
        for (long w : d.hex("values")) assertEquals(Long.MIN_VALUE + w, h.nextLong(Long.MIN_VALUE, Long.MAX_VALUE));
    }

    @Test
    void wideBoundFillsAreTheU64Fill() {
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
        for (int r = 0; r < 6; r++) {
            Case c = Conformance.find("fill_below.json", "CROSS_BELOW32[" + r + "]");
            long[] a = new long[c.n()];
            Tandem g = c.generator();
            g.nextLongs(a, c.word("range"));
            assertArrayEquals(c.hex("values"), a, c.id());
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

    /** The float normal evaluates the same formula on two float draws, to a few ulps. */
    @Test
    void floatNormalFollowsTheFormula() {
        Tandem g = Tandem.seed(42), probe = Tandem.seed(42);
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
