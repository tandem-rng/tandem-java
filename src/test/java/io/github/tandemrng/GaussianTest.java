package io.github.tandemrng;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import org.junit.jupiter.api.Test;

/**
 * Scalar calls and fills give one sequence: one ziggurat draw per double normal, and Box-Muller
 * pairs for float normals.
 */
class GaussianTest {
    @Test
    void scalarCallsEqualTheFill() {
        double[] want = new double[1001];
        Tandem fill = Tandem.seed(5);
        fill.fillGaussian(want);
        Tandem g = Tandem.seed(5);
        for (int i = 0; i < want.length; i++) assertEquals(want[i], g.nextGaussian(), "double " + i);
        assertEquals(fill.position(), g.position());

        // An odd count consumed a whole last pair in the fill, the scalar loop kept its sine half.
        float[] wantF = new float[1001];
        Tandem fillF = Tandem.seed(5);
        fillF.fillGaussian(wantF);
        Tandem h = Tandem.seed(5);
        for (int i = 0; i < wantF.length; i++) assertEquals(wantF[i], h.nextGaussianFloat(), "float " + i);
        assertEquals(fillF.position(), h.position());
    }

    @Test
    void floatPairCallsEqualTheFill() {
        float[] wantF = new float[600];
        Tandem.seed(6).fillGaussian(wantF);
        Tandem h = Tandem.seed(6);
        for (int j = 0; j < 300; j++)
            assertArrayEquals(new float[] {wantF[2 * j], wantF[2 * j + 1]}, h.nextGaussianFloat2());
    }

    /**
     * A double fill cut at any element equals the whole fill, at the first element that misses
     * the fast path and just after it, from a start whose draw index is not the element index.
     */
    @Test
    void doubleFillCutAnywhereEqualsTheWholeFill() {
        Tandem probe = Tandem.seed(12);
        probe.setPosition(77);
        int miss = 0;
        while (true) {
            long r = probe.nextLong();
            if (r >>> 11 >= ZigTables.K[(int) r & 1023]) break;
            miss++;
        }
        double[] whole = new double[3000];
        Tandem w = Tandem.seed(12);
        w.setPosition(77);
        w.fillGaussian(whole);
        for (int cut : new int[] {1, 1001, miss, miss + 1}) {
            double[] parts = new double[whole.length];
            Tandem p = Tandem.seed(12);
            p.setPosition(77);
            p.fillGaussian(parts, 0, cut);
            p.fillGaussian(parts, cut, whole.length - cut);
            assertArrayEquals(whole, parts, "cut " + cut);
            assertEquals(w.position(), p.position());
        }
    }

    @Test
    void emptyDoubleFillAlignsThePosition() {
        Tandem g = Tandem.seed(13);
        g.setPosition(65);
        g.fillGaussian(new double[0]);
        assertEquals(128, g.position());
    }

    @Test
    void oddFloatFillUsesTheCosineHalfAndConsumesBothUniforms() {
        float[] four = new float[4], three = new float[3];
        Tandem a = Tandem.seed(7), b = Tandem.seed(7);
        a.fillGaussian(four);
        b.fillGaussian(three);
        assertArrayEquals(java.util.Arrays.copyOf(four, 3), three);
        assertEquals(a.position(), b.position());
    }

    @Test
    void fillOfASliceWritesOnlyTheSlice() {
        double[] whole = new double[10], want = new double[6];
        Tandem.seed(8).fillGaussian(whole, 2, 6);
        Tandem.seed(8).fillGaussian(want);
        assertArrayEquals(want, java.util.Arrays.copyOfRange(whole, 2, 8));
        assertEquals(0.0, whole[0]);
        assertEquals(0.0, whole[9]);
    }

    @Test
    void keptHalfIsDroppedByASeek() {
        Tandem f = Tandem.seed(9);
        f.nextGaussianFloat();
        f.setPosition(0);
        assertEquals(Tandem.seed(9).nextGaussianFloat(), f.nextGaussianFloat());
    }

    @Test
    void keptHalfIsDroppedByDerivingChildren() {
        for (int how = 0; how < 4; how++) {
            Tandem g = Tandem.seed(10);
            g.nextGaussianFloat();
            switch (how) {
                case 0 -> g.split(3);
                case 1 -> g.sub(3);
                case 2 -> g.fork(0);
                default -> g.fork(2);
            }
            Tandem fresh = new Tandem(g.keyLo(), g.keyHi(), g.position(), 32);
            assertEquals(fresh.nextGaussianFloat(), g.nextGaussianFloat(), "case " + how);
        }
    }

    @Test
    void keptHalfIsNotSerialized() throws IOException, ClassNotFoundException {
        Tandem g = Tandem.seed(11);
        g.nextGaussianFloat();
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (ObjectOutputStream out = new ObjectOutputStream(bytes)) {
            out.writeObject(g);
        }
        Tandem h;
        try (ObjectInputStream in = new ObjectInputStream(new ByteArrayInputStream(bytes.toByteArray()))) {
            h = (Tandem) in.readObject();
        }
        Tandem fresh = new Tandem(g.keyLo(), g.keyHi(), g.position(), 32);
        assertEquals(fresh.nextGaussianFloat(), h.nextGaussianFloat());
    }
}
