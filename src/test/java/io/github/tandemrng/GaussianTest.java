package io.github.tandemrng;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import org.junit.jupiter.api.Test;

/** Normals come in Box-Muller pairs: scalar calls, pair calls and fills give one sequence. */
class GaussianTest {
    @Test
    void scalarCallsEqualTheFill() {
        double[] want = new double[1001];
        Tandem fill = Tandem.seed(5);
        fill.fillGaussian(want);
        Tandem g = Tandem.seed(5);
        for (int i = 0; i < want.length; i++) assertEquals(want[i], g.nextGaussian(), "double " + i);
        // An odd count consumed a whole last pair in the fill, the scalar loop kept its sine half.
        assertEquals(fill.position(), g.position());

        float[] wantF = new float[1001];
        Tandem fillF = Tandem.seed(5);
        fillF.fillGaussian(wantF);
        Tandem h = Tandem.seed(5);
        for (int i = 0; i < wantF.length; i++) assertEquals(wantF[i], h.nextGaussianFloat(), "float " + i);
        assertEquals(fillF.position(), h.position());
    }

    @Test
    void pairCallsEqualTheFill() {
        double[] want = new double[600];
        Tandem.seed(6).fillGaussian(want);
        Tandem g = Tandem.seed(6);
        float[] wantF = new float[600];
        Tandem.seed(6).fillGaussian(wantF);
        Tandem h = Tandem.seed(6);
        for (int j = 0; j < 300; j++) {
            assertArrayEquals(new double[] {want[2 * j], want[2 * j + 1]}, g.nextGaussian2());
            assertArrayEquals(new float[] {wantF[2 * j], wantF[2 * j + 1]}, h.nextGaussianFloat2());
        }
    }

    @Test
    void oddFillUsesTheCosineHalfAndConsumesBothUniforms() {
        double[] four = new double[4], three = new double[3];
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
        double first = Tandem.seed(9).nextGaussian();
        Tandem g = Tandem.seed(9);
        g.nextGaussian();
        g.setPosition(0);
        assertEquals(first, g.nextGaussian());

        Tandem f = Tandem.seed(9);
        f.nextGaussianFloat();
        f.setPosition(0);
        assertEquals(Tandem.seed(9).nextGaussianFloat(), f.nextGaussianFloat());
    }

    @Test
    void keptHalfIsDroppedByDerivingChildren() {
        for (int how = 0; how < 4; how++) {
            Tandem g = Tandem.seed(10);
            g.nextGaussian();
            switch (how) {
                case 0 -> g.split(3);
                case 1 -> g.sub(3);
                case 2 -> g.fork(0);
                default -> g.fork(2);
            }
            Tandem fresh = new Tandem(g.keyLo(), g.keyHi(), g.position(), 32);
            assertEquals(fresh.nextGaussian(), g.nextGaussian(), "case " + how);
        }
    }

    @Test
    void keptHalfIsNotSerialized() throws IOException, ClassNotFoundException {
        Tandem g = Tandem.seed(11);
        g.nextGaussian();
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (ObjectOutputStream out = new ObjectOutputStream(bytes)) {
            out.writeObject(g);
        }
        Tandem h;
        try (ObjectInputStream in = new ObjectInputStream(new ByteArrayInputStream(bytes.toByteArray()))) {
            h = (Tandem) in.readObject();
        }
        Tandem fresh = new Tandem(g.keyLo(), g.keyHi(), g.position(), 32);
        assertEquals(fresh.nextGaussian(), h.nextGaussian());
    }
}
