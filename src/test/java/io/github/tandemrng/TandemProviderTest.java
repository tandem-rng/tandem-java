package io.github.tandemrng;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.apache.commons.rng.UniformRandomProvider;
import org.junit.jupiter.api.Test;

/** The Commons RNG adaptor returns the wrapped generator's values through the interface. */
class TandemProviderTest {
    @Test
    void drawsEqualTheGenerator() {
        UniformRandomProvider p = new TandemProvider(Tandem.seed(5));
        Tandem g = Tandem.seed(5);
        assertEquals(g.nextInt(), p.nextInt());
        assertEquals(g.nextLong(), p.nextLong());
        assertEquals(g.nextBoolean(), p.nextBoolean());
        assertEquals(g.nextFloat(), p.nextFloat());
        assertEquals(g.nextDouble(), p.nextDouble());
        assertEquals(g.nextInt(1000), p.nextInt(1000));
        assertEquals(g.nextInt(-7, 12), p.nextInt(-7, 12));
        assertEquals(g.nextLong(1L << 40), p.nextLong(1L << 40));
        assertEquals(g.nextLong(-3, 1L << 50), p.nextLong(-3, 1L << 50));
    }

    @Test
    void bytesAndStreams() {
        UniformRandomProvider p = new TandemProvider(Tandem.seed(5));
        Tandem g = Tandem.seed(5);
        byte[] a = new byte[300], b = new byte[300];
        p.nextBytes(a, 20, 250);
        g.fill(b, 20, 250);
        assertArrayEquals(b, a);
        assertEquals(g.nextDouble(), p.doubles(1).findFirst().getAsDouble());
    }
}
