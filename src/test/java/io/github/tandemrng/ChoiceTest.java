package io.github.tandemrng;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** The law of the weighted choice. {@link ConformanceTest} checks its values against tandem-c. */
class ChoiceTest {
    /** Only relative mass matters: a power-of-two scale changes no table entry. */
    @Test
    void scalingTheWeightsByAPowerOfTwoKeepsTheTable() {
        double[] w = {0.5, 3, 0, 1, 7, 2.25, 0.1, 4, 1, 6}, big = new double[w.length];
        for (int i = 0; i < w.length; i++) big[i] = w[i] * 0x1p600;
        ChoiceTable a = new ChoiceTable(w), b = new ChoiceTable(big);
        assertEquals(a.capacity(), b.capacity());
        assertArrayEquals(a.cut(), b.cut());
        assertArrayEquals(a.alias(), b.alias());
    }

    /**
     * A zero weight never appears. Nine positive weights leave 8 degrees of freedom, whose
     * 0.0005 and 0.9995 quantiles are 0.71 and 27.87.
     */
    @Test
    void countsFollowTheWeights() {
        double[] w = {0.5, 3, 0, 1, 7, 2.25, 0.1, 4, 1, 6};
        int n = 1_000_000;
        int[] a = new int[n];
        Tandem.seed(2028).fillChoice(a, new ChoiceTable(w));
        long[] counts = new long[w.length];
        for (int x : a) counts[x]++;
        assertEquals(0, counts[2]);
        double sum = 0, chi2 = 0;
        for (double x : w) sum += x;
        for (int i = 0; i < w.length; i++) {
            if (w[i] == 0) continue;
            double expected = n * w[i] / sum;
            chi2 += (counts[i] - expected) * (counts[i] - expected) / expected;
        }
        assertTrue(0.71 < chi2 && chi2 < 27.87, "chi2 " + chi2);
    }
}
