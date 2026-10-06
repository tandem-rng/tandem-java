package io.github.tandemrng;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import org.junit.jupiter.api.Test;

/**
 * Exponentials {@code -ln(1 - u)}: long fills against scalar draws and the law of Exp(1).
 * {@link ConformanceTest} checks the values of tandem-c.
 */
class ExponentialTest {
    private static long[] bits(double[] a) {
        return Arrays.stream(a).mapToLong(Double::doubleToRawLongBits).toArray();
    }

    private static int[] bits(float[] a) {
        int[] b = new int[a.length];
        for (int i = 0; i < a.length; i++) b[i] = Float.floatToRawIntBits(a[i]);
        return b;
    }

    @Test
    void cutFillsEqualTheScalarDrawsFromAnUnalignedStart() {
        int max = 3000;
        int[] ns = {1, 2, 3, 1023, 1024, 1025, max}, cuts = {1, 7, 1000, 1024, 2049};
        for (int n : ns) {
            Tandem start = Tandem.seed(7, 9);
            start.nextByte();
            Tandem a = copy(start);
            double[] want = new double[n], got = new double[n];
            for (int i = 0; i < n; i++) want[i] = a.nextExponential();
            Tandem b = copy(start);
            b.fillExponential(got);
            assertArrayEquals(bits(want), bits(got), "n " + n);
            assertEquals(a.position(), b.position());
            for (int cut : cuts) {
                int k = Math.min(cut, n);
                Tandem e = copy(start);
                Arrays.fill(got, 0);
                e.fillExponential(got, 0, k);
                e.fillExponential(got, k, n - k);
                assertArrayEquals(bits(want), bits(got), "n " + n + " cut " + cut);
                assertEquals(a.position(), e.position());
            }

            Tandem a32 = copy(start);
            float[] want32 = new float[n], got32 = new float[n];
            for (int i = 0; i < n; i++) want32[i] = a32.nextExponentialFloat();
            for (int cut : cuts) {
                int k = Math.min(cut, n);
                Tandem e = copy(start);
                e.fillExponential(got32, 0, k);
                e.fillExponential(got32, k, n - k);
                assertArrayEquals(bits(want32), bits(got32), "float n " + n + " cut " + cut);
                assertEquals(a32.position(), e.position());
            }
        }
    }

    private static Tandem copy(Tandem g) {
        return new Tandem(g.keyLo(), g.keyHi(), g.position(), 32);
    }

    @Test
    void emptyFillLeavesThePositionAndTheArray() {
        Tandem g = Tandem.seed(3);
        g.nextByte();
        long pos = g.position();
        double[] d = {1.5, 2.5};
        float[] f = {1.5f, 2.5f};
        g.fillExponential(d, 1, 0);
        g.fillExponential(f, 2, 0);
        g.fillExponential(new double[0]);
        assertEquals(pos, g.position());
        assertArrayEquals(new double[] {1.5, 2.5}, d);
        assertArrayEquals(new float[] {1.5f, 2.5f}, f);
    }

    /**
     * Exp(1) has raw moments E[x^k] = k! and Var(x^k) = (2k)! - (k!)^2. Each sample moment lies
     * within 5 standard errors and the Kolmogorov-Smirnov statistic sqrt(n) D is below 1.95, the
     * 0.1 % point of the Kolmogorov distribution.
     */
    private static void checkExponential(double[] x) {
        double[] fact = {1, 1, 2, 6, 24, 120, 720, 5040, 40320};
        int n = x.length;
        double[] m = new double[5];
        for (double v : x) {
            double xk = 1;
            for (int k = 1; k <= 4; k++) m[k] += xk *= v;
        }
        for (int k = 1; k <= 4; k++) {
            double se = Math.sqrt((fact[2 * k] - fact[k] * fact[k]) / n);
            assertTrue(Math.abs(m[k] / n - fact[k]) < 5 * se, "moment " + k);
        }
        Arrays.sort(x);
        double d = 0;
        for (int i = 0; i < n; i++) {
            double f = -Math.expm1(-x[i]);
            d = Math.max(d, Math.max(f - (double) i / n, (double) (i + 1) / n - f));
        }
        assertTrue(d * Math.sqrt(n) < 1.95, "KS " + d * Math.sqrt(n));
    }

    @Test
    void samplesFollowTheStandardExponential() {
        int n = 10_000_000;
        double[] x = new double[n];
        float[] f = new float[n];
        Tandem g = Tandem.seed(2026, 10);
        g.fillExponential(x);
        checkExponential(x);
        g.fillExponential(f);
        for (int i = 0; i < n; i++) x[i] = f[i];
        checkExponential(x);
    }
}
