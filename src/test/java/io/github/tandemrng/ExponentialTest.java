package io.github.tandemrng;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.random.RandomGenerator;
import org.junit.jupiter.api.Test;

/** Exponentials {@code -ln(1 - u)} are bit identical to tandem-c and tandem-cuda. */
class ExponentialTest {
    private static final long EXPECTED_HASH = 0x47f8f98297d94ee2L;
    private static final int N = 64;

    private static long[] bits(double[] a) {
        return Arrays.stream(a).mapToLong(Double::doubleToRawLongBits).toArray();
    }

    private static int[] bits(float[] a) {
        int[] b = new int[a.length];
        for (int i = 0; i < a.length; i++) b[i] = Float.floatToRawIntBits(a[i]);
        return b;
    }

    private static Tandem at(long start) {
        Tandem g = Tandem.seed(42);
        g.setPosition(start);
        return g;
    }

    @Test
    void fixtureFromTandemCIsExact() {
        for (int c = 0; c < Derived.EXP_START.length; c++) {
            long start = Derived.EXP_START[c];
            double[] fill = new double[N], scalar = new double[N];
            Tandem a = at(start);
            a.fillExponential(fill);
            RandomGenerator b = at(start);
            for (int i = 0; i < N; i++) scalar[i] = b.nextExponential();
            assertArrayEquals(bits(Derived.EXP[c]), bits(fill), "fill " + c);
            assertArrayEquals(bits(Derived.EXP[c]), bits(scalar), "scalar " + c);
            assertEquals(Derived.EXP_END[c], a.position());
            assertEquals(Derived.EXP_END[c], ((Tandem) b).position());

            float[] fillF = new float[N], scalarF = new float[N];
            Tandem d = at(start), e = at(start);
            d.fillExponential(fillF);
            for (int i = 0; i < N; i++) scalarF[i] = e.nextExponentialFloat();
            assertArrayEquals(bits(Derived.EXPF[c]), bits(fillF), "float fill " + c);
            assertArrayEquals(bits(Derived.EXPF[c]), bits(scalarF), "float scalar " + c);
            assertEquals(Derived.EXPF_END[c], d.position());
            assertEquals(Derived.EXPF_END[c], e.position());
        }
    }

    private static long fnv(long h, long bits, int bytes) {
        for (int i = 0; i < bytes; i++) h = (h ^ ((bits >>> (8 * i)) & 0xff)) * 0x100000001b3L;
        return h;
    }

    /** The bytes are those of tandem-c's tools/dump_exponentials.c, SHA-256 5c035a4e... . */
    @Test
    void fillsMatchTheCReferenceHash() {
        int n = 1_000_000;
        double[] d = new double[n];
        float[] f = new float[n];
        long h = 0xcbf29ce484222325L;
        for (long start : new long[] {0, 1, 77, 12345, 1L << 30}) {
            Tandem g = Tandem.seed(2026, 7);
            g.setPosition(start);
            g.fillExponential(d);
            for (double x : d) h = fnv(h, Double.doubleToRawLongBits(x), 8);
            g.fillExponential(f);
            for (float x : f) h = fnv(h, Float.floatToRawIntBits(x) & 0xffffffffL, 4);
        }
        assertEquals(EXPECTED_HASH, h);
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
