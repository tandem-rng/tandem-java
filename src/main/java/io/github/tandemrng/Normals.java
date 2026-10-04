package io.github.tandemrng;

/**
 * Box-Muller pairs in plain double and float arithmetic. Every operation is a correctly rounded
 * add, multiply, divide or square root, so the results are the same on every JVM, unlike
 * {@code Math.log}, whose intrinsics may differ by an ulp between platforms. They agree with
 * a libm to a few ulps, which the cross-port tolerances allow.
 */
final class Normals {
    private Normals() {}

    private static final double LN2 = 0.6931471805599453;
    private static final long OFF = 0x3fe6a09e667f3bcdL;
    private static final long EXP_MASK = 0xfff0000000000000L;
    private static final double QUARTER_PI = 0.7853981633974483;
    private static final double R = 0.7071067811865476;
    // cos and sin of k pi / 4.
    private static final double[] OCT_COS = {1, R, 0, -R, -1, -R, 0, R};
    private static final double[] OCT_SIN = {0, R, 1, R, 0, -R, -1, -R};

    // atanh series: log m = 2 (s + s^3/3 + s^5/5 + ...), s = (m - 1) / (m + 1), |s| < 0.1716.
    // The first omitted term is below 1e-17 after ten terms.
    private static final double[] ATANH = new double[10];
    // Taylor series of sin and cos on [0, pi/4], nine terms, first omitted term below 1e-19.
    private static final double[] SIN = new double[9];
    private static final double[] COS = new double[9];

    static {
        for (int k = 0; k < 10; k++) ATANH[k] = 2.0 / (2 * k + 1);
        for (int k = 0; k < 9; k++) {
            SIN[k] = inverseFactorial(2 * k + 1, k);
            COS[k] = inverseFactorial(2 * k, k);
        }
    }

    /** (-1)^k / n!, computed in double from the exact integer product where it fits, else by division. */
    private static double inverseFactorial(int n, int k) {
        double f = 1;
        for (int i = 2; i <= n; i++) f *= i;
        return (k % 2 == 0 ? 1.0 : -1.0) / f;
    }

    /** Sum of c[k] y^k for ten terms, grouped so the fused multiply-adds form a short tree. */
    private static double estrin10(double[] c, double y) {
        double y2 = y * y, y4 = y2 * y2, y8 = y4 * y4;
        double a0 = Math.fma(c[1], y, c[0]), a1 = Math.fma(c[3], y, c[2]);
        double a2 = Math.fma(c[5], y, c[4]), a3 = Math.fma(c[7], y, c[6]);
        double a4 = Math.fma(c[9], y, c[8]);
        double b0 = Math.fma(a1, y2, a0), b1 = Math.fma(a3, y2, a2);
        return Math.fma(a4, y8, Math.fma(b1, y4, b0));
    }

    /** The same for nine terms. */
    private static double estrin9(double[] c, double y) {
        double y2 = y * y, y4 = y2 * y2, y8 = y4 * y4;
        double a0 = Math.fma(c[1], y, c[0]), a1 = Math.fma(c[3], y, c[2]);
        double a2 = Math.fma(c[5], y, c[4]), a3 = Math.fma(c[7], y, c[6]);
        double b0 = Math.fma(a1, y2, a0), b1 = Math.fma(a3, y2, a2);
        return Math.fma(c[8], y8, Math.fma(b1, y4, b0));
    }

    /** Natural log of u in (0, 1], to about 1e-16 relative. */
    static double ln(double u) {
        // Integer-only range reduction to m in [sqrt(1/2), sqrt 2): no data-dependent branch.
        long bits = Double.doubleToRawLongBits(u);
        long i = bits - OFF;
        int e = (int) (i >> 52);
        double m = Double.longBitsToDouble(bits - (i & EXP_MASK));
        double s = (m - 1.0) / (m + 1.0);
        double s2 = s * s;
        return Math.fma(e, LN2, s * estrin10(ATANH, s2));
    }

    /**
     * Writes {@code {r cos(2 pi b), r sin(2 pi b)}} with {@code r = sqrt(-2 ln(1 - a))} to
     * {@code out[o]} and {@code out[o + 1]}.
     */
    static void pair(double a, double b, double[] out, int o) {
        double r = Math.sqrt(-2.0 * ln(1.0 - a));
        sinCos(b, out, o);
        out[o] *= r;
        out[o + 1] *= r;
    }

    /** Writes {@code cos(2 pi b)} and {@code sin(2 pi b)} for b in [0, 1) to {@code out[o]} and {@code out[o + 1]}. */
    static void sinCos(double b, double[] out, int o) {
        // 8b = k + f is exact. The angle is k pi/4 + x with x in [0, pi/4), and a table of the
        // octant's cosine and sine rotates the series values, so there is no branch on the data.
        double t = b * 8.0;
        int k = (int) t;
        double x = (t - k) * QUARTER_PI;
        double x2 = x * x;
        double sx = x * estrin9(SIN, x2), cx = estrin9(COS, x2);
        double ck = OCT_COS[k], sk = OCT_SIN[k];
        out[o] = Math.fma(ck, cx, -sk * sx);
        out[o + 1] = Math.fma(sk, cx, ck * sx);
    }

    private static final float LN2_F = 0.6931472f;
    private static final float[] ATANH_F = new float[6];

    static {
        for (int k = 0; k < 6; k++) ATANH_F[k] = (float) (2.0 / (2 * k + 1));
    }

    /** Natural log of u in (0, 1] in float, to a few ulps. */
    static float lnf(float u) {
        int bits = Float.floatToRawIntBits(u);
        int i = bits - 0x3f3504f3;
        int e = i >> 23;
        float m = Float.intBitsToFloat(bits - (i & 0xff800000));
        float s = (m - 1.0f) / (m + 1.0f);
        float s2 = s * s;
        float p = ATANH_F[5];
        for (int k = 4; k >= 0; k--) p = Math.fma(p, s2, ATANH_F[k]);
        return Math.fma((float) e, LN2_F, s * p);
    }

    /**
     * Writes the float pair to {@code out[o]} and {@code out[o + 1]}: the radius in float, the
     * angle {@code 2 pi b} through the double series and rounded to float, then a float product.
     */
    static void pairF(float a, float b, float[] out, int o) {
        float r = (float) Math.sqrt(-2.0f * lnf(1.0f - a));
        double[] z = new double[2];
        sinCos(b, z, 0);
        out[o] = r * (float) z[0];
        out[o + 1] = r * (float) z[1];
    }
}
