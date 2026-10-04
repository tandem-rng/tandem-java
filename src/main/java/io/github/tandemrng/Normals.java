package io.github.tandemrng;

/**
 * Box-Muller pairs in explicit fused multiply-adds, the same formulation and coefficients as
 * {@code tandem.c}, so every compiler, JVM and target gives the same bits. Every operation is a
 * correctly rounded add, multiply, divide, square root or {@link Math#fma}.
 *
 * <p>For uniforms {@code a} and {@code b}: {@code r = sqrt(-2 ln(1 - a))} and the pair is
 * {@code r cos(2 pi b), r sin(2 pi b)}, cosine first.
 */
final class Normals {
    private Normals() {}

    private static final long SIGN64 = 0x8000000000000000L;
    private static final int SIGN32 = 0x80000000;

    /** -2 ln x for x in (0, 1], the logarithm of the normals and the exponentials. */
    private static double neg2Log(double x) {
        // x = mant 2^k with mant in [sqrt(1/2), sqrt 2), from the bits: adding the bits of
        // sqrt(1/2) to the exponent field makes the mantissa rollover pick k.
        long ix = Double.doubleToRawLongBits(x) + 0x00095f6200000000L;
        double nk = (double) (1023 - (ix >>> 52));
        ix = (ix & 0x000fffffffffffffL) + 0x3fe6a09e00000000L;
        double mant = Double.longBitsToDouble(ix);
        double s = (mant - 1.0) / (mant + 1.0), zz = s * s;
        double p = Math.fma(zz, Math.fma(zz, Math.fma(zz, Math.fma(zz, Math.fma(zz, Math.fma(zz,
                0.08312363319426472, 0.09070001083303751), 0.11111433317907482),
                0.14285712049336274), 0.2000000000566491), 0.33333333333331017), 1.0);
        // -2 ln x = 2 nk ln 2 - 4 s p, with ln 2 split so that nk * ln2_hi is exact.
        return Math.fma(nk, 3.816429394731813e-10, Math.fma(nk, 1.3862943607382476, (s * -4.0) * p));
    }

    private static float neg2LogF(float x) {
        int ix = Float.floatToRawIntBits(x) + 0x004afb0d;
        float nk = (float) (127 - (ix >>> 23));
        ix = (ix & 0x007fffff) + 0x3f3504f3;
        float mant = Float.intBitsToFloat(ix);
        float s = (mant - 1.0f) / (mant + 1.0f), zz = s * s;
        float p = Math.fma(zz, Math.fma(zz, Math.fma(zz, 0.14275366f, 0.20000061f), 0.33333334f), 1.0f);
        return Math.fma(nk, 2.857213530660374e-06f, Math.fma(nk, 1.38629150390625f, (s * -4.0f) * p));
    }

    /** {@code -ln(1 - u)}; the halving of {@code neg2Log} is exact. */
    static double exponential(double u) {
        return 0.5 * neg2Log(1.0 - u);
    }

    static float exponentialF(float u) {
        return 0.5f * neg2LogF(1.0f - u);
    }

    /** Writes the double pair to {@code out[o]} and {@code out[o + 1]}. */
    static void pair(double a, double b, double[] out, int o) {
        double r = Math.sqrt(neg2Log(1.0 - a));

        // The nearest quarter turn q and the angle left over in [-pi/4, pi/4].
        long q = (long) (b * 4.0 + 0.5);
        double f = Math.fma(-(double) q, 0.25, b), th = f * 6.283185307179586, w = th * th;
        double hs = Math.fma(w, Math.fma(w, Math.fma(w, Math.fma(w, Math.fma(w, 1.5914650986900946e-10,
                -2.5051097984389413e-08), 2.755731600073921e-06), -0.00019841269836630226),
                0.008333333333330813), -0.16666666666666669);
        double hc = Math.fma(w, Math.fma(w, Math.fma(w, Math.fma(w, Math.fma(w, 2.0665708703855164e-09,
                -2.7555858522576447e-07), 2.480158263811954e-05), -0.0013888888882156126),
                0.04166666666663108), -0.4999999999999997);
        double sn = th * Math.fma(w, hs, 1.0), cs = Math.fma(w, hc, 1.0);

        // Rotate by q quarter turns with bit operations: odd q swaps the two, bit 1 of q negates
        // the sine, and bit 1 of q + 1 negates the cosine.
        long sm = -(q & 1L), sb = Double.doubleToRawLongBits(sn), cb = Double.doubleToRawLongBits(cs);
        long xb = (sb & sm) | (cb & ~sm), yb = (cb & sm) | (sb & ~sm);
        xb ^= ((q + 1) << 62) & SIGN64;
        yb ^= (q << 62) & SIGN64;
        out[o] = r * Double.longBitsToDouble(xb);
        out[o + 1] = r * Double.longBitsToDouble(yb);
    }

    /** Writes the float pair to {@code out[o]} and {@code out[o + 1]}, all in float arithmetic. */
    static void pairF(float a, float b, float[] out, int o) {
        float r = (float) Math.sqrt(neg2LogF(1.0f - a));

        int q = (int) (b * 4.0f + 0.5f);
        float f = Math.fma(-(float) q, 0.25f, b);
        // 2 pi as a float pair, so that the angle is good to the last bit of the float.
        float th = Math.fma(f, -1.7484555e-7f, f * 6.2831855f), w = th * th;
        float hs = Math.fma(w, Math.fma(w, Math.fma(w, 2.72499e-06f, -0.00019840087f), 0.008333332f), -0.16666667f);
        float hc = Math.fma(w, Math.fma(w, Math.fma(w, 2.4463761e-05f, -0.0013887589f), 0.04166665f), -0.5f);
        float sn = th * Math.fma(w, hs, 1.0f), cs = Math.fma(w, hc, 1.0f);

        int sm = -(q & 1), sb = Float.floatToRawIntBits(sn), cb = Float.floatToRawIntBits(cs);
        int xb = (sb & sm) | (cb & ~sm), yb = (cb & sm) | (sb & ~sm);
        xb ^= ((q + 1) << 30) & SIGN32;
        yb ^= (q << 30) & SIGN32;
        out[o] = r * Float.intBitsToFloat(xb);
        out[o + 1] = r * Float.intBitsToFloat(yb);
    }
}
