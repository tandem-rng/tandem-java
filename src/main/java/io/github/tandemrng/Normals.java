package io.github.tandemrng;

/**
 * The derived draws of spec Appendix A in the formulation and coefficients of {@code tandem.c},
 * so every JVM and target gives the same bits. Every operation is a correctly rounded add,
 * multiply, divide, square root or {@link Math#fma}, and Java never fuses the others.
 *
 * <p>Double normals are the 1024-layer ziggurat of one 64-bit draw. Float normals are
 * Box-Muller pairs: for uniforms {@code a} and {@code b}, {@code r = sqrt(-2 ln(1 - a))} and the
 * pair is {@code r cos(2 pi b), r sin(2 pi b)}, cosine first.
 */
final class Normals {
    private Normals() {}

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

    /**
     * The ziggurat normal of the 64-bit draw {@code r} whose global draw index is {@code g}
     * under the key of {@code keyed}. Bits 0-9 pick the layer, bit 10 the sign, bits 11-63 are
     * the magnitude {@code ra}, and {@code +-ra W[i]} is the normal when {@code ra < K[i]},
     * 99.57 % of the time. The sign flips the sign bit, which gives -0.0 for {@code ra = 0}
     * with the sign set, and takes no branch, because the sign is a coin flip.
     */
    static double gaussian(long r, Tandem keyed, long g) {
        int s = (int) r & 2047;
        long ra = r >>> 11;
        double x = ra * Double.longBitsToDouble(ZIG[2 * s + 1]);
        return ra < ZIG[2 * s] ? x : gaussianMiss(r, x, keyed.gaussianFallback(g));
    }

    /**
     * Fills {@code a[off, off + n)} with the ziggurat normals of {@code r[0, n)}, whose global
     * draw indices start at g. The first loop has no call, so C2 unrolls it, and only records
     * the misses, which the second loop finishes.
     */
    static void gaussians(long[] r, int n, double[] a, int off, int[] miss, Tandem keyed, long g) {
        long[] t = ZIG;
        int misses = 0;
        for (int j = 0; j < n; j++) {
            long rj = r[j];
            int s2 = (int) rj << 1 & 4094;
            long ra = rj >>> 11;
            a[off + j] = ra * Double.longBitsToDouble(t[s2 + 1]);
            if (ra >= t[s2]) miss[misses++] = j;
        }
        Tandem fallbacks = misses > 0 ? keyed.gaussianFallbacks() : null;
        for (int q = 0; q < misses; q++) {
            int j = miss[q];
            a[off + j] = gaussianMiss(r[j], a[off + j], fallbacks.split(g + j));
        }
    }

    /**
     * Entry 2s is the threshold K[s mod 1024] and entry 2s + 1 the bits of the width
     * W[s mod 1024] with the sign of bit 10 of s, for the 11 low bits s of a draw. One table
     * keeps both loads on one base address, and the signed width replaces the sign flip:
     * {@code ra * -w} is {@code -(ra * w)} exactly, -0.0 for {@code ra = 0} as well.
     */
    private static final long[] ZIG = new long[4096];

    static {
        for (int s = 0; s < 2048; s++) {
            double w = ZigTables.W[s & 1023];
            ZIG[2 * s] = ZigTables.K[s & 1023];
            ZIG[2 * s + 1] = Double.doubleToRawLongBits((s & 1024) != 0 ? -w : w);
        }
    }

    /**
     * A miss continues on the draws of its own fallback generator: the tail beyond R by
     * Marsaglia's method, else the wedge test {@code ln y < -x^2 / 2}, else a new draw. Draw d
     * of the fallback is its element d by random access, which computes one lane: a miss takes
     * one to three draws, and a scalar draw would generate the generator's whole block cache.
     */
    private static double gaussianMiss(long r, double x, Tandem g) {
        Fallback fallback = new Fallback(g);
        double[] w = ZigTables.W, y = ZigTables.Y;
        long[] k = ZigTables.K;
        long d = 0;
        for (;;) {
            int i = (int) r & 1023;
            if (i == 0) {
                double a, b;
                do {
                    a = exponential(fallback.uniform(d++)) / ZigTables.R;
                    b = exponential(fallback.uniform(d++));
                } while (b + b < a * a);
                double t = ZigTables.R + a;
                return (r & 1024) != 0 ? -t : t;
            }
            double h = y[i] + fallback.uniform(d++) * (y[i + 1] - y[i]);
            if (-0.5 * neg2Log(h) < -0.5 * (x * x)) return x;
            r = fallback.draw(d++);
            i = (int) r & 1023;
            long ra = r >>> 11;
            x = ra * w[i];
            if ((r & 1024) != 0) x = -x;
            if (ra < k[i]) return x;
        }
    }

    /** The draws of a fallback generator at position 0. Draws 2i and 2i + 1 share one F. */
    private static final class Fallback {
        private final Tandem g;
        private final long[] pair = new long[2];
        private long held = -1;

        Fallback(Tandem g) {
            this.g = g;
        }

        long draw(long d) {
            if (d >>> 1 != held) {
                held = d >>> 1;
                g.atLongPair(held, pair);
            }
            return pair[(int) d & 1];
        }

        /** As {@code g.atDouble(d)}. */
        double uniform(long d) {
            return (draw(d) >>> 11) * 0x1p-53;
        }
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
