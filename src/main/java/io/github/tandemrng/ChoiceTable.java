package io.github.tandemrng;

/**
 * The alias table of a weighted choice, Appendix C of the specification: Walker's alias method
 * in exact integers. {@link Tandem#nextChoice} and {@link Tandem#fillChoice} draw an index
 * {@code i} in {@code [0, m)} with probability proportional to {@code weights[i]}, one 64-bit
 * draw each, with no retry. The table and the indices equal tandem-c's
 * {@code tandem_choice_build} and {@code tandem_fill_choice} in every port.
 *
 * <p>Column {@code j} gives {@code cut[j]} of its capacity {@code S} to index {@code j} and the
 * rest to {@code alias[j]}. The table is immutable, so threads may share it.
 */
public final class ChoiceTable {
    private final long capacity;
    private final long[] cut;
    private final int[] alias;

    /**
     * Builds the table. It consumes no draws.
     *
     * @param weights {@code m >= 1} weights, finite and not negative, at least one positive
     * @throws IllegalArgumentException for other weights
     */
    public ChoiceTable(double[] weights) {
        int m = weights.length;
        if (m == 0) throw new IllegalArgumentException("a choice needs at least one weight");
        double wmax = 0;
        for (double w : weights) {
            if (!(w >= 0 && w < Double.POSITIVE_INFINITY))
                throw new IllegalArgumentException("choice weights must be finite and not negative");
            wmax = Math.max(wmax, w);
        }
        if (wmax == 0) throw new IllegalArgumentException("a choice needs a positive weight");

        // The first pass bounds the sum below 2^64, the second scales it to just below 2^63.
        int t = 63 - nbits(m) - floorLog2(wmax);
        long total = 0;
        for (double w : weights) total += ceilScaled(w, t);
        t += 63 - nbits(total);
        long[] q = new long[m];
        int[] a = new int[m];
        int big = 0;
        total = 0;
        for (int i = 0; i < m; i++) {
            q[i] = ceilScaled(weights[i], t);
            total += q[i];
            if (Long.compareUnsigned(q[i], q[big]) > 0) big = i;
            a[i] = i;
        }
        // The total and the masses may reach 2^63, so they and the capacity are unsigned.
        long d = Long.remainderUnsigned(m - Long.remainderUnsigned(total, m), m);
        q[big] += d;
        long s = Long.divideUnsigned(total + d, m);

        // Vose's pairing in place: q[] holds the masses until a column is paired.
        int l = 0;
        while (Long.compareUnsigned(q[l], s) < 0) l++;
        for (int i = 0; i < m; i++)
            for (int j = i; j <= i && Long.compareUnsigned(q[j], s) < 0; ) {
                a[j] = l;
                q[l] -= s - q[j];
                j = l;
                if (Long.compareUnsigned(q[l], s) < 0)
                    do l++;
                    while (l < m && Long.compareUnsigned(q[l], s) < 0);
            }
        capacity = s;
        cut = q;
        alias = a;
    }

    private static int nbits(long x) {
        return 64 - Long.numberOfLeadingZeros(x);
    }

    /** The integer significand of a finite w > 0: w = s 2^e with 1 <= s < 2^53. */
    private static long significand(double w) {
        long b = Double.doubleToRawLongBits(w), frac = b & ((1L << 52) - 1);
        return (b >>> 52) == 0 ? frac : frac | 1L << 52;
    }

    private static int exponent(double w) {
        int biased = (int) (Double.doubleToRawLongBits(w) >>> 52);
        return biased == 0 ? -1074 : biased - 1075;
    }

    /** floor(log2 w) for finite w > 0, subnormals included. */
    private static int floorLog2(double w) {
        return exponent(w) + nbits(significand(w)) - 1;
    }

    /**
     * ceil(w 2^t) for finite w >= 0, exact. A multiply by 2^t would round a positive w to 0 where
     * the product is subnormal. The caller keeps the result below 2^64.
     */
    private static long ceilScaled(double w, int t) {
        if (w == 0) return 0;
        long s = significand(w);
        int k = exponent(w) + t;
        if (k >= 0) return s << k;
        if (k <= -53) return 1;
        return (s >>> -k) + ((s & ((1L << -k) - 1)) != 0 ? 1 : 0);
    }

    /** The index of the UInt64 draw r. */
    int index(long r) {
        long m = cut.length;
        int j = (int) Math.unsignedMultiplyHigh(r, m);
        long v = Math.unsignedMultiplyHigh(r * m, capacity);
        return Long.compareUnsigned(v, cut[j]) < 0 ? j : alias[j];
    }

    /** Returns the number of weights {@code m}. */
    public int size() {
        return cut.length;
    }

    /** Returns the column capacity {@code S}, an unsigned 64-bit value. */
    public long capacity() {
        return capacity;
    }

    /** Returns a copy of {@code cut}, unsigned 64-bit values from 0 to {@code S}. */
    public long[] cut() {
        return cut.clone();
    }

    /** Returns a copy of {@code alias}, indices in {@code [0, m)}. */
    public int[] alias() {
        return alias.clone();
    }
}
