package io.github.tandemrng;

import java.io.IOException;
import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.util.Objects;
import java.util.random.RandomGenerator;
import java.util.stream.Stream;

/**
 * Tandem8x32, a noncryptographic pseudorandom number generator, as defined by
 * <a href="https://github.com/tandem-rng/spec">the specification</a>. The output is the
 * specified stream bit for bit.
 *
 * <p>A generator is its transport form, a 128-bit key, a 64-bit bit position and the chunk
 * length {@code K}, plus a cache of the current 1024-bit row. The cache is a pure function of
 * the transport form, so it is not serialized and never changes a drawn value.
 *
 * <p>Positions are unsigned 64-bit bit offsets held in a {@code long}. Every draw first aligns
 * the position to the width of its type, as section 5 of the specification requires.
 *
 * <p>A generator is not thread-safe. Give each thread its own child from {@link #split(long)},
 * {@link #fork(int)} or {@link #split()}.
 *
 * <p>This class implements {@link RandomGenerator.SplittableGenerator}, so it works with
 * {@code ints()}, {@code doubles()} and, on JDK 21 and later, {@code Collections.shuffle}. The
 * interface's default {@code nextGaussian} is overridden by the Box-Muller transform of
 * {@link #nextGaussian()}, which agrees with every other Tandem port. {@code nextInt(int)} and
 * {@code nextLong(long)} use Lemire's method and likewise agree with the other ports.
 */
public final class Tandem implements RandomGenerator.SplittableGenerator, Serializable {
    private static final long serialVersionUID = 1L;

    /** The chunk length of the canonical variant Tandem8x32-K32. */
    public static final int DEFAULT_CHUNK_LENGTH = 32;
    /** The largest chunk length the specification allows. */
    public static final int MAX_CHUNK_LENGTH = 65536;

    private static final int CLOCK_WEYL = 0x9e3779b9;
    private static final int DOMAIN_STREAM = 0x9e3779b9;
    private static final int DOMAIN_SPLIT = 0xbb67ae85;
    private static final int DOMAIN_FORK = 0xd2511f53;
    private static final int DOMAIN_FOLD = 0xcd9e8d57;
    private static final int DOMAIN_SEED = 0xa54ff53a;
    private static final int AUX_STREAM = 0x94d049bb;
    private static final int[] RC = {
        0xd17cc1b7, 0xa7220a94, 0xfe13abe8, 0xfa9a6ee0, 0xedb14acc, 0x9e21c820, 0xff28b1d5, 0xef5de2b0
    };
    private static final long MASK32 = 0xffffffffL;
    private static final double TWO_PI = 6.283185307179586;

    private final int k0, k1, k2, k3;
    private final int chunk;
    private long pos;
    private transient Cache cache;
    /** Kept sine halves of the last scalar Box-Muller pairs, see {@link #nextGaussian()}. */
    private transient boolean hasSpare, hasSpareF;
    private transient double spare;
    private transient float spareF;

    private void dropSpares() {
        hasSpare = false;
        hasSpareF = false;
    }

    /**
     * A block of consecutive rows in stream order, and the states of the eight lanes of their
     * group, four exposed and four hidden words per lane. Each lane steps through a whole block
     * with its state in registers, which is why the cache holds a block and not a single row.
     */
    private static final class Cache {
        final int[] o = new int[32];
        final int[] h = new int[32];
        final int[] buf = new int[BLOCK_ROWS * 32];
        final int[] scratch = new int[8];
        /** First row held in {@code buf} and how many rows it holds. */
        long start;
        int count;
        /** The group the lane states belong to and the number of steps they have taken. */
        long group = -1;
        int next;
    }

    private static final int BLOCK_ROWS = 32;

    // ---- Construction and transport ---------------------------------------------------------

    /**
     * Builds a generator from a raw key.
     *
     * @param key four 32-bit key words
     * @param position start bit position, from 0 to 2^63 - 1
     * @param chunkLength power of two from 1 to 65536
     */
    public Tandem(int[] key, long position, int chunkLength) {
        this(fourWords(key)[0], key[1], key[2], key[3], position, chunkLength);
    }

    private static int[] fourWords(int[] key) {
        if (key.length != 4) throw new IllegalArgumentException("key needs four words");
        return key;
    }

    /**
     * Builds a generator from a raw 128-bit key. Word 0 is the low half of {@code keyLo}.
     *
     * @param position start bit position, from 0 to 2^63 - 1
     * @param chunkLength power of two from 1 to 65536
     */
    public Tandem(long keyLo, long keyHi, long position, int chunkLength) {
        this((int) keyLo, (int) (keyLo >>> 32), (int) keyHi, (int) (keyHi >>> 32), position, chunkLength);
    }

    private Tandem(int k0, int k1, int k2, int k3, long position, int chunkLength) {
        if (position < 0) throw new IllegalArgumentException("position must be below 2^63");
        checkChunkLength(chunkLength);
        this.k0 = k0;
        this.k1 = k1;
        this.k2 = k2;
        this.k3 = k3;
        this.pos = position;
        this.chunk = chunkLength;
    }

    private static void checkChunkLength(int k) {
        if (k < 1 || k > MAX_CHUNK_LENGTH || (k & (k - 1)) != 0)
            throw new IllegalArgumentException("chunk length must be a power of two from 1 to 65536");
    }

    /** Seeds from an unsigned 64-bit value with the specification's whitening, at {@code K = 32}. */
    public static Tandem seed(long seed) {
        return seed(seed, 0L, DEFAULT_CHUNK_LENGTH);
    }

    /** Seeds from the 128-bit value {@code hi * 2^64 + lo}, whitened, at {@code K = 32}. */
    public static Tandem seed(long lo, long hi) {
        return seed(lo, hi, DEFAULT_CHUNK_LENGTH);
    }

    /** Seeds from the 128-bit value {@code hi * 2^64 + lo}, whitened, with chunk length {@code K}. */
    public static Tandem seed(long lo, long hi, int chunkLength) {
        int[] s = {0, 0, DOMAIN_SEED, 0, (int) lo, (int) (lo >>> 32), (int) hi, (int) (hi >>> 32)};
        f(s);
        return new Tandem(s[0], s[1], s[2], s[3], 0L, chunkLength);
    }

    /** Returns the key as four words. */
    public int[] key() {
        return new int[] {k0, k1, k2, k3};
    }

    /** Returns key words 0 and 1, word 0 in the low half. */
    public long keyLo() {
        return (k0 & MASK32) | ((long) k1 << 32);
    }

    /** Returns key words 2 and 3, word 2 in the low half. */
    public long keyHi() {
        return (k2 & MASK32) | ((long) k3 << 32);
    }

    /** Returns the bit position as an unsigned 64-bit value. */
    public long position() {
        return pos;
    }

    /** Returns the chunk length {@code K}. */
    public int chunkLength() {
        return chunk;
    }

    /** Moves to a bit position from 0 to 2^63 - 1. */
    public void setPosition(long position) {
        if (position < 0) throw new IllegalArgumentException("position must be below 2^63");
        pos = position;
        dropSpares();
    }

    /** Returns an independent generator with the same key, position and chunk length. */
    public Tandem copy() {
        return new Tandem(k0, k1, k2, k3, pos, chunk);
    }

    @Override
    public String toString() {
        return String.format(
                "Tandem8x32-K%d[key=%08x%08x%08x%08x, position=%s]",
                chunk, k0, k1, k2, k3, Long.toUnsignedString(pos));
    }

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        if (chunk < 1 || chunk > MAX_CHUNK_LENGTH || (chunk & (chunk - 1)) != 0)
            throw new InvalidObjectException("invalid chunk length " + chunk);
    }

    // ---- Step and seeding function ----------------------------------------------------------

    /** T on s = (o0..o3, h0..h3). */
    static void t(int[] s) {
        long p0 = (s[0] & MASK32) * ((s[4] | 1) & MASK32);
        long p1 = (s[2] & MASK32) * ((s[5] | 1) & MASK32);
        int lo0 = (int) p0, hi0 = (int) (p0 >>> 32);
        int lo1 = (int) p1, hi1 = (int) (p1 >>> 32);
        int n0 = s[1] ^ hi1 ^ lo1;
        int n1 = Integer.rotateLeft(lo1, 16) ^ s[6];
        int n2 = s[3] ^ hi0 ^ lo0;
        int n3 = Integer.rotateLeft(lo0, 16) ^ s[7];
        int h0 = s[4] ^ Integer.rotateLeft(s[5], 7);
        int h1 = s[5] ^ Integer.rotateLeft(s[6], 13);
        int h2 = s[6] ^ Integer.rotateLeft(s[7], 22);
        int h3 = s[7] ^ Integer.rotateLeft(h0, 3);
        s[0] = n0;
        s[1] = n1;
        s[2] = n2;
        s[3] = n3;
        s[4] = (h0 + CLOCK_WEYL) ^ n0;
        s[5] = h1;
        s[6] = h2;
        s[7] = h3;
    }

    /** F on s = (o0..o3, h0..h3), in place. */
    static void f(int[] s) {
        for (int r = 0; r < 8; r++) {
            t(s);
            s[0] ^= RC[r];
            for (int w = 0; w < 4; w++) {
                int x = s[w];
                s[w] = s[4 + w];
                s[4 + w] = x;
            }
        }
    }

    static void fKeyed(int k0,int k1, int k2, int k3, long counter, int domain, int aux, int[] s) {
        s[0] = (int) counter;
        s[1] = (int) (counter >>> 32);
        s[2] = domain;
        s[3] = aux;
        s[4] = k0;
        s[5] = k1;
        s[6] = k2;
        s[7] = k3;
        f(s);
    }

    /**
     * Steps every lane {@code rows} times and writes the exposed halves to {@code buf} in
     * stream order: row j, lane l, word w at {@code 32 j + 4 l + w}.
     */
    private static void generate(int[] o, int[] h, int[] buf, int rows) {
        for (int l = 0; l < 8; l++) {
            int o0 = o[4 * l], o1 = o[4 * l + 1], o2 = o[4 * l + 2], o3 = o[4 * l + 3];
            int h0 = h[4 * l], h1 = h[4 * l + 1], h2 = h[4 * l + 2], h3 = h[4 * l + 3];
            for (int j = 0, b = 4 * l; j < rows; j++, b += 32) {
                long p0 = (o0 & MASK32) * ((h0 | 1) & MASK32);
                long p1 = (o2 & MASK32) * ((h1 | 1) & MASK32);
                int n0 = o1 ^ (int) (p1 >>> 32) ^ (int) p1;
                int n1 = Integer.rotateLeft((int) p1, 16) ^ h2;
                int n2 = o3 ^ (int) (p0 >>> 32) ^ (int) p0;
                int n3 = Integer.rotateLeft((int) p0, 16) ^ h3;
                h0 ^= Integer.rotateLeft(h1, 7);
                h1 ^= Integer.rotateLeft(h2, 13);
                h2 ^= Integer.rotateLeft(h3, 22);
                h3 ^= Integer.rotateLeft(h0, 3);
                h0 = (h0 + CLOCK_WEYL) ^ n0;
                o0 = n0;
                o1 = n1;
                o2 = n2;
                o3 = n3;
                buf[b] = o0;
                buf[b + 1] = o1;
                buf[b + 2] = o2;
                buf[b + 3] = o3;
            }
            o[4 * l] = o0;
            o[4 * l + 1] = o1;
            o[4 * l + 2] = o2;
            o[4 * l + 3] = o3;
            h[4 * l] = h0;
            h[4 * l + 1] = h1;
            h[4 * l + 2] = h2;
            h[4 * l + 3] = h3;
        }
    }

    // ---- Rows -------------------------------------------------------------------------------

    /**
     * Returns the offset in the cache buffer of the row that holds bit position p, producing
     * the block of that row first if needed.
     */
    private int locate(long p) {
        Cache c = cache;
        long r = p >>> 10;
        if (c != null) {
            long d = r - c.start;
            if (d >= 0 && d < c.count) return (int) d << 5;
        }
        return load(r);
    }

    private int load(long r) {
        Cache c = cache;
        if (c == null) cache = c = new Cache();
        c.count = 0;
        int shift = Integer.numberOfTrailingZeros(chunk);
        long g = r >>> shift;
        int rows = Math.min(chunk, BLOCK_ROWS);
        int first = (int) (r & (chunk - 1L)) & -rows;
        // A block is only reachable by stepping forward, so a backward or cross-group move reseeds.
        if (c.group != g || c.next > first) {
            for (int l = 0; l < 8; l++) {
                fKeyed(k0, k1, k2, k3, 8L * g + l, DOMAIN_STREAM, AUX_STREAM, c.scratch);
                System.arraycopy(c.scratch, 0, c.o, 4 * l, 4);
                System.arraycopy(c.scratch, 4, c.h, 4 * l, 4);
            }
            c.group = g;
            c.next = 0;
        }
        for (; c.next <= first; c.next += rows) generate(c.o, c.h, c.buf, rows);
        c.start = (g << shift) + first;
        c.count = rows;
        return (int) (r - c.start) << 5;
    }

    /** The word at bit position p, which need not be aligned: the 32-bit word that contains it. */
    private int word(long p) {
        int base = locate(p);
        return cache.buf[base + ((int) (p >>> 5) & 31)];
    }

    /** The w bits at bit position p, aligned to w, for w up to 64, without touching the cache. */
    private long rawAt(long p, int w) {
        long r = p >>> 10;
        int shift = Integer.numberOfTrailingZeros(chunk);
        int[] s = new int[8];
        fKeyed(k0, k1, k2, k3, 8L * (r >>> shift) + ((p >>> 7) & 7), DOMAIN_STREAM, AUX_STREAM, s);
        for (long j = r & (chunk - 1L); j >= 0; j--) t(s);
        return extract(s, (int) (p >>> 5) & 3, (int) (p & 31), w);
    }

    private static long extract(int[] x, int word, int bit, int w) {
        if (w == 64) return (x[word] & MASK32) | ((long) x[word + 1] << 32);
        return (x[word] >>> bit) & (-1 >>> (32 - w)) & MASK32;
    }

    /** The w bits at the aligned position p, through the row cache. */
    private long rawCached(long p, int w) {
        int base = locate(p);
        return extract(cache.buf, base + ((int) (p >>> 5) & 31), (int) (p & 31), w);
    }

    private static long align(long p, int w) {
        return (p + (w - 1)) & -(long) w;
    }

    // ---- Scalar draws -----------------------------------------------------------------------

    /** Draws one bit. */
    @Override
    public boolean nextBoolean() {
        long p = pos;
        pos = p + 1;
        return ((word(p) >>> (int) (p & 31)) & 1) != 0;
    }

    /** Draws a signed 8-bit integer. */
    public byte nextByte() {
        long p = align(pos, 8);
        pos = p + 8;
        return (byte) (word(p) >>> (int) (p & 31));
    }

    /** Draws a signed 16-bit integer. */
    public short nextShort() {
        long p = align(pos, 16);
        pos = p + 16;
        return (short) (word(p) >>> (int) (p & 31));
    }

    /** Draws a signed 32-bit integer. */
    @Override
    public int nextInt() {
        long p = align(pos, 32);
        pos = p + 32;
        return word(p);
    }

    /** Draws a signed 64-bit integer. */
    @Override
    public long nextLong() {
        long p = align(pos, 64);
        pos = p + 64;
        int i = locate(p) + ((int) (p >>> 5) & 31);
        int[] x = cache.buf;
        return (x[i] & MASK32) | ((long) x[i + 1] << 32);
    }

    /** Draws a 128-bit integer as {@code {lo, hi}}. */
    public long[] nextLong128() {
        long p = align(pos, 128);
        pos = p + 128;
        int i = locate(p) + ((int) (p >>> 5) & 31);
        int[] x = cache.buf;
        return new long[] {(x[i] & MASK32) | ((long) x[i + 1] << 32), (x[i + 2] & MASK32) | ((long) x[i + 3] << 32)};
    }

    /** Draws a float in [0, 1) with 24 random bits. */
    @Override
    public float nextFloat() {
        return toFloat(nextInt());
    }

    /** Draws a double in [0, 1) with 53 random bits. */
    @Override
    public double nextDouble() {
        return toDouble(nextLong());
    }

    /** Draws a binary16 value in [0, 1) with 11 random bits, as its bit pattern. */
    public short nextFloat16Bits() {
        return toFloat16Bits(nextShort());
    }

    /** Draws a Unicode scalar value, uniform over the 1112064 non-surrogate code points. */
    public int nextCodePoint() {
        return toCodePoint(nextLong());
    }

    /** Draws a complex number as {@code {re, im}}, two float draws. */
    public float[] nextComplexFloat() {
        float re = nextFloat();
        return new float[] {re, nextFloat()};
    }

    /** Draws a complex number as {@code {re, im}}, two double draws. */
    public double[] nextComplexDouble() {
        double re = nextDouble();
        return new double[] {re, nextDouble()};
    }

    private static double toDouble(long raw) {
        return (raw >>> 11) * 0x1p-53;
    }

    private static float toFloat(int raw) {
        return (raw >>> 8) * 0x1p-24f;
    }

    /** (raw >> 5) * 2^-11 is zero or a normal binary16 whose significand is the 11-bit integer, so the encoding is exact. */
    private static short toFloat16Bits(short raw) {
        int k = (raw & 0xffff) >>> 5;
        if (k == 0) return 0;
        int m = 31 - Integer.numberOfLeadingZeros(k);
        return (short) (((m + 4) << 10) | ((k << (10 - m)) & 0x3ff));
    }

    /** floor(raw * 1112064 / 2^64) for unsigned raw, then skip the surrogate range. */
    private static int toCodePoint(long raw) {
        long m = 1112064L;
        long hi = Math.multiplyHigh(raw, m) + ((raw >> 63) & m);
        return (int) (hi < 0xd800 ? hi : hi + 0x800);
    }

    // ---- Bounded integers and normals -------------------------------------------------------

    /**
     * Draws uniformly from {@code [0, range)} by Lemire's multiply and reject on 32-bit draws,
     * with {@code range} read as unsigned. The result is unsigned too. Not part of the
     * specification. It returns the values of {@code below_u32} in the other ports.
     *
     * @throws IllegalArgumentException if {@code range} is 0
     */
    public int belowU32(int range) {
        if (range == 0) throw new IllegalArgumentException("range must be nonzero");
        long r = range & MASK32;
        long m = (nextInt() & MASK32) * r;
        if ((m & MASK32) < r) {
            long t = Integer.remainderUnsigned(-range, range) & MASK32;
            while ((m & MASK32) < t) m = (nextInt() & MASK32) * r;
        }
        return (int) (m >>> 32);
    }

    /**
     * Draws uniformly from {@code [0, range)} by Lemire's multiply and reject on 64-bit draws,
     * with {@code range} read as unsigned. The result is unsigned too. Not part of the
     * specification. It returns the values of {@code below_u64} in the other ports.
     *
     * @throws IllegalArgumentException if {@code range} is 0
     */
    public long belowU64(long range) {
        if (range == 0) throw new IllegalArgumentException("range must be nonzero");
        long x = nextLong(), lo = x * range;
        if (Long.compareUnsigned(lo, range) < 0) {
            long t = Long.remainderUnsigned(-range, range);
            while (Long.compareUnsigned(lo, t) < 0) {
                x = nextLong();
                lo = x * range;
            }
        }
        return unsignedMultiplyHigh(x, range);
    }

    private static final long PURPOSE_BELOW32 = 0x424c573332L;
    private static final long PURPOSE_BELOW64 = 0x424c573634L;

    /**
     * Fills {@code a[off, off + len)} uniformly from {@code [0, range)}, with {@code range} read
     * as unsigned, by the shared parallel-friendly contract of the other ports. Element i takes
     * draw i of {@link #fill(int[])} and the fill moves the position by exactly {@code len}
     * draws. A rejected draw retries with Lemire's rule on the draws of
     * {@code sub(0x424c573332).split(i)} of a generator with this key and chunk length, from
     * position 0. Without rejections this equals repeated {@link #belowU32}, but after a
     * rejection the sequential scalar loop and the fill differ.
     */
    public void fillBelowU32(int[] a, int off, int len, int range) {
        if (range == 0) throw new IllegalArgumentException("range must be nonzero");
        fill(a, off, len);
        long r = range & MASK32;
        for (int e = 0; e < len; e++) {
            long m = (a[off + e] & MASK32) * r;
            if ((m & MASK32) < r) {
                long t = Integer.remainderUnsigned(-range, range) & MASK32;
                if ((m & MASK32) < t) {
                    Tandem f = new Tandem(k0, k1, k2, k3, 0L, chunk).sub(PURPOSE_BELOW32).split(e);
                    do m = (f.nextInt() & MASK32) * r;
                    while ((m & MASK32) < t);
                }
            }
            a[off + e] = (int) (m >>> 32);
        }
    }

    /** As {@link #fillBelowU32}, on 64-bit draws and {@code sub(0x424c573634)}. */
    public void fillBelowU64(long[] a, int off, int len, long range) {
        if (range == 0) throw new IllegalArgumentException("range must be nonzero");
        fill(a, off, len);
        for (int e = 0; e < len; e++) {
            long x = a[off + e], lo = x * range;
            if (Long.compareUnsigned(lo, range) < 0) {
                long t = Long.remainderUnsigned(-range, range);
                if (Long.compareUnsigned(lo, t) < 0) {
                    Tandem f = new Tandem(k0, k1, k2, k3, 0L, chunk).sub(PURPOSE_BELOW64).split(e);
                    do {
                        x = f.nextLong();
                        lo = x * range;
                    } while (Long.compareUnsigned(lo, t) < 0);
                }
            }
            a[off + e] = unsignedMultiplyHigh(x, range);
        }
    }

    /** Fills {@code a} uniformly from {@code [0, bound)} with {@link #fillBelowU32}. */
    public void nextInts(int[] a, int bound) {
        if (bound <= 0) throw new IllegalArgumentException("bound must be positive");
        fillBelowU32(a, 0, a.length, bound);
    }

    /** Fills {@code a} uniformly from {@code [0, bound)} with {@link #fillBelowU64}. */
    public void nextLongs(long[] a, long bound) {
        if (bound <= 0) throw new IllegalArgumentException("bound must be positive");
        fillBelowU64(a, 0, a.length, bound);
    }

    /** {@code Math.unsignedMultiplyHigh} needs JDK 18. */
    private static long unsignedMultiplyHigh(long a, long b) {
        return Math.multiplyHigh(a, b) + ((a >> 63) & b) + ((b >> 63) & a);
    }

    /** Draws uniformly from {@code [0, bound)} with {@link #belowU32}. */
    @Override
    public int nextInt(int bound) {
        if (bound <= 0) throw new IllegalArgumentException("bound must be positive");
        return belowU32(bound);
    }

    /** Draws uniformly from {@code [origin, bound)} with {@link #belowU32}. */
    @Override
    public int nextInt(int origin, int bound) {
        if (origin >= bound) throw new IllegalArgumentException("bound must exceed origin");
        // The span is read as unsigned, so it may exceed Integer.MAX_VALUE.
        return origin + belowU32(bound - origin);
    }

    /** Draws uniformly from {@code [0, bound)} with {@link #belowU64}. */
    @Override
    public long nextLong(long bound) {
        if (bound <= 0) throw new IllegalArgumentException("bound must be positive");
        return belowU64(bound);
    }

    /** Draws uniformly from {@code [origin, bound)} with {@link #belowU64}. */
    @Override
    public long nextLong(long origin, long bound) {
        if (origin >= bound) throw new IllegalArgumentException("bound must exceed origin");
        return origin + belowU64(bound - origin);
    }

    /**
     * Returns the cosine half of the next Box-Muller pair, {@code sqrt(-2 ln(1 - a)) cos(2 pi b)}
     * from two double draws {@code a} and {@code b}, and keeps the sine half for the next call,
     * which returns it without drawing. Repeated calls therefore give exactly the sequence of
     * {@link #fillGaussian(double[])}. The kept half is dropped by {@code setPosition},
     * {@code split}, {@code fork} and {@code sub}, and is not serialized, so the first call after
     * deserialization starts a fresh pair. Other draws do not touch it. It uses
     * {@link StrictMath}, so the result is the same on every JVM, and agrees with the other
     * ports to a relative 1e-12.
     */
    @Override
    public double nextGaussian() {
        if (hasSpare) {
            hasSpare = false;
            return spare;
        }
        double a = nextDouble();
        double b = nextDouble();
        double r = StrictMath.sqrt(-2.0 * StrictMath.log(1.0 - a));
        spare = r * StrictMath.sin(TWO_PI * b);
        hasSpare = true;
        return r * StrictMath.cos(TWO_PI * b);
    }

    /** Draws a Box-Muller pair {@code {cos, sin}} from two double draws, ignoring the kept half. */
    public double[] nextGaussian2() {
        double a = nextDouble();
        double b = nextDouble();
        return gaussianPair(a, b);
    }

    private static double[] gaussianPair(double a, double b) {
        double r = StrictMath.sqrt(-2.0 * StrictMath.log(1.0 - a));
        return new double[] {r * StrictMath.cos(TWO_PI * b), r * StrictMath.sin(TWO_PI * b)};
    }

    /**
     * Fills with standard normals in pairs: elements {@code 2j} and {@code 2j + 1} are the cosine
     * and sine halves from uniforms {@code 2j} and {@code 2j + 1} of {@link #fill(double[])}. An
     * odd length uses the cosine half of its last pair and still consumes both uniforms. The
     * kept half of {@link #nextGaussian()} is neither used nor changed.
     */
    public void fillGaussian(double[] a) {
        fillGaussian(a, 0, a.length);
    }

    /** As {@link #fillGaussian(double[])} on {@code a[off, off + len)}. */
    public void fillGaussian(double[] a, int off, int len) {
        Objects.checkFromIndexSize(off, len, a.length);
        double[] u = new double[2 * Math.min((len + 1) / 2, GAUSSIAN_BLOCK)];
        for (int i = off, end = off + len; i < end; ) {
            int pairs = Math.min((end - i + 1) / 2, GAUSSIAN_BLOCK);
            fill(u, 0, 2 * pairs);
            for (int j = 0; j < pairs; j++, i += 2) {
                double[] z = gaussianPair(u[2 * j], u[2 * j + 1]);
                a[i] = z[0];
                if (i + 1 < end) a[i + 1] = z[1];
            }
        }
    }

    private static final int GAUSSIAN_BLOCK = 256;

    /**
     * As {@link #nextGaussian()} computed in float from two float draws, with its own kept sine
     * half. The radius is computed in float, the angle {@code 2 pi b} in double, and the cosine and
     * sine are rounded to float before the float product, as in the other hosts. Results agree
     * with other ports to a few ulps.
     */
    public float nextGaussianFloat() {
        if (hasSpareF) {
            hasSpareF = false;
            return spareF;
        }
        float u = nextFloat();
        float v = nextFloat();
        float r = (float) StrictMath.sqrt(-2.0f * (float) StrictMath.log(1.0f - u));
        double ang = TWO_PI * v;
        spareF = r * (float) StrictMath.sin(ang);
        hasSpareF = true;
        return r * (float) StrictMath.cos(ang);
    }

    /** Draws a float Box-Muller pair {@code {cos, sin}} from two float draws, ignoring the kept half. */
    public float[] nextGaussianFloat2() {
        float u = nextFloat();
        float v = nextFloat();
        return gaussianPairF(u, v);
    }

    private static float[] gaussianPairF(float u, float v) {
        float r = (float) StrictMath.sqrt(-2.0f * (float) StrictMath.log(1.0f - u));
        double ang = TWO_PI * v;
        return new float[] {r * (float) StrictMath.cos(ang), r * (float) StrictMath.sin(ang)};
    }

    /** As {@link #fillGaussian(double[])} in float, from uniforms of {@link #fill(float[])}. */
    public void fillGaussian(float[] a) {
        fillGaussian(a, 0, a.length);
    }

    /** As {@link #fillGaussian(float[])} on {@code a[off, off + len)}. */
    public void fillGaussian(float[] a, int off, int len) {
        Objects.checkFromIndexSize(off, len, a.length);
        float[] u = new float[2 * Math.min((len + 1) / 2, GAUSSIAN_BLOCK)];
        for (int i = off, end = off + len; i < end; ) {
            int pairs = Math.min((end - i + 1) / 2, GAUSSIAN_BLOCK);
            fill(u, 0, 2 * pairs);
            for (int j = 0; j < pairs; j++, i += 2) {
                float[] z = gaussianPairF(u[2 * j], u[2 * j + 1]);
                a[i] = z[0];
                if (i + 1 < end) a[i + 1] = z[1];
            }
        }
    }

    // ---- Random access ----------------------------------------------------------------------

    /** Returns bit {@code i} of the fill that would start here, without moving the position. */
    public boolean atBoolean(long i) {
        return rawAt(pos + i, 1) != 0;
    }

    /** Returns element {@code i} of the {@code fill(byte[])} that would start here. */
    public byte atByte(long i) {
        return (byte) rawAt(align(pos, 8) + 8 * i, 8);
    }

    /** Returns element {@code i} of the {@code fill(short[])} that would start here. */
    public short atShort(long i) {
        return (short) rawAt(align(pos, 16) + 16 * i, 16);
    }

    /** Returns element {@code i} of the {@code fill(int[])} that would start here. */
    public int atInt(long i) {
        return (int) rawAt(align(pos, 32) + 32 * i, 32);
    }

    /** Returns element {@code i} of the {@code fill(long[])} that would start here. */
    public long atLong(long i) {
        return rawAt(align(pos, 64) + 64 * i, 64);
    }

    /** Returns element {@code i} of the {@code fill(float[])} that would start here. */
    public float atFloat(long i) {
        return toFloat(atInt(i));
    }

    /** Returns element {@code i} of the {@code fill(double[])} that would start here. */
    public double atDouble(long i) {
        return toDouble(atLong(i));
    }

    /** Returns element {@code i} of the {@code fillFloat16Bits} that would start here. */
    public short atFloat16Bits(long i) {
        return toFloat16Bits(atShort(i));
    }

    /** Returns element {@code i} of the {@code fillCodePoints} that would start here. */
    public int atCodePoint(long i) {
        return toCodePoint(atLong(i));
    }

    // ---- Fills ------------------------------------------------------------------------------

    /**
     * Aligns the position to w bits, checks that n elements end below 2^64, moves the position
     * to the end and returns the aligned start. It checks before the caller writes any output.
     */
    private long beginFill(int w, int n) {
        long p = align(pos, w);
        if (Long.compareUnsigned(p, pos) < 0) throw new IllegalStateException("position overflow");
        long end = p + (long) w * n;
        if (Long.compareUnsigned(end, p) < 0) throw new IllegalStateException("position overflow");
        pos = end;
        return p;
    }

    /** Fills with the same 32-bit values as repeated {@link #nextInt()}. */
    public void fill(int[] a) {
        fill(a, 0, a.length);
    }

    /** Fills {@code a[off, off + len)} with the same values as repeated {@link #nextInt()}. */
    public void fill(int[] a, int off, int len) {
        Objects.checkFromIndexSize(off, len, a.length);
        long p = beginFill(32, len);
        int i = off, end = off + len;
        for (; i < end && (p & 1023) != 0; i++, p += 32) a[i] = word(p);
        while (end - i >= 32) {
            int base = locate(p), rows = Math.min(cache.count - (base >> 5), (end - i) >> 5);
            System.arraycopy(cache.buf, base, a, i, rows << 5);
            i += rows << 5;
            p += (long) rows << 10;
        }
        for (; i < end; i++, p += 32) a[i] = word(p);
    }

    /** Fills with the same 64-bit values as repeated {@link #nextLong()}. */
    public void fill(long[] a) {
        fill(a, 0, a.length);
    }

    /** Fills {@code a[off, off + len)} with the same values as repeated {@link #nextLong()}. */
    public void fill(long[] a, int off, int len) {
        Objects.checkFromIndexSize(off, len, a.length);
        long p = beginFill(64, len);
        int i = off, end = off + len;
        for (; i < end && (p & 1023) != 0; i++, p += 64) a[i] = rawCached(p, 64);
        while (end - i >= 16) {
            int base = locate(p), rows = Math.min(cache.count - (base >> 5), (end - i) >> 4);
            int[] x = cache.buf;
            for (int k = 0, n = rows << 4; k < n; k++) a[i + k] = (x[base + 2 * k] & MASK32) | ((long) x[base + 2 * k + 1] << 32);
            i += rows << 4;
            p += (long) rows << 10;
        }
        for (; i < end; i++, p += 64) a[i] = rawCached(p, 64);
    }

    /** Fills with the same values as repeated {@link #nextFloat()}. Complex fills use this with length 2n. */
    public void fill(float[] a) {
        fill(a, 0, a.length);
    }

    /** Fills {@code a[off, off + len)} with the same values as repeated {@link #nextFloat()}. */
    public void fill(float[] a, int off, int len) {
        Objects.checkFromIndexSize(off, len, a.length);
        long p = beginFill(32, len);
        int i = off, end = off + len;
        for (; i < end && (p & 1023) != 0; i++, p += 32) a[i] = toFloat(word(p));
        while (end - i >= 32) {
            int base = locate(p), rows = Math.min(cache.count - (base >> 5), (end - i) >> 5);
            int[] x = cache.buf;
            for (int k = 0, n = rows << 5; k < n; k++) a[i + k] = toFloat(x[base + k]);
            i += rows << 5;
            p += (long) rows << 10;
        }
        for (; i < end; i++, p += 32) a[i] = toFloat(word(p));
    }

    /** Fills with the same values as repeated {@link #nextDouble()}. Complex fills use this with length 2n. */
    public void fill(double[] a) {
        fill(a, 0, a.length);
    }

    /** Fills {@code a[off, off + len)} with the same values as repeated {@link #nextDouble()}. */
    public void fill(double[] a, int off, int len) {
        Objects.checkFromIndexSize(off, len, a.length);
        long p = beginFill(64, len);
        int i = off, end = off + len;
        for (; i < end && (p & 1023) != 0; i++, p += 64) a[i] = toDouble(rawCached(p, 64));
        while (end - i >= 16) {
            int base = locate(p), rows = Math.min(cache.count - (base >> 5), (end - i) >> 4);
            int[] x = cache.buf;
            for (int k = 0, n = rows << 4; k < n; k++)
                a[i + k] = toDouble((x[base + 2 * k] & MASK32) | ((long) x[base + 2 * k + 1] << 32));
            i += rows << 4;
            p += (long) rows << 10;
        }
        for (; i < end; i++, p += 64) a[i] = toDouble(rawCached(p, 64));
    }

    /** Fills with the same values as repeated {@link #nextByte()}. */
    public void fill(byte[] a) {
        fill(a, 0, a.length);
    }

    /** Fills {@code a[off, off + len)} with the same values as repeated {@link #nextByte()}. */
    public void fill(byte[] a, int off, int len) {
        Objects.checkFromIndexSize(off, len, a.length);
        long p = beginFill(8, len);
        int i = off, end = off + len;
        for (; i < end && (p & 1023) != 0; i++, p += 8) a[i] = (byte) rawCached(p, 8);
        while (end - i >= 128) {
            int base = locate(p), rows = Math.min(cache.count - (base >> 5), (end - i) >> 7);
            int[] x = cache.buf;
            for (int k = 0, n = rows << 5; k < n; k++) {
                int v = x[base + k];
                a[i + 4 * k] = (byte) v;
                a[i + 4 * k + 1] = (byte) (v >>> 8);
                a[i + 4 * k + 2] = (byte) (v >>> 16);
                a[i + 4 * k + 3] = (byte) (v >>> 24);
            }
            i += rows << 7;
            p += (long) rows << 10;
        }
        for (; i < end; i++, p += 8) a[i] = (byte) rawCached(p, 8);
    }

    /** Fills with the same values as repeated {@link #nextShort()}. */
    public void fill(short[] a) {
        fill(a, 0, a.length);
    }

    /** Fills {@code a[off, off + len)} with the same values as repeated {@link #nextShort()}. */
    public void fill(short[] a, int off, int len) {
        Objects.checkFromIndexSize(off, len, a.length);
        long p = beginFill(16, len);
        int i = off, end = off + len;
        for (; i < end && (p & 1023) != 0; i++, p += 16) a[i] = (short) rawCached(p, 16);
        while (end - i >= 64) {
            int base = locate(p), rows = Math.min(cache.count - (base >> 5), (end - i) >> 6);
            int[] x = cache.buf;
            for (int k = 0, n = rows << 5; k < n; k++) {
                a[i + 2 * k] = (short) x[base + k];
                a[i + 2 * k + 1] = (short) (x[base + k] >>> 16);
            }
            i += rows << 6;
            p += (long) rows << 10;
        }
        for (; i < end; i++, p += 16) a[i] = (short) rawCached(p, 16);
    }

    /** Fills with the same values as repeated {@link #nextBoolean()}. */
    public void fill(boolean[] a) {
        fill(a, 0, a.length);
    }

    /** Fills {@code a[off, off + len)} with the same values as repeated {@link #nextBoolean()}. */
    public void fill(boolean[] a, int off, int len) {
        Objects.checkFromIndexSize(off, len, a.length);
        long p = beginFill(1, len);
        int i = off, end = off + len;
        while (i < end) {
            int bit = (int) (p & 31);
            int rest = word(p) >>> bit;
            int bits = Math.min(32 - bit, end - i);
            for (int b = 0; b < bits; b++) a[i + b] = ((rest >>> b) & 1) != 0;
            i += bits;
            p += bits;
        }
    }

    /** Fills with the same values as repeated {@link #nextFloat16Bits()}. */
    public void fillFloat16Bits(short[] a) {
        long p = beginFill(16, a.length);
        for (int i = 0; i < a.length; i++, p += 16) a[i] = toFloat16Bits((short) rawCached(p, 16));
    }

    /** Fills with the same values as repeated {@link #nextCodePoint()}. */
    public void fillCodePoints(int[] a) {
        long p = beginFill(64, a.length);
        for (int i = 0; i < a.length; i++, p += 64) a[i] = toCodePoint(rawCached(p, 64));
    }

    /** Fills {@code a} with 128-bit integers as {@code lo, hi} pairs, like repeated {@link #nextLong128()}. */
    public void fillLong128(long[] a) {
        if ((a.length & 1) != 0) throw new IllegalArgumentException("length must be even");
        long p = beginFill(128, a.length / 2);
        for (int i = 0; i < a.length; i += 2, p += 128) {
            a[i] = rawCached(p, 64);
            a[i + 1] = rawCached(p + 64, 64);
        }
    }

    /** Fills {@code bytes} like {@link #fill(byte[])}. */
    @Override
    public void nextBytes(byte[] bytes) {
        fill(bytes);
    }

    // ---- Derived generators -----------------------------------------------------------------

    /** The key of a child: the exposed or hidden half of F under this key. */
    private Tandem child(long counter, int domain, int aux, boolean hidden) {
        dropSpares();
        int[] s = new int[8];
        fKeyed(k0, k1, k2, k3, counter, domain, aux, s);
        int o = hidden ? 4 : 0;
        return new Tandem(s[o], s[o + 1], s[o + 2], s[o + 3], 0L, chunk);
    }

    /**
     * Returns the child with index {@code index}, read as unsigned, at position 0. It depends
     * on the key only and leaves this generator alone.
     */
    public Tandem split(long index) {
        return child(index >>> 1, DOMAIN_SPLIT, 0, (index & 1) != 0);
    }

    /**
     * Returns {@code n} children derived from the current 128-bit block, then moves this
     * generator to the following block boundary, also for {@code n = 0}.
     */
    public Tandem[] fork(int n) {
        if (n < 0) throw new IllegalArgumentException("n must not be negative");
        dropSpares();
        long b = pos >>> 7;
        Tandem[] kids = new Tandem[n];
        for (int i = 0; i < n; i++) kids[i] = child(b, DOMAIN_FORK, i >>> 1, (i & 1) != 0);
        pos = (b + 1) << 7;
        return kids;
    }

    /** Returns the child for the unsigned purpose identifier {@code purpose}, at position 0. */
    public Tandem sub(long purpose) {
        return child(purpose, DOMAIN_FOLD, 0, false);
    }

    /** Returns one child as {@code fork(1)[0]} does. */
    @Override
    public Tandem split() {
        return fork(1)[0];
    }

    /** Seeds a child from two 64-bit draws of {@code source}, with the specification's whitening. */
    @Override
    public Tandem split(SplittableGenerator source) {
        long lo = source.nextLong();
        return seed(lo, source.nextLong(), chunk);
    }

    @Override
    public Stream<SplittableGenerator> splits() {
        return Stream.generate(this::split);
    }

    @Override
    public Stream<SplittableGenerator> splits(long streamSize) {
        return splits().limit(checkSize(streamSize));
    }

    @Override
    public Stream<SplittableGenerator> splits(SplittableGenerator source) {
        return Stream.generate(() -> split(source));
    }

    @Override
    public Stream<SplittableGenerator> splits(long streamSize, SplittableGenerator source) {
        return splits(source).limit(checkSize(streamSize));
    }

    private static long checkSize(long n) {
        if (n < 0) throw new IllegalArgumentException("size must not be negative");
        return n;
    }
}
