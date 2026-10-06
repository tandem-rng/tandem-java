package io.github.tandemrng;

import java.io.IOException;
import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.util.Objects;
import java.util.random.RandomGenerator;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.LongStream;
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
 * {@code ints()}, {@code doubles()} and, {@code Collections.shuffle}. The
 * interface's default {@code nextGaussian} is overridden by the 1024-layer ziggurat of
 * {@link #nextGaussian()}, which agrees with every other Tandem port. Likewise the default
 * {@code nextExponential} is overridden by {@code -ln(1 - u)} of one double draw.
 * Every bounded integer draw uses Lemire's method, and the bounded {@code ints}, {@code longs}
 * and {@code doubles} streams are loops over the scalar bounded draws. Those replace the JDK's
 * default algorithms. The streams without bounds were already built on our draws.
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

    private final int k0, k1, k2, k3;
    private final int chunk;
    private long pos;
    private transient Cache cache;
    /** The cached block as the scalar draws read it: its longs, its first bit position and its length in bits. */
    private transient long[] fb = NO_BLOCK;
    private transient long fbStart, fbBits;
    private static final long[] NO_BLOCK = new long[0];
    /** The kept sine half of the last scalar float Box-Muller pair, see {@link #nextGaussianFloat()}. */
    private transient boolean hasSpareF;
    private transient float[] pairBufF;
    private transient float spareF;

    private void dropSpares() {
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
        final long[] buf = new long[BLOCK_ROWS * 16];
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
        fb = NO_BLOCK;
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

    /**
     * F on s = (o0..o3, h0..h3), in place, followed by {@code steps} steps T. The state stays
     * in locals: every miss of a double normal runs F at least twice.
     */
    static void f(int[] s, long steps) {
        int o0 = s[0], o1 = s[1], o2 = s[2], o3 = s[3], h0 = s[4], h1 = s[5], h2 = s[6], h3 = s[7];
        for (long r = -8; r < steps; r++) {
            long p0 = (o0 & MASK32) * ((h0 | 1) & MASK32);
            long p1 = (o2 & MASK32) * ((h1 | 1) & MASK32);
            int n0 = (int) (p1 ^ (p1 >>> 32)) ^ o1;
            int n1 = Integer.rotateLeft((int) p1, 16) ^ h2;
            int n2 = (int) (p0 ^ (p0 >>> 32)) ^ o3;
            int n3 = Integer.rotateLeft((int) p0, 16) ^ h3;
            h0 ^= Integer.rotateLeft(h1, 7);
            h1 ^= Integer.rotateLeft(h2, 13);
            h2 ^= Integer.rotateLeft(h3, 22);
            h3 ^= Integer.rotateLeft(h0, 3);
            h0 = (h0 + CLOCK_WEYL) ^ n0;
            if (r < 0) {
                // A round of F ends with o0 ^= RC and swaps the halves.
                o0 = h0; o1 = h1; o2 = h2; o3 = h3;
                h0 = n0 ^ RC[(int) r + 8]; h1 = n1; h2 = n2; h3 = n3;
            } else {
                o0 = n0; o1 = n1; o2 = n2; o3 = n3;
            }
        }
        s[0] = o0; s[1] = o1; s[2] = o2; s[3] = o3; s[4] = h0; s[5] = h1; s[6] = h2; s[7] = h3;
    }

    static void f(int[] s) {
        f(s, 0);
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

    // The lane loops below run two lanes side by side and two rows per iteration. One lane
    // alone waits on the latency of its multiply-xor chain. Four lanes need 32 state words and
    // spill out of the AArch64 registers. The row loops end on b != end, which C2 does not
    // count: it unrolls counted loops of many xors twice, and the unrolled loop spilled the lane
    // state. Two rows per iteration halve the loop overhead of the uncounted loop instead. C2
    // has no 32 x 32 to 64-bit multiply, so o0 and o2 stay zero-extended in longs, which saves
    // the zero extension of the fold before the next product. The row count must be even.

    /** F on the eight chunks 8g .. 8g + 7 into the lane state arrays. */
    private void seedLanes(int[] o, int[] h, long g) {
        for (int l = 0; l < 8; l += 2) {
            long ctr = 8L * g + l;
            int o0a = (int) ctr, o1a = (int) (ctr >>> 32), o2a = DOMAIN_STREAM, o3a = AUX_STREAM;
            int h0a = k0, h1a = k1, h2a = k2, h3a = k3;
            int o0b = (int) (ctr + 1), o1b = (int) ((ctr + 1) >>> 32), o2b = DOMAIN_STREAM, o3b = AUX_STREAM;
            int h0b = k0, h1b = k1, h2b = k2, h3b = k3;
            for (int r = 0; r < 8; r++) {
                long p0a = (o0a & MASK32) * ((h0a | 1) & MASK32);
                long p1a = (o2a & MASK32) * ((h1a | 1) & MASK32);
                long p0b = (o0b & MASK32) * ((h0b | 1) & MASK32);
                long p1b = (o2b & MASK32) * ((h1b | 1) & MASK32);
                int n0a = (int) (p1a ^ (p1a >>> 32)) ^ o1a;
                int n1a = Integer.rotateLeft((int) p1a, 16) ^ h2a;
                int n2a = (int) (p0a ^ (p0a >>> 32)) ^ o3a;
                int n3a = Integer.rotateLeft((int) p0a, 16) ^ h3a;
                int n0b = (int) (p1b ^ (p1b >>> 32)) ^ o1b;
                int n1b = Integer.rotateLeft((int) p1b, 16) ^ h2b;
                int n2b = (int) (p0b ^ (p0b >>> 32)) ^ o3b;
                int n3b = Integer.rotateLeft((int) p0b, 16) ^ h3b;
                h0a ^= Integer.rotateLeft(h1a, 7);
                h1a ^= Integer.rotateLeft(h2a, 13);
                h2a ^= Integer.rotateLeft(h3a, 22);
                h3a ^= Integer.rotateLeft(h0a, 3);
                h0a = (h0a + CLOCK_WEYL) ^ n0a;
                h0b ^= Integer.rotateLeft(h1b, 7);
                h1b ^= Integer.rotateLeft(h2b, 13);
                h2b ^= Integer.rotateLeft(h3b, 22);
                h3b ^= Integer.rotateLeft(h0b, 3);
                h0b = (h0b + CLOCK_WEYL) ^ n0b;
                // The round ends with o0 ^= RC[r] and swaps the halves.
                o0a = h0a; o1a = h1a; o2a = h2a; o3a = h3a;
                h0a = n0a ^ RC[r]; h1a = n1a; h2a = n2a; h3a = n3a;
                o0b = h0b; o1b = h1b; o2b = h2b; o3b = h3b;
                h0b = n0b ^ RC[r]; h1b = n1b; h2b = n2b; h3b = n3b;
            }
            o[4 * l] = o0a; o[4 * l + 1] = o1a; o[4 * l + 2] = o2a; o[4 * l + 3] = o3a;
            h[4 * l] = h0a; h[4 * l + 1] = h1a; h[4 * l + 2] = h2a; h[4 * l + 3] = h3a;
            o[4 * l + 4] = o0b; o[4 * l + 5] = o1b; o[4 * l + 6] = o2b; o[4 * l + 7] = o3b;
            h[4 * l + 4] = h0b; h[4 * l + 5] = h1b; h[4 * l + 6] = h2b; h[4 * l + 7] = h3b;
        }
    }

    /**
     * Steps every lane {@code rows} times, an even number, and writes the exposed halves to
     * {@code buf} in stream order, two words per long: row j, lane l at {@code 16 j + 2 l} and
     * the next long.
     */
    private static void generate(int[] o, int[] h, long[] buf, int off, int rows) {
        for (int l = 0; l < 8; l += 2) {
            long o0a = o[4 * l] & MASK32, o2a = o[4 * l + 2] & MASK32;
            int o1a = o[4 * l + 1], o3a = o[4 * l + 3];
            int h0a = h[4 * l], h1a = h[4 * l + 1], h2a = h[4 * l + 2], h3a = h[4 * l + 3];
            long o0b = o[4 * l + 4] & MASK32, o2b = o[4 * l + 6] & MASK32;
            int o1b = o[4 * l + 5], o3b = o[4 * l + 7];
            int h0b = h[4 * l + 4], h1b = h[4 * l + 5], h2b = h[4 * l + 6], h3b = h[4 * l + 7];
            for (int b = off + 2 * l, end = b + 16 * rows; b != end; b += 32) {
                long p0, p1;
                p0 = o0a * ((h0a | 1) & MASK32);
                p1 = o2a * ((h1a | 1) & MASK32);
                o0a = ((int) (p1 ^ (p1 >>> 32)) ^ o1a) & MASK32;
                o1a = Integer.rotateLeft((int) p1, 16) ^ h2a;
                o2a = ((int) (p0 ^ (p0 >>> 32)) ^ o3a) & MASK32;
                o3a = Integer.rotateLeft((int) p0, 16) ^ h3a;
                h0a ^= Integer.rotateLeft(h1a, 7);
                h1a ^= Integer.rotateLeft(h2a, 13);
                h2a ^= Integer.rotateLeft(h3a, 22);
                h3a ^= Integer.rotateLeft(h0a, 3);
                h0a = (h0a + CLOCK_WEYL) ^ (int) o0a;
                p0 = o0b * ((h0b | 1) & MASK32);
                p1 = o2b * ((h1b | 1) & MASK32);
                o0b = ((int) (p1 ^ (p1 >>> 32)) ^ o1b) & MASK32;
                o1b = Integer.rotateLeft((int) p1, 16) ^ h2b;
                o2b = ((int) (p0 ^ (p0 >>> 32)) ^ o3b) & MASK32;
                o3b = Integer.rotateLeft((int) p0, 16) ^ h3b;
                h0b ^= Integer.rotateLeft(h1b, 7);
                h1b ^= Integer.rotateLeft(h2b, 13);
                h2b ^= Integer.rotateLeft(h3b, 22);
                h3b ^= Integer.rotateLeft(h0b, 3);
                h0b = (h0b + CLOCK_WEYL) ^ (int) o0b;
                buf[b] = o0a | ((long) o1a << 32);
                buf[b + 1] = o2a | ((long) o3a << 32);
                buf[b + 2] = o0b | ((long) o1b << 32);
                buf[b + 3] = o2b | ((long) o3b << 32);
                p0 = o0a * ((h0a | 1) & MASK32);
                p1 = o2a * ((h1a | 1) & MASK32);
                o0a = ((int) (p1 ^ (p1 >>> 32)) ^ o1a) & MASK32;
                o1a = Integer.rotateLeft((int) p1, 16) ^ h2a;
                o2a = ((int) (p0 ^ (p0 >>> 32)) ^ o3a) & MASK32;
                o3a = Integer.rotateLeft((int) p0, 16) ^ h3a;
                h0a ^= Integer.rotateLeft(h1a, 7);
                h1a ^= Integer.rotateLeft(h2a, 13);
                h2a ^= Integer.rotateLeft(h3a, 22);
                h3a ^= Integer.rotateLeft(h0a, 3);
                h0a = (h0a + CLOCK_WEYL) ^ (int) o0a;
                p0 = o0b * ((h0b | 1) & MASK32);
                p1 = o2b * ((h1b | 1) & MASK32);
                o0b = ((int) (p1 ^ (p1 >>> 32)) ^ o1b) & MASK32;
                o1b = Integer.rotateLeft((int) p1, 16) ^ h2b;
                o2b = ((int) (p0 ^ (p0 >>> 32)) ^ o3b) & MASK32;
                o3b = Integer.rotateLeft((int) p0, 16) ^ h3b;
                h0b ^= Integer.rotateLeft(h1b, 7);
                h1b ^= Integer.rotateLeft(h2b, 13);
                h2b ^= Integer.rotateLeft(h3b, 22);
                h3b ^= Integer.rotateLeft(h0b, 3);
                h0b = (h0b + CLOCK_WEYL) ^ (int) o0b;
                buf[b + 16] = o0a | ((long) o1a << 32);
                buf[b + 17] = o2a | ((long) o3a << 32);
                buf[b + 18] = o0b | ((long) o1b << 32);
                buf[b + 19] = o2b | ((long) o3b << 32);
            }
            o[4 * l] = (int) o0a; o[4 * l + 1] = o1a; o[4 * l + 2] = (int) o2a; o[4 * l + 3] = o3a;
            h[4 * l] = h0a; h[4 * l + 1] = h1a; h[4 * l + 2] = h2a; h[4 * l + 3] = h3a;
            o[4 * l + 4] = (int) o0b; o[4 * l + 5] = o1b; o[4 * l + 6] = (int) o2b; o[4 * l + 7] = o3b;
            h[4 * l + 4] = h0b; h[4 * l + 5] = h1b; h[4 * l + 6] = h2b; h[4 * l + 7] = h3b;
        }
    }

    /** As {@link #generate}, converting each long to a double as {@link #nextDouble()} does. */
    private static void generateDouble(int[] o, int[] h, double[] buf, int off, int rows) {
        for (int l = 0; l < 8; l += 2) {
            long o0a = o[4 * l] & MASK32, o2a = o[4 * l + 2] & MASK32;
            int o1a = o[4 * l + 1], o3a = o[4 * l + 3];
            int h0a = h[4 * l], h1a = h[4 * l + 1], h2a = h[4 * l + 2], h3a = h[4 * l + 3];
            long o0b = o[4 * l + 4] & MASK32, o2b = o[4 * l + 6] & MASK32;
            int o1b = o[4 * l + 5], o3b = o[4 * l + 7];
            int h0b = h[4 * l + 4], h1b = h[4 * l + 5], h2b = h[4 * l + 6], h3b = h[4 * l + 7];
            for (int b = off + 2 * l, end = b + 16 * rows; b != end; b += 32) {
                long p0, p1;
                p0 = o0a * ((h0a | 1) & MASK32);
                p1 = o2a * ((h1a | 1) & MASK32);
                o0a = ((int) (p1 ^ (p1 >>> 32)) ^ o1a) & MASK32;
                o1a = Integer.rotateLeft((int) p1, 16) ^ h2a;
                o2a = ((int) (p0 ^ (p0 >>> 32)) ^ o3a) & MASK32;
                o3a = Integer.rotateLeft((int) p0, 16) ^ h3a;
                h0a ^= Integer.rotateLeft(h1a, 7);
                h1a ^= Integer.rotateLeft(h2a, 13);
                h2a ^= Integer.rotateLeft(h3a, 22);
                h3a ^= Integer.rotateLeft(h0a, 3);
                h0a = (h0a + CLOCK_WEYL) ^ (int) o0a;
                p0 = o0b * ((h0b | 1) & MASK32);
                p1 = o2b * ((h1b | 1) & MASK32);
                o0b = ((int) (p1 ^ (p1 >>> 32)) ^ o1b) & MASK32;
                o1b = Integer.rotateLeft((int) p1, 16) ^ h2b;
                o2b = ((int) (p0 ^ (p0 >>> 32)) ^ o3b) & MASK32;
                o3b = Integer.rotateLeft((int) p0, 16) ^ h3b;
                h0b ^= Integer.rotateLeft(h1b, 7);
                h1b ^= Integer.rotateLeft(h2b, 13);
                h2b ^= Integer.rotateLeft(h3b, 22);
                h3b ^= Integer.rotateLeft(h0b, 3);
                h0b = (h0b + CLOCK_WEYL) ^ (int) o0b;
                buf[b] = toDouble((int) o0a, o1a);
                buf[b + 1] = toDouble((int) o2a, o3a);
                buf[b + 2] = toDouble((int) o0b, o1b);
                buf[b + 3] = toDouble((int) o2b, o3b);
                p0 = o0a * ((h0a | 1) & MASK32);
                p1 = o2a * ((h1a | 1) & MASK32);
                o0a = ((int) (p1 ^ (p1 >>> 32)) ^ o1a) & MASK32;
                o1a = Integer.rotateLeft((int) p1, 16) ^ h2a;
                o2a = ((int) (p0 ^ (p0 >>> 32)) ^ o3a) & MASK32;
                o3a = Integer.rotateLeft((int) p0, 16) ^ h3a;
                h0a ^= Integer.rotateLeft(h1a, 7);
                h1a ^= Integer.rotateLeft(h2a, 13);
                h2a ^= Integer.rotateLeft(h3a, 22);
                h3a ^= Integer.rotateLeft(h0a, 3);
                h0a = (h0a + CLOCK_WEYL) ^ (int) o0a;
                p0 = o0b * ((h0b | 1) & MASK32);
                p1 = o2b * ((h1b | 1) & MASK32);
                o0b = ((int) (p1 ^ (p1 >>> 32)) ^ o1b) & MASK32;
                o1b = Integer.rotateLeft((int) p1, 16) ^ h2b;
                o2b = ((int) (p0 ^ (p0 >>> 32)) ^ o3b) & MASK32;
                o3b = Integer.rotateLeft((int) p0, 16) ^ h3b;
                h0b ^= Integer.rotateLeft(h1b, 7);
                h1b ^= Integer.rotateLeft(h2b, 13);
                h2b ^= Integer.rotateLeft(h3b, 22);
                h3b ^= Integer.rotateLeft(h0b, 3);
                h0b = (h0b + CLOCK_WEYL) ^ (int) o0b;
                buf[b + 16] = toDouble((int) o0a, o1a);
                buf[b + 17] = toDouble((int) o2a, o3a);
                buf[b + 18] = toDouble((int) o0b, o1b);
                buf[b + 19] = toDouble((int) o2b, o3b);
            }
            o[4 * l] = (int) o0a; o[4 * l + 1] = o1a; o[4 * l + 2] = (int) o2a; o[4 * l + 3] = o3a;
            h[4 * l] = h0a; h[4 * l + 1] = h1a; h[4 * l + 2] = h2a; h[4 * l + 3] = h3a;
            o[4 * l + 4] = (int) o0b; o[4 * l + 5] = o1b; o[4 * l + 6] = (int) o2b; o[4 * l + 7] = o3b;
            h[4 * l + 4] = h0b; h[4 * l + 5] = h1b; h[4 * l + 6] = h2b; h[4 * l + 7] = h3b;
        }
    }

    /** As {@link #generate}, one word per int: row j, lane l at {@code 32 j + 4 l} to {@code 32 j + 4 l + 3}. */
    private static void generateInt(int[] o, int[] h, int[] buf, int off, int rows) {
        for (int l = 0; l < 8; l += 2) {
            long o0a = o[4 * l] & MASK32, o2a = o[4 * l + 2] & MASK32;
            int o1a = o[4 * l + 1], o3a = o[4 * l + 3];
            int h0a = h[4 * l], h1a = h[4 * l + 1], h2a = h[4 * l + 2], h3a = h[4 * l + 3];
            long o0b = o[4 * l + 4] & MASK32, o2b = o[4 * l + 6] & MASK32;
            int o1b = o[4 * l + 5], o3b = o[4 * l + 7];
            int h0b = h[4 * l + 4], h1b = h[4 * l + 5], h2b = h[4 * l + 6], h3b = h[4 * l + 7];
            for (int b = off + 4 * l, end = b + 32 * rows; b != end; b += 64) {
                long p0, p1;
                p0 = o0a * ((h0a | 1) & MASK32);
                p1 = o2a * ((h1a | 1) & MASK32);
                o0a = ((int) (p1 ^ (p1 >>> 32)) ^ o1a) & MASK32;
                o1a = Integer.rotateLeft((int) p1, 16) ^ h2a;
                o2a = ((int) (p0 ^ (p0 >>> 32)) ^ o3a) & MASK32;
                o3a = Integer.rotateLeft((int) p0, 16) ^ h3a;
                h0a ^= Integer.rotateLeft(h1a, 7);
                h1a ^= Integer.rotateLeft(h2a, 13);
                h2a ^= Integer.rotateLeft(h3a, 22);
                h3a ^= Integer.rotateLeft(h0a, 3);
                h0a = (h0a + CLOCK_WEYL) ^ (int) o0a;
                p0 = o0b * ((h0b | 1) & MASK32);
                p1 = o2b * ((h1b | 1) & MASK32);
                o0b = ((int) (p1 ^ (p1 >>> 32)) ^ o1b) & MASK32;
                o1b = Integer.rotateLeft((int) p1, 16) ^ h2b;
                o2b = ((int) (p0 ^ (p0 >>> 32)) ^ o3b) & MASK32;
                o3b = Integer.rotateLeft((int) p0, 16) ^ h3b;
                h0b ^= Integer.rotateLeft(h1b, 7);
                h1b ^= Integer.rotateLeft(h2b, 13);
                h2b ^= Integer.rotateLeft(h3b, 22);
                h3b ^= Integer.rotateLeft(h0b, 3);
                h0b = (h0b + CLOCK_WEYL) ^ (int) o0b;
                buf[b] = (int) o0a;
                buf[b + 1] = o1a;
                buf[b + 2] = (int) o2a;
                buf[b + 3] = o3a;
                buf[b + 4] = (int) o0b;
                buf[b + 5] = o1b;
                buf[b + 6] = (int) o2b;
                buf[b + 7] = o3b;
                p0 = o0a * ((h0a | 1) & MASK32);
                p1 = o2a * ((h1a | 1) & MASK32);
                o0a = ((int) (p1 ^ (p1 >>> 32)) ^ o1a) & MASK32;
                o1a = Integer.rotateLeft((int) p1, 16) ^ h2a;
                o2a = ((int) (p0 ^ (p0 >>> 32)) ^ o3a) & MASK32;
                o3a = Integer.rotateLeft((int) p0, 16) ^ h3a;
                h0a ^= Integer.rotateLeft(h1a, 7);
                h1a ^= Integer.rotateLeft(h2a, 13);
                h2a ^= Integer.rotateLeft(h3a, 22);
                h3a ^= Integer.rotateLeft(h0a, 3);
                h0a = (h0a + CLOCK_WEYL) ^ (int) o0a;
                p0 = o0b * ((h0b | 1) & MASK32);
                p1 = o2b * ((h1b | 1) & MASK32);
                o0b = ((int) (p1 ^ (p1 >>> 32)) ^ o1b) & MASK32;
                o1b = Integer.rotateLeft((int) p1, 16) ^ h2b;
                o2b = ((int) (p0 ^ (p0 >>> 32)) ^ o3b) & MASK32;
                o3b = Integer.rotateLeft((int) p0, 16) ^ h3b;
                h0b ^= Integer.rotateLeft(h1b, 7);
                h1b ^= Integer.rotateLeft(h2b, 13);
                h2b ^= Integer.rotateLeft(h3b, 22);
                h3b ^= Integer.rotateLeft(h0b, 3);
                h0b = (h0b + CLOCK_WEYL) ^ (int) o0b;
                buf[b + 32] = (int) o0a;
                buf[b + 33] = o1a;
                buf[b + 34] = (int) o2a;
                buf[b + 35] = o3a;
                buf[b + 36] = (int) o0b;
                buf[b + 37] = o1b;
                buf[b + 38] = (int) o2b;
                buf[b + 39] = o3b;
            }
            o[4 * l] = (int) o0a; o[4 * l + 1] = o1a; o[4 * l + 2] = (int) o2a; o[4 * l + 3] = o3a;
            h[4 * l] = h0a; h[4 * l + 1] = h1a; h[4 * l + 2] = h2a; h[4 * l + 3] = h3a;
            o[4 * l + 4] = (int) o0b; o[4 * l + 5] = o1b; o[4 * l + 6] = (int) o2b; o[4 * l + 7] = o3b;
            h[4 * l + 4] = h0b; h[4 * l + 5] = h1b; h[4 * l + 6] = h2b; h[4 * l + 7] = h3b;
        }
    }

    /** As {@link #generateInt}, converting each word to a float as {@link #nextFloat()} does. */
    private static void generateFloat(int[] o, int[] h, float[] buf, int off, int rows) {
        for (int l = 0; l < 8; l += 2) {
            long o0a = o[4 * l] & MASK32, o2a = o[4 * l + 2] & MASK32;
            int o1a = o[4 * l + 1], o3a = o[4 * l + 3];
            int h0a = h[4 * l], h1a = h[4 * l + 1], h2a = h[4 * l + 2], h3a = h[4 * l + 3];
            long o0b = o[4 * l + 4] & MASK32, o2b = o[4 * l + 6] & MASK32;
            int o1b = o[4 * l + 5], o3b = o[4 * l + 7];
            int h0b = h[4 * l + 4], h1b = h[4 * l + 5], h2b = h[4 * l + 6], h3b = h[4 * l + 7];
            for (int b = off + 4 * l, end = b + 32 * rows; b != end; b += 64) {
                long p0, p1;
                p0 = o0a * ((h0a | 1) & MASK32);
                p1 = o2a * ((h1a | 1) & MASK32);
                o0a = ((int) (p1 ^ (p1 >>> 32)) ^ o1a) & MASK32;
                o1a = Integer.rotateLeft((int) p1, 16) ^ h2a;
                o2a = ((int) (p0 ^ (p0 >>> 32)) ^ o3a) & MASK32;
                o3a = Integer.rotateLeft((int) p0, 16) ^ h3a;
                h0a ^= Integer.rotateLeft(h1a, 7);
                h1a ^= Integer.rotateLeft(h2a, 13);
                h2a ^= Integer.rotateLeft(h3a, 22);
                h3a ^= Integer.rotateLeft(h0a, 3);
                h0a = (h0a + CLOCK_WEYL) ^ (int) o0a;
                p0 = o0b * ((h0b | 1) & MASK32);
                p1 = o2b * ((h1b | 1) & MASK32);
                o0b = ((int) (p1 ^ (p1 >>> 32)) ^ o1b) & MASK32;
                o1b = Integer.rotateLeft((int) p1, 16) ^ h2b;
                o2b = ((int) (p0 ^ (p0 >>> 32)) ^ o3b) & MASK32;
                o3b = Integer.rotateLeft((int) p0, 16) ^ h3b;
                h0b ^= Integer.rotateLeft(h1b, 7);
                h1b ^= Integer.rotateLeft(h2b, 13);
                h2b ^= Integer.rotateLeft(h3b, 22);
                h3b ^= Integer.rotateLeft(h0b, 3);
                h0b = (h0b + CLOCK_WEYL) ^ (int) o0b;
                buf[b] = toFloat((int) o0a);
                buf[b + 1] = toFloat(o1a);
                buf[b + 2] = toFloat((int) o2a);
                buf[b + 3] = toFloat(o3a);
                buf[b + 4] = toFloat((int) o0b);
                buf[b + 5] = toFloat(o1b);
                buf[b + 6] = toFloat((int) o2b);
                buf[b + 7] = toFloat(o3b);
                p0 = o0a * ((h0a | 1) & MASK32);
                p1 = o2a * ((h1a | 1) & MASK32);
                o0a = ((int) (p1 ^ (p1 >>> 32)) ^ o1a) & MASK32;
                o1a = Integer.rotateLeft((int) p1, 16) ^ h2a;
                o2a = ((int) (p0 ^ (p0 >>> 32)) ^ o3a) & MASK32;
                o3a = Integer.rotateLeft((int) p0, 16) ^ h3a;
                h0a ^= Integer.rotateLeft(h1a, 7);
                h1a ^= Integer.rotateLeft(h2a, 13);
                h2a ^= Integer.rotateLeft(h3a, 22);
                h3a ^= Integer.rotateLeft(h0a, 3);
                h0a = (h0a + CLOCK_WEYL) ^ (int) o0a;
                p0 = o0b * ((h0b | 1) & MASK32);
                p1 = o2b * ((h1b | 1) & MASK32);
                o0b = ((int) (p1 ^ (p1 >>> 32)) ^ o1b) & MASK32;
                o1b = Integer.rotateLeft((int) p1, 16) ^ h2b;
                o2b = ((int) (p0 ^ (p0 >>> 32)) ^ o3b) & MASK32;
                o3b = Integer.rotateLeft((int) p0, 16) ^ h3b;
                h0b ^= Integer.rotateLeft(h1b, 7);
                h1b ^= Integer.rotateLeft(h2b, 13);
                h2b ^= Integer.rotateLeft(h3b, 22);
                h3b ^= Integer.rotateLeft(h0b, 3);
                h0b = (h0b + CLOCK_WEYL) ^ (int) o0b;
                buf[b + 32] = toFloat((int) o0a);
                buf[b + 33] = toFloat(o1a);
                buf[b + 34] = toFloat((int) o2a);
                buf[b + 35] = toFloat(o3a);
                buf[b + 36] = toFloat((int) o0b);
                buf[b + 37] = toFloat(o1b);
                buf[b + 38] = toFloat((int) o2b);
                buf[b + 39] = toFloat(o3b);
            }
            o[4 * l] = (int) o0a; o[4 * l + 1] = o1a; o[4 * l + 2] = (int) o2a; o[4 * l + 3] = o3a;
            h[4 * l] = h0a; h[4 * l + 1] = h1a; h[4 * l + 2] = h2a; h[4 * l + 3] = h3a;
            o[4 * l + 4] = (int) o0b; o[4 * l + 5] = o1b; o[4 * l + 6] = (int) o2b; o[4 * l + 7] = o3b;
            h[4 * l + 4] = h0b; h[4 * l + 5] = h1b; h[4 * l + 6] = h2b; h[4 * l + 7] = h3b;
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
            if (d >= 0 && d < c.count) return (int) d << 4;
        }
        return load(r);
    }

    private int load(long r) {
        int rows = Math.min(chunk, BLOCK_ROWS);
        long rb = r & -(long) rows;
        Cache c = blockState(rb, rows);
        // The kernel steps an even number of rows: at K = 1 the second row is spare.
        int steps = (rows + 1) & -2;
        generate(c.o, c.h, c.buf, 0, steps);
        c.next += steps;
        c.start = rb;
        c.count = rows;
        fb = c.buf;
        fbStart = rb << 10;
        fbBits = (long) rows << 10;
        return (int) (r - rb) << 4;
    }

    /**
     * Moves the lane states to the start of the block of {@code rows} rows that begins at row
     * rb, and empties the block cache, whose buffer the caller or the skipped blocks overwrite.
     * A block is only reachable by stepping forward, so a backward or cross-group move reseeds.
     */
    private Cache blockState(long rb, int rows) {
        Cache c = cache;
        if (c == null) cache = c = new Cache();
        c.count = 0;
        fbBits = 0;
        int shift = Integer.numberOfTrailingZeros(chunk);
        long g = rb >>> shift;
        int first = (int) (rb & (chunk - 1L));
        if (c.group != g || c.next > first) {
            seedLanes(c.o, c.h, g);
            c.group = g;
            c.next = 0;
        }
        for (; c.next < first; c.next += rows) generate(c.o, c.h, c.buf, 0, rows);
        return c;
    }

    /** The word at bit position p, which need not be aligned: the 32-bit word that contains it. */
    private int word(long p) {
        long off = p - fbStart;
        if (Long.compareUnsigned(off, fbBits) >= 0) off = refill(p);
        return (int) (fb[(int) (off >>> 6)] >>> (int) (off & 32));
    }

    /** Out of line, so the draw paths stay small enough to inline. */
    private long refill(long p) {
        load(p >>> 10);
        return p - fbStart;
    }

    /** The w bits at bit position p, aligned to w, for w up to 64, without touching the cache. */
    private long rawAt(long p, int w) {
        return extract(laneAt(p), (int) (p >>> 5) & 3, (int) (p & 31), w);
    }

    /** The state of the lane whose row holds bit position p, after that row's step. */
    private int[] laneAt(long p) {
        long r = p >>> 10;
        int shift = Integer.numberOfTrailingZeros(chunk);
        long ctr = 8L * (r >>> shift) + ((p >>> 7) & 7);
        int[] s = {(int) ctr, (int) (ctr >>> 32), DOMAIN_STREAM, AUX_STREAM, k0, k1, k2, k3};
        f(s, (r & (chunk - 1L)) + 1);
        return s;
    }

    /**
     * Writes elements 2i and 2i + 1 of the {@code fill(long[])} that would start at position 0
     * to {@code out}. Both come from one lane of one row, so they take one F.
     */
    void atLongPair(long i, long[] out) {
        int[] s = laneAt(128 * i);
        out[0] = extract(s, 0, 0, 64);
        out[1] = extract(s, 2, 0, 64);
    }

    private static long extract(int[] x, int word, int bit, int w) {
        if (w == 64) return (x[word] & MASK32) | ((long) x[word + 1] << 32);
        return (x[word] >>> bit) & (-1 >>> (32 - w)) & MASK32;
    }

    /** The w bits at the aligned position p, through the row cache. */
    private long rawCached(long p, int w) {
        int base = locate(p);
        long l = cache.buf[base + ((int) (p >>> 6) & 15)];
        return w == 64 ? l : (l >>> (int) (p & 63)) & (-1L >>> (64 - w));
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
        // align(pos, 32) + 32 as one add and mask, which shortens the dependency through pos.
        long e = (pos + 63) & -32L;
        pos = e;
        return word(e - 32);
    }

    /** Draws a signed 64-bit integer. */
    @Override
    public long nextLong() {
        long e = (pos + 127) & -64L;
        pos = e;
        long p = e - 64;
        long off = p - fbStart;
        if (Long.compareUnsigned(off, fbBits) >= 0) off = refill(p);
        return fb[(int) (off >>> 6)];
    }

    /** Draws a 128-bit integer as {@code {lo, hi}}. */
    public long[] nextLong128() {
        long p = align(pos, 128);
        pos = p + 128;
        int i = locate(p) + ((int) (p >>> 6) & 15);
        long[] x = cache.buf;
        return new long[] {x[i], x[i + 1]};
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

    /** {@link #toDouble(long)} of the long with low word lo and high word hi. */
    private static double toDouble(int lo, int hi) {
        // Both terms and their sum are exact, and the work moves from the integer units, which
        // the row step fills, to the floating-point units.
        return Math.fma((double) (hi & MASK32), 0x1p-32, (lo >>> 11) * 0x1p-53);
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
     * A range of 0 returns 0 and consumes one draw, as Appendix A of the specification says.
     */
    public int belowU32(int range) {
        if (range == 0) {
            nextInt();
            return 0;
        }
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
     * A range of 0 returns 0 and consumes one draw, as Appendix A of the specification says.
     */
    public long belowU64(long range) {
        if (range == 0) {
            nextLong();
            return 0;
        }
        long x = nextLong(), lo = x * range;
        if (Long.compareUnsigned(lo, range) < 0) {
            long t = Long.remainderUnsigned(-range, range);
            while (Long.compareUnsigned(lo, t) < 0) {
                x = nextLong();
                lo = x * range;
            }
        }
        return Math.unsignedMultiplyHigh(x, range);
    }

    private static final long PURPOSE_BELOW32 = 0x424c573332L;
    private static final long PURPOSE_BELOW64 = 0x424c573634L;

    /**
     * Fills {@code a[off, off + len)} uniformly from {@code [0, range)}, with {@code range} read
     * as unsigned, by the shared parallel-friendly contract of the other ports. Element i takes
     * draw i of {@link #fill(int[])} and the fill moves the position by exactly {@code len}
     * draws. A rejected draw retries with Lemire's rule on the draws of
     * {@code sub(0x424c573332).split(g)} of a generator with this key and chunk length, from
     * position 0. Here g is the aligned start position divided by 32, plus i, so a fill cut into
     * chunks equals the whole fill. Without rejections this equals repeated {@link #belowU32}, but after a
     * rejection the sequential scalar loop and the fill differ.
     */
    public void fillBelowU32(int[] a, int off, int len, int range) {
        long first = align(pos, 32) >>> 5;
        fill(a, off, len);
        if (range == 0) {
            java.util.Arrays.fill(a, off, off + len, 0);
            return;
        }
        long r = range & MASK32;
        for (int e = 0; e < len; e++) {
            long m = (a[off + e] & MASK32) * r;
            if ((m & MASK32) < r) {
                long t = Integer.remainderUnsigned(-range, range) & MASK32;
                if ((m & MASK32) < t) {
                    Tandem f = new Tandem(k0, k1, k2, k3, 0L, chunk).sub(PURPOSE_BELOW32).split(first + e);
                    do m = (f.nextInt() & MASK32) * r;
                    while ((m & MASK32) < t);
                }
            }
            a[off + e] = (int) (m >>> 32);
        }
    }

    /** As {@link #fillBelowU32}, on 64-bit draws and {@code sub(0x424c573634)}. */
    public void fillBelowU64(long[] a, int off, int len, long range) {
        long first = align(pos, 64) >>> 6;
        fill(a, off, len);
        if (range == 0) {
            java.util.Arrays.fill(a, off, off + len, 0L);
            return;
        }
        for (int e = 0; e < len; e++) {
            long x = a[off + e], lo = x * range;
            if (Long.compareUnsigned(lo, range) < 0) {
                long t = Long.remainderUnsigned(-range, range);
                if (Long.compareUnsigned(lo, t) < 0) {
                    Tandem f = new Tandem(k0, k1, k2, k3, 0L, chunk).sub(PURPOSE_BELOW64).split(first + e);
                    do {
                        x = f.nextLong();
                        lo = x * range;
                    } while (Long.compareUnsigned(lo, t) < 0);
                }
            }
            a[off + e] = Math.unsignedMultiplyHigh(x, range);
        }
    }

    /** Fills {@code a} uniformly from {@code [0, bound)} with {@link #fillBelowU32}. */
    public void nextInts(int[] a, int bound) {
        if (bound <= 0) throw new IllegalArgumentException("bound must be positive");
        fillBelowU32(a, 0, a.length, bound);
    }

    // Appendix A: an interface typed by result or bounds takes its draw width from the range, so a
    // long result with a range up to 2^32 uses 32-bit draws and equals the int result.
    private static boolean narrowRange(long range) {
        return (range - 1) >>> 32 == 0;
    }

    private long belowByRange(long range) {
        if (!narrowRange(range)) return belowU64(range);
        if (range == 1L << 32) return nextInt() & MASK32;
        return belowU32((int) range) & MASK32;
    }

    /**
     * Fills {@code a} uniformly from {@code [0, bound)}. A bound up to 2^32 uses
     * {@link #fillBelowU32} and widens, so the fill consumes 32-bit draws. A larger bound uses
     * {@link #fillBelowU64}.
     */
    public void nextLongs(long[] a, long bound) {
        if (bound <= 0) throw new IllegalArgumentException("bound must be positive");
        if (!narrowRange(bound)) {
            fillBelowU64(a, 0, a.length, bound);
            return;
        }
        // Blocks bound the scratch array. Cut bounded fills equal the whole fill.
        int[] narrow = new int[Math.min(a.length, NARROW_BLOCK)];
        for (int i = 0; i < a.length; ) {
            int m = Math.min(a.length - i, NARROW_BLOCK);
            if (bound == 1L << 32) fill(narrow, 0, m);
            else fillBelowU32(narrow, 0, m, (int) bound);
            for (int j = 0; j < m; j++) a[i++] = narrow[j] & MASK32;
        }
    }

    private static final int NARROW_BLOCK = 1024;

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

    /** Draws uniformly from {@code [0, bound)}. A bound up to 2^32 draws 32 bits and equals {@link #nextInt(int)}. */
    @Override
    public long nextLong(long bound) {
        if (bound <= 0) throw new IllegalArgumentException("bound must be positive");
        return belowByRange(bound);
    }

    /** Draws uniformly from {@code [origin, bound)} on the span, with the width rule of {@link #nextLong(long)}. */
    @Override
    public long nextLong(long origin, long bound) {
        if (origin >= bound) throw new IllegalArgumentException("bound must exceed origin");
        return origin + belowByRange(bound - origin);
    }

    /** Draws a double in {@code [0, bound)} from {@link #nextDouble()}. */
    @Override
    public double nextDouble(double bound) {
        if (!(bound > 0 && bound < Double.POSITIVE_INFINITY)) throw new IllegalArgumentException("bound must be positive and finite");
        double r = nextDouble() * bound;
        return r < bound ? r : Math.nextDown(bound);
    }

    /** Draws a double in {@code [origin, bound)} from {@link #nextDouble()}. */
    @Override
    public double nextDouble(double origin, double bound) {
        if (!(origin < bound && bound - origin < Double.POSITIVE_INFINITY))
            throw new IllegalArgumentException("bound must exceed origin and the range must be finite");
        double r = origin + nextDouble() * (bound - origin);
        return r < bound ? r : Math.nextDown(bound);
    }

    /** Draws a float in {@code [0, bound)} from {@link #nextFloat()}. */
    @Override
    public float nextFloat(float bound) {
        if (!(bound > 0 && bound < Float.POSITIVE_INFINITY)) throw new IllegalArgumentException("bound must be positive and finite");
        float r = nextFloat() * bound;
        return r < bound ? r : Math.nextDown(bound);
    }

    /** Draws a float in {@code [origin, bound)} from {@link #nextFloat()}. */
    @Override
    public float nextFloat(float origin, float bound) {
        if (!(origin < bound && bound - origin < Float.POSITIVE_INFINITY))
            throw new IllegalArgumentException("bound must exceed origin and the range must be finite");
        float r = origin + nextFloat() * (bound - origin);
        return r < bound ? r : Math.nextDown(bound);
    }

    // The bounded streams are sequential generators over the scalar bounded draws, so each
    // element is the value of nextInt(origin, bound) and so on, and a stream equals a loop of
    // scalar calls. They replace the JDK's own bounded algorithms. Do not run them in parallel:
    // the generator is not thread-safe.

    @Override
    public IntStream ints(long streamSize, int origin, int bound) {
        checkSize(streamSize);
        checkIntRange(origin, bound);
        return IntStream.generate(() -> nextInt(origin, bound)).limit(streamSize);
    }

    @Override
    public IntStream ints(int origin, int bound) {
        checkIntRange(origin, bound);
        return IntStream.generate(() -> nextInt(origin, bound));
    }

    @Override
    public LongStream longs(long streamSize, long origin, long bound) {
        checkSize(streamSize);
        checkLongRange(origin, bound);
        return LongStream.generate(() -> nextLong(origin, bound)).limit(streamSize);
    }

    @Override
    public LongStream longs(long origin, long bound) {
        checkLongRange(origin, bound);
        return LongStream.generate(() -> nextLong(origin, bound));
    }

    @Override
    public DoubleStream doubles(long streamSize, double origin, double bound) {
        checkSize(streamSize);
        nextDoubleRangeCheck(origin, bound);
        return DoubleStream.generate(() -> nextDouble(origin, bound)).limit(streamSize);
    }

    @Override
    public DoubleStream doubles(double origin, double bound) {
        nextDoubleRangeCheck(origin, bound);
        return DoubleStream.generate(() -> nextDouble(origin, bound));
    }

    private static void checkIntRange(int origin, int bound) {
        if (origin >= bound) throw new IllegalArgumentException("bound must exceed origin");
    }

    private static void checkLongRange(long origin, long bound) {
        if (origin >= bound) throw new IllegalArgumentException("bound must exceed origin");
    }

    private static void nextDoubleRangeCheck(double origin, double bound) {
        if (!(origin < bound && bound - origin < Double.POSITIVE_INFINITY))
            throw new IllegalArgumentException("bound must exceed origin and the range must be finite");
    }

    private static final long PURPOSE_NORMAL64 = 0x4e524d3634L;

    /** The fallback of the double normal at global draw index g: {@code gaussianFallbacks().split(g)}. */
    Tandem gaussianFallback(long g) {
        return gaussianFallbacks().split(g);
    }

    /** {@code sub(0x4e524d3634)} of this key and chunk length at position 0, the parent of every fallback. */
    Tandem gaussianFallbacks() {
        return new Tandem(k0, k1, k2, k3, 0L, chunk).sub(PURPOSE_NORMAL64);
    }

    /**
     * Returns a standard normal by the 1024-layer ziggurat of spec Appendix A from the next
     * long draw. A draw that misses the fast path, 0.43 % of them, continues on the draws of
     * {@code sub(0x4e524d3634).split(g)} of a generator with this key and chunk length at
     * position 0, g being the draw's index, which leaves the position alone. Repeated calls
     * therefore give exactly the sequence of {@link #fillGaussian(double[])}. It uses only
     * correctly rounded double arithmetic, so the result is bit identical on every JVM and in
     * tandem-c.
     */
    @Override
    public double nextGaussian() {
        long g = align(pos, 64) >>> 6;
        return Normals.gaussian(nextLong(), this, g);
    }

    /**
     * Fills with standard normals: element {@code i} is the ziggurat of draw {@code i} of
     * {@link #fill(long[])}, so a fill equals the sequence of {@link #nextGaussian()} and a fill
     * cut at any element equals the whole fill. An empty fill aligns the position to 64 bits.
     */
    public void fillGaussian(double[] a) {
        fillGaussian(a, 0, a.length);
    }

    /** As {@link #fillGaussian(double[])} on {@code a[off, off + len)}. */
    public void fillGaussian(double[] a, int off, int len) {
        Objects.checkFromIndexSize(off, len, a.length);
        if (len >= NativeFills.MIN && NativeFills.ENABLED) {
            NativeFills.fillNormalF64(this, beginFill(64, len), a, off, len);
            return;
        }
        long first = align(pos, 64) >>> 6;
        long[] r = new long[Math.min(len, GAUSSIAN_BLOCK)];
        int[] miss = new int[r.length];
        if (len == 0) fill(r, 0, 0);
        for (int i = 0; i < len; ) {
            int m = Math.min(len - i, GAUSSIAN_BLOCK);
            fill(r, 0, m);
            Normals.gaussians(r, m, a, off + i, miss, this, first + i);
            i += m;
        }
    }

    private static final int GAUSSIAN_BLOCK = 1024;

    /**
     * Returns the cosine half of the next float Box-Muller pair,
     * {@code sqrt(-2 ln(1 - a)) cos(2 pi b)} from two float draws {@code a} and {@code b}, computed
     * entirely in float, and keeps the sine half for the next call, which returns it without
     * drawing. Repeated calls therefore give exactly the sequence of {@link #fillGaussian(float[])}.
     * The kept half is dropped by {@code setPosition}, {@code split}, {@code fork} and {@code sub},
     * and is not serialized. Bit identical to tandem-c.
     */
    public float nextGaussianFloat() {
        if (hasSpareF) {
            hasSpareF = false;
            return spareF;
        }
        float u = nextFloat();
        float v = nextFloat();
        float[] z = pairBufF;
        if (z == null) pairBufF = z = new float[2];
        Normals.pairF(u, v, z, 0);
        spareF = z[1];
        hasSpareF = true;
        return z[0];
    }

    /** Draws a float Box-Muller pair {@code {cos, sin}} from two float draws, ignoring the kept half. */
    public float[] nextGaussianFloat2() {
        float u = nextFloat();
        float v = nextFloat();
        return gaussianPairF(u, v);
    }

    private static float[] gaussianPairF(float u, float v) {
        float[] z = new float[2];
        Normals.pairF(u, v, z, 0);
        return z;
    }

    /**
     * Fills with float normals in Box-Muller pairs: elements {@code 2j} and {@code 2j + 1} are the
     * cosine and sine halves from uniforms {@code 2j} and {@code 2j + 1} of {@link #fill(float[])}.
     * An odd length uses the cosine half of its last pair and still consumes both uniforms. The
     * kept half of {@link #nextGaussianFloat()} is neither used nor changed.
     */
    public void fillGaussian(float[] a) {
        fillGaussian(a, 0, a.length);
    }

    /** As {@link #fillGaussian(float[])} on {@code a[off, off + len)}. */
    public void fillGaussian(float[] a, int off, int len) {
        Objects.checkFromIndexSize(off, len, a.length);
        if (len >= NativeFills.MIN && NativeFills.ENABLED) {
            // An odd length still consumes both uniforms of its last pair.
            NativeFills.fillNormalF32(this, beginFill(32, (len + 1L) / 2 * 2), a, off, len);
            return;
        }
        float[] u = new float[2 * Math.min((len + 1) / 2, GAUSSIAN_BLOCK)];
        for (int i = off, end = off + len; i < end; ) {
            int pairs = Math.min((end - i + 1) / 2, GAUSSIAN_BLOCK);
            fill(u, 0, 2 * pairs);
            for (int j = 0; j < pairs; j++, i += 2) {
                if (i + 1 < end) {
                    Normals.pairF(u[2 * j], u[2 * j + 1], a, i);
                } else {
                    a[i] = gaussianPairF(u[2 * j], u[2 * j + 1])[0];
                }
            }
        }
    }

    /**
     * Returns {@code -ln(1 - u)} for the next double draw {@code u}, a standard exponential. It
     * uses only correctly rounded double arithmetic ({@link Math#fma}, add, multiply, divide), so
     * the result is bit identical on every JVM and in tandem-c.
     */
    @Override
    public double nextExponential() {
        return Normals.exponential(nextDouble());
    }

    /** As {@link #nextExponential()} computed in float from the next float draw. */
    public float nextExponentialFloat() {
        return Normals.exponentialF(nextFloat());
    }

    /**
     * Fills with standard exponentials: element {@code i} comes from uniform {@code i} of
     * {@link #fill(double[])}, so a fill equals the sequence of {@link #nextExponential()}.
     */
    public void fillExponential(double[] a) {
        fillExponential(a, 0, a.length);
    }

    /** As {@link #fillExponential(double[])} on {@code a[off, off + len)}. */
    public void fillExponential(double[] a, int off, int len) {
        Objects.checkFromIndexSize(off, len, a.length);
        if (len >= NativeFills.MIN && NativeFills.ENABLED) {
            NativeFills.fillExponentialF64(this, beginFill(64, len), a, off, len);
            return;
        }
        // Blocks keep the uniforms in cache for the transform.
        for (int i = off, end = off + len; i < end; ) {
            int m = Math.min(end - i, EXPONENTIAL_BLOCK);
            fill(a, i, m);
            for (int stop = i + m; i < stop; i++) a[i] = Normals.exponential(a[i]);
        }
    }

    /** As {@link #fillExponential(double[])} in float, from uniforms of {@link #fill(float[])}. */
    public void fillExponential(float[] a) {
        fillExponential(a, 0, a.length);
    }

    /** As {@link #fillExponential(float[])} on {@code a[off, off + len)}. */
    public void fillExponential(float[] a, int off, int len) {
        Objects.checkFromIndexSize(off, len, a.length);
        if (len >= NativeFills.MIN && NativeFills.ENABLED) {
            NativeFills.fillExponentialF32(this, beginFill(32, len), a, off, len);
            return;
        }
        for (int i = off, end = off + len; i < end; ) {
            int m = Math.min(end - i, EXPONENTIAL_BLOCK);
            fill(a, i, m);
            for (int stop = i + m; i < stop; i++) a[i] = Normals.exponentialF(a[i]);
        }
    }

    private static final int EXPONENTIAL_BLOCK = 1024;

    /**
     * Draws an index in {@code [0, table.size())} with probability proportional to its weight,
     * Appendix C of the specification. It consumes one {@link #nextLong()} draw, with no retry,
     * and equals element 0 of {@link #fillChoice}.
     */
    public int nextChoice(ChoiceTable table) {
        return table.index(nextLong());
    }

    /**
     * Fills with indices drawn from {@code table}: element {@code i} maps draw {@code i} of
     * {@link #fill(long[])}, so a fill equals the sequence of {@link #nextChoice} and a fill cut
     * anywhere equals the whole fill. An empty fill aligns the position to 64 bits.
     */
    public void fillChoice(int[] a, ChoiceTable table) {
        fillChoice(a, 0, a.length, table);
    }

    /** As {@link #fillChoice(int[], ChoiceTable)} on {@code a[off, off + len)}. */
    public void fillChoice(int[] a, int off, int len, ChoiceTable table) {
        Objects.checkFromIndexSize(off, len, a.length);
        Objects.requireNonNull(table);
        long[] r = new long[Math.min(len, CHOICE_BLOCK)];
        // One pass even for len = 0, whose empty fill aligns the position.
        int i = off, end = off + len;
        do {
            int m = Math.min(end - i, CHOICE_BLOCK);
            fill(r, 0, m);
            for (int k = 0; k < m; k++) a[i++] = table.index(r[k]);
        } while (i < end);
    }

    private static final int CHOICE_BLOCK = 1024;

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
    private long beginFill(int w, long n) {
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
        if (len >= NativeFills.MIN && NativeFills.ENABLED) {
            NativeFills.fillU32(this, beginFill(32, len), a, off, len);
            return;
        }
        long p = beginFill(32, len);
        int i = off, end = off + len;
        for (; i < end && (p & 1023) != 0; i++, p += 32) a[i] = word(p);
        while (end - i >= 32) {
            int br = Math.min(chunk, BLOCK_ROWS);
            if (br > 1 && (p >>> 10 & (br - 1)) == 0 && end - i >= br << 5) {
                Cache c = blockState(p >>> 10, br);
                generateInt(c.o, c.h, a, i, br);
                c.next += br;
                i += br << 5;
                p += (long) br << 10;
                continue;
            }
            int base = locate(p), rows = Math.min(cache.count - (base >> 4), (end - i) >> 5);
            long[] x = cache.buf;
            for (int k = 0, n = rows << 4; k < n; k++) {
                long v = x[base + k];
                a[i + 2 * k] = (int) v;
                a[i + 2 * k + 1] = (int) (v >>> 32);
            }
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
        if (len >= NativeFills.MIN && NativeFills.ENABLED) {
            NativeFills.fillU64(this, beginFill(64, len), a, off, len);
            return;
        }
        long p = beginFill(64, len);
        int i = off, end = off + len;
        for (; i < end && (p & 1023) != 0; i++, p += 64) a[i] = rawCached(p, 64);
        while (end - i >= 16) {
            int br = Math.min(chunk, BLOCK_ROWS);
            if (br > 1 && (p >>> 10 & (br - 1)) == 0 && end - i >= br << 4) {
                Cache c = blockState(p >>> 10, br);
                generate(c.o, c.h, a, i, br);
                c.next += br;
                i += br << 4;
                p += (long) br << 10;
                continue;
            }
            int base = locate(p), rows = Math.min(cache.count - (base >> 4), (end - i) >> 4);
            System.arraycopy(cache.buf, base, a, i, rows << 4);
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
        if (len >= NativeFills.MIN && NativeFills.ENABLED) {
            NativeFills.fillF32(this, beginFill(32, len), a, off, len);
            return;
        }
        long p = beginFill(32, len);
        int i = off, end = off + len;
        for (; i < end && (p & 1023) != 0; i++, p += 32) a[i] = toFloat(word(p));
        while (end - i >= 32) {
            int br = Math.min(chunk, BLOCK_ROWS);
            if (br > 1 && (p >>> 10 & (br - 1)) == 0 && end - i >= br << 5) {
                Cache c = blockState(p >>> 10, br);
                generateFloat(c.o, c.h, a, i, br);
                c.next += br;
                i += br << 5;
                p += (long) br << 10;
                continue;
            }
            int base = locate(p), rows = Math.min(cache.count - (base >> 4), (end - i) >> 5);
            long[] x = cache.buf;
            for (int k = 0, n = rows << 4; k < n; k++) {
                long v = x[base + k];
                a[i + 2 * k] = toFloat((int) v);
                a[i + 2 * k + 1] = toFloat((int) (v >>> 32));
            }
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
        if (len >= NativeFills.MIN && NativeFills.ENABLED) {
            NativeFills.fillF64(this, beginFill(64, len), a, off, len);
            return;
        }
        long p = beginFill(64, len);
        int i = off, end = off + len;
        for (; i < end && (p & 1023) != 0; i++, p += 64) a[i] = toDouble(rawCached(p, 64));
        while (end - i >= 16) {
            int br = Math.min(chunk, BLOCK_ROWS);
            if (br > 1 && (p >>> 10 & (br - 1)) == 0 && end - i >= br << 4) {
                Cache c = blockState(p >>> 10, br);
                generateDouble(c.o, c.h, a, i, br);
                c.next += br;
                i += br << 4;
                p += (long) br << 10;
                continue;
            }
            int base = locate(p), rows = Math.min(cache.count - (base >> 4), (end - i) >> 4);
            long[] x = cache.buf;
            for (int k = 0, n = rows << 4; k < n; k++) a[i + k] = toDouble(x[base + k]);
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
            int base = locate(p), rows = Math.min(cache.count - (base >> 4), (end - i) >> 7);
            long[] x = cache.buf;
            for (int k = 0, n = rows << 4; k < n; k++) {
                long v = x[base + k];
                for (int b = 0; b < 8; b++) a[i + 8 * k + b] = (byte) (v >>> (8 * b));
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
            int base = locate(p), rows = Math.min(cache.count - (base >> 4), (end - i) >> 6);
            long[] x = cache.buf;
            for (int k = 0, n = rows << 4; k < n; k++) {
                long v = x[base + k];
                for (int b = 0; b < 4; b++) a[i + 4 * k + b] = (short) (v >>> (16 * b));
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
