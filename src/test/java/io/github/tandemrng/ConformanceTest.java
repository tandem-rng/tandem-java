package io.github.tandemrng;

import static io.github.tandemrng.Conformance.align;
import static io.github.tandemrng.Conformance.cases;
import static io.github.tandemrng.Conformance.find;
import static io.github.tandemrng.Conformance.list;
import static io.github.tandemrng.Conformance.load;
import static io.github.tandemrng.Conformance.obj;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.tandemrng.Conformance.Case;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.function.ToLongFunction;
import java.util.stream.LongStream;
import org.junit.jupiter.api.Test;

/**
 * The spec's conformance cases and every item of its {@code conformance/CHECKLIST.md} at
 * tandem-spec b31af72, from the copies in {@code src/test/resources/conformance}. Every value
 * compares bit for bit: the Float32 normals and exponentials copy the C polynomials, so they
 * need no tolerance.
 */
class ConformanceTest {
    private static final long MASK32 = 0xffffffffL;
    private static final List<String> FILL_FILES = List.of("fill_below.json", "normal.json", "exponential.json", "choice.json");

    private interface Fill {
        long[] apply(Tandem g, int n);
    }

    /** A kind's fill and scalar draw as bit patterns, and its draw width. */
    private record Draws(int w, Fill fill, ToLongFunction<Tandem> scalar) {}

    private static long[] u32(int[] a) {
        return Arrays.stream(a).mapToLong(x -> x & MASK32).toArray();
    }

    private static long[] bits(double[] a) {
        return Arrays.stream(a).mapToLong(Double::doubleToRawLongBits).toArray();
    }

    private static long[] bits(float[] a) {
        return LongStream.range(0, a.length).map(i -> Float.floatToRawIntBits(a[(int) i]) & MASK32).toArray();
    }

    static ChoiceTable table(Case c) {
        return new ChoiceTable(Arrays.stream(c.hex("weights")).mapToDouble(Double::longBitsToDouble).toArray());
    }

    /** The draws of a kind. The case gives the range or the weights. */
    private static Draws draws(String kind, Case c) {
        return switch (kind) {
            case "below_u32", "fill_below_u32" -> {
                int r = (int) c.word("range");
                yield new Draws(32, (g, n) -> {
                    int[] a = new int[n];
                    g.fillBelowU32(a, 0, n, r);
                    return u32(a);
                }, g -> g.belowU32(r) & MASK32);
            }
            case "below_u64", "fill_below_u64" -> {
                long r = c.word("range");
                yield new Draws(64, (g, n) -> {
                    long[] a = new long[n];
                    g.fillBelowU64(a, 0, n, r);
                    return a;
                }, g -> g.belowU64(r));
            }
            case "fill_normal_f64" -> new Draws(64, (g, n) -> {
                double[] a = new double[n];
                g.fillGaussian(a);
                return bits(a);
            }, g -> Double.doubleToRawLongBits(g.nextGaussian()));
            case "fill_normal_f32" -> new Draws(32, (g, n) -> {
                float[] a = new float[n];
                g.fillGaussian(a);
                return bits(a);
            }, g -> Float.floatToRawIntBits(g.nextGaussianFloat()) & MASK32);
            case "fill_exponential_f64" -> new Draws(64, (g, n) -> {
                double[] a = new double[n];
                g.fillExponential(a);
                return bits(a);
            }, g -> Double.doubleToRawLongBits(g.nextExponential()));
            case "fill_exponential_f32" -> new Draws(32, (g, n) -> {
                float[] a = new float[n];
                g.fillExponential(a);
                return bits(a);
            }, g -> Float.floatToRawIntBits(g.nextExponentialFloat()) & MASK32);
            case "fill_choice" -> {
                ChoiceTable t = table(c);
                yield new Draws(64, (g, n) -> {
                    int[] a = new int[n];
                    g.fillChoice(a, t);
                    return u32(a);
                }, g -> g.nextChoice(t));
            }
            default -> throw new IllegalArgumentException(kind);
        };
    }

    private static Draws draws(Case c) {
        return draws(c.kind(), c);
    }

    /** The end position, which the file pins for some cases. A Float32 normal consumes whole pairs. */
    private static long end(Case c) {
        if (c.kind().equals("fill_normal_f32") && !c.has("end"))
            return align(c.start(), 32) + 64L * ((c.n() + 1) / 2);
        return c.end(draws(c).w);
    }

    // ---- Fallback by global draw index, n = 0, odd n, weighted choice: the case values ------

    @Test
    void everyFillCaseGivesItsValuesAndEnd() {
        int checked = 0;
        for (String file : FILL_FILES)
            for (Case c : cases(file)) {
                Tandem g = c.generator();
                assertArrayEquals(c.hex("values"), draws(c).fill.apply(g, c.n()), c.id());
                assertEquals(end(c), g.position(), c.id());
                checked++;
            }
        assertEquals(77 + 20 + 12 + 24, checked);
    }

    /** Scalar draws equal each fill case without a bounded fallback, with the same end. */
    @Test
    void scalarDrawsEqualTheFillCases() {
        for (String file : List.of("below.json", "normal.json", "exponential.json", "choice.json"))
            for (Case c : cases(file)) {
                if (c.n() == 0) continue;
                Draws d = draws(c);
                Tandem g = c.generator();
                long[] got = new long[c.n()];
                for (int i = 0; i < got.length; i++) got[i] = d.scalar.applyAsLong(g);
                assertArrayEquals(c.hex("values"), got, c.id());
                assertEquals(end(c), g.position(), c.id());
            }
    }

    /** Element i of the later case equals element i + shift of the earlier one. */
    private static void assertShifted(String file, String earlier, String later, int shift) {
        long[] base = find(file, earlier).hex("values");
        Case c = find(file, later);
        long[] got = draws(c).fill.apply(c.generator(), c.n());
        assertArrayEquals(Arrays.copyOfRange(base, shift, c.n()), Arrays.copyOf(got, c.n() - shift), later);
    }

    @Test
    void fallbacksAreKeyedByTheGlobalDrawIndex() {
        assertEquals(41, find("fill_below.json", "CROSS_BELOW32[4]").num("rejected"));
        assertShifted("fill_below.json", "CROSS_BELOW32[4]", "CROSS_BELOW32_AT[4]", 1);
        assertShifted("fill_below.json", "CROSS_BELOW64[6]", "CROSS_BELOW64_AT[6]", 1);
        assertShifted("normal.json", "CROSS_NORMAL[0]", "CROSS_NORMAL[1]", 1);
    }

    // ---- Width from range ----------------------------------------------------------------

    @Test
    void widthFollowsTheRangeWhenOnlyTheResultTypeIsNamed() {
        Case narrow = find("fill_below.json", "CROSS_BELOW32[3]"), wide = find("fill_below.json", "CROSS_BELOW64[3]");
        long[] a = new long[64];
        Tandem g = narrow.generator();
        g.nextLongs(a, 1000);
        assertArrayEquals(narrow.hex("values"), a);
        assertEquals(64 * 32, g.position());
        int[] b = new int[64];
        narrow.generator().nextInts(b, 1000);
        assertArrayEquals(narrow.hex("values"), u32(b));
        long[] c = new long[64];
        wide.generator().fillBelowU64(c, 0, 64, 1000);
        assertArrayEquals(wide.hex("values"), c);
    }

    // ---- Odd n and the pair rule of Float32 Box-Muller -----------------------------------

    @Test
    void floatNormalsComeInPairs() {
        Case c0 = find("normal.json", "CROSS_NORMAL32[0]");
        Tandem g = c0.generator();
        draws(c0).fill.apply(g, 33);
        assertEquals(1088, g.position());

        long[] pairs = find("normal.json", "CROSS_NORMALF").hex("values");
        assertArrayEquals(find("normal.json", "CROSS_NORMAL32[1]").hex("values"), Arrays.copyOf(pairs, 33));
        assertShifted("normal.json", "CROSS_NORMAL32[0]", "CROSS_NORMAL32[2]", 2);

        Tandem h = find("normal.json", "CROSS_NORMALF").generator();
        assertEquals(pairs[0], Float.floatToRawIntBits(h.nextGaussianFloat()) & MASK32);
        assertEquals(32 + 64, h.position());
    }

    // ---- Weighted choice -----------------------------------------------------------------

    @Test
    void choiceTablesEqualTheSpecVectors() {
        int vectors = 0;
        for (Case c : cases("choice.json")) {
            ChoiceTable t = table(c);
            assertEquals(c.word("capacity"), t.capacity(), c.id());
            if (!c.has("cut")) continue;
            assertArrayEquals(c.hex("cut"), t.cut(), c.id());
            assertArrayEquals(c.hex("alias"), Arrays.stream(t.alias()).asLongStream().toArray(), c.id());
            vectors++;
        }
        assertEquals(5, vectors);
    }

    @Test
    void choiceDrawsOneLongAndCutsAnywhere() {
        assertShifted("choice.json", "CROSS_CHOICE[0]", "CROSS_CHOICE[1]", 1);
        Tandem g = Tandem.seed(42);
        g.setPosition(33);
        assertEquals(0, g.nextChoice(new ChoiceTable(new double[] {0.25})));
        assertEquals(128, g.position());
    }

    @Test
    void invalidWeightsBuildNoTable() {
        double[][] bad = {{}, {1, -1}, {1, Double.NaN}, {1, Double.POSITIVE_INFINITY}, {Double.NEGATIVE_INFINITY}, {0.0, -0.0}};
        for (double[] w : bad) assertThrows(IllegalArgumentException.class, () -> new ChoiceTable(w), Arrays.toString(w));
    }

    // ---- Cut fill ------------------------------------------------------------------------

    /**
     * Pieces filled in order on one generator equal the whole fill. A Float32 normal fill writes
     * whole pairs, so its cuts fall at pair boundaries: 2, 8, 20 and the largest even element
     * below n.
     */
    @Test
    void fillsCutAtElementBoundariesEqualTheWholeFill() {
        for (String file : FILL_FILES)
            for (Case c : cases(file)) {
                int n = c.n();
                int[] cuts = c.kind().equals("fill_normal_f32") ? new int[] {2, 8, 20, (n - 1) & ~1} : new int[] {1, 7, 20, 21, n - 1};
                Fill fill = draws(c).fill;
                for (int cut : cuts) {
                    if (cut <= 0 || cut >= n) continue;
                    Tandem g = c.generator();
                    long[] head = fill.apply(g, cut), tail = fill.apply(g, n - cut);
                    long[] got = LongStream.concat(Arrays.stream(head), Arrays.stream(tail)).toArray();
                    assertArrayEquals(c.hex("values"), got, c.id() + " cut " + cut);
                    assertEquals(end(c), g.position(), c.id() + " cut " + cut);
                }
            }
    }

    // ---- Block and 2^63 position boundaries ----------------------------------------------

    private static String sha256(byte[] b) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(b));
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
    }

    /** The little-endian bytes of n elements of a type of the stream dumps. */
    private static byte[] streamBytes(Tandem g, String type, int n) {
        return switch (type) {
            case "UInt8" -> {
                byte[] a = new byte[n];
                g.fill(a);
                yield a;
            }
            case "Bool" -> {
                boolean[] a = new boolean[n];
                g.fill(a);
                byte[] b = new byte[n];
                for (int i = 0; i < n; i++) b[i] = (byte) (a[i] ? 1 : 0);
                yield b;
            }
            case "Float16" -> {
                short[] a = new short[n];
                g.fillFloat16Bits(a);
                ByteBuffer b = le(2 * n);
                b.asShortBuffer().put(a);
                yield b.array();
            }
            case "UInt32", "Char" -> {
                int[] a = new int[n];
                if (type.equals("Char")) g.fillCodePoints(a);
                else g.fill(a);
                ByteBuffer b = le(4 * n);
                b.asIntBuffer().put(a);
                yield b.array();
            }
            case "Float32", "ComplexF32" -> {
                float[] a = new float[type.equals("Float32") ? n : 2 * n];
                g.fill(a);
                ByteBuffer b = le(4 * a.length);
                b.asFloatBuffer().put(a);
                yield b.array();
            }
            case "UInt64", "UInt128" -> {
                long[] a = new long[type.equals("UInt64") ? n : 2 * n];
                if (type.equals("UInt64")) g.fill(a);
                else g.fillLong128(a);
                ByteBuffer b = le(8 * a.length);
                b.asLongBuffer().put(a);
                yield b.array();
            }
            case "Float64", "ComplexF64" -> {
                double[] a = new double[type.equals("Float64") ? n : 2 * n];
                g.fill(a);
                ByteBuffer b = le(8 * a.length);
                b.asDoubleBuffer().put(a);
                yield b.array();
            }
            default -> throw new IllegalArgumentException(type);
        };
    }

    private static ByteBuffer le(int bytes) {
        return ByteBuffer.allocate(bytes).order(ByteOrder.LITTLE_ENDIAN);
    }

    @Test
    void streamHashes() {
        List<Object> streams = list(load("hashes.json").get("streams"));
        assertEquals(12, streams.size());
        for (Object o : streams) {
            Case s = new Case(obj(o));
            byte[] b = streamBytes(s.generator(), (String) s.f().get("type"), s.n());
            assertEquals(s.num("bytes"), b.length, (String) s.f().get("file"));
            assertEquals(s.f().get("sha256"), sha256(b), (String) s.f().get("file"));
        }
    }

    /** Long derived outputs: each start runs the listed fills in order on one generator. */
    @Test
    void dumpHashes() throws NoSuchAlgorithmException {
        for (Object o : list(load("hashes.json").get("dumps"))) {
            Map<String, Object> d = obj(o);
            String id = (String) d.get("id");
            MessageDigest sha = MessageDigest.getInstance("SHA-256");
            long fnv = 0xcbf29ce484222325L, bytes = 0;
            Tandem g = null;
            for (Object start : list(d.get("starts"))) {
                g = Conformance.generator(d.get("key"), ((Number) d.get("K")).intValue(), ((Number) start).longValue());
                for (Object draw : list(d.get("draws"))) {
                    Case c = new Case(obj(draw));
                    Draws k = draws(c.kind(), c);
                    ByteBuffer b = le(k.w / 8 * c.n());
                    for (long x : k.fill.apply(g, c.n())) {
                        if (k.w == 64) b.putLong(x);
                        else b.putInt((int) x);
                    }
                    byte[] a = b.array();
                    for (byte x : a) fnv = (fnv ^ (x & 0xff)) * 0x100000001b3L;
                    sha.update(a);
                    bytes += a.length;
                }
            }
            assertEquals(((Number) d.get("bytes")).longValue(), bytes, id);
            assertEquals(Long.parseUnsignedLong((String) d.get("fnv1a"), 16), fnv, id);
            if (d.containsKey("sha256")) assertEquals(d.get("sha256"), HexFormat.of().formatHex(sha.digest()), id);
            if (d.containsKey("end")) assertEquals(((Number) d.get("end")).longValue(), g.position(), id);
        }
    }

    @Test
    void complexDrawsTakeTheirImaginaryPartFromTheNextBlock() {
        double[] d = new double[3];
        Tandem.seed(42).fill(d);
        Tandem g = Tandem.seed(42);
        g.setPosition(64);
        assertArrayEquals(new double[] {d[1], d[2]}, g.nextComplexDouble());
        assertEquals(192, g.position());
        float[] f = new float[5];
        Tandem.seed(42).fill(f);
        Tandem h = Tandem.seed(42);
        h.setPosition(96);
        assertArrayEquals(new float[] {f[3], f[4]}, h.nextComplexFloat());
        assertEquals(160, h.position());
    }

    /**
     * Positions are unsigned. A start is below 2^63, a draw may then pass it. The item on a fill
     * that reaches 2^64 does not apply: starts lie below 2^63 and a fill holds at most 2^31
     * elements of at most 128 bits, so a fill cannot reach 2^64.
     */
    @Test
    void startsBelow2To63AndDrawsPastIt() {
        int[] key = {1, 2, 3, 4};
        Tandem g = new Tandem(key, Long.MAX_VALUE, 32);
        for (long bad : new long[] {Long.MIN_VALUE, -1L}) {
            assertThrows(IllegalArgumentException.class, () -> new Tandem(key, bad, 32));
            assertThrows(IllegalArgumentException.class, () -> g.setPosition(bad));
            assertEquals(Long.MAX_VALUE, g.position());
        }
        long r = g.nextLong();
        assertEquals(Long.MIN_VALUE + 64, g.position());
        assertEquals(new Tandem(key, 0L, 32).atLong(1L << 57), r);
    }
}
