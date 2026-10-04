package io.github.tandemrng;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.random.RandomGenerator;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

/** Transport, child generators and the {@link RandomGenerator} contract. */
class TandemTest {
    private static final int[] KEY = {11, 22, 33, 44};

    private static int[] half(int[] key, long counter, int domain, int aux, int h) {
        int[] s = new int[8];
        Tandem.fKeyed(key[0], key[1], key[2], key[3], counter, domain, aux, s);
        return new int[] {s[4 * h], s[4 * h + 1], s[4 * h + 2], s[4 * h + 3]};
    }

    @Test
    void keyAsLongsMatchesTheWords() {
        Tandem g = new Tandem(KEY, 0L, 32);
        assertEquals(11L | (22L << 32), g.keyLo());
        assertEquals(33L | (44L << 32), g.keyHi());
        Tandem h = new Tandem(g.keyLo(), g.keyHi(), 0L, 32);
        assertArrayEquals(KEY, h.key());
        assertEquals(g.nextLong(), h.nextLong());
    }

    /** The seed is a 128-bit value: the high half changes the key and the whitening is not the identity. */
    @Test
    void seedUsesBothHalvesAndTheChunkLength() {
        assertNotEquals(Tandem.seed(1, 0).keyLo(), Tandem.seed(1, 1).keyLo());
        assertEquals(1, Tandem.seed(5, 6, 1).chunkLength());
        assertEquals(Tandem.seed(5, 0).keyLo(), Tandem.seed(5).keyLo());
    }

    @Test
    void positionAlignsToTheWidthOfEachDraw() {
        Tandem g = Tandem.seed(1);
        g.nextBoolean();
        g.nextByte();
        assertEquals(16, g.position());
        g.nextBoolean();
        g.nextShort();
        assertEquals(48, g.position());
        g.nextInt();
        assertEquals(96, g.position());
        g.nextBoolean();
        g.nextLong();
        assertEquals(192, g.position());
        g.nextBoolean();
        g.nextLong128();
        assertEquals(384, g.position());
    }

    @Test
    void splitDependsOnTheKeyOnly() {
        Tandem g = new Tandem(KEY, 12345, 8);
        for (long i : new long[] {0, 1, 2, 3, 1L << 40, -1L, -2L}) {
            Tandem c = g.split(i);
            assertArrayEquals(half(KEY, i >>> 1, 0xbb67ae85, 0, (int) (i & 1)), c.key(), "index " + i);
            assertEquals(0, c.position());
            assertEquals(8, c.chunkLength());
        }
        assertEquals(12345, g.position());
    }

    @Test
    void subDependsOnTheKeyOnly() {
        Tandem g = new Tandem(KEY, 999, 4);
        for (long u : new long[] {0, 7, -1L}) {
            Tandem c = g.sub(u);
            assertArrayEquals(half(KEY, u, 0xcd9e8d57, 0, 0), c.key());
            assertEquals(0, c.position());
            assertEquals(4, c.chunkLength());
        }
        assertEquals(999, g.position());
    }

    @Test
    void forkDerivesFromTheCurrentBlockAndMovesOn() {
        Tandem g = new Tandem(KEY, 128L * 5 + 77, 16);
        Tandem[] kids = g.fork(5);
        for (int i = 0; i < 5; i++) {
            assertArrayEquals(half(KEY, 5, 0xd2511f53, i >> 1, i & 1), kids[i].key(), "child " + i);
            assertEquals(0, kids[i].position());
            assertEquals(16, kids[i].chunkLength());
        }
        assertEquals(128L * 6, g.position());
    }

    @Test
    void forkAtABlockBoundaryAndEmptyForkStillAdvance() {
        Tandem g = new Tandem(KEY, 128L * 3, 32);
        assertEquals(0, g.fork(0).length);
        assertEquals(128L * 4, g.position());
        g.fork(1);
        assertEquals(128L * 5, g.position());
    }

    @Test
    void copyIsIndependent() {
        Tandem g = Tandem.seed(8);
        g.nextLong();
        Tandem c = g.copy();
        long want = g.nextLong();
        assertEquals(64, c.position());
        assertEquals(want, c.nextLong());
    }

    @Test
    void serializationKeepsTheStream() throws IOException, ClassNotFoundException {
        Tandem g = Tandem.seed(21, 22, 64);
        for (int i = 0; i < 100; i++) g.nextLong();
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (ObjectOutputStream out = new ObjectOutputStream(bytes)) {
            out.writeObject(g);
        }
        Tandem h;
        try (ObjectInputStream in = new ObjectInputStream(new ByteArrayInputStream(bytes.toByteArray()))) {
            h = (Tandem) in.readObject();
        }
        assertArrayEquals(g.key(), h.key());
        assertEquals(g.position(), h.position());
        assertEquals(64, h.chunkLength());
        for (int i = 0; i < 300; i++) assertEquals(g.nextLong(), h.nextLong());
    }

    // ---- RandomGenerator contract -----------------------------------------------------------

    @Test
    void streamsAreTheDraws() {
        Tandem g = Tandem.seed(4), h = Tandem.seed(4);
        int[] ints = g.ints(50).toArray();
        for (int i = 0; i < 50; i++) assertEquals(h.nextInt(), ints[i]);
        double[] doubles = g.doubles(50).toArray();
        for (int i = 0; i < 50; i++) assertEquals(h.nextDouble(), doubles[i]);
        assertTrue(Tandem.seed(4).doubles(1000).allMatch(d -> d >= 0 && d < 1));
        assertTrue(Tandem.seed(4).ints(1000, -5, 17).allMatch(v -> v >= -5 && v < 17));
    }

    @Test
    void splitIsTheForkOfOneChild() {
        Tandem g = Tandem.seed(6), h = g.copy();
        Tandem a = g.split(), b = h.fork(1)[0];
        assertArrayEquals(b.key(), a.key());
        assertEquals(h.position(), g.position());
    }

    @Test
    void splitFromASourceIsDeterministicInTheSource() {
        Tandem a = Tandem.seed(2).split(Tandem.seed(100)), b = Tandem.seed(2).split(Tandem.seed(100));
        assertArrayEquals(a.key(), b.key());
        assertEquals(a.chunkLength(), b.chunkLength());
        assertNotEquals(a.keyLo(), Tandem.seed(2).split(Tandem.seed(101)).keyLo());
    }

    @Test
    void splitsYieldDistinctChildren() {
        Set<Long> firsts = new HashSet<>();
        List<RandomGenerator.SplittableGenerator> kids =
                Tandem.seed(3).splits(64).collect(Collectors.toList());
        assertEquals(64, kids.size());
        for (RandomGenerator.SplittableGenerator k : kids) firsts.add(k.nextLong());
        assertEquals(64, firsts.size());
        assertEquals(10, Tandem.seed(3).splits(Tandem.seed(8)).limit(10).count());
    }

    /** Collections.shuffle takes a RandomGenerator from JDK 21. */
    @Test
    void shuffleDrivenByTandem() throws ReflectiveOperationException {
        assumeTrue(Runtime.version().feature() >= 21);
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < 100; i++) list.add(i);
        List<Integer> a = new ArrayList<>(list), b = new ArrayList<>(list);
        var shuffle = Collections.class.getMethod("shuffle", List.class, RandomGenerator.class);
        shuffle.invoke(null, a, Tandem.seed(12));
        shuffle.invoke(null, b, Tandem.seed(12));
        assertEquals(a, b);
        assertNotEquals(list, a);
        Collections.sort(a);
        assertEquals(list, a);
    }

    @Test
    void boundedStreamsAreLoopsOfScalarDraws() {
        Tandem g = Tandem.seed(13), h = Tandem.seed(13);
        int[] ints = g.ints(300, 0, 1000).toArray();
        for (int i = 0; i < ints.length; i++) assertEquals(h.nextInt(1000), ints[i]);
        Tandem a = Tandem.seed(13);
        int[] shifted = a.ints(300, 10, 1010).toArray();
        for (int i = 0; i < ints.length; i++) assertEquals(ints[i] + 10, shifted[i]);
        long[] longs = Tandem.seed(14).longs(200, 0, 1L << 40).toArray();
        Tandem l = Tandem.seed(14);
        for (int i = 0; i < longs.length; i++) assertEquals(l.nextLong(1L << 40), longs[i]);
        long[] lshift = Tandem.seed(14).longs(200, -7, (1L << 40) - 7).toArray();
        for (int i = 0; i < longs.length; i++) assertEquals(longs[i] - 7, lshift[i]);
        double[] ds = Tandem.seed(15).doubles(100, 2.0, 5.0).toArray();
        Tandem d = Tandem.seed(15);
        for (int i = 0; i < ds.length; i++) assertEquals(2.0 + d.nextDouble() * 3.0, ds[i]);
        assertEquals(h.nextInt(0, 1000), g.ints(0, 1000).findFirst().getAsInt());
    }

    @Test
    void boundedFloatsAndDoublesUseOurUniforms() {
        Tandem g = Tandem.seed(16), h = Tandem.seed(16);
        assertEquals(h.nextDouble() * 7.0, g.nextDouble(7.0));
        assertEquals(-1.0 + h.nextDouble() * 3.0, g.nextDouble(-1.0, 2.0));
        assertEquals(h.nextFloat() * 7f, g.nextFloat(7f));
        assertEquals(-1f + h.nextFloat() * 3f, g.nextFloat(-1f, 2f));
    }
}
