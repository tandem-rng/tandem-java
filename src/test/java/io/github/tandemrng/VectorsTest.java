package io.github.tandemrng;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

/** Every vector of section 8 of the specification, from the generated {@link Vectors}. */
class VectorsTest {
    private static final int DOMAIN_STREAM = 0x9e3779b9;
    private static final int AUX_STREAM = 0x94d049bb;

    private static Tandem keyed() {
        return new Tandem(Vectors.KEY, 0L, Vectors.K);
    }

    @Test
    void stepT() {
        for (int[][] v : Vectors.T) {
            int[] s = {v[0][0], v[0][1], v[0][2], v[0][3], v[1][0], v[1][1], v[1][2], v[1][3]};
            Tandem.t(s);
            assertArrayEquals(v[2], new int[] {s[0], s[1], s[2], s[3]});
            assertArrayEquals(v[3], new int[] {s[4], s[5], s[6], s[7]});
        }
    }

    @Test
    void seedingFunction() {
        int[] k = Vectors.KEY;
        for (int i = 0; i < Vectors.F_COUNTER.length; i++) {
            int[] s = new int[8];
            Tandem.fKeyed(k[0], k[1], k[2], k[3], Vectors.F_COUNTER[i], DOMAIN_STREAM, AUX_STREAM, s);
            assertArrayEquals(Vectors.F_O[i], new int[] {s[0], s[1], s[2], s[3]});
            assertArrayEquals(Vectors.F_H[i], new int[] {s[4], s[5], s[6], s[7]});
        }
    }

    @Test
    void streamWordsByRandomAccess() {
        Tandem g = keyed();
        for (int i = 0; i < Vectors.STREAM_FIRST_WORD.length; i++)
            for (int j = 0; j < 4; j++)
                assertEquals(Vectors.STREAM_WORDS[i][j], g.atInt(Vectors.STREAM_FIRST_WORD[i] + j));
    }

    @Test
    void streamWordsBySequentialDraws() {
        Tandem g = keyed();
        int[] all = new int[36];
        for (int i = 0; i < all.length; i++) all[i] = g.nextInt();
        for (int i = 0; i < Vectors.STREAM_FIRST_WORD.length; i++)
            for (int j = 0; j < 4; j++)
                assertEquals(Vectors.STREAM_WORDS[i][j], all[(int) Vectors.STREAM_FIRST_WORD[i] + j]);
    }

    @Test
    void derivedDraws() {
        Tandem g = keyed();
        for (int i = 0; i < Vectors.F64_INDEX.length; i++)
            assertEquals(Vectors.F64_VALUE[i], g.atDouble(Vectors.F64_INDEX[i]));
        for (int i = 0; i < Vectors.F32_INDEX.length; i++)
            assertEquals(Vectors.F32_VALUE[i], g.atFloat(Vectors.F32_INDEX[i]));
        for (int i = 0; i < Vectors.BOOL_INDEX.length; i++)
            assertEquals(Vectors.BOOL_VALUE[i], g.atBoolean(Vectors.BOOL_INDEX[i]));
    }

    @Test
    void derivedDrawsBySequentialDraws() {
        double[] d = new double[17];
        keyed().fill(d);
        for (int i = 0; i < Vectors.F64_INDEX.length; i++)
            assertEquals(Vectors.F64_VALUE[i], d[(int) Vectors.F64_INDEX[i]]);
        boolean[] b = new boolean[129];
        keyed().fill(b);
        for (int i = 0; i < Vectors.BOOL_INDEX.length; i++)
            assertEquals(Vectors.BOOL_VALUE[i], b[(int) Vectors.BOOL_INDEX[i]]);
    }

    @Test
    void derivedKeys() {
        assertArrayEquals(Vectors.SPLIT0, keyed().split(0).key());
        assertArrayEquals(Vectors.SPLIT1, keyed().split(1).key());
        assertArrayEquals(Vectors.FORK0, keyed().fork(1)[0].key());
        assertArrayEquals(Vectors.PURPOSE7, keyed().sub(7).key());
    }

    @Test
    void seedWhitening() {
        Tandem g = Tandem.seed(Vectors.SEED_LO, Vectors.SEED_HI);
        assertArrayEquals(Vectors.SEED_KEY, g.key());
        for (int i = 0; i < Vectors.SEED_F64_INDEX.length; i++)
            assertEquals(Vectors.SEED_F64_VALUE[i], g.atDouble(Vectors.SEED_F64_INDEX[i]));
        for (int i = 0; i < Vectors.SEED_U32_INDEX.length; i++)
            assertEquals(Vectors.SEED_U32_VALUE[i], g.atInt(Vectors.SEED_U32_INDEX[i]));
    }
}
