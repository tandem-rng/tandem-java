package io.github.tandemrng;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

/**
 * A fill of {@link NativeFills#MIN} elements or more runs in libtandem when it is loaded. It must
 * equal the same fill cut into pieces below that length, which run in Java, and end at the same
 * position. Without the library both sides run in Java.
 */
class NativeFillsTest {
    private static final int[] KEY = {5, 6, 7, 8};
    private static final long[] STARTS = {0, 1, 77, 1023, 12345, (1L << 40) + 5};
    private static final int[] CHUNKS = {1, 32, 1024};
    // Over one native call of 2^16 elements, odd for the float normal pair.
    private static final int[] LENGTHS = {NativeFills.MIN, NativeFills.MIN + 1, 70001};
    // Even, so the float normal pairs of the pieces line up with those of the whole fill.
    private static final int PIECE = NativeFills.MIN - 2;

    /** The property names a library, else java.library.path may hold one. Either must load. */
    @Test
    void theLibraryLoadsExactlyWhenNamedOrOnTheLibraryPath() {
        boolean onPath = false;
        for (String dir : System.getProperty("java.library.path", "").split(java.io.File.pathSeparator))
            onPath |= !dir.isEmpty() && java.nio.file.Files.isRegularFile(java.nio.file.Path.of(dir, System.mapLibraryName("tandem")));
        assertEquals(System.getProperty("tandem.native") != null || onPath, NativeFills.ENABLED);
    }

    private interface Fill<A> {
        void apply(Tandem g, A a, int off, int len);
    }

    private static <A> void check(String name, java.util.function.IntFunction<A> make, Fill<A> fill) {
        for (int k : CHUNKS)
            for (long start : STARTS)
                for (int n : LENGTHS) {
                    String where = name + " K=" + k + " start=" + start + " n=" + n;
                    Tandem whole = new Tandem(KEY, start, k), pieces = whole.copy();
                    A a = make.apply(n + 3), b = make.apply(n + 3);
                    fill.apply(whole, a, 3, n);
                    for (int i = 0; i < n; i += PIECE) fill.apply(pieces, b, 3 + i, Math.min(PIECE, n - i));
                    assertArrayEquals(new Object[] {b}, new Object[] {a}, where);
                    assertEquals(pieces.position(), whole.position(), where);
                }
    }

    @Test
    void longFillsEqualTheirPieces() {
        check("int", int[]::new, Tandem::fill);
        check("long", long[]::new, Tandem::fill);
        check("float", float[]::new, Tandem::fill);
        check("double", double[]::new, Tandem::fill);
        check("normal double", double[]::new, Tandem::fillGaussian);
        check("normal float", float[]::new, Tandem::fillGaussian);
        check("exponential double", double[]::new, Tandem::fillExponential);
        check("exponential float", float[]::new, Tandem::fillExponential);
        ChoiceTable t = new ChoiceTable(new double[] {1, 2, 3, 4});
        check("choice", int[]::new, (g, a, off, len) -> g.fillChoice(a, off, len, t));
    }
}
