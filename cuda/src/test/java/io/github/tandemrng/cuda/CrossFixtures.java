package io.github.tandemrng.cuda;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The tables of tandem-cuda's tests/cross_fill_below.h and cross_fill_normal.h, and of tandem-c's
 * tests/cross_normal.h and cross_exponential.h, copied verbatim into the test resources by
 * tools/build_ptx.sh. A row is a range or a position, a count, and the values as written. For the _AT tables the row is the start position, the range and the values.
 */
final class CrossFixtures {
    record Row(long head, long count, String[] values) {}

    private static final Pattern TABLE = Pattern.compile("static const \\w+ (\\w+)\\[\\] = \\{\\n(.*?)\\n\\};", Pattern.DOTALL);
    private static final Pattern ROW = Pattern.compile("\\{(\\w+), (\\w+), \\{([^}]*)\\}\\}");
    // The _AT tables of cross_fill_below.h lead with a start position and a range, then the rejection count.
    private static final Pattern ROW_AT = Pattern.compile("\\{(\\w+), (\\w+), \\w+, \\{([^}]*)\\}\\}");

    final Map<String, List<Row>> tables = new LinkedHashMap<>();

    // tandem-c's tables name an anonymous struct, and each row ends with the end position.
    private static final Pattern C_TABLE = Pattern.compile("\\} (\\w+)\\[\\] = \\{\\n(.*?)\\n\\};", Pattern.DOTALL);
    private static final Pattern C_ROW = Pattern.compile("\\{(\\w+),\\s*\\{([^}]*)\\},\\s*(\\w+)\\}");

    private static String text(String resource) throws IOException {
        try (InputStream in = CrossFixtures.class.getResourceAsStream("/cross/" + resource)) {
            return new String(in.readAllBytes(), StandardCharsets.US_ASCII);
        }
    }

    /**
     * A table of tandem-c's tests/cross_normal.h or cross_exponential.h, fills from the key of
     * seed 42 at K = 32. A row's head is the start position and its count the end position.
     */
    static List<Row> tandemC(String resource, String table) throws IOException {
        Matcher t = C_TABLE.matcher(text(resource));
        while (t.find()) {
            if (!t.group(1).equals(table)) continue;
            List<Row> rows = new ArrayList<>();
            Matcher r = C_ROW.matcher(t.group(2));
            while (r.find()) rows.add(new Row(integer(r.group(1)), integer(r.group(3)), r.group(2).split(",\\s*")));
            return rows;
        }
        throw new IllegalStateException("no table " + table + " in " + resource);
    }

    CrossFixtures(String resource) throws IOException {
        Matcher t = TABLE.matcher(text(resource));
        while (t.find()) {
            List<Row> rows = new ArrayList<>();
            Matcher r = (t.group(1).endsWith("_AT") ? ROW_AT : ROW).matcher(t.group(2));
            while (r.find())
                rows.add(new Row(integer(r.group(1)), integer(r.group(2)), r.group(3).split(",\\s*")));
            tables.put(t.group(1), rows);
        }
    }

    /** The key line, {@code {0x...u, ...}}. */
    static int[] key() throws IOException {
        Matcher m = Pattern.compile("CROSS_FILL_KEY\\[4\\] = \\{([^}]*)\\}").matcher(text("cross_fill_below.h"));
        if (!m.find()) throw new IllegalStateException("no key in cross_fill_below.h");
        String[] w = m.group(1).split(",\\s*");
        int[] key = new int[4];
        for (int i = 0; i < 4; i++) key[i] = (int) integer(w[i]);
        return key;
    }

    /** An unsigned C literal with an optional u or ull suffix. */
    static long integer(String s) {
        s = s.replaceAll("[uUlL]+$", "");
        return s.startsWith("0x") ? Long.parseUnsignedLong(s.substring(2), 16) : Long.parseUnsignedLong(s);
    }
}
