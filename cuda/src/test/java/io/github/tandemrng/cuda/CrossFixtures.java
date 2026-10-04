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
 * The tables of tandem-cuda's tests/cross_fill_below.h and cross_fill_normal.h, copied verbatim
 * into the test resources by tools/build_ptx.sh. A row is a range or a position, a count, and
 * the values as written. For the _AT tables the row is the start position, the range and the values.
 */
final class CrossFixtures {
    record Row(long head, long count, String[] values) {}

    private static final Pattern TABLE = Pattern.compile("static const \\w+ (\\w+)\\[\\] = \\{\\n(.*?)\\n\\};", Pattern.DOTALL);
    private static final Pattern ROW = Pattern.compile("\\{(\\w+), (\\w+), \\{([^}]*)\\}\\}");
    // The _AT tables of cross_fill_below.h lead with a start position and a range, then the rejection count.
    private static final Pattern ROW_AT = Pattern.compile("\\{(\\w+), (\\w+), \\w+, \\{([^}]*)\\}\\}");

    final Map<String, List<Row>> tables = new LinkedHashMap<>();

    CrossFixtures(String resource) throws IOException {
        String text;
        try (InputStream in = CrossFixtures.class.getResourceAsStream("/cross/" + resource)) {
            text = new String(in.readAllBytes(), StandardCharsets.US_ASCII);
        }
        Matcher t = TABLE.matcher(text);
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
        String text;
        try (InputStream in = CrossFixtures.class.getResourceAsStream("/cross/cross_fill_below.h")) {
            text = new String(in.readAllBytes(), StandardCharsets.US_ASCII);
        }
        Matcher m = Pattern.compile("CROSS_FILL_KEY\\[4\\] = \\{([^}]*)\\}").matcher(text);
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
