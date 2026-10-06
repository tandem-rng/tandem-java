package io.github.tandemrng;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The spec's conformance files in {@code src/test/resources/conformance}, byte-identical copies
 * of tandem-spec's {@code conformance/*.json}, and readers for their fields. The JSON reader
 * takes what those files hold: objects, arrays, strings without escapes, numbers.
 */
final class Conformance {
    private Conformance() {}

    /** One case of a {@code cases} list. */
    record Case(Map<String, Object> f) {
        String id() {
            return (String) f.get("id");
        }

        String kind() {
            return (String) f.get("kind");
        }

        long start() {
            return num("start");
        }

        int n() {
            return (int) num("n");
        }

        long num(String k) {
            return ((Number) f.get(k)).longValue();
        }

        boolean has(String k) {
            return f.containsKey(k);
        }

        /** A new generator at the case's start. */
        Tandem generator() {
            return Conformance.generator(f.get("key"), (int) num("K"), start());
        }

        /** The hexadecimal string of field k as an unsigned 64-bit value. */
        long word(String k) {
            return Long.parseUnsignedLong((String) f.get(k), 16);
        }

        /** The hexadecimal strings of field k as unsigned 64-bit values. */
        long[] hex(String k) {
            return Conformance.hex(f.get(k));
        }

        /** The end position: the file's {@code end} when it pins one, else {@code align(start, w) + w * n}. */
        long end(int w) {
            return has("end") ? num("end") : align(start(), w) + (long) w * n();
        }
    }

    static Map<String, Object> load(String file) {
        try (InputStream in = Conformance.class.getResourceAsStream("/conformance/" + file)) {
            return obj(new Reader(new String(in.readAllBytes(), StandardCharsets.UTF_8)).value());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    static List<Case> cases(String file) {
        return list(load(file).get("cases")).stream().map(c -> new Case(obj(c))).toList();
    }

    /** The case whose id ends with {@code suffix}. */
    static Case find(String file, String suffix) {
        return cases(file).stream().filter(c -> c.id().endsWith(" " + suffix)).findFirst().orElseThrow();
    }

    static Tandem generator(Object key, int k, long start) {
        long[] w = hex(key);
        return new Tandem(new int[] {(int) w[0], (int) w[1], (int) w[2], (int) w[3]}, start, k);
    }

    static long[] hex(Object strings) {
        return list(strings).stream().mapToLong(s -> Long.parseUnsignedLong((String) s, 16)).toArray();
    }

    static long align(long p, int w) {
        return (p + w - 1) / w * w;
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> obj(Object o) {
        return (Map<String, Object>) o;
    }

    @SuppressWarnings("unchecked")
    static List<Object> list(Object o) {
        return (List<Object>) o;
    }

    private static final class Reader {
        private final String s;
        private int i;

        Reader(String s) {
            this.s = s;
        }

        private char peek() {
            while (Character.isWhitespace(s.charAt(i))) i++;
            return s.charAt(i);
        }

        private void expect(char c) {
            if (peek() != c) throw new IllegalArgumentException("expected " + c + " at " + i);
            i++;
        }

        /** True and past c when c is next, for the separators of a list. */
        private boolean take(char c) {
            if (peek() != c) return false;
            i++;
            return true;
        }

        Object value() {
            char c = peek();
            if (c == '{') {
                i++;
                Map<String, Object> m = new LinkedHashMap<>();
                if (take('}')) return m;
                do {
                    String k = string();
                    expect(':');
                    m.put(k, value());
                } while (take(','));
                expect('}');
                return m;
            }
            if (c == '[') {
                i++;
                List<Object> a = new ArrayList<>();
                if (take(']')) return a;
                do a.add(value());
                while (take(','));
                expect(']');
                return a;
            }
            if (c == '"') return string();
            int j = i;
            while (i < s.length() && "+-.eE0123456789truefalsn".indexOf(s.charAt(i)) >= 0) i++;
            String t = s.substring(j, i);
            return switch (t) {
                case "true" -> true;
                case "false" -> false;
                case "null" -> null;
                default -> t.matches("-?\\d+") ? (Object) Long.parseLong(t) : (Object) Double.parseDouble(t);
            };
        }

        private String string() {
            expect('"');
            int j = s.indexOf('"', i);
            String t = s.substring(i, j);
            if (t.indexOf('\\') >= 0) throw new IllegalArgumentException("escapes are not read, at " + i);
            i = j + 1;
            return t;
        }
    }
}
