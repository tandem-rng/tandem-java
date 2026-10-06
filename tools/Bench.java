import io.github.tandemrng.Tandem;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * Single-thread speed of Tandem against the JDK generators.
 *
 * <pre>pixi run bench</pre>
 *
 * Each generator runs in its own JVM, so no call site sees more than one generator class and
 * the JIT inlines every draw as it would in an application. Times are the minimum of seven runs
 * after three warm-up runs, in GiB/s of output: 2^25 scalar draws, counted at 4 bytes for an int
 * and 8 for the others, and arrays of 2^24 elements. A JDK generator fills an array by a loop of
 * its scalar draws, the baseline of each Tandem fill.
 */
public final class Bench {
    private static final int DRAWS = 1 << 25;
    private static final int ARRAY = 1 << 24;
    private static final int WARMUP = 3, RUNS = 7;
    private static final String BASELINE = "SplittableRandom";
    private static final String[] SCALARS = {
        "nextInt", "nextLong", "nextDouble", "nextGaussian", "nextExponential", "int[] loop", "double[] loop"
    };
    private static final String[] FILLS = {
        "fill(int[])", "fill(long[])", "fill(float[])", "fill(double[])",
        "fillGaussian(double[])", "fillGaussian(float[])", "fillExponential(double[])", "fillExponential(float[])"
    };
    private static volatile long sink;

    private interface Run {
        void run();
    }

    private static double best(Run r) {
        for (int i = 0; i < WARMUP; i++) r.run();
        long min = Long.MAX_VALUE;
        for (int i = 0; i < RUNS; i++) {
            long t = System.nanoTime();
            r.run();
            min = Math.min(min, System.nanoTime() - t);
        }
        return min;
    }

    /** GiB/s of output: count values of the given bytes in ns nanoseconds. */
    private static String gib(double ns, long count, int bytes) {
        return String.format("%.2f", (double) count * bytes / ns * 1e9 / (1L << 30));
    }

    private static RandomGenerator create(String name) {
        return switch (name) {
            case "Tandem", "Tandem libtandem" -> Tandem.seed(42);
            case "L64X128MixRandom" -> RandomGenerator.of("L64X128MixRandom");
            case "SplittableRandom" -> new SplittableRandom(42);
            case "Random" -> new java.util.Random(42);
            default -> throw new IllegalArgumentException(name);
        };
    }

    private static void put(String key, double ns, long count, int bytes) {
        System.out.println(key + "\t" + gib(ns, count, bytes));
    }

    private static void scalars(RandomGenerator r) {
        int[] ints = new int[ARRAY];
        double[] doubles = new double[ARRAY];
        put("nextInt", best(() -> {
            int s = 0;
            for (int i = 0; i < DRAWS; i++) s += r.nextInt();
            sink = s;
        }), DRAWS, 4);
        put("nextLong", best(() -> {
            long s = 0;
            for (int i = 0; i < DRAWS; i++) s += r.nextLong();
            sink = s;
        }), DRAWS, 8);
        put("nextDouble", best(() -> {
            double s = 0;
            for (int i = 0; i < DRAWS; i++) s += r.nextDouble();
            sink = (long) s;
        }), DRAWS, 8);
        put("nextGaussian", best(() -> {
            double s = 0;
            for (int i = 0; i < DRAWS; i++) s += r.nextGaussian();
            sink = (long) s;
        }), DRAWS, 8);
        put("nextExponential", best(() -> {
            double s = 0;
            for (int i = 0; i < DRAWS; i++) s += r.nextExponential();
            sink = (long) s;
        }), DRAWS, 8);
        put("int[] loop", best(() -> {
            for (int i = 0; i < ARRAY; i++) ints[i] = r.nextInt();
            sink = ints[ARRAY - 1];
        }), ARRAY, 4);
        put("double[] loop", best(() -> {
            for (int i = 0; i < ARRAY; i++) doubles[i] = r.nextDouble();
            sink = (long) doubles[ARRAY - 1];
        }), ARRAY, 8);
    }

    private static void tandemFills(Tandem t) {
        int[] ints = new int[ARRAY];
        long[] longs = new long[ARRAY];
        float[] floats = new float[ARRAY];
        double[] doubles = new double[ARRAY];
        put(FILLS[0], best(() -> {
            t.fill(ints);
            sink = ints[ARRAY - 1];
        }), ARRAY, 4);
        put(FILLS[1], best(() -> {
            t.fill(longs);
            sink = longs[ARRAY - 1];
        }), ARRAY, 8);
        put(FILLS[2], best(() -> {
            t.fill(floats);
            sink = (long) floats[ARRAY - 1];
        }), ARRAY, 4);
        put(FILLS[3], best(() -> {
            t.fill(doubles);
            sink = (long) doubles[ARRAY - 1];
        }), ARRAY, 8);
        put(FILLS[4], best(() -> {
            t.fillGaussian(doubles);
            sink = (long) doubles[ARRAY - 1];
        }), ARRAY, 8);
        put(FILLS[5], best(() -> {
            t.fillGaussian(floats);
            sink = (long) floats[ARRAY - 1];
        }), ARRAY, 4);
        put(FILLS[6], best(() -> {
            t.fillExponential(doubles);
            sink = (long) doubles[ARRAY - 1];
        }), ARRAY, 8);
        put(FILLS[7], best(() -> {
            t.fillExponential(floats);
            sink = (long) floats[ARRAY - 1];
        }), ARRAY, 4);
    }

    /** The baseline of each fill: a loop of the generator's scalar draws into the array. */
    private static void loopFills(RandomGenerator r) {
        int[] ints = new int[ARRAY];
        long[] longs = new long[ARRAY];
        float[] floats = new float[ARRAY];
        double[] doubles = new double[ARRAY];
        put(FILLS[0], best(() -> {
            for (int i = 0; i < ARRAY; i++) ints[i] = r.nextInt();
            sink = ints[ARRAY - 1];
        }), ARRAY, 4);
        put(FILLS[1], best(() -> {
            for (int i = 0; i < ARRAY; i++) longs[i] = r.nextLong();
            sink = longs[ARRAY - 1];
        }), ARRAY, 8);
        put(FILLS[2], best(() -> {
            for (int i = 0; i < ARRAY; i++) floats[i] = r.nextFloat();
            sink = (long) floats[ARRAY - 1];
        }), ARRAY, 4);
        put(FILLS[3], best(() -> {
            for (int i = 0; i < ARRAY; i++) doubles[i] = r.nextDouble();
            sink = (long) doubles[ARRAY - 1];
        }), ARRAY, 8);
        put(FILLS[4], best(() -> {
            for (int i = 0; i < ARRAY; i++) doubles[i] = r.nextGaussian();
            sink = (long) doubles[ARRAY - 1];
        }), ARRAY, 8);
        put(FILLS[5], best(() -> {
            for (int i = 0; i < ARRAY; i++) floats[i] = (float) r.nextGaussian();
            sink = (long) floats[ARRAY - 1];
        }), ARRAY, 4);
        put(FILLS[6], best(() -> {
            for (int i = 0; i < ARRAY; i++) doubles[i] = r.nextExponential();
            sink = (long) doubles[ARRAY - 1];
        }), ARRAY, 8);
        put(FILLS[7], best(() -> {
            for (int i = 0; i < ARRAY; i++) floats[i] = (float) r.nextExponential();
            sink = (long) floats[ARRAY - 1];
        }), ARRAY, 4);
    }

    private static void child(String name) {
        RandomGenerator r = create(name);
        if (name.equals("Tandem libtandem")) {
            tandemFills((Tandem) r);
            return;
        }
        scalars(r);
        if (r instanceof Tandem t) tandemFills(t);
        else if (name.equals(BASELINE)) loopFills(r);
    }

    /** Runs one generator in its own JVM and returns its figures by key. */
    private static Map<String, String> run(String name, String lib) throws Exception {
        List<String> cmd = new ArrayList<>(List.of(System.getProperty("java.home") + "/bin/java", "-Xmx2g"));
        if (name.equals("Tandem libtandem")) cmd.addAll(List.of("--enable-native-access=ALL-UNNAMED", "-Dtandem.native=" + lib));
        cmd.addAll(List.of("-cp", System.getProperty("java.class.path"), "tools/Bench.java", name));
        Process p = new ProcessBuilder(cmd).redirectError(ProcessBuilder.Redirect.INHERIT).start();
        Map<String, String> out = new LinkedHashMap<>();
        try (BufferedReader in = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
            for (String line; (line = in.readLine()) != null; ) {
                String[] kv = line.split("\t");
                out.put(kv[0], kv[1]);
            }
        }
        if (p.waitFor() != 0) throw new IllegalStateException(name + " failed");
        return out;
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 1) {
            child(args[0]);
            return;
        }
        String lib = System.getProperty("tandem.native");
        Map<String, Map<String, String>> res = new LinkedHashMap<>();
        for (String name : List.of("Tandem", "L64X128MixRandom", "SplittableRandom", "Random")) res.put(name, run(name, lib));
        if (lib != null) res.put("Tandem libtandem", run("Tandem libtandem", lib));

        System.out.printf("JDK %s, %s %s%n%n", Runtime.version(), System.getProperty("os.name"), System.getProperty("os.arch"));
        StringBuilder head = new StringBuilder("| generator |"), rule = new StringBuilder("|---|");
        for (String s : SCALARS) {
            head.append(' ').append(s).append(" GiB/s |");
            rule.append("---|");
        }
        System.out.println(head);
        System.out.println(rule);
        for (String name : List.of("Tandem", "L64X128MixRandom", "SplittableRandom", "Random")) {
            StringBuilder row = new StringBuilder("| " + name + " |");
            for (String s : SCALARS) row.append(' ').append(res.get(name).get(s)).append(" |");
            System.out.println(row);
        }
        System.out.println();
        System.out.println("| fill | Java GiB/s | libtandem GiB/s | " + BASELINE + " loop GiB/s |");
        System.out.println("|---|---|---|---|");
        for (String f : FILLS) {
            String nat = lib == null ? "-" : res.get("Tandem libtandem").get(f);
            System.out.printf("| `%s` | %s | %s | %s |%n", f, res.get("Tandem").get(f), nat, res.get(BASELINE).get(f));
        }
    }
}
