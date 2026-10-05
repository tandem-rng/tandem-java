import io.github.tandemrng.Tandem;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * Single-thread speed of Tandem against two JDK generators.
 *
 * <pre>pixi run bench</pre>
 *
 * Each generator runs in its own JVM, so no call site sees more than one generator class and
 * the JIT inlines every draw as it would in an application. Times are the minimum of seven runs
 * after three warm-up runs: nanoseconds per draw for scalar draws over 2^25 draws, GiB/s for
 * arrays of 2^24 elements.
 */
public final class Bench {
    private static final int DRAWS = 1 << 25;
    private static final int ARRAY = 1 << 24;
    private static final int WARMUP = 3, RUNS = 7;
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

    private static RandomGenerator create(String name) {
        return switch (name) {
            case "Tandem" -> Tandem.seed(42);
            case "L64X128MixRandom" -> RandomGenerator.of("L64X128MixRandom");
            case "SplittableRandom" -> new SplittableRandom(42);
            case "Random" -> new java.util.Random(42);
            default -> throw new IllegalArgumentException(name);
        };
    }

    private static void child(String name) {
        RandomGenerator r = create(name);
        int[] ints = new int[ARRAY];
        double[] doubles = new double[ARRAY];
        double nextInt = best(() -> {
            int s = 0;
            for (int i = 0; i < DRAWS; i++) s += r.nextInt();
            sink = s;
        }) / DRAWS;
        double nextLong = best(() -> {
            long s = 0;
            for (int i = 0; i < DRAWS; i++) s += r.nextLong();
            sink = s;
        }) / DRAWS;
        double nextDouble = best(() -> {
            double s = 0;
            for (int i = 0; i < DRAWS; i++) s += r.nextDouble();
            sink = (long) s;
        }) / DRAWS;
        double gauss = best(() -> {
            double s = 0;
            for (int i = 0; i < DRAWS; i++) s += r.nextGaussian();
            sink = (long) s;
        }) / DRAWS;
        double expo = best(() -> {
            double s = 0;
            for (int i = 0; i < DRAWS; i++) s += r.nextExponential();
            sink = (long) s;
        }) / DRAWS;
        double intLoop = best(() -> {
            for (int i = 0; i < ARRAY; i++) ints[i] = r.nextInt();
            sink = ints[ARRAY - 1];
        });
        double doubleLoop = best(() -> {
            for (int i = 0; i < ARRAY; i++) doubles[i] = r.nextDouble();
            sink = (long) doubles[ARRAY - 1];
        });
        String intFill = "-", doubleFill = "-", expFill = "-", expFillF = "-", gaussFill = "-", gaussFillF = "-";
        if (r instanceof Tandem t) {
            intFill = gib(best(() -> {
                t.fill(ints);
                sink = ints[ARRAY - 1];
            }), 4);
            doubleFill = gib(best(() -> {
                t.fill(doubles);
                sink = (long) doubles[ARRAY - 1];
            }), 8);
            expFill = gib(best(() -> {
                t.fillExponential(doubles);
                sink = (long) doubles[ARRAY - 1];
            }), 8);
            float[] floats = new float[ARRAY];
            expFillF = gib(best(() -> {
                t.fillExponential(floats);
                sink = (long) floats[ARRAY - 1];
            }), 4);
            gaussFill = gib(best(() -> {
                t.fillGaussian(doubles);
                sink = (long) doubles[ARRAY - 1];
            }), 8);
            gaussFillF = gib(best(() -> {
                t.fillGaussian(floats);
                sink = (long) floats[ARRAY - 1];
            }), 4);
        }
        System.out.printf(
                "| %s | %.2f | %.2f | %.2f | %.2f | %.2f | %s | %s | %s | %s | %s | %s | %s | %s |%n",
                name, nextInt, nextLong, nextDouble, gauss, expo, gib(intLoop, 4), gib(doubleLoop, 8), intFill, doubleFill, expFill, expFillF,
                gaussFill, gaussFillF);
    }

    private static String gib(double ns, int bytes) {
        return String.format("%.2f", (double) ARRAY * bytes / ns * 1e9 / (1L << 30));
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 1) {
            child(args[0]);
            return;
        }
        System.out.printf("JDK %s, %s %s%n%n", Runtime.version(), System.getProperty("os.name"), System.getProperty("os.arch"));
        System.out.println("| generator | nextInt ns | nextLong ns | nextDouble ns | nextGaussian ns | nextExponential ns | int[] loop GiB/s | double[] loop GiB/s | int[] fill GiB/s | double[] fill GiB/s | exponential double[] fill GiB/s | exponential float[] fill GiB/s | normal double[] fill GiB/s | normal float[] fill GiB/s |");
        System.out.println("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
        String java = System.getProperty("java.home") + "/bin/java";
        for (String name : new String[] {"Tandem", "L64X128MixRandom", "SplittableRandom", "Random"}) {
            Process p = new ProcessBuilder(java, "-Xmx2g", "-cp", System.getProperty("java.class.path"), "tools/Bench.java", name)
                    .inheritIO()
                    .start();
            if (p.waitFor() != 0) throw new IllegalStateException(name + " failed");
        }
    }
}
