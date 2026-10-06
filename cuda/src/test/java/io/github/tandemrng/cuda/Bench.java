package io.github.tandemrng.cuda;

import io.github.tandemrng.Tandem;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * GiB/s written by each device fill at 2^26 and 2^28 elements, and by cuRAND Philox4x32-10 with
 * the same output type: ten launches back to back and one synchronize per sample, minimum of 21
 * samples after a half-second warm-up. Then the double array fill at 2^26, which adds the
 * allocation and the copy into the Java heap. `pixi run bench`.
 */
public final class Bench {
    private static final int SAMPLES = 21, LAUNCHES = 10;

    public static void main(String[] args) {
        try (TandemCuda gpu = TandemCuda.open(); Arena arena = Arena.ofConfined();
                Curand curand = new Curand(42)) {
            Tandem g = Tandem.seed(42);
            MemorySegment d = gpu.allocate(arena, 8L << 28);
            System.out.println("cuRAND " + curand.version);
            System.out.println("| fill | 2^26 GiB/s | 2^28 GiB/s | cuRAND 2^26 GiB/s | cuRAND 2^28 GiB/s | cuRAND call |");
            System.out.println("|---|---|---|---|---|---|");
            for (TandemCuda.Kind kind : TandemCuda.Kind.values()) {
                long range = kind == TandemCuda.Kind.U32_BELOW || kind == TandemCuda.Kind.U64_BELOW ? 1000 : 0;
                String label = kind == TandemCuda.Kind.NORMAL_F64 ? "fill_normal_f64 (tandem.cuh)" : kind.entry;
                StringBuilder row = new StringBuilder("| " + label + " |");
                for (int log = 26; log <= 28; log += 2) {
                    long n = 1L << log;
                    row.append(String.format(" %.0f |", rate(gpu, n * kind.bytes,
                            () -> gpu.launch(kind, d.address(), n, range, g))));
                }
                for (int log = 26; log <= 28; log += 2) {
                    long n = 1L << log;
                    row.append(String.format(" %.0f |", rate(gpu, n * kind.bytes,
                            () -> curand.fill(kind, d.address(), n))));
                }
                System.out.println(row.append(" `").append(Curand.call(kind)).append("` |"));
            }
            double[] host = new double[1 << 26];
            long best = Long.MAX_VALUE;
            for (int s = 0; s < 5 + SAMPLES; s++) {
                long t = System.nanoTime();
                gpu.fill(host, g);
                if (s >= 5) best = Math.min(best, System.nanoTime() - t);
            }
            System.out.printf("fill(double[2^26]) with the copy to the heap: %.1f GiB/s%n",
                    8.0 * host.length / best * 1e9 / (1L << 30));
        }
    }

    private static double rate(TandemCuda gpu, long bytes, Runnable launch) {
        long end = System.nanoTime() + 500_000_000L;
        while (System.nanoTime() < end) {
            launch.run();
            gpu.synchronize();
        }
        long best = Long.MAX_VALUE;
        for (int s = 0; s < SAMPLES; s++) {
            long t = System.nanoTime();
            for (int i = 0; i < LAUNCHES; i++) launch.run();
            gpu.synchronize();
            best = Math.min(best, System.nanoTime() - t);
        }
        return (double) bytes * LAUNCHES / best * 1e9 / (1L << 30);
    }
}
