package io.github.tandemrng;

import java.util.Objects;
import org.apache.commons.rng.UniformRandomProvider;

/**
 * Adapts a {@link Tandem} to Apache Commons RNG. The dependency
 * {@code commons-rng-client-api} is optional: add it to your own build to use this class.
 *
 * <p>The adaptor draws from the wrapped generator, so its position moves with every call. It is
 * not thread-safe. Bounded integers use Lemire's method as {@link Tandem#nextInt(int)} does.
 */
public final class TandemProvider implements UniformRandomProvider {
    private final Tandem tandem;

    /** Wraps {@code tandem}, which the adaptor then shares. */
    public TandemProvider(Tandem tandem) {
        this.tandem = Objects.requireNonNull(tandem);
    }

    /** Returns the wrapped generator. */
    public Tandem tandem() {
        return tandem;
    }

    @Override
    public void nextBytes(byte[] bytes) {
        tandem.fill(bytes);
    }

    @Override
    public void nextBytes(byte[] bytes, int start, int len) {
        tandem.fill(bytes, start, len);
    }

    @Override
    public int nextInt() {
        return tandem.nextInt();
    }

    @Override
    public int nextInt(int n) {
        return tandem.nextInt(n);
    }

    @Override
    public int nextInt(int origin, int bound) {
        return tandem.nextInt(origin, bound);
    }

    @Override
    public long nextLong() {
        return tandem.nextLong();
    }

    @Override
    public long nextLong(long n) {
        return tandem.nextLong(n);
    }

    @Override
    public long nextLong(long origin, long bound) {
        return tandem.nextLong(origin, bound);
    }

    @Override
    public boolean nextBoolean() {
        return tandem.nextBoolean();
    }

    @Override
    public float nextFloat() {
        return tandem.nextFloat();
    }

    @Override
    public double nextDouble() {
        return tandem.nextDouble();
    }

    @Override
    public String toString() {
        return "TandemProvider[" + tandem + "]";
    }
}
