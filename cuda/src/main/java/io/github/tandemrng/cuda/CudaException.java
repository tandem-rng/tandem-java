package io.github.tandemrng.cuda;

/** A CUDA driver call failed, or no CUDA driver could be loaded. */
public final class CudaException extends RuntimeException {
    private static final long serialVersionUID = 1L;

    CudaException(String message) {
        super(message);
    }
}
