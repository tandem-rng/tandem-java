package io.github.tandemrng.cuda;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** Runs on hosts without a CUDA driver, such as CI; GPU hosts run only the gpu tag. */
class NoDriverTest {
    @Test
    void openWithoutDriverThrows() {
        CudaException e = assertThrows(CudaException.class, TandemCuda::open);
        assertTrue(e.getMessage().startsWith("no CUDA driver"), e.getMessage());
    }
}
