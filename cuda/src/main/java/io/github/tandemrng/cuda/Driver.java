package io.github.tandemrng.cuda;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.util.Locale;

/** The CUDA driver API calls TandemCuda needs, bound through the native linker. */
final class Driver {
    private static Driver loaded;

    private final MethodHandle cuInit, cuDeviceGet, cuDeviceGetAttribute, cuDevicePrimaryCtxRetain,
            cuDevicePrimaryCtxRelease, cuCtxSetCurrent, cuCtxSynchronize, cuModuleLoadData,
            cuModuleUnload, cuModuleGetFunction, cuMemAlloc, cuMemFree, cuMemcpyDtoH,
            cuLaunchKernel, cuGetErrorString;

    /** Loads the driver once; a failed load is retried by the next call. */
    static synchronized Driver get() {
        if (loaded == null) loaded = new Driver();
        return loaded;
    }

    private Driver() {
        String os = System.getProperty("os.name").toLowerCase(Locale.ROOT);
        String name = os.startsWith("windows") ? "nvcuda.dll" : "libcuda.so.1";
        SymbolLookup lib;
        try {
            lib = SymbolLookup.libraryLookup(name, Arena.global());
        } catch (IllegalArgumentException e) {
            throw new CudaException("no CUDA driver: cannot load " + name + " on " + System.getProperty("os.name"));
        }
        Binder b = new Binder(lib);
        cuInit = b.bind("cuInit", JAVA_INT);
        cuDeviceGet = b.bind("cuDeviceGet", ADDRESS, JAVA_INT);
        cuDeviceGetAttribute = b.bind("cuDeviceGetAttribute", ADDRESS, JAVA_INT, JAVA_INT);
        cuDevicePrimaryCtxRetain = b.bind("cuDevicePrimaryCtxRetain", ADDRESS, JAVA_INT);
        cuDevicePrimaryCtxRelease = b.bind("cuDevicePrimaryCtxRelease_v2", JAVA_INT);
        cuCtxSetCurrent = b.bind("cuCtxSetCurrent", ADDRESS);
        cuCtxSynchronize = b.bind("cuCtxSynchronize");
        cuModuleLoadData = b.bind("cuModuleLoadData", ADDRESS, ADDRESS);
        cuModuleUnload = b.bind("cuModuleUnload", ADDRESS);
        cuModuleGetFunction = b.bind("cuModuleGetFunction", ADDRESS, ADDRESS, ADDRESS);
        cuMemAlloc = b.bind("cuMemAlloc_v2", ADDRESS, JAVA_LONG);
        cuMemFree = b.bind("cuMemFree_v2", JAVA_LONG);
        // Critical, so the copy may write straight into a Java array.
        cuMemcpyDtoH = Linker.nativeLinker().downcallHandle(
                b.find("cuMemcpyDtoH_v2"),
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_LONG),
                Linker.Option.critical(true));
        cuLaunchKernel = b.bind("cuLaunchKernel", ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT,
                JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS, ADDRESS);
        cuGetErrorString = b.bind("cuGetErrorString", JAVA_INT, ADDRESS);
    }

    private record Binder(SymbolLookup lib) {
        MemorySegment find(String name) {
            return lib.find(name).orElseThrow(() -> new CudaException("CUDA driver lacks " + name));
        }

        MethodHandle bind(String name, java.lang.foreign.MemoryLayout... args) {
            return Linker.nativeLinker().downcallHandle(find(name), FunctionDescriptor.of(JAVA_INT, args));
        }
    }

    private void check(int code, String call) {
        if (code == 0) return;
        String text = "unknown error";
        try (Arena a = Arena.ofConfined()) {
            MemorySegment p = a.allocate(ADDRESS);
            if ((int) cuGetErrorString.invokeExact(code, p) == 0)
                text = p.get(ADDRESS, 0).reinterpret(Long.MAX_VALUE).getString(0);
        } catch (Throwable t) {
            // The code alone still names the failure.
        }
        throw new CudaException(call + " failed with CUresult " + code + ": " + text);
    }

    private static RuntimeException rethrow(Throwable t) {
        if (t instanceof RuntimeException r) return r;
        if (t instanceof Error e) throw e;
        return new IllegalStateException(t);
    }

    void init() {
        try {
            check((int) cuInit.invokeExact(0), "cuInit");
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    int device(int ordinal) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment p = a.allocate(JAVA_INT);
            check((int) cuDeviceGet.invokeExact(p, ordinal), "cuDeviceGet");
            return p.get(JAVA_INT, 0);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    int attribute(int attribute, int device) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment p = a.allocate(JAVA_INT);
            check((int) cuDeviceGetAttribute.invokeExact(p, attribute, device), "cuDeviceGetAttribute");
            return p.get(JAVA_INT, 0);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    MemorySegment retainPrimaryContext(int device) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment p = a.allocate(ADDRESS);
            check((int) cuDevicePrimaryCtxRetain.invokeExact(p, device), "cuDevicePrimaryCtxRetain");
            return p.get(ADDRESS, 0);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    void releasePrimaryContext(int device) {
        try {
            check((int) cuDevicePrimaryCtxRelease.invokeExact(device), "cuDevicePrimaryCtxRelease");
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    void setCurrent(MemorySegment context) {
        try {
            check((int) cuCtxSetCurrent.invokeExact(context), "cuCtxSetCurrent");
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    void synchronize() {
        try {
            check((int) cuCtxSynchronize.invokeExact(), "cuCtxSynchronize");
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Loads a module from a NUL-terminated PTX image. */
    MemorySegment loadModule(MemorySegment image) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment p = a.allocate(ADDRESS);
            check((int) cuModuleLoadData.invokeExact(p, image), "cuModuleLoadData");
            return p.get(ADDRESS, 0);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    void unloadModule(MemorySegment module) {
        try {
            check((int) cuModuleUnload.invokeExact(module), "cuModuleUnload");
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    MemorySegment function(MemorySegment module, String name) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment p = a.allocate(ADDRESS);
            check((int) cuModuleGetFunction.invokeExact(p, module, a.allocateFrom(name)), "cuModuleGetFunction " + name);
            return p.get(ADDRESS, 0);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    long alloc(long bytes) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment p = a.allocate(JAVA_LONG);
            check((int) cuMemAlloc.invokeExact(p, bytes), "cuMemAlloc");
            return p.get(JAVA_LONG, 0);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    void free(long devicePointer) {
        try {
            check((int) cuMemFree.invokeExact(devicePointer), "cuMemFree");
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Copies to host memory, a native or heap segment. Waits for earlier work in the context. */
    void copyToHost(MemorySegment host, long devicePointer, long bytes) {
        try {
            check((int) cuMemcpyDtoH.invokeExact(host, devicePointer, bytes), "cuMemcpyDtoH");
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    void launch(MemorySegment function, int blocks, int threads, int sharedBytes, MemorySegment params) {
        try {
            check((int) cuLaunchKernel.invokeExact(function, blocks, 1, 1, threads, 1, 1, sharedBytes,
                    MemorySegment.NULL, params, MemorySegment.NULL), "cuLaunchKernel");
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }
}
