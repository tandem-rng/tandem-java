/* Kernel entry points for TandemCuda, compiled to PTX by tools/build_ptx.sh against a pinned
 * tandem-cuda checkout. The bodies are the fill kernels of tandem.cuh. Here each entry takes
 * the key, the start position, K, n, the range where relevant and the output pointer, and
 * derives the plan the host launchers of tandem.cuh compute, so the Java side only sizes the
 * grid and picks the even or odd normal entry. The other template choices of tandem.cuh (tile
 * or rows, aligned output) become branches on values that are uniform over the grid. Every
 * thread repeats the plan, so it shifts by log2 K where the host launchers divide by K.
 *
 * Copyright 2026 Jessica Cox. Apache License 2.0, see LICENSE.
 */
#include "tandem.cuh"

namespace tandem::detail {

/* The bounded kinds of the Java CPU fills. Their fallback stream is split(g) with g the global
 * draw index, the aligned start over the width plus the element index, so a fill cut at any
 * boundary equals the whole fill. tandem.cuh at the pinned commit keys it by the element index
 * alone, which agrees for fills from position 0. */
struct below32_global {};
struct below64_global {};

/* Ctx plus the global index of the fill's first draw. */
struct CtxG : Ctx {
    uint64_t first;
};

template <> struct elem<below32_global> {
    using out_t = uint32_t;
    static constexpr unsigned bits = 32;
    __device__ static out_t make(const uint32_t w[4], unsigned i, uint64_t e, const Ctx &x) {
        return below_u32(w[i], (uint32_t)x.range, x.key, x.K, static_cast<const CtxG &>(x).first + e);
    }
};
template <> struct elem<below64_global> {
    using out_t = uint64_t;
    static constexpr unsigned bits = 64;
    __device__ static out_t make(const uint32_t w[4], unsigned i, uint64_t e, const Ctx &x) {
        return below_u64(w[2 * i] | ((uint64_t)w[2 * i + 1] << 32), x.range, x.key, x.K,
                         static_cast<const CtxG &>(x).first + e);
    }
};

} // namespace tandem::detail

namespace {

using namespace tandem;
using namespace tandem::detail;

template <class E, bool ALIGNED>
__device__ __forceinline__ void rows(const uint32_t key[4], uint32_t K, uint64_t g0, uint64_t r0,
                                     uint64_t r1, uint64_t b0, uint64_t b1, uint64_t range,
                                     uint64_t first, typename elem<E>::out_t *out) {
    uint64_t c = 8u * g0 + blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t g = c >> 3, lane = c & 7u;
    if (g > r1 / K) return;
    uint32_t o[4], h[4];
    F_keyed(key, c, DOMAIN_STREAM, AUX_STREAM, o, h);
    const CtxG x{{key, K, range}, first};
    uint64_t row = g * K;
    uint32_t j0 = row < r0 ? (uint32_t)(r0 - row) : 0u;
    uint32_t j1 = (uint32_t)(r1 - row < K - 1u ? r1 - row : K - 1u);
    for (uint32_t j = 0; j <= j1; j++) {
        T(o, h);
        if (j >= j0) store_block<E, ALIGNED>(out, b0, b1, (row + j) * 128u + lane * 16u, o, x);
    }
}

template <class E, bool ALIGNED>
__device__ __forceinline__ void tile(uint4 *tile, const uint32_t key[4], uint32_t K, uint64_t g0,
                                     uint64_t r1, uint64_t b0, uint64_t b1, uint64_t range,
                                     uint64_t first, typename elem<E>::out_t *out) {
    constexpr unsigned GROUPS = THREADS / 8, SLOTS = GROUPS * TILE_STEPS * 8;
    uint64_t gb = g0 + blockIdx.x * (uint64_t)GROUPS;
    unsigned gi = threadIdx.x >> 3, lane = threadIdx.x & 7u;
    uint64_t c = 8u * (gb + gi) + lane;
    bool mine = gb + gi <= r1 / K;
    uint32_t o[4], h[4];
    F_keyed(key, c, DOMAIN_STREAM, AUX_STREAM, o, h);
    const CtxG x{{key, K, range}, first};
    uint64_t block_first = gb * K * 128u;
    for (uint32_t jb = 0; jb < K; jb += TILE_STEPS) {
        if (block_first + jb * 128u > b1) break;
        if (mine) {
            for (unsigned j = 0; j < TILE_STEPS; j++) {
                T(o, h);
                tile[(gi * TILE_STEPS + j) * 8 + lane] = make_uint4(o[0], o[1], o[2], o[3]);
            }
        }
        __syncthreads();
        for (unsigned s = threadIdx.x; s < SLOTS; s += THREADS) {
            unsigned sg = s / (TILE_STEPS * 8), within = s % (TILE_STEPS * 8);
            uint64_t at = ((gb + sg) * K + jb) * 128u + within * 16u;
            if (at >= b1) continue;
            uint4 v = tile[s];
            uint32_t w[4] = {v.x, v.y, v.z, v.w};
            store_block<E, ALIGNED>(out, b0, b1, at, w, x);
        }
        __syncthreads();
    }
}

/* The launcher of tandem.cuh for n > 0. The tile path stages 32 KiB in dynamic shared memory,
 * which the host passes only for K >= TILE_STEPS, so the rows path keeps its occupancy. */
template <class E>
__device__ __forceinline__ void fill(uint4 *smem, uint32_t k0, uint32_t k1, uint32_t k2,
                                     uint32_t k3, uint64_t pos, uint32_t K, uint64_t n,
                                     uint64_t range, typename elem<E>::out_t *out) {
    constexpr unsigned bits = elem<E>::bits;
    const uint32_t key[4] = {k0, k1, k2, k3};
    uint64_t p0 = align_pos(pos, bits), p1 = p0 + n * bits;
    uint64_t r0 = p0 >> 10, r1 = (p1 - 1) >> 10, g0 = r0 >> (__ffs(K) - 1);
    uint64_t b0 = p0 / 8, b1 = p1 / 8;
    bool aligned = ((reinterpret_cast<uintptr_t>(out) - b0) & 15u) == 0;
    if (K >= TILE_STEPS) {
        if (aligned) tile<E, true>(smem, key, K, g0, r1, b0, b1, range, p0 / bits, out);
        else tile<E, false>(smem, key, K, g0, r1, b0, b1, range, p0 / bits, out);
    } else {
        if (aligned) rows<E, true>(key, K, g0, r0, r1, b0, b1, range, p0 / bits, out);
        else rows<E, false>(key, K, g0, r0, r1, b0, b1, range, p0 / bits, out);
    }
}

template <bool ODD>
__device__ __forceinline__ void normal64(const uint32_t key[4], uint32_t K, uint64_t g0,
                                         uint64_t ba, uint64_t bb, uint64_t n, double *out) {
    uint64_t c = 8u * g0 + blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t g = c >> 3, lane = c & 7u;
    if (g * K * 8u > bb) return;
    const bool vec = (reinterpret_cast<uintptr_t>(out) & 15u) == 0;
    uint32_t o[4], h[4], po[4] = {0, 0, 0, 0}, ph[4] = {0, 0, 0, 0};
    F_keyed(key, c, DOMAIN_STREAM, AUX_STREAM, o, h);
    if (ODD) F_keyed(key, lane ? c - 1u : 8u * g + 7u, DOMAIN_STREAM, AUX_STREAM, po, ph);
    for (uint32_t j = 0; j < K; j++) {
        uint64_t beta = (g * K + j) * 8u + lane;
        if (beta > bb) break;
        T(o, h);
        if (ODD && (lane || j)) T(po, ph);
        if (beta < ba) continue;
        uint32_t q[4];
        const uint32_t *prev = po;
        if (ODD && lane == 0 && j == 0) {
            block(key, 8u * (g - 1u) + 7u, K - 1u, q);
            prev = q;
        }
        uint64_t u = ODD ? prev[2] | ((uint64_t)prev[3] << 32) : o[0] | ((uint64_t)o[1] << 32);
        uint64_t v = ODD ? o[0] | ((uint64_t)o[1] << 32) : o[2] | ((uint64_t)o[3] << 32);
        Pair2<double> z = box_muller2(to_f64(u), to_f64(v));
        uint64_t e = 2u * (beta - ba);
        if (e + 1 < n) {
            if (vec) *reinterpret_cast<double2 *>(out + e) = make_double2(z.z0, z.z1);
            else { out[e] = z.z0; out[e + 1] = z.z1; }
        } else {
            out[e] = z.z0;
        }
    }
}

template <bool ODD>
__device__ __forceinline__ void normal32(const uint32_t key[4], uint32_t K, uint64_t g0,
                                         uint64_t s0, uint64_t n, uint64_t np, uint64_t ba,
                                         uint64_t bb, float *out) {
    uint64_t c = 8u * g0 + blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t g = c >> 3, lane = c & 7u;
    if (g * K * 8u > bb) return;
    const bool vec = (reinterpret_cast<uintptr_t>(out) & 7u) == 0;
    const bool quad = !ODD && (s0 & 3u) == 0 && (reinterpret_cast<uintptr_t>(out) & 15u) == 0;
    uint32_t o[4], h[4], po[4] = {0, 0, 0, 0}, ph[4] = {0, 0, 0, 0};
    F_keyed(key, c, DOMAIN_STREAM, AUX_STREAM, o, h);
    if (ODD) F_keyed(key, lane ? c - 1u : 8u * g + 7u, DOMAIN_STREAM, AUX_STREAM, po, ph);
    for (uint32_t j = 0; j < K; j++) {
        uint64_t beta = (g * K + j) * 8u + lane;
        if (beta > bb) break;
        T(o, h);
        if (ODD && (lane || j)) T(po, ph);
        if (beta < ba) continue;
        if (quad && 4u * (beta - ba) + 4u <= n) {
            Pair2<float> z0 = normal_step_f32(to_f32(o[0]), to_f32(o[1]));
            Pair2<float> z1 = normal_step_f32(to_f32(o[2]), to_f32(o[3]));
            *reinterpret_cast<float4 *>(out + 4u * (beta - ba)) = make_float4(z0.z0, z0.z1, z1.z0, z1.z1);
            continue;
        }
        int64_t e = ((int64_t)(4u * beta) - (ODD ? 1 : 0) - (int64_t)s0) / 2;
        uint32_t q[4];
        const uint32_t *prev = po;
        if (ODD && lane == 0 && j == 0 && g > 0) {
            block(key, 8u * (g - 1u) + 7u, K - 1u, q);
            prev = q;
        }
        for (int k = 0; k < 2; k++) {
            int64_t pj = e + k;
            if (pj < 0 || pj >= (int64_t)np) continue;
            uint32_t ua = ODD ? (k ? o[1] : prev[3]) : o[2 * k];
            uint32_t ub = ODD ? (k ? o[2] : o[0]) : o[2 * k + 1];
            Pair2<float> z = normal_step_f32(to_f32(ua), to_f32(ub));
            uint64_t at = 2u * (uint64_t)pj;
            if (at + 1 < n) {
                if (vec) *reinterpret_cast<float2 *>(out + at) = make_float2(z.z0, z.z1);
                else { out[at] = z.z0; out[at + 1] = z.z1; }
            } else {
                out[at] = z.z0;
            }
        }
    }
}

} // namespace

/* The tile path holds 32 KiB of shared memory per block, so at most 5 blocks fit an SM of an
 * A100 and registers beyond 48 would only cost occupancy. The bounded fills spill under that
 * cap and run at 70 to 80% of their speed, so they keep their registers. */
#define FILL_ENTRY(name, E, bounds)                                                                \
    extern "C" __global__ void bounds                                                             \
        name(uint32_t k0, uint32_t k1, uint32_t k2, uint32_t k3, uint64_t pos, uint32_t K,         \
             uint64_t n, uint64_t range, typename elem<E>::out_t *out) {                           \
        extern __shared__ uint4 smem[];                                                            \
        fill<E>(smem, k0, k1, k2, k3, pos, K, n, range, out);                                      \
    }

/* The plain fills take a range too, unused, so every fill entry has one signature. */
FILL_ENTRY(fill_u32, uint32_t, __launch_bounds__(THREADS, 5))
FILL_ENTRY(fill_u64, uint64_t, __launch_bounds__(THREADS, 5))
FILL_ENTRY(fill_f32, float, __launch_bounds__(THREADS, 5))
FILL_ENTRY(fill_f64, double, __launch_bounds__(THREADS, 5))
FILL_ENTRY(fill_u32_below, below32_global, __launch_bounds__(THREADS))
FILL_ENTRY(fill_u64_below, below64_global, __launch_bounds__(THREADS))

/* A start at an odd draw needs a second chunk per thread and about 20 more registers, so it
 * has its own entry, _odd, as it has its own template instance in tandem.cuh. The host picks
 * it by the parity of the first draw. */
template <bool ODD>
__device__ __forceinline__ void normal64_entry(uint32_t k0, uint32_t k1, uint32_t k2, uint32_t k3,
                                               uint64_t pos, uint32_t K, uint64_t n, double *out) {
    const uint32_t key[4] = {k0, k1, k2, k3};
    uint64_t ba = (align_pos(pos, 64) >> 7) + (ODD ? 1u : 0u), bb = ba + (n + 1u) / 2u - 1u;
    normal64<ODD>(key, K, (ba >> 3) >> (__ffs(K) - 1), ba, bb, n, out);
}

template <bool ODD>
__device__ __forceinline__ void normal32_entry(uint32_t k0, uint32_t k1, uint32_t k2, uint32_t k3,
                                               uint64_t pos, uint32_t K, uint64_t n, float *out) {
    const uint32_t key[4] = {k0, k1, k2, k3};
    uint64_t np = (n + 1u) / 2u, s0 = align_pos(pos, 32) >> 5;
    uint64_t ba = s0 >> 2, bb = (s0 + 2u * np - 1u) >> 2;
    normal32<ODD>(key, K, (ba >> 3) >> (__ffs(K) - 1), s0, n, np, ba, bb, out);
}

#define NORMAL_ENTRY(name, impl, O, odd)                                                           \
    extern "C" __global__ void __launch_bounds__(THREADS)                                         \
        name(uint32_t k0, uint32_t k1, uint32_t k2, uint32_t k3, uint64_t pos, uint32_t K,         \
             uint64_t n, O *out) {                                                                 \
        impl<odd>(k0, k1, k2, k3, pos, K, n, out);                                                 \
    }

NORMAL_ENTRY(fill_normal_f64, normal64_entry, double, false)
NORMAL_ENTRY(fill_normal_f64_odd, normal64_entry, double, true)
NORMAL_ENTRY(fill_normal_f32, normal32_entry, float, false)
NORMAL_ENTRY(fill_normal_f32_odd, normal32_entry, float, true)
