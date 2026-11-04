// PFA engine, native single-core version: after flipping spin i, ONE pass over all spins updates every
// delta energy and finds the next spin to flip, argmax_j(-beta * dE_j + noise_j) (Gumbel-max sampling).
// Same algorithm and same floating-point operations, in the same order, as pfa_plain.py, so results are
// bit-identical to it.
//
// Compile-time SIMD paths, chosen from the target flags (see the Makefile):
//   SVE      vector-length agnostic, predicated   ARM with SVE (Graviton3, A64FX, ...)
//   NEON     2 doubles per vector                 every other 64-bit ARM (Apple Silicon, Graviton4, ...)
//   AVX-512  8 doubles per vector                 x86-64 with AVX-512
//   AVX2     4 doubles per vector                 x86-64 with AVX2
//   scalar                                        anything else
// Every path breaks argmax ties toward the smallest index (like numpy.argmax). Build with -ffp-contract=off:
// fusing multiply-adds would change rounding and break bit-identity with pfa_plain.
//
// Test-only switches, for checking paths on hardware that lacks them:
//   -DTEST_STREAMING_SVE  runs the SVE path in SME streaming mode (e.g. Apple M4, -mcpu=apple-m4)
//   -DTEST_AVX512_SIMDE   runs the AVX-512 path through the SIMDe emulation headers (-I<simde include dir>)
#include <stdint.h>
#if defined(TEST_STREAMING_SVE)
#include <arm_sme.h>
#include <arm_sve.h>
#define USE_SVE 1
#define SVE_ATTR __arm_locally_streaming
#define SIMD_PATH "sve-streaming"
#elif defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#define USE_SVE 1
#define SVE_ATTR
#define SIMD_PATH "sve"
#elif defined(TEST_AVX512_SIMDE)
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx512.h>
#define USE_AVX512 1
#define SIMD_PATH "avx512-simde"
#elif defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define USE_NEON 1
#define SIMD_PATH "neon"
#elif defined(__AVX512F__)
#include <immintrin.h>
#define USE_AVX512 1
#define SIMD_PATH "avx512"
#elif defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2 1
#define SIMD_PATH "avx2"
#else
#define SIMD_PATH "scalar"
#endif
#if !defined(USE_SVE)
#include <string.h>
#endif
#if defined(TEST_AVX512_SIMDE)
typedef simde__mmask8 mask8_t;
#elif defined(USE_AVX512)
typedef __mmask8 mask8_t;
#endif

const char *pfa_native_simd_path(void) { return SIMD_PATH; }

#if defined(USE_SVE)
// Whole anneal in one SVE function (in streaming mode no non-streaming calls are made, so no memcpy).
SVE_ATTR void pfa_native_anneal(const double *J4, double *s, double *dE, const double *noise2, int n, long nflips,
                            double beta_max, int bi, double *E_io, double *best_io, double *sbest) {
    double E = *E_io, best = *best_io, beta = 0.0, dbeta = beta_max / (double)(nflips - 1);
    int shift = 0;
    const int64_t VL = (int64_t)svcntd(), N = n;
    const svbool_t all = svptrue_b64();
    for (long t = 0; t < nflips; t++) {
        int i = bi; const double *Ji = J4 + (long)i * n;
        E += dE[i]; double si = s[i]; dE[i] = -dE[i]; s[i] = -si;
        if (E < best - 1e-6) {
            best = E;
            for (int64_t j = 0; j < N; j += VL) { svbool_t p = svwhilelt_b64(j, N); svst1(p, sbest + j, svld1(p, s + j)); }
        }
        beta += dbeta; if (++shift == n) shift = 0;
        const double *nz = noise2 + (n - shift);
        const svfloat64_t vsi = svdup_f64(si), vnb = svdup_f64(-beta);
        svfloat64_t B0 = svdup_f64(-1e300), B1 = B0;
        svint64_t X0 = svdup_s64(0), X1 = X0;
#define LANE(P, B, X, J0) { \
            svfloat64_t D = svld1(P, dE + (J0)); \
            D = svadd_x(P, D, svmul_x(P, svmul_x(P, svld1(P, Ji + (J0)), svld1(P, s + (J0))), vsi)); \
            svst1(P, dE + (J0), D); \
            svfloat64_t V = svadd_x(P, svmul_x(P, vnb, D), svld1(P, nz + (J0))); \
            svbool_t m = svcmpgt(P, V, B); B = svsel(m, V, B); X = svsel(m, svindex_s64((J0), 1), X); }
        for (int64_t j = 0; j < N; j += 2 * VL) {          // predicates cover the tail: no scalar remainder loop
            svbool_t p0 = svwhilelt_b64(j, N), p1 = svwhilelt_b64(j + VL, N);
            LANE(p0, B0, X0, j) LANE(p1, B1, X1, j + VL)
        }
#undef LANE
        double b0 = svmaxv(all, B0), b1 = svmaxv(all, B1), bv = b0 > b1 ? b0 : b1;
        int64_t i0 = svminv(svcmpeq(all, B0, bv), X0), i1 = svminv(svcmpeq(all, B1, bv), X1);   // ties -> smallest index
        bi = (int)(i0 < i1 ? i0 : i1);
    }
    *E_io = E; *best_io = best;
}

#else
void pfa_native_anneal(const double *J4, double *s, double *dE, const double *noise2, int n, long nflips,
                   double beta_max, int bi, double *E_io, double *best_io, double *sbest) {
    double E = *E_io, best = *best_io, beta = 0.0, dbeta = beta_max / (double)(nflips - 1);
    int shift = 0;
    for (long t = 0; t < nflips; t++) {
        int i = bi; const double *Ji = J4 + (long)i * n;
        E += dE[i]; double si = s[i]; dE[i] = -dE[i]; s[i] = -si;
        if (E < best - 1e-6) { best = E; memcpy(sbest, s, sizeof(double) * n); }
        beta += dbeta; if (++shift == n) shift = 0;
        const double *nz = noise2 + (n - shift);
        double bv = -1e300; long bl = 0; int j0 = 0;
#if defined(USE_NEON)
        {   // 4 accumulators x 2 doubles = 8 spins per iteration
            const int nv = n & ~7;
            float64x2_t vsi = vdupq_n_f64(si), vnb = vdupq_n_f64(-beta);
            float64x2_t b0 = vdupq_n_f64(-1e300), b1 = b0, b2 = b0, b3 = b0;
            uint64x2_t x0 = {0, 1}, x1 = {2, 3}, x2 = {4, 5}, x3 = {6, 7};
            uint64x2_t k0 = x0, k1 = x1, k2 = x2, k3 = x3, step = vdupq_n_u64(8);
#define LANE(B, X, K, OFF) { \
                float64x2_t D = vld1q_f64(dE + j + OFF); \
                D = vaddq_f64(D, vmulq_f64(vmulq_f64(vld1q_f64(Ji + j + OFF), vld1q_f64(s + j + OFF)), vsi)); \
                vst1q_f64(dE + j + OFF, D); \
                float64x2_t V = vaddq_f64(vmulq_f64(vnb, D), vld1q_f64(nz + j + OFF)); \
                uint64x2_t m = vcgtq_f64(V, B); B = vbslq_f64(m, V, B); X = vbslq_u64(m, K, X); K = vaddq_u64(K, step); }
            for (int j = 0; j < nv; j += 8) { LANE(b0, x0, k0, 0) LANE(b1, x1, k1, 2) LANE(b2, x2, k2, 4) LANE(b3, x3, k3, 6) }
#undef LANE
            double vals[8]; uint64_t idx[8];
            vst1q_f64(vals, b0); vst1q_f64(vals + 2, b1); vst1q_f64(vals + 4, b2); vst1q_f64(vals + 6, b3);
            vst1q_u64(idx, x0); vst1q_u64(idx + 2, x1); vst1q_u64(idx + 4, x2); vst1q_u64(idx + 6, x3);
            for (int q = 0; q < 8; q++)
                if (vals[q] > bv || (vals[q] == bv && (long)idx[q] < bl)) { bv = vals[q]; bl = (long)idx[q]; }
            j0 = nv;
        }
#elif defined(USE_AVX512)
        {   // 4 accumulators x 8 doubles = 32 spins per iteration; mask registers select the running best
            const int nv = n & ~31;
            __m512d vsi = _mm512_set1_pd(si), vnb = _mm512_set1_pd(-beta);
            __m512d b0 = _mm512_set1_pd(-1e300), b1 = b0, b2 = b0, b3 = b0;
            __m512i k0 = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0), k1 = _mm512_add_epi64(k0, _mm512_set1_epi64(8));
            __m512i k2 = _mm512_add_epi64(k0, _mm512_set1_epi64(16)), k3 = _mm512_add_epi64(k0, _mm512_set1_epi64(24));
            __m512i x0 = k0, x1 = k1, x2 = k2, x3 = k3, step = _mm512_set1_epi64(32);
#define LANE(B, X, K, OFF) { \
                __m512d D = _mm512_loadu_pd(dE + j + OFF); \
                D = _mm512_add_pd(D, _mm512_mul_pd(_mm512_mul_pd(_mm512_loadu_pd(Ji + j + OFF), _mm512_loadu_pd(s + j + OFF)), vsi)); \
                _mm512_storeu_pd(dE + j + OFF, D); \
                __m512d V = _mm512_add_pd(_mm512_mul_pd(vnb, D), _mm512_loadu_pd(nz + j + OFF)); \
                mask8_t m = _mm512_cmp_pd_mask(V, B, _CMP_GT_OQ); \
                B = _mm512_mask_blend_pd(m, B, V); X = _mm512_mask_blend_epi64(m, X, K); K = _mm512_add_epi64(K, step); }
            for (int j = 0; j < nv; j += 32) { LANE(b0, x0, k0, 0) LANE(b1, x1, k1, 8) LANE(b2, x2, k2, 16) LANE(b3, x3, k3, 24) }
#undef LANE
            double vals[32]; int64_t idx[32];
            _mm512_storeu_pd(vals, b0); _mm512_storeu_pd(vals + 8, b1); _mm512_storeu_pd(vals + 16, b2); _mm512_storeu_pd(vals + 24, b3);
            _mm512_storeu_si512((void *)idx, x0); _mm512_storeu_si512((void *)(idx + 8), x1);
            _mm512_storeu_si512((void *)(idx + 16), x2); _mm512_storeu_si512((void *)(idx + 24), x3);
            for (int q = 0; q < 32; q++)
                if (vals[q] > bv || (vals[q] == bv && (long)idx[q] < bl)) { bv = vals[q]; bl = (long)idx[q]; }
            j0 = nv;
        }
#elif defined(USE_AVX2)
        {   // 4 accumulators x 4 doubles = 16 spins per iteration; indices kept as doubles (exact below 2^53)
            const int nv = n & ~15;
            __m256d vsi = _mm256_set1_pd(si), vnb = _mm256_set1_pd(-beta);
            __m256d b0 = _mm256_set1_pd(-1e300), b1 = b0, b2 = b0, b3 = b0;
            __m256d k0 = _mm256_set_pd(3, 2, 1, 0), k1 = _mm256_set_pd(7, 6, 5, 4);
            __m256d k2 = _mm256_set_pd(11, 10, 9, 8), k3 = _mm256_set_pd(15, 14, 13, 12);
            __m256d x0 = k0, x1 = k1, x2 = k2, x3 = k3, step = _mm256_set1_pd(16.0);
#define LANE(B, X, K, OFF) { \
                __m256d D = _mm256_loadu_pd(dE + j + OFF); \
                D = _mm256_add_pd(D, _mm256_mul_pd(_mm256_mul_pd(_mm256_loadu_pd(Ji + j + OFF), _mm256_loadu_pd(s + j + OFF)), vsi)); \
                _mm256_storeu_pd(dE + j + OFF, D); \
                __m256d V = _mm256_add_pd(_mm256_mul_pd(vnb, D), _mm256_loadu_pd(nz + j + OFF)); \
                __m256d m = _mm256_cmp_pd(V, B, _CMP_GT_OQ); B = _mm256_blendv_pd(B, V, m); X = _mm256_blendv_pd(X, K, m); \
                K = _mm256_add_pd(K, step); }
            for (int j = 0; j < nv; j += 16) { LANE(b0, x0, k0, 0) LANE(b1, x1, k1, 4) LANE(b2, x2, k2, 8) LANE(b3, x3, k3, 12) }
#undef LANE
            double vals[16], idx[16];
            _mm256_storeu_pd(vals, b0); _mm256_storeu_pd(vals + 4, b1); _mm256_storeu_pd(vals + 8, b2); _mm256_storeu_pd(vals + 12, b3);
            _mm256_storeu_pd(idx, x0); _mm256_storeu_pd(idx + 4, x1); _mm256_storeu_pd(idx + 8, x2); _mm256_storeu_pd(idx + 12, x3);
            for (int q = 0; q < 16; q++)
                if (vals[q] > bv || (vals[q] == bv && (long)idx[q] < bl)) { bv = vals[q]; bl = (long)idx[q]; }
            j0 = nv;
        }
#endif
        for (int j = j0; j < n; j++) {   // scalar path, or the tail after the vector loop
            double d = dE[j] + Ji[j] * s[j] * si;
            dE[j] = d;
            double v = -beta * d + nz[j];
            if (v > bv) { bv = v; bl = j; }
        }
        bi = (int)bl;
    }
    *E_io = E; *best_io = best;
}
#endif
