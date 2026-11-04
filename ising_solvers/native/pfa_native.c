// PFA engine, native single-core version: after flipping spin i, ONE pass over all spins updates every
// delta energy and finds the next spin to flip, argmax_j(-beta * dE_j + noise_j) (Gumbel-max sampling).
// Same algorithm and same floating-point results as pfa_plain.py, so results are bit-identical to it.
//
// Two exact shortcuts keep the inner loop at 5 vector arithmetic ops per vector of spins:
//   - the update dE_j + (J_ij * s_j) * s_i equals dE_j + J_ij * s_j for s_i = +1 and dE_j - J_ij * s_j for
//     s_i = -1, bit for bit, so the sign is chosen once per flip instead of multiplied in per spin;
//   - the loop keeps only a running maximum of the scores per chunk of spins; afterwards the winning chunk
//     is rescored (same arithmetic, same values) to find the first spin that reaches the maximum.
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

const char *pfa_native_simd_path(void) { return SIMD_PATH; }

#if defined(USE_SVE)
// Whole anneal in one SVE function (in streaming mode no non-streaming calls are made, so no memcpy).
SVE_ATTR void pfa_native_anneal(const double *J4, double *s, double *dE, const double *noise2, int n, long nflips,
                            double beta_max, int bi, double *E_io, double *best_io, double *sbest) {
    double E = *E_io, best = *best_io, beta = 0.0, dbeta = beta_max / (double)(nflips - 1);
    int shift = 0;
    const int64_t VL = (int64_t)svcntd(), N = n, CH = 8 * VL;
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
        const svfloat64_t vnb = svdup_f64(-beta);
        svfloat64_t G = svdup_f64(-1e300);                 // per-lane best over chunks, and the chunk it came from
        svint64_t X = svdup_s64(0);
#define LANE(P, B, OP, J0) { \
            svfloat64_t D = svld1(P, dE + (J0)); \
            D = OP(P, D, svmul_x(P, svld1(P, Ji + (J0)), svld1(P, s + (J0)))); \
            svst1(P, dE + (J0), D); \
            B = svmax_m(P, B, svadd_x(P, svmul_x(P, vnb, D), svld1(P, nz + (J0)))); }
#define PASS(OP) \
        for (int64_t j = 0; j < N; j += CH) {               /* predicates cover the tail: no scalar remainder loop */ \
            svfloat64_t B0 = svdup_f64(-1e300), B1 = B0; \
            for (int64_t k = j; k < j + CH && k < N; k += 2 * VL) { \
                svbool_t p0 = svwhilelt_b64(k, N), p1 = svwhilelt_b64(k + VL, N); \
                LANE(p0, B0, OP, k) LANE(p1, B1, OP, k + VL) \
            } \
            svfloat64_t B = svmax_x(all, B0, B1); \
            svbool_t m = svcmpgt(all, B, G); G = svsel(m, B, G); X = svsel(m, svdup_s64(j), X); \
        }
        if (si < 0) { PASS(svsub_x) } else { PASS(svadd_x) }
#undef PASS
#undef LANE
        const double bv = svmaxv(all, G);
        const int64_t bc = svminv(svcmpeq(all, G, bv), X);  // first chunk that reaches the maximum
        for (int64_t j = bc; ; j += VL) {                   // first spin of the winning chunk at the maximum
            svbool_t p = svwhilelt_b64(j, N);
            svbool_t eq = svcmpeq(p, svadd_x(p, svmul_x(p, vnb, svld1(p, dE + j)), svld1(p, nz + j)), bv);
            if (svptest_any(p, eq)) { bi = (int)(j + (int64_t)svcntp_b64(p, svbrkb_z(p, eq))); break; }
        }
    }
    *E_io = E; *best_io = best;
}

#else
#if defined(USE_AVX2)
static inline double hmax256(__m256d x) {
    __m128d m = _mm_max_pd(_mm256_castpd256_pd128(x), _mm256_extractf128_pd(x, 1));
    return _mm_cvtsd_f64(_mm_max_sd(m, _mm_unpackhi_pd(m, m)));
}
#endif

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
        double bv = -1e300; long bl = -1; int bc = 0, j0 = 0;
#if defined(USE_NEON)
        {   // chunks of 64 spins, 4 accumulators x 2 doubles = 8 spins per iteration
            const int nv = n - n % 64;
            const float64x2_t vnb = vdupq_n_f64(-beta);
#define LANE(B, OP, OFF) { \
                float64x2_t D = vld1q_f64(dE + k + OFF); \
                D = OP(D, vmulq_f64(vld1q_f64(Ji + k + OFF), vld1q_f64(s + k + OFF))); \
                vst1q_f64(dE + k + OFF, D); \
                B = vmaxq_f64(B, vaddq_f64(vmulq_f64(vnb, D), vld1q_f64(nz + k + OFF))); }
#define PASS(OP) \
            for (int j = 0; j < nv; j += 64) { \
                float64x2_t b0 = vdupq_n_f64(-1e300), b1 = b0, b2 = b0, b3 = b0; \
                for (int k = j; k < j + 64; k += 8) { LANE(b0, OP, 0) LANE(b1, OP, 2) LANE(b2, OP, 4) LANE(b3, OP, 6) } \
                double v = vmaxvq_f64(vmaxq_f64(vmaxq_f64(b0, b1), vmaxq_f64(b2, b3))); \
                if (v > bv) { bv = v; bc = j; } \
            }
            if (si < 0) { PASS(vsubq_f64) } else { PASS(vaddq_f64) }
#undef PASS
#undef LANE
            j0 = nv;
        }
#elif defined(USE_AVX512)
        {   // chunks of 128 spins, 4 accumulators x 8 doubles = 32 spins per iteration
            const int nv = n - n % 128;
            const __m512d vnb = _mm512_set1_pd(-beta);
#define LANE(B, OP, OFF) { \
                __m512d D = _mm512_loadu_pd(dE + k + OFF); \
                D = OP(D, _mm512_mul_pd(_mm512_loadu_pd(Ji + k + OFF), _mm512_loadu_pd(s + k + OFF))); \
                _mm512_storeu_pd(dE + k + OFF, D); \
                B = _mm512_max_pd(B, _mm512_add_pd(_mm512_mul_pd(vnb, D), _mm512_loadu_pd(nz + k + OFF))); }
#define PASS(OP) \
            for (int j = 0; j < nv; j += 128) { \
                __m512d b0 = _mm512_set1_pd(-1e300), b1 = b0, b2 = b0, b3 = b0; \
                for (int k = j; k < j + 128; k += 32) { LANE(b0, OP, 0) LANE(b1, OP, 8) LANE(b2, OP, 16) LANE(b3, OP, 24) } \
                double v = _mm512_reduce_max_pd(_mm512_max_pd(_mm512_max_pd(b0, b1), _mm512_max_pd(b2, b3))); \
                if (v > bv) { bv = v; bc = j; } \
            }
            if (si < 0) { PASS(_mm512_sub_pd) } else { PASS(_mm512_add_pd) }
#undef PASS
#undef LANE
            j0 = nv;
        }
#elif defined(USE_AVX2)
        {   // chunks of 64 spins, 4 accumulators x 4 doubles = 16 spins per iteration
            const int nv = n - n % 64;
            const __m256d vnb = _mm256_set1_pd(-beta);
#define LANE(B, OP, OFF) { \
                __m256d D = _mm256_loadu_pd(dE + k + OFF); \
                D = OP(D, _mm256_mul_pd(_mm256_loadu_pd(Ji + k + OFF), _mm256_loadu_pd(s + k + OFF))); \
                _mm256_storeu_pd(dE + k + OFF, D); \
                B = _mm256_max_pd(B, _mm256_add_pd(_mm256_mul_pd(vnb, D), _mm256_loadu_pd(nz + k + OFF))); }
#define PASS(OP) \
            for (int j = 0; j < nv; j += 64) { \
                __m256d b0 = _mm256_set1_pd(-1e300), b1 = b0, b2 = b0, b3 = b0; \
                for (int k = j; k < j + 64; k += 16) { LANE(b0, OP, 0) LANE(b1, OP, 4) LANE(b2, OP, 8) LANE(b3, OP, 12) } \
                double v = hmax256(_mm256_max_pd(_mm256_max_pd(b0, b1), _mm256_max_pd(b2, b3))); \
                if (v > bv) { bv = v; bc = j; } \
            }
            if (si < 0) { PASS(_mm256_sub_pd) } else { PASS(_mm256_add_pd) }
#undef PASS
#undef LANE
            j0 = nv;
        }
#endif
        for (int j = j0; j < n; j++) {   // scalar path, or the tail after the vector chunks
            double d = dE[j] + Ji[j] * s[j] * si;
            dE[j] = d;
            double v = -beta * d + nz[j];
            if (v > bv) { bv = v; bl = j; }
        }
        if (bl < 0)                      // the maximum is in a vector chunk: find its first spin there
            for (int j = bc; ; j++)
                if (-beta * dE[j] + nz[j] == bv) { bl = j; break; }
        bi = (int)bl;
    }
    *E_io = E; *best_io = best;
}
#endif
