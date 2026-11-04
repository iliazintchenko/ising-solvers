// PFA engine, native single-core version: after flipping spin i, ONE pass over all spins updates every
// delta energy and finds the next spin to flip, argmax_j(-beta * dE_j + noise_j) (Gumbel-max sampling).
// Same algorithm as pfa_plain.py. The delta energies, energies and couplings are float64; only the selection
// scores -beta * dE_j + noise_j are compared in float32, which changes a pick only if two scores agree to
// about 7 digits, far below the noise scale.
//
// Arrays are padded to np spins (a multiple of 16) by inert spins with dE = +inf, so no remainder loops.
// Inner loop, per vector of spins:
//   - update: dE_j + (J_ij * s_j) * s_i equals dE_j + J_ij * s_j for s_i = +1 and dE_j - J_ij * s_j for
//     s_i = -1, so the sign is chosen once per flip instead of multiplied in per spin;
//   - score: converted to float32, so each vector holds twice as many scores;
//   - selection: only a running maximum per chunk of 64 spins; afterwards the winning chunk is rescored with
//     the same operations to find the first spin that reaches the maximum.
//
// Compile-time SIMD paths, chosen from the target flags (see the Makefile):
//   SVE      vector-length agnostic, predicated   ARM with SVE (Graviton3, A64FX, ...)
//   NEON     2 doubles / 4 floats per vector      every other 64-bit ARM (Apple Silicon, Graviton4, ...)
//   AVX-512  8 doubles per vector                 x86-64 with AVX-512
//   AVX2     4 doubles per vector                 x86-64 with AVX2
//   scalar                                        anything else
// Every path breaks argmax ties toward the smallest index (like numpy.argmax). Build with -ffp-contract=off:
// the rescore must reproduce the main loop's scores exactly, so the compiler may not fuse multiply-adds in
// one place and not the other.
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
#if defined(TEST_AVX512_SIMDE)   // SIMDe lacks _mm512_cvtpd_ps: convert the two 256-bit halves
#define CVT512(D) _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(_mm512_castpd512_pd256(D))), \
                                       _mm256_cvtpd_ps(_mm512_extractf64x4_pd(D, 1)), 1)
#elif defined(USE_AVX512)
#define CVT512(D) _mm512_cvtpd_ps(D)
#endif

const char *pfa_native_simd_path(void) { return SIMD_PATH; }

// J4: n rows with stride np (pad columns zero); s, dE: np entries (pads s = 1, dE = +inf);
// noise: n + np floats, noise[k] = noise_vec[k mod n]; sbest: n entries.
#if defined(USE_SVE)
// Whole anneal in one SVE function (in streaming mode no non-streaming calls are made, so no memcpy).
SVE_ATTR void pfa_native_anneal(const double *J4, double *s, double *dE, const float *noise, int n, int np,
                                long nflips, double beta_max, int bi, double *E_io, double *best_io, double *sbest) {
    double E = *E_io, best = *best_io, beta = 0.0, dbeta = beta_max / (double)(nflips - 1);
    int shift = 0;
    const int64_t VL = (int64_t)svcntd(), N = np, CH = 8 * VL;   // a group is 2 f64 vectors = one f32 vector
    const svbool_t all64 = svptrue_b64(), all32 = svptrue_b32();
    for (long t = 0; t < nflips; t++) {
        int i = bi; const double *Ji = J4 + (long)i * np;
        E += dE[i]; double si = s[i]; dE[i] = -dE[i]; s[i] = -si;
        if (E < best - 1e-6) {
            best = E;
            for (int64_t j = 0; j < n; j += VL) { svbool_t p = svwhilelt_b64(j, (int64_t)n); svst1(p, sbest + j, svld1(p, s + j)); }
        }
        beta += dbeta; if (++shift == n) shift = 0;
        const float *nz = noise + (n - shift);
        const svfloat32_t vnb = svdup_f32((float)-beta);
        svfloat32_t G = svdup_f32(-__builtin_inff());       // per-lane best over chunks, and the chunk it came from
        svint32_t X = svdup_s32(0);
#define SCORE(PF, K, D0, D1) \
            svadd_x(PF, svmul_x(PF, vnb, svuzp1(svcvt_f32_f64_x(all64, D0), svcvt_f32_f64_x(all64, D1))), svld1(PF, nz + (K)))
#define PASS(OP) \
        for (int64_t j = 0; j < N; j += CH) { \
            svfloat32_t B = svdup_f32(-__builtin_inff()); \
            for (int64_t k = j; k < j + CH && k < N; k += 2 * VL) { \
                svbool_t p0 = svwhilelt_b64(k, N), p1 = svwhilelt_b64(k + VL, N), pf = svwhilelt_b32(k, N); \
                svfloat64_t D0 = OP(p0, svld1(p0, dE + k), svmul_x(p0, svld1(p0, Ji + k), svld1(p0, s + k))); \
                svfloat64_t D1 = OP(p1, svld1(p1, dE + k + VL), svmul_x(p1, svld1(p1, Ji + k + VL), svld1(p1, s + k + VL))); \
                svst1(p0, dE + k, D0); svst1(p1, dE + k + VL, D1); \
                B = svmax_m(pf, B, SCORE(pf, k, D0, D1)); \
            } \
            svbool_t m = svcmpgt(all32, B, G); G = svsel(m, B, G); X = svsel(m, svdup_s32((int32_t)j), X); \
        }
        if (si < 0) { PASS(svsub_x) } else { PASS(svadd_x) }
#undef PASS
        const float bv = svmaxv(all32, G);
        const int64_t bc = svminv(svcmpeq(all32, G, bv), X);  // first chunk that reaches the maximum
        bi = -1;
        for (int64_t k = bc; k < bc + CH && k < N; k += 2 * VL) {   // first spin of that chunk at the maximum
            svbool_t p0 = svwhilelt_b64(k, N), p1 = svwhilelt_b64(k + VL, N), pf = svwhilelt_b32(k, N);
            svbool_t eq = svcmpeq(pf, SCORE(pf, k, svld1(p0, dE + k), svld1(p1, dE + k + VL)), bv);
            if (svptest_any(pf, eq)) { bi = (int)(k + (int64_t)svcntp_b32(pf, svbrkb_z(pf, eq))); break; }
        }
#undef SCORE
        if (bi < 0) __builtin_trap();
    }
    *E_io = E; *best_io = best;
}

#else
#if defined(USE_AVX2) || defined(USE_AVX512)
static inline float hmax8(__m256 x) {
    __m128 m = _mm_max_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    return _mm_cvtss_f32(_mm_max_ss(m, _mm_shuffle_ps(m, m, 1)));
}
#endif

void pfa_native_anneal(const double *J4, double *s, double *dE, const float *noise, int n, int np,
                       long nflips, double beta_max, int bi, double *E_io, double *best_io, double *sbest) {
    double E = *E_io, best = *best_io, beta = 0.0, dbeta = beta_max / (double)(nflips - 1);
    int shift = 0;
    for (long t = 0; t < nflips; t++) {
        int i = bi; const double *Ji = J4 + (long)i * np;
        E += dE[i]; double si = s[i]; dE[i] = -dE[i]; s[i] = -si;
        if (E < best - 1e-6) { best = E; memcpy(sbest, s, sizeof(double) * n); }
        beta += dbeta; if (++shift == n) shift = 0;
        const float *nz = noise + (n - shift);
        const float nb = (float)-beta;
        float bv = -__builtin_inff(); int bc = 0; long bl = -1;
#if defined(USE_NEON)
        {   // chunks of 64 spins; a group of 16 spins = 8 f64 vectors -> 4 f32 score vectors
            const float32x4_t vnb = vdupq_n_f32(nb);
#define UPD(D, OP, O) float64x2_t D = OP(vld1q_f64(dE + k + O), vmulq_f64(vld1q_f64(Ji + k + O), vld1q_f64(s + k + O))); \
                vst1q_f64(dE + k + O, D);
#define SC(DA, DB, O) vaddq_f32(vmulq_f32(vnb, vcvt_high_f32_f64(vcvt_f32_f64(DA), DB)), vld1q_f32(nz + k + O))
#define PASS(OP) \
            for (int j = 0; j < np; j += 64) { \
                float32x4_t b0 = vdupq_n_f32(-__builtin_inff()), b1 = b0, b2 = b0, b3 = b0; \
                for (int k = j; k < j + 64 && k < np; k += 16) { \
                    UPD(D0, OP, 0) UPD(D1, OP, 2) UPD(D2, OP, 4) UPD(D3, OP, 6) \
                    UPD(D4, OP, 8) UPD(D5, OP, 10) UPD(D6, OP, 12) UPD(D7, OP, 14) \
                    b0 = vmaxq_f32(b0, SC(D0, D1, 0)); b1 = vmaxq_f32(b1, SC(D2, D3, 4)); \
                    b2 = vmaxq_f32(b2, SC(D4, D5, 8)); b3 = vmaxq_f32(b3, SC(D6, D7, 12)); \
                } \
                float v = vmaxvq_f32(vmaxq_f32(vmaxq_f32(b0, b1), vmaxq_f32(b2, b3))); \
                if (v > bv) { bv = v; bc = j; } \
            }
            if (si < 0) { PASS(vsubq_f64) } else { PASS(vaddq_f64) }
#undef PASS
#undef UPD
            const float32x4_t vm = vdupq_n_f32(bv);
            for (int k = bc; k < bc + 64 && k < np; k += 4) {   // first spin of the winning chunk at the maximum
                uint32x4_t eq = vceqq_f32(SC(vld1q_f64(dE + k), vld1q_f64(dE + k + 2), 0), vm);
                if (vmaxvq_u32(eq)) { uint32_t e[4]; vst1q_u32(e, eq); bl = k + (e[0] ? 0 : e[1] ? 1 : e[2] ? 2 : 3); break; }
            }
#undef SC
        }
#elif defined(USE_AVX512)
        {   // chunks of 64 spins; a group of 16 spins = 2 f64 vectors -> 2 f32 score vectors of 8
            const __m256 vnb = _mm256_set1_ps(nb);
#define UPD(D, OP, O) __m512d D = OP(_mm512_loadu_pd(dE + k + O), _mm512_mul_pd(_mm512_loadu_pd(Ji + k + O), _mm512_loadu_pd(s + k + O))); \
                _mm512_storeu_pd(dE + k + O, D);
#define SC(D, O) _mm256_add_ps(_mm256_mul_ps(vnb, CVT512(D)), _mm256_loadu_ps(nz + k + O))
#define PASS(OP) \
            for (int j = 0; j < np; j += 64) { \
                __m256 b0 = _mm256_set1_ps(-__builtin_inff()), b1 = b0; \
                for (int k = j; k < j + 64 && k < np; k += 16) { \
                    UPD(D0, OP, 0) UPD(D1, OP, 8) \
                    b0 = _mm256_max_ps(b0, SC(D0, 0)); b1 = _mm256_max_ps(b1, SC(D1, 8)); \
                } \
                float v = hmax8(_mm256_max_ps(b0, b1)); \
                if (v > bv) { bv = v; bc = j; } \
            }
            if (si < 0) { PASS(_mm512_sub_pd) } else { PASS(_mm512_add_pd) }
#undef PASS
#undef UPD
#undef SC
        }
#elif defined(USE_AVX2)
        {   // chunks of 64 spins; a group of 16 spins = 4 f64 vectors -> 2 f32 score vectors of 8
            const __m256 vnb = _mm256_set1_ps(nb);
#define UPD(D, OP, O) __m256d D = OP(_mm256_loadu_pd(dE + k + O), _mm256_mul_pd(_mm256_loadu_pd(Ji + k + O), _mm256_loadu_pd(s + k + O))); \
                _mm256_storeu_pd(dE + k + O, D);
#define SC(DA, DB, O) _mm256_add_ps(_mm256_mul_ps(vnb, \
                _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(DA)), _mm256_cvtpd_ps(DB), 1)), _mm256_loadu_ps(nz + k + O))
#define PASS(OP) \
            for (int j = 0; j < np; j += 64) { \
                __m256 b0 = _mm256_set1_ps(-__builtin_inff()), b1 = b0; \
                for (int k = j; k < j + 64 && k < np; k += 16) { \
                    UPD(D0, OP, 0) UPD(D1, OP, 4) UPD(D2, OP, 8) UPD(D3, OP, 12) \
                    b0 = _mm256_max_ps(b0, SC(D0, D1, 0)); b1 = _mm256_max_ps(b1, SC(D2, D3, 8)); \
                } \
                float v = hmax8(_mm256_max_ps(b0, b1)); \
                if (v > bv) { bv = v; bc = j; } \
            }
            if (si < 0) { PASS(_mm256_sub_pd) } else { PASS(_mm256_add_pd) }
#undef PASS
#undef UPD
#undef SC
        }
#endif
#if defined(USE_AVX2) || defined(USE_AVX512)
        {   // first spin of the winning chunk at the maximum, 4 spins at a time (same per-lane operations)
            const __m128 vnb4 = _mm_set1_ps(nb), vm = _mm_set1_ps(bv);
            for (int k = bc; k < bc + 64 && k < np; k += 4) {
                __m128 w = _mm_add_ps(_mm_mul_ps(vnb4, _mm256_cvtpd_ps(_mm256_loadu_pd(dE + k))), _mm_loadu_ps(nz + k));
                int msk = _mm_movemask_ps(_mm_cmpeq_ps(w, vm));
                if (msk) { bl = k + __builtin_ctz(msk); break; }
            }
        }
#endif
#if !defined(USE_NEON) && !defined(USE_AVX512) && !defined(USE_AVX2)
        for (int j = 0; j < np; j++) {   // scalar path
            double d = dE[j] + Ji[j] * s[j] * si;
            dE[j] = d;
            float v = nb * (float)d + nz[j];
            if (v > bv) { bv = v; bl = j; }
        }
#endif
        if (bl < 0) __builtin_trap();
        bi = (int)bl;
    }
    *E_io = E; *best_io = best;
}
#endif
