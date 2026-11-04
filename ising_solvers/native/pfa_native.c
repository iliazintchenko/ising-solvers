// PFA engine, native single-core version: after flipping spin i, ONE pass over all spins updates every
// local field and finds the next spin to flip, argmax_j(-beta * dE_j + noise_j) (Gumbel-max sampling).
//
// State per spin j: phi_j = (h_j + sum_k J_jk s_k) / 2 in float64, and s_j as a float32 +-1. Then
// dE_j = -4 s_j phi_j, and:
//   - flipping spin i changes every phi_j by -s_i J_ij (s_i before the flip): one add or subtract of the
//     caller's coupling row, chosen once per flip; no spin load, no multiply, no copy of the couplings;
//   - phi_i itself does not change when spin i flips (only s_i does), and the fields never need updating;
//   - the selection score -beta * dE_j + noise_j, divided by 4 beta, is s_j phi_j + noise_j / (4 beta):
//     computed in float32 with one multiply and one fused multiply-add. Float32 changes a pick only if two
//     scores agree to about 7 digits, far below the noise scale.
// The noise is one Gumbel vector stored twice; each step reads it from a random offset (a cyclic shift), so every
// spin sees a new noise value at every step. A fixed shift of one per step would make the large noise values travel
// through the spin indices, which ties results to how the spins happen to be ordered.
// Selection keeps only a running maximum per chunk of 64 spins; afterwards the winning chunk is rescored
// with the same operations to find the first spin that reaches the maximum.
//
// This file holds the anneal loop and picks the SIMD path at compile time from the target flags (see the
// Makefile). Each path lives in its own file and defines one function, flip_pass(), which does the pass:
//   pfa_native_sve.c     vector-length agnostic, predicated   ARM with SVE (Graviton3 and 4, NVIDIA Grace, A64FX, ...)
//   pfa_native_neon.c    2 doubles / 4 floats per vector      every other 64-bit ARM (Apple Silicon, Graviton2, ...)
//   pfa_native_avx512.c  8 doubles per vector                 x86-64 with AVX-512
//   pfa_native_avx2.c    4 doubles per vector                 x86-64 with AVX2 (and FMA)
//   pfa_native_scalar.c                                       anything else
// The SIMD passes handle spins in groups of 16 and the last n mod 16 spins with tail_pass() below.
// Every path breaks argmax ties toward the smallest index (like numpy.argmax). Build with -ffp-contract=off:
// the rescore must reproduce the main loop's scores exactly, so the compiler may not fuse multiply-adds in
// one place and not the other.
//
// Test-only switches, for checking paths on hardware that lacks them:
//   -DTEST_STREAMING_SVE  runs the SVE path in SME streaming mode (e.g. Apple M4, -mcpu=apple-m4)
//   -DTEST_AVX512_SIMDE   runs the AVX-512 path through the SIMDe emulation headers (-I<simde include dir>)
#include <stdint.h>

#if defined(__aarch64__) || defined(__FMA__)
#define FMAF(A, B, C) __builtin_fmaf(A, B, C)
#else
#define FMAF(A, B, C) ((A) * (B) + (C))   // no hardware FMA: a library fmaf would be far slower
#endif

// Spins [j0, n) one at a time, same update and score as the SIMD groups; updates *bv, *bl on a strictly
// larger score.
static inline void tail_pass(const double *Ji, double *phi, const float *s, const float *nz, int j0, int n,
                             double si, float inv, float *bv, int *bl) {
    for (int j = j0; j < n; j++) {
        double p = phi[j] - Ji[j] * si;
        phi[j] = p;
        float v = FMAF(nz[j], inv, s[j] * (float)p);
        if (v > *bv) { *bv = v; *bl = j; }
    }
}

#if defined(TEST_STREAMING_SVE)
#include <arm_sme.h>
#include <arm_sve.h>
#define SIMD_PATH "sve-streaming"
#define SVE_STREAMING __arm_streaming
#define ANNEAL_ATTR __arm_locally_streaming
#include "pfa_native_sve.c"
#elif defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#define SIMD_PATH "sve"
#define SVE_STREAMING
#include "pfa_native_sve.c"
#elif defined(TEST_AVX512_SIMDE)
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx512.h>
#define SIMD_PATH "avx512-simde"
#define CVT512(D) _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(_mm512_castpd512_pd256(D))), \
                                       _mm256_cvtpd_ps(_mm512_extractf64x4_pd(D, 1)), 1)   // SIMDe lacks _mm512_cvtpd_ps
#include "pfa_native_avx512.c"
#elif defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define SIMD_PATH "neon"
#include "pfa_native_neon.c"
#elif defined(__AVX512F__)
#include <immintrin.h>
#define SIMD_PATH "avx512"
#define CVT512(D) _mm512_cvtpd_ps(D)
#include "pfa_native_avx512.c"
#elif defined(__AVX2__)
#include <immintrin.h>
#define SIMD_PATH "avx2"
#include "pfa_native_avx2.c"
#else
#define SIMD_PATH "scalar"
#include "pfa_native_scalar.c"
#endif

#ifndef ANNEAL_ATTR
#define ANNEAL_ATTR
#endif

const char *pfa_native_simd_path(void) { return SIMD_PATH; }

// J: the n x n couplings, row-major; phi: n local fields / 2 (updated in place); s: n spins as float32 +-1
// (updated in place); noise: the noise vector stored twice (2n floats); sbest: n floats (out).
// seed: for the random noise offsets.
ANNEAL_ATTR void pfa_native_anneal(const double *J, double *phi, float *s, const float *noise, int n,
                                   long nflips, double beta_max, int bi, double *E_io, double *best_io, float *sbest,
                                   uint64_t seed) {
    double E = *E_io, best = *best_io, beta = 0.0, dbeta = beta_max / (double)(nflips - 1);
    uint64_t x = seed | 1;   // xorshift64* generator for the offsets
    for (long t = 0; t < nflips; t++) {
        int i = bi; double si = (double)s[i];
        E -= 4.0 * si * phi[i]; s[i] = -s[i];   // E += dE_i
        if (E < best - 1e-6) { best = E; __builtin_memcpy(sbest, s, sizeof(float) * n); }
        beta += dbeta;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        int offset = (int)(((x * 2685821657736338717ULL) >> 32) * (uint64_t)n >> 32);   // uniform in [0, n)
        bi = flip_pass(J + (long)i * n, phi, s, noise + offset, n, si, (float)(0.25 / beta));
    }
    *E_io = E; *best_io = best;
}
