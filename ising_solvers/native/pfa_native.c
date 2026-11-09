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
// Tabu: for the tenure flips after a spin flips (default n/4), its dE is raised by lambda temperatures at the time
// of its flip (lambda / beta_i, default lambda = 3): its rate is divided by e^lambda right after the flip and by
// more as the anneal cools. This spreads the flips over more distinct spins, mostly while the anneal is still hot;
// a flip worth many more temperatures (such as stepping back down from an expensive climb) still wins. The penalty
// is applied by shifting the spin's stored field, phi~_i = phi_i - s_i * c_i with c_i = lambda / (4 beta) at the
// time of its flip, so the pass reads nothing extra; the anneal loop adds and removes the shift, O(1) per flip.
//
// This file holds the anneal loop and picks the SIMD path at compile time from the target flags (see the
// Makefile). Each path lives in its own file and defines the passes, flip_pass() for spins and cat_chunks() for
// categorical variables:
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
//   -DTEST_AVX2_SIMDE     runs the AVX2 path through SIMDe
//   -DTEST_SCALAR         builds the scalar path on any target
#include <stdint.h>
#include <stdlib.h>

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

// Categorical score of value slot k: (o[grp[k]] - F[k]) + g[k] * T, after its update F[k] = (F[k] - ra[k]) + rb[k].
// Every path does these operations in this order, so all paths give bit-identical results.
static inline double cat_score(const double *F, const int *grp, const double *o, const float *g, double T, int k) {
    return (o[grp[k]] - F[k]) + (double)g[k] * T;
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
#define CVTPS512(X) _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_cvtps_pd(_mm256_castps256_ps128(X))), \
                                       _mm256_cvtps_pd(_mm256_extractf128_ps(X, 1)), 1)   // nor _mm512_cvtps_pd
#define GATHER512(I, B) _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_i32gather_pd(B, _mm256_castsi256_si128(I), 8)), \
                                           _mm256_i32gather_pd(B, _mm256_extracti128_si256(I, 1), 8), 1)   // nor gathers
#include "pfa_native_avx512.c"
#elif defined(TEST_AVX2_SIMDE)
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx2.h>
#define SIMD_PATH "avx2-simde"
#include "pfa_native_avx2.c"
#elif defined(TEST_SCALAR)
#define SIMD_PATH "scalar-test"
#include "pfa_native_scalar.c"
#elif defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define SIMD_PATH "neon"
#include "pfa_native_neon.c"
#elif defined(__AVX512F__)
#include <immintrin.h>
#define SIMD_PATH "avx512"
#define CVT512(D) _mm512_cvtpd_ps(D)
#define CVTPS512(X) _mm512_cvtps_pd(X)
#define GATHER512(I, B) _mm512_i32gather_pd(I, B, 8)
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
#ifndef SVE_STREAMING
#define SVE_STREAMING
#endif

const char *pfa_native_simd_path(void) { return SIMD_PATH; }

// pen[j]: the shift c_j currently applied to phi[j] (0 when j is not tabu); last[j]: step of j's last flip;
// recent[t % tenure]: the spin flipped at step t.
ANNEAL_ATTR static void anneal_loop(const double *J, double *phi, float *s, const float *noise, int n, long nflips,
                                    double beta_start, double beta_max, int bi, double *E_io, double *best_io,
                                    float *sbest, uint64_t seed, double lambda, int tenure, double *pen, long *last,
                                    int *recent) {
    double E = *E_io, best = *best_io, beta = beta_start, dbeta = (beta_max - beta_start) / (double)(nflips - 1);
    uint64_t x = seed | 1;   // xorshift64* generator for the offsets
    for (long t = 0; t < nflips; t++) {
        int i = bi; double si = (double)s[i];
        double phii = phi[i] + si * pen[i];   // true local field of i
        E -= 4.0 * si * phii;                 // E += dE_i
        phi[i] = phii; pen[i] = 0.0; s[i] = -s[i];
        beta += dbeta;
        if (tenure > 0) {
            if (t >= tenure) {   // the spin flipped tenure steps ago leaves the tabu list, unless it flipped again since
                int j = recent[t % tenure];
                if (last[j] == t - tenure) { phi[j] += (double)s[j] * pen[j]; pen[j] = 0.0; }
            }
            pen[i] = lambda * 0.25 / beta; phi[i] -= (double)s[i] * pen[i];
            recent[t % tenure] = i; last[i] = t;
        }
        if (E < best) { best = E; __builtin_memcpy(sbest, s, sizeof(float) * n); }
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        int offset = (int)(((x * 2685821657736338717ULL) >> 32) * (uint64_t)n >> 32);   // uniform in [0, n)
        bi = flip_pass(J + (long)i * n, phi, s, noise + offset, n, si, (float)(0.25 / beta));
    }
    for (int j = 0; j < n; j++) phi[j] += (double)s[j] * pen[j];   // leave the true local fields
    *E_io = E; *best_io = best;
}

// J: the n x n couplings, row-major; phi: n local fields / 2 (updated in place); s: n spins as float32 +-1
// (updated in place); noise: the noise vector stored twice (2n floats); sbest: n floats (out).
// beta rises linearly from beta_start to beta_max; seed: for the random noise offsets; tabu: penalty lambda in
// temperatures for the last tenure flips (tenure 0: no tabu).
void pfa_native_anneal(const double *J, double *phi, float *s, const float *noise, int n,
                       long nflips, double beta_start, double beta_max, int bi, double *E_io, double *best_io,
                       float *sbest, uint64_t seed, double lambda, int tenure) {
    double *pen = calloc(n, sizeof(double));
    long *last = malloc(sizeof(long) * n);
    int *recent = malloc(sizeof(int) * (tenure > 0 ? tenure : 1));
    for (int j = 0; j < n; j++) last[j] = -1;
    anneal_loop(J, phi, s, noise, n, nflips, beta_start, beta_max, bi, E_io, best_io, sbest, seed, lambda, tenure, pen,
                last, recent);
    free(pen); free(last); free(recent);
}

// Categorical (Potts) variables. Variable r takes one of q_r >= 2 values; the first is its reference, and each other
// value owns a slot k, with grp[k] = r (slots grouped by variable). Values are stored relative to the reference: F[k]
// is the energy of slot k's value minus that of the reference value, given the other variables' current values, and
// Jt (m x m, symmetric, zero within a variable) holds the matching pair terms, so every state keeps its energy.
// cur[r] is the slot of r's value, or -1 at the reference.
// A move from value a to b changes every field by Jt[b] - Jt[a] (the reference has no row: a row of zeros stands in),
// applied in the next step's pass. Moves to another slot k have dE = F[k] - Fc[r], with Fc[r] = F[cur[r]] (0 at the
// reference): with o[r] = Fc[r] - pen[r], the score -dE - pen + noise / beta is (o[r] - F[k]) + g[k] / beta. The pass
// updates F and scores every slot, keeping the maximum per chunk of 64 slots, as in the spin problem. Fc lives apart:
// F holds +inf at the current slots, so they score -inf. The moves back to the reference, dE = -Fc[r], are scored with
// o (before the pass, one per variable): (o[r]) + g[cur[r]] / beta; on a tie the pass wins. The tabu acts per variable,
// through o. With q = 2 every variable has one slot and this is the spin problem; pfa_native_anneal solves it.
static inline int cat_pass(double *F, const double *ra, const double *rb, const int *grp, const double *o,
                           const float *g, double T, int m, double *bv_io) SVE_STREAMING {
    const int nv = m & ~63; double bv = -__builtin_inf(); int bc = 0, bl = -1;
    cat_chunks(F, ra, rb, grp, o, g, T, nv, &bv, &bc);
    for (int k = nv; k < m; k++) {
        F[k] = (F[k] - ra[k]) + rb[k];
        const double sc = cat_score(F, grp, o, g, T, k);
        if (sc > bv) { bv = sc; bl = k; }
    }
    *bv_io = bv;
    if (bl >= 0 || bv == -__builtin_inf()) return bl;
    for (int k = bc; k < bc + 64; k++) if (cat_score(F, grp, o, g, T, k) == bv) return k;
    __builtin_trap();
}

ANNEAL_ATTR static void cat_loop(const double *Jt, double *F, int m, int V, const int *grp, int *cur,
                                 const float *noise, long nmoves, double beta_start, double beta_max, double *E_io,
                                 double *best_io, int *best_cur, uint64_t seed, double lambda, int tenure, double *pen,
                                 long *last, int *recent, double *o, double *Fc, const double *zero) {
    double E = *E_io, best = *best_io, beta = beta_start, dbeta = (beta_max - beta_start) / (double)(nmoves - 1);
    uint64_t x = seed | 1;
    const double *ra = zero, *rb = zero;   // rows of the last move, applied in the next pass
    for (long t = 0; t < nmoves; t++) {
        beta += dbeta;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        const float *g = noise + (int)(((x * 2685821657736338717ULL) >> 32) * (uint64_t)m >> 32);
        const double T = 1.0 / beta;
        double sr = -__builtin_inf(); int rr = -1;   // best move back to a reference
        for (int r = 0; r < V; r++) {
            const int c = cur[r];
            if (c >= 0) {
                Fc[r] = (Fc[r] - ra[c]) + rb[c];
                o[r] = Fc[r] - pen[r];
                const double sc = o[r] + (double)g[c] * T;
                if (sc > sr) { sr = sc; rr = r; }
            } else o[r] = -pen[r];
        }
        double bv; int k = cat_pass(F, ra, rb, grp, o, g, T, m, &bv);
        int r, c, b; double dE;
        if (sr > bv) { r = rr; c = cur[r]; b = -1; dE = -Fc[r]; }
        else { r = grp[k]; c = cur[r]; b = k; dE = F[k] - (c >= 0 ? Fc[r] : 0.0); }
        if (c >= 0) F[c] = Fc[r];
        if (b >= 0) { Fc[r] = F[b]; F[b] = __builtin_inf(); }
        ra = c >= 0 ? Jt + (long)c * m : zero; rb = b >= 0 ? Jt + (long)b * m : zero;
        cur[r] = b; E += dE;
        if (tenure > 0) {
            if (t >= tenure) {
                const int q = recent[t % tenure];
                if (last[q] == t - tenure) pen[q] = 0.0;
            }
            pen[r] = lambda / beta; recent[t % tenure] = r; last[r] = t;
        }
        if (E < best) { best = E; __builtin_memcpy(best_cur, cur, sizeof(int) * V); }
    }
    for (int k = 0; k < m; k++) F[k] = (F[k] - ra[k]) + rb[k];   // leave the fields of the final state
    for (int r = 0; r < V; r++) if (cur[r] >= 0) F[cur[r]] = (Fc[r] - ra[cur[r]]) + rb[cur[r]];
    *E_io = E; *best_io = best;
}

// Jt: m x m pair terms; F: m fields (updated in place); grp: variable of each slot, 0..V-1, grouped; cur: V current
// slots (updated in place); noise: m noise values stored twice; best_cur: V slots (out). As pfa_native_anneal otherwise.
void pfa_categorical_anneal(const double *Jt, double *F, int m, int V, const int *grp, int *cur, const float *noise,
                            long nmoves, double beta_start, double beta_max, double *E_io, double *best_io,
                            int *best_cur, uint64_t seed, double lambda, int tenure) {
    double *pen = calloc(V, sizeof(double)), *o = calloc(V + 64, sizeof(double)), *Fc = malloc(sizeof(double) * V);
    double *zero = calloc(m, sizeof(double));
    long *last = malloc(sizeof(long) * V);
    int *recent = malloc(sizeof(int) * (tenure > 0 ? tenure : 1));
    for (int r = 0; r < V; r++) {
        last[r] = -1;
        if (cur[r] >= 0) { Fc[r] = F[cur[r]]; F[cur[r]] = __builtin_inf(); }
    }
    cat_loop(Jt, F, m, V, grp, cur, noise, nmoves, beta_start, beta_max, E_io, best_io, best_cur, seed, lambda,
             tenure, pen, last, recent, o, Fc, zero);
    free(pen); free(o); free(Fc); free(zero); free(last); free(recent);
}

// One-hot input -> reference-value layout, in one pass. C: n x n couplings over all slots, f: n fields; ref[r]: the
// reference slot of variable r; slots[i], sgrp[i]: the i-th other slot and its variable (m of them, grouped by
// variable). Writes Jt (m x m, zero within a variable), ut (m) and returns the energy of the all-reference state.
double pfa_categorical_prepare(const double *C, const double *f, int n, int V, const int *ref, int m, const int *slots,
                               const int *sgrp, double *Jt, double *ut) {
    double e0 = 0.0;
    for (int r = 0; r < V; r++) {
        const double *row = C + (long)ref[r] * n; double s = 0.0;
        for (int q = 0; q < V; q++) if (q != r) s += row[ref[q]];
        e0 += f[ref[r]] + 0.5 * s;
    }
    for (int i = 0; i < m; i++) {
        int gi = sgrp[i];
        const double *ra = C + (long)slots[i] * n, *rr = C + (long)ref[gi] * n;
        double *out = Jt + (long)i * m;
        for (int j = 0; j < m; j++) {
            int gj = sgrp[j], b = slots[j], rb = ref[gj];
            out[j] = gj == gi ? 0.0 : (ra[b] + rr[rb]) - (ra[rb] + rr[b]);   // same rounding for (i, j) and (j, i)
        }
        double s = f[slots[i]] - f[ref[gi]];
        for (int q = 0; q < V; q++) if (q != gi) s += ra[ref[q]] - rr[ref[q]];
        ut[i] = s;
    }
    return e0;
}
