// PFA engine, native single-core version, compiled as one unit: this file picks the SIMD path at compile time from
// the target flags (see the Makefile) and includes the kernels,
//   pfa_native_spin.c                 spins: pfa_native_anneal()
//   pfa_native_categorical.c          categorical variables: pfa_categorical_anneal()
//   pfa_native_sparse_spin.c          spins, sparse couplings: pfa_sparse_anneal()
//   pfa_native_sparse_categorical.c   categorical variables, sparse couplings: pfa_sparse_categorical_anneal()
// Each holds its anneal loop. The first two include their pass over memory for the chosen path,
// pfa_native_spin_<path>.c or pfa_native_categorical_<path>.c; the sparse two share the selection tree of
// pfa_native_tree.c, which includes pfa_native_tree_<path>.c:
//   sve      vector-length agnostic, predicated   ARM with SVE (Graviton3 and 4, NVIDIA Grace, A64FX, ...)
//   neon     2 doubles / 4 floats per vector      every other 64-bit ARM (Apple Silicon, Graviton2, ...)
//   avx512   8 doubles per vector                 x86-64 with AVX-512
//   avx2     4 doubles per vector                 x86-64 with AVX2 (and FMA)
//   scalar                                        anything else
// Build with -ffp-contract=off: each kernel rescores the winning chunk and must reproduce the pass's scores exactly,
// so the compiler may not fuse multiply-adds in one place and not the other.
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

#if defined(TEST_STREAMING_SVE)
#include <arm_sme.h>
#include <arm_sve.h>
#define SIMD_PATH "sve-streaming"
#define SVE_STREAMING __arm_streaming
#define ANNEAL_ATTR __arm_locally_streaming
#define SPIN_PASS "pfa_native_spin_sve.c"
#define CATEGORICAL_PASS "pfa_native_categorical_sve.c"
#define TREE_BLOCK "pfa_native_tree_sve.c"
#elif defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#define SIMD_PATH "sve"
#define SVE_STREAMING
#define SPIN_PASS "pfa_native_spin_sve.c"
#define CATEGORICAL_PASS "pfa_native_categorical_sve.c"
#define TREE_BLOCK "pfa_native_tree_sve.c"
#elif defined(TEST_AVX512_SIMDE)
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx512.h>
#define SIMD_PATH "avx512-simde"
#define CVT512(D) _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(_mm512_castpd512_pd256(D))), \
                                       _mm256_cvtpd_ps(_mm512_extractf64x4_pd(D, 1)), 1)   // SIMDe lacks _mm512_cvtpd_ps
#define X86_PASSES
#define SPIN_PASS "pfa_native_spin_avx512.c"
#define CATEGORICAL_PASS "pfa_native_categorical_avx512.c"
#define TREE_BLOCK "pfa_native_tree_avx512.c"
#elif defined(TEST_AVX2_SIMDE)
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx2.h>
#include <simde/x86/fma.h>
#define SIMD_PATH "avx2-simde"
#define X86_PASSES
#define SPIN_PASS "pfa_native_spin_avx2.c"
#define CATEGORICAL_PASS "pfa_native_categorical_avx2.c"
#define TREE_BLOCK "pfa_native_tree_avx2.c"
#elif defined(TEST_SCALAR)
#define SIMD_PATH "scalar-test"
#define SPIN_PASS "pfa_native_spin_scalar.c"
#define CATEGORICAL_PASS "pfa_native_categorical_scalar.c"
#define TREE_BLOCK "pfa_native_tree_scalar.c"
#elif defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define SIMD_PATH "neon"
#define SPIN_PASS "pfa_native_spin_neon.c"
#define CATEGORICAL_PASS "pfa_native_categorical_neon.c"
#define TREE_BLOCK "pfa_native_tree_neon.c"
#elif defined(__AVX512F__)
#include <immintrin.h>
#define SIMD_PATH "avx512"
#define CVT512(D) _mm512_cvtpd_ps(D)
#define X86_PASSES
#define SPIN_PASS "pfa_native_spin_avx512.c"
#define CATEGORICAL_PASS "pfa_native_categorical_avx512.c"
#define TREE_BLOCK "pfa_native_tree_avx512.c"
#elif defined(__AVX2__)
#include <immintrin.h>
#define SIMD_PATH "avx2"
#define X86_PASSES
#define SPIN_PASS "pfa_native_spin_avx2.c"
#define CATEGORICAL_PASS "pfa_native_categorical_avx2.c"
#define TREE_BLOCK "pfa_native_tree_avx2.c"
#else
#define SIMD_PATH "scalar"
#define SPIN_PASS "pfa_native_spin_scalar.c"
#define CATEGORICAL_PASS "pfa_native_categorical_scalar.c"
#define TREE_BLOCK "pfa_native_tree_scalar.c"
#endif

#ifndef ANNEAL_ATTR
#define ANNEAL_ATTR
#endif
#ifndef SVE_STREAMING
#define SVE_STREAMING
#endif

#ifdef X86_PASSES   // shared by the x86 passes: fused multiply-adds when built with FMA (any AVX2 CPU has it)
#if defined(__FMA__) || defined(TEST_AVX2_SIMDE)
#define FMA8(A, B, C) _mm256_fmadd_ps(A, B, C)
#define FMA4(A, B, C) _mm_fmadd_ps(A, B, C)
#else
#define FMA8(A, B, C) _mm256_add_ps(_mm256_mul_ps(A, B), C)
#define FMA4(A, B, C) _mm_add_ps(_mm_mul_ps(A, B), C)
#endif
static inline float hmax8(__m256 x) {
    __m128 m = _mm_max_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    return _mm_cvtss_f32(_mm_max_ss(m, _mm_shuffle_ps(m, m, 1)));
}
#endif

const char *pfa_native_simd_path(void) { return SIMD_PATH; }

#include "pfa_native_spin.c"
#include "pfa_native_categorical.c"
#include "pfa_native_tree.c"
#include "pfa_native_sparse_spin.c"
#include "pfa_native_sparse_categorical.c"
