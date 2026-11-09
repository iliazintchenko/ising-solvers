// AVX2 pass (included by pfa_native.c). Chunks of 64 spins; a group of 16 spins = 4 float64 vectors of local
// fields -> 2 float32 score vectors of 8. Subtracts si * Ji from phi and returns the next spin, the first to reach
// max_j(s_j phi_j + nz_j * inv). Uses fused multiply-adds when built with FMA (any AVX2 CPU has it).
#if defined(__FMA__)
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

static inline int flip_pass(const double *Ji, double *phi, const float *s, const float *nz, int n, double si, float inv) {
    const __m256 vinv = _mm256_set1_ps(inv);
    const int nv = n & ~15;
    float bv = -__builtin_inff(); int bc = 0, bl = -1;
#define UPD(D, OP, O) __m256d D = OP(_mm256_loadu_pd(phi + k + O), _mm256_loadu_pd(Ji + k + O)); _mm256_storeu_pd(phi + k + O, D);
#define CV8(DA, DB) _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(DA)), _mm256_cvtpd_ps(DB), 1)
#define SC(DA, DB, O) FMA8(_mm256_loadu_ps(nz + k + O), vinv, _mm256_mul_ps(_mm256_loadu_ps(s + k + O), CV8(DA, DB)))
#define PASS(OP) \
    for (int j = 0; j < nv; j += 64) { \
        __m256 b0 = _mm256_set1_ps(-__builtin_inff()), b1 = b0; \
        for (int k = j; k < j + 64 && k < nv; k += 16) { \
            UPD(D0, OP, 0) UPD(D1, OP, 4) UPD(D2, OP, 8) UPD(D3, OP, 12) \
            b0 = _mm256_max_ps(b0, SC(D0, D1, 0)); b1 = _mm256_max_ps(b1, SC(D2, D3, 8)); \
        } \
        float v = hmax8(_mm256_max_ps(b0, b1)); \
        if (v > bv) { bv = v; bc = j; } \
    }
    if (si > 0) { PASS(_mm256_sub_pd) } else { PASS(_mm256_add_pd) }
#undef PASS
#undef UPD
#undef SC
#undef CV8
    tail_pass(Ji, phi, s, nz, nv, n, si, inv, &bv, &bl);
    if (bl >= 0) return bl;
    const __m128 vinv4 = _mm_set1_ps(inv), vm = _mm_set1_ps(bv);
    for (int k = bc; k < bc + 64 && k < nv; k += 4) {   // first spin of the winning chunk at the maximum
        __m128 w = FMA4(_mm_loadu_ps(nz + k), vinv4, _mm_mul_ps(_mm_loadu_ps(s + k), _mm256_cvtpd_ps(_mm256_loadu_pd(phi + k))));
        int msk = _mm_movemask_ps(_mm_cmpeq_ps(w, vm));
        if (msk) return k + __builtin_ctz(msk);
    }
    __builtin_trap();
}

// Categorical pass over whole chunks of 64 slots [0, nv), 8 slots per step: update, score, and keep the first chunk
// that reaches the largest score in *bv, *bc.
static inline double hmax4d(__m256d x) {
    __m128d m = _mm_max_pd(_mm256_castpd256_pd128(x), _mm256_extractf128_pd(x, 1));
    return _mm_cvtsd_f64(_mm_max_sd(m, _mm_unpackhi_pd(m, m)));
}
static inline void cat_chunks(double *F, const double *ra, const double *rb, const int *grp, const double *o,
                              const float *g, double T, int nv, double *bv, int *bc) {
    const __m256d vT = _mm256_set1_pd(T);
#define CS(B, O) { \
        const __m256d f = _mm256_add_pd(_mm256_sub_pd(_mm256_loadu_pd(F + k + O), _mm256_loadu_pd(ra + k + O)), \
                                        _mm256_loadu_pd(rb + k + O)); \
        _mm256_storeu_pd(F + k + O, f); \
        const __m256d oo = _mm256_i32gather_pd(o, _mm_loadu_si128((const __m128i *)(grp + k + O)), 8); \
        B = _mm256_max_pd(B, _mm256_add_pd(_mm256_sub_pd(oo, f), \
                                           _mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(g + k + O)), vT))); }
    for (int j = 0; j < nv; j += 64) {
        __m256d b0 = _mm256_set1_pd(-__builtin_inf()), b1 = b0;
        for (int k = j; k < j + 64; k += 8) { CS(b0, 0) CS(b1, 4) }
        const double v = hmax4d(_mm256_max_pd(b0, b1));
        if (v > *bv) { *bv = v; *bc = j; }
    }
#undef CS
}
