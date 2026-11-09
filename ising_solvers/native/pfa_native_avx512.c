// AVX-512 pass (included by pfa_native.c). Chunks of 64 spins; a group of 16 spins = 2 float64 vectors of local
// fields -> 2 float32 score vectors of 8. Subtracts si * Ji from phi and returns the next spin, the first to reach
// max_j(s_j phi_j + nz_j * inv).
static inline float hmax8(__m256 x) {
    __m128 m = _mm_max_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    return _mm_cvtss_f32(_mm_max_ss(m, _mm_shuffle_ps(m, m, 1)));
}

static inline int flip_pass(const double *Ji, double *phi, const float *s, const float *nz, int n, double si, float inv) {
    const __m256 vinv = _mm256_set1_ps(inv);
    const int nv = n & ~15;
    float bv = -__builtin_inff(); int bc = 0, bl = -1;
#define UPD(D, OP, O) __m512d D = OP(_mm512_loadu_pd(phi + k + O), _mm512_loadu_pd(Ji + k + O)); _mm512_storeu_pd(phi + k + O, D);
#define SC(D, O) _mm256_fmadd_ps(_mm256_loadu_ps(nz + k + O), vinv, _mm256_mul_ps(_mm256_loadu_ps(s + k + O), CVT512(D)))
#define PASS(OP) \
    for (int j = 0; j < nv; j += 64) { \
        __m256 b0 = _mm256_set1_ps(-__builtin_inff()), b1 = b0; \
        for (int k = j; k < j + 64 && k < nv; k += 16) { \
            UPD(D0, OP, 0) UPD(D1, OP, 8) \
            b0 = _mm256_max_ps(b0, SC(D0, 0)); b1 = _mm256_max_ps(b1, SC(D1, 8)); \
        } \
        float v = hmax8(_mm256_max_ps(b0, b1)); \
        if (v > bv) { bv = v; bc = j; } \
    }
    if (si > 0) { PASS(_mm512_sub_pd) } else { PASS(_mm512_add_pd) }
#undef PASS
#undef UPD
#undef SC
    tail_pass(Ji, phi, s, nz, nv, n, si, inv, &bv, &bl);
    if (bl >= 0) return bl;
    const __m128 vinv4 = _mm_set1_ps(inv), vm = _mm_set1_ps(bv);
    for (int k = bc; k < bc + 64 && k < nv; k += 4) {   // first spin of the winning chunk at the maximum
        __m128 w = _mm_fmadd_ps(_mm_loadu_ps(nz + k), vinv4, _mm_mul_ps(_mm_loadu_ps(s + k), _mm256_cvtpd_ps(_mm256_loadu_pd(phi + k))));
        int msk = _mm_movemask_ps(_mm_cmpeq_ps(w, vm));
        if (msk) return k + __builtin_ctz(msk);
    }
    __builtin_trap();
}

// Categorical pass over whole chunks of 64 slots [0, nv), 16 slots per step: update, score, and keep the first chunk
// that reaches the largest score in *bv, *bc.
static inline void cat_chunks(double *F, const double *ra, const double *rb, const int *grp, const double *o,
                              const float *g, double T, int nv, double *bv, int *bc) {
    const __m512d vT = _mm512_set1_pd(T);
#define CS(B, O) { \
        const __m512d f = _mm512_add_pd(_mm512_sub_pd(_mm512_loadu_pd(F + k + O), _mm512_loadu_pd(ra + k + O)), \
                                        _mm512_loadu_pd(rb + k + O)); \
        _mm512_storeu_pd(F + k + O, f); \
        const __m512d oo = GATHER512(_mm256_loadu_si256((const __m256i *)(grp + k + O)), o); \
        B = _mm512_max_pd(B, _mm512_add_pd(_mm512_sub_pd(oo, f), \
                                           _mm512_mul_pd(CVTPS512(_mm256_loadu_ps(g + k + O)), vT))); }
    for (int j = 0; j < nv; j += 64) {
        __m512d b0 = _mm512_set1_pd(-__builtin_inf()), b1 = b0;
        for (int k = j; k < j + 64; k += 16) { CS(b0, 0) CS(b1, 8) }
        const double v = _mm512_reduce_max_pd(_mm512_max_pd(b0, b1));
        if (v > *bv) { *bv = v; *bc = j; }
    }
#undef CS
}
