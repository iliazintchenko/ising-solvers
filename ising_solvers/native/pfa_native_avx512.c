// AVX-512 pass (included by pfa_native.c). Chunks of 64 spins; a group of 16 spins = 2 float64 vectors of delta
// energies -> 2 float32 score vectors of 8. Updates dE with row Ji of the flipped spin (sign si) and returns the
// next spin, the first to reach max_j(nb * dE_j + nz_j).
static inline float hmax8(__m256 x) {
    __m128 m = _mm_max_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    return _mm_cvtss_f32(_mm_max_ss(m, _mm_shuffle_ps(m, m, 1)));
}

static inline int flip_pass(const double *Ji, const double *s, double *dE, const float *nz, int np, double si, float nb) {
    const __m256 vnb = _mm256_set1_ps(nb);
    float bv = -__builtin_inff(); int bc = 0;
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
    const __m128 vnb4 = _mm_set1_ps(nb), vm = _mm_set1_ps(bv);
    for (int k = bc; k < bc + 64 && k < np; k += 4) {   // first spin of the winning chunk at the maximum
        __m128 w = _mm_add_ps(_mm_mul_ps(vnb4, _mm256_cvtpd_ps(_mm256_loadu_pd(dE + k))), _mm_loadu_ps(nz + k));
        int msk = _mm_movemask_ps(_mm_cmpeq_ps(w, vm));
        if (msk) return k + __builtin_ctz(msk);
    }
    __builtin_trap();
}
