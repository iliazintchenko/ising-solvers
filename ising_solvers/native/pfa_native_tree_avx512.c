// AVX-512 block for the selection tree (included by pfa_native_tree.c): u[k] = a[k] + g[k] * T for 64 k, 16 per
// step. Returns the largest u[k]; *amax is the largest a[k]. tree_mask(): bit k is set where u[k] >= v.
static inline float tree_block(const float *a, const float *g, float T, float *u, float *amax) {
    const __m512 vT = _mm512_set1_ps(T);
    __m512 b = _mm512_set1_ps(-__builtin_inff()), m = b;
    for (int k = 0; k < 64; k += 16) {
        const __m512 va = _mm512_loadu_ps(a + k), vu = _mm512_fmadd_ps(_mm512_loadu_ps(g + k), vT, va);
        _mm512_storeu_ps(u + k, vu);
        b = _mm512_max_ps(b, vu); m = _mm512_max_ps(m, va);
    }
#define HMAX16(X) hmax8(_mm256_max_ps(_mm512_castps512_ps256(X), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(X), 1))))
    *amax = HMAX16(m);
    return HMAX16(b);
#undef HMAX16
}

static inline uint64_t tree_mask(const float *u, float v) {
    const __m512 vv = _mm512_set1_ps(v);
    uint64_t m = 0;
    for (int k = 0; k < 64; k += 16) m |= (uint64_t)_mm512_cmp_ps_mask(_mm512_loadu_ps(u + k), vv, _CMP_GE_OQ) << k;
    return m;
}
