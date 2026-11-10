// AVX2 block for the selection tree (included by pfa_native_tree.c): u[k] = a[k] + g[k] * T for 64 k, 16 per step.
// Returns the largest u[k]; *amax is the largest a[k]. tree_mask(): bit k is set where u[k] >= v.
static inline float tree_block(const float *a, const float *g, float T, float *u, float *amax) {
    const __m256 vT = _mm256_set1_ps(T);
    __m256 b0 = _mm256_set1_ps(-__builtin_inff()), b1 = b0, m0 = b0, m1 = b0;
    for (int k = 0; k < 64; k += 16) {
        const __m256 a0 = _mm256_loadu_ps(a + k), a1 = _mm256_loadu_ps(a + k + 8);
        const __m256 u0 = FMA8(_mm256_loadu_ps(g + k), vT, a0), u1 = FMA8(_mm256_loadu_ps(g + k + 8), vT, a1);
        _mm256_storeu_ps(u + k, u0); _mm256_storeu_ps(u + k + 8, u1);
        b0 = _mm256_max_ps(b0, u0); b1 = _mm256_max_ps(b1, u1);
        m0 = _mm256_max_ps(m0, a0); m1 = _mm256_max_ps(m1, a1);
    }
    *amax = hmax8(_mm256_max_ps(m0, m1));
    return hmax8(_mm256_max_ps(b0, b1));
}

static inline uint64_t tree_mask(const float *u, float v) {
    const __m256 vv = _mm256_set1_ps(v);
    uint64_t m = 0;
    for (int k = 0; k < 64; k += 8)
        m |= (uint64_t)(uint32_t)_mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(u + k), vv, _CMP_GE_OQ)) << k;
    return m;
}
