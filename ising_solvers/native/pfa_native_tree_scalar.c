// Scalar block for the selection tree (included by pfa_native_tree.c), for targets without a SIMD path:
// u[k] = a[k] + g[k] * T for 64 k. Returns the largest u[k]; *amax is the largest a[k]. tree_mask(): bit k is set
// where u[k] >= v.
static inline float tree_block(const float *a, const float *g, float T, float *u, float *amax) {
    float b = -__builtin_inff(), m = b;
    for (int k = 0; k < 64; k++) {
        u[k] = FMAF(g[k], T, a[k]);
        if (u[k] > b) b = u[k];
        if (a[k] > m) m = a[k];
    }
    *amax = m;
    return b;
}

static inline uint64_t tree_mask(const float *u, float v) {
    uint64_t m = 0;
    for (int k = 0; k < 64; k++) m |= (uint64_t)(u[k] >= v) << k;
    return m;
}
