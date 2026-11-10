// SVE block for the selection tree (included by pfa_native_tree.c), vector-length agnostic: u[k] = a[k] + g[k] * T
// for 64 k. Returns the largest u[k]; *amax is the largest a[k]. tree_mask(): bit k is set where u[k] >= v.
static inline float tree_block(const float *a, const float *g, float T, float *u, float *amax) SVE_STREAMING {
    const int64_t VL = (int64_t)svcntw();
    const svfloat32_t vT = svdup_f32(T);
    svfloat32_t b = svdup_f32(-__builtin_inff()), m = b;
    for (int64_t k = 0; k < 64; k += VL) {
        const svbool_t p = svwhilelt_b32(k, (int64_t)64);
        const svfloat32_t va = svld1(p, a + k), vu = svmla_x(p, va, svld1(p, g + k), vT);
        svst1(p, u + k, vu);
        b = svmax_m(p, b, vu); m = svmax_m(p, m, va);
    }
    *amax = svmaxv(svptrue_b32(), m);
    return svmaxv(svptrue_b32(), b);
}

static inline uint64_t tree_mask(const float *u, float v) SVE_STREAMING {
    const int64_t VL = (int64_t)svcntw();
    uint64_t m = 0;
    for (int64_t k = 0; k < 64; k += VL) {   // bit k of the result = bit k mod 32 of its low or high half
        const svbool_t p = svwhilelt_b32(k, (int64_t)64), ge = svcmpge(p, svld1(p, u + k), v);
        const svuint32_t idx = svindex_u32((uint32_t)k, 1), bit = svlsl_x(p, svdup_u32(1), svand_x(p, idx, 31));
        const svbool_t low = svcmplt(p, idx, 32);
        m |= (uint64_t)svorv(svand_z(p, ge, low), bit) | (uint64_t)svorv(svbic_z(p, ge, low), bit) << 32;
    }
    return m;
}
