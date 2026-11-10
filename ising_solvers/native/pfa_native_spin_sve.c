// SVE pass for spins (included by pfa_native_spin.c), vector-length agnostic: a group is 2 float64 vectors of local
// fields = one float32 score vector, a chunk is 4 groups, predicates cover the end of the arrays. The chunk holding
// the maximum is tracked per lane without branches. Subtracts si * Ji from phi and returns the next spin, the first
// to reach max_j(s_j phi_j + nz_j * inv).
static inline int flip_pass(const double *Ji, double *phi, const float *s, const float *nz, int n, double si,
                            float inv) SVE_STREAMING {
    const int64_t VL = (int64_t)svcntd(), N = n, CH = 8 * VL;
    const svbool_t all64 = svptrue_b64(), all32 = svptrue_b32();
    const svfloat32_t vinv = svdup_f32(inv);
    svfloat32_t G = svdup_f32(-__builtin_inff());   // per-lane best over chunks, and the chunk it came from
    svint32_t X = svdup_s32(0);
#define SCORE(PF, K, D0, D1) svmla_x(PF, svmul_x(PF, svld1(PF, s + (K)), \
    svuzp1(svcvt_f32_f64_x(all64, D0), svcvt_f32_f64_x(all64, D1))), svld1(PF, nz + (K)), vinv)
#define PASS(OP) \
    for (int64_t j = 0; j < N; j += CH) { \
        svfloat32_t B = svdup_f32(-__builtin_inff()); \
        for (int64_t k = j; k < j + CH && k < N; k += 2 * VL) { \
            svbool_t p0 = svwhilelt_b64(k, N), p1 = svwhilelt_b64(k + VL, N), pf = svwhilelt_b32(k, N); \
            svfloat64_t D0 = OP(p0, svld1(p0, phi + k), svld1(p0, Ji + k)); \
            svfloat64_t D1 = OP(p1, svld1(p1, phi + k + VL), svld1(p1, Ji + k + VL)); \
            svst1(p0, phi + k, D0); svst1(p1, phi + k + VL, D1); \
            B = svmax_m(pf, B, SCORE(pf, k, D0, D1)); \
        } \
        svbool_t m = svcmpgt(all32, B, G); G = svsel(m, B, G); X = svsel(m, svdup_s32((int32_t)j), X); \
    }
    if (si > 0) { PASS(svsub_x) } else { PASS(svadd_x) }
#undef PASS
    const float bv = svmaxv(all32, G);
    const int64_t bc = svminv(svcmpeq(all32, G, bv), X);   // first chunk that reaches the maximum
    for (int64_t k = bc; k < bc + CH && k < N; k += 2 * VL) {   // first spin of that chunk at the maximum
        svbool_t p0 = svwhilelt_b64(k, N), p1 = svwhilelt_b64(k + VL, N), pf = svwhilelt_b32(k, N);
        svbool_t eq = svcmpeq(pf, SCORE(pf, k, svld1(p0, phi + k), svld1(p1, phi + k + VL)), bv);
        if (svptest_any(pf, eq)) return (int)(k + (int64_t)svcntp_b32(pf, svbrkb_z(pf, eq)));
    }
#undef SCORE
    __builtin_trap();
}
