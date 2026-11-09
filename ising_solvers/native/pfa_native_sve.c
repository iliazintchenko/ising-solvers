// SVE pass (included by pfa_native.c), vector-length agnostic: a group is 2 float64 vectors of local fields
// = one float32 score vector, a chunk is 4 groups, predicates cover the end of the arrays. The chunk holding the
// maximum is tracked per lane without branches. Subtracts si * Ji from phi and returns the next spin, the first
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

// Categorical pass over whole chunks of 64 slots [0, nv), one vector per step: update, score, and keep the first chunk
// that reaches the largest score in *bv, *bc. Slots are grouped by variable and every variable has one, so the
// variables of a vector's slots are grp[k] .. grp[k] + VL - 1: o[grp[k]] is one contiguous load and a table lookup
// (no gather, which streaming mode lacks; o has 64 entries of padding). The noise is loaded as 32-bit values into
// 64-bit lanes and widened.
static inline void cat_chunks(double *F, const double *ra, const double *rb, const int *grp, const double *o,
                              const float *g, double T, int nv, double *bv, int *bc) SVE_STREAMING {
    const int64_t VL = (int64_t)svcntd();
    const svfloat64_t vT = svdup_f64(T);
    for (int64_t j = 0; j < nv; j += 64) {
        svfloat64_t B = svdup_f64(-__builtin_inf());
        for (int64_t k = j; k < j + 64; k += VL) {
            const svbool_t pg = svwhilelt_b64(k, j + 64);
            const svfloat64_t f = svadd_x(pg, svsub_x(pg, svld1(pg, F + k), svld1(pg, ra + k)), svld1(pg, rb + k));
            svst1(pg, F + k, f);
            const svint64_t gi = svld1sw_s64(pg, grp + k);
            const svfloat64_t oo = svtbl(svld1(pg, o + grp[k]), svreinterpret_u64(svsub_x(pg, gi, (int64_t)grp[k])));
            const svfloat64_t gg = svcvt_f64_f32_x(pg, svreinterpret_f32_u64(svld1uw_u64(pg, (const uint32_t *)(g + k))));
            B = svmax_m(pg, B, svadd_x(pg, svsub_x(pg, oo, f), svmul_x(pg, gg, vT)));
        }
        const double v = svmaxv(svptrue_b64(), B);
        if (v > *bv) { *bv = v; *bc = (int)j; }
    }
}
