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

// Categorical pass over one chunk of slots [j, end), one float64 vector of fields per step: update (subtract a row,
// add a row, or both), score, and return the largest score. The float32 scores stay in the low halves of the 64-bit
// lanes: signs, noise and o are loaded as 32-bit values into 64-bit lanes. Where blk gives the vector's slots one o
// (vectors that divide a block of 8), it is one broadcast. Otherwise: slots are grouped by variable and every variable
// has one, so the variables of a vector's slots are grp[k] .. grp[k] + VL - 1, and o[grp[k]] is one contiguous load
// and a table lookup (no gather, which streaming mode lacks; o has 64 entries of padding).
static inline float cat_chunk(double *F, const float *S, const double *ra, const double *rb, const int *grp,
                              const int *blk, const float *o, const float *g, float T, int j, int end) SVE_STREAMING {
    const int64_t VL = (int64_t)svcntd(), END = end;
    const int whole = 8 % VL == 0;
    const svfloat32_t vT = svdup_f32(T);
    svfloat32_t B = svdup_f32(-__builtin_inff());
#define W(P) svld1uw_u64(pg, (const uint32_t *)(P))
#define SUB svsub_x(pg, svld1(pg, F + k), svld1(pg, ra + k))
#define ADD svadd_x(pg, svld1(pg, F + k), svld1(pg, rb + k))
#define BOTH svadd_x(pg, SUB, svld1(pg, rb + k))
#define CHUNK(UPD) \
    for (int64_t k = j; k < END; k += VL) { \
        const svbool_t pg = svwhilelt_b64(k, END); \
        const svfloat64_t f = UPD; \
        svst1(pg, F + k, f); \
        const int v = whole ? blk[k >> 3] : -1; \
        svfloat32_t oo; \
        if (v >= 0) oo = svdup_f32(o[v]); \
        else oo = svreinterpret_f32(svtbl(W(o + grp[k]), \
                      svreinterpret_u64(svsub_x(pg, svld1sw_s64(pg, grp + k), (int64_t)grp[k])))); \
        B = svmax_m(pg, B, svmla_x(pg, svmla_x(pg, oo, svreinterpret_f32(W(S + k)), svcvt_f32_f64_x(pg, f)), \
                                   svreinterpret_f32(W(g + k)), vT)); \
    }
    if (ra && rb) { CHUNK(BOTH) } else if (ra) { CHUNK(SUB) } else { CHUNK(ADD) }
#undef CHUNK
#undef BOTH
#undef ADD
#undef SUB
#undef W
    return svmaxv(svptrue_b32(), B);
}
