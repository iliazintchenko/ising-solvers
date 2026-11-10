// SVE pass for categorical variables (included by pfa_native.c), over one chunk of slots [j, end). One float64
// vector of fields per step: update (subtract a row, add a row, or both), score, and return the largest score. The
// float32 scores stay in the low halves of the 64-bit lanes: signs, noise and o are loaded as 32-bit values into
// 64-bit lanes. Where blk gives the vector's slots one o (vectors that divide a block of 8), it is one broadcast.
// Otherwise: slots are grouped by variable and every variable has one, so the variables of a vector's slots are
// grp[k] .. grp[k] + VL - 1, and o[grp[k]] is one contiguous load and a table lookup (no gather, which streaming
// mode lacks; o has 64 entries of padding).
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
