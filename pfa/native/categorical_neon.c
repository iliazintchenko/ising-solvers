// NEON pass for categorical variables (included by categorical.c), over one chunk of slots [j, end).
// 8 slots per step = 4 float64 vectors of fields -> 2 float32 score vectors: update (subtract a row, add a row, or
// both), score, and return the largest score. Where blk gives the 8 slots one o, it is one broadcast. Otherwise (NEON
// has no gather): slots are grouped by variable and every variable has one, so the variables of 4 slots are grp[k] ..
// grp[k] + 3, and o[grp[k]] is one contiguous load of 4 and a byte table lookup with indices
// 4 (grp[k + i] - grp[k]) + (0, 1, 2, 3) (o has 64 entries of padding).
static inline float cat_chunk(double *F, const float *S, const double *ra, const double *rb, const int *grp,
                              const int *blk, const float *o, const float *g, float T, int j, int end) {
    const float32x4_t vT = vdupq_n_f32(T);
    const uint32x4_t b0123 = vdupq_n_u32(0x03020100), x4 = vdupq_n_u32(0x04040404);
    float32x4_t b0 = vdupq_n_f32(-__builtin_inff()), b1 = b0;
#define OL(O) vreinterpretq_f32_u8(vqtbl1q_u8(vreinterpretq_u8_f32(vld1q_f32(o + grp[k + O])), vreinterpretq_u8_u32( \
        vmlaq_u32(b0123, vsubq_u32(vld1q_u32((const uint32_t *)(grp + k + O)), vdupq_n_u32((uint32_t)grp[k + O])), x4))))
#define SUB(O) vsubq_f64(vld1q_f64(F + k + O), vld1q_f64(ra + k + O))
#define ADD(O) vaddq_f64(vld1q_f64(F + k + O), vld1q_f64(rb + k + O))
#define BOTH(O) vaddq_f64(SUB(O), vld1q_f64(rb + k + O))
#define SC(DA, DB, O, OO) vfmaq_f32(vfmaq_f32(OO, vld1q_f32(S + k + O), vcvt_high_f32_f64(vcvt_f32_f64(DA), DB)), \
                                    vld1q_f32(g + k + O), vT)
#define CHUNK(UPD) \
    for (int k = j; k < end; k += 8) { \
        const int v = blk[k >> 3]; \
        float32x4_t o0, o1; \
        if (v >= 0) o0 = o1 = vld1q_dup_f32(o + v); \
        else { o0 = OL(0); o1 = OL(4); } \
        const float64x2_t f0 = UPD(0), f1 = UPD(2), f2 = UPD(4), f3 = UPD(6); \
        vst1q_f64(F + k, f0); vst1q_f64(F + k + 2, f1); vst1q_f64(F + k + 4, f2); vst1q_f64(F + k + 6, f3); \
        b0 = vmaxq_f32(b0, SC(f0, f1, 0, o0)); b1 = vmaxq_f32(b1, SC(f2, f3, 4, o1)); \
    }
    if (ra && rb) { CHUNK(BOTH) } else if (ra) { CHUNK(SUB) } else { CHUNK(ADD) }
#undef CHUNK
#undef SC
#undef BOTH
#undef ADD
#undef SUB
#undef OL
    return vmaxvq_f32(vmaxq_f32(b0, b1));
}
