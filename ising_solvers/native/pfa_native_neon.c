// NEON pass (included by pfa_native.c). Chunks of 64 spins; a group of 16 spins = 8 float64 vectors of local
// fields -> 4 float32 score vectors. Subtracts si * Ji from phi and returns the next spin, the first to reach
// max_j(s_j phi_j + nz_j * inv).
static inline int flip_pass(const double *Ji, double *phi, const float *s, const float *nz, int n, double si, float inv) {
    const float32x4_t vinv = vdupq_n_f32(inv);
    const int nv = n & ~15;
    float bv = -__builtin_inff(); int bc = 0, bl = -1;
#define UPD(D, OP, O) float64x2_t D = OP(vld1q_f64(phi + k + O), vld1q_f64(Ji + k + O)); vst1q_f64(phi + k + O, D);
#define SC(DA, DB, O) vfmaq_f32(vmulq_f32(vld1q_f32(s + k + O), vcvt_high_f32_f64(vcvt_f32_f64(DA), DB)), vld1q_f32(nz + k + O), vinv)
#define PASS(OP) \
    for (int j = 0; j < nv; j += 64) { \
        float32x4_t b0 = vdupq_n_f32(-__builtin_inff()), b1 = b0, b2 = b0, b3 = b0; \
        for (int k = j; k < j + 64 && k < nv; k += 16) { \
            UPD(D0, OP, 0) UPD(D1, OP, 2) UPD(D2, OP, 4) UPD(D3, OP, 6) \
            UPD(D4, OP, 8) UPD(D5, OP, 10) UPD(D6, OP, 12) UPD(D7, OP, 14) \
            b0 = vmaxq_f32(b0, SC(D0, D1, 0)); b1 = vmaxq_f32(b1, SC(D2, D3, 4)); \
            b2 = vmaxq_f32(b2, SC(D4, D5, 8)); b3 = vmaxq_f32(b3, SC(D6, D7, 12)); \
        } \
        float v = vmaxvq_f32(vmaxq_f32(vmaxq_f32(b0, b1), vmaxq_f32(b2, b3))); \
        if (v > bv) { bv = v; bc = j; } \
    }
    if (si > 0) { PASS(vsubq_f64) } else { PASS(vaddq_f64) }
#undef PASS
#undef UPD
    tail_pass(Ji, phi, s, nz, nv, n, si, inv, &bv, &bl);
    if (bl >= 0) return bl;
    const float32x4_t vm = vdupq_n_f32(bv);
    for (int k = bc; k < bc + 64 && k < nv; k += 4) {   // first spin of the winning chunk at the maximum
        uint32x4_t eq = vceqq_f32(SC(vld1q_f64(phi + k), vld1q_f64(phi + k + 2), 0), vm);
        if (vmaxvq_u32(eq)) { uint32_t e[4]; vst1q_u32(e, eq); return k + (e[0] ? 0 : e[1] ? 1 : e[2] ? 2 : 3); }
    }
#undef SC
    __builtin_trap();
}

// Categorical pass over one chunk of slots [j, end), 8 slots per step = 4 float64 vectors of fields -> 2 float32
// score vectors: update (subtract a row, add a row, or both), score, and return the largest score. Where blk gives the
// 8 slots one o, it is one broadcast. Otherwise (NEON has no gather): slots are grouped by variable and every variable
// has one, so the variables of 4 slots are grp[k] .. grp[k] + 3, and o[grp[k]] is one contiguous load of 4 and a byte
// table lookup with indices 4 (grp[k + i] - grp[k]) + (0, 1, 2, 3) (o has 64 entries of padding).
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
