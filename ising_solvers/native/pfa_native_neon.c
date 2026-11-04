// NEON pass (included by pfa_native.c). Chunks of 64 spins; a group of 16 spins = 8 float64 vectors of delta
// energies -> 4 float32 score vectors. Updates dE with row Ji of the flipped spin (sign si) and returns the next
// spin, the first to reach max_j(nb * dE_j + nz_j).
static inline int flip_pass(const double *Ji, const double *s, double *dE, const float *nz, int np, double si, float nb) {
    const float32x4_t vnb = vdupq_n_f32(nb);
    float bv = -__builtin_inff(); int bc = 0;
#define UPD(D, OP, O) float64x2_t D = OP(vld1q_f64(dE + k + O), vmulq_f64(vld1q_f64(Ji + k + O), vld1q_f64(s + k + O))); \
    vst1q_f64(dE + k + O, D);
#define SC(DA, DB, O) vaddq_f32(vmulq_f32(vnb, vcvt_high_f32_f64(vcvt_f32_f64(DA), DB)), vld1q_f32(nz + k + O))
#define PASS(OP) \
    for (int j = 0; j < np; j += 64) { \
        float32x4_t b0 = vdupq_n_f32(-__builtin_inff()), b1 = b0, b2 = b0, b3 = b0; \
        for (int k = j; k < j + 64 && k < np; k += 16) { \
            UPD(D0, OP, 0) UPD(D1, OP, 2) UPD(D2, OP, 4) UPD(D3, OP, 6) \
            UPD(D4, OP, 8) UPD(D5, OP, 10) UPD(D6, OP, 12) UPD(D7, OP, 14) \
            b0 = vmaxq_f32(b0, SC(D0, D1, 0)); b1 = vmaxq_f32(b1, SC(D2, D3, 4)); \
            b2 = vmaxq_f32(b2, SC(D4, D5, 8)); b3 = vmaxq_f32(b3, SC(D6, D7, 12)); \
        } \
        float v = vmaxvq_f32(vmaxq_f32(vmaxq_f32(b0, b1), vmaxq_f32(b2, b3))); \
        if (v > bv) { bv = v; bc = j; } \
    }
    if (si < 0) { PASS(vsubq_f64) } else { PASS(vaddq_f64) }
#undef PASS
#undef UPD
    const float32x4_t vm = vdupq_n_f32(bv);
    for (int k = bc; k < bc + 64 && k < np; k += 4) {   // first spin of the winning chunk at the maximum
        uint32x4_t eq = vceqq_f32(SC(vld1q_f64(dE + k), vld1q_f64(dE + k + 2), 0), vm);
        if (vmaxvq_u32(eq)) { uint32_t e[4]; vst1q_u32(e, eq); return k + (e[0] ? 0 : e[1] ? 1 : e[2] ? 2 : 3); }
    }
#undef SC
    __builtin_trap();
}
