// NEON pass for spins (included by spin.c). Chunks of 64 spins; a group of 16 spins = 8 float64 vectors of
// local fields -> 4 float32 score vectors. Subtracts si * Ji from phi and returns the next spin, the first to reach
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
