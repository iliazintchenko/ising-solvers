// NEON block for the selection tree (included by pfa_native_tree.c): u[k] = a[k] + g[k] * T for 64 k, 16 per step.
// Returns the largest u[k]; *amax is the largest a[k]. tree_mask(): bit k is set where u[k] >= v.
static inline float tree_block(const float *a, const float *g, float T, float *u, float *amax) {
    const float32x4_t vT = vdupq_n_f32(T);
    float32x4_t b0 = vdupq_n_f32(-__builtin_inff()), b1 = b0, b2 = b0, b3 = b0, m0 = b0, m1 = b0, m2 = b0, m3 = b0;
    for (int k = 0; k < 64; k += 16) {
        const float32x4_t a0 = vld1q_f32(a + k), a1 = vld1q_f32(a + k + 4), a2 = vld1q_f32(a + k + 8), a3 = vld1q_f32(a + k + 12);
        const float32x4_t u0 = vfmaq_f32(a0, vld1q_f32(g + k), vT), u1 = vfmaq_f32(a1, vld1q_f32(g + k + 4), vT);
        const float32x4_t u2 = vfmaq_f32(a2, vld1q_f32(g + k + 8), vT), u3 = vfmaq_f32(a3, vld1q_f32(g + k + 12), vT);
        vst1q_f32(u + k, u0); vst1q_f32(u + k + 4, u1); vst1q_f32(u + k + 8, u2); vst1q_f32(u + k + 12, u3);
        b0 = vmaxq_f32(b0, u0); b1 = vmaxq_f32(b1, u1); b2 = vmaxq_f32(b2, u2); b3 = vmaxq_f32(b3, u3);
        m0 = vmaxq_f32(m0, a0); m1 = vmaxq_f32(m1, a1); m2 = vmaxq_f32(m2, a2); m3 = vmaxq_f32(m3, a3);
    }
    *amax = vmaxvq_f32(vmaxq_f32(vmaxq_f32(m0, m1), vmaxq_f32(m2, m3)));
    return vmaxvq_f32(vmaxq_f32(vmaxq_f32(b0, b1), vmaxq_f32(b2, b3)));
}

static inline uint64_t tree_mask(const float *u, float v) {
    static const uint8_t bit[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    const float32x4_t vv = vdupq_n_f32(v);
    const uint8x16_t vbit = vld1q_u8(bit);
    uint64_t m = 0;
#define GE(O) vmovn_u32(vcgeq_f32(vld1q_f32(u + k + O), vv))
    for (int k = 0; k < 64; k += 16) {   // 16 compares -> 16 bytes -> 2 bytes of bits
        const uint8x16_t b = vandq_u8(vcombine_u8(vmovn_u16(vcombine_u16(GE(0), GE(4))), vmovn_u16(vcombine_u16(GE(8), GE(12)))), vbit);
        m |= ((uint64_t)vaddv_u8(vget_low_u8(b)) | (uint64_t)vaddv_u8(vget_high_u8(b)) << 8) << k;
    }
#undef GE
    return m;
}
