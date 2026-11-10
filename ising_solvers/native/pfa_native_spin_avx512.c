// AVX-512 pass for spins (included by pfa_native_spin.c). Chunks of 64 spins; a group of 16 spins = 2 float64 vectors
// of local fields -> 2 float32 score vectors of 8. Subtracts si * Ji from phi and returns the next spin, the first to
// reach max_j(s_j phi_j + nz_j * inv).
static inline int flip_pass(const double *Ji, double *phi, const float *s, const float *nz, int n, double si, float inv) {
    const __m256 vinv = _mm256_set1_ps(inv);
    const int nv = n & ~15;
    float bv = -__builtin_inff(); int bc = 0, bl = -1;
#define UPD(D, OP, O) __m512d D = OP(_mm512_loadu_pd(phi + k + O), _mm512_loadu_pd(Ji + k + O)); _mm512_storeu_pd(phi + k + O, D);
#define SC(D, O) _mm256_fmadd_ps(_mm256_loadu_ps(nz + k + O), vinv, _mm256_mul_ps(_mm256_loadu_ps(s + k + O), CVT512(D)))
#define PASS(OP) \
    for (int j = 0; j < nv; j += 64) { \
        __m256 b0 = _mm256_set1_ps(-__builtin_inff()), b1 = b0; \
        for (int k = j; k < j + 64 && k < nv; k += 16) { \
            UPD(D0, OP, 0) UPD(D1, OP, 8) \
            b0 = _mm256_max_ps(b0, SC(D0, 0)); b1 = _mm256_max_ps(b1, SC(D1, 8)); \
        } \
        float v = hmax8(_mm256_max_ps(b0, b1)); \
        if (v > bv) { bv = v; bc = j; } \
    }
    if (si > 0) { PASS(_mm512_sub_pd) } else { PASS(_mm512_add_pd) }
#undef PASS
#undef UPD
#undef SC
    tail_pass(Ji, phi, s, nz, nv, n, si, inv, &bv, &bl);
    if (bl >= 0) return bl;
    const __m128 vinv4 = _mm_set1_ps(inv), vm = _mm_set1_ps(bv);
    for (int k = bc; k < bc + 64 && k < nv; k += 4) {   // first spin of the winning chunk at the maximum
        __m128 w = _mm_fmadd_ps(_mm_loadu_ps(nz + k), vinv4, _mm_mul_ps(_mm_loadu_ps(s + k), _mm256_cvtpd_ps(_mm256_loadu_pd(phi + k))));
        int msk = _mm_movemask_ps(_mm_cmpeq_ps(w, vm));
        if (msk) return k + __builtin_ctz(msk);
    }
    __builtin_trap();
}
