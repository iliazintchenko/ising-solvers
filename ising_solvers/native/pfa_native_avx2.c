// AVX2 pass (included by pfa_native.c). Chunks of 64 spins; a group of 16 spins = 4 float64 vectors of local
// fields -> 2 float32 score vectors of 8. Subtracts si * Ji from phi and returns the next spin, the first to reach
// max_j(s_j phi_j + nz_j * inv). Uses fused multiply-adds when built with FMA (any AVX2 CPU has it).
#if defined(__FMA__) || defined(TEST_AVX2_SIMDE)
#define FMA8(A, B, C) _mm256_fmadd_ps(A, B, C)
#define FMA4(A, B, C) _mm_fmadd_ps(A, B, C)
#else
#define FMA8(A, B, C) _mm256_add_ps(_mm256_mul_ps(A, B), C)
#define FMA4(A, B, C) _mm_add_ps(_mm_mul_ps(A, B), C)
#endif
static inline float hmax8(__m256 x) {
    __m128 m = _mm_max_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    return _mm_cvtss_f32(_mm_max_ss(m, _mm_shuffle_ps(m, m, 1)));
}

static inline int flip_pass(const double *Ji, double *phi, const float *s, const float *nz, int n, double si, float inv) {
    const __m256 vinv = _mm256_set1_ps(inv);
    const int nv = n & ~15;
    float bv = -__builtin_inff(); int bc = 0, bl = -1;
#define UPD(D, OP, O) __m256d D = OP(_mm256_loadu_pd(phi + k + O), _mm256_loadu_pd(Ji + k + O)); _mm256_storeu_pd(phi + k + O, D);
#define CV8(DA, DB) _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(DA)), _mm256_cvtpd_ps(DB), 1)
#define SC(DA, DB, O) FMA8(_mm256_loadu_ps(nz + k + O), vinv, _mm256_mul_ps(_mm256_loadu_ps(s + k + O), CV8(DA, DB)))
#define PASS(OP) \
    for (int j = 0; j < nv; j += 64) { \
        __m256 b0 = _mm256_set1_ps(-__builtin_inff()), b1 = b0; \
        for (int k = j; k < j + 64 && k < nv; k += 16) { \
            UPD(D0, OP, 0) UPD(D1, OP, 4) UPD(D2, OP, 8) UPD(D3, OP, 12) \
            b0 = _mm256_max_ps(b0, SC(D0, D1, 0)); b1 = _mm256_max_ps(b1, SC(D2, D3, 8)); \
        } \
        float v = hmax8(_mm256_max_ps(b0, b1)); \
        if (v > bv) { bv = v; bc = j; } \
    }
    if (si > 0) { PASS(_mm256_sub_pd) } else { PASS(_mm256_add_pd) }
#undef PASS
#undef UPD
#undef SC
#undef CV8
    tail_pass(Ji, phi, s, nz, nv, n, si, inv, &bv, &bl);
    if (bl >= 0) return bl;
    const __m128 vinv4 = _mm_set1_ps(inv), vm = _mm_set1_ps(bv);
    for (int k = bc; k < bc + 64 && k < nv; k += 4) {   // first spin of the winning chunk at the maximum
        __m128 w = FMA4(_mm_loadu_ps(nz + k), vinv4, _mm_mul_ps(_mm_loadu_ps(s + k), _mm256_cvtpd_ps(_mm256_loadu_pd(phi + k))));
        int msk = _mm_movemask_ps(_mm_cmpeq_ps(w, vm));
        if (msk) return k + __builtin_ctz(msk);
    }
    __builtin_trap();
}

// Categorical pass over one chunk of slots [j, end), 8 slots per step = 2 float64 vectors of fields -> 1 float32
// score vector: update (subtract a row, add a row, or both), score, and return the largest score. Where blk gives the
// 8 slots one o, it is one broadcast; otherwise o[grp[k]] is gathered.
static inline float cat_chunk(double *F, const float *S, const double *ra, const double *rb, const int *grp,
                              const int *blk, const float *o, const float *g, float T, int j, int end) {
    const __m256 vT = _mm256_set1_ps(T);
    __m256 b = _mm256_set1_ps(-__builtin_inff());
#define SUB(O) _mm256_sub_pd(_mm256_loadu_pd(F + k + O), _mm256_loadu_pd(ra + k + O))
#define ADD(O) _mm256_add_pd(_mm256_loadu_pd(F + k + O), _mm256_loadu_pd(rb + k + O))
#define BOTH(O) _mm256_add_pd(SUB(O), _mm256_loadu_pd(rb + k + O))
#define CHUNK(UPD) \
    for (int k = j; k < end; k += 8) { \
        const int v = blk[k >> 3]; \
        const __m256 oo = v >= 0 ? _mm256_broadcast_ss(o + v) \
                                 : _mm256_i32gather_ps(o, _mm256_loadu_si256((const __m256i *)(grp + k)), 4); \
        const __m256d f0 = UPD(0), f1 = UPD(4); \
        _mm256_storeu_pd(F + k, f0); _mm256_storeu_pd(F + k + 4, f1); \
        const __m256 d = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(f0)), _mm256_cvtpd_ps(f1), 1); \
        b = _mm256_max_ps(b, FMA8(_mm256_loadu_ps(g + k), vT, _mm256_add_ps(oo, _mm256_mul_ps(_mm256_loadu_ps(S + k), d)))); \
    }
    if (ra && rb) { CHUNK(BOTH) } else if (ra) { CHUNK(SUB) } else { CHUNK(ADD) }
#undef CHUNK
#undef BOTH
#undef ADD
#undef SUB
    return hmax8(b);
}
