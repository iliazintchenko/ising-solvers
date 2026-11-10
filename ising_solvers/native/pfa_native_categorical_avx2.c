// AVX2 pass for categorical variables (included by pfa_native.c), over one chunk of slots [j, end). 8 slots per
// step = 2 float64 vectors of fields -> 1 float32 score vector: update (subtract a row, add a row, or both), score,
// and return the largest score. Where blk gives the 8 slots one o, it is one broadcast; otherwise o[grp[k]] is
// gathered.
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
