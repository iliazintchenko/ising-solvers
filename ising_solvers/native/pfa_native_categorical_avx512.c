// AVX-512 pass for categorical variables (included by pfa_native_categorical.c), over one chunk of slots [j, end).
// 16 slots per step = 2 float64 vectors of fields -> 2 float32 score vectors of 8: update (subtract a row, add a row,
// or both), score, and return the largest score. Where blk gives 8 slots one o, it is one broadcast; otherwise
// o[grp[k]] is gathered.
static inline float cat_chunk(double *F, const float *S, const double *ra, const double *rb, const int *grp,
                              const int *blk, const float *o, const float *g, float T, int j, int end) {
    const __m256 vT = _mm256_set1_ps(T);
    __m256 b0 = _mm256_set1_ps(-__builtin_inff()), b1 = b0;
#define SUB(O) _mm512_sub_pd(_mm512_loadu_pd(F + k + O), _mm512_loadu_pd(ra + k + O))
#define ADD(O) _mm512_add_pd(_mm512_loadu_pd(F + k + O), _mm512_loadu_pd(rb + k + O))
#define BOTH(O) _mm512_add_pd(SUB(O), _mm512_loadu_pd(rb + k + O))
#define OV(O) (blk[(k + O) >> 3] >= 0 ? _mm256_broadcast_ss(o + blk[(k + O) >> 3]) \
                                      : _mm256_i32gather_ps(o, _mm256_loadu_si256((const __m256i *)(grp + k + O)), 4))
#define SC(D, O) _mm256_fmadd_ps(_mm256_loadu_ps(g + k + O), vT, \
                                 _mm256_add_ps(OV(O), _mm256_mul_ps(_mm256_loadu_ps(S + k + O), CVT512(D))))
#define CHUNK(UPD) \
    for (int k = j; k < end; k += 16) { \
        const __m512d f0 = UPD(0), f1 = UPD(8); \
        _mm512_storeu_pd(F + k, f0); _mm512_storeu_pd(F + k + 8, f1); \
        b0 = _mm256_max_ps(b0, SC(f0, 0)); b1 = _mm256_max_ps(b1, SC(f1, 8)); \
    }
    if (ra && rb) { CHUNK(BOTH) } else if (ra) { CHUNK(SUB) } else { CHUNK(ADD) }
#undef CHUNK
#undef SC
#undef OV
#undef BOTH
#undef ADD
#undef SUB
    return hmax8(_mm256_max_ps(b0, b1));
}
