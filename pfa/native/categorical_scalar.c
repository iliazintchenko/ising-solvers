// Scalar pass for categorical variables (included by categorical.c), for targets without a SIMD path,
// over one chunk of slots [j, end), one slot at a time: update, score, and return the largest score.
static inline float cat_chunk(double *F, const float *S, const double *ra, const double *rb, const int *grp,
                              const int *blk, const float *o, const float *g, float T, int j, int end) {
    (void)blk;
    float v = -__builtin_inff();
    for (int k = j; k < end; k++) {
        F[k] = cat_update(F[k], ra, rb, k);
        const float sc = cat_score(F, S, grp, o, g, T, k);
        if (sc > v) v = sc;
    }
    return v;
}
