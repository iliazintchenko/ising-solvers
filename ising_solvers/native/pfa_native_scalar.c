// Scalar pass (included by pfa_native.c), for targets without a SIMD path. Subtracts si * Ji from phi and
// returns the next spin, the first to reach max_j(s_j phi_j + nz_j * inv).
static inline int flip_pass(const double *Ji, double *phi, const float *s, const float *nz, int n, double si, float inv) {
    float bv = -__builtin_inff(); int bl = 0;
    tail_pass(Ji, phi, s, nz, 0, n, si, inv, &bv, &bl);
    return bl;
}

// Categorical pass over one chunk of slots [j, end), one slot at a time: update, score, and return the largest score.
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
