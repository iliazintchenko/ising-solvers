// Scalar pass (included by pfa_native.c), for targets without a SIMD path. Subtracts si * Ji from phi and
// returns the next spin, the first to reach max_j(s_j phi_j + nz_j * inv).
static inline int flip_pass(const double *Ji, double *phi, const float *s, const float *nz, int n, double si, float inv) {
    float bv = -__builtin_inff(); int bl = 0;
    tail_pass(Ji, phi, s, nz, 0, n, si, inv, &bv, &bl);
    return bl;
}

// Categorical pass over whole chunks of 64 slots [0, nv), one slot at a time: update, score, and keep the first chunk
// that reaches the largest score in *bv, *bc.
static inline void cat_chunks(double *F, const double *ra, const double *rb, const int *grp, const double *o,
                              const float *g, double T, int nv, double *bv, int *bc) {
    for (int j = 0; j < nv; j += 64) {
        double v = -__builtin_inf();
        for (int k = j; k < j + 64; k++) {
            F[k] = (F[k] - ra[k]) + rb[k];
            const double sc = cat_score(F, grp, o, g, T, k);
            if (sc > v) v = sc;
        }
        if (v > *bv) { *bv = v; *bc = j; }
    }
}
