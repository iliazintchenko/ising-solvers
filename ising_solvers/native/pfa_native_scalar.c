// Scalar pass (included by pfa_native.c), for targets without a SIMD path. Updates dE with row Ji of the flipped
// spin (sign si) and returns the next spin, the first to reach max_j(nb * dE_j + nz_j).
static inline int flip_pass(const double *Ji, const double *s, double *dE, const float *nz, int np, double si, float nb) {
    float bv = -__builtin_inff(); int bl = 0;
    for (int j = 0; j < np; j++) {
        double d = dE[j] + Ji[j] * s[j] * si;
        dE[j] = d;
        float v = nb * (float)d + nz[j];
        if (v > bv) { bv = v; bl = j; }
    }
    return bl;
}
