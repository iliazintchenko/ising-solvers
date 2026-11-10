// Scalar pass for spins (included by pfa_native.c), for targets without a SIMD path. Subtracts si * Ji from phi and
// returns the next spin, the first to reach max_j(s_j phi_j + nz_j * inv).
static inline int flip_pass(const double *Ji, double *phi, const float *s, const float *nz, int n, double si, float inv) {
    float bv = -__builtin_inff(); int bl = 0;
    tail_pass(Ji, phi, s, nz, 0, n, si, inv, &bv, &bl);
    return bl;
}
