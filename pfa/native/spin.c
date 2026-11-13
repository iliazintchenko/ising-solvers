// PFA spin kernel (included by pfa.c): after flipping spin i, ONE pass over all spins updates every
// local field and finds the next spin to flip, argmax_j(-beta * dE_j + noise_j) (Gumbel-max sampling).
//
// State per spin j: phi_j = (h_j + sum_k J_jk s_k) / 2 in float64, and s_j as a float32 +-1. Then
// dE_j = -4 s_j phi_j, and:
//   - flipping spin i changes every phi_j by -s_i J_ij (s_i before the flip): one add or subtract of the
//     caller's coupling row, chosen once per flip; no spin load, no multiply, no copy of the couplings;
//   - phi_i itself does not change when spin i flips (only s_i does), and the fields never need updating;
//   - the selection score -beta * dE_j + noise_j, divided by 4 beta, is s_j phi_j + noise_j / (4 beta):
//     computed in float32 with one multiply and one fused multiply-add. Float32 changes a pick only if two
//     scores agree to about 7 digits, far below the noise scale.
// The noise is one Gumbel vector stored twice; each step reads it from a random offset (a cyclic shift), so every
// spin sees a new noise value at every step. A fixed shift of one per step would make the large noise values travel
// through the spin indices, which ties results to how the spins happen to be ordered.
// Selection keeps only a running maximum per chunk of 64 spins; afterwards the winning chunk is rescored
// with the same operations to find the first spin that reaches the maximum.
// Tabu: for the tenure flips after a spin flips (default n/4), its dE is raised by lambda temperatures at the time
// of its flip (lambda / beta_i, default lambda = 3): its rate is divided by e^lambda right after the flip and by
// more as the anneal cools. This spreads the flips over more distinct spins, mostly while the anneal is still hot;
// a flip worth many more temperatures (such as stepping back down from an expensive climb) still wins. The penalty
// is applied by shifting the spin's stored field, phi~_i = phi_i - s_i * c_i with c_i = lambda / (4 beta) at the
// time of its flip, so the pass reads nothing extra; the anneal loop adds and removes the shift, O(1) per flip.
//
// The pass is flip_pass(), one version per SIMD path in spin_<path>.c. The SIMD passes handle spins in
// groups of 16 and the last n mod 16 spins with tail_pass() below. Every path breaks argmax ties toward the smallest
// index (like numpy.argmax).

// Spins [j0, n) one at a time, same update and score as the SIMD groups; updates *bv, *bl on a strictly
// larger score.
static inline void tail_pass(const double *Ji, double *phi, const float *s, const float *nz, int j0, int n,
                             double si, float inv, float *bv, int *bl) {
    for (int j = j0; j < n; j++) {
        double p = phi[j] - Ji[j] * si;
        phi[j] = p;
        float v = FMAF(nz[j], inv, s[j] * (float)p);
        if (v > *bv) { *bv = v; *bl = j; }
    }
}

#include SPIN_PASS   // flip_pass()

// pen[j]: the shift c_j currently applied to phi[j] (0 when j is not tabu); last[j]: step of j's last flip;
// recent[t % tenure]: the spin flipped at step t.
// The lowest state is kept for each half of the flips: sbest for the first, slate for the second.
ANNEAL_ATTR static void anneal_loop(const double *J, double *phi, float *s, const float *noise, int n, long nflips,
                                    double beta_start, double beta_max, int bi, double *E_io, double *best_io,
                                    float *sbest, double *late_io, float *slate, uint64_t seed, double lambda,
                                    int tenure, double *pen, long *last, int *recent) {
    double E = *E_io, best = *best_io, beta = beta_start, dbeta = (beta_max - beta_start) / (double)(nflips - 1);
    uint64_t x = seed | 1;   // xorshift64* generator for the offsets
    int at_best = 0;         // the current state is the best one of this half and sbest does not hold it yet
    for (long t = 0; t < nflips; t++) {
        if (t == nflips / 2) {   // the second half starts: its lowest state so far is the current one
            if (at_best) __builtin_memcpy(sbest, s, sizeof(float) * n);
            *best_io = best; best = E; sbest = slate; at_best = 1;
        }
        int i = bi; double si = (double)s[i];
        double phii = phi[i] + si * pen[i];   // true local field of i
        double dE = -4.0 * si * phii;
        if (at_best && dE >= 0.0) { __builtin_memcpy(sbest, s, sizeof(float) * n); at_best = 0; }   // leaving it
        E += dE;
        phi[i] = phii; pen[i] = 0.0; s[i] = -s[i];
        beta += dbeta;
        if (tenure > 0) {
            if (t >= tenure) {   // the spin flipped tenure steps ago leaves the tabu list, unless it flipped again since
                int j = recent[t % tenure];
                if (last[j] == t - tenure) { phi[j] += (double)s[j] * pen[j]; pen[j] = 0.0; }
            }
            pen[i] = lambda * 0.25 / beta; phi[i] -= (double)s[i] * pen[i];
            recent[t % tenure] = i; last[i] = t;
        }
        if (E < best) { best = E; at_best = 1; }
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        int offset = (int)(((x * 2685821657736338717ULL) >> 32) * (uint64_t)n >> 32);   // uniform in [0, n)
        bi = flip_pass(J + (long)i * n, phi, s, noise + offset, n, si, (float)(0.25 / beta));
    }
    if (at_best) __builtin_memcpy(sbest, s, sizeof(float) * n);
    for (int j = 0; j < n; j++) phi[j] += (double)s[j] * pen[j];   // leave the true local fields
    *E_io = E; *late_io = best;
}

// J: the n x n couplings, row-major; phi: n local fields / 2 (updated in place); s: n spins as float32 +-1
// (updated in place); noise: the noise vector stored twice (2n floats). The lowest state of the first half of the
// flips: sbest (n floats, in: the initial spins) and *best_io (in: the initial energy); of the second half, the
// state it starts from included: slate and *late_io (out).
// beta rises linearly from beta_start to beta_max; seed: for the random noise offsets; tabu: penalty lambda in
// temperatures for the last tenure flips (tenure 0: no tabu).
void pfa_anneal_spin(const double *J, double *phi, float *s, const float *noise, int n, long nflips,
                     double beta_start, double beta_max, int bi, double *E_io, double *best_io, float *sbest,
                     double *late_io, float *slate, uint64_t seed, double lambda, int tenure) {
    double *pen = calloc(n, sizeof(double));
    long *last = malloc(sizeof(long) * n);
    int *recent = malloc(sizeof(int) * (tenure > 0 ? tenure : 1));
    for (int j = 0; j < n; j++) last[j] = -1;
    anneal_loop(J, phi, s, noise, n, nflips, beta_start, beta_max, bi, E_io, best_io, sbest, late_io, slate, seed, lambda,
                tenure, pen, last, recent);
    free(pen); free(last); free(recent);
}
