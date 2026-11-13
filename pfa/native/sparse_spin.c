// PFA sparse spin kernel (included by pfa.c): the anneal of spin.c for couplings stored by row as
// (column, value) pairs, with the same schedule, fields, tabu and bookkeeping.
//   - Flipping spin i changes phi_j only for the neighbours j of i: one add or subtract per stored coupling.
//   - The next spin is drawn from a tree of sums (sums.c) with probability proportional to
//     exp(beta x_j), x_j = 4 s_j phi_j = -dE_j, the tabu penalty folded into phi as in the spin kernel. Only the
//     neighbours, the flipped spin and a spin leaving the tabu list get a new x_j.
//   - The lowest state of each half is kept up to date from a list of the spins flipped since it was last written.

// pen, last, recent: as in the spin kernel. moved: spins flipped since sbest last held the state.
static void sparse_loop(const int64_t *ptr, const int *idx, const double *val, double *phi, float *s, int n,
                        long nflips, double beta_start, double beta_max, double *E_io, double *best_io, float *sbest,
                        double *late_io, float *slate, uint64_t seed, double lambda, int tenure, double *pen,
                        long *last, int *recent, int *moved, Sums *sums) {
    double E = *E_io, best = *best_io, beta = beta_start, dbeta = (beta_max - beta_start) / (double)(nflips - 1);
    uint64_t rng = seed | 1;
    int at_best = 0;     // the current state is the best one of this half and sbest does not hold it yet
    long nmoved = 0;     // more than n: too many to list, copy the state
#define KEEP_BEST { \
        if (nmoved > n) __builtin_memcpy(sbest, s, sizeof(float) * n); \
        else for (long q = 0; q < nmoved; q++) sbest[moved[q]] = s[moved[q]]; \
        nmoved = 0; at_best = 0; }
#define NEW_X(J) { sums->x[J] = 4.0 * (double)s[J] * phi[J]; sums_refresh(sums, J, (J) + 1); }
    for (int j = 0; j < n; j++) sums->x[j] = 4.0 * (double)s[j] * phi[j];
    for (long t = 0; t < nflips; t++) {
        if (t == nflips / 2) {   // the second half starts: its lowest state so far is the current one
            if (at_best) KEEP_BEST
            *best_io = best; best = E; sbest = slate;
            __builtin_memcpy(sbest, s, sizeof(float) * n); nmoved = 0;
        }
        beta += dbeta;
        const int i = sums_draw(sums, beta, &rng);
        const double si = (double)s[i];
        const double phii = phi[i] + si * pen[i];   // true local field of i
        const double dE = -4.0 * si * phii;
        if (at_best && dE >= 0.0) KEEP_BEST   // leaving it
        E += dE;
        phi[i] = phii; pen[i] = 0.0; s[i] = -s[i];
        if (nmoved < n) moved[nmoved] = i;
        nmoved++;
        if (tenure > 0) {
            if (t >= tenure) {   // the spin flipped tenure steps ago leaves the tabu list, unless it flipped again since
                const int j = recent[t % tenure];
                if (last[j] == t - tenure) {
                    phi[j] += (double)s[j] * pen[j]; pen[j] = 0.0;
                    NEW_X(j)
                }
            }
            pen[i] = lambda * 0.25 / beta; phi[i] -= (double)s[i] * pen[i];
            recent[t % tenure] = i; last[i] = t;
        }
        if (E < best) { best = E; at_best = 1; }
        for (int64_t e = ptr[i]; e < ptr[i + 1]; e++) {
            const int j = idx[e];
            phi[j] -= val[e] * si;
            sums->x[j] = 4.0 * (double)s[j] * phi[j];
        }
        sums_refresh_at(sums, idx + ptr[i], (int)(ptr[i + 1] - ptr[i]));
        NEW_X(i)
    }
    if (at_best) KEEP_BEST
#undef KEEP_BEST
#undef NEW_X
    for (int j = 0; j < n; j++) phi[j] += (double)s[j] * pen[j];   // leave the true local fields
    *E_io = E; *late_io = best;
}

// ptr, idx, val: the couplings by row, row i in idx/val[ptr[i] .. ptr[i + 1]); seed: for the draws. No noise buffer
// and no first spin; everything else as pfa_anneal_spin.
void pfa_anneal_sparse_spin(const int64_t *ptr, const int *idx, const double *val, double *phi, float *s, int n,
                            long nflips, double beta_start, double beta_max, double *E_io, double *best_io,
                            float *sbest, double *late_io, float *slate, uint64_t seed, double lambda, int tenure) {
    double *pen = calloc(n, sizeof(double));
    long *last = malloc(sizeof(long) * n);
    int *recent = malloc(sizeof(int) * (tenure > 0 ? tenure : 1)), *moved = malloc(sizeof(int) * n);
    Sums sums;
    sums_new(&sums, n);
    for (int j = 0; j < n; j++) last[j] = -1;
    sparse_loop(ptr, idx, val, phi, s, n, nflips, beta_start, beta_max, E_io, best_io, sbest, late_io, slate, seed,
                lambda, tenure, pen, last, recent, moved, &sums);
    sums_free(&sums);
    free(pen); free(last); free(recent); free(moved);
}
