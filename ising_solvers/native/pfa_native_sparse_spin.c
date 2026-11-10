// PFA sparse spin kernel (included by pfa_native.c): the spin kernel of pfa_native_spin.c for couplings stored by
// row as (column, value) pairs. It makes the same flips: same state, same scores, same noise, same tabu.
//   - Flipping spin i changes phi_j only for the neighbours j of i: one add or subtract per stored coupling.
//   - The scores are a_j + noise_j / (4 beta) with a_j = s_j * (float32)phi_j, kept in a selection tree
//     (pfa_native_tree.c) that finds the next spin without scoring every spin. Only the neighbours, the flipped spin
//     and a spin leaving the tabu list get a new a_j.
//   - The best state is kept up to date from a list of the spins flipped since it was last written.

// pen, last, recent: as in the spin kernel. moved: spins flipped since sbest last held the state.
ANNEAL_ATTR static void sparse_loop(const int64_t *ptr, const int *idx, const double *val, double *phi, float *s,
                                    int n, long nflips, double beta_start, double beta_max, int bi, double *E_io,
                                    double *best_io, float *sbest, uint64_t seed, double lambda, int tenure,
                                    double *pen, long *last, int *recent, int *moved, Tree *tree) {
    double E = *E_io, best = *best_io, beta = beta_start, dbeta = (beta_max - beta_start) / (double)(nflips - 1);
    uint64_t x = seed | 1;
    int at_best = 0;     // the current state is the best one and sbest does not hold it yet
    long nmoved = 0;     // more than n: too many to list, copy the state
#define KEEP_BEST { \
        if (nmoved > n) __builtin_memcpy(sbest, s, sizeof(float) * n); \
        else for (long q = 0; q < nmoved; q++) sbest[moved[q]] = s[moved[q]]; \
        nmoved = 0; at_best = 0; }
    for (int j = 0; j < n; j++) tree_set(tree, j, s[j] * (float)phi[j]);
    for (long t = 0; t < nflips; t++) {
        const int i = bi; const double si = (double)s[i];
        const double phii = phi[i] + si * pen[i];   // true local field of i
        const double dE = -4.0 * si * phii;
        if (at_best && dE >= 0.0) KEEP_BEST   // leaving it
        E += dE;
        phi[i] = phii; pen[i] = 0.0; s[i] = -s[i];
        if (nmoved < n) moved[nmoved] = i;
        nmoved++;
        beta += dbeta;
        if (tenure > 0) {
            if (t >= tenure) {   // the spin flipped tenure steps ago leaves the tabu list, unless it flipped again since
                const int j = recent[t % tenure];
                if (last[j] == t - tenure) {
                    phi[j] += (double)s[j] * pen[j]; pen[j] = 0.0;
                    tree_set(tree, j, s[j] * (float)phi[j]);
                }
            }
            pen[i] = lambda * 0.25 / beta; phi[i] -= (double)s[i] * pen[i];
            recent[t % tenure] = i; last[i] = t;
        }
        if (E < best) { best = E; at_best = 1; }
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        const int offset = (int)(((x * 2685821657736338717ULL) >> 32) * (uint64_t)n >> 32);   // uniform in [0, n)
        for (int64_t e = ptr[i]; e < ptr[i + 1]; e++) {
            const int j = idx[e];
            const double p = phi[j] - val[e] * si;
            phi[j] = p;
            tree_set(tree, j, s[j] * (float)p);
        }
        tree_set(tree, i, s[i] * (float)phi[i]);
        bi = tree_argmax(tree, offset, (float)(0.25 / beta));
    }
    if (at_best) KEEP_BEST
#undef KEEP_BEST
    for (int j = 0; j < n; j++) phi[j] += (double)s[j] * pen[j];   // leave the true local fields
    *E_io = E; *best_io = best;
}

// ptr, idx, val: the couplings by row, row i in idx/val[ptr[i] .. ptr[i + 1]); sbest: n floats (in: the initial
// spins, out: the best). Everything else as pfa_native_anneal.
void pfa_sparse_anneal(const int64_t *ptr, const int *idx, const double *val, double *phi, float *s,
                       const float *noise, int n, long nflips, double beta_start, double beta_max, int bi,
                       double *E_io, double *best_io, float *sbest, uint64_t seed, double lambda, int tenure) {
    double *pen = calloc(n, sizeof(double));
    long *last = malloc(sizeof(long) * n);
    int *recent = malloc(sizeof(int) * (tenure > 0 ? tenure : 1)), *moved = malloc(sizeof(int) * n);
    Tree tree;
    tree_new(&tree, n, noise);
    for (int j = 0; j < n; j++) last[j] = -1;
    sparse_loop(ptr, idx, val, phi, s, n, nflips, beta_start, beta_max, bi, E_io, best_io, sbest, seed, lambda,
                tenure, pen, last, recent, moved, &tree);
    tree_free(&tree);
    free(pen); free(last); free(recent); free(moved);
}
