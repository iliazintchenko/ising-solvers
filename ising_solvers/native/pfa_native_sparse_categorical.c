// PFA sparse categorical kernel (included by pfa_native.c): the categorical kernel of pfa_native_categorical.c for
// problems where each variable is coupled to few others. It makes the same moves: same state, scores, noise, tabu.
// Slots, F, S, o, cur and the tabu are as there; first[r] is the first slot of variable r (first[V] = m).
//   - Couplings are stored per pair of coupled variables, as full tables. The neighbours of variable r are
//     adj[vptr[r] .. vptr[r + 1]); the row of a slot k of r holds its couplings to the slots of those neighbours, one
//     neighbour after the other: vals[rows[r] + (k - first[r]) * len[r] ..), len[r] values.
//   - A move of r from value a to b changes the fields of its neighbours' slots only, by the row of b minus the row
//     of a, applied at once.
//   - The scores are a_k + noise_k / beta with a_k = o[r] + S[k] * (float32)F[k], kept in a selection tree
//     (pfa_native_tree.c). A move gives new a_k to the slots of the moved variable and of its neighbours (their o
//     changes with the field of their current slot), and to those of a variable leaving the tabu list.
//   - The best state is kept up to date from a list of the variables moved since it was last written.

// New a_k for the slots of variable r, from F, S, cur and pen.
static inline void sparse_cat_values(Tree *tree, const double *F, const float *S, const int *first, const int *cur,
                                     const double *pen, int r) SVE_STREAMING {
    const int lo = first[r], hi = first[r + 1], c = cur[r];
    if (hi - lo == 1) { tree_set(tree, lo, 0.0f + S[lo] * (float)F[lo]); return; }   // a spin: its tabu is in F
    const float o = (float)((c >= 0 ? F[c] : 0.0) - pen[r]);
    float *a = tree->a[0];
    for (int k = lo; k < hi; k++) a[k] = o + S[k] * (float)F[k];
    for (int c64 = lo >> 6; c64 <= (hi - 1) >> 6; c64++) tree_fit(tree, c64);
}

ANNEAL_ATTR static void sparse_cat_loop(const int *first, const int *vptr, const int *adj, const int64_t *rows,
                                        const double *vals, double *F, int m, int V, int *cur, long nmoves,
                                        double beta_start, double beta_max, double *E_io, double *best_io,
                                        int *best_cur, uint64_t seed, double lambda, int tenure, double *pen,
                                        long *last, int *recent, int *moved, float *S, const int *grp,
                                        const long *len, Tree *tree) {
    double E = *E_io, best = *best_io, beta = beta_start, dbeta = (beta_max - beta_start) / (double)(nmoves - 1);
    uint64_t x = seed | 1;
    int at_best = 0;     // the current state is the best one and best_cur does not hold it yet
    long nmoved = 0;     // more than V: too many to list, copy the state
#define KEEP_BEST { \
        if (nmoved > V) __builtin_memcpy(best_cur, cur, sizeof(int) * V); \
        else for (long q = 0; q < nmoved; q++) best_cur[moved[q]] = cur[moved[q]]; \
        nmoved = 0; at_best = 0; }
    for (int r = 0; r < V; r++) sparse_cat_values(tree, F, S, first, cur, pen, r);
    for (long t = 0; t < nmoves; t++) {
        beta += dbeta;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        const int offset = (int)(((x * 2685821657736338717ULL) >> 32) * (uint64_t)m >> 32);
        const int k = tree_argmax(tree, offset, (float)(1.0 / beta));
        const int r = grp[k], c = cur[r], b = k == c ? -1 : k, spin = first[r + 1] - first[r] == 1;
        if (spin) { F[k] += (double)S[k] * pen[r]; pen[r] = 0.0; }   // true field of k
        const double dE = b < 0 ? -F[k] : F[k] - (c >= 0 ? F[c] : 0.0);
        if (at_best && dE >= 0.0) KEEP_BEST   // leaving it
        E += dE; cur[r] = b;
        if (nmoved < V) moved[nmoved] = r;
        nmoved++;
        if (c >= 0) S[c] = -1.0f;
        if (b >= 0) S[b] = spin ? 1.0f : 0.0f;
        if (tenure > 0) {
            if (t >= tenure) {   // the variable moved tenure steps ago leaves the tabu list, unless it moved again since
                const int q = recent[t % tenure];
                if (last[q] == t - tenure) {
                    if (first[q + 1] - first[q] == 1) F[first[q]] += (double)S[first[q]] * pen[q];
                    pen[q] = 0.0;
                    sparse_cat_values(tree, F, S, first, cur, pen, q);
                }
            }
            pen[r] = lambda / beta; recent[t % tenure] = r; last[r] = t;
            if (spin) F[k] -= (double)S[k] * pen[r];
        }
        if (E < best) { best = E; at_best = 1; }
        const double *ra = c >= 0 ? vals + rows[r] + (c - first[r]) * len[r] : 0;   // rows of the old and new value
        const double *rb = b >= 0 ? vals + rows[r] + (b - first[r]) * len[r] : 0;
        for (int e = vptr[r]; e < vptr[r + 1]; e++) {
            const int q = adj[e], w = first[q + 1] - first[q];
            double *Fq = F + first[q];
            if (ra && rb) { for (int i = 0; i < w; i++) Fq[i] = (Fq[i] - ra[i]) + rb[i]; ra += w; rb += w; }
            else if (ra) { for (int i = 0; i < w; i++) Fq[i] -= ra[i]; ra += w; }
            else { for (int i = 0; i < w; i++) Fq[i] += rb[i]; rb += w; }
            sparse_cat_values(tree, F, S, first, cur, pen, q);
        }
        sparse_cat_values(tree, F, S, first, cur, pen, r);
    }
    if (at_best) KEEP_BEST
#undef KEEP_BEST
    for (int r = 0; r < V; r++)   // leave the true fields
        if (first[r + 1] - first[r] == 1) F[first[r]] += (double)S[first[r]] * pen[r];
    *E_io = E; *best_io = best;
}

// first: V + 1 slot offsets; vptr, adj, rows, vals: the couplings, see above; F: m fields (updated in place); cur: V
// current slots (updated in place); best_cur: V slots (in: the initial state, out: the best). Everything else as
// pfa_categorical_anneal.
void pfa_sparse_categorical_anneal(const int *first, const int *vptr, const int *adj, const int64_t *rows,
                                   const double *vals, double *F, int m, int V, int *cur, const float *noise,
                                   long nmoves, double beta_start, double beta_max, double *E_io, double *best_io,
                                   int *best_cur, uint64_t seed, double lambda, int tenure) {
    double *pen = calloc(V, sizeof(double));
    float *S = malloc(sizeof(float) * m);
    long *last = malloc(sizeof(long) * V), *len = malloc(sizeof(long) * V);
    int *recent = malloc(sizeof(int) * (tenure > 0 ? tenure : 1)), *moved = malloc(sizeof(int) * V);
    int *grp = malloc(sizeof(int) * m);
    Tree tree;
    tree_new(&tree, m, noise);
    for (int r = 0; r < V; r++) {
        last[r] = -1; len[r] = 0;
        for (int e = vptr[r]; e < vptr[r + 1]; e++) len[r] += first[adj[e] + 1] - first[adj[e]];
        for (int k = first[r]; k < first[r + 1]; k++) { grp[k] = r; S[k] = -1.0f; }
        if (cur[r] >= 0) S[cur[r]] = first[r + 1] - first[r] == 1 ? 1.0f : 0.0f;
    }
    sparse_cat_loop(first, vptr, adj, rows, vals, F, m, V, cur, nmoves, beta_start, beta_max, E_io, best_io, best_cur,
                    seed, lambda, tenure, pen, last, recent, moved, S, grp, len, &tree);
    tree_free(&tree);
    free(pen); free(S); free(last); free(len); free(recent); free(moved); free(grp);
}
