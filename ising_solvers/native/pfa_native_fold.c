// PFA fold (included by pfa_native.c): state a takes from state b every part that b has lower.
// The variables on which a and b differ are put in groups: two of them share a group when their choices interact,
// that is when taking both from b does not change the energy by the sum of taking each alone. For spins i, j the
// difference is 4 J_ij a_i a_j, nonzero whenever the coupling is; for variables r, s with values a_r, b_r and
// a_s, b_s it is
//     T(a_r, a_s) - T(a_r, b_s) - T(b_r, a_s) + T(b_r, b_s)
// with T the couplings between their values. Variables of different groups do not interact, so the energy change
// of taking a group from b is the same whatever is taken elsewhere:
//     sum over the group of (the change of taking that variable alone) + sum over its pairs of the difference above
// and every group with a negative change is taken. The result is the lowest state that has each group from a or
// from b. An interaction counts when it is not exactly 0: one that is nonzero by rounding only joins two groups,
// which loses a choice and nothing else.
// Each function writes the result into a and returns the energy change, 0 or less.

static inline int fold_root(int *up, int p) {
    while (up[p] != p) { up[p] = up[up[p]]; p = up[p]; }
    return p;
}

static inline void fold_join(int *up, int p, int q) {
    p = fold_root(up, p); q = fold_root(up, q);
    if (p != q) up[q] = p;
}

// delta[p]: the change of taking differing variable p alone plus its interactions with the later ones; up: the
// groups. Sets take[p] for the members of the groups whose total is negative and returns the sum of those totals.
static double fold_groups(int nd, int *up, const double *delta, double *sum, char *take) {
    double gain = 0.0;
    for (int p = 0; p < nd; p++) sum[p] = 0.0;
    for (int p = 0; p < nd; p++) sum[fold_root(up, p)] += delta[p];
    for (int p = 0; p < nd; p++) {
        take[p] = sum[fold_root(up, p)] < 0.0;
        if (up[p] == p && take[p]) gain += sum[p];
    }
    return gain;
}

#define FOLD_NEW(N) \
    int *d = malloc(sizeof(int) * (N)), *up = malloc(sizeof(int) * (N)), nd = 0; \
    double *delta = malloc(sizeof(double) * (N)), *sum = malloc(sizeof(double) * (N)); \
    char *take = malloc(N);
#define FOLD_FREE free(d); free(up); free(delta); free(sum); free(take);

// Spins, couplings as an n x n array: E = s J s / 2 + h s.
double pfa_spin_fold(const double *J, const double *h, int n, float *a, const float *b) {
    FOLD_NEW(n)
    for (int i = 0; i < n; i++) if (a[i] != b[i]) { up[nd] = nd; d[nd++] = i; }
    for (int p = 0; p < nd; p++) {
        const int i = d[p];
        const double *Ji = J + (long)i * n, ai = (double)a[i];
        double f = h[i];
        for (int j = 0; j < n; j++) f += Ji[j] * (double)a[j];
        delta[p] = -2.0 * ai * f;   // flipping i alone
        for (int q = p + 1; q < nd; q++) {
            const int j = d[q];
            if (Ji[j] != 0.0) { delta[p] += 4.0 * Ji[j] * ai * (double)a[j]; fold_join(up, p, q); }
        }
    }
    const double gain = fold_groups(nd, up, delta, sum, take);
    for (int p = 0; p < nd; p++) if (take[p]) a[d[p]] = b[d[p]];
    FOLD_FREE
    return gain;
}

// Spins, couplings by row as in pfa_sparse_anneal.
double pfa_sparse_spin_fold(const int64_t *ptr, const int *idx, const double *val, const double *h, int n, float *a,
                            const float *b) {
    FOLD_NEW(n)
    int *pos = malloc(sizeof(int) * n);   // place of a differing spin in d, else -1
    for (int i = 0; i < n; i++) {
        pos[i] = -1;
        if (a[i] != b[i]) { pos[i] = nd; up[nd] = nd; d[nd++] = i; }
    }
    for (int p = 0; p < nd; p++) {
        const int i = d[p];
        const double ai = (double)a[i];
        double f = h[i], pairs = 0.0;
        for (int64_t e = ptr[i]; e < ptr[i + 1]; e++) {
            const int j = idx[e];
            f += val[e] * (double)a[j];
            if (pos[j] > p && val[e] != 0.0) { pairs += 4.0 * val[e] * ai * (double)a[j]; fold_join(up, p, pos[j]); }
        }
        delta[p] = -2.0 * ai * f + pairs;
    }
    const double gain = fold_groups(nd, up, delta, sum, take);
    for (int p = 0; p < nd; p++) if (take[p]) a[d[p]] = b[d[p]];
    free(pos);
    FOLD_FREE
    return gain;
}

// Categorical variables, with the slots and the couplings read as in pfa_native_eliminate.c (both layouts); a, b:
// the slot of each variable's value, -1 at value 0.
double pfa_categorical_fold(const int *first, int V, const int *vptr, const int *adj, const int64_t *seg,
                            const int64_t *rows, const int64_t *len, const double *vals, const double *fields, int *a,
                            const int *b) {
    FOLD_NEW(V)
    int *pos = malloc(sizeof(int) * V);   // place of a differing variable in d, else -1
    for (int r = 0; r < V; r++) {
        pos[r] = -1;
        if (a[r] != b[r]) { pos[r] = nd; up[nd] = nd; d[nd++] = r; }
    }
    for (int p = 0; p < nd; p++) {
        const int r = d[p], ar = a[r], br = b[r];
        const double *La = ar >= 0 ? vals + rows[r] + (int64_t)(ar - first[r]) * len[r] : 0;   // couplings of a_r
        const double *Lb = br >= 0 ? vals + rows[r] + (int64_t)(br - first[r]) * len[r] : 0;   // and of b_r
        double fa = ar >= 0 ? fields[ar] : 0.0, fb = br >= 0 ? fields[br] : 0.0, pairs = 0.0;
        for (int e = vptr[r]; e < vptr[r + 1]; e++) {
            const int s = adj[e], as = a[s], bs = b[s];
            const int64_t at = seg[e] - first[s];
            const double Taa = La && as >= 0 ? La[at + as] : 0.0, Tba = Lb && as >= 0 ? Lb[at + as] : 0.0;
            fa += Taa; fb += Tba;   // the fields of a_r and b_r in state a
            if (pos[s] > p) {
                const double Tab = La && bs >= 0 ? La[at + bs] : 0.0, Tbb = Lb && bs >= 0 ? Lb[at + bs] : 0.0;
                const double both = ((Taa - Tab) - Tba) + Tbb;
                if (both != 0.0) { pairs += both; fold_join(up, p, pos[s]); }
            }
        }
        delta[p] = (fb - fa) + pairs;
    }
    const double gain = fold_groups(nd, up, delta, sum, take);
    for (int p = 0; p < nd; p++) if (take[p]) a[d[p]] = b[d[p]];
    free(pos);
    FOLD_FREE
    return gain;
}
#undef FOLD_NEW
#undef FOLD_FREE
