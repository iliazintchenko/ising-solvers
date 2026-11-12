// PFA dead-end elimination for categorical problems (included by pfa_native.c): removes values that no optimum
// needs, before the anneal. Value a of variable r is removed when another allowed value b of r has
//     d(a, b) = f(a) - f(b) + sum over the neighbours s of r of  min over the allowed c of s of (T(a, c) - T(b, c))  >=  0
// with f the fields and T the couplings: b is then at least as good as a whatever the other variables do, so every
// state that uses a has one of no higher energy that does not, and an optimum always remains. Removing a value of s
// takes it out of the minima of the neighbours of s, which can make more of their values removable, so the
// variables whose neighbours lost a value are tested again until nothing changes.
//   - Slots and first[] are as in pfa_native_sparse_categorical.c: value v > 0 of variable r is slot
//     first[r] + v - 1, value 0 has no slot, no field and no couplings. alive[first[r] + r + v] says whether value v
//     of r is allowed, v = 0 .. first[r + 1] - first[r].
//   - Couplings are read as there: the neighbours of r are adj[vptr[r] .. vptr[r + 1]), the line of a slot k of r is
//     vals[rows[r] + (k - first[r]) * len[r] ..), and its couplings to the slots of neighbour adj[e] start at seg[e]
//     in that line. An n x n array is the same with rows[r] = first[r] * n, len[r] = n and seg[e] = first[adj[e]].
//   - d(a, b) is summed so that a pair that fails is dropped early. It starts at E(a) - E(b), the energies of the
//     two values with every neighbour at its lowest allowed value: one sum per value, not per pair. Each neighbour
//     then lowers it from its term in that state to its minimum. Once below 0 it stays there, so the pair is
//     dropped, and the neighbour that took it there is read first from then on.
// alive is set here. Returns the number of values removed.
#include <math.h>

// min over the allowed values c of variable s of (T(a, c) - T(b, c)) - u, from the two lines' couplings to the slots
// of s. With u that difference at one allowed value, the result is 0 or less.
static inline double elim_min(const double *A, const double *B, const double *pen, const int *first, int s, double u) {
    const double *P = pen + first[s] + s;
    const int n = first[s + 1] - first[s];
    double m0 = P[0] - u, m1 = INFINITY, m2 = INFINITY, m3 = INFINITY;   // P[0]: value 0 of s, which has no couplings
    int c = 0;
    for (; c + 4 <= n; c += 4) {
        const double t0 = (A[c] - B[c]) + P[c + 1] - u, t1 = (A[c + 1] - B[c + 1]) + P[c + 2] - u;
        const double t2 = (A[c + 2] - B[c + 2]) + P[c + 3] - u, t3 = (A[c + 3] - B[c + 3]) + P[c + 4] - u;
        m0 = t0 < m0 ? t0 : m0; m1 = t1 < m1 ? t1 : m1; m2 = t2 < m2 ? t2 : m2; m3 = t3 < m3 ? t3 : m3;
    }
    for (; c < n; c++) { const double t = (A[c] - B[c]) + P[c + 1] - u; m0 = t < m0 ? t : m0; }
    m0 = m1 < m0 ? m1 : m0; m2 = m3 < m2 ? m3 : m2;
    return m2 < m0 ? m2 : m0;
}

long pfa_categorical_eliminate(const int *first, int V, const int *vptr, const int *adj, const int64_t *seg,
                               const int64_t *rows, const int64_t *len, const double *vals, const double *fields,
                               char *alive) {
    const int m = first[V];
    int qmax = 0;
    int64_t lmax = 0;
    for (int r = 0; r < V; r++) {
        if (first[r + 1] - first[r] + 1 > qmax) qmax = first[r + 1] - first[r] + 1;
        if (len[r] > lmax) lmax = len[r];
    }
    double *pen = calloc(m + V, sizeof(double));       // 0 for an allowed value, infinity for a removed one
    double *zero = calloc(lmax + 1, sizeof(double));   // the line of value 0
    double *E = malloc(sizeof(double) * qmax);         // energy of each allowed value of r, neighbours at their lowest
    int *val = malloc(sizeof(int) * qmax);             // the allowed values of r
    int *low = calloc(V, sizeof(int));                 // the lowest allowed value of each variable
    int *ord = malloc(sizeof(int) * (vptr[V] + 1));    // the neighbours of each variable in the order they are read
    char *dirty = malloc(V);
    for (int i = 0; i < m + V; i++) alive[i] = 1;
    for (int e = 0; e < vptr[V]; e++) ord[e] = e;
    for (int r = 0; r < V; r++) dirty[r] = 1;
    long removed = 0;
    for (int again = 1; again;) {
        again = 0;
        for (int r = 0; r < V; r++) {
            if (!dirty[r]) continue;
            dirty[r] = 0;
            const int lo = first[r], q = first[r + 1] - lo + 1, base = lo + r, e0 = vptr[r], e1 = vptr[r + 1];
            int k = 0, gone = 0;
            for (int v = 0; v < q; v++) if (alive[base + v]) val[k++] = v;
            if (k < 2) continue;
            for (int i = 0; i < k; i++) {
                const int v = val[i];
                const double *L = v ? vals + rows[r] + (int64_t)(v - 1) * len[r] : zero;
                double e = v ? fields[lo + v - 1] : 0.0;
                for (int j = e0; j < e1; j++) { const int c = low[adj[j]]; if (c) e += L[seg[j] + c - 1]; }
                E[i] = e;
            }
            for (int ia = k - 1; ia >= 0; ia--) {   // highest value first, so of equal values the lowest stays
                const int a = val[ia];
                const double *La = a ? vals + rows[r] + (int64_t)(a - 1) * len[r] : zero;
                for (int ib = 0; ib < k; ib++) {
                    const int b = val[ib];
                    double d = E[ia] - E[ib];
                    if (ib == ia || !alive[base + b] || d < 0.0) continue;
                    const double *Lb = b ? vals + rows[r] + (int64_t)(b - 1) * len[r] : zero;
                    for (int p = e0; p < e1 && d >= 0.0; p++) {
                        const int j = ord[p], c = low[adj[j]];
                        d += elim_min(La + seg[j], Lb + seg[j], pen, first, adj[j], c ? La[seg[j] + c - 1] - Lb[seg[j] + c - 1] : 0.0);
                        if (d < 0.0) { ord[p] = ord[e0]; ord[e0] = j; }
                    }
                    if (d >= 0.0) {
                        alive[base + a] = 0; pen[base + a] = INFINITY; removed++; gone = 1;
                        while (!alive[base + low[r]]) low[r]++;
                        break;
                    }
                }
            }
            if (gone) { again = 1; for (int j = e0; j < e1; j++) dirty[adj[j]] = 1; }
        }
    }
    free(pen); free(zero); free(E); free(val); free(low); free(ord); free(dirty);
    return removed;
}
