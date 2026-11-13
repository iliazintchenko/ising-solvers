// PFA categorical (Potts) kernel (included by pfa.c). Variable r takes one of q_r >= 2 values; value 0 is its
// reference, and each other value owns a slot k, with grp[k] = r (slots grouped by variable). F[k] is the energy of
// slot k's value minus that of value 0, given the other variables' current values, and Jt (m x m, symmetric, zero
// within a variable) holds the couplings between slots. cur[r] is the slot of r's value, or -1 at value 0.
// Every slot is one move, scored (o[r] + S[k] F[k]) + g[k] / beta = -dE - pen[r] + noise / beta:
//   - slot k is not r's value: r moves to k, dE = F[k] - F[cur[r]] (F[-1] = 0): S[k] = -1, o[r] = F[cur[r]] - pen[r];
//   - slot k is r's value: r moves back to value 0, dE = -F[k]: S[k] = 0 and o[r] as above, or S[k] = +1 and o[r]
//     without F[cur[r]] when r has two values.
// So a variable with two values is a spin: its score is S[k] F[k] - pen[r], and its tabu is folded into the stored
// field as for spins, F~[k] = F[k] - S[k] pen[r], which leaves o[r] = 0. Only variables with more values need o, the
// field of their current slot.
// A move from value a to b changes every field by Jt[b] - Jt[a], with no row for value 0: one or two rows, applied in
// the next step's pass. The pass updates F and scores every slot, in float32 as for spins, and keeps the maximum per
// chunk of 64 slots; the winning chunk is then rescored with the same operations for its first slot at the maximum
// (the last m mod 16 slots are handled one at a time).
// blk[k / 8] tells the pass where 8 slots share one o: the variable they all belong to, or V (o[V] = 0) when all
// belong to two-value variables; -1 otherwise, and the pass then reads o[grp[k]] slot by slot.
//
// The pass over one chunk of slots is cat_chunk(), one version per SIMD path in categorical_<path>.c;
// cat_pass() below runs it over all chunks. Noise, schedule, tabu and tie-breaking are as in the spin kernel.

// Categorical pass pieces, one slot at a time. Update: F[k] minus ra[k] if there is a row ra, plus rb[k] if there is
// a row rb. Score, in float32 as for spins: (o[grp[k]] + S[k] * F[k]) + g[k] * T, the last multiply-add fused. Every
// path does these operations in this order, so all paths give bit-identical results.
static inline double cat_update(double f, const double *ra, const double *rb, int k) {
    if (ra) f -= ra[k];
    if (rb) f += rb[k];
    return f;
}
static inline float cat_score(const double *F, const float *S, const int *grp, const float *o, const float *g,
                              float T, int k) {
    return FMAF(g[k], T, o[grp[k]] + S[k] * (float)F[k]);
}

#include CATEGORICAL_PASS   // cat_chunk()

// The o of variables with more than two values is set as the pass reaches their slots (multi: those variables in
// order, mfirst: their first slots), from the field of their current slot before its update: the two rows are then
// read where the pass is about to read them.
static inline int cat_pass(double *F, const float *S, const double *ra, const double *rb, const int *grp,
                           const int *blk, float *o, const float *g, float T, int m, const int *cur,
                           const double *pen, const int *multi, const int *mfirst, int nmulti) SVE_STREAMING {
    const int nv = m & ~15; float bv = -__builtin_inff(); int bc = 0, bl = -1, i = 0;   // the passes take 8 or 16 slots
#define SET_O(END) \
    for (; i < nmulti && mfirst[i] < (END); i++) { \
        const int r = multi[i], c = cur[r], k = c >= 0 ? c : mfirst[i]; \
        o[r] = (float)((c >= 0 ? 1.0 : 0.0) * cat_update(F[k], ra, rb, k) - pen[r]); \
    }
    for (int j = 0; j < nv; j += 64) {
        const int end = j + 64 < nv ? j + 64 : nv;
        SET_O(end)
        const float v = cat_chunk(F, S, ra, rb, grp, blk, o, g, T, j, end);
        if (v > bv) { bv = v; bc = j; }
    }
    SET_O(m)
#undef SET_O
    for (int k = nv; k < m; k++) {
        F[k] = cat_update(F[k], ra, rb, k);
        const float sc = cat_score(F, S, grp, o, g, T, k);
        if (sc > bv) { bv = sc; bl = k; }
    }
    if (bl >= 0) return bl;
    for (int k = bc; k < nv; k++) if (cat_score(F, S, grp, o, g, T, k) == bv) return k;
    __builtin_trap();
}

// one[r]: the slot of a two-value variable r, else -1.
ANNEAL_ATTR static void cat_loop(const double *Jt, double *F, int m, int V, const int *grp, int *cur,
                                 const float *noise, long nmoves, double beta_start, double beta_max, double *E_io,
                                 double *best_io, int *best_cur, double *late_io, int *late_cur, uint64_t seed,
                                 double lambda, int tenure, double *pen, long *last, int *recent, float *o, float *S,
                                 const int *multi, const int *mfirst, int nmulti, const int *one, const int *blk,
                                 const double *zero) {
    double E = *E_io, best = *best_io, beta = beta_start, dbeta = (beta_max - beta_start) / (double)(nmoves - 1);
    uint64_t x = seed | 1;
    const double *ra = 0, *rb = zero;   // rows of the last move, applied in the next pass (none yet: add zeros)
    int at_best = 0;                    // the current state is the best one of this half, not yet in best_cur
    for (long t = 0; t < nmoves; t++) {
        if (t == nmoves / 2) {   // the second half starts: its lowest state so far is the current one
            if (at_best) __builtin_memcpy(best_cur, cur, sizeof(int) * V);
            *best_io = best; best = E; best_cur = late_cur; at_best = 1;
        }
        beta += dbeta;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        const float *g = noise + (int)(((x * 2685821657736338717ULL) >> 32) * (uint64_t)m >> 32);
        const int k = cat_pass(F, S, ra, rb, grp, blk, o, g, (float)(1.0 / beta), m, cur, pen, multi, mfirst, nmulti);
        const int r = grp[k], c = cur[r], b = k == c ? -1 : k, spin = one[r] >= 0;
        if (spin) { F[k] += (double)S[k] * pen[r]; pen[r] = 0.0; }   // true field of k
        const double dE = b < 0 ? -F[k] : F[k] - (c >= 0 ? F[c] : 0.0);
        if (at_best && dE >= 0.0) { __builtin_memcpy(best_cur, cur, sizeof(int) * V); at_best = 0; }   // leaving it
        E += dE; cur[r] = b;
        if (c >= 0) S[c] = -1.0f;
        if (b >= 0) S[b] = spin ? 1.0f : 0.0f;
        ra = c >= 0 ? Jt + (long)c * m : 0; rb = b >= 0 ? Jt + (long)b * m : 0;
        if (tenure > 0) {
            if (t >= tenure) {   // the variable moved tenure steps ago leaves the tabu list, unless it moved again since
                const int q = recent[t % tenure];
                if (last[q] == t - tenure) {
                    if (one[q] >= 0) F[one[q]] += (double)S[one[q]] * pen[q];
                    pen[q] = 0.0;
                }
            }
            pen[r] = lambda / beta; recent[t % tenure] = r; last[r] = t;
            if (spin) F[k] -= (double)S[k] * pen[r];
        }
        if (E < best) { best = E; at_best = 1; }
    }
    if (at_best) __builtin_memcpy(best_cur, cur, sizeof(int) * V);
    for (int r = 0; r < V; r++) if (one[r] >= 0) F[one[r]] += (double)S[one[r]] * pen[r];   // leave the true fields
    for (int k = 0; k < m; k++) F[k] = cat_update(F[k], ra, rb, k);                         // of the final state
    *E_io = E; *late_io = best;
}

// Jt: m x m couplings between slots; F: m fields (updated in place); grp: variable of each slot, 0..V-1, grouped;
// cur: V current slots (updated in place); noise: m noise values stored twice; best_cur: V slots (in: the initial
// state, out: the lowest of the first half of the moves); late_cur: V slots (out: the lowest of the second half).
// As pfa_anneal_spin otherwise.
void pfa_anneal_categorical(const double *Jt, double *F, int m, int V, const int *grp, int *cur, const float *noise,
                            long nmoves, double beta_start, double beta_max, double *E_io, double *best_io,
                            int *best_cur, double *late_io, int *late_cur, uint64_t seed, double lambda, int tenure) {
    double *pen = calloc(V, sizeof(double)), *zero = calloc(m, sizeof(double));
    float *o = calloc(V + 64, sizeof(float)), *S = malloc(sizeof(float) * m);
    long *last = malloc(sizeof(long) * V);
    int *recent = malloc(sizeof(int) * (tenure > 0 ? tenure : 1)), *multi = malloc(sizeof(int) * V), nmulti = 0;
    int *mfirst = malloc(sizeof(int) * V), *one = malloc(sizeof(int) * V), *blk = malloc(sizeof(int) * (m / 8 + 1));
    for (int k = 0; k < m; k++) {
        const int r = grp[k];
        S[k] = -1.0f;
        one[r] = (k > 0 && grp[k - 1] == r) || (k + 1 < m && grp[k + 1] == r) ? -1 : k;
    }
    for (int k = 0; k < m; k++)
        if (one[grp[k]] < 0 && (k == 0 || grp[k - 1] != grp[k])) { mfirst[nmulti] = k; multi[nmulti++] = grp[k]; }
    for (int r = 0; r < V; r++) {
        last[r] = -1;
        if (cur[r] >= 0) S[cur[r]] = one[r] >= 0 ? 1.0f : 0.0f;
    }
    for (int j = 0; j < m / 8; j++) {
        const int *gj = grp + 8 * j; int spins = 1;
        for (int i = 0; i < 8; i++) spins &= one[gj[i]] >= 0;
        blk[j] = gj[0] == gj[7] ? gj[0] : spins ? V : -1;
    }
    cat_loop(Jt, F, m, V, grp, cur, noise, nmoves, beta_start, beta_max, E_io, best_io, best_cur, late_io, late_cur,
             seed, lambda, tenure, pen, last, recent, o, S, multi, mfirst, nmulti, one, blk, zero);
    free(pen); free(o); free(S); free(zero); free(last); free(recent); free(multi); free(mfirst); free(one); free(blk);
}

// One-hot input -> reference-value layout, in one pass. C: n x n couplings over all slots, f: n fields; ref[r]: the
// reference slot of variable r; slots[i], sgrp[i]: the i-th other slot and its variable (m of them, grouped by
// variable). Writes Jt (m x m, zero within a variable), ut (m) and returns the energy of the all-reference state.
double pfa_from_one_hot(const double *C, const double *f, int n, int V, const int *ref, int m, const int *slots,
                        const int *sgrp, double *Jt, double *ut) {
    double e0 = 0.0;
    for (int r = 0; r < V; r++) {
        const double *row = C + (long)ref[r] * n; double s = 0.0;
        for (int q = 0; q < V; q++) if (q != r) s += row[ref[q]];
        e0 += f[ref[r]] + 0.5 * s;
    }
    for (int i = 0; i < m; i++) {
        int gi = sgrp[i];
        const double *ra = C + (long)slots[i] * n, *rr = C + (long)ref[gi] * n;
        double *out = Jt + (long)i * m;
        for (int j = 0; j < m; j++) {
            int gj = sgrp[j], b = slots[j], rb = ref[gj];
            out[j] = gj == gi ? 0.0 : (ra[b] + rr[rb]) - (ra[rb] + rr[b]);   // same rounding for (i, j) and (j, i)
        }
        double s = f[slots[i]] - f[ref[gi]];
        for (int q = 0; q < V; q++) if (q != gi) s += ra[ref[q]] - rr[ref[q]];
        ut[i] = s;
    }
    return e0;
}
