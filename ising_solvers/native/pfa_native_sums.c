// Selection for the sparse kernels (included by pfa_native.c). Each of n candidates k (a spin, or a value of a
// variable) has a value x[k] = -dE_k minus its tabu penalty, and the next one is drawn with probability
// exp(beta x[k]) / sum_j exp(beta x[j]). That is the distribution of argmax_k(beta x[k] + g_k) with fresh Gumbel
// noise g at every step, which the array kernels sample with stored noise. A sparse step changes few x[k], and
// nothing here passes over all candidates per step.
//   - Weights w[k] = exp(beta0 (x[k] - x0)) are the leaves of a tree of sums with fan-out 8: every entry of a level
//     is the sum of 8 entries of the level below. A new x[k] recomputes its weight and the sums above it from their
//     8 children, so rounding errors do not build up. A draw takes a uniform number times the total and walks down.
//   - beta0 is the beta at which all weights were last computed and x0 the largest x then. beta has risen since, and
//     the draw is corrected by rejection: candidate k is accepted with probability exp((beta - beta0) (x[k] - xmax)),
//     with xmax >= every x. Its weight times that is proportional to exp(beta x[k]), so the accepted candidate has
//     the distribution above at the beta of this step.
//   - All weights are recomputed at the current beta once the rejected draws since the last time have cost as much
//     as that takes: with rejections becoming more frequent at a steady rate, this is when recomputing costs the
//     least per step. They are also recomputed when the total has left the range of a double (0 or infinite).
#include <math.h>

#define SUMS_LEVELS 12   // 8^11 candidates per entry of the top level

typedef struct {
    int n, top;               // candidates; number of levels above the weights
    double *x;                // the values
    double *w[SUMS_LEVELS];   // w[0]: the weights, padded with 0 to whole groups of 8; w[l]: sums of 8 of w[l - 1]
    double beta0, x0, xmax;   // see above
    long rejected;            // draws rejected since all weights were computed
} Sums;

static inline double sums_8(const double *w) {
    return ((w[0] + w[1]) + (w[2] + w[3])) + ((w[4] + w[5]) + (w[6] + w[7]));
}

// After new x[lo .. hi): their weights and the sums above them.
static inline void sums_refresh(Sums *t, long lo, long hi) {
    const double *x = t->x; double *w = t->w[0];
    for (long k = lo; k < hi; k++) {
        if (x[k] > t->xmax) t->xmax = x[k];
        w[k] = exp(t->beta0 * (x[k] - t->x0));
    }
    for (int l = 1; l <= t->top; l++) {
        lo >>= 3; hi = ((hi - 1) >> 3) + 1;
        for (long c = lo; c < hi; c++) t->w[l][c] = sums_8(t->w[l - 1] + 8 * c);
    }
}

// The same for the candidates idx[0 .. cnt); in ascending order, each sum above them is computed once.
static inline void sums_refresh_at(Sums *t, const int *idx, int cnt) {
    for (int e = 0; e < cnt; e++) {
        const int k = idx[e];
        if (t->x[k] > t->xmax) t->xmax = t->x[k];
        t->w[0][k] = exp(t->beta0 * (t->x[k] - t->x0));
    }
    for (int l = 1; l <= t->top; l++) {
        long last = -1;
        for (int e = 0; e < cnt; e++) {
            const long c = idx[e] >> (3 * l);
            if (c != last) { t->w[l][c] = sums_8(t->w[l - 1] + 8 * c); last = c; }
        }
    }
}

// All weights, at beta.
static void sums_build(Sums *t, double beta) {
    double m = t->x[0];
    for (int k = 1; k < t->n; k++) if (t->x[k] > m) m = t->x[k];
    t->beta0 = beta; t->x0 = t->xmax = m; t->rejected = 0;
    sums_refresh(t, 0, t->n);
}

static inline double sums_uniform(uint64_t *rng) {   // in [0, 1), from a xorshift64* generator
    uint64_t x = *rng;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *rng = x;
    return (double)((x * 2685821657736338717ULL) >> 11) * 0x1p-53;
}

// A candidate drawn with probability proportional to exp(beta x[k]); beta >= the beta of every earlier draw.
static inline int sums_draw(Sums *t, double beta, uint64_t *rng) {
    for (;;) {
        double total = sums_8(t->w[t->top]);
        if (t->rejected * (t->top + 2) >= t->n || !(total > 0.0 && total < __builtin_inf())) {
            sums_build(t, beta);
            total = sums_8(t->w[t->top]);
        }
        double r = sums_uniform(rng) * total;
        long c = 0;
        for (int l = t->top; l >= 0; l--) {   // the entry of this group that r falls in
            const double *w = t->w[l] + 8 * c;
            int k = 0, last = 0;
            for (; k < 8; k++) {
                if (w[k] > 0.0) last = k;
                if (r < w[k]) break;
                r -= w[k];
            }
            if (k == 8) { k = last; r = 0.0; }   // rounding took r past the end
            c = 8 * c + k;
        }
        if (beta == t->beta0 || sums_uniform(rng) < exp((beta - t->beta0) * (t->x[c] - t->xmax))) return (int)c;
        t->rejected++;
    }
}

// n candidates; the caller fills x before the first draw.
static void sums_new(Sums *t, int n) {
    long cnt = ((long)n + 7) & ~7L;
    t->n = n; t->top = 0; t->rejected = n;   // the first draw computes the weights
    t->x = malloc(sizeof(double) * n);
    t->w[0] = calloc(cnt, sizeof(double));
    while (cnt > 8) {
        cnt = ((cnt >> 3) + 7) & ~7L;
        t->w[++t->top] = calloc(cnt, sizeof(double));
    }
}

static void sums_free(Sums *t) {
    free(t->x);
    for (int l = 0; l <= t->top; l++) free(t->w[l]);
}
