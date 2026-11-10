// Selection for the sparse kernels (included by pfa_native.c). Each of n candidates j (a spin, or a value of a
// variable) has a float32 value a[j], its score without the noise, and the pick is the one the dense kernels make:
// the first j to reach max_j(a[j] + T * g[j]), g being the noise read from this step's offset. A dense pass scores
// every j. A sparse kernel changes few a[j] per step and finds the same j here without scoring most of them.
//
// Candidates are grouped in blocks of 64, blocks in nodes of 64 blocks, and so on up to one group of at most 64
// nodes. Every node keeps an upper bound A of the a[j] below it, and a table gives the largest noise value G that
// its candidates can read at this step's offset. No score below the node then exceeds A + T * G, in float32 too
// (rounding is monotone), so a node whose bound is under a score already found is skipped. The search starts at the
// top and enters, at each node, the child with the largest bound first, then every other child whose bound still
// reaches the best score found.
//   - A is raised when an a[j] below the node rises, and set to the largest value of its children whenever the node
//     is entered. A bound left too high costs one visit and never changes the pick.
//   - G: at offset f, the candidates of node c of size B read the noise window [cB + f, cB + f + B). The table of a
//     level holds the maximum of every such window, one row per f mod B, when that fits in about 4n values;
//     otherwise (the largest nodes) the maximum of the two aligned blocks of B noise values that hold the window.
//
// tree_block() scores 64 candidates, or bounds 64 nodes, and tree_mask() marks those that reach a threshold; one
// version of each per SIMD path, in pfa_native_tree_<path>.c.
#define TREE_LEVELS 6   // 64^5 candidates per top node

typedef struct {
    int top;                 // number of levels above the candidates
    float *a[TREE_LEVELS];   // a[0]: the values, padded with -inf to whole blocks; a[l]: the bounds A of level l
    float *g[TREE_LEVELS];   // g[0]: the noise; g[l]: the noise maxima G of level l
    long cols[TREE_LEVELS];  // row length of g[l] where it holds every window, else 0
    int offset, arg;         // of the search in progress: noise offset, best candidate so far
    float T, best;           //                            temperature, best score so far
} Tree;

#include TREE_BLOCK   // tree_block(), tree_mask()

// The 64 candidates c0 .. c0 + 63.
static inline void tree_leaf(Tree *t, int c0) SVE_STREAMING {
    float u[64], amax;
    const float umax = tree_block(t->a[0] + c0, t->g[0] + t->offset + c0, t->T, u, &amax);
    if (t->top) t->a[1][c0 >> 6] = amax;
    if (umax < t->best) return;
    const uint64_t at = tree_mask(u, umax);
    if (!at) __builtin_trap();
    const int j = c0 + __builtin_ctzll(at);
    if (umax > t->best || j < t->arg) { t->best = umax; t->arg = j; }
}

// The 64 nodes c0 .. c0 + 63 of level l >= 1.
static void tree_node(Tree *t, int l, int c0) SVE_STREAMING {
    const int f = t->offset, sh = 6 * l;
    const float *g = t->cols[l] ? t->g[l] + (long)(f & ((1 << sh) - 1)) * t->cols[l] + (f >> sh) + c0
                                : t->g[l] + (f >> sh) + c0;
    float u[64], amax;
    const float umax = tree_block(t->a[l] + c0, g, t->T, u, &amax);
    if (l < t->top) t->a[l + 1][c0 >> 6] = amax;
    if (umax < t->best) return;
#define ENTER(I) { if (l == 1) tree_leaf(t, (c0 + (I)) << 6); else tree_node(t, l - 1, (c0 + (I)) << 6); }
    const uint64_t at = tree_mask(u, umax);
    if (!at) __builtin_trap();
    const int k = __builtin_ctzll(at);
    ENTER(k)
    for (uint64_t rest = tree_mask(u, t->best) & ~(1ULL << k); rest; rest &= rest - 1) {
        const int i = __builtin_ctzll(rest);
        if (u[i] >= t->best) ENTER(i)
    }
#undef ENTER
}

// The first j to reach max_j(a[j] + T * noise[offset + j]).
static inline int tree_argmax(Tree *t, int offset, float T) SVE_STREAMING {
    t->offset = offset; t->T = T; t->best = -__builtin_inff(); t->arg = -1;
    if (t->top) tree_node(t, t->top, 0); else tree_leaf(t, 0);
    return t->arg;
}

// Raises the bounds above candidate j to v, from level l up.
static inline void tree_raise(Tree *t, int j, float v, int l) SVE_STREAMING {
    for (; l <= t->top; l++) {
        float *b = t->a[l] + (j >> (6 * l));
        if (v <= *b) return;
        *b = v;
    }
}

// a[j] = v.
static inline void tree_set(Tree *t, int j, float v) SVE_STREAMING {
    t->a[0][j] = v;
    tree_raise(t, j, v, 1);
}

// After any change to the values of block c: its bound becomes their maximum.
static inline void tree_fit(Tree *t, int c) SVE_STREAMING {
    if (t->top == 0) return;
    const float *a = t->a[0] + 64 * c;
    float m[8];
    for (int i = 0; i < 8; i++) m[i] = a[i];
    for (int k = 8; k < 64; k += 8) for (int i = 0; i < 8; i++) m[i] = a[k + i] > m[i] ? a[k + i] : m[i];
    for (int i = 1; i < 8; i++) m[0] = m[i] > m[0] ? m[i] : m[0];
    t->a[1][c] = m[0];
    tree_raise(t, 64 * c, m[0], 2);
}

// A tree over n candidates, every value -inf, for the noise vector stored twice (2n values).
static void tree_new(Tree *t, int n, const float *noise) {
    const float ninf = -__builtin_inff();
    long cnt[TREE_LEVELS];
    cnt[0] = ((long)n + 63) & ~63L; t->top = 0;
    while (cnt[t->top] > 64) { t->top++; cnt[t->top] = ((cnt[t->top - 1] >> 6) + 63) & ~63L; }
    for (int l = 0; l <= t->top; l++) {
        t->a[l] = malloc(sizeof(float) * cnt[l]);
        for (long c = 0; c < cnt[l]; c++) t->a[l][c] = ninf;
    }
    t->g[0] = malloc(sizeof(float) * (2L * n + 64)); t->cols[0] = 0;
    __builtin_memcpy(t->g[0], noise, sizeof(float) * 2L * n);
    for (long x = 2L * n; x < 2L * n + 64; x++) t->g[0][x] = 0.0f;
    for (int l = 1; l <= t->top; l++) {
        const long B = 1L << (6 * l), cols = cnt[l] + (n >> (6 * l)) + 1;   // columns c + (offset >> 6l)
        if (B * cols <= 4L * n + 8192) {   // w[x] = max noise[x .. x + B), then one row per x mod B
            const long len = B * cols;
            float *w = malloc(sizeof(float) * (len + B));
            for (long x = 0; x < len + B; x++) w[x] = x < 2L * n ? noise[x] : ninf;
            for (long b = 1; b < B; b *= 2) for (long x = 0; x < len; x++) if (w[x + b] > w[x]) w[x] = w[x + b];
            t->g[l] = malloc(sizeof(float) * len); t->cols[l] = cols;
            for (long r = 0; r < B; r++) for (long q = 0; q < cols; q++) t->g[l][r * cols + q] = w[q * B + r];
            free(w);
        } else {                           // w[q] = max of aligned block q, then pairs of blocks
            float *w = malloc(sizeof(float) * (cols + 1));
            for (long q = 0; q <= cols; q++) {
                long lo = q * B, hi = lo + B < 2L * n ? lo + B : 2L * n;
                w[q] = ninf;
                for (long x = lo; x < hi; x++) if (noise[x] > w[q]) w[q] = noise[x];
            }
            t->g[l] = malloc(sizeof(float) * cols); t->cols[l] = 0;
            for (long q = 0; q < cols; q++) t->g[l][q] = w[q] > w[q + 1] ? w[q] : w[q + 1];
            free(w);
        }
    }
}

static void tree_free(Tree *t) {
    for (int l = 0; l <= t->top; l++) { free(t->a[l]); free(t->g[l]); }
}
