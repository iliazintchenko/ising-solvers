# Pretty Fast Annealer

PFA solves Ising problems: find the spins s ∈ {−1, +1}ⁿ that minimize

```
E(s) = ½ sᵀ J s + hᵀ s
```

for symmetric couplings J (zero diagonal) and fields h. Flipping spin j changes the energy by

```
ΔE_j = −2 s_j (h_j + Σ_k J_jk s_k)
```

## Results vs. Simulated Annealing

The reference is optimized simulated annealing (SA, details below), each method with `beta_max` tuned for every run
length. A PFA flip costs **1.2–1.4×** an accepted SA flip, and a rejected SA proposal about 1/n of one. Counting both,
PFA reaches the same energy faster by:

| problems | speedup |
|---|---|
| spin glasses | **1.5–2×** |
| constrained problems with large penalties | **15–1000×**, or SA never gets there |

## Algorithm

Each step flips one spin, chosen from all n with probability

```
p_j = e^(−β ΔE_j) / Σ_k e^(−β ΔE_k)
```

while the inverse temperature β rises linearly from 0 to `beta_max`.

**Equilibrium.** Read as a continuous-time process, each flip fires at rate e^(−β ΔE) and its reverse at
e^(+β ΔE). The ratio e^(−2β ΔE) is detailed balance for e^(−2β E): at β, PFA equilibrates at 2β. Skipping the waiting
times also weights each state by its exit rate R(s) = Σ_k e^(−β ΔE_k), favoring states with many cheap exits. An
optimizer doesn't care: it keeps the best state it has seen.

**Tabu.** When spin i flips at inverse temperature β_i, its ΔE is raised by λ/β_i, λ = 3 temperatures, for the
next n/4 flips (the tenure):

```
p_i ∝ e^(−β (ΔE_i + λ/β_i))
```

Right after the flip this divides its weight by e^λ; as the anneal cools, by e^(λβ/β_i). Early on, while β grows fast
relative to β_i, the tabu is nearly hard. A tabu spin i still outweighs a free spin k when ΔE_i + λ/β_i < ΔE_k, so a
move worth more than the penalty, like stepping back down from an expensive climb, still happens. Among moves of
similar cost, the chain picks spins it hasn't just flipped. Hot, this spreads the flips over more spins; cold, it
keeps the chain moving between low-energy states instead of re-flipping the same few spins.

λ is in temperatures and the tenure a fraction of n, so the rule is unchanged when the energy is rescaled
(J, h, β → cJ, ch, β/c) or n grows. Both values are measured, not derived, and set by `tabu_lambda` and
`tabu_tenure`. The tabu breaks detailed balance, which an optimizer doesn't need. It roughly doubles the best
`beta_max`.

**Repetitions.** Run 1 of R starts from random spins at β = 0. Each later run starts from the best state so far,
with fresh noise and β starting partway up the ramp:

```
run 1:  random spins         β: 0                  → beta_max
run k:  best state so far    β: c/(R−1)·beta_max   → beta_max
```

where c counts the earlier runs that did not lower the best energy. The runs form an outer anneal: a restart reheats
the best state, at first hot enough to leave its basin and search widely, later only enough to refine it. The starting
β rises by one step of `beta_max`/(R−1) each time a run brings nothing, so the restarts stay hot for as long as hot
restarts pay.

**Recombination.** A run that ends above the best state can still have part of the system lower. Take two states a
and b, and the spins on which they differ. Taking spins i and j from b changes the energy by

```
Δ_i + Δ_j + 4 J_ij a_i a_j
```

with Δ_i the change of taking i alone: the two choices are independent unless J_ij ≠ 0. So the differing spins fall
into groups, connected through nonzero couplings, and no coupling joins two groups. Taking a whole group G from b
changes the energy by

```
Δ(G) = Σ_{i ∈ G} Δ_i + Σ_{i < j ∈ G} 4 J_ij a_i a_j
```

whatever is taken elsewhere, so a takes every group with Δ(G) < 0. That is the lowest of the 2^g states that have
each of the g groups from a or from b, found in one pass and never above either. For categorical variables r and s
with values a_r, b_r and a_s, b_s the two choices are independent unless

```
T(a_r, a_s) − T(a_r, b_s) − T(b_r, a_s) + T(b_r, b_s) ≠ 0
```

with T the couplings between their values: two coupled variables can still be in different groups.

After every run, the best state takes the groups that the run has lower, from its lowest state in each half of its
flips. In the first half of a restart that is usually the state it started from; in the second it is where the
reheated chain settled, a different state of low energy. A run that would have been discarded now gives its good
parts, and brings nothing only when the best energy stays where it was.

How much this gives depends on how two low states differ. If every pair is coupled, or the states differ on so many
spins that the groups merge, there is one group and the lower state is kept, as without recombination. If the
couplings are sparse and fields or a few strong couplings hold most spins in place, low states differ in many small
groups, and the best state improves group by group. On problems of that kind (side-chain packing, MAX-2-SAT, sparse
glasses with fields or with couplings on several scales), 10 and 30 runs ended lower than without recombination in 82
of 84 trials. On dense problems and a random sparse glass with equal couplings and no fields, the results stayed
within the spread between seeds.

**Categorical variables.** Often a variable takes exactly one of several values. Written as spins, that is a one-hot
group, which every single flip breaks. Given the number of values of each variable, each step instead changes one
whole variable: variable r moves from its value a_r to value k, chosen among all variables and values with

```
p(r→k) ∝ e^(−β ΔE(r→k)),    ΔE(r→k) = F(k) − F(a_r)
```

where F(k) is the energy of value k given every other variable's current value. Every state has one value per
variable, so there is no penalty to set and no climb over one. Value 0 of each variable adds nothing; every other value
has its own row of couplings, so a move reads the row of the value it leaves and of the value it enters, none for value
0. With two values per variable that is one stored value and one row per move: exactly the spin problem.

**Elimination.** Before the anneal, a categorical problem loses the values that no optimum needs. Value a of
variable r is removed when another value b of r has

```
f(a) − f(b) + Σ_s min_c [ T_rs(a, c) − T_rs(b, c) ]  ≥  0
```

with f the fields, T_rs the couplings between the values of r and of s, the sum over the variables s coupled to r, and
the minimum over the values c of s still allowed. The left side is the least that F(a) − F(b) can be, each neighbour
taken at its worst for b. So b is at least as good as a in every state: any state that uses a has one of no higher
energy that does not, and a state of the lowest energy always remains. With ≥ rather than >, values that are never
better than another go too, duplicates included. A removed value leaves the minima of its variable's neighbours, which
can make more values removable, so the test repeats until nothing changes.

The anneal then runs on the values left, and variables left with one value drop out. There is nothing to set. How
much goes depends on the problem: a value goes when its field costs more than its couplings can ever win back
against some other value. Problems given as spins are not reduced: for a spin the test is |h_i| ≥ Σ_j |J_ij|.

## Compared with simulated annealing

SA here follows Isakov et al.,
[Optimised simulated annealing for Ising spin glasses](https://arxiv.org/abs/1401.1084), built like PFA's
implementation: full float64 coupling rows, one replica, one core. The paper's tricks for low-precision couplings,
many replicas or sparse graphs (lookup tables, multi-spin coding, bipartite updates) are left out on both sides.

SA proposes one spin at a time and accepts the flip with probability min(1, e^(−β ΔE_j)). As the anneal cools, it
rejects almost every proposal; PFA never rejects. SA's two ingredients have PFA counterparts:

| SA | PFA | what it does |
|---|---|---|
| Metropolis acceptance of one proposed spin | Boltzmann selection over all n spins | decides which flip happens |
| sequential sweeps | tabu | spreads the work over distinct spins |

**Cost per step.** A flip in either method updates all n local fields. Per spin, an accepted SA flip moves 24 bytes:
a float64 coupling read, a float64 field read and written. A PFA flip also reads the spin and its noise value, 4 bytes
each: 32 bytes, 4/3 of an accepted SA flip. A rejected SA proposal reads one field and one random number, about 1/n of
a flip, so flips set the cost. Work, rejections included, in accepted SA flips per spin:

```
PFA:  (4/3) · flips / n          SA:  (accepted flips + proposals / n) / n
```

**Descent.** Metropolis accepts every downhill move, so it takes them in sweep order, whatever their size. PFA
weights each by e^(β |ΔE_j|) and takes the biggest first. Each PFA flip does more of the descent.

**Sweeps and the tabu.** Random proposals repeat some spins and miss others: after n of them, 1/e ≈ 37% of the spins
are still untried. A sweep proposes each spin once per n proposals, and does better. PFA has no proposals to order;
its repeat-avoidance is the tabu. A sweep is a hard rule on proposals that keeps the Boltzmann distribution
stationary; the tabu is a soft rule on flips that does not.

**Constraints.** Many hard problems are an objective plus constraints written as penalties:

```
E(s) = P · C(s) + O(s)
```

C(s) ≥ 0 is zero exactly on feasible states, O(s) is the objective, and P is much larger than the scale ε of changes
in O. The hard case: every single flip from a feasible state breaks a constraint, as with one-hot or cardinality
constraints. Then every move costs about P or more, and getting from one feasible state to another takes two flips,
one that breaks a constraint and one that repairs it.

SA accepts the breaking flip with probability e^(−β (P + ΔO_i)), so it moves between feasible states only while
β ≲ 1/P. It needs β ≳ 1/ε to tell good objective values from bad. With P/ε ≫ 1, no β does both: at β = 1/ε each move
takes about e^(P/ε) proposals, and the constraints freeze long before the objective is optimized.

PFA is blind to a common shift of all ΔE, which cancels in the normalization of p_j. The moves that break one
constraint all carry the same P, so

```
p_i ∝ e^(−β ΔO_i)
```

The objective alone chooses which constraint to break, whatever P is, and PFA leaves every state in one step, at any
β. It can run cold enough to resolve the objective without freezing on the constraints. Any rejection-free method has
this advantage.

**Long runs.** Late in the anneal, what counts is how far the chain still moves without climbing. At a fixed low
temperature, PFA sits lower in energy than SA and still drifts further from where it started. (Equal temperature means
PFA at β against SA at 2β.)

## Implementation

The annealer runs in compiled C with SIMD, on one core, on any couplings and fields. On the array kernels, every flip
reads one full float64 row, zeros included, and updates all n local fields; the selection rides along in the same
pass. A flip therefore costs the same on every problem of a given size. Problems with few couplings per spin run on
sparse kernels instead (see Sparse couplings). The tricks of the array kernels:

- **Gumbel-max sampling.** Drawing from p_j directly costs n exponentials and a normalization per step. The
  [Gumbel-max trick](https://lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/) removes both:
  with each g_j an independent standard Gumbel, g = −log(−log u) for u uniform on (0, 1),

  ```
  P( argmax_j (−β ΔE_j + g_j) = i ) = p_i
  ```

  so picking a spin costs one multiply, one add and one comparison per spin.
- **Noise drawn once.** At step t, spin j reads g_((j + o_t) mod n), with a fresh random offset o_t. Each step on its
  own is exact Gumbel-max sampling; across steps the same values return permuted, a cheap stand-in for fresh noise. A
  random offset, unlike a shift by one, keeps large values from marching through neighbouring indices, so results
  don't depend on the spin order. The vector is stored as g followed by g, so the n values from any offset are one
  contiguous stretch: no wraparound, no modulo. Offsets come from a xorshift64* generator, mapped to [0, n) by a
  multiply and a shift, without division.
- **Local fields.** Each spin stores φ_j = (h_j + Σ_k J_jk s_k) / 2 in float64, so ΔE_j = −4 s_j φ_j. After F flips,
  rounding has added at most about F · 2⁻⁵³ of the largest field: negligible.
- **Add or subtract, chosen once.** Flipping spin i changes every φ_j by −s_i J_ij. With s_i = ±1 fixed for the pass,
  that is a plain subtract or add of row i, read in place: no multiply, no spin load. φ_i itself does not change.
- **Scores.** Dividing −β ΔE_j + g_j by 4β leaves the argmax unchanged and gives

  ```
  s_j φ_j + g_j / (4β)
  ```

  computed in float32, with spins stored as float32 ±1, so no branch. Rounding can change a pick only when two scores
  agree to 7 digits (2⁻²⁴ ≈ 6·10⁻⁸).
- **Argmax without indices.** A vector argmax carries an index next to every running maximum, which roughly doubles
  the selection work. The pass instead keeps only maxima and records which chunk of 64 spins (8 vectors on SVE)
  reached the largest one. Afterwards it rescores that chunk, still in cache, and returns its first spin at the
  maximum: about 64/n of a pass extra. The rescore repeats the pass's float operations exactly (the build stops the
  compiler from fusing multiply-adds on its own), so it finds the same maximum, and ties go to the smallest index.
- **Tabu for free.** The penalty, c_i = λ / (4β_i) in score units, is folded into the stored field,
  φ̃_i = φ_i − s_i c_i, so that s_i φ̃_i = s_i φ_i − c_i, and removed when i leaves the list or flips again. The pass
  reads nothing extra; the anneal loop adds and removes shifts at O(1) per flip.
- **Energy tracked, not computed.** E += ΔE_i = −4 s_i φ_i at each flip, O(1). The lowest state of each half of a run
  is copied only when the anneal steps up from a new low, so a descent through many new lows costs one copy.
- **Categorical moves.** One pass per move, as for spins: every value k other than 0 is one move, scored in float32

  ```
  o_r + S_k F(k) + g_k / β
  ```

  A move of variable r to k has S_k = −1 and o_r = F(a_r) − c_r, with c_r the tabu of r; the move from the current
  value back to 0 has S_k = 0. A variable with two values is a spin: its one value scores with S_k = ±1 and o_r = 0,
  its tabu folded into F as above. The pass updates every F(k) with the row of the value left and of the value
  entered, none for value 0, and reads o_r only for variables with more than two values, as it reaches them.

### Sparse couplings

When each spin has few neighbours, `run()` uses kernels that keep only the nonzero couplings, and no step passes over
all spins. The input format plays no part: the same couplings as an array or as a `scipy.sparse` matrix give the same
run. Schedule, fields, tabu and bookkeeping are those of the array kernels. The selection is not, so a seed gives a
different run on the sparse kernels than on the array kernels.

- **Fields.** Flipping spin i changes φ_j only for the d neighbours of i: d adds instead of n.
- **Selection without a pass.** Stored noise changes every score at every step, because the offset moves. Fresh
  noise at every step gives p_j ∝ e^(−β ΔE_j), and p_j can be drawn directly: it costs one exponential per ΔE that
  changed, n with an array, d + 1 here. The weights w_j = e^(−β ΔE_j) are the leaves of a tree of sums with fan-out 8,
  every entry the sum of the 8 below it. A uniform number times the total, followed down the tree, lands on spin j
  with probability

  ```
  w_j / Σ_k w_k
  ```

  A flip gives new weights to the flipped spin and its neighbours. The sums above them are recomputed from their 8
  children, so rounding errors do not build up. A draw or an update touches log₈ n entries.
- **β moves at every flip**, and with it every weight. The tree keeps the weights of an earlier β₀, and the draw is
  corrected: spin j is accepted with probability

  ```
  e^(−(β − β₀) (ΔE_j − ΔE_min))        ΔE_min ≤ every ΔE
  ```

  and otherwise drawn again. Weight times acceptance is proportional to e^(−β ΔE_j), so the spin flipped has exactly
  the distribution of this step's β. A draw that is not accepted costs a walk down the tree, not a step.
- **Recomputing the weights.** Rejections become more frequent as β leaves β₀. If they cost a · k at the k-th step
  after all weights were computed, and computing them costs R, a cycle of K steps costs R/K + a K/2 per step: least
  when a K²/2 = R. So all weights are recomputed, at the current β, when the rejected draws since the last time have
  cost as much as that takes. Weights are kept relative to the largest, and are also recomputed if their total leaves
  the range of a float64.
- **Categorical.** Couplings are kept as full tables between coupled variables. A move of variable r updates F for
  the values of r's neighbours only, and those values get new weights, since o changes with the field of a
  neighbour's current value. The same tree draws the next move.
- **Lowest states.** They are brought up to date from a list of the spins flipped since they were last written, not
  copied.

A sparse step costs a fixed part plus one exponential per changed weight; an array step costs one pass over all n
rows of the couplings. `run()` takes the sparse kernel where that is the cheaper step:

```
spins:        d < n / 30 − 7        d: mean number of neighbours of a spin
categorical:  c < n / 9 − 20        c: mean number of values of a variable's neighbours
```

The constants are fitted to timings of both kernels on one machine (Apple M4). For spins that is fewer than about 3%
of the couplings nonzero from a few hundred spins up; for categorical variables, about a tenth of the value pairs
coupled.

Memory comes before speed. The array kernels need the couplings as an n × n float64 array, 8 n² bytes. If that is
more than half of the machine's memory, the kernel whose couplings take less memory runs: the sparse one when its
entries, 24 bytes per nonzero coupling, plus for categorical variables its tables, take less than the array.

### Elimination

One function, `pfa_native_eliminate.c`, reads both layouts of the couplings, the array and the tables of the sparse
kernel: in either, the couplings of a value to the values of one neighbour are one stretch of memory.

- **A failing pair is dropped early.** Most pairs (a, b) fail the test, so its sum is ordered to show that soon. It
  starts at E(a) − E(b), the energies of the two values with every neighbour at its lowest allowed value: one sum per
  value, not per pair, and never below the full sum. Each neighbour then lowers it, from its term in that state to
  its minimum. Once below 0 it stays there, and the pair is dropped.
- **The neighbour that decides is read first.** The neighbour that took a pair below 0 moves to the front of its
  variable's order, where the next pairs meet it.
- **Only what changed is tested again**: the variables with a neighbour that lost a value.
- **No tolerance.** The comparison is the floating-point one, so two values that tie only up to rounding can both stay.
- **The reduced problem.** The lowest value left of each variable becomes its value 0. With k_s that value of s,

  ```
  T'(a, c) = T(a, c) − T(a, k_s) − T(k_r, c) + T(k_r, k_s)
  f'(a)    = G(a) − G(k_r),        G(a) = f(a) + Σ_s T(a, k_s)
  ```

  which changes the energy by a constant. The couplings of a variable left with one value are then in the fields.
  `run()` picks the kernel again for the reduced problem.

### Recombination

`pfa_native_fold.c` has one function for spins with the couplings as an array, one for spins with the couplings by
row, and one for categorical variables, which reads both layouts as the elimination does.

- **Groups.** The differing variables are disjoint sets, joined for every pair of them that interacts; each set then
  has one sum, Δ(G). The work is the couplings of the differing variables: their stored neighbours on the sparse
  kernels, their rows of the array otherwise.
- **No tolerance.** A pair interacts when the expression above is not exactly 0. Where it is nonzero only by rounding,
  two groups are joined that could have been chosen separately: a choice is lost, exactness is not.
- **Two states per run.** Each kernel keeps the lowest state of the first and of the second half of its flips, with
  the bookkeeping it already had: at the midpoint the current state becomes the lowest so far of the second half.
- **Rounds.** The best state takes groups from the run's two states and from the state the run started from, going
  round until its energy, a float64, no longer falls. A run brought nothing when that energy is not below the
  energy of the state it started from.

## Usage

`make` (GCC or Clang) compiles `ising_solvers/native/pfa_native.c` for the current machine. It includes the four
kernels: `pfa_native_spin.c` and `pfa_native_categorical.c` for couplings as an array, and `pfa_native_sparse_spin.c`
and `pfa_native_sparse_categorical.c`, which draw from the tree of sums of `pfa_native_sums.c`; and
`pfa_native_eliminate.c` and `pfa_native_fold.c`. The first two use the one SIMD path that matches the machine; the
rest is the same code on every path:

| path | files | used on |
|---|---|---|
| SVE | `pfa_native_spin_sve.c`, `pfa_native_categorical_sve.c` | ARM with SVE (Graviton3 and 4, NVIDIA Grace, A64FX, ...) |
| NEON | `pfa_native_spin_neon.c`, `pfa_native_categorical_neon.c` | every other 64-bit ARM (Apple Silicon, Graviton2, Ampere Altra, ...) |
| AVX-512 | `pfa_native_spin_avx512.c`, `pfa_native_categorical_avx512.c` | x86-64 with AVX-512 |
| AVX2 | `pfa_native_spin_avx2.c`, `pfa_native_categorical_avx2.c` | x86-64 with AVX2 |
| scalar | `pfa_native_spin_scalar.c`, `pfa_native_categorical_scalar.c` | anything else |

`pfa_native.simd_path()` reports which one was built; `make -B ARCH_FLAGS="-mavx2 -mfma"` rebuilds for another
target. Then:

```python
from ising_solvers import pfa_native

spins = pfa_native.run(couplings, fields, beta_max=1.0, num_flips=100 * len(couplings), num_reps=10)
values = pfa_native.run(couplings, fields, sizes=sizes, num_flips=100 * len(sizes), num_reps=10)
```

It returns the lowest-energy state found, as ±1 spins (int8), or with `sizes` as the value of each variable (int64).

| parameter | what it is | default | theory |
|---|---|---|---|
| `couplings` | J, an n × n symmetric matrix with zero diagonal: an array or a `scipy.sparse` matrix | required | The format does not change the run: `run()` picks the array or the sparse kernels from the number of nonzero couplings (see Sparse couplings). The array kernels read a C-contiguous float64 array in place; any other input is converted once, a `scipy.sparse` matrix to n × n float64. If that array would take more than half of the machine's memory, the kernel that needs less memory runs. |
| `fields` | h, a length-n vector | `None` (zeros) | Enter only through the local fields φ, so they cost nothing per flip. |
| `sizes` | for categorical problems, the number of values of each variable | `None` | Makes the problem categorical (see Categorical variables): variable r takes a value in 0 … `sizes[r]` − 1. Each value other than 0 has a row and column of `couplings` and an entry of `fields`, variable by variable, so n = Σ (`sizes[r]` − 1). The energy sums the fields of the values taken and half the couplings between every two of them; couplings between values of the same variable must be zero. Each step changes one variable's value. Values that no optimum needs are removed first (see Elimination). |
| `beta_max` | the final inverse temperature | `1.0` | Measured in 1/energy: scaling the energy by c scales the best `beta_max` by 1/c, so the default suits only ΔE of order one. A linear schedule is nearly as good as optimized ones (Isakov et al.). |
| `num_flips` | steps per run, the length of the anneal | `1000` | The budget of one run: num_flips · n field updates, or num_flips / n flips per spin. Longer anneals track the cooling more closely and reach lower energies. |
| `num_reps` | runs of the anneal, R | `1` | The first starts from random spins; each later run restarts from the best state so far with β rising from c/(R−1)·`beta_max`, c the number of earlier runs that did not lower the best energy (see Repetitions). After each run the best state takes the groups of spins that the run has lower (see Recombination). |
| `seed` | random seed | `42` | Sets the first run's spins and every run's Gumbel noise vector and offsets, or on the sparse kernels its draws, so results are reproducible. |
| `tabu_lambda` | the tabu penalty λ, in temperatures | `3.0` | Divides a just-flipped spin's weight by e^λ (see Tabu). Rarely worth changing. |
| `tabu_tenure` | the tabu tenure, as a fraction of n | `0.25` | A spin stays tabu for `tabu_tenure` · n flips; 0 turns the tabu off. With `sizes`, the tabu acts per variable, and n is the number of variables. Rarely worth changing. |

A categorical problem in one-hot form, ½ xᵀ C x + fᵀ x over binary x with one 1 per group, converts with
`pfa_native.categorical_from_one_hot(couplings, fields, groups)`, which returns the couplings, fields, sizes and the
energy offset between the two forms; `pfa_native.categorical_to_one_hot(values, groups)` converts a result back.

`examples/quickstart.py` solves a small problem and measures flips per second.
