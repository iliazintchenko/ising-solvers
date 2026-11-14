# Pretty Fast Annealer

PFA minimizes an energy of single-variable and pairwise terms over discrete variables. Variable r takes one of several
values, a_r, and

```
E = Σ_r f_r(a_r) + Σ_{r<s} T_rs(a_r, a_s)
```

with fields f_r and coupling tables T_rs. A step moves one variable r from a_r to another value k, which changes the
energy by

```
ΔE(r→k) = F(k) − F(a_r),        F(k) = f_r(k) + Σ_s T_rs(k, a_s)
```

F(k), the field of value k, is the energy that value k adds given the current values of the other variables. A
variable with two values is a spin, and its one move a flip. Spins are given as s ∈ {−1, +1}ⁿ with couplings J and
fields h,

```
E(s) = ½ sᵀ J s + hᵀ s,        ΔE_j = −2 s_j (h_j + Σ_l J_jl s_l)
```

the case T_rs(s_r, s_s) = J_rs s_r s_s and f_r(s_r) = h_r s_r. A problem over 0/1 variables x is the same problem in
s = 2x − 1 (`pfa.from_binary`), and constraints enter as penalties (see Constraints). PFA is an annealer that never
rejects: every step makes one move, chosen among all moves of all variables by its Boltzmann weight.

```python
import pfa

spins = pfa.run(couplings, fields, beta_max=1.0, num_steps=100 * len(couplings), num_reps=10)
values = pfa.run(couplings, fields, sizes=sizes, num_steps=100 * len(sizes), num_reps=10)
```

## Algorithm

Each step makes one move, chosen among all moves of all variables with probability

```
p(r→k) = e^(−β ΔE(r→k)) / Σ over all moves e^(−β ΔE)
```

while the inverse temperature β rises linearly from 0 to `beta_max`. Every state has one value per variable. So a
variable that must take exactly one of several values is not written as a one-hot group of spins, where every single
flip breaks the constraint and a penalty must hold it: there is no penalty to set and no barrier between the values,
and a move changes the whole variable in one step.

**Equilibrium.** Read as a continuous-time process, each move fires at rate e^(−β ΔE) and its reverse at
e^(+β ΔE). The ratio e^(−2β ΔE) is detailed balance for e^(−2β E): run at β, PFA samples the Boltzmann distribution of
inverse temperature 2β. Skipping the waiting times also weights each state by its exit rate, Σ over its moves of
e^(−β ΔE), favoring states with many cheap exits. An optimizer doesn't care: it keeps the best state it has seen.

**Tabu.** When variable r moves at inverse temperature β_r, the ΔE of every move of r is raised by λ/β_r for the next
V/4 steps (the tenure), V the number of variables, with λ = 3: three times the temperature at which it moved.

```
p(r→k) ∝ e^(−β (ΔE(r→k) + λ/β_r))
```

Right after the move this divides the weight of r's moves by e^λ; as the anneal cools, by e^(λβ/β_r). Early on, while
β grows fast relative to β_r, the tabu is nearly hard. A move of a tabu variable still outweighs a move of a free one
when ΔE(r→k) + λ/β_r < ΔE(s→l), so a move worth more than the penalty, like stepping back down from an expensive
climb, still happens. Among moves of similar cost, the anneal picks variables it hasn't just moved. Hot, this spreads
the moves over more variables; cold, it keeps the state moving between low-energy states instead of moving the same
few variables back and forth.

λ is in units of the temperature and the tenure a fraction of V, so the rule is unchanged when the energy is rescaled
(f, T, β → αf, αT, β/α) or V grows. Both constants are tuned on benchmarks, not derived, and set by `tabu_lambda` and
`tabu_tenure`. The tabu breaks detailed balance, which an optimizer doesn't need. With it, the best `beta_max` is
roughly twice what it is without.

**Repetitions.** Repetition 1 of R starts from a random value for every variable at β = 0. Each later one, a restart,
starts from the best state so far, with new random numbers and β starting partway up the ramp:

```
repetition 1:  random values        β: 0                  → beta_max
restart:       best state so far    β: u/(R−1)·beta_max   → beta_max
```

where u is 1 plus the number of restarts so far that brought nothing, that is, did not lower the best energy. The
restarts form an outer anneal: each reheats the best state, at first hot enough to leave its basin and search widely,
later only enough to refine it. The starting β rises by one step of `beta_max`/(R−1) each time a restart brings
nothing, so the restarts stay hot for as long as hot restarts pay.

**Elimination.** Before the anneal, the problem loses the values that no optimum needs. Value a of variable r is
removed when another value b of r has

```
f_r(a) − f_r(b) + Σ_s min_c [ T_rs(a, c) − T_rs(b, c) ]  ≥  0
```

with the sum over the neighbours s of r, the variables coupled to it, and the minimum over the values c of s still
allowed. The left side is the least that F(a) − F(b) can be, each neighbour taken at the value worst for b. So b is at
least as good as a in every state: any state that uses a has one of no higher energy that does not, and a state of the
lowest energy always remains. With ≥ rather than >, values that are never better than another go too, duplicates
included. A removed value of s drops out of the minima over c in the tests of the neighbours of s. That can only raise
those minima, so more values can become removable, and the test repeats until nothing changes.

The anneal then runs on the values left, and variables left with one value drop out. The elimination has no
parameter. How much goes depends on the problem: a value goes when its field costs more than its couplings can ever
win back against some other value. Problems given as spins are not reduced. For a spin the test would be
|h_i| ≥ Σ_j |J_ij|: a field that outweighs all the spin's couplings together.

**Recombination.** A repetition whose lowest state is above the best state can still hold some variables in a better
arrangement. Take two states a and b, and the variables on which they differ. Setting variables r and s of a to their
values in b changes its energy by

```
Δ_r + Δ_s + I_rs,        I_rs = T_rs(a_r, a_s) − T_rs(a_r, b_s) − T_rs(b_r, a_s) + T_rs(b_r, b_s)
```

with Δ_r the change of setting r alone: the two choices are independent unless I_rs ≠ 0. For spins
I_rs = 4 J_rs a_r a_s, nonzero whenever the coupling is; two coupled variables with more values can still be
independent. So the differing variables fall into clusters, the connected components of the pairs with I_rs ≠ 0, and
no such pair joins two clusters. Setting a whole cluster G to its values in b changes the energy by

```
Δ(G) = Σ_{r ∈ G} Δ_r + Σ_{r < s ∈ G} I_rs
```

whatever is done to the other clusters, so a takes every cluster with Δ(G) < 0. That is the lowest of the 2^m states
that have each of the m clusters from a or from b, found in one pass and never above either.

Each repetition records two states: the lowest of the first half of its steps and the lowest of the second half. In a
restart the first is usually the state it started from; the second is where the reheated anneal settled, a different
state of low energy. After the repetition, the lowest of three states, these two and the best state so far, becomes
the best state and takes from the other two every cluster they have lower. A repetition that ends above the best
state still contributes its good parts, and brings nothing only when the best energy stays where it was.

How much this gives depends on how two low states differ. If every two variables are coupled, or the states differ on
so many variables that the clusters merge, there is one cluster and the lower state is kept, as without
recombination. If the couplings are sparse and fields or a few strong couplings hold most variables in place, low
states differ in many small clusters, and the best state improves cluster by cluster. On problems of that kind
(side-chain packing, MAX-2-SAT, sparse glasses with fields or with couplings on several scales), with R = 10 and
R = 30 the final energy was lower in 82 of 84 trials (seven problems, six seeds, both values of R) than with the
earlier scheme, no recombination and every restart one step colder than the last whatever it brought. On dense problems and a random sparse glass
with couplings of equal magnitude and no fields, the results stayed within the spread between seeds.

## Compared with simulated annealing

SA here follows Isakov et al.,
[Optimised simulated annealing for Ising spin glasses](https://arxiv.org/abs/1401.1084), built like PFA's
implementation: full float64 coupling rows, one replica, one core. The paper's tricks for low-precision couplings,
many replicas or sparse graphs (lookup tables, multi-spin coding, bipartite updates) are left out on both sides, and
so is PFA's own storage for sparse couplings.

SA proposes one spin at a time and accepts the flip with probability min(1, e^(−β ΔE_j)). As the anneal cools, it
rejects almost every proposal; PFA never rejects. The comparison is of this step: repetitions, elimination and
recombination sit outside it and would apply to either method.

Each method ran with its final inverse temperature tuned for every length of anneal, on spin problems. A PFA flip
costs **1.2–1.4×** an accepted SA flip; a rejected SA proposal costs about 1/n of an accepted flip. With flips and
rejected proposals both counted, PFA reaches the same energy in less work by:

| problems | speedup |
|---|---|
| spin glasses | **1.5–2×** |
| constrained problems with large penalties | **15–1000×**, or SA never reaches that energy |

**Cost per step.** A flip in either method updates all n local fields. Per spin, an accepted SA flip moves 24 bytes:
a float64 coupling read, a float64 field read and written. A PFA flip also reads the spin and a stored random number
(see Gumbel-max sampling), 4 bytes each: 32 bytes, 4/3 of an accepted SA flip. A rejected SA proposal reads one field
and one random number, about 1/n of a flip. Work in units of n accepted SA flips (one per spin), rejected proposals
included:

```
PFA:  (4/3) · flips / n          SA:  (accepted flips + proposals / n) / n
```

**Descent.** Metropolis accepts every downhill flip, so it takes them in sweep order, whatever their size. PFA
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
constraints. Then every flip costs about P or more, and getting from one feasible state to another takes two flips,
one that breaks a constraint and one that repairs it.

SA accepts the breaking flip with probability e^(−β (P + ΔO_i)), so it moves between feasible states only while
β ≲ 1/P. It needs β ≳ 1/ε to tell good objective values from bad. With P/ε ≫ 1, no β does both: at β = 1/ε each
breaking flip takes about e^(P/ε) proposals, and the constraints freeze long before the objective is optimized.

PFA is blind to a common shift of all ΔE, which cancels in the normalization of p. From a feasible state, the flips
that break one constraint all carry the same P, so

```
p_i ∝ e^(−β ΔO_i)
```

The objective alone chooses which constraint to break, whatever P is, and PFA leaves every state in one step, at any
β. It can run cold enough to resolve the objective without freezing on the constraints. Any method that never rejects
has this advantage.

**Long anneals.** Late in the anneal, what counts is how far the state still moves without climbing. Held at the same
low temperature (PFA at β, SA at 2β, see Equilibrium), PFA sits lower in energy than SA and still drifts further from where it started.

## Implementation

The annealer runs in compiled C with SIMD, on one core, on any couplings and fields. It has four kernels, the compiled
anneal loops: one for spins and one for variables with more values, each with the couplings either as an n × n array
(the array kernels) or as lists of the nonzero ones (the sparse kernels, see Sparse couplings). Value 0 of every
variable has no field and no couplings (any problem can be shifted to that form, see `sizes` under Usage), so each of
the other values has a row of the couplings, n rows in all, and a move reads the row of the value it leaves and of
the value it enters, none for value 0. On the array kernels those are full float64 rows, zeros included, and one pass
over the n rows updates every field and selects the next move. A move therefore costs the same on every problem of a
given size. The tricks of the array kernels:

- **Gumbel-max sampling.** Drawing from p directly costs n exponentials and a normalization per step. The
  [Gumbel-max trick](https://lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/) removes both:
  with each g_k an independent standard Gumbel, g = −log(−log u) for u uniform on (0, 1),

  ```
  P( argmax_k (−β ΔE_k + g_k) = i ) = p_i
  ```

  so picking a move costs one multiply, one add and one comparison per row.
- **Noise drawn once per repetition.** At step t, row k reads g_((k + ω_t) mod n), with a fresh random offset ω_t.
  Each step on its own is exact Gumbel-max sampling; across steps the same values return shifted, a cheap stand-in
  for fresh noise. A random offset, unlike a shift by one, keeps large values from marching through neighbouring
  indices, so results don't depend on the order of the variables. The vector is stored as g followed by g, so the n
  values from any offset are one contiguous stretch: no wraparound, no modulo. Offsets come from a xorshift64*
  generator, mapped to [0, n) by a multiply and a shift, without division.
- **Scores.** Row k, a value other than 0 of variable r, is one candidate move, scored in float32 as
  −ΔE − c_r + g_k / β, with c_r = λ / β_r the tabu penalty of r, and written

  ```
  o_r + S_k F(k) + g_k / β
  ```

  If k is not the current value a_r, the move is r → k: S_k = −1 and o_r = F(a_r) − c_r, with F(0) = 0. The row of
  a_r itself stands for the move back to value 0: S_k = 0. The pass updates every F(k) with the row of the value left
  and of the value entered, and reads o_r only for variables with more than two values, as it reaches them. Rounding
  can change a pick only when two scores agree to 7 digits (2⁻²⁴ ≈ 6·10⁻⁸).
- **Spins.** A spin has one row and one move, and the spin kernel stores φ_j = (h_j + Σ_l J_jl s_l) / 2 in float64,
  so ΔE_j = −4 s_j φ_j. Dividing −β ΔE_j + g_j by 4β leaves the argmax unchanged and gives the score

  ```
  s_j φ_j + g_j / (4β)
  ```

  with spins stored as float32 ±1: no branch and no o_r. Flipping spin i changes every φ_j by −s_i J_ij, with s_i its
  value before the flip. With s_i = ±1 fixed for the pass, that is a plain subtract or add of row i, read in place: no
  multiply, and s_i is not read again. φ_i itself does not change. After N flips, rounding has added at most about
  N · 2⁻⁵³ of the largest field: negligible. A variable with two values among variables with more is scored as a
  spin, not by the rule above: S_k = +1 when k is its value and −1 when it is at value 0, with o_r = 0.
- **Argmax without indices.** A vector argmax carries an index next to every running maximum, which roughly doubles
  the selection work. The pass instead keeps only maxima and records which chunk of 64 rows (8 vectors on SVE)
  reached the largest one. Afterwards it rescores that chunk, still in cache, and returns its first row at the
  maximum: about 64/n of a pass extra. The rescore repeats the pass's float operations exactly (the build stops the
  compiler from fusing multiply-adds on its own), so it finds the same maximum, and ties go to the smallest index.
- **Tabu for free.** For a spin, the penalty, c_i = λ / (4β_i) in score units, is added into the stored field,
  φ̃_i = φ_i − s_i c_i, so that s_i φ̃_i = s_i φ_i − c_i, and taken out when the tenure of i ends or i flips again. A
  variable with two values among variables with more has it added into F the same way; a variable with more values
  has it in o_r. The pass reads nothing extra; the anneal loop adds and removes penalties
  at O(1) per step.
- **Energy tracked, not computed.** E += ΔE at each move, O(1). The lowest state of each half of a repetition is
  copied only when the anneal steps up from a new low, so a descent through many new lows costs one copy.

### Sparse couplings

When each variable has few neighbours, `run()` uses kernels that keep only the nonzero couplings, and no step passes
over all rows. The input format plays no part: the same couplings as an array or as a `scipy.sparse` matrix give the
same result. Schedule, fields, tabu and bookkeeping are those of the array kernels. The selection is not, so a seed
gives a different result on the sparse kernels than on the array kernels.

- **Fields.** A move of variable r changes F only on the rows of r's neighbours: for a spin with d neighbours, d adds
  instead of n. For variables with more values, the couplings are kept as full tables between coupled variables.
- **Selection without a pass.** Gumbel-max with stored noise needs the pass: the offset moves, so every score changes
  at every step. The sparse kernels draw from p directly instead, which costs one exponential per ΔE that changed:
  for a spin d + 1, n if every spin were coupled to every other; for a variable with more values, every row of its
  neighbours, since each ΔE of a neighbour s, F(k) − F(a_s) over the values k of s, changes with F(a_s). The weights w_k = e^(−β ΔE_k) of
  all moves are the leaves of a tree of sums with fan-out 8, every entry the sum of the 8 below it. A uniform number
  times the total, followed down the tree, lands on move k with probability

  ```
  w_k / Σ_l w_l
  ```

  A move gives new weights to the rows it changed. The sums above them are recomputed from their 8 children, so
  rounding errors do not build up. A draw or an update touches log₈ n entries.
- **β moves at every step**, and with it every weight. The tree keeps the weights of an earlier β₀, and the draw is
  corrected: move k is accepted with probability

  ```
  e^(−(β − β₀) (ΔE_k − ΔE_min))        ΔE_min ≤ every ΔE
  ```

  and otherwise drawn again. Weight times acceptance is proportional to e^(−β ΔE_k), so the move made has exactly the
  distribution of this step's β. A draw that is not accepted costs a walk down the tree, not a step.
- **Recomputing the weights.** Rejected draws become more frequent as β leaves β₀. If they cost ρ · t at the t-th
  step after all weights were computed, and computing them costs Q, a cycle of K steps costs Q/K + ρ K/2 per step:
  least when ρ K²/2 = Q. So all weights are recomputed, at the current β, when the rejected draws since the last time
  have cost as much as that takes. Weights are kept relative to the largest, and are also recomputed if their total
  leaves the range of a float64.
- **Lowest states.** They are brought up to date from a list of the variables moved since they were last written,
  not copied.

A sparse step costs a fixed part plus one exponential per changed weight; an array step costs one pass over all n
rows. `run()` takes the sparse kernel where that is the cheaper step:

```
spins:        d < n / 30 − 7        d: mean number of neighbours of a spin
more values:  v < n / 9 − 20        v: rows a move changes, the values other than 0 of the moved variable's
                                    neighbours, averaged over the n rows
```

The constants are fitted to timings of both kernels on one machine (Apple M4). For spins that is fewer than about 3%
of the couplings nonzero from a few hundred spins up; for variables with more values, about a tenth of the value
pairs coupled.

Memory comes before speed. The array kernels need the couplings as an n × n float64 array, 8 n² bytes. If that is
more than half of the machine's memory, the kernel whose couplings take less memory runs: the sparse one when its
entries, 24 bytes per nonzero coupling, plus its tables where there are any, take less than the array.

### Elimination

One function, in `eliminate.c`, reads both layouts of the couplings, the array and the tables of the sparse kernel:
in either, the couplings of a value to the values of one neighbour are one stretch of memory.

- **A failing test stops early.** The test is made for every two values a and b of one variable, and most of them
  fail. Its sum starts from an upper bound, U(a) − U(b): the fields of a and of b with every neighbour of their
  variable held at its first value still allowed. That is one sum per value, used in every test the value enters.
  Each neighbour in turn then replaces its term by its minimum, which can only lower the total. Once the total is
  below 0 the test has failed, and the remaining neighbours are skipped.
- **The neighbour that decides is read first.** The neighbour that took a test below 0 moves to the front of the
  order in which that variable's neighbours are read, where the next tests meet it.
- **Only what changed is tested again**: the variables with a neighbour that lost a value.
- **No tolerance.** The comparison is the floating-point one, so two values that tie only up to rounding can both stay.
- **The reduced problem.** Value 0 must have no field and no couplings, so the first value left of each variable,
  k_r, becomes its value 0. For a value a of r and a value c of s,

  ```
  T'_rs(a, c) = T_rs(a, c) − T_rs(a, k_s) − T_rs(k_r, c) + T_rs(k_r, k_s)
  f'_r(a)     = G(a) − G(k_r),        G(a) = f_r(a) + Σ_s T_rs(a, k_s)
  ```

  which changes the energy by a constant. The couplings of a variable left with one value are then in the fields.
  `run()` picks the kernel again for the reduced problem.

### Recombination

`fold.c` has one function for spins on the array kernel, one for spins on the sparse kernel, and one for variables
with more values, which reads both layouts as the elimination does.

- **Clusters.** Each differing variable starts as its own set, and two sets are merged whenever a variable of one
  interacts with a variable of the other (union-find). Each set then has one sum, Δ(G). The work is the couplings of
  the differing variables: their stored neighbours on the sparse kernels, their rows of the array otherwise.
- **No tolerance.** Two differing variables interact when I_rs is not exactly 0. Where it is nonzero only by rounding,
  two clusters are merged that could have been chosen separately: a choice is lost, exactness is not.
- **Two states per repetition.** Each kernel keeps the lowest state of the first and of the second half of its
  steps: at the midpoint the current state becomes the lowest so far of the second half. Where the split falls
  matters little: at a quarter, three quarters or nine tenths of the steps the results were the same within the
  spread between seeds. Without a split most of the gain is lost, because the lowest state of a whole restart is
  usually the state it started from.
- **Rounds.** The lowest of three states, the repetition's two and the state it started from, becomes the best state
  and takes clusters from the other two, round after round until its energy, a float64, no longer falls. The first
  repetition has only its two. A repetition brought nothing when that energy is not below the energy of the state it
  started from.

## Usage

`make` (GCC or Clang) compiles `pfa/native/pfa.c` for the current machine. It includes the four kernels: `spin.c`
and `categorical.c` for couplings as an array, and `sparse_spin.c` and `sparse_categorical.c`, which draw from the
tree of sums of `sums.c`; and `eliminate.c` and `fold.c`. The two array kernels use the one SIMD path that matches
the machine; the rest is the same code on every path:

| path | files | used on |
|---|---|---|
| SVE | `spin_sve.c`, `categorical_sve.c` | ARM with SVE (Graviton3 and 4, NVIDIA Grace, A64FX, ...) |
| NEON | `spin_neon.c`, `categorical_neon.c` | every other 64-bit ARM (Apple Silicon, Graviton2, Ampere Altra, ...) |
| AVX-512 | `spin_avx512.c`, `categorical_avx512.c` | x86-64 with AVX-512 |
| AVX2 | `spin_avx2.c`, `categorical_avx2.c` | x86-64 with AVX2 |
| scalar | `spin_scalar.c`, `categorical_scalar.c` | anything else |

`pfa.simd_path()` reports which one was built; `make -B ARCH_FLAGS="-mavx2 -mfma"` rebuilds for another
target.

`pfa.run()` returns the lowest-energy state found, as ±1 spins (int8), or with `sizes` as the value of each variable
(int64). Its parameters:

| parameter | what it is | default | notes |
|---|---|---|---|
| `couplings` | the couplings, an n × n symmetric matrix with zero diagonal: J for spins, or with `sizes` between the values; an array or a `scipy.sparse` matrix | required | The format does not change the result: `run()` picks the array or the sparse kernels from the number of nonzero couplings (see Sparse couplings). The array kernels read a C-contiguous float64 array in place; any other input is converted once, a `scipy.sparse` matrix to n × n float64. If that array would take more than half of the machine's memory, the kernel that needs less memory runs. |
| `fields` | the fields, a length-n vector: h for spins, or with `sizes` of the values | `None` (zeros) | Enter only through F, so they cost nothing per step. |
| `sizes` | the number of values of each variable | `None` (spins) | Variable r takes a value in 0 … `sizes[r]` − 1. Each value other than 0 has a row and column of `couplings` and an entry of `fields`, ordered by variable (variable 0's values 1 … `sizes[0]` − 1 first), so n = Σ (`sizes[r]` − 1). The energy is the fields of the values taken plus each coupling between two of them once: ½ xᵀ C x + fᵀ x with x_k = 1 for the values taken. Couplings between values of the same variable must be zero. Values that no optimum needs are removed first (see Elimination). |
| `beta_max` | the final inverse temperature | `1.0` | Measured in 1/energy: scaling the energy by α scales the best `beta_max` by 1/α, so the default suits only ΔE of order one. A linear schedule is nearly as good as optimized ones (Isakov et al.). |
| `num_steps` | steps per repetition, the length of the anneal | `1000` | The budget of one repetition: on the array kernels num_steps · n field updates, or num_steps / V moves per variable. Longer anneals track the cooling more closely and reach lower energies. |
| `num_reps` | repetitions of the anneal, R | `1` | The first starts from a random value for every variable; each later one restarts from the best state so far with β rising from u/(R−1)·`beta_max`, u being 1 plus the number of restarts so far that did not lower the best energy (see Repetitions). After each repetition the best state takes the clusters of variables that the repetition has lower (see Recombination). |
| `seed` | random seed | `42` | Sets the first repetition's state and every repetition's Gumbel noise vector and offsets, or on the sparse kernels its draws, so results are reproducible. |
| `tabu_lambda` | the tabu penalty λ, in temperatures | `3.0` | Divides the weight of a variable's moves by e^λ just after it moved (see Tabu). Rarely worth changing. |
| `tabu_tenure` | the tabu tenure, as a fraction of the number of variables V | `0.25` | A variable stays tabu for `tabu_tenure` · V steps; 0 turns the tabu off. Rarely worth changing. |

A problem in one-hot form, ½ xᵀ C x + fᵀ x over binary x with one 1 per group, converts with
`pfa.from_one_hot(couplings, fields, groups)`, with `groups[i]` the variable that x_i belongs to. It returns the
couplings, fields, sizes and the energy offset between the two forms; `pfa.to_one_hot(values, groups)` converts a
result back. A problem over 0/1 variables converts to spins with `pfa.from_binary(couplings, fields, offset)`, and
`pfa.energy(state, couplings, fields, offset)` is ½ sᵀ J s + hᵀ s + offset for a ±1 or a 0/1 state.

`examples/spins.py` solves a small problem given over 0/1 variables, `examples/colouring.py` colours a graph with
several values per variable, and `examples/throughput.py` measures steps per second.
