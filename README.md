# Pretty Fast Annealer

PFA solves Ising problems: find the spins s ∈ {−1, +1}ⁿ that minimize

```
E(s) = ½ sᵀ J s + hᵀ s
```

for symmetric couplings J (zero diagonal) and fields h. Flipping spin j changes the energy by

```
ΔE_j = −2 s_j (h_j + Σ_k J_jk s_k)
```

## Results

Against optimized simulated annealing (SA, compared in detail below), with `beta_max` tuned for every run length:

| what | measured as | result |
|---|---|---|
| spin glasses | work for the same energy | SA needs **1.5–2×** PFA's work |
| constrained problems with large penalties | work for the same energy | SA never gets there: with **15–1000×** PFA's work, it still ends above PFA at 10 flips per spin |
| cost of a flip | time per executed flip | a PFA flip costs **1.2–1.4×** an accepted SA flip; a rejected SA proposal, about 1/n of one |

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
(J, h, β → cJ, ch, β/c) or n grows. Both values are measured, not derived, and fixed. The tabu breaks detailed
balance, which an optimizer doesn't need. It roughly doubles the best `beta_max`.

## Compared with simulated annealing

The reference is SA as optimized by Isakov et al.,
[Optimised simulated annealing for Ising spin glasses](https://arxiv.org/abs/1401.1084), built like PFA's
implementation: full float64 coupling rows, one replica, one core. Their tricks for low-precision couplings, many
replicas or sparse graphs (lookup tables, multi-spin coding, bipartite updates) are left out on both sides.

SA proposes one spin at a time and accepts the flip with probability min(1, e^(−β ΔE_j)). As the anneal cools, it
rejects almost every proposal; PFA never rejects. Its two ingredients have PFA counterparts:

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

**Sparse implementations.** The comparison holds on sparse problems as it stands. Written for sparsity, SA updates
only the flipped spin's d neighbours, and that is all it needs. PFA's selection would still compare all n weights per
step, because every new noise offset and every change of β moves every weight. A sparse PFA needs a sum tree over the
rates (the n-fold way) or a priority queue of random clocks (the next-reaction method), at about d log n per flip with
β held fixed between rebuilds. That version is not compared here.

## Implementation

The annealer runs in compiled C with SIMD, on one core, on any couplings and fields. Every flip reads one full
float64 row of couplings, zeros included, and updates all n local fields; the selection rides along in the same pass.
A flip therefore costs the same on every problem of a given size. The tricks:

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
- **Energy tracked, not computed.** E += ΔE_i = −4 s_i φ_i at each flip, O(1). The best state is copied only when E
  reaches a new low.

## Usage

`make` (GCC or Clang) compiles `ising_solvers/native/pfa_native.c` for the current machine, with the one SIMD path
that matches it:

| path | file | used on |
|---|---|---|
| SVE | `pfa_native_sve.c` | ARM with SVE (Graviton3 and 4, NVIDIA Grace, A64FX, ...) |
| NEON | `pfa_native_neon.c` | every other 64-bit ARM (Apple Silicon, Graviton2, Ampere Altra, ...) |
| AVX-512 | `pfa_native_avx512.c` | x86-64 with AVX-512 |
| AVX2 | `pfa_native_avx2.c` | x86-64 with AVX2 |
| scalar | `pfa_native_scalar.c` | anything else |

`pfa_native.simd_path()` reports which one was built; `make -B ARCH_FLAGS="-mavx2 -mfma"` rebuilds for another
target. Then:

```python
from ising_solvers import pfa_native

spins = pfa_native.run(couplings, fields, beta_max=1.0, num_flips=100 * len(couplings), num_reps=10)
```

It returns the lowest-energy state seen, as ±1 spins (int8).

| parameter | what it is | default | theory |
|---|---|---|---|
| `couplings` | J, an n × n symmetric matrix with zero diagonal | required | Read in place when it is a C-contiguous float64 array, otherwise copied once. |
| `fields` | h, a length-n vector | `None` (zeros) | Enter only through the local fields φ, so they cost nothing per flip. |
| `beta_max` | the final inverse temperature | `1.0` | Measured in 1/energy: scaling the energy by c scales the best `beta_max` by 1/c, so the default suits only ΔE of order one. A linear schedule is nearly as good as optimized ones (Isakov et al.). |
| `num_flips` | flips per repetition, the length of the anneal | `1000` | The whole budget: num_flips · n field updates, or num_flips / n flips per spin. Longer anneals track the cooling more closely and reach lower energies. |
| `num_reps` | independent repetitions | `1` | Each starts from fresh random spins and noise; the best state over all of them is returned. |
| `seed` | random seed | `42` | Sets the initial spins, the Gumbel noise vector and the per-step offsets, so runs are reproducible. |

`examples/quickstart.py` solves a small problem and measures flips per second.
