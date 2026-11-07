# Pretty Fast Annealer

PFA is a fast solver for Ising problems: find the spins s ∈ {−1, +1}ⁿ that minimize

```
E(s) = ½ sᵀ J s + hᵀ s
```

for symmetric couplings J (zero diagonal) and fields h. Flipping spin j changes the energy by

```
ΔE_j = −2 s_j (h_j + Σ_k J_jk s_k)
```

## Algorithm

PFA flips one spin per step, chosen from all n with probability

```
p_j = e^(−β ΔE_j) / Σ_k e^(−β ΔE_k)
```

while the inverse temperature β rises linearly from 0 to `beta_max`. Downhill flips are strongly favored; uphill flips
stay possible.

**What it samples.** As a continuous-time process, a flip with energy change ΔE happens at rate e^(−β ΔE) and its
reverse at rate e^(+β ΔE). Their ratio, e^(−2β ΔE), is detailed balance for e^(−2β E): PFA at β equilibrates at 2β.
Taking one flip per step, without the waiting times, also weights each state by its total exit rate
R(s) = Σ_k e^(−β ΔE_k), so states with many cheap exits are visited more often. An optimizer doesn't care: it keeps
the best state it has seen.

**Sampling a step.** Drawing from p_j directly costs n exponentials and a normalization per step. The
[Gumbel-max trick](https://lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/) removes both: with
each g_j an independent standard Gumbel, g = −log(−log u) for u uniform on (0, 1),

```
P( argmax_j (−β ΔE_j + g_j) = i ) = p_i
```

so picking a spin costs one multiply, one add and one comparison per spin. The noise is drawn once: at step t, spin j
reads g_((j + o_t) mod n), with a fresh random offset o_t. Each step on its own is exact Gumbel-max sampling; across
steps the same values return permuted, a cheap stand-in for fresh noise. A random offset, rather than a shift by one,
keeps large noise values from marching through neighbouring indices, so results don't depend on the spin order.

**Soft tabu.** When spin i flips, at inverse temperature β_i, its ΔE is raised by λ/β_i, λ = 3 temperatures at that
moment, for the next n/4 flips (the tenure):

```
p_i ∝ e^(−β (ΔE_i + λ/β_i))
```

Right after the flip this divides its weight by e^λ; as the anneal cools, by e^(λβ/β_i). Early on, while β grows fast
relative to β_i, the tabu is nearly hard. A tabu spin i still outweighs a free spin k when ΔE_i + λ/β_i < ΔE_k, so a
move worth more than the penalty, like stepping back down from an expensive climb, still happens. Among moves of
similar cost, the chain picks spins it hasn't just flipped. Hot, this spreads the flips over more spins; cold, it
keeps the chain moving between low-energy states instead of re-flipping the same few spins.

λ is in temperatures and the tenure a fraction of n, so the rule is unchanged when the energy is rescaled (J, h, β →
cJ, ch, β/c) or n grows. The two values are measured, not derived, and fixed in the code. The tabu breaks detailed
balance; an optimizer doesn't need it. With the tabu, the anneal reaches lower energies in the same number of flips,
and its best `beta_max` is about 2× higher than without it.

## Compared with simulated annealing

The reference is simulated annealing (SA) with the optimizations of Isakov et al., [Optimised simulated annealing for
Ising spin glasses](https://arxiv.org/abs/1401.1084), built the same way as PFA's implementation below: full float64
coupling rows, one replica, one core. Its optimizations for low-precision couplings, many replicas or sparse graphs
(lookup tables, multi-spin coding of 64 replicas, bipartite updates) are left out on both sides.

SA proposes a random spin and accepts the flip with probability min(1, e^(−β ΔE_j)). As the anneal cools, it rejects
almost every proposal. PFA never rejects. Each ingredient of optimized SA has a counterpart in PFA:

| optimized SA | PFA | what it does |
|---|---|---|
| Metropolis acceptance of one proposed spin | Boltzmann selection over all n spins | decides which flip happens |
| sequential sweeps | soft tabu | spreads the work over distinct spins |
| stored local fields | stored local fields | ΔE is read, not computed |
| precomputed random numbers at random offsets | one Gumbel vector at random offsets | cheap randomness |

**Cost per step.** A flip in either method updates all n local fields. Per spin, an SA flip moves 24 bytes: a float64
coupling read, a float64 field read and written. A PFA flip also reads the spin and its noise value, 4 bytes each:
32 bytes, 4/3 of an SA flip (measured: 1.2–1.4×). A rejected SA proposal reads one field and one random number, about
1/n of a flip, so flips set the cost. In SA flips per spin:

```
PFA:  (4/3) · flips / n          SA:  (accepted flips + proposals / n) / n
```

**Descent.** Metropolis accepts every downhill move, so it takes them in sweep order, whatever their size. PFA
weights each by e^(β |ΔE_j|) and takes the biggest first. Each PFA flip does more of the descent, and the flips saved
outweigh the 4/3.

**Sweeps and the tabu.** SA that picks spins at random proposes some spins again and again while missing others:
after n proposals, about 1/e ≈ 37% of the spins have never been tried. Sweeping in order proposes every spin once per
n proposals, and does better. PFA has no proposals to order; its repeat-avoidance is the soft tabu. A sweep is a hard
rule on proposals that keeps the Boltzmann distribution stationary; the tabu is a soft rule on flips that does not.

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
β. It can run cold enough to resolve the objective without freezing on the constraints. This advantage belongs to
rejection-free selection as such.

**Long runs.** Late in the anneal, what counts is how far the chain still moves without climbing. At a fixed low
temperature, PFA sits lower in energy than SA and still drifts further from where it started. (Equal temperature means
PFA at β against SA at 2β.)

**Sparse implementations.** The comparison holds on sparse problems as it stands. Written for sparsity, SA updates
only the flipped spin's d neighbours, and that is all it needs. PFA's selection would still compare all n scores per
step, because every new offset and every change of β moves every score. A sparse PFA needs a sum tree over the rates
(the n-fold way) or a priority queue of random clocks (the next-reaction method), at about d log n per flip with β
held fixed between rebuilds. That version is not compared here.

## Implementation

`pfa_native` runs PFA in compiled C with SIMD, on one core, on any couplings and fields. A flip changes the ΔE of
every spin it couples to, so every flip reads one full float64 row of couplings and updates all n local fields. The
selection rides along in the same pass, for free, and a flip costs the same on every problem of a given size:

- **Local fields.** Each spin stores φ_j = (h_j + Σ_k J_jk s_k) / 2 in float64, so ΔE_j = −4 s_j φ_j. Flipping spin
  i adds −s_i J_ij to every φ_j (s_i before the flip). After F flips, rounding has added at most about F · 2⁻⁵³ of
  the largest field: negligible.
- **Scores.** Dividing −β ΔE_j + g_j by 4β leaves the argmax unchanged and gives

  ```
  s_j φ_j + g_j / (4β)
  ```

computed in float32. Rounding can change a pick only when two scores agree to 7 digits (2⁻²⁴ ≈ 6·10⁻⁸). - **Argmax
without indices.** A vector argmax carries an index next to every running maximum, which roughly doubles the selection
work. The pass instead keeps only maxima and records which chunk of 64 spins (8 vectors on SVE) reached the largest
one. Afterwards it rescores that chunk, still in cache, and returns its first spin at the maximum. The extra cost is
one reduction per chunk plus 64 rescored spins per flip, about 64/n of a pass. The rescore repeats the pass's float
operations exactly (the build stops the compiler from fusing multiply-adds on its own), so it finds the same maximum,
and ties go to the smallest index. - **Tabu for free.** The penalty, c_i = λ / (4β_i) in score units, is folded into
the stored field, φ̃_i = φ_i − s_i c_i, so that s_i φ̃_i = s_i φ_i − c_i, and removed when i leaves the list or flips
again. The pass reads nothing extra.

## Usage

Build with `make` and GCC or Clang. This compiles `ising_solvers/native/pfa_native.c` for the current machine,
which includes the one SIMD path that matches the target:

| path | file | used on |
|---|---|---|
| SVE | `pfa_native_sve.c` | ARM with SVE (Graviton3 and 4, NVIDIA Grace, A64FX, ...) |
| NEON | `pfa_native_neon.c` | every other 64-bit ARM (Apple Silicon, Graviton2, Ampere Altra, ...) |
| AVX-512 | `pfa_native_avx512.c` | x86-64 with AVX-512 |
| AVX2 | `pfa_native_avx2.c` | x86-64 with AVX2 |
| scalar | `pfa_native_scalar.c` | anything else |

`pfa_native.simd_path()` reports which one was built; `make -B ARCH_FLAGS="-mavx2 -mfma"` rebuilds for another target.
Then:

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
