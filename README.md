# Pretty Fast Annealer

A fast solver for the Ising spin glass problem: find the spins s ∈ {−1, +1}ⁿ that minimize

```
E(s) = ½ sᵀ J s + hᵀ s
```

for symmetric couplings J (zero diagonal) and fields h. Flipping spin j changes the energy by

```
ΔE_j = −2 s_j (h_j + Σ_k J_jk s_k)
```

## The annealer

PFA flips one spin per step while the inverse temperature β rises linearly from 0 to `beta_max`. It differs from
simulated annealing in how it picks the spin.

Simulated annealing proposes a random spin and accepts the flip with probability min(1, e^(−β ΔE_j)). As the anneal
cools, it rejects almost every proposal.

PFA never rejects. Every step flips a spin, chosen from all n with probability

```
p_j = e^(−β ΔE_j) / Σ_k e^(−β ΔE_k)
```

Downhill flips are strongly favored; uphill flips stay possible.

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

**Why it's cheap.** A flip changes the ΔE of every spin it couples to. On a dense problem that is all of them, so
every flip already makes a pass over all n spins, and the selection rides along for free.

**Soft tabu.** For n/4 flips after spin i flips, its weight is divided by e^λ, with λ = 3:

```
p_i ∝ e^(−β ΔE_i − λ)
```

as if its ΔE were λ temperatures (λ/β) higher. A tabu spin i still outweighs a free spin k when
ΔE_i < ΔE_k − λ/β, so a move worth more than λ temperatures, like stepping back down from an expensive climb, still
happens. Among moves of similar cost, the chain picks spins it hasn't just flipped. Hot, this spreads the flips over
more spins; cold, it keeps the chain moving between low-energy states instead of re-flipping the same few spins.

λ is in temperatures and the tenure a fraction of n, so the rule is unchanged when the energy is rescaled
(J, h, β → cJ, ch, β/c) or n grows. The two values are measured, not derived. The tabu breaks detailed balance; an
optimizer doesn't need it. With the tabu, the anneal reaches lower energies in the same number of flips, and its best
`beta_max` is about 2× higher than without it.

## pfa_native

`pfa_native` runs PFA in compiled C with SIMD, on one core. It targets one kind of problem: dense spin glasses with
fields and real-valued couplings. It reads a full float64 coupling row per flip, so sparse problems pay for every
zero, and integer or low-precision couplings gain nothing from their structure. Each flip's update and selection are
fused into one pass over memory:

- **Local fields.** Each spin stores φ_j = (h_j + Σ_k J_jk s_k) / 2 in float64, so ΔE_j = −4 s_j φ_j. Flipping spin
  i adds −s_i J_ij to every φ_j (s_i before the flip): one pass over row i of the couplings, read in place. After F
  flips, rounding has added at most about F · 2⁻⁵³ of the largest field: negligible.
- **Scores.** Dividing −β ΔE_j + g_j by 4β leaves the argmax unchanged and gives

  ```
  s_j φ_j + g_j / (4β)
  ```

  one multiply and one fused multiply-add per spin, in float32. Rounding can change a pick only when two scores agree
  to 7 digits (2⁻²⁴ ≈ 6·10⁻⁸).
- **Tabu for free.** The penalty, c_i = λ / (4β) in score units, is folded into the stored field,
  φ̃_i = φ_i − s_i c_i, so that s_i φ̃_i = s_i φ_i − c_i, and removed when i leaves the list or flips again. The pass
  reads nothing extra. Fixed at the flip, the penalty grows as the anneal cools, to λ β/β_flip temperatures.

Build it with `make` and any C compiler. This compiles `ising_solvers/native/pfa_native.c` for the current machine,
which includes the one SIMD path that matches the target:

| path | file | used on |
|---|---|---|
| SVE | `pfa_native_sve.c` | ARM with SVE (Graviton3 and 4, NVIDIA Grace, A64FX, ...) |
| NEON | `pfa_native_neon.c` | every other 64-bit ARM (Apple Silicon, Graviton2, Ampere Altra, ...) |
| AVX-512 | `pfa_native_avx512.c` | x86-64 with AVX-512 |
| AVX2 | `pfa_native_avx2.c` | x86-64 with AVX2 |
| scalar | `pfa_native_scalar.c` | anything else |

`pfa_native.simd_path()` reports which one was built; `make ARCH_FLAGS=-mavx2` builds for another target. Then:

```python
from ising_solvers import pfa_native

spins = pfa_native.run(couplings, fields, beta_max=1.0, num_flips=100 * len(couplings), num_reps=10)
```

`examples/quickstart.py` solves a small problem and measures throughput on a dense one.

## pfa_native vs. simulated annealing

The reference is the optimized simulated annealing of Isakov et al.,
[Optimised simulated annealing for Ising spin glasses](https://arxiv.org/abs/1401.1084), on the problems `pfa_native`
is built for: fully connected, with fields and real-valued couplings. `pfa_native` borrows two of its techniques:
stored local fields, so ΔE is read rather than computed, and precomputed random numbers read at random offsets. The
others (lookup tables for integer couplings, multi-spin coding of 64 replicas, bipartite updates) need low-precision
couplings, many replicas or sparse graphs. `pfa_native` deliberately runs one replica on one core, and so does the
comparison.

**Cost per step.** A flip in either method updates all n fields. Per spin, an SA flip moves 24 bytes: a float64
coupling read, a float64 field read and written. A PFA flip also reads the spin and its noise value, 4 bytes each:
32 bytes, 4/3 of an SA flip (measured: 1.2–1.4×). A rejected SA proposal reads one field and one random number, about
1/n of a flip. On dense problems rejections are nearly free and flips set the cost. In SA flips per spin:

```
PFA:  (4/3) · flips / n          SA:  (accepted flips + proposals / n) / n
```

**Descent.** Metropolis accepts every downhill move, so it takes them in random order, whatever their size. PFA weights
each by e^(β |ΔE_j|) and takes the biggest first. Each PFA flip does more of the descent, and the flips saved outweigh
the 4/3.

**Constraints.** Many hard problems are an objective plus constraints written as penalties:

```
E(s) = P · C(s) + O(s)
```

C(s) ≥ 0 counts the violated constraints, O(s) is the objective, and P is much larger than the scale ε of changes in
O. The hard case: every single flip from a feasible state breaks a constraint, as with one-hot or cardinality
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
rejection-free selection as such; on dense problems, that selection is free.

**Long runs.** Long anneals spend most of their time cold, where what counts is how far the chain still moves without
climbing. At a fixed low temperature, PFA sits lower in energy than SA and still drifts further from where it started.
(Equal temperature means PFA at β against SA at 2β.)

**Sparse problems.** Both implementations read full coupling rows, so on sparse problems the comparison is unchanged.
Written for sparsity, SA updates only the flipped spin's d neighbours, and that is all it needs. PFA's selection would
still compare all n scores per step, because every new offset and every change of β moves every score. A sparse PFA
needs per-spin random clocks in a priority queue, as in the n-fold way, at about d log n per flip. That version is not
compared here.
