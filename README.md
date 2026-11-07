# Pretty Fast Annealer

Fast solvers for the Ising spin glass problem: find the spins s ∈ {−1, +1}ⁿ that minimize the energy

```
E(s) = ½ sᵀ J s + hᵀ s
```

for a symmetric coupling matrix J (zero diagonal) and fields h. Flipping spin j changes the energy by

```
ΔE_j = −2 s_j (h_j + Σ_k J_jk s_k)
```

Every implementation in this project runs the same annealer (let's call it PFA). It flips one spin at a time while the
inverse temperature β rises linearly from 0 to `beta_max`. What differs from simulated annealing is how it chooses the
spin.

Classic simulated annealing proposes a random spin j and accepts the flip with probability min(1, e^(−β ΔE_j)). As the
temperature drops, almost every proposal is rejected.

PFA is rejection-free: every step flips a spin. It considers all n spins at once and picks spin j with probability

```
p_j = e^(−β ΔE_j) / Σ_k e^(−β ΔE_k)
```

Flips that lower the energy are strongly favored, and flips that raise it stay possible.

**What the chain samples.** Read as a continuous-time process, a flip that changes the energy by ΔE happens at rate
e^(−β ΔE), and the flip back at rate e^(+β ΔE). Their ratio, e^(−2β ΔE), is the detailed-balance condition for the
Boltzmann distribution e^(−2β E): at inverse temperature β, PFA's rates are in equilibrium at 2β. PFA takes one flip
per step and skips the waiting times between flips, so it visits each state in proportion to e^(−2β E(s)) R(s), where
R(s) = Σ_k e^(−β ΔE_k) is the state's total rate of leaving. States with many cheap moves out are visited more often
than their Boltzmann weight. For an optimizer this does not matter: it keeps the lowest state it has visited.

**Sampling a step.** Sampling p_j directly would take n exponentials and a normalization at every step. The
[Gumbel-max trick](https://lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/) avoids both. If
each g_j is drawn independently from the standard Gumbel distribution, g = −log(−log u) with u uniform on (0, 1), then

```
P( argmax_j (−β ΔE_j + g_j) = i ) = p_i
```

Choosing the next spin becomes one multiply, one add and one comparison per spin. To keep the noise cheap, PFA draws
one Gumbel vector g at the start and, at step t, gives spin j the value g_((j + o_t) mod n), with the offset o_t drawn
at random at every step. Within one step, the n spins receive n distinct entries of an independent Gumbel vector, so
each step on its own is exact Gumbel-max sampling. Across steps the same n values come back permuted, a cheap stand-in
for fresh noise. The offset is random, not advanced by one per step: a fixed shift would carry the largest noise
values through neighbouring spin indices, which ties results to how the spins happen to be ordered.

On dense problems the selection is cheap. Flipping a spin changes the ΔE of every other spin, so each flip has to
touch all n spins anyway to update them, and the selection rides along in the same pass.

## Solvers

| module | runs on |
|---|---|
| `pfa_plain` | numpy, one repetition; the readable reference |
| `pfa_cpu` | numpy, many repetitions at once |
| `pfa_gpu` | PyTorch on a GPU |
| `pfa_tpu` | PyTorch/XLA on a TPU |
| `pfa_native` | compiled C with SIMD, one core |

`pfa_native` is built for one kind of problem: dense (all-to-all) spin glasses with fields and floating-point
couplings. Each flip reads a full row of float64 couplings, so on sparse problems it pays for every zero, and
integer or low-precision couplings gain nothing from their structure.

It runs the algorithm of `pfa_plain` with one change to the selection, a soft tabu, and with each flip's update and
selection fused into one pass over memory:

- **Local fields.** Each spin keeps φ_j = (h_j + Σ_k J_jk s_k) / 2 in float64, so that ΔE_j = −4 s_j φ_j. Flipping
  spin i changes every φ_j by −s_i J_ij (with s_i before the flip): one add or subtract of row i of the couplings,
  which is read in place, without a copy. Each flip adds one rounding error per field, so after F flips the error is
  at most about F · 2⁻⁵³ of the largest field: float64 keeps it negligible for any practical run.
- **Scores.** Dividing −β ΔE_j + g_j by 4β does not change the argmax and gives the score

  ```
  s_j φ_j + g_j / (4β)
  ```

  one multiply and one fused multiply-add per spin. Scores are computed and compared in float32. Its relative
  precision, 2⁻²⁴ ≈ 6·10⁻⁸, means rounding can change a pick only when two scores agree to about 7 digits.
- **Soft tabu.** For the n/4 flips after spin i flips, its weight is divided by e^λ, with λ = 3:

  ```
  p_i ∝ e^(−β ΔE_i − λ)
  ```

  This is the weight of a flip whose ΔE is higher by λ temperatures, λ/β. A tabu spin i still beats a free spin k
  whenever ΔE_i < ΔE_k − λ/β, so a move that is better by more than λ temperatures, such as stepping back down from an
  expensive climb, still happens. Among moves of similar cost, the chain prefers spins it has not just flipped, so its
  flips spread over more distinct spins. That matters most early in the anneal, while many moves cost about the same
  in temperature units.

  The penalty costs nothing per spin. In score units it is c_i = λ / (4β), applied by shifting the stored field to
  φ̃_i = φ_i − s_i c_i, so that s_i φ̃_i = s_i φ_i − c_i. The shift is removed when i leaves the list or flips again.
  The shift is fixed at the flip, λ/β_flip in ΔE, so measured in the current temperature it is λ β/β_flip ≥ λ and grows
  while i stays tabu.

  λ is measured in temperatures and the tenure as a fraction of n. So the rule does not change when all energies are
  rescaled (J, h, β → cJ, ch, β/c) or when the problem grows. The two values themselves were chosen by measurement,
  not derived. The tabu breaks detailed balance, so the chain no longer samples a Boltzmann distribution; an
  optimizer that keeps the best state it has seen does not need one.

The soft tabu reaches lower energies at the same number of flips. Its best `beta_max` is about 2× higher than for
the other solvers.

Build it once with a C compiler using `make`. This compiles `ising_solvers/native/pfa_native.c` for the current
machine. That file holds the anneal loop and includes the one SIMD path that matches the target, each in its own file:

| path | file | used on |
|---|---|---|
| SVE | `pfa_native_sve.c` | ARM with SVE (Graviton3 and 4, NVIDIA Grace, A64FX, ...) |
| NEON | `pfa_native_neon.c` | every other 64-bit ARM (Apple Silicon, Graviton2, Ampere Altra, ...) |
| AVX-512 | `pfa_native_avx512.c` | x86-64 with AVX-512 |
| AVX2 | `pfa_native_avx2.c` | x86-64 with AVX2 |
| scalar | `pfa_native_scalar.c` | anything else |

`pfa_native.simd_path()` reports which one was built. To build for a different target, override the
architecture flags, e.g. `make ARCH_FLAGS=-mavx2`.

## pfa_native compared with simulated annealing

The reference is simulated annealing as optimized by Isakov et al.,
[Optimised simulated annealing for Ising spin glasses](https://arxiv.org/abs/1401.1084), on the problems `pfa_native`
is built for: fully connected, with fields and couplings of arbitrary precision. That SA keeps every spin's local
field, so a proposal reads its ΔE instead of computing it. It sweeps the spins in order and reads its random numbers
from a precomputed table at random offsets. `pfa_native` takes the stored fields and the reused random numbers from
it. Its other techniques (lookup tables for integer couplings, multi-spin coding of 64 replicas, bipartite updates)
need low-precision couplings, many replicas or sparse graphs. `pfa_native` deliberately runs one replica on one core,
and the comparison does the same.

**Cost of a step.** In either method a flip updates all n local fields. Per spin, an SA flip reads a float64 coupling
and reads and writes a float64 field, 24 bytes. A PFA flip also reads the spin and its noise value, 4 bytes each, for
32 bytes. So a PFA flip costs about 32/24 = 4/3 of an SA flip (measured: 1.2–1.4×). A rejected SA proposal reads one
field and one random number, about 1/n of a flip. On dense problems rejections are nearly free, and work, counted in
SA flips per spin, is

```
PFA:  (4/3) · flips / n          SA:  (accepted flips + proposals / n) / n
```

**Descent.** Metropolis makes flip j with probability proportional to min(1, e^(−β ΔE_j)): it accepts every downhill
move, so it takes them in random order, whatever their size. PFA weights a downhill move by e^(β |ΔE_j|) and takes the
largest descents first. Uphill moves are weighted e^(−β ΔE_j) in both. So each PFA flip does more of the descent,
and the flips saved outweigh the 4/3 cost of each flip.

**Constraints and large penalties.** Many hard problems are an objective plus constraints written as penalties:

```
E(s) = P · C(s) + O(s)
```

C(s) ≥ 0 counts the violated constraints (C = 0 on feasible states), O(s) is the objective, and the penalty P is much
larger than the scale ε of the changes in O. The hard case is when every single flip from a feasible state violates
a constraint, as with one-hot or cardinality constraints. Then every move costs ΔE_i = P + ΔO_i or more, and moving
from one feasible state to another takes at least two flips: one that breaks a constraint and one that repairs it.

For Metropolis, the first flip of such a pair is accepted with probability e^(−β (P + ΔO_i)). Between feasible
states, SA therefore moves at a rate of about e^(−β P) per proposal, and it can move only while β ≲ 1/P. To tell
better objective values from worse, it needs β ≳ 1/ε. When P/ε ≫ 1 there is no β at which SA does both: at the β that
resolves the objective, a move between feasible states takes about e^(P/ε) proposals, so the feasible configuration
freezes long before the objective is optimized.

PFA's selection is unchanged when every ΔE is shifted by the same constant, because the shift cancels in the
normalization of p_j. From a feasible state, the moves that break one constraint all carry the same P, so

```
p_i ∝ e^(−β ΔO_i)
```

The choice of which constraint to break is set by the objective alone, whatever P is, and PFA leaves every state in
one step, at any β. So it can run cold enough to resolve the objective without the penalties freezing it.

This advantage belongs to rejection-free selection as such. On dense problems that selection is free, because the
update pass visits every spin anyway.

**Long runs.** In long anneals most of the run is spent cold, and what counts there is how far the chain still moves
without climbing. Held at a fixed low temperature, PFA with the soft tabu sits at a lower energy than SA and still
drifts further from where it started. PFA without the tabu drifts no further than SA. So in long runs the advantage
comes from the tabu. (Equal temperature means PFA at β against SA at 2β, from the equilibrium above.)

**Sparse problems.** Both implementations read full coupling rows, so a sparse problem costs each the same as a dense
one, and the comparison carries over. An implementation written for sparse problems updates only the flipped spin's d
neighbours. For SA that is the whole change. PFA's selection, however, still compares all n scores at every step,
because each new noise offset and each change of β changes every score. A sparse selection needs per-spin random clocks
in a priority queue, as in the n-fold way, at about d log n per flip; that version is not compared here.
