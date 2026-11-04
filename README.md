# Pretty Fast Annealer

Fast solvers for the Ising spin glass problem: find the spins s ∈ {−1, +1}ⁿ that minimize the energy

```
E(s) = ½ sᵀ J s + hᵀ s
```

for a symmetric coupling matrix J (zero diagonal) and fields h.

Every implementation in this project runs the same annealer (lets call it PFA). It flips one spin at a time while the
inverse temperature β rises linearly from 0 to `beta_max`. What differs from simulated annealing in how it chooses the spin.

Classic simulated annealing proposes a random spin and accepts the flip with probability min(1, e^(−β ΔE)), where
ΔE is the energy change of the flip. As the temperature drops, almost every proposal is rejected, so most of the
work goes into flips that never happen.

PFA is rejection-free: every step flips a spin. It considers all n spins at once and picks spin j with probability
proportional to e^(−β ΔE_j). Flips that lower the energy are strongly favored, flips that raise it stay possible,
and no step is wasted.

Sampling from that distribution directly would take n exponentials and a normalization at every step. The
[Gumbel-max trick](https://lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/) avoids both: if
each g_j is drawn from the standard Gumbel distribution, then argmax_j(−β ΔE_j + g_j) is distributed exactly as
above. Choosing the next spin becomes one multiply, one add and one comparison per spin. To keep the noise cheap,
PFA draws one Gumbel vector at the start and shifts it by one position after every flip, so each spin sees a
different noise value at each step: a cheap stand-in for fresh noise.

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

It runs the same algorithm as `pfa_plain`, with each flip's update and selection fused into one pass over memory.
Delta energies and energies stay float64. Only the selection scores are compared in float32, which can change a pick
only when two scores agree to about 7 digits; in testing that happened about once in 50,000 steps or less. With
`num_reps=1`, `pfa_native` draws the same random numbers as `pfa_plain`, so the two follow the same trajectory
until such a step, and statistically equivalent ones after it.

Build it once with a C compiler using `make`. This compiles `ising_solvers/native/pfa_native.c` for the current machine. That file holds the anneal loop and
includes the one SIMD path that matches the target, each in its own file:

| path | file | used on |
|---|---|---|
| SVE | `pfa_native_sve.c` | ARM with SVE (Graviton3 and 4, NVIDIA Grace, A64FX, ...) |
| NEON | `pfa_native_neon.c` | every other 64-bit ARM (Apple Silicon, Graviton2, Ampere Altra, ...) |
| AVX-512 | `pfa_native_avx512.c` | x86-64 with AVX-512 |
| AVX2 | `pfa_native_avx2.c` | x86-64 with AVX2 |
| scalar | `pfa_native_scalar.c` | anything else |

`pfa_native.simd_path()` reports which one was built. To build for a different target, override the
architecture flags, e.g. `make ARCH_FLAGS=-mavx2`.
