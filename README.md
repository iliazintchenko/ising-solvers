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
PFA draws one Gumbel vector at the start and reuses it under a cyclic shift, so each spin sees a different noise
value at each step: a cheap stand-in for fresh noise.

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

- Each spin keeps its local field h + J s in float64, and flipping spin i adds −2 s_i times row i of the couplings.
  The coupling matrix is read in place, without a copy.
- Each step reads the noise vector at a random offset, so results do not depend on how the spins are ordered.
- Selection scores are compared in float32, which can change a pick only when two scores agree to about 7 digits.
- Soft tabu: for the n/4 flips after a spin flips, its selection weight is divided by e³, as if its ΔE were higher
  by 3 temperatures (3/β). This spreads the flips over more distinct spins, mostly while the anneal is still hot:
  on a 1000-spin Sherrington–Kirkpatrick instance, the share of flips that re-flip a spin flipped within the last
  n/4 flips falls from about 32% to 5% in the first quarter of the anneal and from 58% to 27% in the second. A flip
  that is worth much more than 3 temperatures, such as stepping back down from an expensive climb, still happens.
  It costs nothing per spin: the penalty is applied as a shift of the flipped spin's stored local field.

The soft tabu reaches lower energies at the same number of flips, by a wide margin on dense spin glasses. Its best
`beta_max` is about 2× higher than for the other solvers.

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
