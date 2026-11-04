# ising-solvers
Pretty fast solvers for Ising spin glasses

## Native single-core engine (`pfa_native`)

The same algorithm as `pfa_plain`, compiled to SIMD code. Each flip updates the delta energies and selects the
next spin in a single pass over memory. The results are bit-identical to `pfa_plain`: with `num_reps=1`,
`pfa_native.run(couplings, fields, beta_max, num_flips, 1, seed)` returns exactly the spins of
`pfa_plain.run(couplings, fields, beta_max, num_flips, seed)`.

Build it once with a C compiler:

```
make
```

This compiles `ising_solvers/native/pfa_native.c` for the current machine. The SIMD path follows from the target:

| path | used on |
|---|---|
| SVE | ARM with SVE (Graviton3, A64FX, ...) |
| NEON | every other 64-bit ARM (Apple Silicon, Graviton4, ...) |
| AVX-512 | x86-64 with AVX-512 |
| AVX2 | x86-64 with AVX2 |
| scalar | anything else |

`pfa_native.simd_path()` reports which one was built. To build for a different target, override the
architecture flags, e.g. `make ARCH_FLAGS=-mavx2`.

Speed on one core of an Apple M4 Max (NEON), random fully connected problem with fields:

| spins | `pfa_plain` | `pfa_native` |
|---|---|---|
| 1000 | 0.30M flips/s | 3.9M flips/s |
| 4000 | 0.064M flips/s | 0.76M flips/s |
