import numpy as np
import time
from ising_solvers import utils

# beta_max does not affect flips per second benchmark
beta_max = 1.0

# number of spin flips per run. does not affect flips per second much, as long
# as it amortizes the initialization time
num_flips = 1000

# system size. larger systems will have lower flips per second
num_spins = 1000

# define hamiltonian

couplings = np.random.normal(0, 1, (num_spins, num_spins))
couplings = (couplings + couplings.T) / 2
np.fill_diagonal(couplings, 0.0)

fields = np.random.normal(0, 1, num_spins)

offset = 2.5

# find num_reps that maximized flips per second
def benchmark_solver(solver):
    flips_per_second_max = 1
    num_reps = 1
    num_reps_factor = 1
    while True:
        t = time.monotonic()
        spins_min = solver.run(
            couplings,
            fields,
            beta_max,
            num_flips,
            num_reps,
        )
        dt = time.monotonic() - t
        flips_per_second = (num_flips * num_reps) / dt

        print(num_reps, dt, flips_per_second, flips_per_second_max)

        if flips_per_second > flips_per_second_max:
            flips_per_second_max = flips_per_second
        else:
            num_reps /= 1.0 + num_reps_factor
            num_reps_factor /= 2

        num_reps *= 1.0 + num_reps_factor
        num_reps = int(num_reps)

        if num_reps_factor < 0.01:
            break

    print(
        f"Optimal perf: {flips_per_second_max} flips per second @ {num_reps} repetitions"
    )


try:
    from ising_solvers import pfa_cpu as solver

    benchmark_solver(solver)
except Exception as e:
    print("Failed running on CPU:", e)

try:
    from ising_solvers import pfa_gpu as solver

    benchmark_solver(solver)
except Exception as e:
    print("Failed running on GPU:", e)

try:
    from ising_solvers import pfa_tpu as solver

    benchmark_solver(solver)
except Exception as e:
    print("Failed running on TPU:", e)
