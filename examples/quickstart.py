import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ising_solvers import pfa_native, utils

# a small problem in binary variables x ∈ {0, 1}, mapped to Ising spins s = 2x − 1
couplings = np.array(
    [
        [0.0, -1.1, 0.2, 0.15234],
        [-1.1, 0.0, -1.5, 0.8],
        [0.2, -1.5, 0.0, -1.0],
        [0.15234, 0.8, -1.0, 0.0],
    ]
)
fields = np.array([0.1, -0.2, 0.3, -0.1])
offset = 2.5

spins_binary = np.array([0, 0, 0, 1])
energy_binary = utils.get_energy(spins_binary, couplings, fields, offset)

couplings, fields, offset = utils.binary_to_ising(couplings, fields, offset)
energy_ising = utils.get_energy(2 * spins_binary - 1, couplings, fields, offset)
np.testing.assert_almost_equal(energy_ising, energy_binary)

spins_min = pfa_native.run(couplings, fields, beta_max=3.0, num_flips=1000, num_reps=100)
print("ground state:", spins_min, utils.get_energy(spins_min, couplings, fields, offset))

# throughput on a dense problem; beta_max does not affect it
n = 1000
rng = np.random.default_rng(0)
couplings = np.triu(rng.normal(0, 1, (n, n)), 1)
couplings += couplings.T
fields = rng.normal(0, 1, n)

num_flips = 100 * n
t = time.perf_counter()
pfa_native.run(couplings, fields, beta_max=1.0, num_flips=num_flips)
dt = time.perf_counter() - t
print(
    f"{n} spins, {pfa_native.simd_path()}: {num_flips / dt / 1e6:.2f} M flips/s, "
    f"{num_flips * n / dt / 1e9:.2f} G spin updates/s"
)
