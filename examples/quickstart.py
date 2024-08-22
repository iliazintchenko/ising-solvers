import numpy as np
from ising_solvers import pfa_cpu, pfa_gpu, utils

couplings = np.array(
    [
        [0.0, -1.1, 0.2, 0.15],
        [-1.1, 0.0, -1.5, 0.8],
        [0.2, -1.5, 0.0, -1.0],
        [0.15, 0.8, -1.0, 0.0],
    ]
)
fields = np.array([0.1, -0.2, 0.3, -0.1])
offset = 2.5

spins_binary = np.array([0, 0, 0, 1])

energy_binary = utils.get_energy(spins_binary, couplings, fields, offset)

couplings, fields, offset = utils.binary_to_ising(couplings, fields, offset)

spins_ising = 2 * spins_binary - 1

energy_ising = utils.get_energy(spins_ising, couplings, fields, offset)

np.testing.assert_almost_equal(energy_ising, energy_binary)

beta_max = 3.0
num_flips = 1000
num_reps = 100

spins_min = pfa_cpu.run(
    couplings,
    fields,
    beta_max,
    num_flips,
)
energy_min = utils.get_energy(spins_min, couplings, fields, offset)
print("Ground state:", spins_min, energy_min)

spins_min = pfa_gpu.run(
    couplings,
    fields,
    beta_max,
    num_flips,
    num_reps,
)
energy_min = utils.get_energy(spins_min, couplings, fields, offset)
print("Ground state GPU:", spins_min, energy_min)
