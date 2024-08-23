import numpy as np
from ising_solvers import utils

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

from ising_solvers import pfa_cpu

spins_min = pfa_cpu.run(
    couplings,
    fields,
    beta_max,
    num_flips,
    num_reps,
)
energy_min = utils.get_energy(spins_min, couplings, fields, offset)
print("Ground state CPU:", spins_min, energy_min)

try:
    from ising_solvers import pfa_gpu

    spins_min = pfa_gpu.run(
        couplings,
        fields,
        beta_max,
        num_flips,
        num_reps,
    )
    energy_min = utils.get_energy(spins_min, couplings, fields, offset)
    print("Ground state GPU:", spins_min, energy_min)
except Exception as e:
    print("Failed running on GPU:", e)

try:
    from ising_solvers import pfa_tpu

    spins_min = pfa_tpu.run(
        couplings,
        fields,
        beta_max,
        num_flips,
        num_reps,
    )
    energy_min = utils.get_energy(spins_min, couplings, fields, offset)
    print("Ground state TPU:", spins_min, energy_min)
except Exception as e:
    print("Failed running on TPU:", e)
