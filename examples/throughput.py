"""Steps per second of the array spin kernel on a dense problem; beta_max does not affect it."""
import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pfa

n = 1000
rng = np.random.default_rng(0)
couplings = np.triu(rng.normal(0, 1, (n, n)), 1)
couplings += couplings.T
fields = rng.normal(0, 1, n)

num_steps = 100 * n
t = time.perf_counter()
pfa.run(couplings, fields, beta_max=1.0, num_steps=num_steps)
dt = time.perf_counter() - t
print(
    f"{n} spins, {pfa.simd_path()}: {num_steps / dt / 1e6:.2f} M flips/s, "
    f"{num_steps * n / dt / 1e9:.2f} G field updates/s"
)
