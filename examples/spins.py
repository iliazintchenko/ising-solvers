"""A small problem over 0/1 variables, rewritten in spins and solved."""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pfa

# E(x) = 1/2 x Q x + q x + c over x in {0, 1}^4
Q = np.array(
    [
        [0.0, -1.1, 0.2, 0.15234],
        [-1.1, 0.0, -1.5, 0.8],
        [0.2, -1.5, 0.0, -1.0],
        [0.15234, 0.8, -1.0, 0.0],
    ]
)
q = np.array([0.1, -0.2, 0.3, -0.1])
c = 2.5

# the same energies over spins s = 2 x - 1
J, h, offset = pfa.from_binary(Q, q, c)
x = np.array([0, 0, 0, 1])
np.testing.assert_almost_equal(pfa.energy(2 * x - 1, J, h, offset), pfa.energy(x, Q, q, c))

s = pfa.run(J, h, beta_max=3.0, num_steps=1000, num_reps=10)
print("lowest state:", (s + 1) // 2, "energy:", pfa.energy(s, J, h, offset))
