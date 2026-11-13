"""Graph colouring as a categorical problem: each vertex takes one of k colours, and every edge whose ends share a
colour costs 1. The energy is the number of such edges, 0 for a proper colouring."""
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pfa

rng = np.random.default_rng(0)
V, k, degree = 500, 4, 7   # vertices, colours, mean degree
edges = rng.integers(0, V, (V * degree // 2, 2))
edges = np.unique(np.sort(edges[edges[:, 0] != edges[:, 1]], axis=1), axis=0)

# Value 0 of every variable has no field and no couplings, so the k - 1 other colours have a row each: colour a > 0 of
# vertex u is row (k - 1) * u + a - 1. Two ends with the same colour a > 0 cost 1 through the coupling between their
# rows; both at colour 0 cost 1 as well, which the fields and a constant supply: with x_u = 1 when u is not at
# colour 0, [both at 0] = 1 - x_u - x_v + x_u x_v, and x_u x_v = 1 iff exactly one row of u and one row of v are
# taken: a coupling of 1 between every row of u and every row of v.
u, v = edges.T
rows_u = (k - 1) * u[:, None] + np.arange(k - 1)
rows_v = (k - 1) * v[:, None] + np.arange(k - 1)
same = sp.coo_matrix((np.ones(edges.shape[0] * (k - 1)), (rows_u.ravel(), rows_v.ravel())), shape=((k - 1) * V,) * 2)
every = sp.coo_matrix(
    (np.ones(edges.shape[0] * (k - 1) ** 2), (np.repeat(rows_u, k - 1, axis=1).ravel(), np.tile(rows_v, k - 1).ravel())),
    shape=((k - 1) * V,) * 2,
)
couplings = (same + every).tocsr()
couplings = couplings + couplings.T
fields = -np.bincount(np.concatenate((rows_u.ravel(), rows_v.ravel())), minlength=(k - 1) * V).astype(float)
constant = edges.shape[0]

colours = pfa.run(couplings, fields, sizes=np.full(V, k), beta_max=4.0, num_steps=50 * V, num_reps=10)
clashes = int(np.sum(colours[u] == colours[v]))
taken = np.zeros((k - 1) * V)
taken[(k - 1) * np.flatnonzero(colours) + colours[colours > 0] - 1] = 1.0
assert pfa.energy(taken, couplings, fields, constant) == clashes
print(f"{V} vertices, {edges.shape[0]} edges, {k} colours: {clashes} edges with both ends the same colour")
