import numpy as np
from .utils import validate_hamiltonian


def run(
    couplings: np.ndarray,
    fields: np.ndarray | None = None,
    beta_max: float = 1.0,
    num_flips: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """
    Attempts to find the ground state configuration of an Ising spin glass
    using PFA - a Pretty Fast implementation of the simulated Annealing
    algorithm.

    The cost function being optimized is the following:

    E = 0.5 * config @ (couplings @ config) + np.dot(fields, config)

    Parameters:
        couplings (np.ndarray): The couplings matrix for the system, such that couplings[i,j] is the coupling between spin_i and spin_j.
        fields (np.ndarray, optional): External field terms. Defaults to None.
        beta_max (float): The maximum inverse temperature. Defaults to 1.0.
        num_flips (int): The number of spin flips to perform. Defaults to 1000.
        seed (int): Seed for the random number generator

    Returns:
        np.ndarray: The configuration of spins that minimizes the energy.
    """

    validate_hamiltonian(couplings, fields)

    if beta_max <= 0.0:
        raise ValueError("beta_max must be positive")
    if num_flips <= 0:
        raise ValueError("num_flips must be positive")

    n = len(couplings)
    couplings = couplings.copy()

    # if we have fields, pack them into the couplings via an additional dummy spin
    if fields is not None:
        couplings = np.vstack((couplings, fields))
        couplings = np.hstack((couplings, np.append(fields, 0.0).reshape(-1, 1)))
        n += 1

    rng = np.random.default_rng(seed)

    # start with random config
    config = 2 * rng.integers(0, 2, n) - 1

    # energy that we are optimizing
    E = 0.5 * config @ (couplings @ config)

    # changes in energy if each respective spin is flipped
    dEs = -2 * config * (couplings @ config)

    # track the lowest energy state we have achieved
    Emin = E
    config_min = config.copy()

    # prepare noise vector to use the Gumbel-Max trick for sampling:
    # https://lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/
    vec = -np.log(-np.log(rng.random(n)))

    # rotate our noise around cyclically to avoid biasing any single spin
    noise_arr = [np.roll(vec, i) for i in range(n)]

    # anneal from beta == 0 to beta = beta_max with num_flips spin flips
    for k, beta in enumerate(np.linspace(0.0, beta_max, num_flips)):

        # sample the spin to flip according to P[i] = exp(-beta*dE[i]) using our noise vector
        i = (-beta * dEs + noise_arr[k % n]).argmax()

        # update total energy
        E += dEs[i]

        # update dEs
        delta_dEs = (4 * config[i]) * couplings[i] * config
        delta_dEs[i] = -2 * dEs[i]
        dEs += delta_dEs

        # actually flip the spin
        config[i] *= -1

        # track the lowest energy state
        if E < Emin - 1e-06:
            Emin = E
            config_min = config.copy()

    # if we had any fields cut away the last dummy spin
    if fields is not None:
        config_min = config_min[-1] * config_min[:-1]

    return config_min
