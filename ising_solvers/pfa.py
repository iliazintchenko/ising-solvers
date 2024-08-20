import numpy as np
from .utils import validate_hamiltonian, pack_fields, get_energy


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

    The cost function being optimized is:

    energy = get_energy(spins, couplings, fields)

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

    # fuse fields into the couplings via an additional dummy spin to make things simpler
    couplings = pack_fields(couplings, fields)

    n = len(couplings)

    rng = np.random.default_rng(seed)

    # start with random spins
    spins = 2 * rng.integers(0, 2, n) - 1

    # energy that we are optimizing
    energy = get_energy(spins, couplings)

    # changes in energy if each respective spin is flipped
    delta_energies = -2 * spins * (couplings @ spins)

    # track the lowest energy state we have achieved
    energy_min = energy
    spins_min = spins.copy()

    # prepare noise vector to use the Gumbel-Max trick for sampling:
    # https://lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/
    noise_vec = -np.log(-np.log(rng.random(n)))

    # rotate noise vector around cyclically to avoid biasing any single spin
    noise_arr = [np.roll(noise_vec, i) for i in range(n)]

    # pre-multiplying couplings by 4 to speed up delta_energies update
    couplings *= 4

    # anneal from beta == 0 to beta = beta_max with num_flips spin flips
    for k, beta in enumerate(np.linspace(0.0, beta_max, num_flips)):

        # sample the spin to flip according to probabilities np.exp(-beta * delta_energies) using our noise vector
        i = (-beta * delta_energies + noise_arr[k % n]).argmax()

        # update total energy
        energy += delta_energies[i]

        # update delta_energies
        if spins[i] == 1:
            delta_energies += couplings[i] * spins
        else:
            delta_energies -= couplings[i] * spins
        delta_energies[i] -= 2 * delta_energies[i]

        # actually flip the spin
        spins[i] *= -1

        # track the lowest energy state
        if energy < energy_min - 1e-06:
            energy_min = energy
            spins_min = spins.copy()

    # if we had any fields, cut away the last dummy spin
    if fields is not None:
        spins_min = spins_min[-1] * spins_min[:-1]

    return spins_min
