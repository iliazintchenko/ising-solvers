import numpy as np
from .utils import validate_hamiltonian, pack_fields


def run(
    couplings: np.ndarray,
    fields: np.ndarray | None = None,
    beta_max: float = 1.0,
    num_flips: int = 1000,
    num_reps: int = 1,
    seed: int = 42,
) -> np.ndarray:
    """
    Attempts to find the ground state configuration of an Ising spin glass
    using PFA - a Pretty Fast implementation of the simulated Annealing
    algorithm.

    Parameters:
        couplings (np.ndarray): The couplings matrix for the system, such that couplings[i,j] is the coupling between spin_i and spin_j.
        fields (np.ndarray, optional): External field terms. Defaults to None.
        beta_max (float): The maximum inverse temperature. Defaults to 1.0.
        num_flips (int): The number of spin flips to perform. Defaults to 1000.
        num_reps (int): Number of repetitions to perform, each starting from a different random configuration. Defaults to 1.
        seed (int): Seed for the random number generator
    Returns:
        np.ndarray: The configuration of spins that minimizes the energy.
    Raises:
        ValueError: If input parameters are invalid.
    """

    validate_hamiltonian(couplings, fields)

    if beta_max <= 0.0:
        raise ValueError("beta_max must be positive")
    if num_flips <= 0:
        raise ValueError("num_flips must be positive")
    if num_reps <= 0:
        raise ValueError("num_reps must be positive")

    couplings = pack_fields(couplings, fields)

    n = len(couplings)

    rng = np.random.default_rng(seed)

    spins = 2 * rng.integers(0, 2, (n, num_reps)) - 1

    energy_arr = 0.5 * np.sum(spins * (couplings @ spins), axis=0)

    delta_energies = -2 * spins * (couplings @ spins)

    j = np.argmin(energy_arr)
    energy_min = energy_arr[j]
    spins_min = spins[:, j].copy()

    noise = -np.log(-np.log(rng.random((n, num_reps))))

    couplings *= 4

    all_reps = np.arange(num_reps)

    for beta in np.linspace(0.0, beta_max, num_flips):

        inds = (-beta * delta_energies + noise).argmax(axis=0)

        energy_arr += delta_energies[inds, all_reps]

        delta_energies += spins[inds, all_reps] * couplings[:, inds] * spins
        delta_energies[inds, all_reps] *= -1

        spins[inds, all_reps] *= -1

        noise = np.roll(noise, 1, axis=0)

        j = np.argmin(energy_arr)
        if energy_arr[j] < energy_min - 1e-06:
            energy_min = energy_arr[j]
            spins_min = spins[:, j].copy()

    if fields is not None:
        spins_min = spins_min[-1] * spins_min[:-1]

    return spins_min
