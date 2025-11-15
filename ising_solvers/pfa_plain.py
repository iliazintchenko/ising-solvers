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
    algorithm. This version runs with only a single repetition and is meant
    to be used for debugging and algorithm iteration.

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
    Raises:
        ValueError: If input parameters are invalid.
    """

    # make some basic validation checks
    validate_hamiltonian(couplings, fields)

    if beta_max <= 0.0:
        raise ValueError("beta_max must be positive")
    if num_flips <= 1:
        raise ValueError("num_flips must be at least 2")

    # fuse fields into the couplings via an additional dummy spin to make things simpler
    couplings = pack_fields(couplings, fields)
    couplings = np.ascontiguousarray(couplings)

    n = len(couplings)

    rng = np.random.default_rng(seed)

    # start with random spins
    spins = 2.0 * rng.integers(0, 2, n).astype(np.float64) - 1.0

    # energy that we are optimizing
    energy = get_energy(spins, couplings)

    # track the lowest energy state we have achieved
    energy_min = energy
    spins_min = spins.copy()

    # prepare noise vector to use the Gumbel-Max trick for sampling:
    # https://lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/
    noise_vec = -np.log(-np.log(rng.random(n)))
    noise_buffer = np.concatenate((noise_vec, noise_vec))
    noise_shift = 0

    # changes in energies if each respective spin is flipped
    delta_energies = -2 * spins * (couplings @ spins)

    # pre-multiplying couplings by 4 to speed up helper vector update
    couplings *= 4

    # scratch space reused throughout the loop to avoid repeated allocations
    work = np.empty_like(delta_energies)

    beta_step = beta_max / (num_flips - 1)
    beta = 0.0

    #caching ufunc objects to avoid repeated lookups
    multiply = np.multiply
    add = np.add
    argmax = np.argmax

    # anneal 
    for _ in range(num_flips):

        start = n - noise_shift
        noise_view = noise_buffer[start : start + n]

        multiply(delta_energies, -beta, out=work)
        add(work, noise_view, out=work)
        i = int(argmax(work))

        # update total energy
        energy += delta_energies[i]

        # update delta energies using the shared scratch buffer
        multiply(couplings[i], spins, out=work)
        multiply(work, spins[i], out=work)
        add(delta_energies, work, out=delta_energies)
        delta_energies[i] *= -1

        # flip the spin
        spins[i] = -spins[i]

        # track the lowest energy state
        if energy < energy_min - 1e-06:
            energy_min = energy
            np.copyto(spins_min, spins)

        noise_shift += 1
        if noise_shift == n:
            noise_shift = 0

        beta += beta_step

    # if we had any fields, fold the last dummy spin back in
    if fields is not None:
        spins_min = spins_min[-1] * spins_min[:-1]

    spins_min = spins_min.astype(np.int8, copy=False)
    return spins_min
