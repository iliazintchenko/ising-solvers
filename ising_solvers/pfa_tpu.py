import numpy as np
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.core.xla_random as xla_random
from torch_xla.amp import autocast as xla_autocast
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
    using a TPU-accelerated PFA - a Pretty Fast implementation of the simulated
    Annealing algorithm.

    Parameters:
        couplings (np.ndarray): The couplings matrix for the system, such that couplings[i,j] is the coupling between spin_i and spin_j.
        fields (np.ndarray, optional): External field terms. Defaults to None.
        beta_max (float): Maximum inverse temperature. Defaults to 1.0.
        num_flips (int): Number of spin flips to perform. Defaults to 1000.
        num_reps (int): Number of repetitions to perform, each starting from a different random configuration. Defaults to 1.
        seed (int): Seed for the random number generator.
    Returns:
        np.ndarray: The configuration of spins that minimizes the energy.
    Raises:
        RuntimeError: If a TPU is not available.
        ValueError: If input parameters are invalid.
    """
    if not torch_xla.is_available():
        raise RuntimeError("TPU is not available. This implementation requires a TPU.")

    validate_hamiltonian(couplings, fields)
    if beta_max <= 0.0:
        raise ValueError("beta_max must be positive")
    if num_flips <= 0:
        raise ValueError("num_flips must be positive")
    if num_reps <= 0:
        raise ValueError("num_reps must be positive")

    couplings = pack_fields(couplings, fields)
    device = xm.xla_device()

    couplings = xm.send_cpu_data_to_device(couplings, device)
    n = couplings.shape[0]

    xm.set_rng_state(seed)

    spins = (2 * xla_random.randint(xm.get_ordinal(), (n, num_reps), 0, 2) - 1).float()
    energy_arr = 0.5 * xm.sum(spins * (couplings @ spins), dim=0)
    delta_energies = -2 * spins * (couplings @ spins)
    j = xm.argmin(energy_arr)
    energy_min = energy_arr[j].clone().detach()
    spins_min = spins[:, j].clone().detach()
    noise = -xm.log(-xm.log(xla_random.uniform(xm.get_ordinal(), (n, num_reps))))
    couplings.mul_(4)
    all_reps = xm.arange(num_reps, device=device)
    with xla_autocast(), xm.compile({xm.XLA_DOWNCAST_BF16: True}):
        for beta in xm.linspace(0.0, beta_max, num_flips, device=device):
            inds = (-beta * delta_energies + noise).argmax(dim=0)
            energy_arr += delta_energies[inds, all_reps]
            delta_energies += spins[inds, all_reps] * couplings[:, inds] * spins
            delta_energies[inds, all_reps] *= -1
            spins[inds, all_reps] *= -1
            noise = xm.roll(noise, 1, dims=0)
            j = xm.argmin(energy_arr)
            if energy_arr[j] < energy_min - 1e-06:
                energy_min = energy_arr[j].clone().detach()
                spins_min = spins[:, j].clone().detach()
            xm.mark_step()

    spins_min = xm.xla_host_copy(spins_min).numpy()

    if fields is not None:
        spins_min = spins_min[-1] * spins_min[:-1]

    return spins_min
