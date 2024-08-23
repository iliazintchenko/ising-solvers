import numpy as np
import torch
from torch.amp import autocast

from torch.backends import cudnn

cudnn.benchmark = True
cudnn.enabled = True

from .utils import validate_hamiltonian, pack_fields

# @torch.compile
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
    using a GPU-accelerated PFA - a Pretty Fast implementation of the simulated
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
    """

    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available. This implementation requires a GPU.")

    device = "cuda"

    validate_hamiltonian(couplings, fields)

    if beta_max <= 0.0:
        raise ValueError("beta_max must be positive")
    if num_flips <= 0:
        raise ValueError("num_flips must be positive")
    if num_reps <= 0:
        raise ValueError("num_reps must be positive")

    couplings = pack_fields(couplings, fields)

    couplings = torch.tensor(couplings, dtype=torch.float32, device=device)

    n = couplings.shape[0]

    torch.manual_seed(seed)

    spins = (2 * torch.randint(0, 2, (n, num_reps), device=device) - 1).float()

    energy_arr = 0.5 * torch.sum(spins * (couplings @ spins), dim=0)

    delta_energies = -2 * spins * (couplings @ spins)

    j = torch.argmin(energy_arr)
    energy_min = energy_arr[j].clone().detach()
    spins_min = spins[:, j].clone().detach()

    noise = -torch.log(-torch.log(torch.rand((n, num_reps), device=device)))

    couplings.mul_(4)

    all_reps = torch.arange(num_reps, device=device)

    with autocast(device_type=device):

        for beta in torch.linspace(0.0, beta_max, num_flips, device=device):

            inds = (-beta * delta_energies + noise).argmax(dim=0)

            energy_arr += delta_energies[inds, all_reps]

            delta_energies += spins[inds, all_reps] * couplings[:, inds] * spins
            delta_energies[inds, all_reps] *= -1

            spins[inds, all_reps] *= -1

            noise = torch.roll(noise, 1, dims=0)

            j = torch.argmin(energy_arr)
            if energy_arr[j] < energy_min - 1e-06:
                energy_min = energy_arr[j].clone().detach()
                spins_min = spins[:, j].clone().detach()

    spins_min = spins_min.cpu().numpy()

    if fields is not None:
        spins_min = spins_min[-1] * spins_min[:-1]

    return spins_min
