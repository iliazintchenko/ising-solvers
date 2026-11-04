import ctypes
import os

import numpy as np

from .utils import validate_hamiltonian, pack_fields, get_energy

# built by running `make` in the repository root
_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "native", "libpfa_native.so")
_lib = ctypes.CDLL(_LIB_PATH)

_f64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
_f64_2d = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
_lib.pfa_native_anneal.argtypes = [
    _f64_2d,  # couplings, pre-multiplied by 4
    _f64,  # spins (updated in place)
    _f64,  # delta energies (updated in place)
    _f64,  # noise buffer, the noise vector stored twice
    ctypes.c_int,  # number of spins
    ctypes.c_long,  # number of flips
    ctypes.c_double,  # beta_max
    ctypes.c_int,  # index of the first spin to flip
    ctypes.POINTER(ctypes.c_double),  # energy (in: initial, out: final)
    ctypes.POINTER(ctypes.c_double),  # lowest energy (in: initial, out: lowest seen)
    _f64,  # spins of the lowest energy state (out)
]
_lib.pfa_native_anneal.restype = None
_lib.pfa_native_simd_path.argtypes = []
_lib.pfa_native_simd_path.restype = ctypes.c_char_p


def simd_path() -> str:
    """Name of the SIMD code path the native library was compiled with."""
    return _lib.pfa_native_simd_path().decode()


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
    algorithm. Same algorithm as pfa_plain, compiled to native SIMD code on a
    single core: each flip updates the delta energies and selects the next spin
    in one pass over memory.

    Repetition 0 draws its random numbers exactly like pfa_plain, so with
    num_reps=1 the result is bit-identical to pfa_plain.run(..., seed=seed).
    Further repetitions continue drawing from the same generator, each starting
    from a different random configuration, and run one after another.

    Parameters:
        couplings (np.ndarray): The couplings matrix for the system, such that couplings[i,j] is the coupling between spin_i and spin_j.
        fields (np.ndarray, optional): External field terms. Defaults to None.
        beta_max (float): The maximum inverse temperature. Defaults to 1.0.
        num_flips (int): The number of spin flips to perform per repetition. Defaults to 1000.
        num_reps (int): Number of repetitions to perform, each starting from a different random configuration. Defaults to 1.
        seed (int): Seed for the random number generator
    Returns:
        np.ndarray: The configuration of spins that minimizes the energy over all repetitions.
    Raises:
        ValueError: If input parameters are invalid.
    """

    validate_hamiltonian(couplings, fields)

    if beta_max <= 0.0:
        raise ValueError("beta_max must be positive")
    if num_flips <= 1:
        raise ValueError("num_flips must be at least 2")
    if num_reps <= 0:
        raise ValueError("num_reps must be positive")

    # fuse fields into the couplings via an additional dummy spin
    couplings = np.ascontiguousarray(pack_fields(couplings, fields), dtype=np.float64)
    couplings_x4 = couplings * 4

    n = len(couplings)

    rng = np.random.default_rng(seed)

    energy_best = np.inf
    spins_best = None

    for _ in range(num_reps):
        # same draws, in the same order, as pfa_plain
        spins = 2.0 * rng.integers(0, 2, n).astype(np.float64) - 1.0
        energy = get_energy(spins, couplings)
        noise_vec = -np.log(-np.log(rng.random(n)))
        noise_buffer = np.concatenate((noise_vec, noise_vec))
        delta_energies = -2 * spins * (couplings @ spins)

        # first pick at beta = 0, computed exactly as pfa_plain does
        first = int(np.argmax(np.multiply(delta_energies, -0.0) + noise_buffer[n:]))

        energy_c = ctypes.c_double(energy)
        energy_min_c = ctypes.c_double(energy)
        spins_min = spins.copy()
        _lib.pfa_native_anneal(
            couplings_x4,
            spins,
            delta_energies,
            noise_buffer,
            n,
            num_flips,
            beta_max,
            first,
            ctypes.byref(energy_c),
            ctypes.byref(energy_min_c),
            spins_min,
        )

        if energy_min_c.value < energy_best:
            energy_best = energy_min_c.value
            spins_best = spins_min

    # if we had any fields, fold the last dummy spin back in
    if fields is not None:
        spins_best = spins_best[-1] * spins_best[:-1]

    return spins_best.astype(np.int8, copy=False)
