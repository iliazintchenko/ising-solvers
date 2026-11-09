import ctypes
import os

import numpy as np

from .utils import validate_hamiltonian, get_energy

# built by running `make` in the repository root
_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "native", "libpfa_native.so")
_lib = ctypes.CDLL(_LIB_PATH)

_f64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
_f32 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
_f64_2d = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
_lib.pfa_native_anneal.argtypes = [
    _f64_2d,  # couplings, read in place
    _f64,  # local fields / 2: (fields + couplings @ spins) / 2 (updated in place)
    _f32,  # spins as +-1 (updated in place)
    _f32,  # noise buffer, the noise vector stored twice
    ctypes.c_int,  # number of spins
    ctypes.c_long,  # number of flips
    ctypes.c_double,  # beta at the start of the ramp
    ctypes.c_double,  # beta_max
    ctypes.c_int,  # index of the first spin to flip
    ctypes.POINTER(ctypes.c_double),  # energy (in: initial, out: final)
    ctypes.POINTER(ctypes.c_double),  # lowest energy (in: initial, out: lowest seen)
    _f32,  # spins of the lowest energy state (out)
    ctypes.c_uint64,  # seed for the per-step noise offsets
    ctypes.c_double,  # tabu penalty lambda, in temperatures
    ctypes.c_int,  # tabu tenure, in flips
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
    tabu_lambda: float = 3.0,
    tabu_tenure: float = 0.25,
) -> np.ndarray:
    """
    Attempts to find the ground state of an Ising problem with PFA, a
    rejection-free annealer (see the README): every step flips one spin,
    chosen with probability proportional to exp(-beta * dE) by Gumbel-max
    sampling, while beta rises linearly to beta_max. For the tabu_tenure * n
    flips after a spin flips, its dE is raised by tabu_lambda temperatures at
    the time of its flip (a tabu). The first repetition starts from random
    spins at beta = 0; repetition k restarts from the best state found so
    far, with beta ramping up from (k - 1) / (num_reps - 1) * beta_max, so
    the restarts cool. Compiled to native SIMD code on one core: each flip
    updates the local fields and selects the next spin in one pass over
    memory. The couplings are read in place, without a copy, when they are a
    C-contiguous float64 array.

    Parameters:
        couplings (np.ndarray): The couplings matrix for the system, such that couplings[i,j] is the coupling between spin_i and spin_j.
        fields (np.ndarray, optional): External field terms. Defaults to None.
        beta_max (float): The maximum inverse temperature. Defaults to 1.0.
        num_flips (int): The number of spin flips to perform per repetition. Defaults to 1000.
        num_reps (int): Number of repetitions: the first from random spins, each further one from the best state so far. Defaults to 1.
        seed (int): Seed for the random number generator
        tabu_lambda (float): Tabu penalty on a just-flipped spin, in temperatures. Defaults to 3.0;
            rarely worth changing.
        tabu_tenure (float): How long a spin stays tabu, as a fraction of n flips; 0 turns the tabu
            off. Defaults to 0.25; rarely worth changing.
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
    if tabu_lambda < 0.0:
        raise ValueError("tabu_lambda must be non-negative")
    if not 0.0 <= tabu_tenure <= 1.0:
        raise ValueError("tabu_tenure must be between 0 and 1")

    couplings = np.ascontiguousarray(couplings, dtype=np.float64)
    n = len(couplings)
    fields = np.zeros(n) if fields is None else np.asarray(fields, dtype=np.float64)
    tenure = int(tabu_tenure * n)

    rng = np.random.default_rng(seed)

    energy_best = np.inf
    spins_best = None

    for rep in range(num_reps):
        if rep == 0:
            spins = 2.0 * rng.integers(0, 2, n).astype(np.float64) - 1.0
            beta_start = 0.0
        else:
            spins = spins_best.astype(np.float64)
            beta_start = beta_max * rep / (num_reps - 1)   # restarts cool: from near 0 to beta_max at the last
        energy = get_energy(spins, couplings, fields)
        noise_vec = -np.log(-np.log(rng.random(n)))
        noise_buffer = np.concatenate((noise_vec, noise_vec)).astype(np.float32)
        offsets_seed = int(rng.integers(1, 2**63))
        half_fields = 0.5 * (fields + couplings @ spins)

        # first pick at beta_start: argmax_j(-beta_start * dE_j + noise_j), with dE_j = -4 s_j phi_j
        first = int(np.argmax(4.0 * beta_start * spins * half_fields + noise_vec))

        spins_f32 = spins.astype(np.float32)
        spins_min = spins_f32.copy()
        energy_c = ctypes.c_double(energy)
        energy_min_c = ctypes.c_double(energy)
        _lib.pfa_native_anneal(
            couplings,
            half_fields,
            spins_f32,
            noise_buffer,
            n,
            num_flips,
            beta_start,
            beta_max,
            first,
            ctypes.byref(energy_c),
            ctypes.byref(energy_min_c),
            spins_min,
            offsets_seed,
            tabu_lambda,
            tenure,
        )

        if energy_min_c.value < energy_best:
            energy_best = energy_min_c.value
            spins_best = spins_min

    return spins_best.astype(np.int8)
