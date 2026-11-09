import ctypes
import os

import numpy as np

from .utils import get_energy, is_symmetric, validate_hamiltonian

# built by running `make` in the repository root
_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "native", "libpfa_native.so")
_lib = ctypes.CDLL(_LIB_PATH)

_f64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
_f32 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
_f64_2d = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
_i32 = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
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
_lib.pfa_categorical_anneal.argtypes = [
    _f64_2d,  # pair terms between value slots, relative to each variable's reference value
    _f64,  # fields of the value slots, relative to the reference (updated in place)
    ctypes.c_int,  # number of value slots
    ctypes.c_int,  # number of variables with more than one value
    _i32,  # variable of each slot, 0 .. V - 1, slots grouped by variable
    _i32,  # current slot of each variable, -1 at the reference value (updated in place)
    _f32,  # noise buffer, the noise vector stored twice
    ctypes.c_long,  # number of moves
    ctypes.c_double,  # beta at the start of the ramp
    ctypes.c_double,  # beta_max
    ctypes.POINTER(ctypes.c_double),  # energy (in: initial, out: final)
    ctypes.POINTER(ctypes.c_double),  # lowest energy (in: initial, out: lowest seen)
    _i32,  # slots of the lowest energy state (out)
    ctypes.c_uint64,  # seed for the per-step noise offsets
    ctypes.c_double,  # tabu penalty lambda, in temperatures
    ctypes.c_int,  # tabu tenure, in moves
]
_lib.pfa_categorical_anneal.restype = None
_lib.pfa_categorical_prepare.argtypes = [
    _f64_2d,  # one-hot couplings over all slots
    _f64,  # one-hot fields
    ctypes.c_int,  # number of slots
    ctypes.c_int,  # number of variables
    _i32,  # reference slot of each variable
    ctypes.c_int,  # number of other slots
    _i32,  # the other slots, grouped by variable
    _i32,  # variable of each other slot
    _f64_2d,  # pair terms relative to the reference values (out)
    _f64,  # fields relative to the reference values (out)
]
_lib.pfa_categorical_prepare.restype = ctypes.c_double
_lib.pfa_native_simd_path.argtypes = []
_lib.pfa_native_simd_path.restype = ctypes.c_char_p


def simd_path() -> str:
    """Name of the SIMD code path the native library was compiled with."""
    return _lib.pfa_native_simd_path().decode()


def run(
    couplings: np.ndarray,
    fields: np.ndarray | None = None,
    groups: np.ndarray | None = None,
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

    With groups, the problem is categorical: each variable takes exactly one of
    several values. It is given in one-hot form, energy 0.5 * x @ couplings @ x
    + fields @ x over binary x, with groups[i] the variable that slot i belongs
    to; couplings within a group are ignored. Every step then changes one
    variable's value, chosen among all variables and values by the same rule,
    so no state breaks the one-value-per-variable constraint and no penalty is
    needed. The tabu acts per variable, for tabu_tenure times the number of
    variables steps.

    Parameters:
        couplings (np.ndarray): The couplings matrix for the system, such that couplings[i,j] is the coupling between spin_i and spin_j.
        fields (np.ndarray, optional): External field terms. Defaults to None.
        groups (np.ndarray, optional): For categorical problems, the variable of each one-hot slot. Defaults to None.
        beta_max (float): The maximum inverse temperature. Defaults to 1.0.
        num_flips (int): Steps per repetition, each flipping a spin or changing one variable's value. Defaults to 1000.
        num_reps (int): Number of repetitions: the first from random spins, each further one from the best state so far. Defaults to 1.
        seed (int): Seed for the random number generator
        tabu_lambda (float): Tabu penalty on a just-flipped spin, in temperatures. Defaults to 3.0;
            rarely worth changing.
        tabu_tenure (float): How long a spin stays tabu, as a fraction of n flips; 0 turns the tabu
            off. Defaults to 0.25; rarely worth changing.
    Returns:
        np.ndarray: The lowest-energy state over all repetitions: +-1 spins, or with groups the one-hot x (0/1).
    Raises:
        ValueError: If input parameters are invalid.
    """

    if groups is None:
        validate_hamiltonian(couplings, fields)
    elif couplings.shape[0] != couplings.shape[1] or (fields is not None and len(fields) != len(couplings)):
        raise ValueError("Couplings must be a square matrix and fields of the same length")

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
    if groups is not None:
        return _run_categorical(couplings, fields, groups, beta_max, num_flips, num_reps, seed, tabu_lambda, tabu_tenure)
    return _run_spins(couplings, fields, beta_max, num_flips, num_reps, seed, tabu_lambda, tabu_tenure)


def _run_spins(couplings, fields, beta_max, num_flips, num_reps, seed, tabu_lambda, tabu_tenure):
    """run() for spins, on validated input."""
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


def _run_categorical(couplings, fields, groups, beta_max, num_flips, num_reps, seed, tabu_lambda, tabu_tenure):
    """run() for categorical problems; values are stored relative to each variable's first value."""
    couplings = np.asarray(couplings, dtype=np.float64)
    n = len(couplings)
    fields = np.zeros(n) if fields is None else np.asarray(fields, dtype=np.float64)
    _, gid = np.unique(np.asarray(groups), return_inverse=True)
    if len(gid) != n:
        raise ValueError("groups must give one variable per slot")
    V = int(gid.max()) + 1
    order = np.argsort(gid, kind="stable")                            # slots grouped by variable, in input order
    sizes = np.bincount(gid, minlength=V)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    ref = order[starts]                                               # each variable's reference value: its first slot
    is_ref = np.zeros(n, dtype=bool)
    is_ref[starts] = True
    slots = order[~is_ref]                                            # the other values, grouped by variable
    sgrp = gid[slots].astype(np.int32)
    multi = sizes > 1                                                 # variables that can change value
    msizes = sizes[multi]
    Vm = len(msizes)
    mgrp = np.repeat(np.arange(Vm), msizes - 1).astype(np.int32)
    gs = np.concatenate(([0], np.cumsum(msizes - 1)))
    m = len(slots)

    # pair terms and fields relative to the reference values, one pass over the input in C
    Jt = np.empty((m, m))
    ut = np.empty(m)
    e0 = _lib.pfa_categorical_prepare(np.ascontiguousarray(couplings), np.ascontiguousarray(fields), n, V,
                                      ref.astype(np.int32), m, slots.astype(np.int32), sgrp, Jt, ut)
    if not is_symmetric(Jt):
        raise ValueError("Couplings between different variables must be symmetric")

    def one_hot(cur):
        x = np.zeros(n, dtype=np.int8)
        x[ref] = 1
        x[ref[multi][cur >= 0]] = 0
        x[slots[cur[cur >= 0]]] = 1
        return x

    if np.all(msizes == 2):   # two values per variable: the spin problem, x = (s + 1) / 2 with s = +1 for the other slot
        h = 0.5 * ut + 0.25 * Jt.sum(axis=1)
        Jt *= 0.25
        spins = _run_spins(Jt, h, beta_max, num_flips, num_reps, seed, tabu_lambda, tabu_tenure)
        return one_hot(np.where(spins > 0, np.arange(Vm), -1))

    rng = np.random.default_rng(seed)
    tenure = int(tabu_tenure * Vm)
    energy_best, cur_best = np.inf, None
    for rep in range(num_reps):
        if rep == 0:
            v = (rng.random(Vm) * msizes).astype(np.int64)               # a uniformly random value per variable
            cur = np.where(v == 0, -1, gs[:-1] + v - 1).astype(np.int32)
            beta_start = 0.0
        else:
            cur = cur_best.copy()
            beta_start = beta_max * rep / (num_reps - 1)
        on = cur[cur >= 0]
        F = ut + Jt[:, on].sum(axis=1)
        energy = e0 + ut[on].sum() + 0.5 * Jt[np.ix_(on, on)].sum()
        noise_vec = -np.log(-np.log(rng.random(m)))
        noise_buffer = np.concatenate((noise_vec, noise_vec)).astype(np.float32)
        offsets_seed = int(rng.integers(1, 2**63))
        best_cur = cur.copy()
        energy_c = ctypes.c_double(energy)
        energy_min_c = ctypes.c_double(energy)
        _lib.pfa_categorical_anneal(
            Jt, F, m, Vm, mgrp, cur, noise_buffer, num_flips, beta_start, beta_max,
            ctypes.byref(energy_c), ctypes.byref(energy_min_c), best_cur, offsets_seed, tabu_lambda, tenure,
        )
        if energy_min_c.value < energy_best:
            energy_best, cur_best = energy_min_c.value, best_cur

    return one_hot(cur_best)
