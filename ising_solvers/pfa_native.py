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
_i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
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
    ctypes.POINTER(ctypes.c_double),  # lowest energy of the first half of the flips (in: initial)
    _f32,  # spins of that state (in: initial)
    ctypes.POINTER(ctypes.c_double),  # lowest energy of the second half (out)
    _f32,  # spins of that state (out)
    ctypes.c_uint64,  # seed for the per-step noise offsets
    ctypes.c_double,  # tabu penalty lambda, in temperatures
    ctypes.c_int,  # tabu tenure, in flips
]
_lib.pfa_native_anneal.restype = None
_lib.pfa_categorical_anneal.argtypes = [
    _f64_2d,  # couplings between the values other than 0
    _f64,  # fields of those values, plus their couplings to the current values (updated in place)
    ctypes.c_int,  # number of values other than 0
    ctypes.c_int,  # number of variables with more than one value
    _i32,  # variable of each value other than 0, 0 .. V - 1, grouped by variable
    _i32,  # current value of each variable as its row, -1 at value 0 (updated in place)
    _f32,  # noise buffer, the noise vector stored twice
    ctypes.c_long,  # number of moves
    ctypes.c_double,  # beta at the start of the ramp
    ctypes.c_double,  # beta_max
    ctypes.POINTER(ctypes.c_double),  # energy (in: initial, out: final)
    ctypes.POINTER(ctypes.c_double),  # lowest energy of the first half of the moves (in: initial)
    _i32,  # rows of that state (in: initial)
    ctypes.POINTER(ctypes.c_double),  # lowest energy of the second half (out)
    _i32,  # rows of that state (out)
    ctypes.c_uint64,  # seed for the per-step noise offsets
    ctypes.c_double,  # tabu penalty lambda, in temperatures
    ctypes.c_int,  # tabu tenure, in moves
]
_lib.pfa_categorical_anneal.restype = None
# the sparse kernels: the couplings argument is replaced by the stored couplings, and they take no noise buffer
# (and no first spin): the seed is for their draws
_spin_args, _cat_args = _lib.pfa_native_anneal.argtypes, _lib.pfa_categorical_anneal.argtypes
_lib.pfa_sparse_anneal.argtypes = [
    _i64,  # where each row of the couplings starts
    _i32,  # column of each stored coupling
    _f64,  # value of each stored coupling
] + _spin_args[1:3] + _spin_args[4:8] + _spin_args[9:]
_lib.pfa_sparse_anneal.restype = None
_lib.pfa_sparse_categorical_anneal.argtypes = [
    _i32,  # first row of each variable, and the number of rows
    _i32,  # where each variable's neighbours start
    _i32,  # the neighbours: the variables it is coupled to
    _i64,  # where each variable's couplings start
    _f64,  # per variable, a table: one line per value other than 0, its couplings to the neighbours' values
] + _cat_args[1:4] + _cat_args[5:6] + _cat_args[7:]
_lib.pfa_sparse_categorical_anneal.restype = None
_lib.pfa_categorical_eliminate.argtypes = [
    _i32,  # first row of each variable, and the number of rows
    ctypes.c_int,  # number of variables
    _i32,  # where each variable's neighbours start
    _i32,  # the neighbours: the variables it is coupled to
    _i64,  # per neighbour, where its rows start in a line of the variable's couplings
    _i64,  # where each variable's lines start
    _i64,  # the length of a line of each variable
    _f64,  # the lines: the tables of the sparse kernel, or the array of the array kernel
    _f64,  # fields
    np.ctypeslib.ndpointer(dtype=np.int8, ndim=1, flags="C_CONTIGUOUS"),  # per value, value 0 included: allowed (out)
]
_lib.pfa_categorical_eliminate.restype = ctypes.c_long
# the folds: the first state takes from the second every cluster of variables that the second has lower (in place),
# and the energy change is returned
_lib.pfa_spin_fold.argtypes = [_f64_2d, _f64, ctypes.c_int, _f32, _f32]   # couplings, fields, spins, two states
_lib.pfa_spin_fold.restype = ctypes.c_double
_lib.pfa_sparse_spin_fold.argtypes = _lib.pfa_sparse_anneal.argtypes[:3] + [_f64, ctypes.c_int, _f32, _f32]
_lib.pfa_sparse_spin_fold.restype = ctypes.c_double
_lib.pfa_categorical_fold.argtypes = _lib.pfa_categorical_eliminate.argtypes[:9] + [_i32, _i32]   # two states, as rows
_lib.pfa_categorical_fold.restype = ctypes.c_double
_lib.pfa_categorical_prepare.argtypes = [
    _f64_2d,  # one-hot couplings over all slots
    _f64,  # one-hot fields
    ctypes.c_int,  # number of slots
    ctypes.c_int,  # number of variables
    _i32,  # reference slot of each variable
    ctypes.c_int,  # number of other slots
    _i32,  # the other slots, grouped by variable
    _i32,  # variable of each other slot
    _f64_2d,  # couplings between the other slots, relative to the reference slots (out)
    _f64,  # fields of the other slots, relative to the reference slots (out)
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
    sizes: np.ndarray | None = None,
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
    spins at beta = 0; each later one restarts from the best state so far,
    with beta ramping up from u / (num_reps - 1) * beta_max, u being 1 plus
    the number of restarts so far that did not lower the best energy, so
    the restarts cool as they stop paying. After each repetition
    the best state takes from the repetition's lowest states (one per half
    of its flips) every cluster of spins that they have lower: the spins on
    which two states differ fall into clusters not coupled to each other,
    and each cluster is taken from the state where it costs less. Compiled
    to native code on one core.

    The couplings are an array or a scipy.sparse matrix; the format does not
    change the run. run() picks the kernel from the number of nonzero
    couplings. The array kernels update the local fields and select the next
    spin in one SIMD pass over memory, and read a C-contiguous float64 array
    in place. When a step changes few enough local fields, the sparse kernels
    are faster: they store only the nonzero couplings, update the fields of
    the neighbours only, and draw the next spin with the same probability
    from a tree of sums of the weights exp(-beta * dE). A seed gives a
    different run on the sparse kernels than on the array kernels. If the
    array of the array kernels would take more than half of the machine's
    memory, the kernel whose couplings take less memory runs instead.

    With sizes, the problem is categorical: variable r takes one integer value
    x_r in 0 .. sizes[r] - 1. Each value other than 0 has a row and column of
    the couplings and an entry of the fields, variable by variable (variable
    0's values 1 .. sizes[0] - 1 first), and the energy sums the fields of the
    values taken plus half the couplings between every two of them; value 0
    adds nothing. Couplings between two values of the same variable must be 0.
    Every step changes one variable's value, chosen among all variables and
    values by the same rule. The tabu acts per variable, for tabu_tenure times
    the number of variables steps. Before the anneal, values that another
    value of the same variable is at least as good as, whatever the other
    variables do, are removed (dead-end elimination): a state of the lowest
    energy always remains. categorical_from_one_hot() converts the one-hot
    form.

    Parameters:
        couplings (np.ndarray or scipy.sparse matrix): The couplings matrix for the system, such that couplings[i,j] is the coupling between spin_i and spin_j.
        fields (np.ndarray, optional): External field terms. Defaults to None.
        sizes (np.ndarray, optional): For categorical problems, the number of values of each variable. Defaults to None.
        beta_max (float): The maximum inverse temperature. Defaults to 1.0.
        num_flips (int): Steps per repetition, each flipping a spin or changing one variable's value. Defaults to 1000.
        num_reps (int): Number of repetitions: the first from random spins, each further one from the best state so far, which takes the parts each repetition has lower. Defaults to 1.
        seed (int): Seed for the random number generator
        tabu_lambda (float): Tabu penalty on a just-flipped spin, in temperatures. Defaults to 3.0;
            rarely worth changing.
        tabu_tenure (float): How long a spin stays tabu, as a fraction of n flips; 0 turns the tabu
            off. Defaults to 0.25; rarely worth changing.
    Returns:
        np.ndarray: The lowest-energy state found: +-1 spins, or with sizes the value of each variable.
    Raises:
        ValueError: If input parameters are invalid.
    """

    validate_hamiltonian(couplings, fields)
    sparse = hasattr(couplings, "tocsr")
    if sparse:
        couplings = couplings.tocsr().astype(np.float64)
        couplings.sum_duplicates()
        couplings.eliminate_zeros()
    if sizes is not None:
        sizes = np.asarray(sizes, dtype=np.int64)
        if np.any(sizes < 1) or int(np.sum(sizes - 1)) != couplings.shape[0]:
            raise ValueError("sizes must be positive, with one couplings row per value other than 0: sum(sizes - 1)")
        ends = np.cumsum(sizes - 1)
        if sparse:
            var = np.repeat(np.arange(len(sizes)), sizes - 1)   # variable of each row
            row, col = couplings.nonzero()
            within = bool(np.any(var[row] == var[col]))
        else:
            within = any(np.any(couplings[e - q + 1 : e, e - q + 1 : e]) for q, e in zip(sizes, ends) if q > 2)
        if within:
            raise ValueError("Couplings between values of the same variable must be zero")

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
    fields = np.zeros(couplings.shape[0]) if fields is None else np.ascontiguousarray(fields, dtype=np.float64)
    if sizes is not None:
        return _run_categorical(couplings, fields, sizes, beta_max, num_flips, num_reps, seed, tabu_lambda, tabu_tenure)
    form = _spin_form(couplings, len(fields))
    return _run_spins(form, fields, beta_max, num_flips, num_reps, seed, tabu_lambda, tabu_tenure)


def _sparse_is_faster(changed, size, spins):
    """Which kernel takes less time per step. A step of a sparse kernel costs a fixed part plus one exponential per
    weight it changes; a step of an array kernel is one pass over all size candidates. changed: the weights a step
    changes on average, the neighbours of a spin or the values of a variable's neighbours. The constants are
    fitted to timings of both kernels (on an Apple M4)."""
    return changed < (size / 30 - 7 if spins else size / 9 - 20)


def _memory():
    """Bytes of memory of this machine."""
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def _use_sparse(changed, size, spins, stored):
    """Whether the sparse kernel runs the problem: where its step is faster. The array kernel needs the couplings as
    a size x size float64 array; if that takes more than half of the machine's memory, memory decides and not speed:
    the sparse kernel runs where its couplings take less, stored bytes."""
    array = 8 * size * size
    if 2 * array > _memory():
        return stored < array
    return _sparse_is_faster(changed, size, spins)


def _entries(couplings):
    """(row, column, value) of the nonzero couplings, row by row with ascending columns: the same arrays for an
    array and for a scipy.sparse CSR matrix with sorted columns and no stored zeros. Entries are returned as given."""
    if isinstance(couplings, tuple):
        return couplings
    if hasattr(couplings, "tocsr"):
        return np.repeat(np.arange(couplings.shape[0]), np.diff(couplings.indptr)), couplings.indices, couplings.data
    row, col = np.nonzero(couplings)
    return row, col, np.asarray(couplings[row, col], dtype=np.float64)


def _array(couplings, n):
    """The couplings as an n x n C-contiguous float64 array: an array of that kind is returned as given."""
    if isinstance(couplings, tuple):
        out = np.zeros((n, n))
        out[couplings[0], couplings[1]] = couplings[2]
        return out
    return couplings.toarray() if hasattr(couplings, "tocsr") else np.ascontiguousarray(couplings, dtype=np.float64)


def _count(couplings):
    """The number of nonzero couplings."""
    if isinstance(couplings, tuple):
        return len(couplings[2])
    return couplings.nnz if hasattr(couplings, "tocsr") else np.count_nonzero(couplings)


def _spin_form(couplings, n):
    """The couplings of a problem of n spins as the kernel chosen for them takes them, from an array, a scipy.sparse
    CSR matrix or entries: a float64 array for the array kernel, or _entries() for the sparse kernel. Whatever is
    computed from either form does not depend on the input format."""
    count = _count(couplings)
    if _use_sparse(count / n, n, True, 24 * count):   # the entries: 24 bytes per nonzero coupling
        return _entries(couplings)
    return _array(couplings, n)


def _folded(fold, states):
    """The lowest of the states, as (energy, state) pairs, after it has taken from the others every cluster of
    variables that one of them has lower (fold(a, b) does that for a from b and returns the energy change), going
    round until its energy no longer falls: its energy and the state."""
    energy, state = min(states, key=lambda pair: pair[0])
    others = [other for _, other in states if other is not state]
    while True:
        lower = energy + sum(fold(state, other) for other in others)
        if not lower < energy:
            return energy, state
        energy = lower


def _run_spins(couplings, fields, beta_max, num_flips, num_reps, seed, tabu_lambda, tabu_tenure):
    """run() for spins, on validated input, with the couplings as _spin_form() returns them."""
    sparse = isinstance(couplings, tuple)
    n = len(fields) if sparse else couplings.shape[0]
    if sparse:
        row, col, data = couplings
        ptr = np.concatenate(([0], np.cumsum(np.bincount(row, minlength=n))))
        stored = (ptr.astype(np.int64), col.astype(np.int32), data)
    tenure = int(tabu_tenure * n)

    rng = np.random.default_rng(seed)
    if sparse:
        fold = lambda a, b: _lib.pfa_sparse_spin_fold(*stored, fields, n, a, b)
    else:
        fold = lambda a, b: _lib.pfa_spin_fold(couplings, fields, n, a, b)

    spins_best = None
    stale = 0   # 1 after the first repetition, plus the restarts so far that did not lower the best energy

    for rep in range(num_reps):
        if rep == 0:
            spins = 2.0 * rng.integers(0, 2, n).astype(np.float64) - 1.0
            beta_start = 0.0
        else:
            spins = spins_best.astype(np.float64)
            beta_start = beta_max * stale / (num_reps - 1)   # restarts cool: one step per repetition that brought nothing
        if sparse:
            coupled = np.bincount(row, weights=data * spins[col], minlength=n)   # couplings @ spins
            energy = 0.5 * spins @ coupled + fields @ spins
        else:
            coupled = couplings @ spins
            energy = get_energy(spins, couplings, fields)
        half_fields = 0.5 * (fields + coupled)
        spins_f32 = spins.astype(np.float32)
        spins_first, spins_late = spins_f32.copy(), np.empty(n, dtype=np.float32)
        energy_c = ctypes.c_double(energy)
        first_c, late_c = ctypes.c_double(energy), ctypes.c_double()
        lowest = ctypes.byref(first_c), spins_first, ctypes.byref(late_c), spins_late
        if sparse:
            _lib.pfa_sparse_anneal(
                *stored, half_fields, spins_f32, n, num_flips, beta_start, beta_max, ctypes.byref(energy_c),
                *lowest, int(rng.integers(1, 2**63)), tabu_lambda, tenure,
            )
        else:
            noise_vec = -np.log(-np.log(rng.random(n)))
            noise_buffer = np.concatenate((noise_vec, noise_vec)).astype(np.float32)
            offsets_seed = int(rng.integers(1, 2**63))
            # first pick at beta_start: argmax_j(-beta_start * dE_j + noise_j), with dE_j = -4 s_j phi_j
            first = int(np.argmax(4.0 * beta_start * spins * half_fields + noise_vec))
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
                *lowest,
                offsets_seed,
                tabu_lambda,
                tenure,
            )

        states = [(first_c.value, spins_first), (late_c.value, spins_late)]
        if rep:
            states.append((energy, spins_best))
        energy_new, spins_best = _folded(fold, states)
        stale += rep == 0 or not energy_new < energy

    return spins_best.astype(np.int8)


def _run_categorical(couplings, fields, sizes, beta_max, num_flips, num_reps, seed, tabu_lambda, tabu_tenure):
    """run() for categorical problems, on validated input (couplings as an array or a scipy.sparse CSR matrix).
    Variables with one value never move and are left out, and so are the values that elimination removes."""
    multi = sizes > 1
    args = beta_max, num_flips, num_reps, seed, tabu_lambda, tabu_tenure
    x = np.zeros(len(sizes), dtype=np.int64)
    if not multi.any():
        return x
    form = _Form(couplings, sizes[multi])
    alive = np.empty(form.starts[-1] + len(form.starts) - 1, dtype=np.int8)
    if not _lib.pfa_categorical_eliminate(*form.lines, np.ascontiguousarray(fields), alive):
        x[multi] = _anneal_categorical(form, fields, *args)
        return x
    couplings, fields, count, value = _reduced(form, fields, alive.astype(bool))
    at = np.cumsum(count) - count          # where each variable's allowed values start in value
    free = count > 1
    if free.any():
        at[free] += _anneal_categorical(_Form(couplings, count[free]), fields, *args)
    x[multi] = value[at]
    return x


class _Form:
    """The couplings of a categorical problem whose variables all have more than one value, as the kernel chosen
    for them reads them. starts: rows of variable r are starts[r] .. starts[r + 1] - 1. sparse: whether the sparse
    kernel runs it. couplings: _entries() for the sparse kernel, a float64 array for the array kernel. lines: how
    both pfa_categorical_eliminate() and the sparse kernel find the couplings of a row to a neighbour's rows (see
    pfa_native_eliminate.c): first row of each variable, number of variables, where each variable's neighbours
    start, the neighbours, where a neighbour's rows start in a line, where each variable's lines start, the length
    of its lines, and the lines, which are the sparse kernel's tables or the array itself."""

    def __init__(self, couplings, sizes):
        V = len(sizes)
        width = sizes - 1                                                     # rows per variable
        starts = np.concatenate(([0], np.cumsum(width)))
        m = int(starts[-1])
        grp = np.repeat(np.arange(V), width)
        if isinstance(couplings, np.ndarray):   # the variables with any coupling between their rows
            nonzero = couplings != 0
            count = np.count_nonzero(nonzero)
            some = np.logical_or.reduceat(np.logical_or.reduceat(nonzero, starts[:-1], axis=1), starts[:-1], axis=0)
            pairs = np.flatnonzero(some)
            del nonzero, some
        else:
            row, col, data = _entries(couplings)
            count = len(data)
            pairs = np.unique(grp[row].astype(np.int64) * V + grp[col])       # coupled (variable, neighbour), as one number
        left, right = pairs // V, pairs % V
        tables = int(np.sum(width[left] * width[right]))                      # the sparse kernel stores these in full
        self.starts, self.grp = starts, grp.astype(np.int32)
        self.sparse = _use_sparse(tables / m, m, False, 8 * tables + 24 * count)   # the tables and the entries
        vptr = np.searchsorted(left, np.arange(V + 1))
        if self.sparse:
            self.couplings = _entries(couplings)
            seg, line, rows, vals = _tables(*self.couplings, pairs, grp, starts)
        else:
            self.couplings = _array(couplings, m)
            seg, line, rows, vals = starts[right], np.full(V, m), starts[:-1] * m, self.couplings.reshape(-1)
        self.lines = (starts.astype(np.int32), V, vptr.astype(np.int32), right.astype(np.int32), seg.astype(np.int64),
                      rows.astype(np.int64), line.astype(np.int64), vals)


def _tables(row, col, data, pairs, grp, starts):
    """Couplings between rows (row, col, data) -> full tables between coupled variables, as the sparse categorical
    kernel reads them. pairs: the coupled (variable, neighbour) as variable * V + neighbour, ascending. Variable r's
    table has one line per row of r, with the couplings to every row of its first neighbour, then of its second,
    and so on. Returns where each neighbour's rows start in a line, the length of each variable's lines, where each
    variable's table starts, and the tables."""
    V = len(starts) - 1
    width = np.diff(starts)                                            # rows per variable
    left, right = pairs // V, pairs % V
    vptr = np.searchsorted(left, np.arange(V + 1))
    seg = np.cumsum(width[right]) - width[right]                       # where each neighbour's rows start in a line,
    seg -= seg[vptr[left]]                                             # counted from the variable's first neighbour
    line = np.bincount(left, weights=width[right], minlength=V).astype(np.int64)   # length of a line
    rows = np.concatenate(([0], np.cumsum(width * line)[:-1]))
    vals = np.zeros(int(np.sum(width * line)))
    e = np.searchsorted(pairs, grp[row].astype(np.int64) * V + grp[col])
    r = left[e]
    vals[rows[r] + (row - starts[r]) * line[r] + seg[e] + col - starts[right[e]]] = data
    return seg, line, rows, vals


def _reduced(form, fields, alive):
    """The problem of form on the allowed values (alive, per value with value 0 included, variable by variable), with
    the energy changed by a constant. The lowest allowed value of each variable becomes its value 0: with k_s the
    row of that value of s (no row if it is the old value 0, and its terms are 0), the couplings and fields are

        T'(a, c) = T(a, c) - (T(a, k_s) + T(k_r, c)) + T(k_r, k_s)
        f'(a) = G(a) - G(k_r),    G(a) = f(a) + sum over s of T(a, k_s)

    Variables left with one value are left out: their couplings are in the fields. Returns the couplings (entries
    or an array, as in form), the fields, the number of allowed values of every variable of form, and the old value
    of each allowed value, variable by variable."""
    starts = form.starts
    V, m = len(starts) - 1, int(starts[-1])
    width = np.diff(starts)
    slot = starts[:-1] + np.arange(V)                    # where each variable's values start in alive
    count = np.add.reduceat(alive.astype(np.int64), slot)
    on = np.flatnonzero(alive)
    value = on - np.repeat(slot, count)
    k = np.where(value[np.cumsum(count) - count] > 0, on[np.cumsum(count) - count] - np.arange(V) - 1, -1)   # row k_r
    is_k = np.zeros(m, dtype=bool)
    is_k[k[k >= 0]] = True
    keep = np.delete(alive, slot) & ~is_k                # the rows of the reduced problem
    new = np.cumsum(keep) - 1                            # their new numbers
    k_of = np.repeat(k, width)                           # per row: k of its variable
    n = int(keep.sum())
    out_fields = np.empty(n)
    _, _, vptr, adj, _, rows, line, vals = form.lines
    if form.sparse:
        wide = width[adj]
        to_row = np.arange(int(wide.sum())) - np.repeat(np.cumsum(wide) - wide - starts[adj], wide)   # row of each column
        ends = np.concatenate(([0], np.cumsum(line)))
        out = [np.empty(0, dtype=np.int64)], [np.empty(0, dtype=np.int64)], [np.empty(0)]
    else:
        to_row = np.arange(m)
        out = np.zeros((n, n))
    for r in np.flatnonzero(count > 1):
        lo, hi = starts[r], starts[r + 1]
        a = np.flatnonzero(keep[lo:hi])
        if form.sparse:
            table = vals[rows[r] : rows[r] + width[r] * line[r]].reshape(width[r], line[r])
            cols = to_row[ends[r] : ends[r + 1]]
        else:
            table, cols = form.couplings[lo:hi], to_row
        G = fields[lo:hi] + table[:, is_k[cols]].sum(axis=1)
        c = np.flatnonzero(keep[cols])
        h = np.flatnonzero(k_of[cols[c]] >= 0)                       # the columns whose variable has a k_s,
        at_k = c[h] - (cols[c[h]] - k_of[cols[c[h]]])                # and the column of k_s
        T = table[np.ix_(a, c)]
        if k[r] >= 0:
            own = table[k[r] - lo]                                   # T(k_r, .)
            rest = np.flatnonzero(k_of[cols[c]] < 0)
            T[:, h] = T[:, h] - (table[np.ix_(a, at_k)] + own[c[h]]) + own[at_k]
            T[:, rest] -= own[c[rest]]
            out_fields[new[lo + a]] = G[a] - G[k[r] - lo]
        else:
            T[:, h] -= table[np.ix_(a, at_k)]
            out_fields[new[lo + a]] = G[a]
        if form.sparse:
            i, j = np.nonzero(T)
            out[0].append(new[lo + a][i])
            out[1].append(new[cols[c]][j])
            out[2].append(T[i, j])
        else:
            out[new[lo + a[0]] : new[lo + a[-1]] + 1] = T           # c is every kept row
    if form.sparse:
        out = tuple(np.concatenate(x) for x in out)
    return out, out_fields, count, value


def _anneal_categorical(form, fields, beta_max, num_flips, num_reps, seed, tabu_lambda, tabu_tenure):
    """The anneal of a categorical problem in the form of _Form: the value of each variable, 0 for its value 0."""
    starts, grp, sparse, couplings = form.starts, form.grp, form.sparse, form.couplings
    V, m = len(starts) - 1, int(starts[-1])
    sizes = np.diff(starts) + 1

    def values(cur):   # current row of each variable (-1: value 0) -> the value of every variable
        return np.where(cur < 0, 0, cur - starts[:-1] + 1)

    if np.all(sizes == 2):
        # x = (s + 1) / 2: the spin problem with couplings C and fields 2 f + C 1 has 4 times the energy (plus a
        # constant), so it runs at beta / 4
        spin = _spin_form(couplings, m)
        sums = np.bincount(spin[0], weights=spin[2], minlength=m) if isinstance(spin, tuple) else spin.sum(axis=1)
        spins = _run_spins(spin, 2.0 * fields + sums, beta_max / 4.0, num_flips, num_reps, seed, tabu_lambda,
                           tabu_tenure)
        return values(np.where(spins > 0, np.arange(V), -1))

    if sparse:
        row, col, data = couplings
        first, _, vptr, adj, _, rows, _, vals = form.lines
        stored = first, vptr, adj, rows, vals
    rng = np.random.default_rng(seed)
    tenure = int(tabu_tenure * V)
    fields = np.ascontiguousarray(fields, dtype=np.float64)
    fold = lambda a, b: _lib.pfa_categorical_fold(*form.lines, fields, a, b)
    cur_best = None
    stale = 0   # 1 after the first repetition, plus the restarts so far that did not lower the best energy
    for rep in range(num_reps):
        if rep == 0:
            v = (rng.random(V) * sizes).astype(np.int64)                  # a uniformly random value per variable
            cur = np.where(v == 0, -1, starts[:-1] + v - 1).astype(np.int32)
            beta_start = 0.0
        else:
            cur = cur_best.copy()
            beta_start = beta_max * stale / (num_reps - 1)
        on = cur[cur >= 0]
        if sparse:   # from the entries, so that the input format plays no part
            taken = np.zeros(m, dtype=bool)
            taken[on] = True
            sel = taken[row]
            F = fields + np.bincount(col[sel], weights=data[sel], minlength=m)
            energy = fields[on].sum() + 0.5 * data[sel & taken[col]].sum()
        else:
            F = fields + couplings[:, on].sum(axis=1)
            energy = fields[on].sum() + 0.5 * couplings[np.ix_(on, on)].sum()
        cur_first, cur_late = cur.copy(), np.empty(V, dtype=np.int32)
        energy_c = ctypes.c_double(energy)
        first_c, late_c = ctypes.c_double(energy), ctypes.c_double()
        lowest = ctypes.byref(first_c), cur_first, ctypes.byref(late_c), cur_late
        if sparse:
            _lib.pfa_sparse_categorical_anneal(
                *stored, F, m, V, cur, num_flips, beta_start, beta_max, ctypes.byref(energy_c), *lowest,
                int(rng.integers(1, 2**63)), tabu_lambda, tenure,
            )
        else:
            noise_vec = -np.log(-np.log(rng.random(m)))
            noise_buffer = np.concatenate((noise_vec, noise_vec)).astype(np.float32)
            _lib.pfa_categorical_anneal(
                couplings, F, m, V, grp, cur, noise_buffer, num_flips, beta_start, beta_max, ctypes.byref(energy_c),
                *lowest, int(rng.integers(1, 2**63)), tabu_lambda, tenure,
            )
        states = [(first_c.value, cur_first), (late_c.value, cur_late)]
        if rep:
            states.append((energy, cur_best))
        energy_new, cur_best = _folded(fold, states)
        stale += rep == 0 or not energy_new < energy

    return values(cur_best)


def _one_hot_layout(groups):
    """Variable of each slot (labels in sorted order), slots grouped by variable in input order, values per variable,
    and where each variable's slots start in that order."""
    _, gid = np.unique(np.asarray(groups), return_inverse=True)
    order = np.argsort(gid, kind="stable")
    sizes = np.bincount(gid)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    return gid, order, sizes, starts


def categorical_from_one_hot(couplings, fields, groups):
    """
    Converts a categorical problem from one-hot form to the form run() takes with sizes.

    One-hot form: energy 0.5 * x @ couplings @ x + fields @ x over binary x with one 1 per variable, groups[i] the
    variable of slot i; couplings within a variable are ignored. Variables are numbered in sorted order of their
    labels; value 0 of each is its first slot, the others follow in input order.

    Returns:
        (couplings, fields, sizes, offset): the energy in one-hot form is the energy in this form plus offset.
    """
    couplings = np.ascontiguousarray(couplings, dtype=np.float64)
    n = len(couplings)
    fields = np.zeros(n) if fields is None else np.ascontiguousarray(fields, dtype=np.float64)
    gid, order, sizes, starts = _one_hot_layout(groups)
    if len(gid) != n:
        raise ValueError("groups must give one variable per slot")
    is_first = np.zeros(n, dtype=bool)
    is_first[starts] = True
    ref = order[starts].astype(np.int32)
    rest = order[~is_first]
    m = len(rest)
    out_couplings = np.empty((m, m))
    out_fields = np.empty(m)
    offset = _lib.pfa_categorical_prepare(couplings, fields, n, len(sizes), ref, m, rest.astype(np.int32),
                                          gid[rest].astype(np.int32), out_couplings, out_fields)
    return out_couplings, out_fields, sizes, offset


def categorical_to_one_hot(x, groups):
    """The one-hot x (0/1) of a state given as one value per variable, as run() returns it with sizes."""
    _, order, _, starts = _one_hot_layout(groups)
    one_hot = np.zeros(len(order), dtype=np.int8)
    one_hot[order[starts + np.asarray(x)]] = 1
    return one_hot
