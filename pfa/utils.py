import numpy as np


def energy(
    state: np.ndarray,
    couplings: np.ndarray,
    fields: np.ndarray | None = None,
    offset: float | None = None,
) -> float:
    """
    The energy of a state, +-1 spins or 0/1 variables:

    energy = 0.5 * state @ (couplings @ state) + fields @ state + offset

    Parameters
    ----------
    state : np.ndarray
        A vector of length n.
    couplings : np.ndarray
        An n x n symmetric matrix with zero diagonal.
    fields : np.ndarray, optional
        A vector of length n. Defaults to zeros.
    offset: float, optional
        A constant. Defaults to 0.

    Returns
    -------
    float
        The energy.
    """

    value = 0.5 * state @ (couplings @ state)

    if fields is not None:
        value += np.dot(fields, state)

    if offset is not None:
        value += offset

    return value


def is_symmetric(matrix: np.ndarray) -> bool:
    """Exact symmetry, comparing the upper strip with the lower strip a block at a time (cache-friendly)."""
    n = len(matrix)
    return all(np.array_equal(matrix[i : i + 256, i:], matrix[i:, i : i + 256].T) for i in range(0, n, 256))


def validate(
    couplings: np.ndarray, fields: np.ndarray | None = None
) -> None:
    """
    Checks the couplings and fields: square, symmetric, zero diagonal, matching lengths.

    Parameters
    ----------
    couplings : np.ndarray
        An n x n matrix.
    fields : np.ndarray, optional
        A vector of length n.

    Raises
    ------
    ValueError
        If any of the checks fail.
    """

    if couplings.shape[0] != couplings.shape[1]:
        raise ValueError("Couplings must be square matrix")
    if fields is not None and len(fields) != couplings.shape[0]:
        raise ValueError("Fields must have the same length as couplings matrix")
    sparse = hasattr(couplings, "tocsr")   # a scipy.sparse matrix
    if not np.allclose(couplings.diagonal() if sparse else np.diag(couplings), 0.0):
        raise ValueError("Couplings diagonal must be zero")
    if (couplings != couplings.T).nnz if sparse else not is_symmetric(couplings):
        raise ValueError("Couplings must be symmetric matrix")


def from_binary(
    couplings: np.ndarray,
    fields: np.ndarray | None = None,
    offset: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Rewrites a problem over 0/1 variables x as one over +-1 spins s = 2 x - 1, with the same energies.

    Parameters
    ----------
    couplings : np.ndarray
        An n x n symmetric matrix with zero diagonal, the couplings between the 0/1 variables.
    fields : np.ndarray, optional
        A vector of length n. Defaults to zeros.
    offset: float, optional
        A constant. Defaults to 0.

    Returns
    -------
    np.ndarray
        The couplings between the spins.
    np.ndarray
        The fields on the spins.
    float
        The constant.
    """

    validate(couplings, fields)

    n = len(couplings)
    couplings = couplings.copy()

    # init fields to zero if not provided
    if fields is None:
        fields = np.zeros(n)
    else:
        fields = fields.copy()

    # shift the offset term
    if offset is None:
        offset = 0.0
    offset += np.sum(fields) / 2 + np.sum(couplings) / 8

    # map the fields
    fields = fields / 2 + np.sum(couplings, axis=1) / 4

    # map the couplings
    couplings /= 4

    return couplings, fields, offset
