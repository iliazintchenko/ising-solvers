import numpy as np


def get_energy(
    spins: np.ndarray,
    couplings: np.ndarray,
    fields: np.ndarray | None = None,
    offset: float | None = None,
) -> float:
    """
    Compute the energy of the system with either binary spins (0, 1) or Ising
    spins (-1, 1).

    energy = 0.5 * spins @ (couplings @ spins) + np.dot(fields, spins) + offset

    Parameters
    ----------
    couplings : np.ndarray
        A square matrix (n x n) representing the interaction strengths between the binary spins.
    fields : np.ndarray, optional
        A vector of length n representing the external fields acting on each binary spin.
        If not provided, defaults to a zero vector.
    offset: float, optional
        Constant energy offset term. If not provided, defaults to 0.

    Returns
    -------
    float
        The energy of the system.
    """

    energy = 0.5 * spins @ (couplings @ spins)

    if fields is not None:
        energy += np.dot(fields, spins)

    if offset is not None:
        energy += offset

    return energy


def is_symmetric(matrix: np.ndarray) -> bool:
    """Exact symmetry, comparing the upper strip with the lower strip a block at a time (cache-friendly)."""
    n = len(matrix)
    return all(np.array_equal(matrix[i : i + 256, i:], matrix[i:, i : i + 256].T) for i in range(0, n, 256))


def validate_hamiltonian(
    couplings: np.ndarray, fields: np.ndarray | None = None
) -> None:
    """
    Validates the couplings matrix and fields vector for consistency.

    Parameters
    ----------
    couplings : np.ndarray
        A square matrix (n x n) representing the interaction strengths between the binary spins.
    fields : np.ndarray, optional
        A vector of length n representing the external fields acting on each binary spin.

    Raises
    ------
    ValueError
        If any of the input checks fail.
    """

    if couplings.shape[0] != couplings.shape[1]:
        raise ValueError("Couplings must be square matrix")
    if fields is not None and len(fields) != len(couplings):
        raise ValueError("Fields must have the same length as couplings matrix")
    if not np.allclose(np.diag(couplings), 0.0):
        raise ValueError("Couplings diagonal must be zero")
    if not is_symmetric(couplings):
        raise ValueError("Couplings must be symmetric matrix")


def binary_to_ising(
    couplings: np.ndarray,
    fields: np.ndarray | None = None,
    offset: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Maps the Hamiltonian of a system with binary spins (0, 1) to an equivalent Hamiltonian
    of a system with Ising spins (-1, +1).

    Parameters
    ----------
    couplings : np.ndarray
        A square matrix (n x n) representing the interaction strengths between the binary spins.
    fields : np.ndarray, optional
        A vector of length n representing the external fields acting on each binary spin.
        If not provided, it defaults to a zero vector.
    offset: float, optional
        Constant energy offset term. If not provided, defaults to 0.

    Returns
    -------
    np.ndarray
        The transformed coupling matrix for the Ising spins.
    np.ndarray
        The transformed field vector for the Ising spins.
    float
        The shifted offset term.
    """

    validate_hamiltonian(couplings, fields)

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
