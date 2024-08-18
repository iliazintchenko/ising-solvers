import numpy as np


def pack_fields(couplings: np.ndarray, fields: np.ndarray = None) -> np.ndarray:
    """
    Packs transverse fields into an additional dummy spin that is coupled to all other spins.

    Parameters
    ----------
    couplings : np.ndarray
        A square matrix (n x n) representing the interaction strengths between the binary spins.
    fields : np.ndarray, optional
        A vector of length n representing the external fields acting on each binary spin.

    """

    couplings = couplings.copy()

    # if we have fields, pack them into the couplings via an additional dummy spin
    if fields is not None:
        couplings = np.vstack((couplings, fields))
        couplings = np.hstack((couplings, np.append(fields, 0.0).reshape(-1, 1)))

    return couplings


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
        raise ValueError("Couplings matrix must be square")
    if fields is not None and len(fields) != len(couplings):
        raise ValueError("Fields must have the same length as couplings matrix")
    if not np.allclose(np.diag(couplings), 0.0):
        raise ValueError("Couplings diagonal must be zero")
    if not np.array_equal(couplings, couplings.T):
        raise ValueError("Couplings must be symmetric")


def binary_to_ising(
    couplings: np.ndarray, fields: np.ndarray | None = None
) -> (np.ndarray, np.ndarray):
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

    Returns
    -------
    np.ndarray
        The transformed coupling matrix for the Ising spins.
    np.ndarray
        The transformed field vector for the Ising spins.
    """

    validate_hamiltonian(couplings, fields)

    n = len(couplings)
    couplings = couplings.copy()

    # init fields to zero if not provided
    if fields is None:
        fields = np.zeros(n)
    else:
        fields = fields.copy()

    # map the fields
    fields = fields / 2 + np.sum(couplings, axis=1) / 4

    # map the couplings
    couplings /= 4

    return couplings, fields
