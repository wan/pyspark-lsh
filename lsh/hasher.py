import numpy as np

from pyspark.mllib.linalg import SparseVector



def minhash(v, a, b, p, m):
    """
    Determines the type and computes the minhash of the vector.
        1: Multiplies the index by the non-zero seed "a".
        2: Adds the bias "b" (can be 0).
        3: Modulo "p", a number larger than the number of elements.
        4: Modulo "m", the number of bins.

    Parameters
    ----------
    v : object
        Python list, NumPy array, or SparseVector.
    a : integer
        Seed, > 0.
    b : integer
        Seed, >= 0.
    p : integer
        Prime number that is larger than the number of elements.
    m : integer
        Number of bins.

    Returns
    -------
    i : integer
        Integer minhash value that is in [0, bins).
    """

    indices = None
    if isinstance(v, SparseVector):
        indices = v.indices
    elif isinstance(v, (np.ndarray, list)):
        indices = np.arange(len(v), dtype = np.int)
    else:
        raise Exception("Unknown array type '%s'." % type(v))
    # Map the indices to hash values and take the minimum.
    return np.array((((a * indices) + b) % p) % m).min()
