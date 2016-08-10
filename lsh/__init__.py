import hasher
import functools
import numpy as np

from pyspark.mllib.linalg import SparseVector


def run(zdata, p, m, n, b, c):
    """
    Starts the main LSH process.

    Parameters
    ----------
    zdata : RDD[Vector, id]
        RDD of data points. Acceptable vector types are numpy.ndarray
        or PySpark SparseVector.
    p : integer, larger than the largest value in data.
    m : integer, number of bins for hashing.
    n : integer, number of rows to split the signatures into.
    b : integer, number of bands.
    c : integer, minimum allowable cluster size.
    """
    seeds = np.vstack([np.random.random_integers(p, size = n), np.random.random_integers(0, p, size = n)]).T
    hashes = [functools.partial(hasher.minhash, a = s[0], b = s[1], p = p, m = m) for s in seeds]

    # Start by generating the signatures for each data point.
    # Output format is:
    # <(vector id, band idx), minhash>
    sigs = zdata.flatMap(lambda x: [[(x[1], i % b), hashes[i](x[0])] for i, h in enumerate(hashes)])

    # Put together the vector minhashes in the same band.
    # Output format is:
    # <(band idx, minhash list), vector idx>
    bands = sigs.combineByKey(lambda vector: frozenset([vector]),
            lambda vectors, vector: vectors.union(frozenset([vector])),
            lambda left, right: left.union(right)) \
        .map(lambda ((v_idx, b_idx), v_set): ((b_idx, hash(v_set)), v_idx)) \
        .combineByKey(lambda v_set: [v_set],
            lambda v_sets, v_set: v_sets + [v_set],
            lambda left, right: left + right)

    if c > 0:
        bands = bands.filter(lambda (bucket, vectors): len(vectors) >= c)

    # Remaps each element to a cluster / bucket index.
    # Output format is:
    # <vector id, bucket idx>
    vector_bucket = bands.map(lambda (bucket, vectors): frozenset(sorted(vectors))) \
        .distinct() \
        .zipWithIndex() \
        .flatMap(lambda (vectors, idx): [(v, idx) for v in vectors])

    return vector_bucket
