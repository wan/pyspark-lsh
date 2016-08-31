from collections import defaultdict
import functools

import numpy as np
from pyspark.mllib.linalg import SparseVector

import hasher


def run(zdata,
        vector_size,
        num_bins,
        signature_size,
        num_bands,
        min_cluster_size):
    """
    Starts the main LSH process.

    Parameters
    ----------
    zdata : RDD[Vector, id]
        RDD of data points. Acceptable vector types are numpy.ndarray
        or PySpark SparseVector.
    vector_size : integer, Prime number larger than the largest value in data.
    num_bins : integer, number of bins for hashing.
    signature_size : integer, number of rows to split the signatures into.
    num_bands : integer, number of bands.
    min_cluster_size : integer, minimum allowable cluster size.
    """
    seeds = np.vstack([np.random.random_integers(vector_size, size = signature_size), np.random.random_integers(0, vector_size, size = signature_size)]).T
    hashers = [functools.partial(hasher.minhash, a = seed[0], b = seed[1], p = vector_size, m = num_bins) for seed in seeds]

    def hash_data(data, _id):
        # Hash vector data into bands
        hash_vals = defaultdict(set)
        for band, hash_func in enumerate(hashers):
            hash_vals[band % num_bands].add(hash_func(data))
        return [((band, hash(frozenset(vals))), [_id])
                  for band, vals in hash_vals.iteritems()]

    sigs = zdata.flatMap(lambda (data, _id): hash_data(data, _id))
    bands = sigs.reduceByKey(lambda x,y: x+y)
    if min_cluster_size:
        bands = bands.filter(lambda (bucket, vectors): len(vectors) >= min_cluster_size)

    # Remaps each element to a cluster / bucket index.
    # Output format is:
    # <vector id, bucket idx>
    vector_bucket = bands.map(lambda (bucket, vectors): frozenset(vectors)) \
        .distinct() \
        .zipWithIndex() \
        .flatMap(lambda (vectors, idx): [(v, idx) for v in vectors])

    return vector_bucket
