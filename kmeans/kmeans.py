import time
import os
from itertools import cycle

import dislib as ds
from dislib.data.array import Array
from dislib.cluster import KMeans

import numpy as np

from pycompss.api.task import task
from pycompss.api.api import compss_wait_on, compss_barrier
from pycompss.api.parameter import COLLECTION_IN, IN

from tad4bj.slurm import handler as tadh


#############################################
# Constants / experiment values:
#############################################

POINTS_PER_FRAGMENT = int(os.environ["POINTS_PER_FRAGMENT"])
NUMBER_OF_FRAGMENTS = int(os.environ["NUMBER_OF_FRAGMENTS"])
DIMENSIONS = int(os.getenv("DIMENSIONS", "20"))
NUMBER_OF_CENTERS = int(os.getenv("NUMBER_OF_CENTERS", "8"))
NUMBER_OF_KMEANS_ITERATIONS = int(os.getenv("NUMBER_OF_KMEANS_ITERATIONS", "10"))

USE_DATACLAY = bool(int(os.environ["USE_DATACLAY"]))
EVAL_SPLIT_OVERHEAD = bool(int(os.getenv("EVAL_SPLIT_OVERHEAD", 0)))
USE_SPLIT = bool(int(os.environ["USE_SPLIT"]))
COMPUTE_IN_SPLIT = bool(int(os.environ["COMPUTE_IN_SPLIT"]))

SEED = 42

#############################################
#############################################

import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.constraint import constraint
from pycompss.api.parameter import COLLECTION_IN, Depth, Type
from pycompss.api.task import task
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances
from sklearn.utils import check_random_state, validation

from dislib.data.array import Array


class KMeansDataClay(BaseEstimator):

    def __init__(self, n_clusters=8, init='random', max_iter=10, tol=1e-4,
                 arity=50, random_state=None, verbose=False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.arity = arity
        self.verbose = verbose
        self.init = init

    def fit(self, x, y=None):
        """ Compute K-means clustering.

        Parameters
        ----------
        x : ds-array
            Samples to cluster.
        y : ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : KMeans
        """
        self.random_state = check_random_state(self.random_state)
        self._init_centers(x.shape[1], x._sparse)

        old_centers = None
        iteration = 0

        if USE_SPLIT:
            flatten_blocks = [row[0] for row in x._blocks]

            from dataclay.contrib.splitting import split_1d
            from dislib_model.split import GenericSplit

            self.split = split_1d(flatten_blocks, split_class=GenericSplit, multiplicity=24)

        while not self._converged(old_centers, iteration):
            old_centers = self.centers.copy()
            partials = []

            if USE_SPLIT:
                for partition in self.split:
                    partials.append(_partial_sum_partition(partition, old_centers))
            else:
                for row in x._blocks:
                    partials.append(_partial_sum_row(row, old_centers))

            self._recompute_centers(partials)
            iteration += 1

        self.n_iter = iteration

        return self

    def _converged(self, old_centers, iteration):
        if old_centers is None:
            return False

        diff = np.sum(paired_distances(self.centers, old_centers))

        if self.verbose:
            print("Iteration %s - Convergence crit. = %s" % (iteration, diff))

        return diff < self.tol ** 2 or iteration >= self.max_iter

    def _recompute_centers(self, partials):
        while len(partials) > 1:
            partials_subset = partials[:self.arity]
            partials = partials[self.arity:]
            partials.append(_merge(*partials_subset))

        partials = compss_wait_on(partials)

        for idx, sum_ in enumerate(partials[0]):
            if sum_[1] != 0:
                self.centers[idx] = sum_[0] / sum_[1]

    def _init_centers(self, n_features, sparse):
        if isinstance(self.init, np.ndarray) \
                or isinstance(self.init, csr_matrix):
            if self.init.shape != (self.n_clusters, n_features):
                raise ValueError("Init array must be of shape (n_clusters, "
                                 "n_features)")
            self.centers = self.init.copy()
        elif self.init == "random":
            shape = (self.n_clusters, n_features)
            self.centers = self.random_state.random_sample(shape)

            if sparse:
                self.centers = csr_matrix(self.centers)
        else:
            raise ValueError("Init must be random, an nd-array, "
                             "or an sp.matrix")


@constraint(computing_units="${ComputingUnits}")
@task(returns=dict)
def _partial_sum_partition(partition, centers):
    intermediate_results = list()
    for block in partition:
        intermediate_results.append(block.partial_sum(centers))

    return _merge(*intermediate_results)


@constraint(computing_units="${ComputingUnits}")
@task(returns=dict)
def _partial_sum_row(row, centers):
    return row[0].partial_sum(centers)


@constraint(computing_units="${ComputingUnits}")
@task(returns=dict)
def _merge(*data):
    accum = data[0].copy()

    for d in data[1:]:
        accum += d

    return accum


if USE_DATACLAY:
    print("Using dataClay for this execution")

    from dataclay.api import init

    init()

    KMeans = KMeansDataClay


def main():

    print(f"""Starting experiment with the following:

POINTS_PER_FRAGMENT = {POINTS_PER_FRAGMENT}
NUMBER_OF_FRAGMENTS = {NUMBER_OF_FRAGMENTS}
DIMENSIONS = {DIMENSIONS}
NUMBER_OF_CENTERS = {NUMBER_OF_CENTERS}
NUMBER_OF_KMEANS_ITERATIONS = {NUMBER_OF_KMEANS_ITERATIONS}

USE_DATACLAY = {USE_DATACLAY}
USE_SPLIT = {USE_SPLIT}
COMPUTE_IN_SPLIT = {COMPUTE_IN_SPLIT}
""")
    start_time = time.time()

    rand_state = np.random.RandomState()

    x = ds.random_array((POINTS_PER_FRAGMENT * NUMBER_OF_FRAGMENTS, DIMENSIONS),
                        (POINTS_PER_FRAGMENT, DIMENSIONS), rand_state)

    compss_barrier()

    # The following line solves issues down the road.
    # Should I have to do that?
    # Not really.
    #
    # This "requirement" is related to the undefined behaviour
    # of COMPSs when you run compss_wait_on twice on the same 
    # Future object.
    if USE_DATACLAY:
        x._blocks = compss_wait_on(x._blocks)

    print("Generation/Load done")
    initialization_time = time.time() - start_time
    print("Starting kmeans")

    print("-----------------------------------------")
    print("-------------- RESULTS ------------------")
    print("-----------------------------------------")
    print("Initialization time: %f" % initialization_time)

    tadh["initialization_time"] = initialization_time
    tadh.write_all()
    tadh["iteration_time"] = list()

    time.sleep(10)

    # Run kmeans
    start_t = time.time()

    k = KMeans()
    k.fit(x)
    compss_barrier()
    end_t = time.time()

    kmeans_time = end_t - start_t
    print("k-means time #1: %f" % kmeans_time)
    tadh["iteration_time"].append(kmeans_time)
    tadh.write_all()


if __name__ == "__main__":
    main()
