import os
import time

import numpy as np

import dislib as ds
from dislib.data.array import Array
from dislib.neighbors import NearestNeighbors

from sklearn.base import BaseEstimator
from sklearn.utils import validation

from scipy.spatial.transform import Rotation

from pycompss.api.constraint import constraint
from pycompss.api.task import task
from pycompss.api.api import compss_wait_on, compss_barrier
from pycompss.api.parameter import IN, Depth, Type, COLLECTION_IN, INOUT

from tad4bj.slurm import handler as tadh


#############################################
# Constants / experiment values:
#############################################
USE_DATACLAY = bool(int(os.environ["USE_DATACLAY"]))

POINTS_PER_BLOCK = int(os.environ["POINTS_PER_BLOCK"])
N_BLOCKS_FIT = int(os.environ["N_BLOCKS_FIT"])
N_BLOCKS_NN = int(os.environ["N_BLOCKS_NN"])
NUMBER_OF_STEPS = int(os.environ["NUMBER_OF_STEPS"])
POINT_DIMENSION = 3

BLOCKSIZE = (POINTS_PER_BLOCK, POINT_DIMENSION)

SEED = 42
CHECK_RESULT = False

#############################################
#############################################

# Used in dataClay execution
class SplitItemIndexCoordinator:
    """Split coordinator that performs global item tracking."""
    def __init__(self):
        self.offset = 0
        
    def __call__(self, backend):
        return ItemIndexAwareSplit(backend, self)


@constraint(computing_units="${ComputingUnits}")
@task(returns=2)
def _merge_kqueries(k, *queries):
    # Reorganize and flatten
    dist, ind = zip(*queries)
    aggr_dist = np.hstack(dist)
    aggr_ind = np.hstack(ind)

    # Final indexes of the indexes (sic)
    final_ii = np.argsort(aggr_dist)[:,:k]

    # Final results
    final_dist = np.take_along_axis(aggr_dist, final_ii, 1)
    final_ind = np.take_along_axis(aggr_ind, final_ii, 1)

    return final_dist, final_ind


@task(block=INOUT)
def rotate_block(block, rotation):
    block = block @ rotation


class NearestNeighborsDataClay(BaseEstimator):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
    
    def fit(self, x):
        self._fit_data = list()

        if len(x._blocks[0]) != 1:
            raise ValueError("I only know how to work with dsarray of one column")

        flatten_blocks = [row[0] for row in x._blocks]

        from dislib_model.nn_fit import PersistentFitStructure
        for partition in split_1d(flatten_blocks, split_class=SplitItemIndexCoordinator()):
            nn = PersistentFitStructure()
            nn.make_persistent()
            nn.fit_split(partition)
            self._fit_data.append(nn)
    
    def kneighbors(self, x, n_neighbors=None, return_distance=True):
        validation.check_is_fitted(self, '_fit_data')
        
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        
        distances = []
        indices = []

        for q_row in x._iterator(axis=0):
            queries = []

            for nn_fit_struct in self._fit_data:
                queries.append(nn_fit_struct.get_kneighbors(q_row._blocks, n_neighbors))

            dist, ind = _merge_kqueries(n_neighbors, *queries)
            distances.append([dist])
            indices.append([ind])

        ind_arr = Array(blocks=indices,
                        top_left_shape=(x._top_left_shape[0], n_neighbors),
                        reg_shape=(x._reg_shape[0], n_neighbors),
                        shape=(x.shape[0], n_neighbors), sparse=False)

        if return_distance:
            dst_arr = Array(blocks=distances,
                            top_left_shape=(x._top_left_shape[0], n_neighbors),
                            reg_shape=(x._reg_shape[0], n_neighbors),
                            shape=(x.shape[0], n_neighbors), sparse=False)
            return dst_arr, ind_arr

        return ind_arr


if USE_DATACLAY:
    print("Using dataClay for this execution")

    from dataclay.api import init
    from dataclay.contrib.splitting import split_1d

    init()

    from dislib_model.split import ItemIndexAwareSplit

    NearestNeighbors = NearestNeighborsDataClay


def main():

    print(f"""Starting experiment with the following:

BLOCKSIZE = {BLOCKSIZE}
POINTS_PER_BLOCK = {POINTS_PER_BLOCK}
N_BLOCKS_FIT = {N_BLOCKS_FIT}
N_BLOCKS_NN = {N_BLOCKS_NN}
POINT_DIMENSION = {POINT_DIMENSION}
NUMBER_OF_STEPS = {NUMBER_OF_STEPS}
""")

    start_time = time.time()

    rand_state = np.random.RandomState()
    x = ds.random_array((POINTS_PER_BLOCK * N_BLOCKS_FIT, POINT_DIMENSION),
                        BLOCKSIZE, rand_state)

    xq = ds.random_array((POINTS_PER_BLOCK * N_BLOCKS_NN, POINT_DIMENSION),
                         BLOCKSIZE, rand_state)

    compss_barrier()

    # The following line solved a bug.
    # Does it make sense?
    # Not really.
    #
    # ...
    #
    # My wild guess is related to the fact that this forces a compss_wait_on
    # on the blocks and stores the result, linked to the undefined behaviour
    # of COMPSs when you run compss_wait_on twice on the same Future object.
    if USE_DATACLAY:
        _ = x.collect()
        _ = xq.collect()
    # If it works, I won't break it.
    # Maybe it is no longer necessary.
    # I won't be the one taking risks. Good luck. Have fun.
    #
    # Addendum:
    # At first only x was collected. But after adding the code that contains
    # block.rotate_in_place an error was given because xq had future objects.
    # Thus, xq.collect was also added. What we really want is to have actual
    # objects instead of futures in the x*._blocks structure.

    print("Generation/Load done")
    initialization_time = time.time() - start_time
    print("Starting NearestNeighbors")

    print("-----------------------------------------")
    print("-------------- RESULTS ------------------")
    print("-----------------------------------------")
    print("Initialization time: %f" % initialization_time)

    time.sleep(10)

    # Run fit
    start_t = time.time()

    nn = NearestNeighbors()
    nn.fit(x)

    compss_barrier()

    end_t = time.time()

    fit_time = end_t - start_t
    print("Fit time: %f" % fit_time)

    tadh["initialization_time"] = initialization_time
    tadh["fit_time"] = fit_time
    tadh["kneighbors_time"] = list()
    tadh["rotation_time"] = list()
    tadh.write_all()

    for _ in range(NUMBER_OF_STEPS):
        # Run a kneighbors
        start_t = time.time()
        dist, ind = nn.kneighbors(xq)

        compss_barrier()
        end_t = time.time()

        kneighbors_time = end_t - start_t

        print("k-neighbors time: %f" % kneighbors_time)
        tadh["kneighbors_time"].append(kneighbors_time)

        start_t = time.time()
        r = Rotation.from_euler("XYZ", angles=np.random.randint(0, 360, 3), degrees=True).as_matrix()
        for row in xq._blocks:
            for block in row:
                if USE_DATACLAY:
                    block.rotate_in_place(r)
                else:
                    rotate_block(block, r)

        compss_barrier()
        end_t = time.time()

        rotation_time = end_t - start_t
        print("rotation time: %f" % rotation_time)
        tadh["rotation_time"].append(rotation_time)

        # Write both times for each iteration
        tadh.write_all()

    print()
    print("-----------------------------------------")
    print()

    if CHECK_RESULT:
        from sklearn.neighbors import NearestNeighbors as SKNN
        nn = SKNN()
        nn.fit(x.collect())
        skdist, skind = nn.kneighbors(xq.collect())

        print("Results according to dislib:")
        # Some debugging stuff to check the result
        print(dist.collect())
        print(ind.collect())
        print("-----------------------------------------")
        print("Results according to sklearn:")
        print(skdist)
        print(skind)
        print("-----------------------------------------")


if __name__ == "__main__":
    main()
