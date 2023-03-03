import os
import time
import random

import numpy as np

import dislib as ds
from dislib.data.array import Array
from dislib.neighbors import NearestNeighbors

from sklearn.base import BaseEstimator
from sklearn.utils import validation

from scipy.spatial.transform import Rotation

from pycompss.api.api import compss_delete_object
from pycompss.api.constraint import constraint
from pycompss.api.task import task
from pycompss.api.api import compss_wait_on, compss_barrier
from pycompss.api.parameter import IN, Depth, Type, COLLECTION_IN, INOUT

from tad4bj.slurm import handler as tadh


#############################################
# Constants / experiment values:
#############################################
USE_DATACLAY = bool(int(os.getenv("USE_DATACLAY", "0")))
USE_SPLIT = bool(int(os.getenv("USE_SPLIT", "0")))
COPY_FIT_STRUCT = bool(int(os.getenv("COPY_FIT_STRUCT", "1")))

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


# @task(block=INOUT)
# def rotate_block(block, rotation):
#     block = block @ rotation

@task(q_blocks={Type: COLLECTION_IN, Depth: 2}, returns=1)
def perform_np_block(q_blocks):
    return np.block(q_blocks)


class NearestNeighborsDataClay(BaseEstimator):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
    
    def fit(self, x):
        self._fit_data = list()

        if len(x._blocks[0]) != 1:
            raise ValueError("I only know how to work with dsarray of one column")

        from dislib_model.nn_fit import PersistentFitStructure

        if USE_SPLIT:
            flatten_blocks = [row[0] for row in x._blocks]

            for partition in split_1d(flatten_blocks, split_class=SplitItemIndexCoordinator()):
                print("Creating a PersistentFitStructure on backend %s that encompasses #%d blocks" %
                       (partition.backend, len(partition._chunks)))
                nn = PersistentFitStructure()
                nn.make_persistent(backend_id=partition.backend)
                nn.fit_split(partition)
                self._fit_data.append(nn)
        else:
            offset = 0

            from itertools import cycle
            from dataclay.api import get_backends_info

            for backend, row in zip(cycle(list(get_backends_info().keys())), x._iterator(axis=0)):
                nn = PersistentFitStructure()
                nn.make_persistent(backend_id=backend)
                nn.fit(row._blocks, offset)
                self._fit_data.append(nn)
                # Carry the offset by counting samples
                offset += row.shape[0]
    
    def kneighbors(self, x, n_neighbors=None, return_distance=True):
        validation.check_is_fitted(self, '_fit_data')
        
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        
        distances = []
        indices = []

        for q_row in x._iterator(axis=0):
        #for row_blocks in x._blocks:
            queries = []

            for nn_fit_struct in self._fit_data:
                if COPY_FIT_STRUCT:
                    queries.append(nn_fit_struct.get_kneighbors(q_row._blocks, n_neighbors))
                else:
                    queries.append(nn_fit_struct.get_kneighbors_nocopy(q_row._blocks, n_neighbors))

            # compss_delete_object(q_samples)
            dist, ind = _merge_kqueries(n_neighbors, *queries)
            compss_delete_object(*queries)
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

USE_DATACLAY = {USE_DATACLAY}
USE_SPLIT = {USE_SPLIT}
COPY_FIT_STRUCT = {COPY_FIT_STRUCT}
""")

    start_time = time.time()

    rand_state = np.random.RandomState()
    x = ds.random_array((POINTS_PER_BLOCK * N_BLOCKS_FIT, POINT_DIMENSION),
                        BLOCKSIZE, rand_state)

    xq = ds.random_array((POINTS_PER_BLOCK * N_BLOCKS_NN, POINT_DIMENSION),
                         BLOCKSIZE, rand_state)

    compss_barrier()

    # The following line solves issues down the road.
    # Should I have to do that?
    # Not really.
    #
    # This "requirement" is related to the undefined behaviour
    # of COMPSs when you run compss_wait_on twice on the same 
    # Future object. Related to the NearestNeighbors 
    # implementation with persistent objects.
    if USE_DATACLAY:
        x._blocks = compss_wait_on(x._blocks)
        xq._blocks = compss_wait_on(xq._blocks)

    print("Generation/Load done")
    initialization_time = time.time() - start_time
    print("Starting NearestNeighbors")

    print("-----------------------------------------")
    print("-------------- RESULTS ------------------")
    print("-----------------------------------------")
    print("Initialization time: %f" % initialization_time)

    time.sleep(20)

    # Run fit
    start_t = time.time()

    nn = NearestNeighbors()
    nn.fit(x)

    compss_barrier()

    end_t = time.time()

    fit_time = end_t - start_t
    print("Fit time: %f" % fit_time)

    time.sleep(20)

    tadh["initialization_time"] = initialization_time
    tadh["fit_time"] = fit_time
    tadh["kneighbors_time"] = list()
    tadh["rotation_time"] = list()
    tadh.write_all()

    # Uncomment that if you are only interested in evaluating fit_time
    #return

    if USE_SPLIT:
        from dataclay.api import batch_object_info
        for obj, backend in batch_object_info(nn._fit_data).items():
            print("Object %r:" % obj)
            print(" - Backend: %s" % backend)
            #print(" - #elements: %d" % len(obj._itemindexes))

    for _ in range(NUMBER_OF_STEPS):
        # Run a kneighbors
        start_t = time.time()
        dist, ind = nn.kneighbors(xq)

        compss_barrier()
        end_t = time.time()

        kneighbors_time = end_t - start_t

        print("k-neighbors time: %f" % kneighbors_time)
        tadh["kneighbors_time"].append(kneighbors_time)
        
        # start_t = time.time()
        # r = Rotation.from_euler("XYZ", angles=np.random.randint(0, 360, 3), degrees=True).as_matrix()
        # for row in xq._blocks:
        #     for block in row:
        #         if USE_DATACLAY:
        #             block.rotate_in_place(r)
        #         else:
        #             rotate_block(block, r)

        # compss_barrier()
        # end_t = time.time()

        # rotation_time = end_t - start_t
        # print("rotation time: %f" % rotation_time)
        # tadh["rotation_time"].append(rotation_time)

        # Write for each iteration
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
