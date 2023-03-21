import os
import time
from copy import copy

import numpy as np

import dask.array as da
from dask.distributed import wait, Client, get_worker

from sklearn.neighbors import NearestNeighbors

from tad4bj.slurm import handler as tadh


#############################################
# Constants / experiment values:
#############################################
USE_SPLIT = bool(int(os.getenv("USE_SPLIT", "0")))
COPY_FIT_STRUCT = bool(int(os.getenv("COPY_FIT_STRUCT", "1")))

POINTS_PER_BLOCK = int(os.environ["POINTS_PER_BLOCK"])
N_BLOCKS_FIT = int(os.environ["N_BLOCKS_FIT"])
N_BLOCKS_NN = int(os.environ["N_BLOCKS_NN"])
NUMBER_OF_STEPS = int(os.environ["NUMBER_OF_STEPS"])
POINT_DIMENSION = 3

BLOCKSIZE = (POINTS_PER_BLOCK, POINT_DIMENSION)
DASK_RECHUNK = int(os.getenv("DASK_RECHUNK", "0"))
if DASK_RECHUNK > 0:
    EFFECTIVE_POINTS_PER_BLOCK = POINTS_PER_BLOCK * DASK_RECHUNK
else:
    EFFECTIVE_POINTS_PER_BLOCK = POINTS_PER_BLOCK

SEED = 42
CHECK_RESULT = False

#############################################
#############################################

class SplitPartitionHelper:
    def __init__(self):
        self.worker_indexed = dict()

    def add(self, key, ind_rang, value):
        if key not in self.worker_indexed:
            self.worker_indexed[key] = (list(), list())
        indices, value_list = self.worker_indexed[key]
        indices.append(ind_rang)
        value_list.append(value)

    def get_partitions(self):
        return self.worker_indexed.items()


def dask_split(client, array):
    sph = SplitPartitionHelper()
    for f_str, v in client.who_has(array).items():
        i = int(f_str.split(',')[1])
        index_ranges = (i * EFFECTIVE_POINTS_PER_BLOCK, (i + 1) * EFFECTIVE_POINTS_PER_BLOCK)
        sph.add(v[0], index_ranges, f_str)
    return sph.get_partitions()


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


def compute_nn(block):
    nn = NearestNeighbors(n_jobs=1)
    nn.fit(block)
    return nn


def compute_nn_from_partition(blocks):
    worker = get_worker()

    np_blocks = [worker.data[block] for block in blocks]
    nn_data = np.vstack(np_blocks)

    nn = NearestNeighbors(n_jobs=1)
    nn.fit(nn_data)
    return nn


def fit(client, x: da.Array):
    if DASK_RECHUNK > 0:
        x = x.rechunk((EFFECTIVE_POINTS_PER_BLOCK, POINT_DIMENSION))

    result = list()
    waitables = list()

    if USE_SPLIT:
        partitions = dask_split(client, x)

        for worker, (ind_ranges, block_ids) in partitions:
            computation = client.submit(compute_nn_from_partition, block_ids, workers=[worker], pure=False)

            waitables.append(computation)
            result.append(
                (worker, computation, ind_ranges)
            )
    else:
        result = list()
        waitables = list()
        for i, block in enumerate(x.blocks):
            computation = compute_nn(block)
            waitables.append(computation)
            result.append(
                (None, computation,
                 [(i * EFFECTIVE_POINTS_PER_BLOCK, (i + 1) * EFFECTIVE_POINTS_PER_BLOCK)]
                )
            )

    wait(waitables)
    return result


def get_kneighbors(nn, block, ind_ranges, copy_fit_struct=False):
    if copy_fit_struct:
        # Prepare a new structure for the tree walk
        # (due to the lack of readonly/concurrent implementation in the KDTree sklearn implementation)
        original = nn
        nn = copy(original)
        nn._tree = copy(original._tree)

    # Note that the merge requires distances, so we ask for them
    dist, ind = nn.kneighbors(X=block, n_neighbors=5)

    global_ind = np.hstack([np.arange(*i_r) for i_r in ind_ranges])
    return dist, global_ind[ind]
    #            ^****** This converts the local indexes to global ones


def process_merge_results(merged_results):
    indices = list()
    dist = list()
    for res in merged_results:
        d, ind = res.result()
        dist.append(d)
        indices.append(ind)
    return np.vstack(dist), np.vstack(indices)


def kneighbors(client, nn_struct, x):
    merged_results = list()

    for block in x.blocks:
        queries = []

        for worker, nn_fit, ind_ranges in nn_struct:
            comp = client.submit(get_kneighbors, nn_fit, block, ind_ranges, COPY_FIT_STRUCT, workers=worker, pure=False)
            queries.append(comp)

        # This is a list of pairs, later process_merge_result waits and reorganizes
        merged_results.append(client.submit(_merge_kqueries, 5, *queries, pure=False))

    return process_merge_results(merged_results)


def main():
    # dask client
    client = Client(scheduler_file=os.path.expandvars("$HOME/dask-scheduler-$SLURM_JOB_ID.json"))
    print("Dask client: %s" % client)
    print(client)

    print(f"""Starting experiment with the following:

BLOCKSIZE = {BLOCKSIZE}
POINTS_PER_BLOCK = {POINTS_PER_BLOCK}
N_BLOCKS_FIT = {N_BLOCKS_FIT}
N_BLOCKS_NN = {N_BLOCKS_NN}
POINT_DIMENSION = {POINT_DIMENSION}
NUMBER_OF_STEPS = {NUMBER_OF_STEPS}

DASK_RECHUNK = {DASK_RECHUNK}
USE_SPLIT = {USE_SPLIT}
COPY_FIT_STRUCT = {COPY_FIT_STRUCT}
""")
    start_time = time.time()

    x = da.random.random((POINTS_PER_BLOCK * N_BLOCKS_FIT, POINT_DIMENSION),
                         chunks=BLOCKSIZE)

    xq = da.random.random((POINTS_PER_BLOCK * N_BLOCKS_NN, POINT_DIMENSION),
                          chunks=BLOCKSIZE)

    x = x.persist()
    xq = xq.persist()
    wait(x)
    wait(xq)

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

    nn = fit(client, x)

    end_t = time.time()

    fit_time = end_t - start_t
    print("Fit time: %f" % fit_time)

    time.sleep(10)

    tadh["initialization_time"] = initialization_time
    tadh["fit_time"] = fit_time
    tadh["kneighbors_time"] = list()
    tadh["rotation_time"] = list()
    tadh.write_all()

    # Uncomment that if you are only interested in evaluating fit_time
    #return

    for _ in range(NUMBER_OF_STEPS):
        # Run a kneighbors
        start_t = time.time()

        dist, ind = kneighbors(client, nn, xq)

        end_t = time.time()

        kneighbors_time = end_t - start_t

        print("k-neighbors time: %f" % kneighbors_time)
        tadh["kneighbors_time"].append(kneighbors_time)
        
        tadh.write_all()

    print()
    print("-----------------------------------------")
    print()

    if CHECK_RESULT:
        from sklearn.neighbors import NearestNeighbors as SKNN
        nn = SKNN()
        nn.fit(x)
        skdist, skind = nn.kneighbors(xq)

        print("Results according to dislib:")
        # Some debugging stuff to check the result
        print(dist)
        print(ind)
        print("-----------------------------------------")
        print("Results according to sklearn:")
        print(skdist)
        print(skind)
        print("-----------------------------------------")


if __name__ == "__main__":
    main()
