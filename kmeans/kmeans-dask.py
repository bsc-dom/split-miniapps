from itertools import cycle
import time
import os

import dask
import dask.array as da
from dask.distributed import wait, Client, get_worker
import numpy as np
from sklearn.metrics import pairwise_distances

from tad4bj.slurm import handler as tadh


#############################################
# Constants / experiment values:
#############################################

POINTS_PER_FRAGMENT = int(os.environ["POINTS_PER_FRAGMENT"])
NUMBER_OF_FRAGMENTS = int(os.environ["NUMBER_OF_FRAGMENTS"])
DIMENSIONS = int(os.getenv("DIMENSIONS", "20"))
NUMBER_OF_CENTERS = int(os.getenv("NUMBER_OF_CENTERS", "8"))
NUMBER_OF_KMEANS_ITERATIONS = int(os.getenv("NUMBER_OF_KMEANS_ITERATIONS", "10"))

EVAL_SPLIT_OVERHEAD = bool(int(os.getenv("EVAL_SPLIT_OVERHEAD", 0)))
USE_SPLIT = bool(int(os.environ["USE_SPLIT"]))

DASK_RECHUNK = int(os.getenv("DASK_RECHUNK", "0"))

SEED = 42

PROFILE_SPLIT = bool(int(os.getenv("PROFILE_SPLIT", "0")))

#############################################
#############################################


class SplitPartitionHelper:
    def __init__(self, multi):
        self.all_cycles = dict()
        self.multi = multi
        self.worker_indexed = list()

    def __getitem__(self, key):
        try:
            return self.all_cycles[key]
        except KeyError:
            new_lists = [list() for _ in range(self.multi)]
            self.all_cycles[key] = cycle(new_lists)
            self.worker_indexed.extend((key, l) for l in new_lists)
            return self.all_cycles[key]

    def get_partitions(self):
        return self.worker_indexed


def dask_split(client, array, multiplicity=24):
    sph = SplitPartitionHelper(multiplicity)
    for f_str, v in client.who_has(array).items():
        next(sph[v[0]]).append(f_str)
    return sph.get_partitions()


def _partial_sum(block, centers):
    partials = np.zeros((centers.shape[0], 2), dtype=object)
    close_centers = pairwise_distances(block, centers).argmin(axis=1)

    for center_idx, _ in enumerate(centers):
        indices = np.argwhere(close_centers == center_idx).flatten()
        partials[center_idx][0] = np.sum(block[indices], axis=0)
        partials[center_idx][1] = indices.shape[0]

    return partials


def _direct_merge(blocks):
    accum = blocks[0].copy()

    for d in blocks[1:]:
        accum += d

    return accum

_merge = dask.delayed(_direct_merge)


def partition_partial_sum(partition, centers):
    worker = get_worker()
    inner_partials = [_partial_sum(worker.data[block], centers) for block in partition]

    return _direct_merge(inner_partials)


@dask.delayed
def _new_centers(premerge):
    merge_result = _direct_merge(premerge)
    centers = np.zeros((NUMBER_OF_CENTERS, DIMENSIONS))
    for idx, sum_ in enumerate(merge_result):
        centers[idx] = sum_[0] / sum_[1]
    return centers

def main():

    # dask client
    client = Client(scheduler_file=os.path.expandvars("$HOME/dask-scheduler-$SLURM_JOB_ID.json"))
    print("Dask client: %s" % client)
    print(client)

    print(f"""Starting experiment with the following:

POINTS_PER_FRAGMENT = {POINTS_PER_FRAGMENT}
NUMBER_OF_FRAGMENTS = {NUMBER_OF_FRAGMENTS}
DIMENSIONS = {DIMENSIONS}
NUMBER_OF_CENTERS = {NUMBER_OF_CENTERS}
NUMBER_OF_KMEANS_ITERATIONS = {NUMBER_OF_KMEANS_ITERATIONS}

USE_SPLIT = {USE_SPLIT}
DASK_RECHUNK = {DASK_RECHUNK}
""")
    start_time = time.time()

    x = da.random.random((POINTS_PER_FRAGMENT * NUMBER_OF_FRAGMENTS, DIMENSIONS),
                         chunks=(POINTS_PER_FRAGMENT, DIMENSIONS))
    x = x.persist()

    wait(x)

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

    if DASK_RECHUNK > 0:
        print("Issuing a rechunk operation")
        x = x.rechunk((POINTS_PER_FRAGMENT * DASK_RECHUNK, DIMENSIONS))

    if USE_SPLIT:
        print("Performing the split")
        partitions = dask_split(client, x)

    centers = np.random.random((NUMBER_OF_CENTERS, DIMENSIONS))

    for _ in range(NUMBER_OF_KMEANS_ITERATIONS):
        centers = client.scatter(centers, broadcast=True)
        if USE_SPLIT:
            partial_merge = [client.submit(partition_partial_sum, part, centers, workers=w) for w, part in partitions]
        else:
            partial_sums = x.map_blocks(_partial_sum, centers, dtype=object)
        
            blocks = partial_sums.blocks

            partial_merge = list()
            for i in range(0, blocks.shape[0], 50):
                partial_merge.append(_merge([blocks[k] for k in range(i,min(i+50,blocks.shape[0]))]))
        
        centers = _new_centers(partial_merge).compute()

    end_t = time.time()

    kmeans_time = end_t - start_t
    print("k-means time #1: %f" % kmeans_time)
    tadh["iteration_time"].append(kmeans_time)
    tadh.write_all()


if __name__ == "__main__":
    main()
