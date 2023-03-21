from itertools import cycle
import time
import numpy as np
import os

import dask
import dask.array as da
from dask.distributed import wait, Client, get_worker

from tad4bj.slurm import handler as tadh


#############################################
# Constants / experiment values:
#############################################

POINTS_PER_FRAGMENT = int(os.environ["POINTS_PER_FRAGMENT"])
NUMBER_OF_FRAGMENTS = int(os.environ["NUMBER_OF_FRAGMENTS"])
DIMENSIONS = int(os.getenv("DIMENSIONS", "5"))
NUMBER_OF_ITERATIONS = int(os.getenv("NUMBER_OF_ITERATIONS", "10"))

USE_SPLIT = bool(int(os.environ["USE_SPLIT"]))

DASK_RECHUNK = int(os.getenv("DASK_RECHUNK", "0"))

SEED = 420

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


@dask.delayed
def sum_partials(partials):
    return np.sum(partials, axis=0)


def compute_partition(partition, n_bins):
    worker = get_worker()

    subresults = list()
    for frag_id in partition:
        frag = worker.data[frag_id]
        partial, _ = np.histogramdd(frag, n_bins, [(0, 1)] * DIMENSIONS)
        subresults.append(partial)

    return np.sum(subresults, axis=0)


def partial_histogram(fragment, n_bins, n_dimensions):
    values, _ = np.histogramdd(fragment, n_bins, [(0, 1)] * n_dimensions)
    return values


def histogram(client, experiment):
    n_bins = 4  # that is 4 *per dimension*

    partials = list()

    if DASK_RECHUNK > 0:
        experiment = experiment.rechunk((POINTS_PER_FRAGMENT * DASK_RECHUNK, DIMENSIONS))

    if USE_SPLIT:
        partitions = dask_split(client, experiment)

        for w, part in partitions:
            # The function is pure, but we don't want Dask to cache results (because we are evaluating
            # the execution, not the cache). pure=False is a way to do avoid cache artificially
            partial = client.submit(compute_partition, part, n_bins, workers=w, pure=False)
            partials.append(partial)
    else:
        inner_partials = experiment.map_blocks(partial_histogram, n_bins, DIMENSIONS, dtype=float)
        blocks = inner_partials.blocks

        partials = list()
        for i in range(0, blocks.shape[0], 50):
            partials.append(sum_partials([blocks[k] for k in range(i, min(i+50, blocks.shape[0]))]))

    result = sum_partials(partials)
    return result.compute()


def main():
    # dask client
    client = Client(scheduler_file=os.path.expandvars("$HOME/dask-scheduler-$SLURM_JOB_ID.json"))
    print("Dask client: %s" % client)
    print(client)

    print(f"""Starting experiment with the following:

POINTS_PER_FRAGMENT = {POINTS_PER_FRAGMENT}
NUMBER_OF_FRAGMENTS = {NUMBER_OF_FRAGMENTS}
DIMENSIONS = {DIMENSIONS}
NUMBER_OF_ITERATIONS = {NUMBER_OF_ITERATIONS}

USE_SPLIT = {USE_SPLIT}
DASK_RECHUNK = {DASK_RECHUNK}
""")
    start_time = time.time()

    dataset = da.random.random((POINTS_PER_FRAGMENT * NUMBER_OF_FRAGMENTS, DIMENSIONS),
                               chunks=(POINTS_PER_FRAGMENT, DIMENSIONS))

    dataset = dataset.persist()
    wait(dataset)

    print("Generation/Load done")
    initialization_time = time.time() - start_time
    print("Starting histogram")

    print("-----------------------------------------")
    print("-------------- RESULTS ------------------")
    print("-----------------------------------------")
    print("Initialization time: %f" % initialization_time)

    tadh["initialization_time"] = initialization_time
    tadh.write_all()
    tadh["iteration_time"] = list()

    time.sleep(10)

    # Run histogram
    result = None
    for i in range(NUMBER_OF_ITERATIONS):
        start_t = time.time()
        result = histogram(client, dataset)
        end_t = time.time()

        histogram_time = end_t - start_t
        print("Histogram time (#%d/%d): %f" % (i + 1, NUMBER_OF_ITERATIONS, histogram_time))
        tadh["iteration_time"].append(histogram_time)
        tadh.write_all()

    #print(result)


if __name__ == "__main__":
    main()
