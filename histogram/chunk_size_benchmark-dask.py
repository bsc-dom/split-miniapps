import time
import numpy as np
import os

from pprint import pprint

from dask import delayed
from dask.distributed import wait, Client
import dask.array as da

SWEEP_VALUES = [1, 4, 16, 48]

BLOCK_SIZES = {k: 18432000//k for k in SWEEP_VALUES}

DIMENSIONS = 5
N_BINS = 4  # that is per dimensions
REPETITIONS = 15


def compute_histogram(block_list):
    partials = list()
    for block in block_list:
        partials.append(np.histogramdd(block, N_BINS, [(0, 1)] * DIMENSIONS))
    return np.sum(partials, axis=0)


def gen_block_list(k):
    return [np.random.random((BLOCK_SIZES[k], DIMENSIONS)) for _ in range(k)]


if __name__ == "__main__":
    # dask client
    client = Client(scheduler_file=os.path.expandvars("$HOME/dask-scheduler-$SLURM_JOB_ID.json"))
    print("Dask client: %s" % client)
    print(client)
    si = client.scheduler_info()
    client.retire_workers(list(si["workers"].keys())[1:])

    print("Blocks sizes to use:")
    pprint(BLOCK_SIZES)
    for k in SWEEP_VALUES:
        print("Results (k=#%d)" % k)
        futures = client.map(gen_block_list, [k] * 24, pure=False)
        wait(futures)

        stopwatch = list()
        for _ in range(REPETITIONS):
            start_t = time.time()
            res = client.map(compute_histogram, futures, pure=False)
            wait(res)
            end_t = time.time()
            stopwatch.append(end_t - start_t)
        
        print("Value (client.map): %f" % (sum(stopwatch) / REPETITIONS,))
        print("             * min: %f" % (min(stopwatch),))

        stopwatch = list()
        for _ in range(REPETITIONS):
            start_t = time.time()
            res = client.submit(compute_histogram, futures[0], pure=False)
            wait(res)
            end_t = time.time()
            stopwatch.append(end_t - start_t)
        
        print("Value (client.submit): %f" % (sum(stopwatch) / REPETITIONS,))
        print("                * min: %f" % (min(stopwatch),))
        print("###################################################")
