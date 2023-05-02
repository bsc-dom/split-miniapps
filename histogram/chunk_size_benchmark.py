import timeit
import numpy as np

from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint

SWEEP_VALUES = [1, 4, 16, 48]

BLOCK_SIZES = {k: 18432000//k for k in SWEEP_VALUES}

DIMENSIONS = 5
N_BINS = 4  # that is per dimensions
TIMEIT_NUMBER = 5


def compute_histogram(block_list):
    partials = list()
    for block in block_list:
        partials.append(np.histogramdd(block, N_BINS, [(0, 1)] * DIMENSIONS))
    return np.sum(partials, axis=0)


def compute_histogram_mth(block_nested_list):
    with ThreadPool(24) as p:
        p.map(compute_histogram, block_nested_list)


def compute_histogram_mth_tpe(block_nested_list):
    with ThreadPoolExecutor(max_workers=24) as executor:
        executor.map(compute_histogram, block_nested_list)


if __name__ == "__main__":
    print("Blocks sizes to use:")
    pprint(BLOCK_SIZES)
    for k in SWEEP_VALUES:
        print("Results (k=#%d)" % k)
        blocks = [
            [np.random.random((BLOCK_SIZES[k], DIMENSIONS)) for _ in range(k)]
            for _ in range(24)
        ]
        res = timeit.repeat("compute_histogram_mth(blocks)",
                            globals=globals(),
                            number=TIMEIT_NUMBER)
        
        print("Value (24 thread pool): %f" % (min(res) / TIMEIT_NUMBER,))

        res = timeit.repeat("compute_histogram_mth_tpe(blocks)",
                            globals=globals(),
                            number=TIMEIT_NUMBER)
        
        print("Value (#24 ThreadPoolExecutor): %f" % (min(res) / TIMEIT_NUMBER,))

        single_block_list = blocks[0]
        res = timeit.repeat("compute_histogram(single_block_list)",
                            globals=globals(),
                            number=TIMEIT_NUMBER)

        print("Value (single thread): %f" % (min(res) / TIMEIT_NUMBER,))

        print("###################################################")
