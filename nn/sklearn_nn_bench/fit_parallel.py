from concurrent.futures import ThreadPoolExecutor
import time

from sklearn.neighbors import NearestNeighbors
import numpy as np


FIT_BLOCKS = 24
FIT_POINTS = 500000


def prepare_fit_points():
    ret = list()

    for _ in range(FIT_BLOCKS):
        points = np.random.random((FIT_POINTS, 3))
        ret.append(points)

    return ret


def perform_fit(points):
    nn = NearestNeighbors()
    nn.fit(points)
    # No need to return, we won't be using the result


def parallel_run(blocks, parallel_threads):
    print("Ready to run %d parallel threads . . . " % parallel_threads)

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
        _ = [executor.submit(perform_fit, block) for block in blocks]
    end_time = time.time()

    print("Time for this run: %f" % (end_time - start_time))


if __name__ == "__main__":
    blocks_for_fit = prepare_fit_points()

    print("Length of the blocks_for_fit structure: %d" % len(blocks_for_fit))

    print("Running sequential baseline")
    start_time = time.time()
    for block in blocks_for_fit:
        perform_fit(block)
    print("Time for this run: %f" % (time.time() - start_time))

    for n_threads in [1, 2, 4, 8, 12, 24]:
        parallel_run(blocks_for_fit, n_threads)
