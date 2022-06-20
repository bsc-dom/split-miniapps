from copy import copy, deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import cProfile

from sklearn.neighbors import NearestNeighbors
import numpy as np

# for Cython profiling only:
# import pyximport
# pyximport.install()


MAX_WORKERS = 4
N_BLOCKS = 4
DIMENSIONS =  3
NN_POINTS = 1000000
FIT_POINTS = 1000000

def prepare_fit():
    points = np.random.random((FIT_POINTS, 3))
    nn = NearestNeighbors()
    nn.fit(points)
    return nn


def prepare_blocks():
    return [np.random.random((NN_POINTS, 3)) for _ in range(N_BLOCKS)]


def perform_kneighbors(nn, block, copy_nn=False, profile=False):
    if copy_nn:
        nn_orig = nn

        nn = copy(nn_orig)
        nn._tree = copy(nn_orig._tree)

    if profile:
        prof = cProfile.Profile()
        prof.enable()

    start_time = time.time()
    result = nn.kneighbors(block)
    end_time = time.time()

    if profile:
        prof.disable()
        sufix = "copy" if copy_nn else "regular"
        prof.dump_stats(f"exec-{ sufix }.prof")

    return end_time - start_time, result

if __name__ == "__main__":
    start_time = time.time()
    print("Preparing the NearestNeighbors structure . . . ")
    nn = prepare_fit()
    print("... done! (%fs)" % (time.time() - start_time))

    print()

    start_time = time.time()
    print("Preparing blocks . . .")
    blocks = prepare_blocks()
    print("... done! (%fs)" % (time.time() - start_time))

    print()

    print("Submitting kneighbors call to the executor")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        print("Execution w/o copy:")
        futures = [executor.submit(perform_kneighbors, nn, block) for block in blocks]
        for future in as_completed(futures):
            t, result = future.result()
            print("- An execution has been done in %fs" % t)

        print("Execution w/  copy:")
        futures = [executor.submit(perform_kneighbors, nn, block, copy_nn=True) for block in blocks]
        for future in as_completed(futures):
            t, result = future.result()
            print("- An execution has been done in %fs" % t)
