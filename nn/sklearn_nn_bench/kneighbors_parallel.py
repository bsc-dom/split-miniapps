from copy import copy, deepcopy
from concurrent.futures import ThreadPoolExecutor, wait
import time
import cProfile

from sklearn.neighbors import NearestNeighbors
import numpy as np

# for Cython profiling only:
# import pyximport
# pyximport.install()


N_BLOCKS = 24
DIMENSIONS =  3
NN_POINTS = 200000
FIT_POINTS = 500000

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

    print("**************************************************************")
    print("Sequential run as baseline")
    print("**************************************************************")
    print("Execution w/o copy:")
    start_time = time.time()
    for block in blocks:
        perform_kneighbors(nn, block)
    print("- Total time: %f" % (time.time() - start_time))

    print("Execution w/  copy:")
    start_time = time.time()
    for block in blocks:
        perform_kneighbors(nn, block, copy_nn=True)
    print("- Total time: %f" % (time.time() - start_time))

    for parallelism in [1, 2, 4, 8, 12, 24]:
        print("**************************************************************")
        print("Submitting kneighbors call to the executor with parallelism equal to %d" % parallelism)
        print("**************************************************************")

        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            print("Execution w/o copy:")
            start_time = time.time()
            futures = [executor.submit(perform_kneighbors, nn, block) for block in blocks]
            wait(futures)
            print("- Total time: %f" % (time.time() - start_time))

            print("Execution w/  copy:")
            start_time = time.time()
            futures = [executor.submit(perform_kneighbors, nn, block, copy_nn=True) for block in blocks]
            wait(futures)
            print("- Total time: %f" % (time.time() - start_time))
