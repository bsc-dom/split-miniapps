import time
import os
from itertools import cycle

import dislib as ds
from dislib.data.array import Array
from dislib.cluster import KMeans

import numpy as np

from pycompss.api.task import task
from pycompss.api.api import compss_wait_on, compss_barrier
from pycompss.api.parameter import COLLECTION_IN, IN

from tad4bj.slurm import handler as tadh


#############################################
# Constants / experiment values:
#############################################

POINTS_PER_FRAGMENT = int(os.environ["POINTS_PER_FRAGMENT"])
NUMBER_OF_FRAGMENTS = int(os.environ["NUMBER_OF_FRAGMENTS"])
DIMENSIONS = int(os.getenv("DIMENSIONS", "20"))
NUMBER_OF_CENTERS = int(os.getenv("NUMBER_OF_CENTERS", "8"))
NUMBER_OF_KMEANS_ITERATIONS = int(os.getenv("NUMBER_OF_KMEANS_ITERATIONS", "10"))

USE_DATACLAY = bool(int(os.environ["USE_DATACLAY"]))
EVAL_SPLIT_OVERHEAD = bool(int(os.getenv("EVAL_SPLIT_OVERHEAD", 0)))
USE_SPLIT = bool(int(os.environ["USE_SPLIT"]))
COMPUTE_IN_SPLIT = bool(int(os.environ["COMPUTE_IN_SPLIT"]))

SEED = 42

#############################################
#############################################

if USE_DATACLAY:
    from dataclay import api
    from dataclay.contrib.splitting import split_1d

#############################################
#############################################

def main():

    print(f"""Starting experiment with the following:

POINTS_PER_FRAGMENT = {POINTS_PER_FRAGMENT}
NUMBER_OF_FRAGMENTS = {NUMBER_OF_FRAGMENTS}
DIMENSIONS = {DIMENSIONS}
NUMBER_OF_CENTERS = {NUMBER_OF_CENTERS}
NUMBER_OF_KMEANS_ITERATIONS = {NUMBER_OF_KMEANS_ITERATIONS}

USE_DATACLAY = {USE_DATACLAY}
USE_SPLIT = {USE_SPLIT}
COMPUTE_IN_SPLIT = {COMPUTE_IN_SPLIT}
""")
    start_time = time.time()

    rand_state = np.random.RandomState()

    x = ds.random_array((POINTS_PER_FRAGMENT * NUMBER_OF_FRAGMENTS, DIMENSIONS),
                        (POINTS_PER_FRAGMENT, DIMENSIONS), rand_state)

    compss_barrier()

    # The following line solves issues down the road.
    # Should I have to do that?
    # Not really.
    #
    # This "requirement" is related to the undefined behaviour
    # of COMPSs when you run compss_wait_on twice on the same 
    # Future object.
    if USE_DATACLAY:
        x._blocks = compss_wait_on(x._blocks)

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

    k = KMeans()
    k.fit(x)
    compss_barrier()
    end_t = time.time()

    kmeans_time = end_t - start_t
    print("k-means time #1: %f" % kmeans_time)
    tadh.write_all()


if __name__ == "__main__":
    main()
