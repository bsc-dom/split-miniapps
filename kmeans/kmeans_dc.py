import time
import numpy as np
import os

from itertools import cycle

from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import COLLECTION_IN

from dataclay import api
from dataclay.contrib.splitting import split

## Not needed while using PyCOMPSs
#api.init()

from model.fragment import Fragment
from model.pointcloud import PointCloud
from model.split import ChunkSplit

from tad4bj.slurm import handler as tadh

#############################################
# Constants / experiment values:
#############################################

POINTS_PER_FRAGMENT = int(os.environ["POINTS_PER_FRAGMENT"])
NUMBER_OF_FRAGMENTS = int(os.environ["NUMBER_OF_FRAGMENTS"])
DIMENSIONS = int(os.getenv("DIMENSIONS", "500"))
NUMBER_OF_CENTERS = int(os.getenv("NUMBER_OF_CENTERS", "20"))
NUMBER_OF_KMEANS_ITERATIONS = int(os.getenv("NUMBER_OF_KMEANS_ITERATIONS", "10"))

USE_SPLIT = bool(int(os.environ["USE_SPLIT"]))
ROUNDROBIN_PERSISTENCE = bool(int(os.environ["ROUNDROBIN_PERSISTENCE"]))

SEED = 42
MODE = 'uniform'

#############################################
#############################################
@task(partials=COLLECTION_IN, returns=object)
def recompute_centers(partials):
    
    aggr = np.sum(partials, axis=0)

    centers = list()
    for sum_ in aggr:
        # centers with no elements are removed
        if sum_[1] != 0:
            centers.append(sum_[0] / sum_[1])
    return np.array(centers)


@task(returns=object)
def compute_partition(partition, centers):
    subresults = list()
    for frag in partition:
        partial = frag.partial_sum(centers)
        subresults.append(partial)

    return np.sum(subresults, axis=0)


def kmeans_alg(pointcloud):
    np.random.seed(SEED + (NUMBER_OF_FRAGMENTS * 2))
    centers = np.asarray(
        [np.random.random(DIMENSIONS) for _ in range(NUMBER_OF_CENTERS)]
    )

    for it in range(NUMBER_OF_KMEANS_ITERATIONS):
        print("Doing k-means iteration #%d/%d" % (it + 1, NUMBER_OF_KMEANS_ITERATIONS))

        partials = list()

        start_t = time.time()

        if USE_SPLIT:
            for partition in split(pointcloud, split_class=ChunkSplit):
                partial = compute_partition(partition, centers)
                partials.append(partial)
        else:
            for fragment in pointcloud.chunks:
                partial = fragment.partial_sum(centers)
                partials.append(partial)

        centers = recompute_centers(partials)

        # Ignoring any convergence criteria --always doing all iterations for timing purposes.
        centers = compss_wait_on(centers)

        elapsed_time = time.time() - start_t
        print(" ... evaluation time: %f" % elapsed_time)
        tadh["iteration_time"].append(elapsed_time)

        # REMOVEME
        # I am debugging
        if it > 3:
            return

    return centers


def main():

    print(f"""Starting experiment with the following:

POINTS_PER_FRAGMENT = {POINTS_PER_FRAGMENT}
NUMBER_OF_FRAGMENTS = {NUMBER_OF_FRAGMENTS}
DIMENSIONS = {DIMENSIONS}
NUMBER_OF_CENTERS = {NUMBER_OF_CENTERS}
NUMBER_OF_KMEANS_ITERATIONS = {NUMBER_OF_KMEANS_ITERATIONS}

USE_SPLIT = {USE_SPLIT}
ROUNDROBIN_PERSISTENCE = {ROUNDROBIN_PERSISTENCE}
""")
    start_time = time.time()

    # Generate the data
    pc = PointCloud()
    pc.make_persistent()

    backends = list(api.get_backends_info().keys())
    print("Using the following dataClay backends:\n%s" % backends)

    for i, backend in zip(range(NUMBER_OF_FRAGMENTS), cycle(backends)):
        print("Generating fragment #%d" % (i + 1))
        
        fragment = Fragment()
        if ROUNDROBIN_PERSISTENCE:
            fragment.make_persistent(backend_id=backend)
        else:
            fragment.make_persistent()
        
        fragment.generate_points(POINTS_PER_FRAGMENT, DIMENSIONS, MODE, SEED + i)

        pc.add_fragment(fragment)

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

    # We are ignoring the resulting centers because we don't really care
    # --we are just timing the execution.
    centers = kmeans_alg(pointcloud=pc)

    return

    end_t = time.time()

    kmeans_time = end_t - start_t
    print("k-means time #1: %f" % kmeans_time)
    tadh.write_all()

    # Run kmeans again
    start_t = time.time()

    # We are ignoring the resulting centers because we don't really care
    # --we are just timing the execution.
    centers = kmeans_alg(pointcloud=pc)

    end_t = time.time()

    kmeans_time = end_t - start_t
    print("k-means time #2: %f" % kmeans_time)
    tadh.write_all()

    print(centers[0])
    print(centers[1])

if __name__ == "__main__":
    main()
    ## Not needed while using PyCOMPSs
    api.finish()
