import time
import numpy as np
import os

from itertools import cycle

from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
from pycompss.api.reduction import reduction
from pycompss.api.parameter import COLLECTION_IN, IN

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
COMPUTE_IN_SPLIT = bool(int(os.environ["COMPUTE_IN_SPLIT"]))

USE_REDUCTION_DECORATOR = bool(int(os.environ["USE_REDUCTION_DECORATOR"]))
REDUCTION_CHUNK_SIZE = int(os.getenv("REDUCTION_CHUNK_SIZE", "48"))

SEED = 42
MODE = 'uniform'

#############################################
#############################################
@task(partials=COLLECTION_IN, returns=object)
def recompute_centers_sum(partials):
    return np.sum(partials, axis=0)


if USE_REDUCTION_DECORATOR:
    # Non-canonical way of applying a decorator programatically
    recompute_centers_sum = reduction(chunk_size=str(REDUCTION_CHUNK_SIZE))(recompute_centers_sum)
    # dear reader: despite your reluctance, I assure you this is legit.
    # p.s.: unless you are wondering why chunk_size parameter is a str. Beats me. COMPSs' doc says so.


@task(partials=IN, returns=object)
def recompute_centers_for_split(partials):
    return np.sum(partials, axis=0)


@task(returns=object)
def recompute_centers_end(added_partials):
    centers = list()
    for sum_ in added_partials:
        # centers with no elements are removed
        if sum_[1] != 0:
            centers.append(sum_[0] / sum_[1])
    return np.array(centers)


@task(returns=object)
def generate_points(num_points, dim, mode, seed, backend=None):
    fragment = Fragment()
    fragment.make_persistent(backend_id=backend)    
    fragment.generate_points(num_points, dim, mode, seed)
    return fragment


@task(returns=object)
def compute_partition(partition, centers):
    subresults = list()
    for frag in partition:
        partial = frag.partial_sum(centers)
        subresults.append(partial)

    return subresults


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
                if COMPUTE_IN_SPLIT:
                    nested_partial = partition.compute(centers)
                else:
                    nested_partial = compute_partition(partition, centers)
                partials.append(recompute_centers_for_split(nested_partial))

        else:
            for fragment in pointcloud.chunks:
                partial = fragment.partial_sum(centers)
                partials.append(partial)

        reduction_step = recompute_centers_sum(partials)
        centers = recompute_centers_end(reduction_step)

        # Ignoring any convergence criteria --always doing all iterations for timing purposes.
        centers = compss_wait_on(centers)

        elapsed_time = time.time() - start_t
        print(" ... evaluation time: %f" % elapsed_time)
        tadh["iteration_time"].append(elapsed_time)

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

    backends = list(api.get_backends_info().keys())
    print("Using the following dataClay backends:\n%s" % backends)

    fragments = list()
    for i, backend in zip(range(NUMBER_OF_FRAGMENTS), cycle(backends)):
        print("Generating fragment #%d" % (i + 1))
        
        if ROUNDROBIN_PERSISTENCE:
            backend_destination = backend
        else:
            backend_destination = None
        fragment = generate_points(POINTS_PER_FRAGMENT, DIMENSIONS, MODE, SEED + i, backend_destination)
        fragments.append(fragment)

    # Create the pointcloud data structure
    pc = PointCloud()
    # dataClay would have done the implicit compss_wait_on, but I prefer to explicitly wait here
    pc.chunks = compss_wait_on(fragments)
    pc.make_persistent()

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

    print("First and last vectors of centroids:\n%s\n%s" % (centers[0], centers[-1]))


if __name__ == "__main__":
    main()
    ## Not needed while using PyCOMPSs
    api.finish()
