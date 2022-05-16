import time
import os
from itertools import cycle

from sklearn.metrics import pairwise_distances
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

    ## Not needed while using PyCOMPSs
    #api.init()

    from model.fragment import Fragment
    from model.split import KMeansSplit

#############################################
#############################################

@task(partials=COLLECTION_IN, returns=object)
def _merge(partials):
    return np.sum(partials, axis=0)


@task(returns=object)
def _finish(added_partials):
    centers = list()
    for sum_ in added_partials:
        # centers with no elements are removed
        if sum_[1] != 0:
            centers.append(sum_[0] / sum_[1])
    return np.array(centers)


def recompute_centers(partials, arity=48):
    while len(partials) > 1:
        partials_subset = partials[:arity]
        partials = partials[arity:]
        partials.append(_merge(partials_subset))

    return _finish(partials[0])


@task(returns=object)
def generate_points(num_points, dim, seed, backend=None):
    if USE_DATACLAY:
        fragment = Fragment()
        fragment.make_persistent(backend_id=backend)    
        fragment.generate_points(num_points, dim, seed)
    else:

        np.random.seed(seed)
        mat = np.random.random((num_points, dim))

        # Normalize all points between 0 and 1
        mat -= np.min(mat)
        mx = np.max(mat)
        if mx > 0.0:
            mat /= mx

        fragment = mat

    # Wait for things to settle, ensure distribution is consistent,
    # avoid quick runs to destabilize the de facto good balancing
    time.sleep(5)
    return fragment


@task(returns=object)
def compute_partition(partition, centers):
    subresults = list()
    for frag in partition:
        partial = frag.partial_sum(centers)
        subresults.append(partial)

    return np.sum(subresults, axis=0)


@task(returns=object)
def partial_sum(arr, centers):
    partials = np.zeros((centers.shape[0], 2), dtype=object)
    close_centers = pairwise_distances(arr, centers).argmin(axis=1)
    for center_idx in range(len(centers)):
        indices = np.argwhere(close_centers == center_idx).flatten()
        partials[center_idx][0] = np.sum(arr[indices], axis=0)
        partials[center_idx][1] = indices.shape[0]
    
    return partials


def kmeans_alg(pointcloud):
    np.random.seed(SEED + (NUMBER_OF_FRAGMENTS * 2))
    centers = np.asarray(
        [np.random.random(DIMENSIONS) for _ in range(NUMBER_OF_CENTERS)]
    )

    if USE_DATACLAY and USE_SPLIT:
        partitions = split_1d(pointcloud, split_class=KMeansSplit, multiplicity=24)

    for it in range(NUMBER_OF_KMEANS_ITERATIONS):
        print("Doing k-means iteration #%d/%d" % (it + 1, NUMBER_OF_KMEANS_ITERATIONS))

        partials = list()

        start_t = time.time()

        if USE_DATACLAY and USE_SPLIT:
            for partition in partitions:
                if COMPUTE_IN_SPLIT:
                    compute_partial = partition.compute(centers)
                else:
                    compute_partial = compute_partition(partition, centers)
                partials.append(compute_partial)

            # Trivial (non-arity-enabled flow) reduction because 
            # split guarantees a low number of partials.
            reduction_step = _merge(partials)
            centers = _finish(reduction_step)

        else:
            for fragment in pointcloud:
                if USE_DATACLAY:
                    partial = fragment.partial_sum(centers)
                else:
                    partial = partial_sum(fragment, centers)
                
                partials.append(partial)

            centers = recompute_centers(partials)

        # Ignoring any convergence criteria --always doing all iterations for timing purposes.
        centers = compss_wait_on(centers)

        elapsed_time = time.time() - start_t
        print(" ... evaluation time: %f" % elapsed_time)
        tadh["iteration_time"].append(elapsed_time)
        tadh.write_all()

    # Now that system is not cold, evaluate the split
    if USE_DATACLAY and USE_SPLIT and EVAL_SPLIT_OVERHEAD:
        start_t = time.time()
        split_1d(pointcloud, split_class=KMeansSplit, multiplicity=24)
        tadh["split_time"] = time.time() - start_t

    return centers


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

    # Generate the data
    pc = list()

    if USE_DATACLAY:
        backends = list(api.get_backends_info().keys())
        print("Using the following dataClay backends:\n%s" % backends)
    else:
        backends = [None]

    for i, backend in zip(range(NUMBER_OF_FRAGMENTS), cycle(backends)):
        print("Generating fragment #%d" % (i + 1))
        
        fragment = generate_points(POINTS_PER_FRAGMENT, DIMENSIONS, SEED + i, backend)
        pc.append(fragment)

    if USE_DATACLAY:
        # COMPSs doesn't like to perform a compss_wait_on twice on an object
        # and split needs to compss_wait_on
        pc = compss_wait_on(pc)
    else:
        # We don't want to gather data, we only want to include a synchronization point
        compss_barrier()

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

    # Just for the sake of using the centers and seeing that there is nothing fishy
    print("First and last vectors of centroids:\n%s\n%s" % (centers[0], centers[-1]))


if __name__ == "__main__":
    main()
