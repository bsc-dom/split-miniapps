import time
import os

from sklearn.metrics import pairwise_distances
import numpy as np

from pycompss.api.task import task
from pycompss.api.api import compss_wait_on, compss_barrier
from pycompss.api.parameter import COLLECTION_IN

from tad4bj.slurm import handler as tadh


#############################################
# Constants / experiment values:
#############################################

POINTS_PER_FRAGMENT = int(os.environ["POINTS_PER_FRAGMENT"])
NUMBER_OF_FRAGMENTS = int(os.environ["NUMBER_OF_FRAGMENTS"])
DIMENSIONS = int(os.getenv("DIMENSIONS", "500"))
NUMBER_OF_CENTERS = int(os.getenv("NUMBER_OF_CENTERS", "20"))
NUMBER_OF_KMEANS_ITERATIONS = int(os.getenv("NUMBER_OF_KMEANS_ITERATIONS", "10"))

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
def generate_points(num_points, dim, mode, seed):
    """
    Generate a random fragment of the specified number of points using the
    specified mode and the specified seed. Note that the generation is
    distributed (the master will never see the actual points).
    :param num_points: Number of points
    :param dim: Number of dimensions
    :param mode: Dataset generation mode
    :param seed: Random seed
    :return: Dataset fragment
    """
    # Random generation distributions
    rand = {
        'normal': lambda k: np.random.normal(0, 1, k),
        'uniform': lambda k: np.random.random(k),
    }
    r = rand[mode]
    np.random.seed(seed)
    mat = np.array(
        [r(dim) for __ in range(num_points)]
    )
    # Normalize all points between 0 and 1
    mat -= np.min(mat)
    mx = np.max(mat)
    if mx > 0.0:
        mat /= mx

    return mat


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

    for it in range(NUMBER_OF_KMEANS_ITERATIONS):
        print("Doing k-means iteration #%d/%d" % (it + 1, NUMBER_OF_KMEANS_ITERATIONS))

        partials = list()

        start_t = time.time()

        for fragment in pointcloud:
            partial = partial_sum(fragment, centers)
            partials.append(partial)

        centers = recompute_centers(partials)

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
""")
    start_time = time.time()

    # Generate the data
    pc = list()

    for i in range(NUMBER_OF_FRAGMENTS):
        print("Generating fragment #%d" % (i + 1))
        
        fragment = generate_points(POINTS_PER_FRAGMENT, DIMENSIONS, MODE, SEED + i)

        pc.append(fragment)

    compss_barrier()

    print("Generation/Load done")
    initialization_time = time.time() - start_time
    print("Starting kmeans")

    print("-----------------------------------------")
    print("-------------- RESULTS ------------------")
    print("-----------------------------------------")
    print("Initialization time: %f" % initialization_time)

    tadh["initialization_time"] = initialization_time
    tadh["iteration_time"] = list()
    tadh.write_all()

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
    

if __name__ == "__main__":
    main()
