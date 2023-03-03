import time
import numpy as np
import os

import dislib as ds

from pycompss.api.task import task
from pycompss.api.api import compss_wait_on, compss_barrier
from pycompss.api.reduction import reduction
from pycompss.api.parameter import COLLECTION_IN, IN

from tad4bj.slurm import handler as tadh


#############################################
# Constants / experiment values:
#############################################

POINTS_PER_FRAGMENT = int(os.environ["POINTS_PER_FRAGMENT"])
NUMBER_OF_FRAGMENTS = int(os.environ["NUMBER_OF_FRAGMENTS"])
DIMENSIONS = int(os.getenv("DIMENSIONS", "5"))
NUMBER_OF_ITERATIONS = int(os.getenv("NUMBER_OF_ITERATIONS", "10"))

USE_DATACLAY = bool(int(os.environ["USE_DATACLAY"]))
USE_SPLIT = bool(int(os.environ["USE_SPLIT"]))
USE_ACTIVE = bool(int(os.environ["USE_ACTIVE"]))

SEED = 420

#############################################
#############################################

# chunk size is referred as "arity" in most dislib applications,
# semantically speaking
@reduction(chunk_size="48")
@task(partials=COLLECTION_IN, returns=object)
def sum_partials(partials):
    return np.sum(partials, axis=0)


@task(returns=object)
def compute_partition(partition, n_bins):
    subresults = list()
    for frag in partition:
        if USE_ACTIVE and USE_DATACLAY:
            partial = frag.partial_histogram(n_bins, DIMENSIONS)
        else:
            partial, _ = np.histogramdd(frag, n_bins, [(0, 1)] * DIMENSIONS)

        subresults.append(partial)

    return np.sum(subresults, axis=0)


@task(returns=object)
def partial_histogram(fragment, n_bins, n_dimensions):
    values, _ = np.histogramdd(fragment, n_bins, [(0, 1)] * n_dimensions)
    return values


def histogram(experiment):
    n_bins = 4  # that is 4 *per dimension*

    partials = list()

    if USE_DATACLAY and USE_SPLIT:
        flatten_blocks = [row[0] for row in experiment._blocks]

        from dataclay.contrib.splitting import split_1d
        from dislib_model.split import GenericSplit

        for partition in split_1d(flatten_blocks, split_class=GenericSplit, multiplicity=24):
            partial = compute_partition(partition, n_bins)
            partials.append(partial)

    else:
        for row in experiment._blocks:
            # Note that, by design, rows have a single block
            fragment = row[0]
            if USE_DATACLAY and USE_ACTIVE:
                partial = fragment.partial_histogram(n_bins, DIMENSIONS)
            else:
                # This works both for dataClay and non-dataClay executions
                partial = partial_histogram(fragment, n_bins, DIMENSIONS)
                
            partials.append(partial)

    result = sum_partials(partials)
    return compss_wait_on(result)


if USE_DATACLAY:
    print("Using dataClay for this execution")

    from dataclay.api import init

    init()


def main():
    print(f"""Starting experiment with the following:

POINTS_PER_FRAGMENT = {POINTS_PER_FRAGMENT}
NUMBER_OF_FRAGMENTS = {NUMBER_OF_FRAGMENTS}
DIMENSIONS = {DIMENSIONS}
NUMBER_OF_ITERATIONS = {NUMBER_OF_ITERATIONS}

USE_DATACLAY = {USE_DATACLAY}
USE_SPLIT = {USE_SPLIT}
USE_ACTIVE = {USE_ACTIVE}
""")
    start_time = time.time()

    rand_state = np.random.RandomState()

    dataset = ds.random_array((POINTS_PER_FRAGMENT * NUMBER_OF_FRAGMENTS, DIMENSIONS),
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
        dataset._blocks = compss_wait_on(dataset._blocks)

    print("Generation/Load done")
    initialization_time = time.time() - start_time
    print("Starting histogram")

    print("-----------------------------------------")
    print("-------------- RESULTS ------------------")
    print("-----------------------------------------")
    print("Initialization time: %f" % initialization_time)

    tadh["initialization_time"] = initialization_time
    tadh.write_all()
    tadh["iteration_time"] = list()

    time.sleep(10)

    # Run histogram
    result = None
    for i in range(NUMBER_OF_ITERATIONS):
        start_t = time.time()
        result = histogram(dataset)
        end_t = time.time()

        histogram_time = end_t - start_t
        print("Histogram time (#%d/%d): %f" % (i + 1, NUMBER_OF_ITERATIONS, histogram_time))
        tadh["iteration_time"].append(histogram_time)
        tadh.write_all()

    print(result)


if __name__ == "__main__":
    main()
