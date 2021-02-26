import time
import numpy as np
import os

from pycompss.api.task import task
from pycompss.api.api import compss_wait_on, compss_barrier
from pycompss.api.reduction import reduction
from pycompss.api.parameter import COLLECTION_IN

from tad4bj.slurm import handler as tadh


#############################################
# Constants / experiment values:
#############################################

POINTS_PER_FRAGMENT = int(os.environ["POINTS_PER_FRAGMENT"])
NUMBER_OF_FRAGMENTS = int(os.environ["NUMBER_OF_FRAGMENTS"])
NUMBER_OF_ITERATIONS = int(os.getenv("NUMBER_OF_ITERATIONS", "10"))

USE_REDUCTION_DECORATOR = bool(int(os.environ["USE_REDUCTION_DECORATOR"]))
REDUCTION_CHUNK_SIZE = int(os.getenv("REDUCTION_CHUNK_SIZE", "48"))

SEED = 420

#############################################
#############################################

@task(partials=COLLECTION_IN, returns=object)
def sum_partials(partials):
    return np.sum(partials, axis=0)


if USE_REDUCTION_DECORATOR:
    # Non-canonical way of applying a decorator programatically
    sum_partials = reduction(chunk_size=str(REDUCTION_CHUNK_SIZE))(sum_partials)
    # dear reader: despite your reluctance, I assure you this is legit.
    # p.s.: unless you are wondering why chunk_size parameter is a str. Beats me. COMPSs' doc says so.


@task(returns=object)
def generate_fragment(seed):
    # Random generation distributions
    np.random.seed(seed)
    return np.random.f(10, 2, POINTS_PER_FRAGMENT)


@task(returns=object)
def partial_histogram(values, bins):
    values, _ = np.histogram(values, bins)
    return values


def histogram(experiment):
    bins = np.concatenate((np.arange(0,10, 0.1), np.arange(10, 50), [np.infty]))

    partials = list()

    for fragment in experiment:
        partial = partial_histogram(fragment, bins)
        partials.append(partial)

    result = sum_partials(partials)

    return compss_wait_on(result)


def main():
    print(f"""Starting experiment with the following:

POINTS_PER_FRAGMENT = {POINTS_PER_FRAGMENT}
NUMBER_OF_FRAGMENTS = {NUMBER_OF_FRAGMENTS}
NUMBER_OF_ITERATIONS = {NUMBER_OF_ITERATIONS}
""")
    start_time = time.time()

    experiment = list()
    for i in range(NUMBER_OF_FRAGMENTS):
        print("Proceeding to generate fragment #%d" % (i + 1))
        experiment.append(generate_fragment(SEED + i))

    compss_barrier()

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
        result = histogram(experiment)
        end_t = time.time()

        histogram_time = end_t - start_t
        print("Histogram time (#%d/%d): %f" % (i + 1, NUMBER_OF_ITERATIONS, histogram_time))
        tadh["iteration_time"].append(histogram_time)

    tadh.write_all()
    print(result)


if __name__ == "__main__":
    main()
