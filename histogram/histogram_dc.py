import time
import numpy as np
import os

from itertools import cycle

from pycompss.api.task import task
from pycompss.api.api import compss_wait_on

from dataclay import api
from dataclay.contrib.splitting import split

## Not needed while using PyCOMPSs
#api.init()

from model.fragment import Fragment, FragmentList
from model.split import ChunkSplit

from tad4bj.slurm import handler as tadh


#############################################
# Constants / experiment values:
#############################################

POINTS_PER_FRAGMENT = int(os.environ["POINTS_PER_FRAGMENT"])
NUMBER_OF_FRAGMENTS = int(os.environ["NUMBER_OF_FRAGMENTS"])
NUMBER_OF_ITERATIONS = int(os.getenv("NUMBER_OF_ITERATIONS", "10"))

USE_SPLIT = bool(int(os.environ["USE_SPLIT"]))
ROUNDROBIN_PERSISTENCE = bool(int(os.environ["ROUNDROBIN_PERSISTENCE"]))

SEED = 420

#############################################
#############################################


@task(returns=object)
def generate_fragment(seed, backend=None):
    frag = Fragment()
    frag.make_persistent(backend_id=backend)
    frag.generate_values(POINTS_PER_FRAGMENT, seed)

    return frag


@task(returns=object)
def compute_partition(partition, bins):
    subresults = list()
    for frag in partition:
        partial = frag.partial_histogram(bins)
        subresults.append(partial)

    return np.sum(subresults, axis=0)


def histogram(experiment):
    bins = np.concatenate((np.arange(0,10, 0.1), np.arange(10, 50), [np.infty]))

    partials = list()
    if USE_SPLIT:
        for partition in split(experiment, split_class=ChunkSplit):
            partial = compute_partition(partition, bins)
            partials.append(partial)
    else:
        for fragment in experiment.chunks:
            partial = fragment.partial_histogram(bins)
            partials.append(partial)

    partials = compss_wait_on(partials)

    return np.sum(partials, axis=0)


def main():
    print(f"""Starting experiment with the following:

POINTS_PER_FRAGMENT = {POINTS_PER_FRAGMENT}
NUMBER_OF_FRAGMENTS = {NUMBER_OF_FRAGMENTS}
NUMBER_OF_ITERATIONS = {NUMBER_OF_ITERATIONS}

USE_SPLIT = {USE_SPLIT}
ROUNDROBIN_PERSISTENCE = {ROUNDROBIN_PERSISTENCE}
""")
    start_time = time.time()

    backends = list(api.get_backends_info().keys())
    experiment_fragments = list()
    for i, backend in zip(range(NUMBER_OF_FRAGMENTS), cycle(backends)):
        print("Proceeding to generate fragment #%d" % (i + 1))
        if ROUNDROBIN_PERSISTENCE:
            b = backend
        else:
            b = None

        experiment_fragments.append(generate_fragment(SEED + i, b))

    experiment_fragments = compss_wait_on(experiment_fragments)

    experiment = FragmentList()
    experiment.make_persistent()
    experiment.chunks = experiment_fragments

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
    ## Not needed while using PyCOMPSs
    #api.finish()
