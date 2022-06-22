from itertools import cycle
import os
import time

from dislib.data.array import Array
from dislib.classification import CascadeSVM
import dislib as ds

from sklearn import datasets
from sklearn.metrics import confusion_matrix
import numpy as np

from pycompss.api.task import task
from pycompss.api.api import compss_wait_on, compss_barrier

from tad4bj.slurm import handler as tadh


#############################################
# Constants / experiment values:
#############################################

POINTS_PER_FRAGMENT = int(os.environ["POINTS_PER_FRAGMENT"])
NUMBER_OF_FRAGMENTS = int(os.environ["NUMBER_OF_FRAGMENTS"])
DIMENSIONS = int(os.getenv("DIMENSIONS", "30"))
NUMBER_OF_CENTERS = int(os.getenv("NUMBER_OF_CENTERS", "15"))
NUMBER_OF_CSVM_ITERATIONS = int(os.getenv("NUMBER_OF_CSVM_ITERATIONS", "10"))

USE_DATACLAY = bool(int(os.environ["USE_DATACLAY"]))
USE_SPLIT = bool(int(os.environ["USE_SPLIT"]))

CASCADE_ARITY = int(os.getenv("REDUCTION_ARITY", "4"))

SEED = 42

CHECK_RESULT = True

#############################################
#############################################

if USE_DATACLAY:
    from dataclay import api

    api.init()
    
    from dislib_model.block import PersistentBlock


##################
# Note to reader #
##################
#
# The CascadeSVM for this application has been patched onto
# the modified dislib version used. This is because CSVM is
# a little bit more convoluted that other applications, and
# integration is a little bit lengthier than the other ones.


@task(returns=2)
def generate_points(num_points, num_centers, dim, seed, backend=None):
    # We are mimicking the center generation that datasets.make_blobs do,
    # but using a fixed seed which makes the centers equal for all generate_points call
    np.random.seed(777)
    centers = np.random.uniform(-1, 1, (num_centers, dim)) * 10

    points, labels = datasets.make_blobs(
        n_samples=num_points, n_features=dim, centers=centers, shuffle=True, random_state=seed)

    if USE_DATACLAY:
        block = PersistentBlock(points)
        block.make_persistent(backend_id=backend)
        ret = block
    else:
        ret = points

    return ret, labels.reshape(-1, 1)


def main():

    print(f"""Starting experiment with the following:

POINTS_PER_FRAGMENT = {POINTS_PER_FRAGMENT}
NUMBER_OF_FRAGMENTS = {NUMBER_OF_FRAGMENTS}
DIMENSIONS = {DIMENSIONS}
NUMBER_OF_CENTERS = {NUMBER_OF_CENTERS}
NUMBER_OF_CSVM_ITERATIONS = {NUMBER_OF_CSVM_ITERATIONS}
CASCADE_ARITY = {CASCADE_ARITY}

USE_DATACLAY = {USE_DATACLAY}
USE_SPLIT = {USE_SPLIT}
""")

    start_time = time.time()

    if USE_DATACLAY:
        backends = list(api.get_backends_info().keys())
        print("Using the following dataClay backends:\n%s" % backends)
    else:
        # Dummy unused, it will be cycle-d, makes code more consistent (less branching)
        backends = [None]

    dataset_blocks = list()
    nested_labels = list()

    for i, backend in zip(range(NUMBER_OF_FRAGMENTS), cycle(backends)):
        d, l = generate_points(POINTS_PER_FRAGMENT, NUMBER_OF_CENTERS, DIMENSIONS, SEED+i, backend=backend)
        dataset_blocks.append(d)
        nested_labels.append(l)

    compss_barrier()

    print("Generation/Load done")
    initialization_time = time.time() - start_time
    print("Starting CSVM")

    # Make list of rows; each row has a single block element
    dataset_blocks = [[block] for block in dataset_blocks]
    labels_blocks = [[block] for block in nested_labels]

    dataset = Array(blocks=dataset_blocks, top_left_shape=(POINTS_PER_FRAGMENT, DIMENSIONS), 
                    reg_shape=(POINTS_PER_FRAGMENT, DIMENSIONS), 
                    shape=(POINTS_PER_FRAGMENT * NUMBER_OF_FRAGMENTS, DIMENSIONS), 
                    sparse=False)

    arr_labels = Array(blocks=labels_blocks, top_left_shape=(POINTS_PER_FRAGMENT, 1), 
                       reg_shape=(POINTS_PER_FRAGMENT, 1), 
                       shape=(POINTS_PER_FRAGMENT * NUMBER_OF_FRAGMENTS, 1), 
                       sparse=False)

    print("-----------------------------------------")
    print("-------------- RESULTS ------------------")
    print("-----------------------------------------")
    print("Initialization time: %f" % initialization_time)

    tadh["initialization_time"] = initialization_time
    tadh["execution_time"] = list()
    tadh.write_all()

    time.sleep(10)

    # Run csvm
    start_t = time.time()

    svm = CascadeSVM(cascade_arity=CASCADE_ARITY, max_iter=NUMBER_OF_CSVM_ITERATIONS, check_convergence=False, random_state=SEED)
    svm.fit(dataset, arr_labels)

    compss_barrier()

    end_t = time.time()

    csvm_time = end_t - start_t
    print("CSVM time #1: %f" % csvm_time)
    tadh["execution_time"].append(csvm_time)
    tadh.write_all()

    # Let's check behaviour
    if CHECK_RESULT:
        print("Checking result ...")
        test_data, y_true = generate_points(1000, NUMBER_OF_CENTERS, DIMENSIONS, SEED - 1)
        y_pred = svm.predict(ds.array(compss_wait_on(test_data), block_size=(1000, DIMENSIONS)))
        print("Confusion matrix for the result:\n%s" % (confusion_matrix(compss_wait_on(y_true), y_pred.collect())))


if __name__ == "__main__":
    main()
