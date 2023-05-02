from multiprocessing.pool import ThreadPool, Pool
import os.path
import sys
import pickle
from time import time

import numpy as np
from sklearn.neighbors import NearestNeighbors

NUM_POINTS = 50000
DIMENSIONS = 3

BLOCKS = [np.random.random((NUM_POINTS, DIMENSIONS)) for i in range(16)]
SWEEP_MULTIPLIER = [1, 2, 3, 4, 5, 6]


def __sizeof__(self):
    return int(1.5 * (sys.getsizeof(self._fit_X) + sys.getsizeof(self._tree) + sys.getsizeof(self.n_samples_fit_)))

NearestNeighbors.__sizeof__ = __sizeof__

def do_fit(i):
    nn = NearestNeighbors()
    start_t = time()
    nn.fit(BLOCKS[i])
    end_t = time()
    return end_t - start_t


def fit_size():
    nn = NearestNeighbors()
    nn.fit(BLOCKS[0])
    print("Size of nn: %d" % sys.getsizeof(nn))
    print("Serialization size: %d" % len(pickle.dumps(nn)))

    # For comparison
    print("Size of the block: %d" % sys.getsizeof(BLOCKS[0]))
    print("Serialization size: %d" % len(pickle.dumps(BLOCKS[0])))


def fit_cost():
    sequential_times = list()
    for i in range(16):
        sequential_times.append(do_fit(i))
    
    print("**************************")
    print("Sequential times:")
    print(sequential_times)
    print("**************************")
    print()

    # With multiprocessing
    with Pool(4) as p:
        print("**************************")
        print("Multiprocessing (4):")
        print(p.map(do_fit, range(16)))
        print("**************************")
        print()

    with Pool(16) as p:
        print("**************************")
        print("Multiprocessing (16):")
        print(p.map(do_fit, range(16)))
        print("**************************")
        print()

    # With multithreading
    with ThreadPool(4) as p:
        print("**************************")
        print("Multithreading (4):")
        print(p.map(do_fit, range(16)))
        print("**************************")
        print()

    with ThreadPool(16) as p:
        print("**************************")
        print("Multithreading (16):")
        print(p.map(do_fit, range(16)))
        print("**************************")
        print()


def kneighbors_cost_fit_growth():
    with open(os.path.expanduser("~/data_nn_benchmark.csv"), "at") as f:
        print("**************************")
        for k in SWEEP_MULTIPLIER:
            print("Multiplier = %d" % k)
            b = np.random.random((NUM_POINTS * k, DIMENSIONS))
            nn = NearestNeighbors()

            start_t = time()
            nn.fit(b)
            fit_time = time() - start_t

            print("fit time = %f" % fit_time)

            start_t = time()
            nn.kneighbors(X=BLOCKS[k], n_neighbors=5)
            kneighbors_time = time() - start_t

            print("kneighbors time = %f" % kneighbors_time)
            print("------------------------")
            f.write("%d,%f,%f\n" % (k, fit_time, kneighbors_time))
        print("**************************")
        print()


if __name__ == "__main__":
    fit_size()

    # fit_cost()

    # for i in range(10):
    #     kneighbors_cost_fit_growth()
