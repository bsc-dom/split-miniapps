import os
import time

import numpy as np

from dislib.data.array import Array
from dislib.math.base import svd
import dislib as ds

from pycompss.api.task import task
from pycompss.api.api import compss_wait_on, compss_barrier

from tad4bj.slurm import handler as tadh


#############################################
# Constants / experiment values:
#############################################

BLOCKSIZE = int(os.environ["BLOCKSIZE"])
ROW_BLOCKS = int(os.environ["ROW_BLOCKS"])
COL_BLOCKS = int(os.environ["COL_BLOCKS"])

BLOCKSIZE = (BLOCKSIZE * COL_BLOCKS // ROW_BLOCKS, BLOCKSIZE)

RANDOMIZE_PAIRINGS = bool(int(os.environ["RANDOMIZE_PAIRINGS"]))

SEED = 42

CHECK_RESULT = False

#############################################
#############################################

@task(returns=1)
def generate_block(blocksize, seed):
    np.random.seed(seed)
    return np.random.random(blocksize)

def generate_matrix():
    seed = SEED
    matrix = list()

    for i in range(ROW_BLOCKS):
        row = list()
        matrix.append(row)

        for j in range(COL_BLOCKS):
            seed += 1
            row.append(generate_block(BLOCKSIZE, seed))

    compss_barrier()

    return Array(blocks=matrix, top_left_shape=BLOCKSIZE,
                 reg_shape=BLOCKSIZE,
                 shape=(BLOCKSIZE[0] * ROW_BLOCKS, BLOCKSIZE[1] * COL_BLOCKS),
                 sparse=False)


def main():

    print(f"""Starting experiment with the following:

BLOCKSIZE = {BLOCKSIZE}
ROW_BLOCKS = {ROW_BLOCKS}
COL_BLOCKS = {COL_BLOCKS}
""")

    start_time = time.time()

    a = generate_matrix()

    compss_barrier()

    print("Generation/Load done")
    initialization_time = time.time() - start_time
    print("Starting SVD")

    print("-----------------------------------------")
    print("-------------- RESULTS ------------------")
    print("-----------------------------------------")
    print("Initialization time: %f" % initialization_time)

    tadh["initialization_time"] = initialization_time
    tadh["execution_time"] = list()
    tadh.write_all()

    time.sleep(10)

    # Run SVD
    start_t = time.time()

    u, s, v = svd(a, sort=False, randomize_pairings=RANDOMIZE_PAIRINGS, max_iterations=1)

    compss_barrier()

    end_t = time.time()

    svd_time = end_t - start_t
    print("SVD time #1: %f" % svd_time)
    tadh["execution_time"].append(svd_time)
    tadh.write_all()

    # Let's check behaviour
    if CHECK_RESULT:
        u = u.collect()
        s = s.collect()
        v = v.collect()
        a = a.collect()
        print("a:\n%s\n" % a)
        print("u:\n%s\n" % u)
        print("s: %s\n" % s)
        print("v:\n%s\n" % v)
        print(np.allclose(a, u @ np.diag(s) @ v.T))


if __name__ == "__main__":
    main()
