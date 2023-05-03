from itertools import cycle
import time
import os

import dask
import dask.array as da
from dask.distributed import wait, Client, get_worker
import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from tad4bj.slurm import handler as tadh


#############################################
# Constants / experiment values:
#############################################

POINTS_PER_FRAGMENT = int(os.environ["POINTS_PER_FRAGMENT"])
NUMBER_OF_FRAGMENTS = int(os.environ["NUMBER_OF_FRAGMENTS"])
DIMENSIONS = int(os.getenv("DIMENSIONS", "20"))
NUMBER_OF_CENTERS = int(os.getenv("NUMBER_OF_CENTERS", "8"))
NUMBER_OF_CSVM_ITERATIONS = int(os.getenv("NUMBER_OF_CSVM_ITERATIONS", "5"))

USE_SPLIT = bool(int(os.environ["USE_SPLIT"]))

DASK_RECHUNK = int(os.getenv("DASK_RECHUNK", "0"))
if DASK_RECHUNK > 0:
    EFFECTIVE_POINTS_PER_FRAGMENT = POINTS_PER_FRAGMENT * DASK_RECHUNK
else:
    EFFECTIVE_POINTS_PER_FRAGMENT = POINTS_PER_FRAGMENT

SEED = 42

PROFILE_SPLIT = bool(int(os.getenv("PROFILE_SPLIT", "0")))
CASCADE_ARITY = int(os.getenv("REDUCTION_ARITY", "48"))

CHECK_RESULT = False

#############################################
#############################################

class SplitPartitionHelper:
    def __init__(self, multi):
        self.all_cycles = dict()
        self.multi = multi
        self.worker_indexed = dict()

    def __getitem__(self, key):
        try:
            return self.all_cycles[key]
        except KeyError:
            new_lists = [list() for _ in range(self.multi)]
            self.all_cycles[key] = cycle(new_lists)
            self.worker_indexed[key] = new_lists
            return self.all_cycles[key]

    def get_partitions(self):
        return self.worker_indexed


def dask_split(client, array, multiplicity=24):
    sph = SplitPartitionHelper(multiplicity)
    for f_str, v in client.who_has(array).items():
        tup = eval(f_str)
        # The object has the form ('concatenate-<identifier>', x, y)
        # Where x is the position and y is zero (because there are no blocks in that dimension)
        next(sph[v[0]]).append((tup[1], f_str))
    return sph.get_partitions()


def generate_points(num_points, num_centers, dim, seed):
    # We are mimicking the center generation that datasets.make_blobs do,
    # but using a fixed seed which makes the centers equal for all generate_points call
    np.random.seed(777)
    centers = np.random.uniform(-1, 1, (num_centers, dim)) * 10

    points, labels = datasets.make_blobs(
        n_samples=num_points, n_features=dim, centers=centers, shuffle=True, random_state=seed)

    return points, labels


class DaskSVM:
    # This resembles the dislib one, just for comparison.
    # A fair comparison should use the same algorithm and 
    # implementation must resemble one another
    def __init__(self, client=None):
        self._sv_extras = None
        self._client = client

    def fit(self, x, y):
        if USE_SPLIT:
            return self._fit_with_split(x, y)
        
        ids = da.random.randint(2**16, size=(POINTS_PER_FRAGMENT * NUMBER_OF_FRAGMENTS,), chunks=(EFFECTIVE_POINTS_PER_FRAGMENT,))

        for _ in range(NUMBER_OF_CSVM_ITERATIONS - 1):
            self._do_iteration(x, y, ids)
        self._do_iteration(x, y, ids, last_iteration=True)
    
    def _fit_with_split(self, x, y):
        print("Performing the split")
        partitions_by_worker = dask_split(self._client, x)
        digested_struct = dict()

        for w, partitions in partitions_by_worker.items():
            nested_parted_x = list()
            nested_parted_y = list()
            nested_sizes = list()

            for partition_data in partitions:
                indexes, part = tuple(zip(*partition_data))
                nested_sizes.append(sum(x.blocks[i].shape[0] for i in indexes))

                nested_parted_x.append(part)
                y_blocks = [y.blocks[i] for i in indexes]
                # I am not sure if `workers` parameter works here, but it should not be harmful
                # and is more consistent in the general approach of worker-local parameters
                nested_parted_y.append(self._client.compute(y_blocks, workers=w))

            nested_parted_ids = self._client.map(_random_ids, nested_sizes, workers=w, pure=False)
            digested_struct[w] = (nested_parted_x, nested_parted_y, nested_parted_ids)

        for _ in range(NUMBER_OF_CSVM_ITERATIONS - 1):
            self._do_iteration_with_split(digested_struct)
        self._do_iteration_with_split(digested_struct, last_iteration=True)

    def predict(self, x):
        return self._clf.predict(x)
    
    def _do_iteration_with_split(self, digested_struct, last_iteration=False):
        q = []

        # first level
        if self._sv_extras is not None:
            extras = self._sv_extras
        else:
            extras = None

        for w, (nested_parted_x, nested_parted_y, nested_parted_ids) in digested_struct.items():
            train_data = self._client.map(
                _train_partition,
                nested_parted_x,
                nested_parted_y,
                nested_parted_ids,
                [extras] * len(nested_parted_x),
                workers=w)
            q.append(self._client.submit(_train_reduction, train_data, workers=w))

        res = self._client.submit(_train_reduction, q, last_iteration)

        if last_iteration:
            clf = self._client.submit(lambda x: x[3], res)
            self._clf = clf.result()
        else:
            self._sv_extras = self._client.replicate(res)

    def _do_iteration(self, x, y, ids, last_iteration=False):
        q = []

        # first level
        if self._sv_extras is not None:
            extras = self._sv_extras
        else:
            extras = None

        for x_block, y_block, id_block in zip(x.blocks, y.blocks, ids.blocks):
            res = _train(x_block, y_block, id_block, extras=extras)
            q.append(res)

        while len(q) > CASCADE_ARITY:
            data = q[:CASCADE_ARITY]
            del q[:CASCADE_ARITY]

            res = _stack_and_train(data)
            q.append(res)

        # last layer
        res = _stack_and_train(q, return_clf=last_iteration)

        if last_iteration:
            clf = dask.delayed(lambda x: x[3])(res)
            self._clf = clf.compute()
        else:
            self._sv_future = res


def _random_ids(size):
    return np.random.randint(2**16, size=size)


def _train_partition(parted_x, parted_y, ids, extras=None):
    worker = get_worker()
    parted_x = [worker.data[block] for block in parted_x]
    
    clf = SVC()
    if extras is not None:
        _svs, _sv_labels, _sv_ids = extras
        x = np.vstack(parted_x + [_svs])
        y = np.hstack(parted_y + [_sv_labels])
        ids = np.hstack([ids, _sv_ids])
    else:
        x = np.vstack(parted_x)
        y = np.hstack(parted_y)

    # Code in _merge function, put here because reasons
    _, uniques = np.unique(ids, return_index=True)
    indices = np.argsort(uniques)
    uniques = uniques[indices]

    x = x[uniques]
    y = y[uniques]
    ids = ids[uniques]
    # _merge is done

    clf.fit(x, y)

    sv = x[clf.support_]
    sv_labels = y[clf.support_]
    sv_ids = ids[clf.support_]

    return sv, sv_labels, sv_ids


def _train_reduction(q, return_clf=False):
    x_list = [tup[0] for tup in q]
    y_list = [tup[1] for tup in q]
    ids_list = [tup[2] for tup in q]

    clf = SVC()

    x = np.vstack(x_list)
    y = np.hstack(y_list)
    ids = np.hstack(ids_list)

    # Code in _merge function, put here because reasons
    _, uniques = np.unique(ids, return_index=True)
    indices = np.argsort(uniques)
    uniques = uniques[indices]

    x = x[uniques]
    y = y[uniques]
    ids = ids[uniques]
    # _merge is done

    clf.fit(x, y)

    sv = x[clf.support_]
    sv_labels = y[clf.support_]
    sv_ids = ids[clf.support_]

    if return_clf:
        return sv, sv_labels, sv_ids, clf
    else:
        return sv, sv_labels, sv_ids


@dask.delayed
def _train(base_x, base_y, base_ids, extras=None):
    clf = SVC()
    if extras is not None:
        _svs, _sv_labels, _sv_ids = extras
        x = np.vstack([base_x, _svs])
        y = np.hstack([base_y, _sv_labels])
        ids = np.hstack([base_ids, _sv_ids])
    else:
        x, y, ids = base_x, base_y, base_ids

    # Code in _merge function, put here because reasons
    _, uniques = np.unique(ids, return_index=True)
    indices = np.argsort(uniques)
    uniques = uniques[indices]

    x = x[uniques]
    y = y[uniques]
    ids = ids[uniques]
    # _merge is done

    clf.fit(x, y)

    sv = x[clf.support_]
    sv_labels = y[clf.support_]
    sv_ids = ids[clf.support_]

    return sv, sv_labels, sv_ids


@dask.delayed
def _stack_and_train(q, return_clf=False):
    # Sorry for repeating myself, but all this is a PoC
    # and further optimizations could be done within dask
    # to exploit even more fundamental structures
    x_list = [tup[0] for tup in q]
    y_list = [tup[1] for tup in q]
    ids_list = [tup[2] for tup in q]

    clf = SVC()

    x = np.vstack(x_list)
    y = np.hstack(y_list)
    ids = np.hstack(ids_list)

    # Code in _merge function, put here because reasons
    _, uniques = np.unique(ids, return_index=True)
    indices = np.argsort(uniques)
    uniques = uniques[indices]

    x = x[uniques]
    y = y[uniques]
    ids = ids[uniques]
    # _merge is done

    clf.fit(x, y)

    sv = x[clf.support_]
    sv_labels = y[clf.support_]
    sv_ids = ids[clf.support_]

    if return_clf:
        return sv, sv_labels, sv_ids, clf
    else:
        return sv, sv_labels, sv_ids


def main():
    # dask client
    client = Client(scheduler_file=os.path.expandvars("$HOME/dask-scheduler-$SLURM_JOB_ID.json"))
    print("Dask client: %s" % client)
    print(client)

    print(f"""Starting experiment with the following:

POINTS_PER_FRAGMENT = {POINTS_PER_FRAGMENT}
NUMBER_OF_FRAGMENTS = {NUMBER_OF_FRAGMENTS}
DIMENSIONS = {DIMENSIONS}
NUMBER_OF_CENTERS = {NUMBER_OF_CENTERS}
NUMBER_OF_CSVM_ITERATIONS = {NUMBER_OF_CSVM_ITERATIONS}
CASCADE_ARITY = {CASCADE_ARITY}

USE_SPLIT = {USE_SPLIT}
DASK_RECHUNK = {DASK_RECHUNK}
""")
    start_time = time.time()

    dataset_blocks = list()
    labels_blocks = list()

    for i in range(NUMBER_OF_FRAGMENTS):
        d, l = dask.delayed(generate_points, nout=2)(POINTS_PER_FRAGMENT, NUMBER_OF_CENTERS, DIMENSIONS, SEED+i)
        dataset_blocks.append(da.from_delayed(d, shape=(POINTS_PER_FRAGMENT, DIMENSIONS), dtype=np.float64))
        labels_blocks.append(da.from_delayed(l, shape=(POINTS_PER_FRAGMENT,), dtype=np.int64))

    dataset = da.concatenate(dataset_blocks).persist()
    labels = da.concatenate(labels_blocks).persist()

    wait(dataset)
    wait(labels)

    print("Generation/Load done")
    initialization_time = time.time() - start_time
    print("Starting CSVM")

    print("-----------------------------------------")
    print("-------------- RESULTS ------------------")
    print("-----------------------------------------")
    print("Initialization time: %f" % initialization_time)

    tadh["initialization_time"] = initialization_time
    tadh.write_all()
    tadh["execution_time"] = list()

    time.sleep(10)

    # Run CSVM
    start_t = time.time()
    svm = DaskSVM(client=client)

    if DASK_RECHUNK > 0:
        print("Issuing a rechunk operation")
        dataset = dataset.rechunk((EFFECTIVE_POINTS_PER_FRAGMENT, DIMENSIONS))
        labels = labels.rechunk((EFFECTIVE_POINTS_PER_FRAGMENT,))

    svm.fit(dataset, labels)

    end_t = time.time()

    csvm_time = end_t - start_t
    print("CSVM time #1: %f" % csvm_time)
    tadh["execution_time"].append(csvm_time)
    tadh.write_all()

    # Let's check behaviour
    if CHECK_RESULT:
        print("Checking result ...")
        test_data, y_true = generate_points(1000, NUMBER_OF_CENTERS, DIMENSIONS, SEED - 1)
        y_pred = svm.predict(test_data)
        print("Confusion matrix for the result:\n%s" % (confusion_matrix(y_true, y_pred)))


if __name__ == "__main__":
    main()
