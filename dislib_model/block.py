import numpy as np
from sklearn.metrics import pairwise_distances

from dataclay import DataClayObject, dclayMethod

try:
    from pycompss.api.task import task
    from pycompss.api.parameter import IN, Depth, Type, COLLECTION_IN

except ImportError:
    from dataclay.contrib.dummy_pycompss import task, IN, Depth, Type, COLLECTION_IN


class PersistentBlock(DataClayObject):
    """A Persistent Block class, intended for usage with dislib library.

    CAUTION! There are some methods, such as .transpose(), that typically
    return a **view** of the array. If the caller relies on that and
    and changes values, those won't reflect on the persistent block _unless_
    this method is called from within the same ExecutionEnvironment. It's
    quite a nasty corner case.

    @dclayImport numpy as np
    @dclayImportFrom sklearn.metrics import pairwise_distances

    @ClassField block_data numpy.ndarray
    @ClassField shape anything
    @ClassField ndim anything
    @ClassField nbytes anything
    @ClassField itemsize anything
    @ClassField size anything
    """
    @dclayMethod(data="numpy.ndarray")
    def __init__(self, data):
        self.block_data = data
        self.shape = data.shape
        self.ndim = data.ndim
        self.size = data.size
        self.itemsize = data.itemsize
        self.nbytes = data.nbytes

    @dclayMethod(key="anything", return_="anything")
    def __getitem__(self, key):
        return self.block_data[key]

    @dclayMethod(key="anything", value="anything")
    def __setitem__(self, key, value):
        self.block_data[key] = value

    @dclayMethod(index="anything")
    def __delitem__(self, key):
        """Delete an item"""
        del self.block_data[key]

    @dclayMethod(return_="numpy.ndarray")
    def __array__(self):
        """This is used internally by numpy.

        This method allows the PersistentBlock class will work seamlessly with
        most of the numpy library, however that may come with a performance
        cost. It may be wise to fine tune certain critical path for efficiency
        reasons.
        """
        return self.block_data

    @dclayMethod(return_="anything")
    def transpose(self):
        # TODO: Address the PersistentBlock.T, which is property/attribute/thingy
        return self.block_data.transpose()

    @dclayMethod(return_=int)
    def __len__(self):
        return len(self.block_data)

    # Not true, it is INOUT, but I will perform a barrier before
    # anything dangerous, so it's safe to lie and cheat a lil' bit
    @task(target_direction=IN)
    @dclayMethod(rotation_matrix="numpy.ndarray")
    def rotate_in_place(self, rotation_matrix):
        self.block_data = self.block_data @ rotation_matrix

    # To be used by KMeans.
    @task(target_direction=IN, returns=object)
    @dclayMethod(centers='numpy.ndarray', return_='anything')
    def partial_sum(self, centers):
        partials = np.zeros((centers.shape[0], 2), dtype=object)
        arr = self.block_data
        close_centers = pairwise_distances(arr, centers).argmin(axis=1)
        for center_idx in range(len(centers)):
            indices = np.argwhere(close_centers == center_idx).flatten()
            partials[center_idx][0] = np.sum(arr[indices], axis=0)
            partials[center_idx][1] = indices.shape[0]

        return partials

    @task(target_direction=IN, returns=object)
    @dclayMethod(n_bins="int", n_dimensions="int", return_="numpy.ndarray")
    def partial_histogram(self, n_bins, n_dimensions):
        values, _ = np.histogramdd(self.block_data, n_bins, [(0, 1)] * n_dimensions)
        return values
