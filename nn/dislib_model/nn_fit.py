import numpy as np
from sklearn.neighbors import NearestNeighbors

from dataclay import DataClayObject, dclayMethod

try:
    from pycompss.api.task import task
    from pycompss.api.parameter import IN, Depth, Type, COLLECTION_IN

except ImportError:
    from dataclay.contrib.dummy_pycompss import task, IN, Depth, Type, COLLECTION_IN


class PersistentFitStructure(DataClayObject):
    """Split that tracks the internal item index for each chunk.

    @dclayImport numpy as np
    @dclayImportFrom sklearn.neighbors import NearestNeighbors
    @ClassField _nn anything
    @ClassField _itemindexes numpy.ndarray
    """

    @dclayMethod()
    def __init__(self):
        self._nn = None

    # Not true, it is INOUT, but I will perform a barrier before
    # anything dangerous, so it's safe to lie and cheat a lil' bit
    @task(target_direction=IN)
    @dclayMethod(partition="storageobject")
    def fit_split(self, partition):
        # partition is a split instance

        # The following may be hidden by implementing __array__ method in the split object
        #  > Explicit is better than implicit.
        #  > Simple is better than complex.
        subdataset = np.vstack(partition._chunks)

        self._nn = NearestNeighbors()
        self._nn.fit(subdataset)

        # This needs to be explicit
        self._itemindexes = partition.get_item_indexes()

    # A _true_ target_direction=IN
    @task(target_direction=IN, q_blocks={Type: COLLECTION_IN, Depth: 2}, returns=tuple)
    @dclayMethod(q_blocks="list", n_neighbors="int", return_="anything")
    def get_kneighbors(self, q_blocks, n_neighbors):
        q_samples = np.block(q_blocks)

        # Note that the merge requires distances, so we ask for them
        dist, ind = self._nn.kneighbors(X=q_samples, n_neighbors=n_neighbors)

        return dist, self._itemindexes[ind]
        #            ******* This converts the local indexes to global ones
