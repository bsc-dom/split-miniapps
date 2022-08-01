from copy import copy

import numpy as np
from sklearn.neighbors import NearestNeighbors

from dataclay import DataClayObject, dclayMethod

try:
    from pycompss.api.task import task
    from pycompss.api.parameter import IN, INOUT, Depth, Type, COLLECTION_IN

except ImportError:
    from dataclay.contrib.dummy_pycompss import task, IN, INOUT, Depth, Type, COLLECTION_IN


class PersistentFitStructure(DataClayObject):
    """Split that tracks the internal item index for each chunk.

    @dclayImportFrom copy import copy
    @dclayImport numpy as np
    @dclayImportFrom sklearn.neighbors import NearestNeighbors
    @ClassField _nn anything
    @ClassField _itemindexes numpy.ndarray
    """

    @dclayMethod()
    def __init__(self):
        self._nn = None

    @task(target_direction=INOUT)
    @dclayMethod(partition="storageobject")
    def fit_split(self, partition):
        # partition is a split instance

        # The following may be hidden by implementing __array__ method in the split object
        #  1. Explicit is better than implicit.
        #  2. Simple is better than complex.
        # I decided that 1. trumps 2.
        subdataset = np.vstack(partition._chunks)

        self._nn = NearestNeighbors(n_jobs=1)
        self._nn.fit(subdataset)

        # This needs to be explicit
        self._itemindexes = partition.get_item_indexes()

    @task(target_direction=INOUT)
    @dclayMethod(blocks="anything", offset="int")
    def fit(self, blocks, offset):        
        subdataset = np.block(blocks)

        self._nn = NearestNeighbors()
        self._nn.fit(subdataset)

        # Trivial (when compared with fit_split)
        # but following the same pattern, so get_kneighbors work as well
        self._itemindexes = np.arange(offset, offset + len(subdataset))

    @task(target_direction=IN, q_blocks={Type: COLLECTION_IN, Depth: 2}, returns=tuple)
    @dclayMethod(q_blocks="list", n_neighbors="int", return_="anything")
    def get_kneighbors(self, q_blocks, n_neighbors):
        q_samples = np.block(q_blocks)

        # Prepare a new structure for the tree walk
        # (due to the lack of readonly/concurrent implementation in the KDTree sklearn implementation)
        nn = copy(self._nn)
        nn._tree = copy(self._nn._tree)

        # Note that the merge requires distances, so we ask for them
        dist, ind = nn.kneighbors(X=q_samples, n_neighbors=n_neighbors)

        return dist, self._itemindexes[ind]
        #            ^****** This converts the local indexes to global ones

    @task(target_direction=IN, q_blocks={Type: COLLECTION_IN, Depth: 2}, returns=tuple)
    @dclayMethod(q_blocks="list", n_neighbors="int", return_="anything")
    def get_kneighbors_nocopy(self, q_blocks, n_neighbors):
        q_samples = np.block(q_blocks)

        # Note that the merge requires distances, so we ask for them
        dist, ind = self._nn.kneighbors(X=q_samples, n_neighbors=n_neighbors)

        return dist, self._itemindexes[ind]
        #            ^****** This converts the local indexes to global ones

    @task(target_direction=IN, q_blocks={Type: COLLECTION_IN, Depth: 2}, returns=1)
    @dclayMethod(q_blocks="list", return_="str")
    def whatarethose(self, q_blocks):
        return str(q_blocks)

    @task(target_direction=IN, returns=tuple)
    @dclayMethod(q_samples="anything", n_neighbors="int", return_="anything")
    def get_kneighbors_preblocked(self, q_samples, n_neighbors):
        # Prepare a new structure for the tree walk
        # (due to the lack of readonly/concurrent implementation in the KDTree sklearn implementation)
        nn = copy(self._nn)
        nn._tree = copy(self._nn._tree)

        # Note that the merge requires distances, so we ask for them
        dist, ind = nn.kneighbors(X=q_samples, n_neighbors=n_neighbors)

        item_indexes = self._itemindexes[ind]

        return dist, item_indexes
        #            ^****** This converts the local indexes to global ones
