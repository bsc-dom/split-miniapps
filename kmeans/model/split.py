"""This class may be better inside dataClay [contrib] code.

However, I was having some issues regarding model registration and general
usability of the classes and splits. So that ended up here.
"""

from dataclay import DataClayObject, dclayMethod
import numpy as np

try:
    from pycompss.api.task import task
    from pycompss.api.parameter import IN
except ImportError:
    from dataclay.contrib.dummy_pycompss import task, IN


class KMeansSplit(DataClayObject):
    """Generic and simple split.

    @ClassField _chunks anything
    @ClassField _idx anything
    @ClassField split_brothers list<storageobject>
    @ClassField backend anything
    @dclayImport numpy as np
    """

    @dclayMethod(backend="anything")
    def __init__(self, backend):
        """Build a LocalIterator through a list of chunks.

        :param chunks: Sequence of (iterable) chunks.
        """
        # If this is not being called remotely, better to coerce to list right now
        self._chunks = list()
        self._idx = list()
        self.backend = backend
        self.split_brothers = list()

    @dclayMethod(idx="anything", obj="anything")
    def add_object(self, idx, obj):
        self._chunks.append(obj)
        self._idx.append(idx)

    @task(target_direction=IN, returns=object)
    @dclayMethod(centers="anything", return_="anything")
    def compute(self, centers):
        return np.sum([frag.partial_sum(centers) for frag in self._chunks], axis=0)
