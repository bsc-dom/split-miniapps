"""This class may be better inside dataClay [contrib] code.

However, I was having some issues regarding model registration and general
usability of the classes and splits. So that ended up here.
"""

import numpy as np
from dataclay import DataClayObject, dclayMethod

try:
    from pycompss.api.task import task
    from pycompss.api.parameter import IN
except ImportError:
    from dataclay.contrib.dummy_pycompss import task, IN


class ChunkSplit(DataClayObject):
    """
    @ClassField _chunks list<storageobject>
    @ClassField storage_location anything
    @dclayImport numpy as np
    """

    @dclayMethod(chunks="list<storageobject>", storage_location="anything")
    def __init__(self, chunks, storage_location):
        """Build a LocalIterator through a list of chunks.

        :param chunks: Sequence of (iterable) chunks.
        """
        # If this is not being called remotely, better to coerce to list right now
        self._chunks = list(chunks)
        self.storage_location = storage_location

    # Note that the return is not serializable, thus the _local flag
    @dclayMethod(return_="anything", _local=True)
    def __iter__(self):
        return iter(self._chunks)

    @task(target_direction=IN, returns=object)
    @dclayMethod(centers="anything", return_="anything")
    def compute(self, centers):
        subresults = list()
        for frag in self._chunks:
            subresults.append(frag.partial_sum(centers))

        return subresults
