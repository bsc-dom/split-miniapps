"""This class may be better inside dataClay [contrib] code.

However, I was having some issues regarding model registration and general
usability of the classes and splits. So that ended up here.
"""

from dataclay import DataClayObject, dclayMethod


class ChunkSplit(DataClayObject):
    """
    @ClassField _chunks list<storageobject>
    @ClassField storage_location anything
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
