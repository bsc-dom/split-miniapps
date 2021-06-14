from dataclay import DataClayObject, dclayMethod


class PersistentBlock(DataClayObject):
    """A Persistent Block class, intended for usage with dislib library.

    CAUTION! There are some methods, such as .transpose(), that typically
    return a **view** of the array. If the caller relies on that and
    and changes values, those won't reflect on the persistent block _unless_
    this method is called from within the same ExecutionEnvironment. It's
    quite a nasty corner case.

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
