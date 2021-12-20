import numpy as np

from dataclay import DataClayObject, dclayMethod

try:
    from pycompss.api.task import task
    from pycompss.api.parameter import INOUT, IN
except ImportError:
    from dataclay.contrib.dummy_pycompss import task, INOUT, IN


class PersistentBlock(DataClayObject):
    """A Persistent Block class, intended for usage with dislib library.

    CAUTION! There are some methods, such as .transpose(), that typically
    return a **view** of the array. If the caller relies on that and
    and changes values, those won't reflect on the persistent block _unless_
    this method is called from within the same ExecutionEnvironment. It's
    quite a nasty corner case.

    @dclayImport numpy as np
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

    @task(target_direction=IN, returns=1)
    @dclayMethod(return_="numpy.ndarray")
    def get_block(self):
        return self.block_data

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

    @dclayMethod(other="numpy.ndarray", return_="numpy.ndarray")
    def __matmul__(self, other):
        return self.block_data @ other

    @dclayMethod(other="numpy.ndarray", return_="storageobject")
    def __iadd__(self, other):
        self.block_data += other
        return self

    @dclayMethod(return_="anything")
    def transpose(self):
        # TODO: Address the PersistentBlock.T, which is property/attribute/thingy
        return self.block_data.transpose()

    @dclayMethod(return_=int)
    def __len__(self):
        return len(self.block_data)

    @task(target_direction=INOUT, other=INOUT)
    @dclayMethod(other="storageobject", j="numpy.ndarray")
    def _rotate_with(self, other, j):
        if j is None:
            return
        
        other_data = np.copy(other.block_data)
        n = self.shape[1]

        other.block_data = self.block_data @ j[:n, n:] + other_data @ j[n:, n:]
        self.block_data = self.block_data @ j[:n, :n] + other_data @ j[n:, :n]

    @task(target_direction=IN, returns=1)
    @dclayMethod(axis="int", return_="numpy.ndarray")
    def norm(self, axis):
        return np.linalg.norm(self.block_data, axis=axis)

    @task(target_direction=INOUT, block_colj=INOUT, returns=2)
    @dclayMethod(block_colj="storageobject", eps="anything", return_="anything")
    def _compute_rotation_and_rotate(self, block_colj, eps):
        coli = self.block_data
        colj = block_colj.block_data

        bii = coli.T @ coli
        bjj = colj.T @ colj
        bij = coli.T @ colj

        min_shape = (min(bii.shape[0], bjj.shape[0]),
                    min(bii.shape[1], bjj.shape[1]))

        tol = eps * np.sqrt(np.sum([[bii[i][j] * bjj[i][j]
                                    for j in range(min_shape[1])]
                                    for i in range(min_shape[0])]))

        if np.linalg.norm(bij) <= tol:
            return None, False
        else:
            b = np.block([[bii, bij], [bij.T, bjj]])
            j, _, _ = np.linalg.svd(b)

            self._rotate_with(block_colj, j)

            return j, True

    @task(target_direction=IN, returns=1)
    @dclayMethod(return_="storageobject")
    def _compute_u_block(self):
        a_col = np.copy(self.block_data)
        norm = np.linalg.norm(a_col, axis=0)

        # replace zero norm columns of a with an arbitrary unitary vector
        zero_idx = np.where(norm == 0)
        a_col[0, zero_idx] = 1
        norm[zero_idx] = 1

        u_col = a_col / norm

        b = PersistentBlock(u_col)
        b.make_persistent()

        return b
