import numpy as np

from dataclay import DataClayObject, dclayMethod
from dataclay.contrib.splitting import SplittableCollectionMixin

try:
    from pycompss.api.task import task
    from pycompss.api.parameter import IN
except ImportError:
    from dataclay.contrib.dummy_pycompss import task, IN


class FragmentList(DataClayObject, SplittableCollectionMixin):
    """
    @ClassField chunks list<storageobject>
    """
    @dclayMethod()
    def __init__(self):
        self.chunks = list()


class Fragment(DataClayObject):
    """
    @ClassField values numpy.ndarray

    @dclayImport numpy as np
    """
    @dclayMethod()
    def __init__(self):
        self.values = None

    @dclayMethod(num_values='int', seed='int')
    def generate_values(self, num_values, seed):
        """Generate values following a certain distribution.

        :param num_values: Number of points
        :param distribution_func_name: The numpy.random's name for the distribution that 
         will be used to generate the values.
        :param seed: Random seed
        :return: Dataset fragment
        """
        # Random generation distributions
        np.random.seed(seed)
        values = np.random.f(10, 2, num_values)

        self.values = values

    @task(target_direction=IN, returns=object)
    @dclayMethod(bins="numpy.ndarray", return_="numpy.ndarray")
    def partial_histogram(self, bins):
        values, _ = np.histogram(self.values, bins)
        return values
