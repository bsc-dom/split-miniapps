from dataclay import DataClayObject, dclayMethod

try:
    from pycompss.api.task import task
    from pycompss.api.parameter import IN
except ImportError:
    from dataclay.contrib.dummy_pycompss import task, IN

import numpy as np
from sklearn.metrics import pairwise_distances


class Fragment(DataClayObject):
    """
    @ClassField points numpy.ndarray

    @dclayImport numpy as np
    @dclayImportFrom sklearn.metrics import pairwise_distances
    """
    @dclayMethod()
    def __init__(self):
        self.points = None

    @task(target_direction=IN, returns=object)
    @dclayMethod(centers='numpy.ndarray', return_='anything')
    def partial_sum(self, centers):
        partials = np.zeros((centers.shape[0], 2), dtype=object)
        arr = self.points
        close_centers = pairwise_distances(arr, centers).argmin(axis=1)
        for center_idx in range(len(centers)):
            indices = np.argwhere(close_centers == center_idx).flatten()
            partials[center_idx][0] = np.sum(arr[indices], axis=0)
            partials[center_idx][1] = indices.shape[0]

        return partials

    @dclayMethod(num_points='int', dim='int', seed='int')
    def generate_points(self, num_points, dim, seed):
        np.random.seed(seed)
        mat = np.random.random((num_points, dim))

        # Normalize all points between 0 and 1
        mat -= np.min(mat)
        mx = np.max(mat)
        if mx > 0.0:
            mat /= mx

        self.points = mat
