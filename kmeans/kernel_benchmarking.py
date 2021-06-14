#!/usr/bin/env python3

from sklearn.metrics import pairwise_distances
import numpy as np


def partial_sum(arr, centers):
    partials = np.zeros((centers.shape[0], 2), dtype=object)
    close_centers = pairwise_distances(arr, centers).argmin(axis=1)
    for center_idx in range(len(centers)):
        indices = np.argwhere(close_centers == center_idx).flatten()
        partials[center_idx][0] = np.sum(arr[indices], axis=0)
        partials[center_idx][1] = indices.shape[0]
    
    return partials


def recompute_centers(partials):
    aggr = np.sum(partials, axis=0)

    centers = list()
    for sum_ in aggr:
        # centers with no elements are removed
        if sum_[1] != 0:
            centers.append(sum_[0] / sum_[1])
    return np.array(centers)
