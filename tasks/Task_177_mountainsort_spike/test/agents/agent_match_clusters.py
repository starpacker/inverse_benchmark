import numpy as np

import matplotlib

matplotlib.use("Agg")

from scipy.optimize import linear_sum_assignment

def match_clusters(gt_labels, pred_labels):
    """Hungarian matching of predicted clusters to GT units."""
    gt_units = np.unique(gt_labels)
    pred_units = np.unique(pred_labels)
    cost = np.zeros((len(gt_units), len(pred_units)))
    for i, gu in enumerate(gt_units):
        for j, pu in enumerate(pred_units):
            cost[i, j] = -np.sum((gt_labels == gu) & (pred_labels == pu))
    row_ind, col_ind = linear_sum_assignment(cost)
    return {pred_units[c]: gt_units[r] for r, c in zip(row_ind, col_ind)}
