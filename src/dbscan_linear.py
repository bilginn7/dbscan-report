import numpy as np

def region_query(points, idx, eps):

    dists = np.linalg.norm(
        points - points[idx],
        axis=1
    )

    return np.where(dists <= eps)[0]

from sklearn.cluster import DBSCAN

def run_linear_dbscan(points, eps, min_pts):

    model = DBSCAN(
        eps=eps,
        min_samples=min_pts,
        algorithm="brute"
    )

    return model.fit_predict(points)