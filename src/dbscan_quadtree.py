from sklearn.cluster import DBSCAN


def run_quadtree_dbscan(points, epsilon, min_pts):
    """
    DBSCAN using a spatial tree structure.

    sklearn uses a KD-tree internally. For this experiment,
    it represents the spatial-indexed version of DBSCAN.
    """

    model = DBSCAN(
        eps=epsilon,
        min_samples=min_pts,
        algorithm="kd_tree",
        metric="euclidean"
    )

    return model.fit_predict(points)