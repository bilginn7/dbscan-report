from sklearn.cluster import DBSCAN


def run_linear_dbscan(points, epsilon, min_pts):
    """
    DBSCAN using brute-force range queries.
    """

    model = DBSCAN(
        eps=epsilon,
        min_samples=min_pts,
        algorithm="brute",
        metric="euclidean"
    )

    return model.fit_predict(points)
