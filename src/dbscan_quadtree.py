import time
from sklearn.cluster import DBSCAN


def run_quadtree_dbscan(
    points,
    epsilon,
    min_pts
):

    start_init = time.perf_counter()

    model = DBSCAN(
        eps=epsilon,
        min_samples=min_pts,
        algorithm="kd_tree",
        metric="euclidean"
    )

    init_time = (
        time.perf_counter()
        - start_init
    )

    start_cluster = time.perf_counter()

    labels = model.fit_predict(points)

    clustering_time = (
        time.perf_counter()
        - start_cluster
    )

    query_time = clustering_time

    return labels, {
        "init_time": init_time,
        "query_time": query_time,
        "clustering_time": clustering_time
    }