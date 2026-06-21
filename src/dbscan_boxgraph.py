import time
import numpy as np
from collections import defaultdict


def build_grid(points, epsilon):
    """
    Create grid cells with side length epsilon / sqrt(2).
    """

    cell_size = epsilon / np.sqrt(2)

    grid = defaultdict(list)

    for idx, point in enumerate(points):
        cell = (
            int(point[0] / cell_size),
            int(point[1] / cell_size)
        )
        grid[cell].append(idx)

    return grid, cell_size


def region_query(points, grid, cell_size, point_idx, epsilon):
    """
    Return all neighbors within epsilon distance of point_idx.
    """

    point = points[point_idx]

    cell_x = int(point[0] / cell_size)
    cell_y = int(point[1] / cell_size)

    neighbors = []

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):

            neighbor_cell = (
                cell_x + dx,
                cell_y + dy
            )

            for idx in grid.get(neighbor_cell, []):

                distance = np.linalg.norm(
                    points[idx] - point
                )

                if distance <= epsilon:
                    neighbors.append(idx)

    return neighbors


def run_boxgraph_dbscan(points, epsilon, min_pts):
    """
    Grid-based DBSCAN inspired by the BoxGraph algorithm.

    Returns
    -------
    labels : numpy.ndarray
        Cluster labels (-1 indicates noise)

    stats : dict
        Runtime statistics
    """

    # ----------------------------
    # Initialization
    # ----------------------------

    init_start = time.perf_counter()

    grid, cell_size = build_grid(
        points,
        epsilon
    )

    init_time = (
        time.perf_counter() - init_start
    )

    # ----------------------------
    # DBSCAN
    # ----------------------------

    n = len(points)

    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)

    cluster_id = 0

    query_time = 0.0

    clustering_start = time.perf_counter()

    for point_idx in range(n):

        if visited[point_idx]:
            continue

        visited[point_idx] = True

        query_start = time.perf_counter()

        neighbors = region_query(
            points,
            grid,
            cell_size,
            point_idx,
            epsilon
        )

        query_time += (
            time.perf_counter() - query_start
        )

        if len(neighbors) < min_pts:
            continue

        labels[point_idx] = cluster_id

        seed_set = list(neighbors)
        seed_members = set(neighbors)

        i = 0

        while i < len(seed_set):

            current = seed_set[i]

            if not visited[current]:

                visited[current] = True

                query_start = time.perf_counter()

                current_neighbors = region_query(
                    points,
                    grid,
                    cell_size,
                    current,
                    epsilon
                )

                query_time += (
                    time.perf_counter() - query_start
                )

                if len(current_neighbors) >= min_pts:

                    for neighbor in current_neighbors:

                        if neighbor not in seed_members:
                            seed_set.append(neighbor)
                            seed_members.add(neighbor)

            if labels[current] == -1:
                labels[current] = cluster_id

            i += 1

        cluster_id += 1

    total_clustering_time = (
        time.perf_counter() - clustering_start
    )

    clustering_time = (
        total_clustering_time - query_time
    )

    stats = {
        "init_time": init_time,
        "query_time": query_time,
        "clustering_time": clustering_time
    }

    return labels, stats