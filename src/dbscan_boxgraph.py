import numpy as np
from collections import defaultdict


def build_grid(points, epsilon):
    """
    Create boxes of size epsilon / sqrt(2).
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
    Return neighbor indices of a point.
    """

    point = points[point_idx]

    cell_x = int(point[0] / cell_size)
    cell_y = int(point[1] / cell_size)

    neighbors = []

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):

            cell = (
                cell_x + dx,
                cell_y + dy
            )

            for idx in grid.get(cell, []):

                distance = np.linalg.norm(
                    points[idx] - point
                )

                if distance <= epsilon:
                    neighbors.append(idx)

    return neighbors


def run_boxgraph_dbscan(points, epsilon, min_pts):
    """
    Grid-based DBSCAN inspired by the BoxGraph approach.
    """

    n = len(points)

    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)

    grid, cell_size = build_grid(points, epsilon)

    cluster_id = 0

    for point_idx in range(n):

        if visited[point_idx]:
            continue

        visited[point_idx] = True

        neighbors = region_query(
            points,
            grid,
            cell_size,
            point_idx,
            epsilon
        )

        if len(neighbors) < min_pts:
            continue

        labels[point_idx] = cluster_id

        seed_set = list(neighbors)

        i = 0

        while i < len(seed_set):

            current = seed_set[i]

            if not visited[current]:
                visited[current] = True

                current_neighbors = region_query(
                    points,
                    grid,
                    cell_size,
                    current,
                    epsilon
                )

                if len(current_neighbors) >= min_pts:

                    for neighbor in current_neighbors:

                        if neighbor not in seed_set:
                            seed_set.append(neighbor)

            if labels[current] == -1:
                labels[current] = cluster_id

            i += 1

        cluster_id += 1

    return labels