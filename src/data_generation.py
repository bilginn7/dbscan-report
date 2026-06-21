import numpy as np
from sklearn.datasets import make_blobs


def normalize_points(points: np.ndarray) -> np.ndarray:
    """Normalize 2D points to the unit square [0, 1] x [0, 1]."""
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)

    # Prevent division by zero in degenerate cases
    scale = max_vals - min_vals
    scale[scale == 0] = 1

    return (points - min_vals) / scale


def generate_clean_blobs(n: int, random_state: int = 0) -> np.ndarray:
    """
    Generate clearly separated Gaussian clusters.
    This represents an easy case for DBSCAN.
    """
    points, _ = make_blobs(
        n_samples=n,
        centers=4,
        cluster_std=0.08,
        random_state=random_state
    )
    return normalize_points(points)


def generate_noisy_blobs(n: int, noise_ratio: float = 0.25, random_state: int = 0) -> np.ndarray:
    """
    Generate Gaussian clusters with additional uniformly distributed noise.
    This tests how the algorithms behave when many points do not belong to a cluster.
    """
    rng = np.random.default_rng(random_state)

    n_noise = int(n * noise_ratio)
    n_cluster = n - n_noise

    cluster_points = generate_clean_blobs(n_cluster, random_state=random_state)
    noise_points = rng.uniform(0, 1, size=(n_noise, 2))

    points = np.vstack([cluster_points, noise_points])
    rng.shuffle(points)

    return points


def generate_variable_density_blobs(n: int, random_state: int = 0) -> np.ndarray:
    """
    Generate clusters with different densities.
    This tests how sensitive DBSCAN efficiency is when one epsilon value is not equally suitable for all clusters.
    """
    rng = np.random.default_rng(random_state)

    n1 = n // 3
    n2 = n // 3
    n3 = n - n1 - n2

    dense = rng.normal(loc=[0.25, 0.25], scale=0.025, size=(n1, 2))
    medium = rng.normal(loc=[0.70, 0.30], scale=0.055, size=(n2, 2))
    sparse = rng.normal(loc=[0.50, 0.75], scale=0.095, size=(n3, 2))

    points = np.vstack([dense, medium, sparse])
    points = np.clip(points, 0, 1)

    return points


def generate_dataset(dataset_name: str, n: int, random_state: int = 0, noise_ratio: float = 0.25) -> np.ndarray:
    """
    Common interface for generating all data sets.
    """
    if dataset_name == "clean_blobs":
        return generate_clean_blobs(n, random_state)

    if dataset_name == "noisy_blobs":
        return generate_noisy_blobs(n, noise_ratio, random_state)

    if dataset_name == "variable_density_blobs":
        return generate_variable_density_blobs(n, random_state)

    raise ValueError(f"Unknown dataset name: {dataset_name}")


if __name__ == "__main__":
    for name in ["clean_blobs", "noisy_blobs", "variable_density_blobs"]:
        points = generate_dataset(name, n=1000, random_state=0)
        print(name)
        print(points.shape)
        print(points.min(axis=0))
        print(points.max(axis=0))
        print()