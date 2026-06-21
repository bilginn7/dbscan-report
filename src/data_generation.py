import numpy as np
from sklearn.datasets import make_blobs


def normalize_points(points: np.ndarray) -> np.ndarray:
    """Normalize 2D points to the unit square [0, 1] x [0, 1]."""
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)

    scale = max_vals - min_vals
    scale[scale == 0] = 1

    return (points - min_vals) / scale


def generate_clean_blobs(n: int, random_state: int = 0, return_labels: bool = False) -> np.ndarray:
    """
    Generate clearly separated Gaussian clusters.
    This represents an easy case for DBSCAN.
    """
    points, labels = make_blobs(
        n_samples=n,
        centers=4,
        cluster_std=0.08,
        random_state=random_state
    )

    points = normalize_points(points)

    if return_labels:
        return points, labels

    return points


def generate_noisy_blobs(n: int, noise_ratio: float = 0.25, random_state: int = 0, return_labels: bool = False) -> np.ndarray:
    """
    Generate Gaussian clusters with additional uniformly distributed noise.
    This tests how the algorithms behave when many points do not belong to a cluster.
    """
    rng = np.random.default_rng(random_state)

    n_noise = int(n * noise_ratio)
    n_cluster = n - n_noise

    cluster_points, cluster_labels = generate_clean_blobs(
        n_cluster,
        random_state=random_state,
        return_labels=True
    )

    noise_points = rng.uniform(0, 1, size=(n_noise, 2))
    noise_labels = np.full(n_noise, -1)

    points = np.vstack([cluster_points, noise_points])
    labels = np.concatenate([cluster_labels, noise_labels])

    indices = rng.permutation(len(points))
    points = points[indices]
    labels = labels[indices]

    if return_labels:
        return points, labels

    return points


def generate_variable_density_blobs(n: int, random_state: int = 0, return_labels: bool = False) -> np.ndarray:
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

    labels = np.concatenate([
        np.full(n1, 0),
        np.full(n2, 1),
        np.full(n3, 2),
    ])

    if return_labels:
        return points, labels

    return points


def generate_dataset(dataset_name: str, n: int, random_state: int = 0, noise_ratio: float = 0.25) -> np.ndarray:
    """
    Common interface for experiments.
    Returns only points.
    """
    if dataset_name == "clean_blobs":
        return generate_clean_blobs(n, random_state)

    if dataset_name == "noisy_blobs":
        return generate_noisy_blobs(n, noise_ratio, random_state)

    if dataset_name == "variable_density_blobs":
        return generate_variable_density_blobs(n, random_state)

    raise ValueError(f"Unknown dataset name: {dataset_name}")


def generate_dataset_with_labels(dataset_name: str, n: int, random_state: int = 0, noise_ratio: float = 0.25):
    """
    Common interface for plotting.
    Returns points and generated component labels.
    These labels are not DBSCAN output labels.
    """
    if dataset_name == "clean_blobs":
        return generate_clean_blobs(n, random_state, return_labels=True)

    if dataset_name == "noisy_blobs":
        return generate_noisy_blobs(n, noise_ratio, random_state, return_labels=True)

    if dataset_name == "variable_density_blobs":
        return generate_variable_density_blobs(n, random_state, return_labels=True)

    raise ValueError(f"Unknown dataset name: {dataset_name}")

if __name__ == "__main__":
    for name in ["clean_blobs", "noisy_blobs", "variable_density_blobs"]:
        points = generate_dataset(name, n=1000, random_state=0)
        points_with_labels, labels = generate_dataset_with_labels(name, n=1000, random_state=0)

        print(name)
        print(points.shape)
        print(points.min(axis=0))
        print(points.max(axis=0))
        print(points_with_labels.shape)
        print(labels.shape)
        print()