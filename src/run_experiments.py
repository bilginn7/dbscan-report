import csv
import time
from pathlib import Path

from config import DATASETS, ALGORITHMS, N_VALUES, EPSILON_VALUES, MIN_PTS, REPEATS, NOISE_RATIO
from data_generation import generate_dataset


RESULTS_PATH = Path("../results/results.csv")


def run_dbscan(points, epsilon, min_pts, algorithm):
    """
    Temporary placeholder.

    Your teammate should replace this function with calls to the real
    DBSCAN implementations:
    - linear
    - quadtree
    - boxgraph
    """
    #raise NotImplementedError(f"Algorithm not connected yet: {algorithm}")
    return [-1] * len(points)

def count_clusters_and_noise(labels):
    """
    Count clusters and noise points from DBSCAN labels.
    Assumes noise is labeled as -1.
    """
    unique_labels = set(labels)

    num_noise = sum(1 for label in labels if label == -1)
    num_clusters = len(unique_labels - {-1})

    return num_clusters, num_noise


def write_header_if_needed():
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not RESULTS_PATH.exists():
        with open(RESULTS_PATH, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "algorithm",
                "dataset",
                "n",
                "epsilon",
                "min_pts",
                "repeat",
                "total_time",
                "num_clusters",
                "num_noise",
            ])


def append_result(row):
    with open(RESULTS_PATH, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(row)


def run_experiments():
    write_header_if_needed()

    for dataset_name in DATASETS:
        for n in N_VALUES:
            for epsilon in EPSILON_VALUES:
                for repeat in range(REPEATS):
                    points = generate_dataset(
                        dataset_name=dataset_name,
                        n=n,
                        random_state=repeat,
                        noise_ratio=NOISE_RATIO,
                    )

                    for algorithm in ALGORITHMS:
                        print(
                            f"Running {algorithm} | "
                            f"dataset={dataset_name} | "
                            f"n={n} | "
                            f"epsilon={epsilon} | "
                            f"repeat={repeat}"
                        )

                        start = time.perf_counter()
                        labels = run_dbscan(points, epsilon, MIN_PTS, algorithm)
                        end = time.perf_counter()

                        total_time = end - start
                        num_clusters, num_noise = count_clusters_and_noise(labels)

                        append_result([
                            algorithm,
                            dataset_name,
                            n,
                            epsilon,
                            MIN_PTS,
                            repeat,
                            total_time,
                            num_clusters,
                            num_noise,
                        ])


if __name__ == "__main__":
    run_experiments()