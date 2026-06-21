import csv
from pathlib import Path

from config import (DATASETS, ALGORITHMS, N_VALUES, EPSILON_VALUES, MIN_PTS, REPEATS, NOISE_RATIO)
from data_generation import generate_dataset
from timing import measure_total_time


RESULTS_PATH = Path("../results/results.csv")


def run_dbscan(points, epsilon, min_pts, algorithm):
    """
    Temporary placeholder.

    Member 2 must replace this with the real DBSCAN calls.
    For now, this dummy version labels every point as noise.
    """
    return [-1] * len(points)


def count_clusters_and_noise(labels):
    """
    Count the number of clusters and noise points.

    DBSCAN convention:
    - noise points have label -1
    - cluster labels are usually 0, 1, 2, ...
    """
    unique_labels = set(labels)

    num_clusters = len(unique_labels - {-1})
    num_noise = sum(1 for label in labels if label == -1)

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

                        labels, total_time = measure_total_time(
                            run_dbscan,
                            points,
                            epsilon,
                            MIN_PTS,
                            algorithm,
                        )

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