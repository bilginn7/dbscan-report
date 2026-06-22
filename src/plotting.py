import pandas as pd
import matplotlib.pyplot as plt

from data_generation import generate_dataset_with_labels

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR.parent / "figures"
RESULTS_DIR = BASE_DIR.parent / "results"
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def plot_datasets() -> None:
    datasets = [
        "clean_blobs",
        "noisy_blobs",
        "variable_density_blobs",
    ]

    titles = [
        "Clean blobs",
        "Noisy blobs",
        "Variable-density blobs",
    ]

    for dataset_name, title in zip(datasets, titles):
        points, labels = generate_dataset_with_labels(
            dataset_name,
            n=2000,
            random_state=0
        )

        plt.figure(figsize=(5, 5))

        for label in sorted(set(labels)):
            label_points = points[labels == label]

            if label == -1:
                plt.scatter(
                    label_points[:, 0],
                    label_points[:, 1],
                    s=4,
                    label="noise"
                )
            else:
                plt.scatter(
                    label_points[:, 0],
                    label_points[:, 1],
                    s=4,
                    label=f"component {label}"
                )

        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True, linewidth=0.5, alpha=0.4)
        plt.legend(markerscale=3, fontsize=8)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{dataset_name}.pdf")
        plt.close()


def plot_runtime_vs_n():
    df = pd.read_csv("../results/results.csv")

    # Choose one dataset and one epsilon for a clean first plot
    df = df[
        (df["dataset"] == "clean_blobs") &
        (df["epsilon"] == 0.10)
    ]

    grouped = (
        df.groupby(["algorithm", "n"])["total_time"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(6, 4))

    for algorithm in grouped["algorithm"].unique():
        part = grouped[grouped["algorithm"] == algorithm]
        plt.plot(part["n"], part["total_time"], marker="o", label=algorithm)

    plt.xlabel("Number of points")
    plt.ylabel("Average total runtime (s)")
    plt.title("Runtime vs Number of Points")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "runtime_vs_n.pdf")
    plt.close()


def plot_runtime_vs_epsilon():

    df = pd.read_csv("../results/results.csv")

    df = df[
        (df["dataset"] == "clean_blobs")
        & (df["n"] == 5000)
    ]

    grouped = (
        df.groupby(
            ["algorithm", "epsilon"]
        )["total_time"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(6, 4))

    for algorithm in grouped["algorithm"].unique():

        part = grouped[
            grouped["algorithm"] == algorithm
        ]

        plt.plot(
            part["epsilon"],
            part["total_time"],
            marker="o",
            label=algorithm
        )

    plt.xlabel("Epsilon")
    plt.ylabel("Average total runtime (s)")
    plt.title("Runtime vs Epsilon")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "runtime_vs_epsilon.pdf")
    plt.close()


def plot_runtime_breakdown():

    df = pd.read_csv("../results/results.csv")

    df = df[
        (df["dataset"] == "clean_blobs")
        & (df["n"] == 5000)
        & (df["epsilon"] == 0.10)
    ]

    grouped = (
        df.groupby("algorithm")[
            [
                "init_time",
                "query_time",
                "clustering_time"
            ]
        ]
        .mean()
    )

    grouped.plot(
        kind="bar",
        stacked=True,
        figsize=(7, 5)
    )
    plt.xlabel("Algorithm")
    plt.ylabel("Time (s)")
    plt.title("Runtime Breakdown")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend()

    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "runtime_breakdown.pdf"
    )
    plt.close()


def plot_runtime_vs_n_per_dataset():

    df = pd.read_csv("../results/results.csv")

    datasets = df["dataset"].unique()

    for dataset in datasets:

        subset = df[
            (df["dataset"] == dataset)
            & (df["epsilon"] == 0.10)
        ]

        grouped = (
            subset.groupby(
                ["algorithm", "n"]
            )["total_time"]
            .mean()
            .reset_index()
        )

        plt.figure(figsize=(6, 4))

        for algorithm in grouped["algorithm"].unique():

            part = grouped[
                grouped["algorithm"] == algorithm
            ]

            plt.plot(
                part["n"],
                part["total_time"],
                marker="o",
                label=algorithm
            )

        plt.xlabel("Number of points")
        plt.ylabel("Average total runtime (s)")
        plt.yscale("log")
        plt.title(
            f"Runtime vs n ({dataset})"
        )

        plt.grid(True, alpha=0.4)
        plt.legend()

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / f"runtime_vs_n_{dataset}.pdf"
        )

        plt.close()




if __name__ == "__main__":

    #plot_datasets()

    plot_runtime_vs_n()

    plot_runtime_vs_epsilon()

    plot_runtime_breakdown()

    plot_runtime_vs_n_per_dataset()