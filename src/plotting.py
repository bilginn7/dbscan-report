import pandas as pd
import matplotlib.pyplot as plt

from data_generation import generate_dataset


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
        points = generate_dataset(dataset_name, n=2000, random_state=0)

        plt.figure(figsize=(5, 5))
        plt.scatter(points[:, 0], points[:, 1], s=4)

        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")

        plt.grid(True, linewidth=0.5, alpha=0.4)

        plt.tight_layout()
        plt.savefig(f"../figures/{dataset_name}.pdf")
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
    plt.ylabel("Total running time (seconds)")
    plt.title("Runtime vs number of points")
    plt.grid(True, linewidth=0.5, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("../figures/runtime_vs_n.pdf")
    plt.close()



if __name__ == "__main__":
    #plot_datasets()
    plot_runtime_vs_n()