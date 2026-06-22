# DBSCAN Performance Report

A comprehensive benchmarking suite for comparing different DBSCAN clustering algorithm implementations on various synthetic datasets with different data characteristics.

## Overview

This project evaluates the performance of three DBSCAN implementations:
- **Linear**: Basic O(n²) pairwise distance computation
- **Quadtree**: Spatial index-based approach using quadtree partitioning
- **Boxgraph**: Graph-based spatial locality approach

The implementations are tested across multiple synthetic datasets with varying characteristics and parameters to measure clustering time, initialization overhead, and solution quality.

## Project Structure

```
.
├── src/
│   ├── config.py                 # Experiment configuration
│   ├── data_generation.py        # Dataset generation utilities
│   ├── run_experiments.py        # Main experiment runner
│   ├── dbscan_linear.py          # Linear DBSCAN implementation
│   ├── dbscan_quadtree.py        # Quadtree DBSCAN implementation
│   ├── dbscan_boxgraph.py        # Boxgraph DBSCAN implementation
│   ├── plotting.py               # Visualization utilities
│   └── timing.py                 # Performance timing utilities
├── results/                       # Experiment output (CSV results)
├── figures/                       # Generated visualizations
└── README.md
```

## Features

### Datasets

Three distinct dataset types are used to evaluate algorithm performance:

1. **Clean Blobs** - Well-separated Gaussian clusters (easy case)
   - 4 clearly distinct clusters with minimal variance
   - Represents ideal clustering scenarios

2. **Noisy Blobs** - Gaussian clusters with added noise (moderate case)
   - 4 Gaussian clusters + 25% uniformly random noise points
   - Tests robustness to outliers and noise

3. **Variable Density Blobs** (hard case) - Clusters with different densities
   - Three clusters with different point densities (dense, medium, sparse)
   - Tests sensitivity to epsilon parameter across varying densities

### Experiment Parameters

- **Dataset Sizes (n)**: 1,000 | 2,500 | 5,000 points
- **Epsilon Values**: 0.03, 0.05, 0.075, 0.10, 0.15
- **Min Points**: 5
- **Repeats**: 3 per configuration
- **Points normalized to [0, 1] × [0, 1] unit square**

## Installation

```bash
# Clone the repository
git clone https://github.com/bilginn7/dbscan-report.git
cd dbscan-report

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- Python 3.8+
- NumPy
- scikit-learn
- SciPy
- Matplotlib (for visualization)

## Usage

### Run Experiments

```bash
cd src
python run_experiments.py
```

This will:
1. Generate all dataset combinations
2. Run each algorithm on each configuration
3. Record timing metrics (initialization, query, clustering, total)
4. Count resulting clusters and noise points
5. Save results to `results/results.csv`

### Generate Visualizations

```bash
cd src
python plotting.py
```

Generates visualizations in the `figures/` directory comparing algorithms across different datasets and parameters.

## Results Output

The experiment runner produces a CSV file (`results/results.csv`) with the following columns:

| Column | Description |
|--------|-------------|
| algorithm | Implementation name (linear, quadtree, boxgraph) |
| dataset | Dataset type used |
| n | Number of points |
| epsilon | Epsilon parameter (neighborhood radius) |
| min_pts | Minimum points for core objects |
| repeat | Run number (1-3) |
| init_time | Initialization time (ms) |
| query_time | Query/neighborhood search time (ms) |
| clustering_time | Cluster formation time (ms) |
| total_time | Total execution time (ms) |
| num_clusters | Number of clusters found |
| num_noise | Number of noise points (-1 labeled) |

## Key Metrics

- **Initialization Time**: Time to build spatial data structures
- **Query Time**: Time to find epsilon-neighborhoods for points
- **Clustering Time**: Time to form clusters from neighborhood information
- **Total Time**: Sum of all timing components
- **Clustering Quality**: Number of clusters and noise points detected

## Configuration

Modify `src/config.py` to customize experiment parameters:

```python
DATASETS = ["clean_blobs", "noisy_blobs", "variable_density_blobs"]
ALGORITHMS = ["linear", "quadtree", "boxgraph"]
N_VALUES = [1000, 2500, 5000]
EPSILON_VALUES = [0.03, 0.05, 0.075, 0.10, 0.15]
MIN_PTS = 5
REPEATS = 3
NOISE_RATIO = 0.25
```

## Authors

- bilginn7
- JanG1411
