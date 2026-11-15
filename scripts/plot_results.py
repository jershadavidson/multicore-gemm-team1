
import csv
import os
import statistics
from collections import defaultdict

import matplotlib.pyplot as plt

CSV_FILE = "results/raw/gemm_results.csv"
PLOT_DIR = "results/plots"

os.makedirs(PLOT_DIR, exist_ok=True)

# Read and group benchmark results
groups = defaultdict(list)

with open(CSV_FILE, newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        implementation = row["implementation"].strip().lower()
        size = int(row["size"])
        threads = row["threads_or_ranks"].strip()

        key = (implementation, size, threads)

        groups[key].append({
            "time": float(row["total_ms"]),
            "gflops": float(row["throughput"])
        })


def get_stats(implementation, size, threads="default"):
    key = (implementation.lower(), size, str(threads))
    runs = groups.get(key, [])

    if not runs:
        return None

    times = [r["time"] for r in runs]
    gflops = [r["gflops"] for r in runs]

    return (
        statistics.mean(times),
        statistics.stdev(times) if len(times) > 1 else 0,
        statistics.mean(gflops),
        statistics.stdev(gflops) if len(gflops) > 1 else 0
    )


# --------------------------------------------------
# 1. Blocked GEMM GFLOP/s heatmap
# --------------------------------------------------

sizes = [512, 1024, 2048]
tiles = [16, 32, 64, 128]

heatmap = []

for tile in tiles:
    values = []

    for size in sizes:
        stats = get_stats(f"blocked_t{tile}", size, "1")
        values.append(stats[2] if stats else float("nan"))

    heatmap.append(values)

plt.figure(figsize=(8, 5))

image = plt.imshow(heatmap, aspect="auto")
plt.colorbar(image, label="GFLOP/s")

plt.xticks(range(len(sizes)), sizes)
plt.yticks(range(len(tiles)), tiles)

plt.xlabel("Matrix size (N)")
plt.ylabel("Tile size (T)")
plt.title("Blocked GEMM: Performance Heatmap")

for i in range(len(tiles)):
    for j in range(len(sizes)):
        value = heatmap[i][j]

        if value == value:
            plt.text(
                j, i, f"{value:.2f}",
                ha="center", va="center"
            )

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/blocked_gflops_heatmap.png", dpi=300)
plt.close()


# --------------------------------------------------
# 2. Optimisation comparison at N=512
# --------------------------------------------------

implementations = [
    ("naive_ijk", "Naive i-j-k", "1"),
    ("reordered_ikj", "Reordered i-k-j", "1"),
    ("blocked_t128", "Blocked T=128", "1"),
    ("openmp_ikj", "OpenMP (10 threads)", "10"),
    ("openblas_sgemm", "OpenBLAS", "default")
]

labels = []
values = []
errors = []

for implementation, label, threads in implementations:
    stats = get_stats(implementation, 512, threads)

    if stats:
        labels.append(label)
        values.append(stats[2])
        errors.append(stats[3])

plt.figure(figsize=(10, 5))

plt.bar(labels, values, yerr=errors, capsize=4)

plt.ylabel("GFLOP/s")
plt.xlabel("Implementation")
plt.title("GEMM Optimisation Comparison (N=512)")
plt.xticks(rotation=20, ha="right")

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/optimisation_comparison.png", dpi=300)
plt.close()


# --------------------------------------------------
# 3. Execution time vs matrix size
# --------------------------------------------------

time_series = [
    ("reordered_ikj", "Reordered i-k-j", "1"),
    ("blocked_t128", "Blocked T=128", "1"),
    ("openmp_ikj", "OpenMP (10 threads)", "10"),
    ("openblas_sgemm", "OpenBLAS", "default")
]

plt.figure(figsize=(9, 5))

for implementation, label, threads in time_series:
    x_values = []
    means = []
    errors = []

    for size in [512, 1024, 2048, 4096, 8192]:
        stats = get_stats(implementation, size, threads)

        if stats:
            x_values.append(size)
            means.append(stats[0])
            errors.append(stats[1])

    if x_values:
        plt.errorbar(
            x_values,
            means,
            yerr=errors,
            marker="o",
            capsize=3,
            label=label
        )

plt.xlabel("Matrix size (N)")
plt.ylabel("Execution time (ms)")
plt.title("GEMM Execution Time vs Matrix Size")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/execution_time_vs_size.png", dpi=300)
plt.close()


# --------------------------------------------------
# 4. OpenMP speedup vs thread count
# --------------------------------------------------

sequential = get_stats("reordered_ikj", 512, "1")

if sequential:
    baseline_time = sequential[0]

    thread_counts = [1, 2, 4, 8, 10]
    valid_threads = []
    speedups = []
    efficiencies = []

    for threads in thread_counts:
        stats = get_stats("openmp_ikj", 512, str(threads))
        
        if stats:
            speedup = baseline_time / stats[0]
            efficiency = speedup / threads

            valid_threads.append(threads)
            speedups.append(speedup)
            efficiencies.append(efficiency * 100)

    plt.figure(figsize=(8, 5))

    plt.plot(
        valid_threads,
        speedups,
        marker="o",
        label="Measured speedup"
    )

    plt.plot(
        valid_threads,
        valid_threads,
        linestyle="--",
        label="Ideal speedup"
    )

    plt.xlabel("Number of threads")
    plt.ylabel("Speedup")
    plt.title("OpenMP Speedup (N=512)")
    plt.xticks(valid_threads)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/openmp_speedup.png", dpi=300)
    plt.close()

    # Parallel efficiency
    plt.figure(figsize=(8, 5))

    plt.plot(
        valid_threads,
        efficiencies,
        marker="o"
    )

    plt.xlabel("Number of threads")
    plt.ylabel("Parallel efficiency (%)")
    plt.title("OpenMP Parallel Efficiency (N=512)")
    plt.xticks(valid_threads)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/openmp_efficiency.png", dpi=300)
    plt.close()


print("Plot generation completed.")

print("\nGenerated plots:")
for filename in [
    "blocked_gflops_heatmap.png",
    "optimisation_comparison.png",
    "execution_time_vs_size.png",
    "openmp_speedup.png",
    "openmp_efficiency.png"
]:
    print(f"{PLOT_DIR}/{filename}")

