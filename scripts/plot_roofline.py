
import csv
import statistics
import os
import matplotlib.pyplot as plt

CSV_FILE = "results/raw/gemm_results.csv"
PLOT_DIR = "results/plots"

# Measured STREAM Triad bandwidth in GB/s
BANDWIDTH_GB_S = 119.27

# Measured OpenBLAS compute ceiling in GFLOP/s
COMPUTE_CEILING_GFLOPS = 760.76

MATRIX_SIZE = 512

# Arithmetic intensity estimate for FP32 GEMM:
# 2*N^3 operations / (12*N^2 bytes) = N/6 FLOP per byte
arithmetic_intensity = MATRIX_SIZE / 6

# Read benchmark data
groups = {}

with open(CSV_FILE, newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        if int(row["size"]) != MATRIX_SIZE:
            continue

        implementation = row["implementation"].lower()
        threads = row["threads_or_ranks"]

        key = (implementation, threads)

        groups.setdefault(key, []).append(float(row["total_ms"]))


def get_gflops(implementation, threads):
    times = groups.get((implementation, str(threads)))

    if not times:
        return None, None

    # Calculate GFLOP/s from each measured execution time
    results = [
        (2 * MATRIX_SIZE**3) / (time_ms * 1_000_000)
        for time_ms in times
    ]

    mean = statistics.mean(results)
    sd = statistics.stdev(results) if len(results) > 1 else 0

    return mean, sd


# Roofline model
intensities = [
    10 ** (i / 20)
    for i in range(-20, 61)
]

bandwidth_roof = [
    BANDWIDTH_GB_S * intensity
    for intensity in intensities
]

roofline = [
    min(value, COMPUTE_CEILING_GFLOPS)
    for value in bandwidth_roof
]

plt.figure(figsize=(10, 7))

plt.loglog(
    intensities,
    bandwidth_roof,
    linestyle="--",
    label="STREAM bandwidth roof"
)

plt.loglog(
    intensities,
    [COMPUTE_CEILING_GFLOPS] * len(intensities),
    linestyle="--",
    label="Measured OpenBLAS compute ceiling"
)

plt.loglog(
    intensities,
    roofline,
    linewidth=2,
    label="Roofline performance limit"
)

# Plot measured GEMM implementations at N=512
implementations = [
    ("naive_ijk", "1", "Naive i-j-k"),
    ("reordered_ikj", "1", "Reordered i-k-j"),
    ("blocked_t128", "1", "Blocked T=128"),
    ("openmp_ikj", "10", "OpenMP 10 threads"),
    ("openblas_sgemm", "default", "OpenBLAS")
]

for implementation, threads, label in implementations:
    mean, sd = get_gflops(implementation, threads)

    if mean is not None:
        plt.errorbar(
            arithmetic_intensity,
            mean,
            yerr=sd,
            marker="o",
            capsize=4,
            linestyle="none",
            label=f"{label}: {mean:.2f} GFLOP/s"
        )

plt.xlabel("Arithmetic intensity (FLOP/byte)")
plt.ylabel("Performance (GFLOP/s)")
plt.title("Roofline Model - GEMM on Apple M5 (N=512)")

plt.grid(True, which="both", linestyle=":")
plt.legend(fontsize=8)
plt.tight_layout()

os.makedirs(PLOT_DIR, exist_ok=True)

plt.savefig(
    f"{PLOT_DIR}/roofline.png",
    dpi=300
)

plt.close()

print("Roofline plot generated successfully.")
print(f"Arithmetic intensity estimate: {arithmetic_intensity:.2f} FLOP/byte")
print(f"STREAM bandwidth: {BANDWIDTH_GB_S} GB/s")
print(f"Measured compute ceiling: {COMPUTE_CEILING_GFLOPS} GFLOP/s")
