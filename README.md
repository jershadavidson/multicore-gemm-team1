# Memory-Hierarchy-Aware GEMM on a Multicore CPU

## 1. Project Overview

This project implements and evaluates different approaches to General Matrix-Matrix Multiplication (GEMM) on a multicore CPU.

The operation performed is:

\[
C = A \times B
\]

where A, B and C are square matrices of size \(N \times N\).

The project investigates the effect of memory access patterns, cache blocking, compiler vectorisation and multicore parallelism on matrix multiplication performance.

### Implementations

| Implementation | Description |
|---|---|
| Naive GEMM | Traditional `i-j-k` triple-loop implementation |
| Reordered GEMM | `i-k-j` loop ordering for improved cache locality |
| Blocked GEMM | Cache blocking using tile sizes 16, 32, 64 and 128 |
| Vectorised GEMM | Compiler-assisted vectorisation |
| OpenMP GEMM | Parallel matrix multiplication using OpenMP |
| OpenBLAS GEMM | Optimised BLAS implementation used for comparison |
| Double GEMM | Double-precision matrix multiplication |

## 2. Hardware and Software Environment

### Hardware

| Component | Specification |
|---|---|
| System | MacBook Air |
| Processor | Apple M5 |
| CPU cores | 10 (4 performance + 6 efficiency) |
| Memory | 16 GB unified memory |
| Architecture | ARM64 (Apple Silicon) |
| Operating System | macOS |

### Software

| Software | Version |
|---|---|
| GCC | 16.2.0 |
| OpenBLAS | 0.3.34 |
| OpenMP | Enabled through GCC |
| Python | Python 3 |
| Matplotlib | Used for plotting |
| Git | Version control |

Cache profiling with `perf stat` was performed separately on a Linux system. These profiling results are not MacBook benchmark measurements.

## 3. Dependencies and Installation

The project requires:

- GCC compiler
- OpenBLAS library
- OpenMP support
- Python 3
- Matplotlib
- Standard C library and math library

### Install dependencies on macOS

Install GCC and OpenBLAS using Homebrew:

```bash
brew install gcc openblas
```

Verify the installations:

```bash
gcc-16 --version
brew list openblas
```

Configure OpenBLAS paths:

```bash
export LDFLAGS="-L/opt/homebrew/opt/openblas/lib"
export CPPFLAGS="-I/opt/homebrew/opt/openblas/include"
```

Install Python plotting dependency:

```bash
python3 -m pip install matplotlib
```

## 4. Repository Structure

```text
multicore-gemm-team1-main/
│
├── benchmarks/
│   └── stream/
│       └── stream.c
│
├── src/
│   ├── gemm_naive.c
│   ├── gemm_reordered.c
│   ├── gemm_blocked.c
│   ├── gemm_vectorized.c
│   ├── gemm_parallel.c
│   ├── gemm_blas.c
│   └── gemm_double.c
│
├── scripts/
│   ├── plot_results.py
│   └── plot_roofline.py
│
├── results/
│   ├── raw/
│   │   ├── gemm_results.csv
│   │   └── cache_miss_results.md
│   │
│   └── plots/
│       ├── blocked_gflops_heatmap.png
│       ├── execution_time_vs_size.png
│       ├── openmp_efficiency.png
│       ├── openmp_speedup.png
│       ├── optimisation_comparison.png
│       └── roofline.png
│
├── vectorization_report.txt
├── LOG.md
├── .gitignore
└── README.md
```

## 5. Compilation Instructions

Run all commands from the repository root directory.

The following commands use Homebrew GCC and OpenBLAS paths for Apple Silicon.

### 5.1 Naive GEMM

```bash
gcc-16 -O3 -march=native \
src/gemm_naive.c \
-o gemm_naive \
-L/opt/homebrew/opt/openblas/lib \
-I/opt/homebrew/opt/openblas/include \
-lopenblas -lm
```

### 5.2 Reordered GEMM

```bash
gcc-16 -O3 -march=native \
src/gemm_reordered.c \
-o gemm_reordered \
-L/opt/homebrew/opt/openblas/lib \
-I/opt/homebrew/opt/openblas/include \
-lopenblas -lm
```

### 5.3 Blocked GEMM

```bash
gcc-16 -O3 -march=native \
src/gemm_blocked.c \
-o gemm_blocked \
-L/opt/homebrew/opt/openblas/lib \
-I/opt/homebrew/opt/openblas/include \
-lopenblas -lm
```

### 5.4 Vectorised GEMM

```bash
gcc-16 -O3 -march=native \
src/gemm_vectorized.c \
-o gemm_vectorized \
-L/opt/homebrew/opt/openblas/lib \
-I/opt/homebrew/opt/openblas/include \
-lopenblas -lm
```

### 5.5 OpenMP GEMM

```bash
gcc-16 -O3 -march=native -fopenmp \
src/gemm_parallel.c \
-o gemm_parallel \
-L/opt/homebrew/opt/openblas/lib \
-I/opt/homebrew/opt/openblas/include \
-lopenblas -lm
```

### 5.6 OpenBLAS GEMM

```bash
gcc-16 -O3 -march=native \
src/gemm_blas.c \
-o gemm_blas \
-L/opt/homebrew/opt/openblas/lib \
-I/opt/homebrew/opt/openblas/include \
-lopenblas -lm
```

### 5.7 Double-Precision GEMM

```bash
gcc-16 -O3 -march=native \
src/gemm_double.c \
-o gemm_double \
-L/opt/homebrew/opt/openblas/lib \
-I/opt/homebrew/opt/openblas/include \
-lopenblas -lm
```

## 6. Running the Implementations

The project uses the following matrix sizes:

```text
512, 1024, 2048, 4096, 8192
```

### Naive GEMM

```bash
./gemm_naive 512
```

### Reordered GEMM

```bash
./gemm_reordered 512
./gemm_reordered 1024
./gemm_reordered 2048
```

### Blocked GEMM

The first argument specifies matrix size and the second specifies tile size.

```bash
./gemm_blocked 512 16
./gemm_blocked 512 32
./gemm_blocked 512 64
./gemm_blocked 512 128
```

### Vectorised GEMM

```bash
./gemm_vectorized 512
```

### OpenMP GEMM

The OpenMP implementation accepts the matrix size and thread count as arguments.

```bash
./gemm_parallel 512 1
./gemm_parallel 512 2
./gemm_parallel 512 4
./gemm_parallel 512 8
./gemm_parallel 512 10
```

### OpenBLAS GEMM

```bash
./gemm_blas 512
./gemm_blas 1024
./gemm_blas 2048
./gemm_blas 4096
./gemm_blas 8192
```

### Double-Precision GEMM

```bash
./gemm_double
```

The double-precision implementation uses its configured matrix size in the source code.

## 7. Correctness Verification

The custom implementations are verified against OpenBLAS using the `cblas_sgemm()` reference implementation for single precision.

The double-precision implementation uses `cblas_dgemm()`.

For single-precision calculations, the project requires:

\[
\text{Maximum Relative Error} < 10^{-4}
\]

The same input matrices must be used for the custom implementation and the BLAS reference. A fixed random seed is used for reproducibility.

A successful verification should report:

```text
Maximum relative error: <measured value>
Verification PASSED
```

The error must be calculated from the actual output matrices.

## 8. Performance Metrics

### Execution Time

Execution time is measured for each implementation.

### GFLOP/s

Performance is calculated using:

\[
GFLOP/s = \frac{2N^3}{T \times 10^9}
\]

where:

- N = matrix dimension
- T = execution time in seconds

### Speedup

\[
Speedup = \frac{T_{\text{baseline}}}{T_{\text{optimised}}}
\]

### Percentage of BLAS Performance

\[
\text{BLAS Performance (\%)} =
\frac{\text{Custom GFLOP/s}}{\text{OpenBLAS GFLOP/s}}
\times 100
\]

## 9. Compiler Vectorisation Report

The project includes `vectorization_report.txt`, which contains compiler vectorisation information.

To generate a new report:

```bash
gcc-16 -O3 -march=native \
-fopt-info-vec-optimized \
src/gemm_vectorized.c \
-o gemm_vectorized
```

This reports loops that GCC successfully vectorised.

## 10. Cache Profiling

Cache profiling using `perf stat` is performed on a supported Linux system.

Example:

```bash
perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
./gemm_reordered 512
```

The profiling records:

- L1 data-cache loads
- L1 data-cache load misses
- Last-level cache loads
- Last-level cache load misses

The collected cache profiling results are documented in:

```text
results/raw/cache_miss_results.md
```

`perf` is Linux-specific and is not used for the macOS benchmark measurements.

## 11. Generating Graphs

The project uses Python and Matplotlib to generate performance visualisations.

Install Matplotlib:

```bash
python3 -m pip install matplotlib
```

Run the main plotting script:

```bash
python3 scripts/plot_results.py
```

Generate the Roofline plot:

```bash
python3 scripts/plot_roofline.py
```

The generated plots are saved in:

```text
results/plots/
```

The visualisations include:

1. Blocked GEMM GFLOP/s heat map.
2. Execution time versus matrix size.
3. OpenMP speedup.
4. OpenMP efficiency.
5. Optimisation comparison.
6. Roofline plot.

## 12. Raw Results

Benchmark data is stored in:

```text
results/raw/gemm_results.csv
```

Cache profiling data is stored in:

```text
results/raw/cache_miss_results.md
```

The raw results are used by the plotting scripts to generate the project graphs.

## 13. Reproducibility

To reproduce the experiments:

1. Clone or download the repository.
2. Install GCC, OpenBLAS, Python and Matplotlib.
3. Configure the OpenBLAS library and include paths.
4. Compile the required GEMM implementations.
5. Execute the programs with the required matrix sizes and tile sizes.
6. Use the same fixed random seed for input generation.
7. Verify the results against OpenBLAS.
8. Record execution time and GFLOP/s.
9. Store benchmark results in the raw CSV file.
10. Generate the graphs using the scripts in `scripts/`.

