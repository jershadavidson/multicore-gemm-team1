## STREAM Memory Bandwidth Benchmark

- Platform: Apple MacBook Air M5
- Compiler: GCC 16
- Compiler flags: -O3 -march=native -fopenmp
- Array size: 4,000,000 elements per array
- Element size: 8 bytes (double precision)
- OpenMP threads: 10
- Repetitions: 10
- Validation: Passed

| Kernel | Best Bandwidth (MB/s) |
|---|---:|
| Copy | 144787.2 |
| Scale | 109431.5 |
| Add | 107747.7 |
| Triad | 119269.3 |

STREAM Triad bandwidth: 119.27 GB/s


## Hardware and Software Configuration

- Device: Apple MacBook Air (M5)
- Architecture: ARM64 (Apple Silicon)
- CPU: Apple M5, 10 cores (4 performance cores + 6 efficiency cores)
- Unified Memory: 16 GB
- GPU: Apple M5 integrated GPU, 8 cores
- GPU Dedicated VRAM: Not applicable (unified memory)
- L1 Instruction Cache: 128 KB
- L1 Data Cache: 64 KB
- L2 Cache: 6 MB
- L3 Cache: Not reported by system
- Operating System: macOS
- Compiler: GCC 16.2.0
- Compiler Flags: -O3 -march=native -fopenmp
- BLAS Library: OpenBLAS 0.3.34
- OpenMP: Enabled
- MPI: Not used
- CUDA: Not applicable on Apple Silicon