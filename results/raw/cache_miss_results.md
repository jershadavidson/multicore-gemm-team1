
# GEMM Cache-Miss Results

## Measurement status

Cache-miss measurements using Linux `perf stat` are pending.
The current benchmark environment is macOS on Apple Silicon.

| Implementation | Matrix size | L1 cache misses | Last-level cache misses | Status |
|---|---:|---:|---:|---|
| Naive i-j-k | 512 | Not measured | Not measured | Pending |
| Reordered i-k-j | 512 | Not measured | Not measured | Pending |
| Blocked T=16 | 512 | Not measured | Not measured | Pending |
| Blocked T=32 | 512 | Not measured | Not measured | Pending |
| Blocked T=64 | 512 | Not measured | Not measured | Pending |
| Blocked T=128 | 512 | Not measured | Not measured | Pending |

## Notes

- Cache-miss counters were not collected on macOS.
- No cache-miss values have been estimated or fabricated.
- Measurements should be collected on a compatible Linux system using `perf stat`.
