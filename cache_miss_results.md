# GEMM Cache-Miss Results

## Measurement environment

- System: College Linux lab machine
- CPU: Intel Core i9-12900 (hybrid architecture)
- OS: Ubuntu Linux
- Tool: Linux perf stat
- Matrix size: 512 x 512
- Data type: float
- Compiler: GCC 11.4.0
- Flags: -O3 -march=native

## Cache measurements

All counters below are from the Intel Core cluster unless marked Atom.

| Implementation | Mean time (ms) | Core L1 loads | Core L1 misses | Core LLC loads | Core LLC misses |
|---|---:|---:|---:|---:|---:|
| Naive i-j-k | 97.249 | 1815416516 | 799373165 | 101269016 | 36166 |
| Reordered i-k-j | 7.480 | 335831771 | 37730422 | 14290011 | 127684 |
| Blocked T=16 | 14.999 | 877628358 | 14104530 | 278925 | 26818 |
| Blocked T=32 | 14.091 | 637535065 | 88744619 | 305313 | 26015 |
| Blocked T=64 | 11.556 | 497944809 | 75983388 | 381754 | 18705 |
| Blocked T=128 | 8.587 | 452456110 | 72998161 | 443996 | 25824 |
| OpenMP 4 threads | 3.532 | 1163380340 | 19911453 | 3048497 | 28013 |
| OpenMP 10 threads | 1.553 | 588132659 | 43503143 | 3179419 | 17893 |

## Atom cluster counters

| Implementation | Atom L1 loads | Atom L1 misses | Atom LLC loads | Atom LLC misses |
|---|---:|---|---:|---:|
| Naive i-j-k | 791815740 | Unsupported | 5000462 | 12151 |
| Reordered i-k-j | 153876919 | Unsupported | 242237 | 653 |
| Blocked T=16 | 192430402 | Unsupported | 348380 | 137 |
| Blocked T=32 | 196451785 | Unsupported | 330784 | 637 |
| Blocked T=64 | 175864813 | Unsupported | 347633 | 4218 |
| Blocked T=128 | 160452779 | Unsupported | 296028 | 4430 |
| OpenMP 4 threads | 1576860433 | Unsupported | 9967885 | 555 |
| OpenMP 10 threads | 878858035 | Unsupported | 3570338 | 2372 |

## Correctness

All measured implementations passed the BLAS comparison.

Relative max-norm errors:
- Naive: 9.73e-07
- Reordered: 9.73e-07
- Blocked (all tile sizes): 9.73e-07
- OpenMP: 1.41e-06

## Notes

- Atom and Core hardware counters are reported separately by perf.
- Atom L1-dcache-load-misses are unsupported on this CPU.
- No Atom L1 miss values have been estimated or fabricated.
- Measurements include the complete program execution, not only the timed GEMM loops.
- These are Linux lab measurements and must not be mixed with the earlier macOS benchmark timings.
