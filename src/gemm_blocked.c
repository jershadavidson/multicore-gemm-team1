
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <cblas.h>

#define RANDOM_SEED 42
#define TIMED_RUNS 5

double get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    return ts.tv_sec * 1000.0 +
           ts.tv_nsec / 1000000.0;
}

/* Cache-blocked GEMM */
void gemm_blocked(int n, int T,
                  const float *A,
                  const float *B,
                  float *C)
{
    for (int ii = 0; ii < n; ii += T)
    {
        for (int kk = 0; kk < n; kk += T)
        {
            for (int jj = 0; jj < n; jj += T)
            {
                int i_end = (ii + T < n) ? ii + T : n;
                int k_end = (kk + T < n) ? kk + T : n;
                int j_end = (jj + T < n) ? jj + T : n;

                for (int i = ii; i < i_end; i++)
                {
                    for (int k = kk; k < k_end; k++)
                    {
                        float a = A[i * n + k];

                        for (int j = jj; j < j_end; j++)
                        {
                            C[i * n + j] +=
                                a * B[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

float calculate_relative_error(
    const float *C,
    const float *reference,
    int n)
{
    float max_abs_error = 0.0f;
    float max_reference = 0.0f;

    for (long long i = 0; i < (long long)n * n; i++)
    {
        float error = fabsf(C[i] - reference[i]);

        if (error > max_abs_error)
            max_abs_error = error;

        if (fabsf(reference[i]) > max_reference)
            max_reference = fabsf(reference[i]);
    }

    if (max_reference == 0.0f)
        return max_abs_error;

    return max_abs_error / max_reference;
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("Usage: %s <matrix_size> <tile_size>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int T = atoi(argv[2]);

    if (n <= 0 || T <= 0)
    {
        printf("Invalid matrix or tile size.\n");
        return 1;
    }

    if (T != 16 && T != 32 && T != 64 && T != 128)
    {
        printf("Tile size must be 16, 32, 64 or 128.\n");
        return 1;
    }

    size_t elements = (size_t)n * n;

    if (elements > SIZE_MAX / sizeof(float))
    {
        printf("Matrix size is too large.\n");
        return 1;
    }

    size_t bytes = elements * sizeof(float);

    float *A = malloc(bytes);
    float *B = malloc(bytes);
    float *C = malloc(bytes);
    float *reference = malloc(bytes);

    if (!A || !B || !C || !reference)
    {
        printf("Memory allocation failed.\n");

        free(A);
        free(B);
        free(C);
        free(reference);

        return 1;
    }

    srand(RANDOM_SEED);

    for (size_t i = 0; i < elements; i++)
    {
        A[i] = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
        B[i] = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
    }

    for (size_t i = 0; i < elements; i++)
        C[i] = 0.0f;

    printf("\nCache-blocked GEMM\n");
    printf("Matrix size: %d x %d\n", n, n);
    printf("Tile size: %d\n", T);
    printf("Precision: float\n\n");

    /* Warm-up run */
    gemm_blocked(n, T, A, B, C);

    double times[TIMED_RUNS];

    for (int r = 0; r < TIMED_RUNS; r++)
    {
        for (size_t i = 0; i < elements; i++)
            C[i] = 0.0f;

        double start = get_time_ms();

        gemm_blocked(n, T, A, B, C);

        double end = get_time_ms();

        times[r] = end - start;

        printf("Run %d: %.3f ms\n", r + 1, times[r]);
    }

    double mean = 0.0;

    for (int r = 0; r < TIMED_RUNS; r++)
        mean += times[r];

    mean /= TIMED_RUNS;

    double variance = 0.0;

    for (int r = 0; r < TIMED_RUNS; r++)
    {
        double difference = times[r] - mean;
        variance += difference * difference;
    }

    variance /= TIMED_RUNS;

    double stddev = sqrt(variance);

    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        n, n, n,
        1.0f,
        A, n,
        B, n,
        0.0f,
        reference, n
    );

    float relative_error =
        calculate_relative_error(C, reference, n);

    double operations = 2.0 * n * n * n;
    double gflops = operations / (mean * 1.0e6);

    printf("\n----- Results -----\n");
    printf("Mean time: %.3f ms\n", mean);
    printf("Standard deviation: %.3f ms\n", stddev);
    printf("Throughput: %.4f GFLOP/s\n", gflops);
    printf("Relative max-norm error: %.8e\n", relative_error);

    free(A);
    free(B);
    free(C);
    free(reference);

    return 0;
}
